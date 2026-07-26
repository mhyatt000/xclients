from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time

# Share the GPU with the WiLoR/SAM3 servers instead of preallocating 75% of it.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Persist XLA compilations: the planner's first solve costs ~1 min of JIT per cold cache.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jax"))

import cv2
import numpy as np
from numpy.typing import NDArray
import tyro
from webpolicy.client import Client

from xclients.core.cfg import Config
from xclients.core.latest_worker import LatestWorker
from xclients.core.tf import FLU2RDF
from xclients.motion import default_units, EndEffector, RetargetPolicy, unit_assets
from xclients.triangulate import lift_hand_pnp
from xclients.viser_webui import ViserWebUI
from xclients.xarm_driver import XArmDriver, XArmDriverConfig

logging.basicConfig(level=logging.INFO)

# Default camera extrinsics (camera FLU -> world): eye at (0, 0, 0.36) looking down world -X, level.
DEFAULT_WORLD_FROM_CAM_FLU = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.36],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


@dataclass
class RetargetConfig(Config):
    port: int = 8084  # WiLoR server
    host: str = "localhost"
    extr: Path | None = None  # 4x4 camera-to-world transform in repo HT convention: camera FLU -> world
    cap: int | Path = 0
    fx: float = 515.0
    fy: float = 515.0
    limit: int | None = None
    ema_n: int = 4  # EMA smoothing horizon for PnP-refined kp3d; 1 disables smoothing
    offset_x: float = 0.0  # world-frame offset added to kp3d to shift hands into the robot workspace
    offset_z: float = 0.0
    len_traj: int = 5  # online-planner horizon
    dt: float = 0.1
    left_ee: EndEffector = "gripper"  # end effector on dxarm-l
    right_ee: EndEffector = "gripper"  # end effector on dxarm-r
    left_ip: str | None = None  # dxarm-l controller (192.168.1.238); None disables driving
    right_ip: str | None = None  # dxarm-r controller (192.168.1.231); None disables driving
    # Ruka self-collision: drop hand-internal pairs, check hand-vs-arm via one fitted
    # bounding sphere (~630 -> ~190 pairs). --no-coarse-hand-coll restores full fidelity.
    coarse_hand_coll: bool = True

    def __post_init__(self) -> None:
        self.extr = Path(self.extr).expanduser().resolve() if self.extr else None
        if isinstance(self.cap, Path):
            self.cap = self.cap.expanduser().resolve()


def load_world_from_camera_flu(path: Path) -> NDArray[np.float64]:
    if path.suffix == ".npz":
        data = np.load(path)
        key = "HT" if "HT" in data.files else data.files[0]
        cam_t_world = np.asarray(data[key], dtype=np.float64)
    elif path.suffix == ".npy":
        cam_t_world = np.asarray(np.load(path), dtype=np.float64)
    else:
        cam_t_world = np.asarray(np.loadtxt(path), dtype=np.float64)

    if cam_t_world.shape == (3, 4):
        cam_t_world = np.vstack([cam_t_world, np.array([0.0, 0.0, 0.0, 1.0])])
    if cam_t_world.shape != (4, 4):
        raise ValueError(f"Expected extrinsics with shape (4, 4) or (3, 4), got {cam_t_world.shape} from {path}")
    return cam_t_world


def opencv_camera_points_to_world(
    world_from_cam_flu: NDArray[np.float64],
    points_cam: NDArray[np.float64],
) -> NDArray[np.float32]:
    points_h = np.concatenate([points_cam, np.ones((len(points_cam), 1), dtype=np.float64)], axis=1)
    world_from_cam_rdf = world_from_cam_flu @ FLU2RDF
    return (points_h @ world_from_cam_rdf.T)[:, :3].astype(np.float32)


def camera_intrinsics(cfg: RetargetConfig, frame: NDArray[np.uint8]) -> NDArray[np.float64]:
    h, w = frame.shape[:2]
    return np.array(
        [
            [cfg.fx, 0.0, w / 2.0],
            [0.0, cfg.fy, h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def ema_alpha(n: int) -> float:
    if n <= 1:
        return 1.0
    return 2.0 / (float(n) + 1.0)


class HandSmoother:
    def __init__(self, n: int) -> None:
        self._alpha = ema_alpha(n)
        self._points: dict[str, NDArray[np.float64]] = {}

    def smooth(self, slot: str, points: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._alpha >= 1.0 or slot not in self._points:
            smoothed = points.copy()
        else:
            smoothed = self._alpha * points + (1.0 - self._alpha) * self._points[slot]
        self._points[slot] = smoothed
        return smoothed


class HandTracker:
    """Assigns detections to 'left'/'right' slots by handedness + wrist continuity."""

    def __init__(self, handedness_penalty: float = 0.3, gate: float = 0.5, max_misses: int = 10) -> None:
        self._penalty = handedness_penalty
        self._gate = gate
        self._max_misses = max_misses
        self._wrists: dict[str, NDArray[np.float64]] = {}
        self._misses: dict[str, int] = {"left": 0, "right": 0}

    def assign(self, wrists: list[NDArray[np.float64]], is_rights: list[bool]) -> dict[str, int]:
        costs = []
        for slot in ("left", "right"):
            for j, (wrist, is_right) in enumerate(zip(wrists, is_rights)):
                cost = np.linalg.norm(wrist - self._wrists[slot]) if slot in self._wrists else 0.0
                cost += self._penalty * ((slot == "right") != is_right)
                costs.append((float(cost), slot, j))
        assigned: dict[str, int] = {}
        used: set[int] = set()
        for cost, slot, j in sorted(costs):
            if cost > self._gate or slot in assigned or j in used:
                continue
            assigned[slot] = j
            used.add(j)
        for slot in ("left", "right"):
            if slot in assigned:
                self._wrists[slot] = wrists[assigned[slot]]
                self._misses[slot] = 0
            else:
                self._misses[slot] += 1
                if self._misses[slot] >= self._max_misses:
                    self._wrists.pop(slot, None)  # forget stale position so the hand can re-enter anywhere
        return assigned


def lift_hands(
    cfg: RetargetConfig,
    frame: NDArray[np.uint8],
    hands: list[dict],
) -> list[tuple[bool, NDArray[np.float64]]]:
    """PnP-lift each detection to camera frame; returns (is_right, kp3d_cam) per liftable hand."""
    k = camera_intrinsics(cfg, frame)
    lifted = []
    for i, hand in enumerate(hands):
        kp2d = np.asarray(hand["keypoints_2d"], dtype=np.float32)
        kp3d_rel = np.asarray(hand["keypoints_3d"], dtype=np.float64)
        try:
            kp3d_cam, _rot, tvec = lift_hand_pnp(kp2d, kp3d_rel, k)
        except RuntimeError as exc:
            logging.warning("Skipping hand %d: %s", i, exc)
            continue
        if tvec[2] <= 0.0:
            logging.warning("Skipping hand %d: PnP placed it behind the camera with z=%.3f", i, tvec[2])
            continue
        lifted.append((bool(hand["is_right"]), kp3d_cam))
    return lifted


def run_wilor(client: Client, frame: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], dict]:
    return frame, client.step({"image": frame, "type": "image"})


def main(cfg: RetargetConfig) -> None:
    world_from_cam_flu = load_world_from_camera_flu(cfg.extr) if cfg.extr else DEFAULT_WORLD_FROM_CAM_FLU
    cap = cv2.VideoCapture(str(cfg.cap) if isinstance(cfg.cap, Path) else cfg.cap)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {cfg.cap}")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read first frame from camera {cfg.cap}")

    units = default_units(
        len_traj=cfg.len_traj,
        dt=cfg.dt,
        left_ee=cfg.left_ee,
        right_ee=cfg.right_ee,
        coarse_hand_coll=cfg.coarse_hand_coll,
    )
    ui = ViserWebUI(assets=unit_assets(units))
    h, w = frame.shape[:2]
    ui.add_camera(
        "cam",
        world_from_cam_flu,
        fov=2.0 * np.arctan2(h / 2.0, cfg.fy),
        aspect=w / h,
        image=frame[..., ::-1],
    )

    client = Client(cfg.host, cfg.port)
    worker = LatestWorker(lambda image: run_wilor(client, image), name="wilor-worker")
    policy = RetargetPolicy(units)
    policy_worker = LatestWorker(lambda hands: policy.step({"hands": hands}), name="retarget-policy")
    drivers: dict[str, XArmDriver] = {}
    driver_workers: dict[str, LatestWorker] = {}
    for name, ip in (("dxarm-l", cfg.left_ip), ("dxarm-r", cfg.right_ip)):
        if ip is not None:
            drivers[name] = XArmDriver(XArmDriverConfig(ip=ip, gripper=False, clear=True))
            driver_workers[name] = LatestWorker(drivers[name].step, name=f"{name}-driver")
    # Warm up the solver JIT immediately so it compiles before hands appear (~1 min cold, seconds cached).
    warmup_kp3d = {}
    for slot, y in (("left", -0.3), ("right", 0.3)):
        kp = np.tile(np.array([0.35, y, 0.45], dtype=np.float32), (21, 1))
        kp[0] += [-0.05, 0.0, -0.03]
        kp[4] += [0.05, 0.02, 0.0]
        kp[8] += [0.05, -0.02, 0.0]
        warmup_kp3d[slot] = kp
    warmup_seq = policy_worker.submit(warmup_kp3d)
    warmup_start = time.monotonic()
    logging.info("compiling planner (first run can take ~1 min; cached in %s)...", os.environ["JAX_COMPILATION_CACHE_DIR"])
    smoother = HandSmoother(cfg.ema_n)
    tracker = HandTracker()
    kp3d_offset = np.array([cfg.offset_x, 0.0, cfg.offset_z], dtype=np.float32)
    step = 0
    last_result_seq = 0
    last_error_seq = 0
    last_policy_seq = 0
    last_policy_error_seq = 0
    last_driver_error_seq = {name: 0 for name in driver_workers}
    policy_updates = 0
    prev_cfgs: dict[str, NDArray[np.float64]] = {}

    logging.info("Polling camera %s and sending latest frames to %s:%s", cfg.cap, cfg.host, cfg.port)
    try:
        while cfg.limit is None or step < cfg.limit:
            worker.submit(frame.copy())
            result = worker.latest()

            if result is not None and result.error is not None and result.seq != last_error_seq:
                error = result.error
                logging.error("WiLoR inference failed", exc_info=(type(error), error, error.__traceback__))
                last_error_seq = result.seq
            elif result is not None and result.value is not None and result.seq != last_result_seq:
                result_frame, out = result.value
                hands = out.get("hands") or []
                lifted = lift_hands(cfg, result_frame, hands)
                assigned = tracker.assign([kp3d_cam[0] for _, kp3d_cam in lifted], [r for r, _ in lifted])
                kp3ds = {
                    slot: opencv_camera_points_to_world(world_from_cam_flu, smoother.smooth(slot, lifted[j][1]))
                    + kp3d_offset
                    for slot, j in assigned.items()
                }
                ui.update_hands(kp3ds)
                if kp3ds:
                    policy_worker.submit(kp3ds)
                last_result_seq = result.seq

            policy_result = policy_worker.latest()
            if (
                policy_result is not None
                and policy_result.error is not None
                and policy_result.seq != last_policy_error_seq
            ):
                error = policy_result.error
                logging.error("retarget policy failed", exc_info=(type(error), error, error.__traceback__))
                last_policy_error_seq = policy_result.seq
            elif policy_result is not None and policy_result.value is not None and policy_result.seq != last_policy_seq:
                if policy_result.seq == warmup_seq:
                    policy.reset()
                    logging.info("planner compiled in %.1fs; retargeting live", time.monotonic() - warmup_start)
                else:
                    cfgs = {name: np.asarray(out["q"]) for name, out in policy_result.value.items()}
                    if policy_updates % 30 == 0:
                        deltas = {
                            name: float(np.abs(q - prev_cfgs[name]).max()) if name in prev_cfgs else float("nan")
                            for name, q in cfgs.items()
                        }
                        logging.info("policy update #%d max|dq|: %s", policy_updates, deltas)
                    prev_cfgs = cfgs
                    policy_updates += 1
                    ui.step(cfgs)
                    for name, driver_worker in driver_workers.items():
                        driver_worker.submit({"q": cfgs[name], "aperture": policy_result.value[name]["aperture"]})
                last_policy_seq = policy_result.seq

            for name, driver_worker in driver_workers.items():
                driver_result = driver_worker.latest()
                if driver_result is not None and driver_result.error is not None and driver_result.seq != last_driver_error_seq[name]:
                    error = driver_result.error
                    logging.error("%s driver failed", name, exc_info=(type(error), error, error.__traceback__))
                    last_driver_error_seq[name] = driver_result.seq

            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame from camera %s", cfg.cap)
                continue
            ui.update_camera_image("cam", frame[..., ::-1])
            step += 1
    finally:
        worker.close()
        policy_worker.close()
        for driver_worker in driver_workers.values():
            driver_worker.close()
        for driver in drivers.values():
            driver.close()
        cap.release()


if __name__ == "__main__":
    main(tyro.cli(RetargetConfig))
