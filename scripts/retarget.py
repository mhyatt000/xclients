from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
import tyro
from webpolicy.client import Client

from xclients.core.cfg import Config

# TEMP(debug): re-enable together with the WiLoR/planner blocks in main().
# from xclients.core.latest_worker import LatestWorker
from xclients.core.tf import FLU2RDF
from xclients.triangulate import lift_hand_pnp
from xclients.viser_webui import ViserWebUI

logging.basicConfig(level=logging.INFO)


@dataclass
class RetargetConfig(Config):
    # TEMP(debug): defaults added while WiLoR + onlineplanner are disabled below;
    # revert to required fields when re-enabling.
    port: int = 8000
    host: str = "localhost"
    onlineplanner_host: str = "localhost"
    onlineplanner_port: int = 8085
    extr: Path | None = None  # 4x4 camera-to-world transform in repo HT convention: camera FLU -> world
    cap: int | Path = 0
    fx: float = 515.0
    fy: float = 515.0
    limit: int | None = None
    ema_n: int = 4  # EMA smoothing horizon for PnP-refined kp3d; 1 disables smoothing

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
        self._points: NDArray[np.float64] | None = None

    def smooth(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._alpha >= 1.0 or self._points is None:
            smoothed = points.copy()
        else:
            smoothed = self._alpha * points + (1.0 - self._alpha) * self._points
        self._points = smoothed
        return smoothed


def lift_first_hand(
    cfg: RetargetConfig,
    frame: NDArray[np.uint8],
    hands: list[dict],
    world_from_cam_flu: NDArray[np.float64],
    smoother: HandSmoother,
) -> NDArray[np.float32] | None:
    k = camera_intrinsics(cfg, frame)
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
        kp3d_cam = smoother.smooth(kp3d_cam)
        return opencv_camera_points_to_world(world_from_cam_flu, kp3d_cam)
    return None


def run_wilor(client: Client, frame: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], dict]:
    return frame, client.step({"image": frame, "type": "image"})


def run_onlineplanner(client: Client, kp3d: NDArray[np.float32]) -> dict:
    return client.step({"kp3d": kp3d})


def main(cfg: RetargetConfig) -> None:
    world_from_cam_flu = load_world_from_camera_flu(cfg.extr) if cfg.extr else np.eye(4)
    cap = cv2.VideoCapture(str(cfg.cap) if isinstance(cfg.cap, Path) else cfg.cap)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {cfg.cap}")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read first frame from camera {cfg.cap}")

    ui = ViserWebUI()

    # TEMP(debug): WiLoR + onlineplanner disabled; the loop below only reads the
    # camera and steps the viser scene. Uncomment the marked blocks to restore.
    # client = Client(cfg.host, cfg.port)
    # worker = LatestWorker(lambda image: run_wilor(client, image), name="wilor-worker")
    # planner_client = Client(cfg.onlineplanner_host, cfg.onlineplanner_port)
    # planner_worker = LatestWorker(lambda kp3d: run_onlineplanner(planner_client, kp3d), name="onlineplanner-worker")
    # smoother = HandSmoother(cfg.ema_n)
    step = 0
    # last_result_seq = 0
    # last_error_seq = 0
    # last_planner_error_seq = 0

    logging.info("Polling camera %s and stepping viser scene (WiLoR/planner disabled)", cfg.cap)
    try:
        while cfg.limit is None or step < cfg.limit:
            ui.step()

            # TEMP(debug): WiLoR inference disabled.
            # worker.submit(frame.copy())
            # result = worker.latest()
            #
            # if result is not None and result.error is not None and result.seq != last_error_seq:
            #     error = result.error
            #     logging.error("WiLoR inference failed", exc_info=(type(error), error, error.__traceback__))
            #     last_error_seq = result.seq
            # elif result is not None and result.value is not None and result.seq != last_result_seq:
            #     result_frame, out = result.value
            #     hands = out.get("hands") or []
            #     kp3d = lift_first_hand(cfg, result_frame, hands, world_from_cam_flu, smoother)
            #     if kp3d is not None:
            #         planner_worker.submit(kp3d)
            #     last_result_seq = result.seq
            #
            # TEMP(debug): onlineplanner disabled.
            # planner_result = planner_worker.latest()
            # if (
            #     planner_result is not None
            #     and planner_result.error is not None
            #     and planner_result.seq != last_planner_error_seq
            # ):
            #     error = planner_result.error
            #     logging.error("onlineplanner failed", exc_info=(type(error), error, error.__traceback__))
            #     last_planner_error_seq = planner_result.seq

            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame from camera %s", cfg.cap)
                continue
            step += 1
    finally:
        # worker.close()
        # planner_worker.close()
        cap.release()


if __name__ == "__main__":
    main(tyro.cli(RetargetConfig))
