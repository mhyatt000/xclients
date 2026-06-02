from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time

import cv2
import numpy as np
from numpy.typing import NDArray
import rerun as rr
import rerun.blueprint as rrb
import tyro
from webpolicy.client import Client

from xclients.core.cfg import Config
from xclients.core.latest_worker import LatestWorker
from xclients.core.tf import FLU2RDF
from xclients.triangulate import lift_hand_pnp, project_points

logging.basicConfig(level=logging.INFO)

HAND_EDGES = np.array(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ],
    dtype=np.int32,
)
HAND_COLORS = np.array(
    [
        [0, 255, 0],
        [255, 0, 255],
        [0, 180, 255],
        [255, 180, 0],
    ],
    dtype=np.uint8,
)


@dataclass
class WilorConfig(Config):
    extr: Path  # 4x4 camera-to-world transform in repo HT convention: camera FLU -> world
    cap: int | Path = 0
    camera_name: str = "wilor"
    app_id: str = "wilor_view"
    world_path: str = "world"
    fx: float = 515.0
    fy: float = 515.0
    spawn: bool = True
    rrd_path: Path | None = None
    show: bool = False
    limit: int | None = None
    jpeg_quality: int = 75
    point_radius: float = 0.008
    ema_n: int = 4  # EMA smoothing horizon for PnP-refined kp3d; 1 disables smoothing

    def __post_init__(self) -> None:
        self.extr = Path(self.extr).expanduser().resolve()
        if isinstance(self.cap, Path):
            self.cap = self.cap.expanduser().resolve()
        if self.rrd_path is not None:
            self.rrd_path = self.rrd_path.expanduser().resolve()


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
    if len(points_cam) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    points_h = np.concatenate([points_cam, np.ones((len(points_cam), 1), dtype=np.float64)], axis=1)
    world_from_cam_rdf = world_from_cam_flu @ FLU2RDF
    return (points_h @ world_from_cam_rdf.T)[:, :3].astype(np.float32)


def camera_intrinsics(cfg: WilorConfig, frame: NDArray[np.uint8]) -> NDArray[np.float64]:
    h, w = frame.shape[:2]
    return np.array(
        [
            [cfg.fx, 0.0, w / 2.0],
            [0.0, cfg.fy, h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def hand_color(index: int) -> NDArray[np.uint8]:
    return HAND_COLORS[index % len(HAND_COLORS)]


def ema_alpha(n: int) -> float:
    if n <= 1:
        return 1.0
    return 2.0 / (float(n) + 1.0)


class HandSmoother:
    def __init__(self, n: int) -> None:
        self._alpha = ema_alpha(n)
        self._points: dict[int, NDArray[np.float64]] = {}

    def smooth(self, hand_index: int, points: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._alpha >= 1.0 or hand_index not in self._points:
            smoothed = points.copy()
        else:
            smoothed = self._alpha * points + (1.0 - self._alpha) * self._points[hand_index]
        self._points[hand_index] = smoothed
        return smoothed


def setup_rerun(cfg: WilorConfig, world_from_cam_flu: NDArray[np.float64], width: int, height: int) -> None:
    rr.init(cfg.app_id)
    if cfg.rrd_path is not None:
        rr.save(cfg.rrd_path)
    elif cfg.spawn:
        rr.spawn()

    cam_root = f"{cfg.world_path}/cam/{cfg.camera_name}"
    image_path = f"{cam_root}/image"
    world_from_cam_rdf = world_from_cam_flu @ FLU2RDF

    rr.log("/", rr.ViewCoordinates.FLU, static=True)
    rr.log(cfg.world_path, rr.CoordinateFrame(frame=cfg.world_path), static=True)
    rr.log(
        cam_root,
        rr.Transform3D(
            translation=world_from_cam_rdf[:3, 3].astype(np.float32),
            mat3x3=world_from_cam_rdf[:3, :3].astype(np.float32),
            parent_frame=cfg.world_path,
            child_frame=cam_root,
            relation=rr.TransformRelation.ParentFromChild,
        ),
        static=True,
    )
    rr.log(
        cam_root,
        rr.Pinhole(
            resolution=[width, height],
            focal_length=[cfg.fx, cfg.fy],
            principal_point=[width / 2.0, height / 2.0],
            camera_xyz=rr.ViewCoordinates.RDF,
            parent_frame=cam_root,
            child_frame=f"{cam_root}/image_plane",
            image_plane_distance=0.1,
            color=[255, 128, 0],
            line_width=0.002,
        ),
        static=True,
    )

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/",
                    contents=[
                        f"+ {cam_root}",
                        f"+ {cfg.world_path}/hands/**",
                    ],
                ),
                rrb.Spatial2DView(origin=cam_root, contents=[f"+ {image_path}/**"]),
                column_shares=[3, 2],
            ),
            collapse_panels=True,
        )
    )


def log_frame(
    cfg: WilorConfig,
    frame: NDArray[np.uint8],
    hands: list[dict],
    world_from_cam_flu: NDArray[np.float64],
    smoother: HandSmoother,
) -> None:
    cam_root = f"{cfg.world_path}/cam/{cfg.camera_name}"
    image_path = f"{cam_root}/image"
    hand_root = f"{cfg.world_path}/hands"

    rr.log(image_path, rr.CoordinateFrame(frame=f"{cam_root}/image_plane"), static=True)
    rr.log(image_path, rr.Image(frame, color_model="BGR").compress(jpeg_quality=cfg.jpeg_quality))
    rr.log(f"{image_path}/hands", rr.Clear(recursive=True))
    rr.log(hand_root, rr.Clear(recursive=True))
    rr.log(hand_root, rr.CoordinateFrame(frame=cfg.world_path), static=True)
    k = camera_intrinsics(cfg, frame)

    for i, hand in enumerate(hands):
        color = hand_color(i)
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
        kp3d_cam = smoother.smooth(i, kp3d_cam)
        kp3d_world = opencv_camera_points_to_world(world_from_cam_flu, kp3d_cam)
        reproj = project_points(kp3d_cam, k).astype(np.float32)
        kp2d_path = f"{image_path}/hands/hand_{i}/kp2d"
        reproj_path = f"{image_path}/hands/hand_{i}/reprojected"
        kp2d_bones_path = f"{image_path}/hands/hand_{i}/bones"
        kp3d_path = f"{hand_root}/hand_{i}/kp3d"
        kp3d_bones_path = f"{hand_root}/hand_{i}/bones"

        rr.log(kp2d_path, rr.CoordinateFrame(frame=f"{cam_root}/image_plane"), static=True)
        rr.log(kp2d_path, rr.Points2D(kp2d, colors=color, radii=4))
        rr.log(reproj_path, rr.CoordinateFrame(frame=f"{cam_root}/image_plane"), static=True)
        rr.log(reproj_path, rr.Points2D(reproj, colors=[255, 255, 255], radii=2))
        rr.log(kp2d_bones_path, rr.CoordinateFrame(frame=f"{cam_root}/image_plane"), static=True)
        rr.log(kp2d_bones_path, rr.LineStrips2D(kp2d[HAND_EDGES], colors=color, radii=2))

        rr.log(kp3d_path, rr.CoordinateFrame(frame=cfg.world_path), static=True)
        rr.log(kp3d_path, rr.Points3D(kp3d_world, colors=color, radii=cfg.point_radius))
        rr.log(kp3d_bones_path, rr.CoordinateFrame(frame=cfg.world_path), static=True)
        rr.log(kp3d_bones_path, rr.LineStrips3D(kp3d_world[HAND_EDGES], colors=color, radii=cfg.point_radius * 0.5))


def run_wilor(client: Client, frame: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], dict]:
    return frame, client.step({"image": frame, "type": "image"})


def main(cfg: WilorConfig) -> None:
    world_from_cam_flu = load_world_from_camera_flu(cfg.extr)
    cap = cv2.VideoCapture(str(cfg.cap) if isinstance(cfg.cap, Path) else cfg.cap)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {cfg.cap}")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read first frame from camera {cfg.cap}")
    setup_rerun(cfg, world_from_cam_flu, frame.shape[1], frame.shape[0])

    client = Client(cfg.host, cfg.port)
    worker = LatestWorker(lambda image: run_wilor(client, image), name="wilor-view-worker")
    smoother = HandSmoother(cfg.ema_n)
    start = time.monotonic()
    step = 0
    last_result_seq = 0
    last_error_seq = 0

    logging.info("Polling camera %s and sending latest frames to %s:%s", cfg.cap, cfg.host, cfg.port)
    try:
        while cfg.limit is None or step < cfg.limit:
            rr.set_time("step", sequence=step)
            rr.set_time("time", duration=time.monotonic() - start)

            worker.submit(frame.copy())
            result = worker.latest()
            display = frame

            if result is not None and result.error is not None and result.seq != last_error_seq:
                error = result.error
                logging.error("WiLoR inference failed", exc_info=(type(error), error, error.__traceback__))
                last_error_seq = result.seq
            elif result is not None and result.value is not None and result.seq != last_result_seq:
                result_frame, out = result.value
                hands = out.get("hands") or []
                log_frame(cfg, result_frame, hands, world_from_cam_flu, smoother)
                display = result_frame
                last_result_seq = result.seq

            if cfg.show:
                cv2.imshow(f"WiLoR {cfg.cap}", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame from camera %s", cfg.cap)
                continue
            step += 1
    finally:
        worker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(tyro.cli(WilorConfig))
