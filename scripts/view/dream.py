from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time

import cv2
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R
import tyro
from webpolicy.client import Client

from xclients.core import tf as xctf
from xclients.core.run.scene import RerunScene

logging.basicConfig(level=logging.INFO)


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 8000
    camera: str | int = 0  # OpenCV camera index to poll from
    camera_name: str = "dream"  # Rerun camera entity name
    image_size: int = 200  # Resize frames to a square image before sending to Dream
    fx: float = 515.0  # Focal length in pixels along x for the payload K matrix
    fy: float = 515.0  # Focal length in pixels along y for the payload K matrix
    q: list[float] = field(default_factory=lambda: [0.0] * 7)  # Joint vector sent to Dream and logged to Rerun
    deg2rad: bool = False  # Convert cfg.q from degrees to radians before sending
    urdf: Path = Path("xarm7_standalone.urdf")
    app_id: str = "dream_view"
    entity_path_prefix: str = "robot"
    transforms_path: str = "robot/transforms"
    spawn: bool = True
    rrd_path: Path | None = None
    limit: int | None = None
    show: bool = False  # Also show the local OpenCV window

    def __post_init__(self) -> None:
        self.urdf = self.urdf.expanduser().resolve()
        if self.rrd_path is not None:
            self.rrd_path = Path(self.rrd_path).expanduser().resolve()


def scale_intrinsics(k: np.ndarray, sx: float, sy: float) -> np.ndarray:
    scaled = np.asarray(k, dtype=np.float32).copy()
    scaled[0, 0] *= sx
    scaled[1, 1] *= sy
    scaled[0, 2] *= sx
    scaled[1, 2] *= sy
    return scaled


def coerce_w2c_pose(w2c: np.ndarray | None) -> np.ndarray | None:
    if w2c is None:
        return None

    pose = np.asarray(w2c, dtype=np.float64)
    while pose.ndim > 2 and pose.shape[0] == 1:
        pose = pose[0]

    if pose.shape == (3, 4):
        pose = np.vstack([pose, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)])

    if pose.shape != (4, 4):
        raise ValueError(f"Expected w2c with shape (4, 4), got {pose.shape}")

    return pose


def dream_camera_calibration(k: np.ndarray, w2c: np.ndarray, width: int, height: int) -> dict[str, np.ndarray | int]:
    return {
        "intrinsics": np.asarray(k, dtype=np.float32),
        "extrinsics": np.asarray(w2c, dtype=np.float32),
        "width": int(width),
        "height": int(height),
    }


def log_dynamic_camera(scene: RerunScene, camera_name: str, calibration: dict[str, np.ndarray | int]) -> None:
    entity_path = f"{scene.world_path}/cam/{camera_name}"
    image_plane_frame = f"{entity_path}/image_plane"
    extrinsic = np.asarray(calibration["extrinsics"], dtype=np.float64)
    intrinsics = np.asarray(calibration["intrinsics"], dtype=np.float64)
    width = int(calibration["width"])
    height = int(calibration["height"])

    rot = extrinsic[:3, :3]
    if not np.isfinite(rot).all():
        raise ValueError(f"Camera rotation has non-finite values: {rot!r}")
    if not np.isfinite(extrinsic[:3, 3]).all():
        raise ValueError(f"Camera translation has non-finite values: {extrinsic[:3, 3]!r}")
    quat_xyzw = R.from_matrix(rot).as_quat()
    t = extrinsic[:3, 3].astype(np.float32)

    rr.log(
        entity_path,
        rr.Transform3D(
            translation=t,
            quaternion=quat_xyzw,
            parent_frame=str(scene.world_path),
            child_frame=entity_path,
            relation=rr.TransformRelation.ChildFromParent,
        ),
        static=False,
    )
    rr.log(
        entity_path,
        rr.Pinhole(
            resolution=[width, height],
            focal_length=[float(intrinsics[0, 0]), float(intrinsics[1, 1])],
            principal_point=[float(intrinsics[0, 2]), float(intrinsics[1, 2])],
            camera_xyz=rr.ViewCoordinates.RDF,
            parent_frame=entity_path,
            child_frame=image_plane_frame,
            image_plane_distance=0.1,
            color=[255, 128, 0],
            line_width=0.002,
        ),
        static=False,
    )


def joint_values_from_q(scene: RerunScene, q: np.ndarray) -> dict[str, float]:
    if q.ndim != 1:
        q = q.reshape(-1)

    arm_joint_names = [name for name in sorted(scene.joint_map) if name.startswith("joint")]
    if len(q) != len(arm_joint_names):
        raise ValueError(f"Expected {len(arm_joint_names)} joint values for {arm_joint_names}, got {len(q)}")

    return {name: float(value) for name, value in zip(arm_joint_names, q, strict=True)}


def main(cfg: Config) -> None:
    client = Client(cfg.host, cfg.port)
    cap = cv2.VideoCapture(cfg.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {cfg.camera}")

    scene = RerunScene(
        cfg.urdf,
        app_id=cfg.app_id,
        entity_path_prefix=cfg.entity_path_prefix,
        transforms_path=cfg.transforms_path,
        spawn=cfg.spawn,
        rrd_path=cfg.rrd_path,
    )
    scene.set_cameras([cfg.camera_name])

    q = np.asarray(cfg.q, dtype=np.float32)
    if cfg.deg2rad:
        q = np.deg2rad(q)

    start = time.monotonic()
    step = 0
    logging.info("Polling camera %s and sending frames to %s:%s", cfg.camera, cfg.host, cfg.port)
    while cfg.limit is None or step < cfg.limit:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to read frame from camera %s", cfg.camera)
            continue

        h, w = frame.shape[:2]
        frame_model = cv2.resize(frame, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_LINEAR)
        k_orig = np.array(
            [
                [cfg.fx, 0.0, w / 2.0],
                [0.0, cfg.fy, h / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        k_model = scale_intrinsics(k_orig, cfg.image_size / float(w), cfg.image_size / float(h))

        rr.set_time("step", sequence=step)
        rr.set_time("time", duration=time.monotonic() - start)
        scene.log_camera_images({cfg.camera_name: frame})
        scene.log_joints(joint_values_from_q(scene, q), step=step)

        payload = {
            "image": frame_model,
            "type": "image",
            "q": q,
            "K": k_model,
        }
        out = client.step(payload)

        pose = coerce_w2c_pose(out.get("w2c") if out else None)
        if pose is not None:
            calibration = dream_camera_calibration(k_orig, pose, w, h)
            # Dream returns camera pose relative to the robot/world in RDF; convert to the
            # scene's FLU world frame the same way the static calibration loader does.
            calibration["extrinsics"] = xctf.RDF2FLU @ pose
            try:
                log_dynamic_camera(scene, cfg.camera_name, calibration)
            except (ValueError, np.linalg.LinAlgError) as exc:
                logging.warning("Skipping invalid Dream camera pose at step %d: %s", step, exc)
        else:
            logging.warning("Dream response did not include a valid w2c pose at step %d", step)

        if cfg.show:
            cv2.imshow(f"Dream {cfg.camera}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        step += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(tyro.cli(Config))
