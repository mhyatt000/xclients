from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation as R
import tyro
from webpolicy.client import Client

from xclients.core import tf as xctf
from xclients.core.run.scene import RerunScene

logging.basicConfig(level=logging.INFO)

GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 8000
    da_host: str | None = None
    da_port: int | None = None
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
    history: int = 100  # Number of recent camera centers to keep as 3D points
    max_camera_distance: float = 3.0  # Skip Dream poses whose camera center is farther than this many meters
    depth_stride: int = 4  # Subsample factor when converting depth maps to 3D points
    max_depth: float = 2.0  # Maximum depth in meters to include in the point cloud
    show: bool = False  # Also show the local OpenCV window

    def __post_init__(self) -> None:
        self.urdf = self.urdf.expanduser().resolve()
        if self.rrd_path is not None:
            self.rrd_path = Path(self.rrd_path).expanduser().resolve()
        if (self.da_host is None) != (self.da_port is None):
            raise ValueError("Pass both da_host and da_port, or neither.")


def scale_intrinsics(k: np.ndarray, sx: float, sy: float) -> np.ndarray:
    scaled = np.asarray(k, dtype=np.float32).copy()
    scaled[0, 0] *= sx
    scaled[1, 1] *= sy
    scaled[0, 2] *= sx
    scaled[1, 2] *= sy
    return scaled


def draw_mask(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None

    arr = np.asarray(mask)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected mask with shape (h, w) or (1, h, w), got {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        maxv = float(arr.max()) if arr.size else 0.0
        if maxv <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def draw_output_image(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None

    arr = np.asarray(image)
    if arr.ndim >= 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected raster image with shape (h, w), (h, w, c), or batched variants, got {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        maxv = float(arr.max()) if arr.size else 0.0
        if maxv <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def draw_depth_image(depth: np.ndarray | None) -> np.ndarray | None:
    if depth is None:
        return None

    arr = np.asarray(depth)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected depth with shape (h, w) or singleton-batched variants, got {arr.shape}")

    arr = arr.astype(np.float32)
    valid = np.isfinite(arr) & (arr > 0.0)
    if not np.any(valid):
        return np.zeros((*arr.shape, 3), dtype=np.uint8)

    lo = float(arr[valid].min())
    hi = float(arr[valid].max())
    norm = np.zeros_like(arr, dtype=np.float32)
    if hi > lo:
        norm[valid] = (arr[valid] - lo) / (hi - lo)
    depth_u8 = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def ensure_bgr_image(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return cv2.cvtColor(arr[..., 0], cv2.COLOR_GRAY2BGR)
    return arr


def log_aux_image(scene: RerunScene, camera_name: str, path: str, image: np.ndarray) -> None:
    rr.log(path, rr.CoordinateFrame(frame=f"{scene.world_path}/cam/{camera_name}/image_plane"), static=True)
    rr.log(path, rr.Image(image, color_model="BGR").compress(jpeg_quality=75), static=False)


def send_dream_blueprint(scene: RerunScene, camera_name: str) -> None:
    cam_root = f"{scene.world_path}/cam/{camera_name}"
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/",
                contents=[
                    "+ /robot/**",
                    f"+ {scene.world_path}/**",
                ],
            ),
            rrb.Vertical(
                contents=[
                    rrb.Spatial2DView(origin=cam_root, contents=["+ $origin/image"]),
                    rrb.Spatial2DView(origin=cam_root, contents=["+ $origin/mask"]),
                    rrb.Spatial2DView(origin=cam_root, contents=["+ $origin/raster"]),
                    rrb.Spatial2DView(origin=cam_root, contents=["+ $origin/depth"]),
                ]
            ),
            column_shares=[4, 1],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)


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


def camera_world_position(extrinsic: np.ndarray) -> np.ndarray:
    rot = np.asarray(extrinsic[:3, :3], dtype=np.float64)
    t = np.asarray(extrinsic[:3, 3], dtype=np.float64)
    return (-rot.T @ t).astype(np.float32)


def history_colors(count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 4), dtype=np.uint8)
    if count == 1:
        return np.array([[255, 128, 0, 255]], dtype=np.uint8)

    oldest = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    newest = np.array([255.0, 128.0, 0.0], dtype=np.float32)
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    rgb = oldest * (1.0 - t) + newest * t
    alpha = (64.0 + 191.0 * t).reshape(-1, 1)
    colors = np.concatenate([rgb, alpha], axis=1)
    return np.round(colors).astype(np.uint8)


def first_array(value: object) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def unproject_depth_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    bgr_image: np.ndarray,
    *,
    stride: int,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(depth, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"Expected depth map with shape (h, w), got {z.shape}")

    k = np.asarray(intrinsics, dtype=np.float32)
    if k.shape != (3, 3):
        raise ValueError(f"Expected intrinsics with shape (3, 3), got {k.shape}")

    if bgr_image.shape[:2] != z.shape:
        bgr_image = cv2.resize(bgr_image, (z.shape[1], z.shape[0]), interpolation=cv2.INTER_LINEAR)

    step = max(int(stride), 1)
    ys, xs = np.mgrid[0 : z.shape[0] : step, 0 : z.shape[1] : step]
    zs = z[::step, ::step]

    valid = np.isfinite(zs) & (zs > 0.0)
    if max_depth > 0.0:
        valid &= zs <= float(max_depth)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    zs = zs[valid].astype(np.float32)

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    x = (xs - cx) * zs / fx
    y = (ys - cy) * zs / fy
    points = np.stack([x, y, zs], axis=1)

    colors_bgr = bgr_image[::step, ::step][valid]
    colors_rgb = colors_bgr[:, ::-1].astype(np.uint8)
    return points, colors_rgb


def main(cfg: Config) -> None:
    client = Client(cfg.host, cfg.port)
    da_client = Client(cfg.da_host, cfg.da_port) if cfg.da_host is not None and cfg.da_port is not None else None
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
    mask_path = f"{scene.world_path}/cam/{cfg.camera_name}/mask"
    raster_path = f"{scene.world_path}/cam/{cfg.camera_name}/raster"
    depth_path = f"{scene.world_path}/cam/{cfg.camera_name}/depth"
    depth_points_path = f"{scene.world_path}/scene/{cfg.camera_name}_depth_points"
    scene.set_cameras([cfg.camera_name])
    send_dream_blueprint(scene, cfg.camera_name)

    q_cfg = np.asarray(cfg.q, dtype=np.float32)
    q_payload = np.deg2rad(q_cfg) if cfg.deg2rad else q_cfg

    start = time.monotonic()
    step = 0
    camera_history: deque[np.ndarray] = deque(maxlen=max(1, int(cfg.history)))
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
        scene.log_joints(joint_values_from_q(scene, q_cfg), step=step, degrees=True)

        payload = {
            "image": frame_model,
            "type": "image",
            "q": q_payload,
            "K": k_model,
        }
        out = client.step(payload)

        da_out = None
        if da_client is not None:
            da_payload = {
                "image": [frame],
                "intrinsics": np.array([k_orig], dtype=np.float32),
            }
            da_out = da_client.step(da_payload)

        mask = draw_mask(out.get("mask") if out else None)
        raster_raw = out.get("raster_image") if out else None
        if raster_raw is None and out is not None:
            raster_raw = out.get("rast_image")
        raster = draw_output_image(raster_raw)
        depth_raw = first_array(da_out.get("depth") if da_out else None)
        depth_vis = draw_depth_image(depth_raw)
        if mask is not None:
            log_aux_image(scene, cfg.camera_name, mask_path, ensure_bgr_image(mask))
        if raster is not None:
            log_aux_image(scene, cfg.camera_name, raster_path, ensure_bgr_image(raster))
        if depth_vis is not None:
            log_aux_image(scene, cfg.camera_name, depth_path, depth_vis)

        if depth_raw is not None:
            depth_intr = first_array(da_out.get("intrinsics") if da_out else None)
            if depth_intr is None:
                depth_intr = k_orig
            try:
                points_cam, colors_rgb = unproject_depth_points(
                    depth_raw,
                    np.asarray(depth_intr, dtype=np.float32),
                    frame,
                    stride=cfg.depth_stride,
                    max_depth=cfg.max_depth,
                )
                scene.log_points3d(
                    points_cam,
                    colors=colors_rgb,
                    radii=0.02,
                    path=depth_points_path,
                    parent_frame=f"{scene.world_path}/cam/{cfg.camera_name}",
                )
            except ValueError as exc:
                logging.warning("Skipping DA depth point cloud at step %d: %s", step, exc)

        pose = coerce_w2c_pose(out.get("w2c") if out else None)
        if pose is not None:
            calibration = dream_camera_calibration(k_orig, pose, w, h)
            try:
                calibration["extrinsics"] = xctf.RDF2FLU @ pose

                camera_position = camera_world_position(np.asarray(calibration["extrinsics"], dtype=np.float64))
                if not np.isfinite(camera_position).all():
                    raise ValueError(f"Camera center has non-finite values: {camera_position!r}")
                camera_distance = float(np.linalg.norm(camera_position))
                if camera_distance > cfg.max_camera_distance:
                    raise ValueError(
                        f"Camera center distance {camera_distance:.3f} m exceeds max_camera_distance={cfg.max_camera_distance:.3f} m"
                    )

                log_dynamic_camera(scene, cfg.camera_name, calibration)
                camera_history.append(camera_position)
                history_points = np.stack(camera_history, axis=0)
                scene.log_points3d(
                    history_points,
                    colors=history_colors(len(history_points)),
                    radii=0.01,
                    path=f"{scene.world_path}/scene/{cfg.camera_name}_history",
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                logging.warning("Skipping invalid Dream camera pose at step %d: %s", step, exc)
        else:
            logging.warning("Dream response did not include a valid w2c pose at step %d", step)

        if cfg.show:
            cv2.imshow(f"Dream {cfg.camera}", frame)
            if mask is not None:
                cv2.imshow(f"Dream {cfg.camera} Mask", mask)
            if raster is not None:
                cv2.imshow(f"Dream {cfg.camera} Raster", raster)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        step += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(tyro.cli(Config))
