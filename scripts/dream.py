from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import logging
import time

import cv2
import numpy as np
from rich import print
import tyro
from webpolicy.client import Client

from xclients.core.cfg import Config, spec


@dataclass
class DreamConfig(Config):
    camera: str | int = 0  # OpenCV camera index to poll from
    show: bool = True  # Display the live camera feed while polling
    conf: float = 0.5  # Minimum confidence required to draw a keypoint
    q: list[float] = field(default_factory=lambda: [0.0] * 7)  # Joint vector to attach to the payload
    deg2rad: bool = False  # Convert cfg.q from degrees to radians before sending
    image_size: int = 200  # Resize frames to a square image before sending to the model
    fx: float = 515.0  # Focal length in pixels along x for the payload K matrix
    fy: float = 515.0  # Focal length in pixels along y for the payload K matrix


def draw_keypoints(
    frame: np.ndarray,
    keypoints_norm: np.ndarray | None,
    confidence: np.ndarray | None,
    conf_thr: float,
) -> np.ndarray:
    annotated = frame.copy()
    if keypoints_norm is None:
        return annotated

    pts = np.asarray(keypoints_norm)
    if pts.ndim == 3:
        pts = pts[0]
    if pts.ndim != 2 or pts.shape[-1] != 2:
        raise ValueError(f"Expected keypoints_norm with shape (n, 2) or (1, n, 2), got {pts.shape}")

    h, w = frame.shape[:2]
    pts = pts.astype(np.float32).copy()
    pts[:, 0] *= w
    pts[:, 1] *= h

    conf = None
    if confidence is not None:
        conf = np.asarray(confidence)
        if conf.ndim == 2:
            conf = conf[0]
        if conf.ndim != 1 or conf.shape[0] != pts.shape[0]:
            raise ValueError(f"Expected confidence to align with keypoints, got {conf.shape} for {pts.shape}")

    for i, xy in enumerate(pts):
        alpha = 1.0
        color = (0, 0, 255)
        if conf is not None:
            value = float(np.clip(conf[i], 0.0, 1.0))
            alpha = value
            if value <= 0.5:
                t = value / 0.5
                color = (255 * (1.0 - t), 255 * t, 0)
            else:
                t = (value - 0.5) / 0.5
                color = (0, 255 * (1.0 - t), 255 * t)
            color = tuple(round(c) for c in color)
            if alpha <= 0.0:
                continue
        x, y = int(xy[0]), int(xy[1])
        overlay = annotated.copy()
        cv2.circle(overlay, (x, y), 5, color, -1)
        cv2.circle(overlay, (x, y), 7, (255, 255, 255), 2)
        cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0.0, dst=annotated)

    return annotated


def draw_mask(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None

    arr = np.asarray(mask)
    if arr.ndim >= 3:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
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
        raise ValueError(f"Expected raster_image with shape (h, w), (h, w, c), or batched variants, got {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        maxv = float(arr.max()) if arr.size else 0.0
        if maxv <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def project_w2c_point(
    w2c: np.ndarray | None,
    k: np.ndarray,
    image_shape: tuple[int, ...] | None = None,
) -> tuple[float, float] | None:
    if w2c is None:
        return None

    pose = np.asarray(w2c, dtype=np.float64)
    if pose.ndim == 3:
        if pose.shape[0] != 1:
            raise ValueError(f"Expected a single batched w2c pose, got {pose.shape}")
        pose = pose[0]
    if pose.shape != (4, 4):
        raise ValueError(f"Expected w2c with shape (4, 4), got {pose.shape}")

    k_mat = np.asarray(k, dtype=np.float64)
    if k_mat.shape != (3, 3):
        raise ValueError(f"Expected k with shape (3, 3), got {k_mat.shape}")

    world_row = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    world_col = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    def _pix_from_cam(cam_xyz: np.ndarray, *, flip_y: bool, neg_z: bool) -> tuple[float, float] | None:
        z = float(cam_xyz[2])
        depth = -z if neg_z else z
        if depth <= 1e-6:
            return None
        x = float(k_mat[0, 0] * (float(cam_xyz[0]) / depth) + k_mat[0, 2])
        y_proj = float(k_mat[1, 1] * (float(cam_xyz[1]) / depth) + k_mat[1, 2])
        y = float(2.0 * k_mat[1, 2] - y_proj) if flip_y else y_proj
        return x, y

    candidates: list[tuple[float, float]] = []
    poses = [pose]
    with contextlib.suppress(np.linalg.LinAlgError):
        poses.append(np.linalg.inv(pose))

    for mat in poses:
        row_cam = world_row @ mat
        col_cam = mat @ world_col
        for cam_xyz in (row_cam[:3], col_cam[:3]):
            for flip_y, neg_z in ((True, True), (False, False)):
                xy = _pix_from_cam(cam_xyz, flip_y=flip_y, neg_z=neg_z)
                if xy is not None and np.isfinite(xy[0]) and np.isfinite(xy[1]):
                    candidates.append(xy)

    if not candidates:
        return None

    if image_shape is not None:
        h, w = image_shape[:2]
        for x, y in candidates:
            if 0.0 <= x < w and 0.0 <= y < h:
                return x, y

    return candidates[0]


def draw_cross(img: np.ndarray, xy: tuple[float, float] | None, color: tuple[int, int, int]) -> np.ndarray:
    if xy is None:
        return img
    if not np.isfinite(xy[0]) or not np.isfinite(xy[1]):
        return img

    annotated = img.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    center = (round(xy[0]), round(xy[1]))
    print(f"Drawing cross at {center} with color {color}")
    size = 5
    cv2.line(
        annotated,
        (center[0] - size, center[1] - size),
        (center[0] + size, center[1] + size),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        annotated,
        (center[0] - size, center[1] + size),
        (center[0] + size, center[1] - size),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        annotated,
        (center[0] - size, center[1] - size),
        (center[0] + size, center[1] + size),
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        annotated,
        (center[0] - size, center[1] + size),
        (center[0] + size, center[1] - size),
        color,
        1,
        cv2.LINE_AA,
    )
    return annotated


def draw_w2c_point(img: np.ndarray, w2c: np.ndarray | None, k: np.ndarray) -> np.ndarray:
    return draw_cross(img, project_w2c_point(w2c, k, img.shape), (255, 0, 255))


def scale_intrinsics(k: np.ndarray, sx: float, sy: float) -> np.ndarray:
    scaled = np.asarray(k, dtype=np.float32).copy()
    scaled[0, 0] *= sx
    scaled[1, 1] *= sy
    scaled[0, 2] *= sx
    scaled[1, 2] *= sy
    return scaled


def main(cfg: DreamConfig) -> None:
    client = Client(cfg.host, cfg.port)
    cap = cv2.VideoCapture(cfg.camera)
    step_times: list[float] = []

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {cfg.camera}")

    logging.info("Polling camera %s and sending frames to %s:%s", cfg.camera, cfg.host, cfg.port)
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to read frame from camera %s", cfg.camera)
            continue

        h, w = frame.shape[:2]
        frame_model = cv2.resize(frame, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_LINEAR)
        q = np.asarray(cfg.q, dtype=np.float32)
        if cfg.deg2rad:
            q = np.deg2rad(q)
        k_orig = np.array(
            [
                [cfg.fx, 0.0, w / 2.0],
                [0.0, cfg.fy, h / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        sx = cfg.image_size / float(w)
        sy = cfg.image_size / float(h)
        k_model = scale_intrinsics(k_orig, sx, sy)

        payload = {
            "image": frame_model,
            "type": "image",
            "q": q,
            "K": k_model,
        }

        print(spec(payload))
        t0 = time.perf_counter()
        out = client.step(payload)
        dt = time.perf_counter() - t0
        step_times.append(dt)
        if len(step_times) > 100:
            step_times.pop(0)
        if len(step_times) == 100:
            mean_s = sum(step_times) / len(step_times)
            mean_hz = 1.0 / mean_s if mean_s > 0 else float("inf")
            print({"step_mean_s": mean_s, "step_mean_hz": mean_hz, "window": len(step_times)})

        print(spec(out) if out is not None else None)
        print(out["w2c"] if out and "w2c" in out else None)
        print(out["mask_iou_reject"])
        print({"mask_iou": out["mask_iou"]})
        print(out["pnp_success"])

        annotated = draw_keypoints(
            frame,
            out.get("keypoints_norm") if out else None,
            out.get("confidence") if out else None,
            cfg.conf,
        )
        annotated = draw_w2c_point(annotated, out.get("w2c") if out else None, k_orig)
        mask = draw_mask(out.get("mask")[0] if "mask" in out else None)
        if mask is not None:
            mask = cv2.resize(mask, (400, 400), interpolation=cv2.INTER_NEAREST)
            k_mask = scale_intrinsics(k_model, 400.0 / cfg.image_size, 400.0 / cfg.image_size)
            mask = draw_w2c_point(mask, out.get("w2c") if out else None, k_mask)
        raster = draw_output_image(out.get("raster_image") if out else None)
        if raster is not None:
            rh, rw = raster.shape[:2]
            k_raster = scale_intrinsics(k_model, rw / float(cfg.image_size), rh / float(cfg.image_size))
            raster = draw_w2c_point(raster, out.get("w2c") if out else None, k_raster)

        if cfg.show:
            cv2.imshow(f"Camera {cfg.camera}", annotated)
            if mask is not None:
                cv2.imshow(f"Mask {cfg.camera}", mask)
            if raster is not None:
                cv2.imshow(f"Raster {cfg.camera}", raster)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(DreamConfig))
