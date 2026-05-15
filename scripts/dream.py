from __future__ import annotations

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
    q_degrees: bool = True  # Convert cfg.q from degrees to radians before sending
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
        if conf is not None and conf[i] <= conf_thr:
            continue
        x, y = int(xy[0]), int(xy[1])
        cv2.circle(annotated, (x, y), 5, (0, 0, 255), -1)
        cv2.circle(annotated, (x, y), 7, (255, 255, 255), 2)

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


def project_world_origin(w2c: np.ndarray | None, k: np.ndarray) -> tuple[float, float] | None:
    if w2c is None:
        return None

    w2c_arr = np.asarray(w2c, dtype=np.float64)
    if w2c_arr.ndim == 3:
        w2c_arr = w2c_arr[0]
    if w2c_arr.shape != (4, 4):
        raise ValueError(f"Expected w2c with shape (4, 4) or (1, 4, 4), got {w2c_arr.shape}")

    world = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    cam = w2c_arr @ world
    depth = cam[2]
    if depth <= 1e-6:
        return None

    norm = np.array([cam[0] / depth, cam[1] / depth, 1.0], dtype=np.float64)
    pix = np.asarray(k, dtype=np.float64) @ norm
    x = float(pix[0])
    y = float(pix[1])
    return x, y


def draw_cross(img: np.ndarray, xy: tuple[float, float] | None, color: tuple[int, int, int]) -> np.ndarray:
    if xy is None:
        return img

    annotated = img.copy()
    center = (round(xy[0]), round(xy[1]))
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
        q = np.asarray(cfg.q, dtype=np.float32)
        if cfg.q_degrees:
            q = np.deg2rad(q)
        K = np.array(
            [
                [cfg.fx, 0.0, w / 2.0],
                [0.0, cfg.fy, h / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        payload = {
            "image": frame,
            "type": "image",
            "q": q,
            "K": K,
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

        annotated = draw_keypoints(
            frame,
            out.get("keypoints_norm") if out else None,
            out.get("confidence") if out else None,
            cfg.conf,
        )
        annotated = draw_cross(annotated, project_world_origin(out.get("w2c") if out else None, K), (255, 0, 255))
        mask = draw_mask(out.get("mask")[0] if "mask" in out else None)

        if cfg.show:
            cv2.imshow(f"Camera {cfg.camera}", annotated)
            if mask is not None:
                cv2.imshow(f"Mask {cfg.camera}", mask)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(DreamConfig))
