from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from rich import print
import tyro
from webpolicy.client import Client

from xclients.core.cfg import Config, spec
from xclients.core.latest_worker import LatestWorker

HAND_EDGES = (
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
)


@dataclass
class MyConfig(Config):
    cap: int | Path = 0
    show: bool = False
    fxy: float | None = None


def project_keypoints_3d(
    keypoints_3d: NDArray[np.float32],
    cam_t: NDArray[np.float64],
    image_shape: tuple[int, int, int],
    fxy: float = 515.0,
) -> NDArray[np.float64]:
    h, w = image_shape[:2]
    points_cam = keypoints_3d.astype(np.float64) + cam_t.astype(np.float64)
    z = np.clip(points_cam[:, 2], 1e-6, None)
    u = fxy * points_cam[:, 0] / z + w / 2
    v = fxy * points_cam[:, 1] / z + h / 2
    return np.stack((u, v), axis=-1)


def draw_keypoints(
    frame: NDArray[np.uint8],
    points: NDArray[np.float64],
    color: tuple[int, int, int],
    radius: int,
    marker: str = "circle",
) -> None:
    h, w = frame.shape[:2]
    pixels = np.rint(points).astype(np.int32)
    valid = (
        np.isfinite(points).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < h)
    )

    for start, end in HAND_EDGES:
        if valid[start] and valid[end]:
            cv2.line(frame, tuple(pixels[start]), tuple(pixels[end]), color, 1, cv2.LINE_AA)

    for point, is_valid in zip(pixels, valid, strict=True):
        if is_valid:
            if marker == "cross":
                x, y = point
                cv2.line(frame, (x - radius, y - radius), (x + radius, y + radius), color, 1, cv2.LINE_AA)
                cv2.line(frame, (x - radius, y + radius), (x + radius, y - radius), color, 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, tuple(point), radius, color, -1, cv2.LINE_AA)


def draw_hand_overlay(frame: NDArray[np.uint8], hand: dict) -> NDArray[np.uint8]:
    overlay = frame.copy()
    keypoints_2d = np.asarray(hand["keypoints_2d"], dtype=np.float64)
    keypoints_3d = np.asarray(hand["keypoints_3d"], dtype=np.float32)
    cam_t = np.asarray(hand["cam_t"], dtype=np.float64)
    fxy = float(hand.get("focal_length", 515.0))
    projected_3d = project_keypoints_3d(keypoints_3d, cam_t, frame.shape, fxy=fxy)

    draw_keypoints(overlay, keypoints_2d, (0, 255, 0), 4, marker="circle")
    draw_keypoints(overlay, projected_3d, (255, 0, 255), 5, marker="cross")
    return overlay


def print_projection_debug(hand: dict, frame: NDArray[np.uint8]) -> None:
    keypoints_3d = np.asarray(hand["keypoints_3d"], dtype=np.float32)
    cam_t = np.asarray(hand["cam_t"], dtype=np.float64)
    fxy = float(hand.get("focal_length", 515.0))
    projected = project_keypoints_3d(keypoints_3d, cam_t, frame.shape, fxy=fxy)
    span = projected.max(axis=0) - projected.min(axis=0)
    print({"cam_t": cam_t, "fxy": fxy, "projected_span": span})


def run_wilor(client: Client, frame: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], dict]:
    payload = {
        "image": frame,
        "type": "image",
    }
    return frame, client.step(payload)


def main(cfg: MyConfig) -> None:
    client = Client(cfg.host, cfg.port)
    cap = cv2.VideoCapture(str(cfg.cap) if isinstance(cfg.cap, Path) else cfg.cap)
    worker = LatestWorker(lambda frame: run_wilor(client, frame), name="wilor-worker")
    last_logged_seq = 0
    last_error_seq = 0

    logging.info("Displaying frames from camera 0; press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame from camera 0")
                continue

            worker.submit(frame.copy())
            display = frame
            result = worker.latest()

            if result is not None and result.error is not None:
                if result.seq != last_error_seq:
                    error = result.error
                    logging.error(
                        "Wilor inference failed",
                        exc_info=(type(error), error, error.__traceback__),
                    )
                    last_error_seq = result.seq
            elif result is not None and result.value is not None:
                result_frame, out = result.value
                display = result_frame
                hands = out.get("hands") or []

                if result.seq != last_logged_seq:
                    print(result_frame.shape)
                    print(spec(out))
                    if not hands:
                        logging.warning("No hand keypoints detected in the frame.")
                    else:
                        hand = hands[0]
                        print(hand["focal_length"])
                        print_projection_debug(hand, result_frame)
                    last_logged_seq = result.seq

                if hands and cfg.show:
                    hand = hands[0]
                    if cfg.fxy is not None:
                        hand = {**hand, "focal_length": cfg.fxy}
                    display = draw_hand_overlay(result_frame, hand)

            if cfg.show or result is None or result.value is None or not (result.value[1].get("hands") or []):
                cv2.imshow("frame", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        worker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(MyConfig))
