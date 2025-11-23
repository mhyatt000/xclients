"""Client bridge for grabbing frames from the camera policy server.

The server at :mod:`src.xclients.server.camera` returns frames as base64-encoded
JPEG strings. This client mirrors the OpenCV :class:`cv2.VideoCapture` API so
callers can request a camera by ID and receive a decoded ``numpy`` image.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import tyro
from webpolicy.client import Client


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 8080
    show: bool = False


def main(cfg: Config) -> None:
    """Request frames from the camera policy server.

    If ``cfg.show`` is true, frames will be displayed in an OpenCV window until
    the "q" key is pressed. Otherwise, a single frame is fetched and its shape
    is logged.
    """

    client = Client(cfg.host, cfg.port)

    logging.info("Displaying frames from camera 0; press 'q' to quit.")
    while True:
        payload = {"id": 0}
        out = client.step(payload)
        if not out:
            logging.error("Failed to read frame from camera 0")
            continue

        frame = (out["image"]).astype(np.uint8)
        if cfg.show:
            cv2.imshow("Camera 0", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    logging.info("Received frame from camera 0 with shape %s", frame.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Config))
