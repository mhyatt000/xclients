"""Client bridge for grabbing frames from the camera policy server.

The server at :mod:`src.xclients.server.camera` returns frames as base64-encoded
JPEG strings. This client mirrors the OpenCV :class:`cv2.VideoCapture` API so
callers can request a camera by ID and receive a decoded ``numpy`` image.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from webpolicy.deploy.client import WebsocketClientPolicy


class CameraClient:
    """Lightweight wrapper for the camera policy server.

    Parameters
    ----------
    host:
        Hostname or IP address of the running camera policy server.
    port:
        Port number exposed by the server. Defaults to ``8080`` which matches
        :class:`src.xclients.server.camera.Config`.
    """

    def __init__(self, host: str, port: int = 8080):
        self._client = WebsocketClientPolicy(host=host, port=port)

    def read(self, camera_id: int = 0) -> Tuple[bool, np.ndarray | None]:
        """Request a single frame from a remote camera.

        The request payload mirrors ``cv2.VideoCapture.read`` by returning a
        boolean success flag alongside the decoded frame.
        """

        payload = {"id": camera_id}
        response = self._client.infer(payload)

        image_b64 = response.get("image")
        if image_b64 is None:
            return False, None

        encoded = base64.b64decode(image_b64)
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return False, None

        return True, frame

    def close(self) -> None:
        """Close the underlying websocket client connection if present."""

        if hasattr(self._client, "close"):
            self._client.close()


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

    client = CameraClient(cfg.host, cfg.port)

    if cfg.show:
        logging.info("Displaying frames from camera 0; press 'q' to quit.")
        while True:
            success, frame = client.read(0)
            if not success or frame is None:
                logging.error("Failed to read frame from camera 0")
                break

            cv2.imshow("Camera 0", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
    else:
        success, frame = client.read(0)
        if not success or frame is None:
            logging.error("Failed to read frame from camera 0")
            return

        logging.info("Received frame from camera 0 with shape %s", frame.shape)

    client.close()


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Config))
