from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Iterator

import cv2
import numpy as np
from rich import print
import tyro
from webpolicy.client import Client

# Import your existing client/config tools
from xclients.core.cfg import Config, spec


# --- Abstract Base Class ---
@dataclass
class IO(ABC):
    @abstractmethod
    def create(self) -> IO: ...

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]: ...


@dataclass
class Camera(IO):
    id: int = 0
    _cap: cv2.VideoCapture | None = field(default=None, init=False, repr=False)

    def create(self) -> Camera:
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.id)
            if not self._cap.isOpened():
                raise RuntimeError(f"cv2.VideoCapture({self.id}) failed to open")
        return self

    def __iter__(self) -> Iterator[np.ndarray]:
        self.create()
        assert self._cap is not None

        logging.info(f"Streaming from Camera {self.id}...")
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame

        self._cap.release()


@dataclass
class NPZ(IO):
    path: Path
    view: str = "side"
    _data: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def create(self) -> NPZ:
        if self._data is None:
            if not self.path.exists():
                raise FileNotFoundError(f"NPZ file not found: {self.path}")
            self._data = dict(np.load(self.path, allow_pickle=True))
        return self

    def __iter__(self) -> Iterator[np.ndarray]:
        self.create()
        assert self._data is not None

        if self.view in self._data:
            raw_data = self._data[self.view]
            logging.info(f"Streaming '{self.view}' view from {self.path} (Shape: {raw_data.shape})...")
        else:
            valid_keys = [k for k in ["side", "over", "low", "image", "rgb", "images"] if k in self._data]
            if not valid_keys:
                raise KeyError(f"Could not find view '{self.view}'. Available keys: {list(self._data.keys())}")

            logging.warning(f"View '{self.view}' not found. Defaulting to '{valid_keys[0]}'")
            raw_data = self._data[valid_keys[0]]

        for frame in raw_data:
            frame = frame.squeeze()

            if frame.ndim != 3:
                if frame.shape[0] == 3:
                    frame = np.transpose(frame, (1, 2, 0))
                else:
                    logging.warning(f"Skipping frame with unexpected shape: {frame.shape}")
                    continue

            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

            yield frame
            time.sleep(0.1)  # Simulate framerate


@dataclass
class SAMConfig(Config):
    """
    Main configuration for SAM Client.
    Tyro automatically creates subcommands (input:camera / input:npz) based on the IO subclasses.
    """

    input: Camera | NPZ = field(default_factory=lambda: Camera(id=0))
    prompt: str | None = None
    confidence: float = 0.5
    show: bool = True

    def __post_init__(self):
        if self.prompt is None:
            logging.warning("No prompt provided. SAM might return empty masks.")


def main(cfg: SAMConfig) -> None:
    client = Client(cfg.host, cfg.port)

    for frame in cfg.input:
        if frame is None:
            continue

        print(f"Frame shape: {frame.shape}")

        payload = {
            "image": frame,
            "type": "image",
            "text": cfg.prompt or "object",
            "confidence": cfg.confidence,
        }

        out = client.step(payload)

        if not out:
            logging.error("Failed to get response from client")
            continue

        print(spec(out))

        if cfg.show:
            display_frame = frame.copy()

            if out.get("masks") is not None:
                mask_sum = out["masks"].sum(0)

                if mask_sum.ndim > 2:
                    mask_sum = mask_sum[0]

                d = mask_sum.astype(np.uint8) * 255
                cv2.imshow("Mask", d)

            cv2.imshow("Input", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(SAMConfig))
