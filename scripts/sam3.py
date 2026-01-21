from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import tyro
from xclients.core.cfg import Config, spec
from rich import print
from webpolicy.client import Client


@dataclass
class SAMConfig(Config):
    prompt: str | None = None
    confidence: float = 0.5

    def __post_init__(self):
        assert self.prompt is not None, "Prompt must be provided for SAM model."


def main(cfg: SAMConfig) -> None:
    client = Client(cfg.host, cfg.port)
    cap = cv2.VideoCapture(0)

    logging.info("Displaying frames from camera 0; press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        print(frame.shape)
        payload = {
            "image": frame,
            "type": "image",
            "text": cfg.prompt,
            "confidence": cfg.confidence,
        }
        out = client.step(payload)
        if not out:
            logging.error("Failed to read frame from camera 0")
            continue

        print(spec(out))
        if out.get("masks") is None:
            if cfg.show:
                cv2.imshow("Camera 0", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        d = out["masks"].sum(0)[0].astype(np.uint8) * 255
        print(d.shape)
        # d = ((d - np.min(d)) / (np.max(d) - np.min(d)) * 255.0).astype(np.uint8)
        # d = 255 - d

        # print(out["extrinsics"])
        # print(out["intrinsics"])

        if cfg.show:
            cv2.imshow("Camera 0", d)
            cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(SAMConfig))
