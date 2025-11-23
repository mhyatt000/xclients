from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Dict

import cv2
from pydantic import TypeAdapter

from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass
class CameraPayload:
    id: int


class CameraPolicy(BasePolicy):
    def __init__(self):
        self.caps: Dict[int, cv2.VideoCapture] = {}
        self.adapter = TypeAdapter(CameraPayload)

    def step(self, raw: Dict) -> Dict:
        payload: CameraPayload = self.adapter.validate_python(raw)

        cap = self.caps.get(payload.id)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(payload.id)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera {payload.id}")
            self.caps[payload.id] = cap

        success, frame = cap.read()
        if not success:
            raise RuntimeError(f"Could not read frame from camera {payload.id}")

        return {"id": payload.id, "image": frame}


def main(cfg: Config):
    policy = CameraPolicy()
    server = Server(policy, cfg.host, cfg.port)
    server.serve()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Config))
