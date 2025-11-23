from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Dict

import cv2
from pydantic import TypeAdapter

from webpolicy.base_policy import BasePolicy
from webpolicy.server.server import Server


@dataclass
class Config:
    host: str
    port: int = 8080


@dataclass
class CameraPayload:
    id: int


class CameraPolicy(BasePolicy):
    def __init__(self):
        self.caps: Dict[int, cv2.VideoCapture] = {}
        self.adapter = TypeAdapter(CameraPayload)

    def infer(self, raw: Dict) -> Dict:
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

        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Could not encode frame as JPEG")

        image_b64 = base64.b64encode(encoded).decode("ascii")
        return {"id": payload.id, "image": image_b64}


def main(cfg: Config):
    policy = CameraPolicy()
    server = Server(policy, cfg.host, cfg.port)
    server.serve()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Config))
