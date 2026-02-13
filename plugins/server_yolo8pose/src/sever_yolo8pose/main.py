from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from rich import print
import torch
import tyro
from ultralytics import YOLO
from ultralytics.engine.results import Results
from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8083
    device: str | None = None  # e.g., "cpu", "cuda:0"

    level: int = 11  # 8 for yolov8 pose
    size: Literal["n", "s", "m", "l", "x"] = "n"


@dataclass
class Y8Payload:
    # boxes: np.ndarray  # shape (N, 4)
    keypoints: dict[np.ndarray] | None  # shape (N, 17, 3)
    masks: np.ndarray | None
    names: dict | None

    orig_img: np.ndarray  # shape (H, W, 3)
    orig_shape: tuple[int, int]  # (H, W)
    path: str
    probs: np.ndarray | None
    save_dir: str
    speed: dict

    @staticmethod
    def from_results(results: Results) -> Y8Payload:
        boxes = results.boxes
        kp = results.keypoints
        keypoints = (
            {
                "xyn": kp.xyn.cpu().numpy(),
                "xy": kp.xy.cpu().numpy(),
                "conf": kp.conf.cpu().numpy(),
            }
            if kp is not None
            else None
        )

        return Y8Payload(
            # boxes=boxes.xyxy.cpu().numpy() if boxes is not None else np.empty((0, 4)),
            scores=boxes.conf.cpu().numpy() if boxes is not None else np.empty((0,)),
            class_ids=boxes.cls.cpu().numpy().astype(int) if boxes is not None else np.empty((0,), dtype=int),
            keypoints=keypoints.xyn.cpu().numpy() if keypoints is not None else np.empty((0, 17, 3)),
        )


class Y8Policy(BasePolicy):
    def __init__(self, cfg: Config):
        self.device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        name = f"yolo{cfg.level}{cfg.size}-pose.pt"
        print("Loading model:", name)
        # name = "yolov8n-pose.torchscript"
        self.model = YOLO(name)
        self.model = self.model.to(device=self.device)

        # self.adapter = TypeAdapter(DA3Payload)

    def step(self, raw: dict) -> dict:
        # payload: DA3Payload = self.adapter.validate_python(raw)
        # print(payload.infer_gs)

        results = self.model(raw["image"])[0]
        kp = results.keypoints
        keypoints = (
            {
                "xyn": kp.xyn.cpu().numpy(),
                "xy": kp.xy.cpu().numpy(),
                "conf": kp.conf.cpu().numpy(),
            }
            if kp
            else {}
        )
        return keypoints

        # payload =  self.oadapter.dump_python({'results':prediction}, mode="python")
        # return payload
        # return prediction


def main(cfg: Config):
    policy = Y8Policy(cfg)
    server = Server(policy, cfg.host, cfg.port)
    server.serve()


if __name__ == "__main__":
    main(tyro.cli(Config))
