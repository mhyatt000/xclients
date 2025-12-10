from dataclasses import dataclass

import cv2
import torch
import tyro
from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
    WiLorHandPose3dEstimationPipeline,
)

# from server_wilor.server import WilorModel


@dataclass
class WilorConfig:
    host: str = "0.0.0.0"
    port: int = 8007


def demo():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    img_path = "assets/img.png"
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    outputs = pipe.predict(image)
    print(outputs)

    print(len(outputs))
    print(outputs[0]["wilor_preds"].keys())


class WilorPolicy(BasePolicy):
    def __init__(self, cfg: WilorConfig):
        self.cfg = cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dtype = torch.float16
        self.pipe = WiLorHandPose3dEstimationPipeline(device=self.device, dtype=dtype)

    def step(self, payload: dict) -> dict:
        image = payload["image"]
        out = self.pipe.predict(image)
        # TODO use jax.tree.map(lambda *xs: np.stack([xs]) , *out) if multiple batch
        return out[0]


def main(cfg: WilorConfig):
    """Standalone debug mode."""

    server = Server(policy=WilorPolicy(cfg), host=cfg.host, port=cfg.port)
    server.serve()
    # demo()


if __name__ == "__main__":
    main(tyro.cli(WilorConfig))
