import sys
from pathlib import Path

from server_wilor.server import WilorModel
from dataclasses import dataclass
import tyro
import torch
import cv2
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline


@dataclass
class WilorConfig:
    # 未来可以添加更多参数，比如模型路径、是否启用GPU等
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
    print(outputs[0]['wilor_preds'].keys())

def main(cfg: WilorConfig):
    """Standalone debug mode."""
    print("[main] Running WilorModel standalone...")

    demo()
    # model = load(cfg)

    # TODO：这里可以加入你自己的调试图片路径
    # image = cv2.imread("test.jpg")
    # result = model.step(image)
    # print(result)

    print("[main] WilorModel loaded successfully (debug mode).")


if __name__ == "__main__":
    main(tyro.cli(WilorConfig))
