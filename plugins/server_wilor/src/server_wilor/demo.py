import tyro
import torch
import cv2
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
import numpy as np


def demo(img: np.ndarray):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    outputs = pipe.predict(img)
    print(outputs)

    print(len(outputs))
    print(outputs[0]['wilor_preds'].keys())
