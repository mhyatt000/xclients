import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

VIDEO_URL = "https://github.com/warmshao/WiLoR-mini/raw/main/assets/video.mp4"
VIDEO_PATH = Path("assets/video.mp4")


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16

    VIDEO_PATH.parent.mkdir(exist_ok=True)
    if not VIDEO_PATH.exists():
        print(f"Downloading {VIDEO_URL} ...")
        subprocess.run(["curl", "-L", "-o", str(VIDEO_PATH), VIDEO_URL], check=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    ret, frame = cap.read()
    cap.release()
    assert ret, f"Could not read frame from {VIDEO_PATH}"
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    outputs = pipe.predict(image)
    print(outputs)



if __name__ == "__main__":
    main()
