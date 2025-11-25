# sam3_server.py
import os
import torch
import cv2
import numpy as np
from webpolicy import ModelServer, register
from typing import List

import sam3
from sam3.model_builder import build_sam3_video_model
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
)

@register("sam3")  # this exposes your server as a webpolicy service
class SAM3VideoServer(ModelServer):
    """
    A WebPolicy server that exposes the SAM3 video tracker as an online service.
    """

    def load(self):
        """
        Load the SAM3 video model once when the server starts.
        """
        print("Loading SAM3 video model...")
        self.model = build_sam3_video_model()
        self.predictor = self.model.tracker
        self.predictor.backbone = self.model.detector.backbone
        print("SAM3 model loaded successfully.")

    def predict(self, video_path: str) -> List[np.ndarray]:
        """
        Run SAM3 video segmentation on an input video file.

        Args:
            video_path: str - the path to a video file accessible to this server

        Returns:
            List of binary mask frames (as numpy arrays)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video")

        all_masks = []

        # Process video frame by frame
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run SAM3 model
            frame_data = load_frame(frame_rgb)
            outputs = self.predictor(frame_data)

            # Extract masks
            masks = outputs.get("masks", None)
            if masks is not None:
                masks = prepare_masks_for_visualization(masks)
                all_masks.append(masks)
            else:
                all_masks.append(None)

            idx += 1

        cap.release()
        return all_masks


# “python sam3_server.py” 的入口
if __name__ == "__main__":
    SAM3VideoServer.run()
