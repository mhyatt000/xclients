import numpy as np
import torch
from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline


class WilorPolicy(BasePolicy):
    def __init__(self, device=None, dtype=torch.float16):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.pipe = WiLorHandPose3dEstimationPipeline(device=self.device, dtype=dtype)

    def step(self, obs: dict) -> dict:
        """
        obs: {"image": np.ndarray [H, W, 3] uint8 RGB}
        returns: {"hands": list of per-hand dicts}
          each hand: {
            "is_right": bool,
            "hand_bbox": [x1, y1, x2, y2],
            "keypoints_3d": [21, 3],   # meters, wrist-relative
            "keypoints_2d": [21, 2],   # pixels
            "global_orient": [1, 3, 3],
            "hand_pose": [15, 3, 3],
            "cam_t": [3],
            "focal_length": float,
          }
        """
        image = obs["image"]
        detections = self.pipe.predict(image)

        hands = []
        for det in detections:
            hand = {
                "is_right": bool(det["is_right"]),
                "hand_bbox": det["hand_bbox"],
            }
            if "wilor_preds" in det:
                p = det["wilor_preds"]
                hand["keypoints_3d"] = p["pred_keypoints_3d"][0]        # [21, 3]
                hand["keypoints_2d"] = p["pred_keypoints_2d"][0]        # [21, 2]
                hand["global_orient"] = p["global_orient"][0]           # [1, 3, 3]
                hand["hand_pose"] = p["hand_pose"][0]                   # [15, 3, 3]
                hand["cam_t"] = p["pred_cam_t_full"][0]                 # [3]
                hand["focal_length"] = float(p["scaled_focal_length"])
            hands.append(hand)

        return {"hands": hands}

    def reset(self, payload: dict | None = None) -> None:
        pass


if __name__ == "__main__":
    policy = WilorPolicy()
    server = Server(policy=policy, host="0.0.0.0", port=8000)
    server.start()
