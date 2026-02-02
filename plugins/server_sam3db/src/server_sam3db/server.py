from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from rich import print
import torch
from webpolicy.base_policy import BasePolicy

from server_sam3db.patch_import import (
    FOVEstimator,
    HumanDetector,
    HumanSegmentor,
    load_sam_3d_body,
    SAM3DBodyEstimator,
    visualize_sample_together,
)

class Sam3dBodyPolicy(BasePolicy):
    def __init__(self,root:Path):
        print("Initializing SAM3D Body server...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # paths from SAM3DB root
        ckpt_path = root / "checkpoints" / "sam3d_body.ckpt"
        mhr_path = root / "assets" / "mhr"
        detector_path = root / "pretrained_models" / "vitdet"
        segmentor_path = root / "pretrained_models" / "sam2"
        fov_path = root / "pretrained_models" / "moge2"

        # load main model
        model, model_cfg = load_sam_3d_body(
            str(ckpt_path),
            device=self.device,
            mhr_path=str(mhr_path),
        )

        self.detector = (
            HumanDetector(
                name="vitdet",
                device=self.device,
                path=str(detector_path),
            )
            if detector_path.exists()
            else None
        )

        self.segmentor = (
            HumanSegmentor(
                name="sam2",
                device=self.device,
                path=str(segmentor_path),
            )
            if segmentor_path.exists()
            else None
        )

        self.fov_estimator = (
            FOVEstimator(
                name="moge2",
                device=self.device,
                path=str(fov_path),
            )
            if fov_path.exists()
            else None
        )

        #  estimator
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=self.detector,
            human_segmentor=self.segmentor,
            fov_estimator=self.fov_estimator,
        )
        self.faces = self.estimator.faces

        print("Model loaded successfully!")

    def step(self, payload: dict, render: bool = False):
        image = payload["image"]

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        outputs = self.estimator.process_one_image(
            image,
            bbox_thr=0.8,
            use_mask=False,
        )

        person = outputs[0]

        def to_numpy_safe(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            return x

        rendered_img = None
        if render:
            rendered_img = visualize_sample_together(image, outputs, self.estimator.faces)

            rendered_img = rendered_img.astype(np.uint8)
        
        mesh_3d = {
            "bbox": to_numpy_safe(person["bbox"]),
            "focal_length": float(person["focal_length"]),

            "pred_vertices": to_numpy_safe(person["pred_vertices"]),
            "pred_keypoints_3d": to_numpy_safe(person["pred_keypoints_3d"]),
            "pred_keypoints_2d": to_numpy_safe(person["pred_keypoints_2d"]),
            "pred_joint_coords": to_numpy_safe(person["pred_joint_coords"]),

            "pred_cam_t": to_numpy_safe(person["pred_cam_t"]),
            "scale_params": to_numpy_safe(person.get("scale_params")),

            "global_rot": to_numpy_safe(person["global_rot"]),
            "pred_global_rots": to_numpy_safe(person["pred_global_rots"]),
            "pred_pose_raw": to_numpy_safe(person["pred_pose_raw"]),
            "body_pose_params": to_numpy_safe(person["body_pose_params"]),
            "hand_pose_params": to_numpy_safe(person["hand_pose_params"]),

            "shape_params": to_numpy_safe(person["shape_params"]),
            "expr_params": to_numpy_safe(person["expr_params"]),
            "mhr_model_params": to_numpy_safe(person["mhr_model_params"]),

            "mask": to_numpy_safe(person["mask"]),
            "lhand_bbox": to_numpy_safe(person["lhand_bbox"]),
            "rhand_bbox": to_numpy_safe(person["rhand_bbox"]),
        }

        return {
            "render": rendered_img,
            "mesh_3d": mesh_3d,
        }