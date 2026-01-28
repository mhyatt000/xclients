import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

from dataclasses import dataclass

from webpolicy.base_policy import BasePolicy

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample, visualize_sample_together
from tqdm import tqdm
from tools.build_detector import HumanDetector
from tools.build_sam import HumanSegmentor
from tools.build_fov_estimator import FOVEstimator

@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8080


class Sam3dBodyPolicy(BasePolicy):
    def __init__(self):
        print("Initializing SAM3D Body server...")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        sam3db_root=Path(os.environ["SAM3DB_ROOT"])

        # paths from env 
        ckpt_path = sam3db_root / "checkpoints" / "sam3d_body.ckpt"
        mhr_path = sam3db_root / "assets" / "mhr"
        detector_path = sam3db_root / "pretrained_models" / "vitdet"
        segmentor_path = sam3db_root / "pretrained_models" / "sam2"
        fov_path = sam3db_root / "pretrained_models" / "moge2"

        #load main model 
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

        # print("outputs type:", type(outputs))
        # print("num persons:", len(outputs))

        person = outputs[0]
        
        rendered_img = None
        if render:
            rendered_img = visualize_sample_together(
                image, 
                outputs, 
                self.estimator.faces
            )

            rendered_img = rendered_img.astype(np.uint8)

        mesh_3d={
            "vertices": person["pred_vertices"],
            "faces": self.faces,          
            "joints_3d": person["pred_keypoints_3d"],

            "camera": {
                "translation": person["pred_cam_t"],
                "focal_length": person["focal_length"],
            },
        }

        return {
            "render": rendered_img, 
            "mesh_3d": mesh_3d,
        }


