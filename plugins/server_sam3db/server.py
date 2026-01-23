import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

from dataclasses import dataclass

# @dataclass
# class Sam3dbConfig:

#     Sam3db_root:Path

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

# paths from env 环境路径
        ckpt_path = sam3db_root / "checkpoints" / "sam3d_body.ckpt"
        mhr_path = sam3db_root / "assets" / "mhr"
        detector_path = sam3db_root / "pretrained_models" / "vitdet"
        segmentor_path = sam3db_root / "pretrained_models" / "sam2"
        fov_path = sam3db_root / "pretrained_models" / "moge2"

        #load main model 加载模型，命令全部写死
        model, model_cfg = load_sam_3d_body(
            str(ckpt_path),
            device=self.device,
            mhr_path=str(mhr_path),
        )
        
        self.detector = HumanDetector(
            name="vitdet", 
            device=self.device, 
            path=str(detector_path)
        )
        self.segmentor = HumanSegmentor(
            name="sam2", 
            device=self.device, 
            path=str(segmentor_path)
        )
        self.fov_estimator = FOVEstimator(
            name="moge2", 
            device=self.device, 
            path=str(fov_path)
        )
       
        #  estimator 工具管家
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

        # 简写了原来的infer 
        outputs = self.estimator.process_one_image(
            image,
            bbox_thr=0.8,
            use_mask=False,
        )

        rendered_img = None
        if render:
            rendered_img = visualize_sample_together(
                image, 
                outputs, 
                self.estimator.faces
            )

            rendered_img = rendered_img.astype(np.uint8)

        return {
            "render": rendered_img 
            #将来要输出3D数据吗？？！！问Matt！
            #"3d_data": outputs 或者 细节一点比如 vertices、joints...
        }


