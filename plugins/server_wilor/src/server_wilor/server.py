import os
import sys
import torch
import cv2
import numpy as np
import tyro
from pathlib import Path

from dataclasses import dataclass

@dataclass
class WilorConfig:

    wilor_root:Path


from webpolicy.base_policy import BasePolicy   

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO
from .dummy import RealWiLoR, RealYOLO, real_cfg
LIGHT_PURPLE=(0.25098039,  0.274117647,  0.65882353)

class WilorModel(BasePolicy):
    def __init__(self):
        print(" Initializing WiLoR server...")
        self.device = torch.device("cpu")
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        wilor_root = Path(os.environ["WILOR_ROOT"])

        ckpt = wilor_root / "pretrained_models" / "wilor_final.ckpt"
        cfg  = wilor_root / "pretrained_models" / "model_config.yaml"
        det  = wilor_root / "pretrained_models" / "detector.pt"

        self.model = RealWiLoR(ckpt, cfg, self.device)
        self.cfg = real_cfg(cfg)
        self.detector = RealYOLO(det, self.device)
        self.renderer = Renderer(
            self.cfg,
            faces = self.model.model.mano.faces,
            
        )

    def render_result(self, image_rgb, preds):
        if preds is None or preds["pred_vertices"].shape[0] == 0:
            return image_rgb

        verts = preds["pred_vertices"][0]   
        cam_t = preds["pred_cam_t"][0]       
        is_right = preds["pred_right"][0]
        focal = preds["focal_length"][0]

        if isinstance(focal, torch.Tensor):
            focal = focal.detach().cpu().numpy()
        focal_length = float(np.max(focal))

        H, W = image_rgb.shape[:2]

        rgba = self.renderer.render_rgba_multiple(
            [verts],
            cam_t=[cam_t],
            render_res=(W, H),
            is_right=[is_right],
            mesh_base_color=LIGHT_PURPLE,
            scene_bg_color=(1, 1, 1),
            focal_length=focal_length,
        )

        rgb = rgba[..., :3]
        alpha = rgba[..., 3:4]

        img_f = image_rgb.astype(np.float32) / 255.0

        out = img_f * (1 - alpha) + rgb * alpha
        return (out * 255).astype(np.uint8)


    def detect_hands(self, image):
        detections = self.detector(image)[0]

        if len(detections) == 0:
            return None, None

        boxes = []
        is_right = []

        for det in detections:
            boxes.append(det.boxes.data[..., :4])
            is_right.append(det.boxes.cls.long())

        boxes = torch.cat(boxes, dim=0)      
        is_right = torch.cat(is_right, dim=0)  

        return boxes, is_right



    def preprocess(self, image, boxes, is_right, rescale_factor):

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        if isinstance(is_right, torch.Tensor):
            is_right = is_right.detach().cpu().numpy()

        dataset = ViTDetDataset(
            self.cfg,
            image,
            boxes,
            is_right,
            rescale_factor=rescale_factor
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        return dataloader



    def infer(self, batch):
        
        batch = recursive_to(batch, self.device)
        
        with torch.no_grad():
            out = self.model(batch)
            print("infer output keys:", out.keys())
        return out



    def postprocess(self, out, batch):
        device = out['pred_cam'].device

        multiplier = (2 * batch['right'] - 1).to(device)
        pred_cam = out['pred_cam']
        pred_cam[:, 1] *= multiplier

        box_center = batch["box_center"].float().to(device)
        box_size   = batch["box_size"].float().to(device)
        img_size   = batch["img_size"].float().to(device)

        focal_length = (
            self.cfg.EXTRA.FOCAL_LENGTH
            / self.cfg.MODEL.IMAGE_SIZE
            * img_size.max()
        )

        cam_t = cam_crop_to_full(
            pred_cam,
            box_center,
            box_size,
            img_size,
            focal_length
        )  

        verts = out['pred_vertices']
        verts[..., 0] *= multiplier[:, None]

        return {
            "verts": verts,          
            "cam_t": cam_t,          
            "right": batch['right'], 
            "focal_length": focal_length,  
        }


    def step(self, payload: dict, render: bool = False):
        image = payload["image"]
        H, W, _ = image.shape

        boxes, is_right = self.detect_hands(image)
        if boxes is None or len(boxes) == 0:
            return {"boxes": [], "right": [], "overlay": None}

        boxes = boxes.float()   
        is_right = is_right.long()

        dataloader = self.preprocess(image, boxes, is_right, rescale_factor=2.0)

        rendered = None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target_idx = 0  # 0 is left; 1 is right

        for i, batch in enumerate(dataloader):
            if i != target_idx:
                continue

            out = self.infer(batch)
            pp = self.postprocess(out, batch)

            if render:
                verts = pp["verts"][0].detach().cpu().numpy()
                cam_t = pp["cam_t"][0].detach().cpu().numpy()
                right = int(pp["right"][0].item())
                focal = pp["focal_length"]

                rendered = self.render_result(
                    image_rgb=image_rgb,
                    preds={
                        "pred_vertices": np.array([verts]),
                        "pred_cam_t": np.array([cam_t]),
                        "pred_right": np.array([right]),
                        "focal_length": np.array([[focal, focal]]),
                    },
                )
            break  

        return {
            "render": rendered,
        }
