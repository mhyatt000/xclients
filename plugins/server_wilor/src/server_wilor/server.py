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
    image: str = None


PLUGIN_DIR = Path(__file__).parent
ROOT_DIR   = (PLUGIN_DIR / "../../../../").resolve()         
WILOR_ROOT = (ROOT_DIR / "external/wilor").resolve()         
sys.path.append(str(WILOR_ROOT))

WEBPOLICY_SRC = (PLUGIN_DIR / "../../external/webpolicy/src").resolve()
sys.path.append(str(WEBPOLICY_SRC))

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = WILOR_ROOT / "pretrained_models" / "wilor_final.ckpt"
        cfg  = WILOR_ROOT / "pretrained_models" / "model_config.yaml"
        det  = WILOR_ROOT / "pretrained_models" / "detector.pt"

        self.model = RealWiLoR(ckpt, cfg, self.device)
        self.cfg = real_cfg(cfg)
        self.detector = RealYOLO(det, self.device)
        self.renderer = Renderer(
            self.cfg,
            faces = self.model.model.mano.faces,
            
        )



    def render_result(self, image_rgb, preds, batch):
        
        if preds is None or preds["pred_vertices"].shape[0] == 0:
            return image_rgb

        cam_crop = preds["pred_cam_t"]            # (B, 3) 
        verts_t  = preds["pred_vertices"]         # (B, V, 3)

#       batch geometry 
        box_center = batch["box_center"]
        box_size   = batch["box_size"]
        img_size   = batch["img_size"]

#       align device here, render mixes torch + numpy(avoid mismatch)
        device = cam_crop.device
        box_center = box_center.to(device)
        box_size   = box_size.to(device)
        img_size   = img_size.to(device)

        cam_t = cam_crop_to_full(
            cam_crop,
            box_center,
            box_size,
            img_size,
        )       

#       take 1 hand and reverse to gets another
        verts = verts_t[0].detach().cpu().numpy() 
        cam_t = cam_t[0].detach().cpu().numpy()    

        is_right = batch["right"][0].item()
        verts[:, 0] *= (2 * is_right - 1)

        joints = preds.get("pred_keypoints_3d", None)
        if joints is not None:
            joints = joints[0].detach().cpu().numpy()
            joints[:, 0] *= (2 * is_right - 1)

 #      render (numpy only)
        H, W = image_rgb.shape[:2]

        rgba = self.renderer.render_rgba(
            verts,
            cam_t=cam_t,
            render_res=(H, W),
        )

        rgb   = rgba[..., :3]
        alpha = rgba[..., 3:4]

#       renderer may return (W, H), align to image shape
        if rgb.shape[0] != image_rgb.shape[0] or rgb.shape[1] != image_rgb.shape[1]:
            rgb   = np.transpose(rgb, (1, 0, 2))
            alpha = np.transpose(alpha, (1, 0, 2))

        img_f = image_rgb.astype(np.float32) / 255.0
        rgb_f = rgb.astype(np.float32)
        if rgb_f.max() > 1.5:
            rgb_f /= 255.0

        overlay = img_f * (1.0 - alpha) + rgb_f * alpha
#       demo is cv2, automatically convert the input to uint8 here
        return (overlay * 255.0).clip(0, 255).astype(np.uint8) 



    def detect_hands(self, image):
        detections = self.detector(image)[0]
        boxes = []
        is_right = []
        for det in detections:
            box = det.boxes.data.cpu().numpy().squeeze()[:4]
            hand_type = int(det.boxes.cls.cpu().item())  # 0=left, 1=right
            boxes.append(box)
            is_right.append(hand_type)
        return np.array(boxes), np.array(is_right)



    def preprocess(self, image, boxes, is_right, rescale_factor):
        dataset = ViTDetDataset(
            self.cfg,
            image,
            boxes,
            is_right,
            rescale_factor=rescale_factor
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0
        )
        return dataloader



    def infer(self, batch):
        
        batch = recursive_to(batch, self.device)
        
        with torch.no_grad():
            out = self.model(batch)
            print("infer output keys:", out.keys())
        return out



    def postprocess(self, out, batch):
 
        multiplier = (2 * batch['right'] - 1).to(out['pred_cam'].device)

        pred_cam = out['pred_cam']
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]

        box_center = batch["box_center"].float().to(pred_cam.device)
        box_size   = batch["box_size"].float().to(pred_cam.device)
        img_size   = batch["img_size"].float().to(pred_cam.device)

        scaled_focal_length = (
            self.cfg.EXTRA.FOCAL_LENGTH
            / self.cfg.MODEL.IMAGE_SIZE
            * img_size.max()
        )

        pred_cam_t_full = cam_crop_to_full(
            pred_cam,
            box_center,
            box_size,
            img_size,
            scaled_focal_length
        ).detach().cpu().numpy()

        all_verts = []
        all_cam_t = []
        all_right = []

        batch_size = batch["img"].shape[0]

        for n in range(batch_size):
            verts = out['pred_vertices'][n].detach().cpu().numpy()

            is_right = batch['right'][n].cpu().numpy()
            verts[:, 0] = (2 * is_right - 1) * verts[:, 0]

            cam_t = pred_cam_t_full[n]

            all_verts.append(verts)
            all_cam_t.append(cam_t)
            all_right.append(is_right)

        return all_verts, all_cam_t, all_right



    def step(self, payload: dict, render: bool = False):
        image = payload["image"]

        dataloader = self.preprocess(image, boxes, is_right, rescale_factor=2.0)

        verts_list, cams_list, right_list = [], [], []
        rendered = None

        for batch in dataloader:
            out = self.infer(batch)
            verts, cams, rights = self.postprocess(out, batch)

            verts_list.extend(verts)
            cams_list.extend(cams)
            right_list.extend(rights)

            #use for debug
            if render:
                # render need RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # render uses geometry info from batch
                rendered = self.render_result(
                    image_rgb=image_rgb,
                    preds={
                        "pred_vertices": out["pred_vertices"],
                        "pred_cam_t": out["pred_cam_t"],
                        "pred_keypoints_3d": out.get("pred_keypoints_3d", None),
                        "pred_right": batch["right"],
                    },
                    batch=batch,
                )           

        result = {
            "status": "ok",
            "verts": verts_list,
            "cams": cams_list,
            "right": right_list,
        }

        if render:
            result["render"] = rendered

        return result



