import os
import yaml
import torch
from ultralytics import YOLO
from wilor.models import load_wilor

class AttrDict(dict):
    
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)
    def __setattr__(self, key, value):
        self[key] = value
    def get(self, key, default=None):
        return super().get(key, default)

def real_cfg(cfg_path):
 
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)

    class C:
        pass

    C.EXTRA = AttrDict(raw["EXTRA"])
    C.MODEL = AttrDict(raw["MODEL"])
    C.MANO  = AttrDict(raw["MANO"])

    return C


class RealWiLoR:
    def __init__(self, ckpt_path, cfg_path, device):
        self.device = device
        print("[RealWiLoR] Loading WiLoR model...")

        self.model, self.cfg = load_wilor(ckpt_path, cfg_path)
        self.model = self.model.to(device).eval()

        
        self.cfg.defrost()

        mano_root = "/home/lliao2/WiLoR/mano_data"
        self.cfg.MANO.MANO_PATH = mano_root
        self.cfg.MANO.MEAN_PARAMS = f"{mano_root}/mano_mean_params.npz"

        self.cfg.freeze()

        print("[RealWiLoR] Loaded WiLoR successfully.")


        
    def __call__(self, batch):
        with torch.no_grad():
            return self.model(batch)

    def __call__(self, batch):
        with torch.no_grad():
            return self.model(batch)


class RealYOLO:
    """Real YOLO detector wrapper."""

    def __init__(self, detector_path, device):
        print("[RealYOLO] Loading YOLO...")
        self.detector = YOLO(detector_path).to(device)

    def __call__(self, image):
        return self.detector(image)
