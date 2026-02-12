import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro
from webpolicy.base_policy import BasePolicy

os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", "")
os.environ["LIDRA_SKIP_INIT"] = "true"

sys.path.insert(0, "/home/nhogg/sam-3d-objects")
sys.path.insert(0, "/home/nhogg/sam-3d-objects/notebook")

from inference import Inference, check_hydra_safety, BLACKLIST_FILTERS, WHITELIST_FILTERS
from omegaconf import OmegaConf


class Policy(BasePolicy):
    def __init__(self, cfg):
        self.cfg = cfg

        config_path = Path(cfg.checkpoint_dir) / "pipeline.yaml"
        config = OmegaConf.load(config_path)
        config.rendering_engine = "pytorch3d"
        config.compile_model = cfg.compile
        config.workspace_dir = str(Path(cfg.checkpoint_dir).parent)
        check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)

        self.inference = Inference(str(config_path), compile=cfg.compile)

        self.device = torch.device(
            cfg.device if cfg.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"sam3dobjects policy initialized on {self.device}")

        self.reset()

        if cfg.warmup_path:
            import cv2

            warmup_img = cv2.imread(str(cfg.warmup_path))
            if warmup_img is not None:
                warmup_img = warmup_img[:, :, ::-1]
                h, w = warmup_img.shape[:2]
                mask = np.ones((h, w), dtype=np.uint8) * 255
                self.step({"image": warmup_img, "mask": mask})
                self.reset()
                print("warmup complete")

    def reset(self, *args, **kwargs) -> None:
        pass

    def step(self, obs: dict) -> dict:
        if obs is None:
            return {}

        image = obs.get("image")
        mask = obs.get("mask")

        if image is None:
            return {"error": "image is required"}

        if isinstance(image, list):
            image = np.array(image)
        if isinstance(mask, list):
            mask = np.array(mask)

        if mask is None:
            h, w = image.shape[:2]
            mask = np.ones((h, w), dtype=np.uint8) * 255

        image = self.inference.merge_mask_to_rgba(image, mask)

        with torch.no_grad():
            result = self.inference(image, None, seed=self.cfg.seed)

        output = {}
        if "gs" in result:
            gs = result["gs"]
            if self.cfg.save_ply_path:
                gs.save_ply(self.cfg.save_ply_path)
                output["ply_saved_to"] = self.cfg.save_ply_path
            output["has_gaussian_splatting"] = True

        if "mesh" in result:
            output["has_mesh"] = True

        output["success"] = True
        return output


@dataclass
class PolicyConfig:
    checkpoint_dir: str = "/home/nhogg/sam-3d-objects/checkpoints/hf"
    compile: bool = False
    seed: int = 42
    save_ply_path: str = ""
    warmup_path: str = ""
    device: int | None = None


@dataclass
class Config:
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    port: int = 8003
    host: str = "0.0.0.0"
