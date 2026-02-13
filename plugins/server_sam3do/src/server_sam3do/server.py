from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf
from rich import print
import torch
from webpolicy.base_policy import BasePolicy

from server_sam3do.inference import (
    BLACKLIST_FILTERS,
    check_hydra_safety,
    Inference,
    WHITELIST_FILTERS,
)


@dataclass
class PolicyConfig:
    ckpt: Path
    compile: bool = True
    seed: int = 42
    save_ply_path: str | None = None
    warmup_path: str | None = None  # TODO change to warmup:bool
    device: int | None = None

    def __post_init__(self):
        self.ckpt = Path(self.ckpt).expanduser().resolve()


@dataclass
class Config:
    port: int = 8003
    host: str = "0.0.0.0"
    policy: PolicyConfig = field(default_factory=PolicyConfig)


class Policy(BasePolicy):
    def __init__(self, cfg):
        self.cfg = cfg

        config_path = Path(cfg.ckpt) / "pipeline.yaml"
        config = OmegaConf.load(config_path)
        config.rendering_engine = "pytorch3d"
        config.compile_model = cfg.compile
        config.workspace_dir = str(Path(cfg.ckpt).parent)
        check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)

        self.inference = Inference(str(config_path), compile=cfg.compile)

        self.device = torch.device(
            cfg.device if cfg.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"sam3dobjects policy initialized on {self.device}")

        self.reset()

        if cfg.warmup_path:  # TODO warmup with np.zeros, not from file
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
        jax.tree.map(lambda x: np.array(x) if isinstance(x, list) else x, obs)
        image = obs["image"]
        mask = obs.get("mask")  # mask is optional

        if mask is None:
            h, w = image.shape[:2]
            mask = np.ones((h, w), dtype=np.uint8) * 255

        # image = self.inference.merge_mask_to_rgba(image, mask)

        with torch.no_grad():
            result = self.inference(image, mask, seed=self.cfg.seed)

        if "gs" in result:
            gs = result["gs"]
            if self.cfg.save_ply_path:
                gs.save_ply(self.cfg.save_ply_path)

        def spec(tree: dict):
            return jax.tree.map(
                lambda x: {"shape": x.shape, "dtype": x.dtype} if isinstance(x, torch.Tensor) else type(x), tree
            )

        print(spec(result))
        print(result["gs"].aabb)

        for k in ["gaussian", "glb", "gs", "mesh", "coords", "coords_original", "shape"]:
            del result[k]
        # tensor to numpy . cpu no gpu. detach
        return jax.tree.map(lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x, result)
