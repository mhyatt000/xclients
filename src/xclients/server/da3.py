
from __future__ import annotations

from typing import Sequence, Optional
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic.dataclasses import dataclass

import glob, os, torch
import tyro
from depth_anything_3.api import DepthAnything3
from pydantic import TypeAdapter

from webpolicy.base_policy import BasePolicy
from webpolicy.server.server import Server

from dataclasses import dataclass

from depth_anything_3.specs import Prediction

@dataclass
class Config:
    host: str 
    port: int = 8080

@dataclass
class DA3Payload:
    image: list[np.ndarray | Image.Image | str] # List of input images (numpy arrays, PIL Images, or file paths)

    extrinsics: np.ndarray | None = None # Camera extrinsics (N, 4, 4)
    intrinsics: np.ndarray | None = None # Camera intrinsics (N, 3, 3)

    align_to_input_ext_scale: bool = True # whether to align the input pose scale to the prediction
    infer_gs: bool = False # Enable the 3D Gaussian branch (needed for `gs_ply`/`gs_video` exports)

    render_exts: np.ndarray | None = None # Optional render extrinsics for Gaussian video export
    render_ixts: np.ndarray | None = None # Optional render intrinsics for Gaussian video export
    render_hw: tuple[int, int] | None = None # Optional render resolution for Gaussian video export

    process_res: int = 504 # Processing resolution
    process_res_method: str = "upper_bound_resize" # Resize method for processing

    export_dir: str | Path | None = None # Directory to export results
    export_format: str = "mini_npz" # Export format (mini_npz, npz, glb, ply, gs, gs_video)
    export_feat_layers: Sequence[int] | None = None # Layer indices to export intermediate features from

    # GLB export parameters
    conf_thresh_percentile: float = 40.0 # [GLB] Lower percentile for adaptive confidence threshold (default: 40.0)
    num_max_points: int = 1_000_000 # [GLB] Maximum number of points in the point cloud (default: 1,000,000)
    show_cameras: bool = True # [GLB] Show camera wireframes in the exported scene (default: True)

    # Feat_vis export parameters
    feat_vis_fps: int = 15 # [FEAT_VIS] Frame rate for output video (default: 15)

    # Other export parameters
    export_kwargs: Optional[dict] = None # additional arguments to export functions.


class DA3Policy(BasePolicy):
    def __init__(self):
        self.device = torch.device("cuda")
        self.model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
        self.model = self.model.to(device=self.device)

        self.adapter = TypeAdapter(DA3Payload)
        self.oadapter = TypeAdapter(Prediction)

    def infer(self, raw: Dict) -> Dict:
        payload : DA3Payload =  self.adapter.validate_python(raw)
        prediction: Prediction = self.model.inference(self.adapter.dump_python(payload, mode="python"))
        return self.oadapter.dump_python(prediction, mode="python")

def main(cfg:Config):

    policy = DA3Policy()
    server = Server(policy, cfg.host, cfg.port)
    server.serve()

if __name__ == "__main__":
    main(tyro.cli(Config))
