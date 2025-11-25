from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Prediction
from PIL import Image
from pydantic import BaseModel, ConfigDict, TypeAdapter
from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8080

    # Model loading
    model_source: str = "huggingface"  # "huggingface", "repo", or explicit path/model id
    hf_model_id: str = "depth-anything/da3nested-giant-large"
    device: str | None = (
        None  # Device to run the model on (e.g., "cuda", "cpu"). If None, auto-select.
    )


class DA3Payload(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: list[
        np.ndarray | Image.Image | str
    ]  # List of input images (numpy arrays, PIL Images, or file paths)

    extrinsics: np.ndarray | None = None  # Camera extrinsics (N, 4, 4)
    intrinsics: np.ndarray | None = None  # Camera intrinsics (N, 3, 3)

    align_to_input_ext_scale: bool = True  # whether to align the input pose scale to the prediction
    infer_gs: bool = False  # Enable the 3D Gaussian branch (needed for `gs_ply`/`gs_video` exports)

    render_exts: np.ndarray | None = None  # Optional render extrinsics for Gaussian video export
    render_ixts: np.ndarray | None = None  # Optional render intrinsics for Gaussian video export
    render_hw: tuple[int, int] | None = None  # Optional render resolution for Gaussian video export

    process_res: int = 504  # Processing resolution
    process_res_method: str = "upper_bound_resize"  # Resize method for processing

    export_dir: str | Path | None = None  # Directory to export results
    export_format: str = "mini_npz"  # Export format (mini_npz, npz, glb, ply, gs, gs_video)
    export_feat_layers: Sequence[int] | None = (
        None  # Layer indices to export intermediate features from
    )

    # GLB export parameters
    conf_thresh_percentile: float = (
        40.0  # [GLB] Lower percentile for adaptive confidence threshold (default: 40.0)
    )
    num_max_points: int = (
        1_000_000  # [GLB] Maximum number of points in the point cloud (default: 1,000,000)
    )
    show_cameras: bool = True  # [GLB] Show camera wireframes in the exported scene (default: True)

    # Feat_vis export parameters
    feat_vis_fps: int = 15  # [FEAT_VIS] Frame rate for output video (default: 15)

    # Other export parameters
    export_kwargs: dict | None = None  # additional arguments to export functions.


# class DA3Prediction(Prediction,BaseModel):
# model_config = ConfigDict(arbitrary_types_allowed=True)


class DA3Policy(BasePolicy):
    def __init__(self, cfg: Config):
        self.device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self._load_model(cfg)
        self.model = self.model.to(device=self.device)

        self.adapter = TypeAdapter(DA3Payload)
        # self.oadapter = TypeAdapter(Prediction)

    @staticmethod
    def _load_model(cfg: Config) -> DepthAnything3:
        if cfg.model_source == "huggingface":
            return DepthAnything3.from_pretrained(cfg.hf_model_id)

        if cfg.model_source == "repo":
            # Uses the raw repository weights that come with the installed package.
            return DepthAnything3.from_pretrained()

        # Allow callers to pass an explicit path or model id.
        return DepthAnything3.from_pretrained(cfg.model_source)

    def step(self, raw: dict) -> dict:
        payload: DA3Payload = self.adapter.validate_python(raw)
        print(payload.infer_gs)
        prediction: Prediction = self.model.inference(
            image=payload.image,
            infer_gs=payload.infer_gs,
            # self.adapter.dump_python(
            # mode="python",
            # )
        )
        return asdict(prediction)

        # return self.oadapter.dump_python(prediction, mode="python")


def main(cfg: Config):
    policy = DA3Policy(cfg)
    server = Server(policy, cfg.host, cfg.port)
    server.serve()


if __name__ == "__main__":
    main(tyro.cli(Config))
