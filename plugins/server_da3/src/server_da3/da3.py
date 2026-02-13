from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import enum
from pathlib import Path

from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export.glb import (
    _compute_alignment_transform_first_cam_glTF_center_by_points,
    _depths_to_world_points_with_colors,
    _filter_and_downsample,
)
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, TypeAdapter
import torch
import tyro
from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server


class ModelChoice(enum.Enum):
    G = "depth-anything/da3nested-giant-large"
    L = "depth-anything/da3metric-large"


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8080

    # Model loading
    model_source: str = "huggingface"  # "huggingface", "repo", or explicit path/model id
    hf_model_id: ModelChoice = ModelChoice.G  # HuggingFace model ID (if using "huggingface" source)
    device: str | None = None  # Device to run the model on (e.g., "cuda", "cpu"). If None, auto-select.


class DA3Payload(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: list[np.ndarray | Image.Image | str]  # List of input images (numpy arrays, PIL Images, or file paths)

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
    export_feat_layers: Sequence[int] | None = None  # Layer indices to export intermediate features from

    # GLB export parameters
    conf_thresh_percentile: float = 40.0  # [GLB] Lower percentile for adaptive confidence threshold (default: 40.0)
    num_max_points: int = 1_000_000  # [GLB] Maximum number of points in the point cloud (default: 1,000,000)
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
            return DepthAnything3.from_pretrained(cfg.hf_model_id.value)

        if cfg.model_source == "repo":
            # Uses the raw repository weights that come with the installed package.
            return DepthAnything3.from_pretrained()

        # Allow callers to pass an explicit path or model id.
        return DepthAnything3.from_pretrained(cfg.model_source)

    def _step_pcd(self, prediction: Prediction):
        # 3) Confidence threshold (if no conf, then no filtering)

        num_max_points: int = 1_000_000
        conf_thresh: float = 1.05
        filter_black_bg, filter_white_bg = True, True
        filter_black_bg, filter_white_bg = False, False
        conf_thresh_percentile: float = 40.0

        if filter_black_bg:
            prediction.conf[(prediction.processed_images < 16).all(axis=-1)] = 1.0
        if filter_white_bg:
            prediction.conf[(prediction.processed_images >= 240).all(axis=-1)] = 1.0

        # conf_thresh = get_conf_thresh(
        # prediction,
        # getattr(prediction, "sky_mask", None),
        # conf_thresh,
        # conf_thresh_percentile,
        # ensure_thresh_percentile,
        # )
        images_u8 = prediction.processed_images  # (N,H,W,3) uint8
        conf_thresh = np.percentile(prediction.conf, conf_thresh_percentile)

        # 4) Back-project to world coordinates and get colors (world frame)
        points, colors = _depths_to_world_points_with_colors(
            prediction.depth,
            prediction.intrinsics,
            prediction.extrinsics,  # w2c
            images_u8,
            prediction.conf,
            conf_thresh,
        )

        # 5) Based on first camera orientation + glTF axis system, center by point cloud,
        # construct alignment transform, and apply to point cloud
        _compute_alignment_transform_first_cam_glTF_center_by_points(prediction.extrinsics[0], points)  # (4,4)

        # if points.shape[0] > 0:
        # points = trimesh.transform_points(points, A)

        # 6) Clean + downsample
        points, colors = _filter_and_downsample(points, colors, num_max_points)
        return points, colors

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
        points, colors = self._step_pcd(prediction)
        prediction = asdict(prediction)
        prediction["points"] = points
        prediction["colors"] = colors
        return prediction

        # return self.oadapter.dump_python(prediction, mode="python")


def main(cfg: Config):
    policy = DA3Policy(cfg)
    server = Server(policy, cfg.host, cfg.port)
    server.serve()


if __name__ == "__main__":
    main(tyro.cli(Config))
