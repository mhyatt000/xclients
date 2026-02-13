# Adapted from sam3d_objects/notebook/inference.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified for server_sam3do to remove external dependencies and conda requirements
from __future__ import annotations

import os

# Skip LIDRA initialization BEFORE importing sam3d_objects
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import builtins
from typing import Callable

from hydra.utils import get_method, instantiate
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

# Lazy import: InferencePipelinePointMap imported in __init__ to avoid building pytorch3d at import time

__all__ = ["BLACKLIST_FILTERS", "WHITELIST_FILTERS", "Inference", "check_hydra_safety"]

WHITELIST_FILTERS = [
    lambda target: target.split(".", 1)[0] in {"sam3d_objects", "torch", "torchvision", "moge"},
]

_DANGEROUS_CALLABLES = {
    builtins.exec,
    builtins.eval,
    builtins.__import__,
    os.kill,
    os.system,
    os.putenv,
    os.remove,
    os.removedirs,
    os.rmdir,
    os.fchdir,
    os.setuid,
    os.fork,
    os.forkpty,
    os.killpg,
    os.rename,
    os.renames,
    os.truncate,
    os.replace,
    os.unlink,
    os.fchmod,
    os.fchown,
    os.chmod,
    os.chown,
    os.chroot,
    os.lchown,
    os.getcwd,
    os.chdir,
}


def _is_blacklisted(target: str) -> bool:
    try:
        return get_method(target) in _DANGEROUS_CALLABLES
    except (ImportError, Exception):
        # If we can't import the target to check, it passes the blacklist check.
        # The whitelist filter already verified the module prefix is allowed.
        return False


BLACKLIST_FILTERS = [_is_blacklisted]


class Inference:
    """Public facing inference API for SAM3D Objects.

    Only exposes the core inference interface needed for 3D object reconstruction.
    """

    def __init__(self, config_file: str, compile: bool = False):
        """Initialize the inference pipeline.

        Args:
            config_file: Path to the pipeline.yaml configuration file
            compile: Whether to compile the model for performance
        """
        # Lazy import to avoid requiring pytorch3d at module load time
        from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap

        config = OmegaConf.load(config_file)
        config.rendering_engine = "pytorch3d"
        config.compile_model = compile
        config.workspace_dir = os.path.dirname(config_file)
        check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)
        self._pipeline: InferencePipelinePointMap = instantiate(config)

    def merge_mask_to_rgba(self, image, mask):
        """Merge mask into RGBA image format."""
        mask = mask.astype(np.uint8) * 255
        mask = mask[..., None]
        rgba_image = np.concatenate([image[..., :3], mask], axis=-1)
        return rgba_image

    def __call__(
        self,
        image: Image.Image | np.ndarray,
        mask: None | Image.Image | np.ndarray | None,
        seed: int | None = None,
        pointmap=None,
    ) -> dict:
        """Run inference on an image.

        Args:
            image: Input image (PIL Image or numpy array)
            mask: Optional mask for segmentation
            seed: Random seed for reproducibility
            pointmap: Optional pointmap

        Returns:
            Dictionary containing inference results with gaussian splatting and/or mesh
        """
        image = self.merge_mask_to_rgba(image, mask)
        return self._pipeline.run(
            image,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=pointmap,
        )


def check_target(
    target: str,
    whitelist_filters: list[Callable],
    blacklist_filters: list[Callable],
):
    """Validate that a target is safe for Hydra instantiation."""
    if any(filt(target) for filt in whitelist_filters) and not any(filt(target) for filt in blacklist_filters):
        return
    raise RuntimeError(
        f"target '{target}' is not allowed to be hydra instantiated, if this is a mistake, please do modify the whitelist_filters / blacklist_filters"
    )


def check_hydra_safety(
    config: DictConfig,
    whitelist_filters: list[Callable],
    blacklist_filters: list[Callable],
):
    """Recursively check all Hydra instantiation targets for safety."""
    to_check = [config]
    while len(to_check) > 0:
        node = to_check.pop()
        if isinstance(node, DictConfig):
            to_check.extend(list(node.values()))
            if "_target_" in node:
                check_target(node["_target_"], whitelist_filters, blacklist_filters)
        elif isinstance(node, ListConfig):
            to_check.extend(list(node))
