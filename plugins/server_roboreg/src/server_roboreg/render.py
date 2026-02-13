"""Renderer utilities for hydra-based robot rendering."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
from roboreg.differentiable import VirtualCamera
from roboreg.util import overlay_mask
from roboreg.util.factories import create_robot_scene
import torch
from tqdm import tqdm

from server_roboreg.common import HydraConfig


@dataclass
class RendererConfig:
    color: str = "b"
    max_jobs: int = 2
    batch_size: int = 1


class Renderer:
    """
    Render robot overlays using an in-memory payload.

    Parameters mirror the legacy CLI in ``old-cli/rr_render.py`` but are organized in an
    object-oriented interface. Images and joint states are supplied directly via
    :meth:`step`, removing any need to read or write from disk.
    """

    def __init__(
        self,
        cfg: HydraConfig,
        rcfg: RendererConfig,
        height: int,
        width: int,
        intr: np.ndarray,
        extr: np.ndarray | None = None,
    ) -> None:
        self.cfg, self.rcfg = cfg, rcfg

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["MAX_JOBS"] = str(self.rcfg.max_jobs)

        camera = {
            "camera": VirtualCamera(
                resolution=[height, width],
                intrinsics=intr,
                device=self.device,
                **({"extrinsics": extr} if extr is not None else {}),
            )
        }

        self.scene = create_robot_scene(
            batch_size=rcfg.batch_size,
            ros_package=cfg.ros_package,
            xacro_path=cfg.xacro_path,
            root_link_name=cfg.root_link_name,
            end_link_name=cfg.end_link_name,
            cameras=camera,
            device=self.device,
            collision=cfg.collision_meshes,
        )

        self.camera = camera
        self.color = rcfg.color
        self.camera_name = next(iter(self.scene.cameras.keys()))
        self.name = self.camera_name

    @staticmethod
    def _ensure_batch(data: np.ndarray | torch.Tensor) -> np.ndarray:
        """Ensure the input has a batch dimension and return a numpy array."""

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if data.ndim == 3:
            data = np.expand_dims(data, axis=0)
        return data

    def observe(self) -> torch.Tensor:
        """Render the current scene from the configured camera."""
        renders = self.scene.observe_from(self.camera_name)
        return renders

    def step(self, payload: dict) -> list[np.ndarray]:
        """
        Render overlays for the provided payload.

        ``payload`` must include ``"images"`` (numpy array or tensor with shape
        ``(B, H, W, C)`` or ``(H, W, C)``) and ``"joints"`` (array or tensor with
        shape ``(B, J)`` or ``(J,)``). The method returns a list of images with the render
        mask overlay applied. No files are read from or written to disk.
        """

        images = payload.get("images")
        if images is None:
            images = payload.get("depth")
            images = np.stack([images] * 3, axis=-1)  # convert depth to 3-channel for overlay
            # normalize
            images = (images - images.min()) / (images.max() - images.min()) * 255.0
            images = images.astype(np.uint8)
        joints = payload["joints"]

        print(payload.keys())

        images_np = self._ensure_batch(np.asarray(images))
        joints = torch.as_tensor(joints, dtype=torch.float32, device=self.device)
        if joints.ndim == 1:
            joints = joints.unsqueeze(0)

        print(joints.shape, images_np.shape)

        overlays: list[np.ndarray] = []
        for j, im in tqdm(zip(joints, images_np, strict=False)):
            print(j.shape, im.shape)
            self.scene.robot.configure(j.reshape(1, -1))
            renders = self.scene.observe_from(self.camera_name)
            render_masks = (renders * 255.0).squeeze(-1).cpu().numpy().astype(np.uint8)
            print(render_masks.shape)

            # for im, render in zip(images_np, render_masks, strict=False):
            overlays.append(overlay_mask(im, render_masks[0], self.color, scale=1.0))

        return {"overlays": overlays}
