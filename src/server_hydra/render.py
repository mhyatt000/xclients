"""Renderer utilities for hydra-based robot rendering."""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
from roboreg.util import overlay_mask
from roboreg.util.factories import create_robot_scene, create_virtual_camera


class Renderer:
    """
    Render robot overlays using an in-memory payload.

    Parameters mirror the legacy CLI in ``old-cli/rr_render.py`` but are organized in an
    object-oriented interface. Images and joint states are supplied directly via
    :meth:`step`, removing any need to read or write from disk.
    """

    def __init__(
        self,
        *,
        camera_info_file: str,
        extrinsics_file: str,
        ros_package: str = "lbr_description",
        xacro_path: str = "urdf/med7/med7.xacro",
        root_link_name: str = "",
        end_link_name: str = "",
        collision_meshes: bool = False,
        color: str = "b",
        max_jobs: int = 2,
        batch_size: int = 1,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        os.environ["MAX_JOBS"] = str(max_jobs)

        camera = {
            "camera": create_virtual_camera(
                camera_info_file=camera_info_file,
                extrinsics_file=extrinsics_file,
                device=self.device,
            )
        }
        self.scene = create_robot_scene(
            batch_size=batch_size,
            ros_package=ros_package,
            xacro_path=xacro_path,
            root_link_name=root_link_name,
            end_link_name=end_link_name,
            cameras=camera,
            device=self.device,
            collision=collision_meshes,
        )
        self.color = color
        self.camera_name = next(iter(self.scene.cameras.keys()))

    @staticmethod
    def _ensure_batch(data: np.ndarray | torch.Tensor) -> np.ndarray:
        """Ensure the input has a batch dimension and return a numpy array."""

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if data.ndim == 3:
            data = np.expand_dims(data, axis=0)
        return data

    def step(self, payload: dict) -> List[np.ndarray]:
        """
        Render overlays for the provided payload.

        ``payload`` must include ``"images"`` (numpy array or tensor with shape
        ``(B, H, W, C)`` or ``(H, W, C)``) and ``"joint_states"`` (array or tensor with
        shape ``(B, J)`` or ``(J,)``). The method returns a list of images with the render
        mask overlay applied. No files are read from or written to disk.
        """

        images = payload["images"]
        joint_states = payload["joint_states"]

        images_np = self._ensure_batch(np.asarray(images))
        joint_states_tensor = torch.as_tensor(joint_states, dtype=torch.float32, device=self.device)
        if joint_states_tensor.ndim == 1:
            joint_states_tensor = joint_states_tensor.unsqueeze(0)

        self.scene.robot.configure(joint_states_tensor)
        renders = self.scene.observe_from(self.camera_name)
        render_masks = (renders * 255.0).squeeze(-1).cpu().numpy().astype(np.uint8)

        overlays: List[np.ndarray] = []
        for image, render in zip(images_np, render_masks, strict=False):
            overlays.append(overlay_mask(image, render, self.color, scale=1.0))
        return overlays
