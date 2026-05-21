"""Renderer utilities for hydra-based robot rendering."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from roboreg.differentiable import NVDiffRastRenderer, Robot, RobotScene, VirtualCamera
from roboreg.io import URDFParser
from roboreg.util import overlay_mask
from roboreg.util.factories import create_robot_scene
import torch
from tqdm import tqdm

from server_roboreg.common import HydraConfig
from server_roboreg.logging_utils import (
    log_renderer_init,
    log_renderer_ros_scene,
    log_renderer_sample,
    log_renderer_step,
    log_renderer_urdf,
)


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
        log_renderer_init(
            self.device,
            self.rcfg.batch_size,
            height,
            width,
            cfg.urdf,
        )

        camera = {
            "camera": VirtualCamera(
                resolution=[height, width],
                intrinsics=intr,
                device=self.device,
                **({"extrinsics": extr} if extr is not None else {}),
            )
        }

        if cfg.urdf:
            log_renderer_urdf(cfg.urdf)
            self.scene = create_robot_scene_from_urdf(
                batch_size=rcfg.batch_size,
                urdf_path=cfg.urdf,
                root_link_name=cfg.root_link_name,
                end_link_name=cfg.end_link_name,
                cameras=camera,
                device=self.device,
                collision=cfg.collision_meshes,
            )
        else:
            log_renderer_ros_scene(cfg.ros_package, cfg.xacro_path)
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

    @staticmethod
    def _normalize_depth_images(depth: object) -> np.ndarray:
        depth_arr = np.asarray(depth, dtype=np.float32)
        images = np.stack([depth_arr] * 3, axis=-1)
        finite = images[np.isfinite(images)]
        if finite.size == 0:
            return np.zeros_like(images, dtype=np.uint8)

        lo = float(finite.min())
        hi = float(finite.max())
        images = (images - lo) / (hi - lo) if hi > lo else np.zeros_like(images, dtype=np.float32)
        return np.clip(images * 255.0, 0.0, 255.0).astype(np.uint8)

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
            images = self._normalize_depth_images(images)
        joints = payload["joints"]

        images_np = self._ensure_batch(np.asarray(images))
        joints = torch.as_tensor(joints, dtype=torch.float32, device=self.device)
        if joints.ndim == 1:
            joints = joints.unsqueeze(0)

        log_renderer_step(images_np.shape, tuple(joints.shape))

        overlays: list[np.ndarray] = []
        for index, (j, im) in enumerate(tqdm(zip(joints, images_np, strict=False))):
            self.scene.robot.configure(j.reshape(1, -1))
            renders = self.scene.observe_from(self.camera_name)
            render_masks = (renders * 255.0).squeeze(-1).cpu().numpy().astype(np.uint8)
            log_renderer_sample(
                index,
                im.shape,
                render_masks.shape,
                float(render_masks.mean() / 255.0),
            )

            overlays.append(overlay_mask(im, render_masks[0], self.color, scale=1.0))

        return {"overlays": overlays}


class LocalURDFParser(URDFParser):
    def __init__(self, urdf_dir: Path) -> None:
        super().__init__()
        self.urdf_dir = urdf_dir

    def ros_package_mesh_paths(
        self, root_link_name: str, end_link_name: str, collision: bool = False
    ) -> dict[str, str]:
        paths = {}
        for link_name, raw_path in self.raw_mesh_paths(root_link_name, end_link_name, collision=collision).items():
            if raw_path.startswith("package://"):
                raise ValueError(f"Standalone URDF cannot resolve ROS package mesh path {raw_path!r}")
            path = Path(raw_path)
            paths[link_name] = str(path if path.is_absolute() else self.urdf_dir / path)
        return paths


def create_robot_scene_from_urdf(
    batch_size: int,
    urdf_path: Path,
    root_link_name: str,
    end_link_name: str,
    cameras: dict[str, VirtualCamera],
    device: torch.device | str,
    collision: bool = False,
) -> RobotScene:
    urdf_path = Path(urdf_path).expanduser().resolve()
    parser = LocalURDFParser(urdf_path.parent)
    parser.from_urdf(urdf_path.read_text())

    if root_link_name == "":
        root_link_name = parser.link_names_with_meshes(collision=collision)[0]
    if end_link_name == "":
        end_link_name = parser.link_names_with_meshes(collision=collision)[-1]

    robot = Robot(
        urdf_parser=parser,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        collision=collision,
        batch_size=batch_size,
        device=device,
    )
    return RobotScene(cameras=cameras, robot=robot, renderer=NVDiffRastRenderer(device=device))
