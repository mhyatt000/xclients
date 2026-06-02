"""Renderer utilities for hydra-based robot rendering."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytorch_kinematics as pk
from roboreg.core import NVDiffRastRenderer, Robot, RobotScene, TorchKinematics, TorchMeshContainer, VirtualCamera
from roboreg.io import (
    apply_mesh_origins,
    load_meshes,
    load_robot_data_from_ros_xacro,
    load_robot_data_from_urdf_file,
    simplify_meshes,
    URDFParser,
)
from roboreg.util import overlay_mask
import torch
from tqdm import tqdm
import transformations

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
            self.scene = create_robot_scene_from_urdf_file(
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
            self.scene = create_robot_scene_from_ros_xacro(
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


class LocalURDFParser:
    def __init__(self, parser: URDFParser, urdf_dir: Path) -> None:
        self.parser = parser
        self.urdf_dir = urdf_dir

    @classmethod
    def from_file(cls, urdf_path: Path) -> LocalURDFParser:
        return cls(URDFParser.from_urdf_file(urdf_path), urdf_path.parent)

    def __getattr__(self, name: str):
        return getattr(self.parser, name)

    def all_mesh_paths(self, collision: bool = False) -> dict[str, Path]:
        paths = {}
        for link in self.parser._robot.links:
            mesh = self._link_mesh_path(link, collision)
            if mesh is None:
                continue
            if mesh.startswith("package://"):
                raise ValueError(f"Standalone URDF cannot resolve ROS package mesh path {mesh!r}")
            path = Path(mesh)
            paths[link.name] = path if path.is_absolute() else self.urdf_dir / path
        return paths

    def all_mesh_origins(self, collision: bool = False) -> dict[str, np.ndarray]:
        origins = {}
        for link in self.parser._robot.links:
            origin = self._link_mesh_origin(link, collision)
            if origin is None:
                continue
            ht = transformations.euler_matrix(origin.rpy[0], origin.rpy[1], origin.rpy[2], "sxyz")
            ht[:3, 3] = origin.xyz
            origins[link.name] = ht
        return origins

    @staticmethod
    def _link_mesh_path(link, collision: bool) -> str | None:
        visual = link.collision if collision else link.visual
        if visual is None or visual.geometry is None:
            return None
        return getattr(visual.geometry, "filename", None)

    @staticmethod
    def _link_mesh_origin(link, collision: bool):
        visual = link.collision if collision else link.visual
        if visual is None or visual.geometry is None:
            return None
        if visual.origin is None:
            return SimpleNamespace(xyz=[0.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0])
        return visual.origin


class FullTreeRobot:
    __slots__ = ["_chain", "_configured_vertices", "_device", "_mesh_container"]

    def __init__(
        self,
        urdf_parser: LocalURDFParser,
        collision: bool = False,
        batch_size: int = 1,
        device: torch.device = "cuda",
    ) -> None:
        meshes = load_meshes(paths=urdf_parser.all_mesh_paths(collision=collision))
        meshes = simplify_meshes(meshes=meshes, target_reduction=0.0)
        meshes = apply_mesh_origins(meshes=meshes, origins=urdf_parser.all_mesh_origins(collision=collision))
        self._mesh_container = TorchMeshContainer(meshes=meshes, batch_size=batch_size, device=device)
        self._chain = pk.build_chain_from_urdf(urdf_parser.urdf).to(device=device)
        self._configured_vertices = self.mesh_container.vertices.clone()
        self._device = torch.device(device) if isinstance(device, str) else device

    def configure(self, q: torch.FloatTensor, ht_root: torch.FloatTensor = None) -> None:
        q = self._normalize_q(q)
        if q.shape[0] != self.mesh_container.batch_size:
            raise ValueError(f"Batch size mismatch. Meshes: {self.mesh_container.batch_size}, joint states: {q.shape[0]}.")
        if ht_root is None:
            ht_root = torch.eye(4, dtype=q.dtype, device=q.device).unsqueeze(0)

        fk = self._chain.forward_kinematics(q)
        self._configured_vertices = self.mesh_container.vertices.clone()
        for link_name in self.mesh_container.names:
            if link_name not in fk:
                continue
            ht = fk[link_name].get_matrix()
            self._configured_vertices[
                :,
                self.mesh_container.lower_vertex_index_lookup[link_name] : self.mesh_container.upper_vertex_index_lookup[link_name],
            ] = torch.matmul(
                torch.matmul(
                    self._configured_vertices[
                        :,
                        self.mesh_container.lower_vertex_index_lookup[link_name] : self.mesh_container.upper_vertex_index_lookup[link_name],
                    ],
                    ht.transpose(-1, -2),
                ),
                ht_root.transpose(-1, -2),
            )

    def _normalize_q(self, q: torch.FloatTensor) -> torch.FloatTensor:
        expected = len(self._chain.get_joint_parameter_names())
        if q.shape[-1] == expected:
            return q
        if expected == 13 and q.shape[-1] in (7, 8):
            full = torch.zeros((q.shape[0], expected), dtype=q.dtype, device=q.device)
            full[:, :7] = q[:, :7]
            if q.shape[-1] == 8:
                full[:, 7:] = q[:, 7:8]
            return full
        raise ValueError(f"Expected joint states of shape {expected}, got {q.shape[-1]}.")

    @property
    def configured_vertices(self) -> torch.FloatTensor:
        return self._configured_vertices

    @property
    def mesh_container(self) -> TorchMeshContainer:
        return self._mesh_container

    @property
    def device(self) -> torch.device:
        return self._device


def robot_from_data(robot_data, batch_size: int, device: torch.device | str) -> Robot:
    mesh_container = TorchMeshContainer(meshes=robot_data.meshes, batch_size=batch_size, device=device)
    kinematics = TorchKinematics(
        urdf=robot_data.urdf,
        root_link_name=robot_data.root_link_name,
        end_link_name=robot_data.end_link_name,
        device=device,
    )
    return Robot(mesh_container=mesh_container, kinematics=kinematics)


def create_robot_scene_from_urdf_file(
    batch_size: int,
    urdf_path: Path,
    root_link_name: str,
    end_link_name: str,
    cameras: dict[str, VirtualCamera],
    device: torch.device | str,
    collision: bool = False,
) -> RobotScene:
    urdf_path = Path(urdf_path).expanduser().resolve()

    if end_link_name in {"__all__", "all", "*"}:
        parser = LocalURDFParser.from_file(urdf_path)
        robot = FullTreeRobot(
            urdf_parser=parser,
            collision=collision,
            batch_size=batch_size,
            device=device,
        )
    else:
        robot_data = load_robot_data_from_urdf_file(
            urdf_path=urdf_path,
            root_link_name=root_link_name,
            end_link_name=end_link_name,
            collision=collision,
        )
        robot = robot_from_data(robot_data, batch_size=batch_size, device=device)
    return RobotScene(cameras=cameras, robot=robot, renderer=NVDiffRastRenderer(device=device))


def create_robot_scene_from_ros_xacro(
    batch_size: int,
    ros_package: str,
    xacro_path: str | Path,
    root_link_name: str,
    end_link_name: str,
    cameras: dict[str, VirtualCamera],
    device: torch.device | str,
    collision: bool = False,
) -> RobotScene:
    robot_data = load_robot_data_from_ros_xacro(
        ros_package=ros_package,
        xacro_path=xacro_path,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        collision=collision,
    )
    robot = robot_from_data(robot_data, batch_size=batch_size, device=device)
    return RobotScene(cameras=cameras, robot=robot, renderer=NVDiffRastRenderer(device=device))
