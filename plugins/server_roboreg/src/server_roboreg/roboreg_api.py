from __future__ import annotations

from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
from roboreg.core import NVDiffRastRenderer, Robot, RobotScene, TorchKinematics, TorchMeshContainer
from roboreg.io import load_robot_data_from_urdf_file, RobotData
from roboreg.io.meshes import apply_mesh_origins, load_meshes, Mesh
from roboreg.io.parsers import URDFParser
import torch

ALL_MESHES_END_LINK = "__all__"
DEFAULT_ARTICULATED_END_LINK = "link7"


def load_robot_data(
    *,
    urdf: Path | str,
    root_link_name: str,
    end_link_name: str,
    collision: bool,
) -> RobotData:
    urdf_path = _resolve_urdf_path(urdf)

    if end_link_name == ALL_MESHES_END_LINK:
        robot_data = load_robot_data_from_urdf_file(
            urdf_path=urdf_path,
            root_link_name=root_link_name,
            end_link_name=DEFAULT_ARTICULATED_END_LINK,
            collision=collision,
        )
        _merge_extra_meshes_into_link(
            robot_data=robot_data,
            urdf_path=urdf_path,
            target_link_name=DEFAULT_ARTICULATED_END_LINK,
            collision=collision,
        )
        return robot_data

    return load_robot_data_from_urdf_file(
        urdf_path=urdf_path,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        collision=collision,
    )


def _resolve_urdf_path(urdf: Path | str) -> Path:
    urdf_path = Path(urdf)
    if not urdf_path.is_absolute():
        repo_urdf_path = Path(__file__).resolve().parents[2] / urdf_path
        if repo_urdf_path.exists():
            urdf_path = repo_urdf_path
    return urdf_path


def _merge_extra_meshes_into_link(
    *,
    robot_data: RobotData,
    urdf_path: Path,
    target_link_name: str,
    collision: bool,
) -> None:
    urdf_parser = URDFParser.from_urdf_file(path=urdf_path)
    extra_meshes = _load_meshes_by_link(
        urdf_parser=urdf_parser,
        urdf_path=urdf_path,
        collision=collision,
    )
    if target_link_name not in robot_data.meshes:
        raise ValueError(f"Target link '{target_link_name}' has no mesh to merge into.")

    transforms = _default_link_transforms(robot_data.urdf)
    if target_link_name not in transforms:
        raise ValueError(f"Target link '{target_link_name}' is not in the URDF kinematic tree.")

    target_from_world = np.linalg.inv(transforms[target_link_name])
    for link_name, mesh in extra_meshes.items():
        if link_name in robot_data.meshes:
            continue
        if link_name not in transforms:
            continue
        target_from_link = target_from_world @ transforms[link_name]
        mesh = _transform_mesh(mesh, target_from_link)
        robot_data.meshes[target_link_name] = _merge_meshes(robot_data.meshes[target_link_name], mesh)
    # Alternative: keep these as separate links and use tree FK with padded/default gripper joints.


def _load_meshes_by_link(
    *,
    urdf_parser: URDFParser,
    urdf_path: Path,
    collision: bool,
) -> dict[str, Mesh]:
    mesh_uris = {}
    mesh_origins = {}
    for link in urdf_parser._robot.links:
        link_name = link.name
        mesh_origin = link.collision if collision else link.visual
        if mesh_origin is None:
            continue
        mesh_uris[link_name] = mesh_origin.geometry.filename
        mesh_origins[link_name] = _origin_matrix(mesh_origin.origin)

    mesh_paths = URDFParser.resolve_relative_uris(mesh_uris, base_path=urdf_path.parent)
    return apply_mesh_origins(load_meshes(mesh_paths), mesh_origins)


def _origin_matrix(origin) -> np.ndarray:
    import transformations

    if origin is None:
        return np.eye(4)
    matrix = transformations.euler_matrix(origin.rpy[0], origin.rpy[1], origin.rpy[2], "sxyz")
    matrix[:3, 3] = origin.xyz
    return matrix


def _default_link_transforms(urdf: str) -> dict[str, np.ndarray]:
    chain = pk.build_chain_from_urdf(urdf)
    q = torch.zeros(1, chain.n_joints)
    transforms = chain.forward_kinematics(q)
    return {name: transform.get_matrix()[0].detach().cpu().numpy() for name, transform in transforms.items()}


def _transform_mesh(mesh: Mesh, transform: np.ndarray) -> Mesh:
    vertices = np.concatenate([mesh.vertices, np.ones((mesh.vertices.shape[0], 1))], axis=1)
    vertices = vertices @ transform.T
    return Mesh(vertices=vertices[:, :3], faces=mesh.faces)


def _merge_meshes(base: Mesh, extra: Mesh) -> Mesh:
    faces = np.concatenate([base.faces, extra.faces + base.vertices.shape[0]], axis=0)
    vertices = np.concatenate([base.vertices, extra.vertices], axis=0)
    return Mesh(vertices=vertices, faces=faces)


def create_robot(
    *,
    robot_data: RobotData,
    batch_size: int,
    device: str,
) -> Robot:
    mesh_container = TorchMeshContainer(
        meshes=robot_data.meshes,
        batch_size=batch_size,
        device=device,
    )
    kinematics = TorchKinematics(
        urdf=robot_data.urdf,
        root_link_name=robot_data.root_link_name,
        end_link_name=robot_data.end_link_name,
        device=device,
    )
    return Robot(mesh_container=mesh_container, kinematics=kinematics)


def create_robot_scene(
    *,
    robot_data: RobotData,
    batch_size: int,
    cameras: dict,
    device: str,
) -> RobotScene:
    return RobotScene(
        cameras=cameras,
        robot=create_robot(robot_data=robot_data, batch_size=batch_size, device=device),
        renderer=NVDiffRastRenderer(device=device),
    )
