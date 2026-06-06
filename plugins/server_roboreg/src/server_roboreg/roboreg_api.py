from __future__ import annotations

from pathlib import Path

from roboreg.core import NVDiffRastRenderer, Robot, RobotScene, TorchKinematics, TorchMeshContainer
from roboreg.io import load_robot_data_from_urdf_file, RobotData


def load_robot_data(
    *,
    urdf: Path | str,
    root_link_name: str,
    end_link_name: str,
    collision: bool,
) -> RobotData:
    urdf_path = Path(urdf)
    if not urdf_path.is_absolute():
        repo_urdf_path = Path(__file__).resolve().parents[2] / urdf_path
        if repo_urdf_path.exists():
            urdf_path = repo_urdf_path

    return load_robot_data_from_urdf_file(
        urdf_path=urdf_path,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        collision=collision,
    )


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
