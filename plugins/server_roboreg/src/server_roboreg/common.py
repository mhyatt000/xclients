from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class REGISTRATION_MODE(Enum):
    DISTANCE_FUNCTION = "distance-function"
    SEGMENTATION = "segmentation"


@dataclass
class URDFConfig:
    ros_package: str = "xarm_description"
    xacro_path: str = "urdf/xarm_device.urdf.xacro"
    root_link_name: str = "link_base"
    end_link_name: str = "link7"
    collision_meshes: bool = False


@dataclass
class DRConfig:
    # Optimization settings
    optimizer: str = "Adam"
    lr: float = 3e-3
    max_iterations: int = int(1e3)
    step_size: int = 100
    gamma: float = 1.0

    mode: REGISTRATION_MODE = REGISTRATION_MODE.DISTANCE_FUNCTION
    display_progress: bool = False

    ros_package: str = "xarm_description"
    xacro_path: str = "urdf/xarm_device.urdf.xacro"
    root_link_name: str = "link_base"
    end_link_name: str = "link7"
    collision_meshes: bool = False

    camera_info_file: str = ""
    extrinsics_file: str = ""
    max_jobs: int = 1


@dataclass
class HydraConfig:
    host: str = "0.0.0.0"
    port: int = 8021

    ros_package: str = "xarm_description"
    xacro_path: str = "urdf/xarm_device.urdf.xacro"
    urdf: Path = Path()
    root_link_name: str = "link_base"
    end_link_name: str = "link7"
    collision_meshes: bool = False

    depth_conversion_factor: float = 1.0
    z_min: float = 0.01
    z_max: float = 2.0
    number_of_points: int = 5000
    max_distance: float = 0.1
    outer_max_iter: int = 50
    inner_max_iter: int = 10
    no_boundary: bool = False
    dilation_kernel_size: int = 3
    erosion_kernel_size: int = 10

    # output_file: str = "HT_hydra_robust.npy"
    dr: DRConfig = field(default_factory=lambda: DRConfig())
