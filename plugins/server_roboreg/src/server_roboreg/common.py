from dataclasses import dataclass
from pathlib import Path


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
