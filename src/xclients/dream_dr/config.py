from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass
class Config:
    data_dir: Path = Path("~/rr_good")  # Directory containing record_data.py npz files
    data_select: list[int] = field(default_factory=lambda: [-1])  # select records idx or -1 for all
    output_dir: Path | None = None  # Run output directory. Defaults under data_dir.
    image_size: int = 200  # Square size for SAM, Dream, and DR
    max_records: int | None = None  # Optional cap for quick tests
    extrinsics_path: Path | None = None  # Optional file/dir with w2c, HT, or extrinsics
    record_w2c_index: int | None = None  # Optional record index for static w2c; defaults to scored best
    intrinsics_path: Path | None = None  # Optional file/dir with K or intrinsics

    sam_host: str = "127.0.0.1"  # SAM3 server host
    sam_port: int = 8080  # SAM3 server port
    sam_prompt: str = "robot arm"  # SAM3 text prompt
    sam_confidence: float = 0.5  # SAM3 confidence threshold
    sam_raw_webpolicy: bool = True  # Send raw payloads for older SAM3 webpolicy servers

    dream_host: str = "127.0.0.1"  # Crossformer DREAM server host
    dream_port: int = 8002  # Crossformer DREAM server port
    dream_joint_units: Literal["deg", "rad"] = "deg"  # DREAM server converts deg to rad internally
    call_dream: bool = False  # Call Dream for initial w2c instead of using extrinsics_path

    run_dr: bool = True  # Run server_roboreg DR after cached masks/Dream poses are ready
    inspect: bool = False  # Only print record/cache state; do not call servers or DR
    refresh_cache: bool = False  # Recompute SAM/Dream outputs even when cache files exist

    dr_optimizer: str = "Adam"  # torch.optim optimizer name
    dr_lr: float = 3e-3  # DR optimizer learning rate
    dr_max_iterations: int = 1000  # DR optimization iterations
    dr_step_size: int = 100  # LR scheduler step size
    dr_gamma: float = 0.8  # LR scheduler gamma
    dr_mode: Literal["distance-function", "segmentation"] = "segmentation"  # DR loss target

    ros_package: str = "xarm_description"  # Robot description package for roboreg
    xacro_path: str = "urdf/xarm_device.urdf.xacro"  # Xacro path relative to ros_package
    urdf_path: Path | None = None  # Direct URDF path; defaults to server_roboreg bundled xArm URDF
    root_link_name: str = "link_base"  # Robot root link
    end_link_name: str = "link7"  # Robot end link
    collision_meshes: bool = False  # Use collision meshes instead of visual meshes

    def __post_init__(self) -> None:
        self.data_dir = self.data_dir.expanduser().resolve()
        if self.extrinsics_path is not None:
            self.extrinsics_path = self.extrinsics_path.expanduser().resolve()
        if self.intrinsics_path is not None:
            self.intrinsics_path = self.intrinsics_path.expanduser().resolve()
        if self.urdf_path is not None:
            self.urdf_path = self.urdf_path.expanduser().resolve()
        if self.output_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = self.data_dir / f"dream_dr_{stamp}"
        self.output_dir = self.output_dir.expanduser().resolve()


@dataclass
class Record:
    stem: str
    path: Path
    image: np.ndarray
    joints: np.ndarray
    intrinsics: np.ndarray
    w2c: np.ndarray | None
