from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Pose:
    position: NDArray[np.float64]  # (3,)
    wxyz: NDArray[np.float64]  # (4,) scalar-first


@dataclass
class Targets:
    """Retargeter output: target pose per robot link, expressed in the robot base frame."""

    poses: dict[str, Pose]
    aperture: float | None = None


@dataclass
class KeypointTargets:
    """Retargeter output for keypoint-space solves: full MANO cloud in the robot base frame."""

    kp3d: NDArray[np.float64]  # (21, 3)
    aperture: float | None = None


@dataclass(frozen=True)
class CostWeights:
    """Solver cost weights — the knobs to tune. Frozen/hashable so each instance maps to one jit cache entry."""

    pose_match_position: float = 50.0
    pose_match_orientation: float = 20.0
    joint_pose_coupling: float = 100.0  # ties the joint trajectory to the latent SE3 trajectory
    pose_smoothness: float = 1.0
    start_anchor: float = 100.0
    traj_smoothness: float = 10.0
    velocity_limit: float = 1.0
    joint_limit: float = 100.0
    rest: float = 0.01
    home: float = 0.1  # small MSE bias of the arm joints toward HOME_DXARM
    manipulability: float = 0.01
    self_collision: float = 10.0
    self_collision_margin: float = 0.02
    world_collision: float = 20.0
    world_collision_margin: float = 0.1
    max_iterations: int = 20


def validate_kp3d(kp3d: NDArray[np.floating]) -> NDArray[np.float64]:
    kp3d = np.asarray(kp3d, dtype=np.float64)
    if kp3d.shape == (1, 21, 3):
        kp3d = kp3d[0]
    if kp3d.shape != (21, 3):
        raise ValueError(f"Expected MANO keypoints with shape (21, 3), got {kp3d.shape}")
    if not np.isfinite(kp3d).all():
        raise ValueError("MANO keypoints contain NaN or inf values")
    return kp3d
