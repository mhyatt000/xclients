from __future__ import annotations

from typing import Protocol

import jaxlie
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from xclients.motion.embodiment import Embodiment
from xclients.motion.types import KeypointTargets, Pose, Targets, validate_kp3d

PALM_KEYPOINT = 0
THUMB_TIP_KEYPOINT = 4
INDEX_TIP_KEYPOINT = 8


class Retargeter(Protocol):
    """Hand keypoints (world frame) -> solver targets (robot base frame). Pure and stateless."""

    def __call__(self, *kp3d: NDArray[np.floating]) -> Targets | KeypointTargets: ...


def normalize(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    return vec / (np.linalg.norm(vec) + 1e-6)


def construct_gripper_axes(
    left_pos: NDArray[np.float64],
    right_pos: NDArray[np.float64],
    midpoint_pos: NDArray[np.float64],
    palm_or_eef_pos: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Lateral and approach-plane axes shared by the hand and the gripper."""
    line_a = normalize(right_pos - left_pos)
    line_b = normalize(midpoint_pos - palm_or_eef_pos)
    line_c = normalize(np.cross(line_a, line_b))
    line_d = normalize(np.cross(line_a, line_c))
    return line_a, line_d


def axes_frame(line_a: NDArray[np.float64], line_d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Orthonormal frame whose first and second columns are A and D."""
    x_axis = normalize(line_a)
    y_axis = normalize(line_d - x_axis * np.dot(line_d, x_axis))
    z_axis = normalize(np.cross(x_axis, y_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1)


class GripperRetargeter:
    """Thumb/index pinch -> single TCP pose + aperture (the 09-style axes-frame mapping)."""

    def __init__(self, emb: Embodiment) -> None:
        (self.target_link,) = emb.target_links
        self.base_T_world = np.linalg.inv(emb.world_T_base)
        self.tcp_frame_from_axes_frame = self._calibrate(emb)

    @staticmethod
    def _calibrate(emb: Embodiment) -> NDArray[np.float64]:
        """How the constructed axes frame maps to the tcp link frame, from FK at rest."""
        names = emb.robot.links.names
        idx = {
            key: names.index(name)
            for key, name in {
                "eef": "xarm_gripper_base_link",
                "left_tip": "left_tip",
                "right_tip": "right_tip",
                "tcp": emb.target_links[0],
            }.items()
        }
        T_base_link = jaxlie.SE3(emb.robot.forward_kinematics(cfg=emb.rest_cfg))
        link_pos = np.array(T_base_link.translation())
        link_wxyz = np.array(T_base_link.rotation().wxyz)

        a_axis, d_axis = construct_gripper_axes(
            link_pos[idx["left_tip"]],
            link_pos[idx["right_tip"]],
            link_pos[idx["tcp"]],
            link_pos[idx["eef"]],
        )
        robot_axes_frame = axes_frame(a_axis, d_axis)
        tcp_frame = R.from_quat(link_wxyz[idx["tcp"]], scalar_first=True).as_matrix()
        return robot_axes_frame.T @ tcp_frame

    def __call__(self, *kp3d: NDArray[np.floating]) -> Targets:
        (kp3d_world,) = kp3d
        kp3d_world = validate_kp3d(kp3d_world)
        kp = kp3d_world @ self.base_T_world[:3, :3].T + self.base_T_world[:3, 3]

        palm = kp[PALM_KEYPOINT]
        left = kp[THUMB_TIP_KEYPOINT]
        right = kp[INDEX_TIP_KEYPOINT]
        tcp_target = (left + right) / 2.0

        a_axis, d_axis = construct_gripper_axes(left, right, tcp_target, palm)
        target_tcp_frame = axes_frame(a_axis, d_axis) @ self.tcp_frame_from_axes_frame
        target_wxyz = R.from_matrix(target_tcp_frame).as_quat(scalar_first=True)
        aperture = float(np.linalg.norm(right - left))
        return Targets(poses={self.target_link: Pose(tcp_target, target_wxyz)}, aperture=aperture)


class RukaRetargeter:
    """MANO keypoints (world frame) -> full keypoint cloud in the robot base frame.

    Pure frame transform: the actual retargeting (21-point local/global
    alignment with a solved human->robot scale) lives in HandPlanner's cost
    structure, so no per-finger scaling or target construction happens here.
    """

    def __init__(self, emb: Embodiment) -> None:
        self.base_T_world = np.linalg.inv(emb.world_T_base)

    def __call__(self, *kp3d: NDArray[np.floating]) -> KeypointTargets:
        (kp3d_world,) = kp3d
        kp3d_world = validate_kp3d(kp3d_world)
        kp = kp3d_world @ self.base_T_world[:3, :3].T + self.base_T_world[:3, 3]
        aperture = float(np.linalg.norm(kp[INDEX_TIP_KEYPOINT] - kp[THUMB_TIP_KEYPOINT]))
        return KeypointTargets(kp3d=kp, aperture=aperture)
