from __future__ import annotations

from typing import Protocol

import jaxlie
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from xclients.target.embodiment import Embodiment
from xclients.target.types import Pose, Targets, validate_kp3d

PALM_KEYPOINT = 0
THUMB_TIP_KEYPOINT = 4
INDEX_TIP_KEYPOINT = 8


class Retargeter(Protocol):
    """Hand keypoints (world frame) -> link targets (robot base frame). Pure and stateless."""

    def __call__(self, *kp3d: NDArray[np.floating]) -> Targets: ...


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


WRIST_KEYPOINT = 0
# MANO/MediaPipe fingertips, ordered to match Embodiment.target_links for the
# ruka rig: thumb, index, middle, ring, pinky (same pairing as dex_retarget.yml).
FINGERTIP_KEYPOINTS = (4, 8, 12, 16, 20)


class RukaRetargeter:
    """MANO fingertips -> per-fingertip position targets on the ruka hand.

    Target orientations are frozen at the rest-FK tip orientations — they only
    exist because the planner matches full SE3 poses. Pair with
    CostWeights(pose_match_orientation=0) so positions alone constrain the solve.
    `scale` shrinks/grows the hand about the wrist to bridge the human-vs-ruka
    size mismatch (1.0 until tuned).
    """

    def __init__(self, emb: Embodiment, scale: float = 1.0) -> None:
        if len(emb.target_links) != len(FINGERTIP_KEYPOINTS):
            raise ValueError(f"Expected {len(FINGERTIP_KEYPOINTS)} fingertip links, got {emb.target_links}")
        self.target_links = list(emb.target_links)
        self.base_T_world = np.linalg.inv(emb.world_T_base)
        self.scale = float(scale)
        wxyz_xyz = np.array(emb.robot.forward_kinematics(cfg=emb.rest_cfg))
        names = emb.robot.links.names
        self.rest_wxyz = {name: wxyz_xyz[names.index(name), :4].copy() for name in self.target_links}

    def __call__(self, *kp3d: NDArray[np.floating]) -> Targets:
        (kp3d_world,) = kp3d
        kp3d_world = validate_kp3d(kp3d_world)
        kp = kp3d_world @ self.base_T_world[:3, :3].T + self.base_T_world[:3, 3]

        wrist = kp[WRIST_KEYPOINT]
        tips = wrist + self.scale * (kp[list(FINGERTIP_KEYPOINTS)] - wrist)
        poses = {
            link: Pose(tips[i].astype(np.float64), self.rest_wxyz[link])
            for i, link in enumerate(self.target_links)
        }
        aperture = float(np.linalg.norm(kp[INDEX_TIP_KEYPOINT] - kp[THUMB_TIP_KEYPOINT]))
        return Targets(poses=poses, aperture=aperture)
