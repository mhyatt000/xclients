from __future__ import annotations

import cv2
from einops import rearrange
from jaxtyping import Float, Int
import jaxtyping as jt # noqa
# TODO rename Float to jt.Float to avoid confusion
import numpy as np
from numpy import ndarray


def batch_triangulate(
    keypoints_2d: Float[ndarray, "nViews nJoints 3"],
    projection_matrices: Float[ndarray, "nViews 3 4"],
    min_views: int = 2,
) -> Float[ndarray, "nJoints 4"]:
    """Triangulate joints from multi-view correspondences with linear least squares.

    Args:
        keypoints_2d: Homogeneous 2D joints ``[n_views, n_joints, 3]``.
        projection_matrices: Camera projection matrices ``[n_views, 3, 4]``.
        min_views: Minimum number of valid views required to triangulate a joint.

    Returns:
        Homogeneous 3D joints ``[n_joints, 4]`` with aggregated confidence.
    """
    num_joints: int = keypoints_2d.shape[1]

    # Count views where each joint is visible
    visibility_count: Int[ndarray, nJoints] = (keypoints_2d[:, :, -1] > 0).sum(axis=0)
    valid_joints = np.where(visibility_count >= min_views)[0]

    # Filter keypoints by valid joints
    filtered_keypoints: Float[ndarray, "nViews nJoints 3"] = keypoints_2d[:, valid_joints]
    conf3d = filtered_keypoints[:, :, -1].sum(axis=0) / visibility_count[valid_joints]

    P0: Float[ndarray, "1 nViews 4"] = projection_matrices[None, :, 0, :]
    P1: Float[ndarray, "1 nViews 4"] = projection_matrices[None, :, 1, :]
    P2: Float[ndarray, "1 nViews 4"] = projection_matrices[None, :, 2, :]

    # x-coords homogenous
    u: Float[ndarray, "nJoints nViews 1"] = rearrange(filtered_keypoints[..., 0], "c j -> j c 1")
    uP2: Float[ndarray, "nJoints nViews 4"] = u * P2

    # y-coords homogenous
    v: Float[ndarray, "nJoints nViews 1"] = rearrange(filtered_keypoints[..., 1], "c j -> j c 1")
    vP2: Float[ndarray, "nJoints nViews 4"] = v * P2

    confidences: Float[ndarray, "nJoints nViews 1"] = rearrange(filtered_keypoints[..., 2], "c j -> j c 1")

    Au: Float[ndarray, "nJoints nViews 4"] = confidences * (uP2 - P0)
    Av: Float[ndarray, "nJoints nViews 4"] = confidences * (vP2 - P1)
    A: Float[ndarray, "nJoints _ 4"] = np.hstack([Au, Av])

    # Solve using SVD
    _, _, Vh = np.linalg.svd(A)
    triangulated_points = Vh[:, -1, :]
    triangulated_points /= triangulated_points[:, 3, None]

    # Construct result
    result: Float[ndarray, "nJoints 4"] = np.zeros((num_joints, 4))
    # convert from homogenous to euclidean and add confidence
    result[valid_joints, :3] = triangulated_points[:, :3]
    result[valid_joints, 3] = conf3d

    return result


def lift_hand_pnp(
    kp2d: Float[ndarray, "nJoints 2"],
    kp3d_rel: Float[ndarray, "nJoints 3"],
    K: Float[ndarray, "3 3"],
    flags: int = cv2.SOLVEPNP_SQPNP,
) -> tuple[Float[ndarray, "nJoints 3"], Float[ndarray, "3 3"], Float[ndarray, "3"]]:
    """Place a known hand shape into one camera's frame via PnP (single view).

    The single-view replacement for multi-view triangulation when the camera's
    world position is unknown. WiLoR gives the hand's 3D *shape* (``kp3d_rel``,
    wrist-relative metres) and where its joints landed in the image (``kp2d``,
    pixels). PnP solves for the rigid transform (R, t) that places that shape into
    the camera frame so it reprojects onto ``kp2d`` under intrinsics ``K``.

    Args:
        kp2d: detected 2D joints in pixels ``[nJoints, 2]``.
        kp3d_rel: hand-relative 3D joints in metres ``[nJoints, 3]`` (MANO/WiLoR).
        K: camera intrinsics ``[3, 3]`` (assumed fx=fy=515, principal point centred).
        flags: ``cv2.solvePnP`` algorithm flag.

    Returns:
        kp3d_cam: joints in the camera frame ``[nJoints, 3]`` (metric).
        R: rotation, hand -> camera ``[3, 3]``.
        t: translation, hand -> camera ``[3]``.
    """
    # force contiguous because OpenCV is picky about that - we dont want error
    obj = np.ascontiguousarray(kp3d_rel, dtype=np.float64)
    img = np.ascontiguousarray(kp2d, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=flags)
    if not ok:
        raise RuntimeError("cv2.solvePnP failed")

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    kp3d_cam = (R @ obj.T).T + t # transform kp3d_rel to kp3d_cam
    return kp3d_cam, R, t


def project_points(
    kp3d_cam: Float[ndarray, "nJoints 3"],
    K: Float[ndarray, "3 3"],
) -> Float[ndarray, "nJoints 2"]:
    """Project camera-frame 3D points to pixels (the forward camera direction)."""
    K = np.asarray(K, dtype=np.float64)
    proj = (K @ np.asarray(kp3d_cam, dtype=np.float64).T).T
    return proj[:, :2] / proj[:, 2:3]


def reprojection_error(
    kp2d: Float[ndarray, "nJoints 2"],
    kp3d_cam: Float[ndarray, "nJoints 3"],
    K: Float[ndarray, "3 3"],
) -> float:
    """Mean pixel distance between detected 2D and reprojected 3D. Low == correct."""
    pred = project_points(kp3d_cam, K)
    return float(np.linalg.norm(pred - np.asarray(kp2d, dtype=np.float64), axis=1).mean())
