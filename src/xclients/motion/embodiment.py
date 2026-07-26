from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pyroki as pk
from pyroki.collision import RobotCollision
from scipy.spatial.transform import Rotation as R
import yourdfpy

from xclients import urdf_compose as uc

REPO_ROOT = Path(__file__).resolve().parents[3]
XARM_GRIPPER_URDF = REPO_ROOT / "retarget_helpers" / "hand" / "xarm" / "xarm7_standalone.urdf"
RUKA_URDF = {
    "left": REPO_ROOT / "urdf" / "ruka" / "robot-left.urdf",
    "right": REPO_ROOT / "urdf" / "ruka" / "robot.urdf",
}
GENERATED_DIR = REPO_ROOT / "urdf" / "generated"

# HOME_DXARM for dxarm: q = np.zeros(7), q[1] = -30 deg. The rest posture inits here and
# the home-bias cost pulls the arm joints back here.
HOME_DXARM = np.deg2rad([0.0, -30.0, 0.0, 0.0, 0.0, 0.0, 0.0])
XARM_ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
XARM_GRIPPER_CUT_JOINT = "gripper_fix"  # link_eef -> xarm_gripper_base_link
XARM_FLANGE_LINK = "link_eef"

RUKA_PREFIX = "ruka_"
RUKA_TIP_LINKS = tuple(
    RUKA_PREFIX + name
    for name in ("thumb_actual_tip", "index_actual_tip", "middle_actual_tip", "ring_actual_tip", "pinky_actual_tip")
)
# Measured flange -> ruka mount offset, from 06-10_xarm_ruka_bimanual_teleop.py's
# HAND_MOUNT_XYZ (the ruka base sits ~3.8 cm off the eef center, not on it).
T_FLANGE_RUKA = np.eye(4)
T_FLANGE_RUKA[:3, 3] = (0.00216402, 0.0382359, 0.00282698)

RUKA_ROOT_LINK = RUKA_PREFIX + "base_new"  # carries the coarse whole-hand collision proxy
COARSE_HAND_PAD = 0.01  # padding added to the fitted hand bounding sphere

# Shared base placements for the dual-arm setup (also used by the client-side viser scene).
# l/r follow the operator's perspective facing the rig: dxarm-l is the arm on your left (+Y).
DXARM_L_POSITION = (0.0, 0.09687363, 0.14767363)
DXARM_L_WXYZ = (0.92387953, -0.38268343, 0.0, 0.0)
DXARM_R_POSITION = (0.0, -0.09687363, 0.14767363)
DXARM_R_WXYZ = (0.92387953, 0.38268343, 0.0, 0.0)


def pose_to_matrix(
    position: tuple[float, float, float],
    wxyz: tuple[float, float, float, float],
) -> NDArray[np.float64]:
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(wxyz), scalar_first=True).as_matrix()
    T[:3, 3] = position
    return T


@dataclass
class Embodiment:
    """WHAT robot — pure data, no behavior."""

    urdf: yourdfpy.URDF
    robot: pk.Robot
    robot_coll: RobotCollision
    target_links: list[str]  # 1 for gripper, N for ruka fingertips
    rest_cfg: NDArray[np.float64]  # warm-start posture
    world_T_base: NDArray[np.float64]  # (4, 4) dxarm-l vs dxarm-r placement
    urdf_path: Path  # source (or generated) file, for client-side visualization


def load_urdf(path: Path) -> yourdfpy.URDF:
    """Load a URDF while resolving relative mesh paths."""

    def filename_handler(fname: str) -> str:
        return yourdfpy.filename_handler_magic(fname, dir=path.parent)

    return yourdfpy.URDF.load(path, filename_handler=filename_handler)


def arm_warmstart_rest_cfg(robot: pk.Robot) -> NDArray[np.float64]:
    """Default config with the arm joints (by name) set to HOME_DXARM."""
    rest_cfg = np.array(robot.joint_var_cls.default_factory())
    actuated = list(robot.joints.actuated_names)
    for name, angle in zip(XARM_ARM_JOINTS, HOME_DXARM):
        rest_cfg[actuated.index(name)] = angle
    return rest_cfg


def arm_home_cfg_and_mask(robot: pk.Robot) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Full-dof HOME_DXARM vector and the 0/1 mask selecting the arm joints (by name)."""
    actuated = list(robot.joints.actuated_names)
    cfg = np.zeros(len(actuated))
    mask = np.zeros(len(actuated))
    for name, angle in zip(XARM_ARM_JOINTS, HOME_DXARM):
        cfg[actuated.index(name)] = angle
        mask[actuated.index(name)] = 1.0
    return cfg, mask


def xarm_gripper(
    world_T_base: NDArray[np.floating],
    urdf_path: Path = XARM_GRIPPER_URDF,
) -> Embodiment:
    urdf = load_urdf(urdf_path)
    rest_cfg = arm_warmstart_rest_cfg(pk.Robot.from_urdf(urdf))
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=rest_cfg)

    required = ("link_tcp", "left_tip", "right_tip", "xarm_gripper_base_link")
    missing = [name for name in required if name not in robot.links.names]
    if missing:
        raise ValueError(f"xArm link(s) missing from {urdf_path}: {missing}")

    return Embodiment(
        urdf=urdf,
        robot=robot,
        robot_coll=RobotCollision.from_urdf(urdf),
        target_links=["link_tcp"],
        rest_cfg=rest_cfg,
        world_T_base=np.asarray(world_T_base, dtype=np.float64),
        urdf_path=urdf_path,
    )


def hand_bounding_sphere(urdf: yourdfpy.URDF, robot: pk.Robot) -> tuple[NDArray[np.float64], float]:
    """(center in RUKA_ROOT_LINK frame, radius) covering all hand collision meshes at zero config."""
    wxyz_xyz = np.array(robot.forward_kinematics(cfg=np.array(robot.joint_var_cls.default_factory())))
    names = robot.links.names

    def T_base_link(link: str) -> NDArray[np.float64]:
        T = np.eye(4)
        T[:3, :3] = R.from_quat(wxyz_xyz[names.index(link), :4], scalar_first=True).as_matrix()
        T[:3, 3] = wxyz_xyz[names.index(link), 4:]
        return T

    T_root_inv = np.linalg.inv(T_base_link(RUKA_ROOT_LINK))
    points = []
    for link in names:
        if not link.startswith(RUKA_PREFIX):
            continue
        mesh = RobotCollision._get_trimesh_collision_geometries(urdf, link)  # noqa: SLF001
        if mesh.vertices.shape[0] == 0:
            continue
        T = T_root_inv @ T_base_link(link)
        points.append(mesh.vertices @ T[:3, :3].T + T[:3, 3])
    if not points:
        raise ValueError("No hand collision meshes found to bound")
    pts = np.concatenate(points, axis=0)
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    radius = float(np.linalg.norm(pts - center, axis=1).max()) + COARSE_HAND_PAD
    return center, radius


def coarse_hand_ignore_pairs(link_names: list[str]) -> tuple[tuple[str, str], ...]:
    """Self-collision pairs to drop: hand-vs-hand, and fine-hand-vs-arm.

    The whole hand stays arm-aware through RUKA_ROOT_LINK, whose collision
    capsule is inflated to bound the hand (see add_collision_sphere), and every
    fine hand link keeps its individual world (floor) collision for free.
    """
    hand = [n for n in link_names if n.startswith(RUKA_PREFIX)]
    arm = [n for n in link_names if not n.startswith(RUKA_PREFIX)]
    pairs = [(a, b) for i, a in enumerate(hand) for b in hand[i + 1 :]]
    pairs += [(h, a) for h in hand if h != RUKA_ROOT_LINK for a in arm]
    return tuple(pairs)


def xarm_ruka(
    side: str,
    world_T_base: NDArray[np.floating],
    arm_urdf_path: Path = XARM_GRIPPER_URDF,
    coarse_hand_coll: bool = True,
) -> Embodiment:
    """xArm7 with the gripper pruned and the ruka hand grafted onto the flange.

    Composed in memory from the unmodified source URDFs; the merged model is
    written to urdf/generated/ so the client-side viser scene can load it too.

    With coarse_hand_coll, hand-internal self-collision pairs are dropped and
    hand-vs-arm is checked against a single fitted whole-hand bounding sphere
    instead of 17 per-link capsules (~630 -> ~190 active pairs).
    """
    model = uc.load_model(arm_urdf_path)
    uc.prune_subtree(model, XARM_GRIPPER_CUT_JOINT)
    uc.attach(model, uc.load_model(RUKA_URDF[side]), XARM_FLANGE_LINK, T_FLANGE_RUKA, prefix=RUKA_PREFIX)
    model.name = f"xarm7_ruka_{side}"
    generated_path = uc.save(model, GENERATED_DIR / f"xarm7_ruka_{side}.urdf")

    urdf = uc.load_urdf(generated_path)
    probe = pk.Robot.from_urdf(urdf)
    rest_cfg = arm_warmstart_rest_cfg(probe)
    # Ruka joints rest at zero (clipped into limits): zero is less curled than
    # the limit midpoint, which is pyroki's own default (per 09-5_ruka.py).
    lower = np.asarray(probe.joints.lower_limits)
    upper = np.asarray(probe.joints.upper_limits)
    for i, joint in enumerate(probe.joints.actuated_names):
        if joint.startswith(RUKA_PREFIX):
            rest_cfg[i] = np.clip(0.0, lower[i], upper[i])
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=rest_cfg)

    missing = [name for name in RUKA_TIP_LINKS if name not in robot.links.names]
    if missing:
        raise ValueError(f"ruka fingertip link(s) missing from {generated_path}: {missing}")

    if coarse_hand_coll:
        center, radius = hand_bounding_sphere(urdf, robot)
        uc.add_collision_sphere(model, RUKA_ROOT_LINK, center, radius, name="hand_coll_proxy")
        generated_path = uc.save(model, GENERATED_DIR / f"xarm7_ruka_{side}_coarse.urdf")
        urdf = uc.load_urdf(generated_path)
        robot = pk.Robot.from_urdf(urdf, default_joint_cfg=rest_cfg)
        robot_coll = RobotCollision.from_urdf(urdf, user_ignore_pairs=coarse_hand_ignore_pairs(list(robot.links.names)))
    else:
        robot_coll = RobotCollision.from_urdf(urdf)

    return Embodiment(
        urdf=urdf,
        robot=robot,
        robot_coll=robot_coll,
        target_links=list(RUKA_TIP_LINKS),
        rest_cfg=rest_cfg,
        world_T_base=np.asarray(world_T_base, dtype=np.float64),
        urdf_path=generated_path,
    )


def dxarm_left(ee: str = "gripper", coarse_hand_coll: bool = True) -> Embodiment:
    return _dxarm("left", pose_to_matrix(DXARM_L_POSITION, DXARM_L_WXYZ), ee, coarse_hand_coll)


def dxarm_right(ee: str = "gripper", coarse_hand_coll: bool = True) -> Embodiment:
    return _dxarm("right", pose_to_matrix(DXARM_R_POSITION, DXARM_R_WXYZ), ee, coarse_hand_coll)


def _dxarm(side: str, world_T_base: NDArray[np.float64], ee: str, coarse_hand_coll: bool) -> Embodiment:
    if ee == "gripper":
        return xarm_gripper(world_T_base)
    if ee == "ruka":
        return xarm_ruka(side, world_T_base, coarse_hand_coll=coarse_hand_coll)
    raise ValueError(f"Unknown end effector {ee!r}; expected 'gripper' or 'ruka'")
