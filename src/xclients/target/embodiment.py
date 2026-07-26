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

XARM_ARM_WARMSTART_DEG = np.array([0.0, -45.0, 0.0, 35.0, 0.0, 65.0, 90.0])
XARM_ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
XARM_GRIPPER_CUT_JOINT = "gripper_fix"  # link_eef -> xarm_gripper_base_link
XARM_FLANGE_LINK = "link_eef"

RUKA_PREFIX = "ruka_"
RUKA_TIP_LINKS = tuple(
    RUKA_PREFIX + name
    for name in ("thumb_actual_tip", "index_actual_tip", "middle_actual_tip", "ring_actual_tip", "pinky_actual_tip")
)
# TODO(calibration): measured flange -> ruka base_new transform; identity until then.
T_FLANGE_RUKA = np.eye(4)

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
    """Default config with the arm joints (by name) set to the warm-start posture."""
    rest_cfg = np.array(robot.joint_var_cls.default_factory())
    actuated = list(robot.joints.actuated_names)
    for name, angle_deg in zip(XARM_ARM_JOINTS, XARM_ARM_WARMSTART_DEG):
        rest_cfg[actuated.index(name)] = np.deg2rad(angle_deg)
    return rest_cfg


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


def xarm_ruka(
    side: str,
    world_T_base: NDArray[np.floating],
    arm_urdf_path: Path = XARM_GRIPPER_URDF,
) -> Embodiment:
    """xArm7 with the gripper pruned and the ruka hand grafted onto the flange.

    Composed in memory from the unmodified source URDFs; the merged model is
    written to urdf/generated/ so the client-side viser scene can load it too.
    """
    model = uc.load_model(arm_urdf_path)
    uc.prune_subtree(model, XARM_GRIPPER_CUT_JOINT)
    uc.attach(model, uc.load_model(RUKA_URDF[side]), XARM_FLANGE_LINK, T_FLANGE_RUKA, prefix=RUKA_PREFIX)
    model.name = f"xarm7_ruka_{side}"
    generated_path = uc.save(model, GENERATED_DIR / f"xarm7_ruka_{side}.urdf")

    urdf = uc.load_urdf(generated_path)
    rest_cfg = arm_warmstart_rest_cfg(pk.Robot.from_urdf(urdf))
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=rest_cfg)

    missing = [name for name in RUKA_TIP_LINKS if name not in robot.links.names]
    if missing:
        raise ValueError(f"ruka fingertip link(s) missing from {generated_path}: {missing}")

    return Embodiment(
        urdf=urdf,
        robot=robot,
        robot_coll=RobotCollision.from_urdf(urdf),
        target_links=list(RUKA_TIP_LINKS),
        rest_cfg=rest_cfg,
        world_T_base=np.asarray(world_T_base, dtype=np.float64),
        urdf_path=generated_path,
    )


def dxarm_left(ee: str = "gripper") -> Embodiment:
    return _dxarm("left", pose_to_matrix(DXARM_L_POSITION, DXARM_L_WXYZ), ee)


def dxarm_right(ee: str = "gripper") -> Embodiment:
    return _dxarm("right", pose_to_matrix(DXARM_R_POSITION, DXARM_R_WXYZ), ee)


def _dxarm(side: str, world_T_base: NDArray[np.float64], ee: str) -> Embodiment:
    if ee == "gripper":
        return xarm_gripper(world_T_base)
    if ee == "ruka":
        return xarm_ruka(side, world_T_base)
    raise ValueError(f"Unknown end effector {ee!r}; expected 'gripper' or 'ruka'")
