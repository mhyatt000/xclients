from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation as R
from webpolicy.base_policy import BasePolicy

from xclients.motion.embodiment import dxarm_left, dxarm_right, Embodiment
from xclients.motion.hand_planner import HandPlanner, RukaWeights
from xclients.motion.planner import OnlinePlanner
from xclients.motion.retargeter import GripperRetargeter, Retargeter, RukaRetargeter
from xclients.motion.types import CostWeights
from xclients.viser_webui import UrdfAsset

EndEffector = Literal["gripper", "ruka"]


@dataclass
class Unit:
    """One driven embodiment: which hand slots it consumes, and how."""

    hands: tuple[str, ...]  # obs slots this unit consumes, e.g. ("left",) or ("left", "right")
    retargeter: Retargeter
    planner: OnlinePlanner | HandPlanner


class RetargetPolicy(BasePolicy):
    """Composition root: slot-keyed hands in, slot-keyed joint commands out.

    obs: {"hands": {"left": (21, 3), "right": (21, 3)}} — either slot may be absent.
    returns: {unit_name: {"q": (dof,), "aperture": float | None}}
    """

    def __init__(self, units: dict[str, Unit]) -> None:
        self.units = units

    def step(self, obs: dict) -> dict:
        hands = obs.get("hands") or {}
        out = {}
        for name, unit in self.units.items():
            kp3ds = [hands.get(slot) for slot in unit.hands]
            if all(kp3d is not None for kp3d in kp3ds):
                targets = unit.retargeter(*kp3ds)
                q = unit.planner.solve(targets)
                out[name] = {"q": np.asarray(q), "aperture": targets.aperture}
            else:
                out[name] = {"q": np.asarray(unit.planner.hold()), "aperture": None}
        return out

    def reset(self, payload: dict | None = None) -> None:
        for unit in self.units.values():
            unit.planner.reset()


def _make_unit(slot: str, emb: Embodiment, ee: EndEffector, weights: CostWeights, len_traj: int, dt: float) -> Unit:
    if ee == "ruka":
        # Keypoint-space retargeting: 21-point local/global alignment with a
        # solved scale (HandPlanner), not SE3 link targets.
        return Unit((slot,), RukaRetargeter(emb), HandPlanner(emb, RukaWeights()))
    return Unit((slot,), GripperRetargeter(emb), OnlinePlanner(emb, weights, len_traj, dt))


def default_units(
    weights: CostWeights = CostWeights(),
    len_traj: int = 5,
    dt: float = 0.1,
    left_ee: EndEffector = "gripper",
    right_ee: EndEffector = "gripper",
    coarse_hand_coll: bool = True,
) -> dict[str, Unit]:
    """Bimanual default: left hand drives dxarm-l, right hand drives dxarm-r."""
    return {
        "dxarm-l": _make_unit("left", dxarm_left(left_ee, coarse_hand_coll), left_ee, weights, len_traj, dt),
        "dxarm-r": _make_unit("right", dxarm_right(right_ee, coarse_hand_coll), right_ee, weights, len_traj, dt),
    }


def unit_assets(units: dict[str, Unit]) -> dict[str, UrdfAsset]:
    """Viser assets matching each unit's solve model (same URDF file, same base pose)."""
    assets = {}
    for name, unit in units.items():
        emb = unit.planner.emb
        T = emb.world_T_base
        assets[name] = UrdfAsset(
            path=emb.urdf_path,
            position=tuple(T[:3, 3]),
            wxyz=tuple(R.from_matrix(T[:3, :3]).as_quat(scalar_first=True)),
        )
    return assets
