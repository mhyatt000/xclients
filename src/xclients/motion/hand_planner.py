"""Keypoint-space hand retargeting solver for the xArm+ruka rig.

Port of the dexpilot-style formulation from pyroki's `09-5_ruka.py` /
`06-10_xarm_ruka_bimanual_teleop.py`, adapted to our Embodiment/World stack:

- ALL 21 MANO keypoints map to ruka links (wrist->backhand, every knuckle),
  not just the 5 fingertips, so every finger segment gets its own gradient.
- Local alignment matches pairwise keypoint deltas (position + bone direction)
  between kinematically-connected pairs, with the human->robot scale solved as
  an optimization variable rather than calibrated by heuristic.
- Global alignment (absolute link-to-keypoint positions) is a separate, weaker
  cost, so hand SHAPE and hand PLACEMENT don't fight over one residual.

Single-step solve warm-started from the previous config (no receding horizon):
quality comes from the cost structure, and the smaller problem solves faster.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxls
import numpy as onp
import pyroki as pk

from xclients.motion.embodiment import Embodiment, RUKA_PREFIX
from xclients.motion.types import KeypointTargets
from xclients.motion.world import World

# MANO keypoint index -> ruka link (pre-prefix), from pyroki's 09-5_ruka.py.
MANO_TO_RUKA = {
    0: "backhand",
    1: "thumb___joint_1",
    2: "thumb___joint_2",
    3: "thumb___joint_3",
    4: "thumb_actual_tip",
    5: "mcp",
    6: "pip",
    7: "finger___joint_3",
    8: "index_actual_tip",
    9: "mcp_2",
    10: "pip_2",
    11: "finger___joint_3_2",
    12: "middle_actual_tip",
    13: "mcp_3",
    14: "pip_3",
    15: "finger___joint_3_3",
    16: "ring_actual_tip",
    17: "mcp_4",
    18: "pinky___joint_2",
    19: "pinky___joint_3",
    20: "pinky_actual_tip",
}


@dataclass(frozen=True)
class RukaWeights:
    """Hand-retargeting cost weights (defaults from 06-10's DEFAULT_WEIGHTS)."""

    local_alignment: float = 10.0
    global_alignment: float = 13.0
    joint_smoothness: float = 2.0
    rest: float = 0.05
    limit: float = 100.0
    self_collision: float = 5.0
    self_collision_margin: float = 0.02
    world_collision: float = 5.0
    world_collision_margin: float = 0.05
    max_iterations: int = 30


def mano_ruka_mapping(robot: pk.Robot, prefix: str = RUKA_PREFIX) -> tuple[jnp.ndarray, jnp.ndarray]:
    """(ruka link indices, MANO keypoint indices), aligned pairs."""
    link_idx, mano_idx = [], []
    for mano_i, name in MANO_TO_RUKA.items():
        full = prefix + name
        if full not in robot.links.names:
            raise ValueError(f"Ruka link {full!r} missing from robot (mapping expects it)")
        link_idx.append(robot.links.names.index(full))
        mano_idx.append(mano_i)
    return jnp.array(link_idx), jnp.array(mano_idx)


def create_conn_tree(robot: pk.Robot, link_indices: jnp.ndarray) -> jnp.ndarray:
    """NxN mask: 1 where two retargeted links are directly chain-connected
    without another retargeted link between them (port of retarget_helpers._utils)."""
    n = len(link_indices)
    parent_joint = onp.asarray(robot.links.parent_joint_indices)
    joint_parent = onp.asarray(robot.joints.parent_indices)
    joint_idx = onp.array([parent_joint[int(li)] for li in link_indices])
    joint_idx_set = set(int(j) for j in joint_idx)

    def directly_connected(i: int, j: int) -> bool:
        for a, b in ((joint_idx[i], joint_idx[j]), (joint_idx[j], joint_idx[i])):
            current = int(b)
            while current != -1:
                parent = int(joint_parent[current])
                if parent == int(a):
                    return True
                if parent in joint_idx_set:
                    break
                current = parent
        return False

    conn = onp.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if directly_connected(i, j):
                conn[i, j] = conn[j, i] = 1.0
    return jnp.array(conn)


class HandPlanner:
    """Warm-started single-step keypoint retargeting: (21,3) base-frame keypoints -> q."""

    def __init__(
        self,
        emb: Embodiment,
        weights: RukaWeights = RukaWeights(),
        world: World | None = None,
    ) -> None:
        self.emb = emb
        self.weights = weights
        world = World.default() if world is None else world
        self.world_coll = world.in_base_frame(emb.world_T_base)
        self.world_masks = world.link_masks(list(emb.robot.links.names))
        self.link_indices, self.mano_indices = mano_ruka_mapping(emb.robot)
        self.mano_mask = create_conn_tree(emb.robot, self.link_indices)
        self.prev_cfg = onp.array(emb.rest_cfg)

    def solve(self, targets: KeypointTargets) -> onp.ndarray:
        cfg = _solve_hand_retarget_jax(
            self.emb.robot,
            self.emb.robot_coll,
            self.world_coll,
            self.world_masks,
            jnp.asarray(targets.kp3d),
            self.link_indices,
            self.mano_indices,
            self.mano_mask,
            jnp.asarray(self.prev_cfg),
            jnp.asarray(self.emb.rest_cfg),
            self.weights,
        )
        self.prev_cfg = onp.array(cfg)
        return self.prev_cfg

    def hold(self) -> onp.ndarray:
        return self.prev_cfg

    def reset(self) -> None:
        self.prev_cfg = onp.array(self.emb.rest_cfg)


@jdc.jit
def _solve_hand_retarget_jax(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: tuple[pk.collision.CollGeom, ...],
    world_masks: tuple[jnp.ndarray, ...],
    target_keypoints: jnp.ndarray,  # (21, 3), robot base frame
    link_indices: jnp.ndarray,
    mano_indices: jnp.ndarray,
    mano_mask: jnp.ndarray,
    prev_cfg: jnp.ndarray,
    rest_cfg: jnp.ndarray,
    weights: jdc.Static[RukaWeights],
) -> jax.Array:
    n = link_indices.shape[0]
    off_diag = 1.0 - jnp.eye(n)

    class ScaleVar(  # pylint: disable=missing-class-docstring
        jaxls.Var[jax.Array],
        default_factory=lambda: jnp.ones((n, n)),
    ): ...

    joint_var = robot.joint_var_cls(0)
    scale_var = ScaleVar(0)

    def mapped_link_positions(cfg: jax.Array) -> jax.Array:
        return robot.forward_kinematics(cfg=cfg)[..., link_indices, 4:7]

    keypoints = target_keypoints[mano_indices]

    @jaxls.Cost.factory(name="LocalAlignmentCost")
    def local_alignment_cost(
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[jnp.ndarray],
        scale_var: ScaleVar,
    ):
        pos = mapped_link_positions(vals[joint_var])
        delta_mano = keypoints[:, None] - keypoints[None, :]
        delta_robot = pos[:, None] - pos[None, :]

        scale = vals[scale_var][..., None]
        res_position = (delta_mano - delta_robot * scale) * off_diag[..., None] * mano_mask[..., None]

        mano_dir = delta_mano / jnp.linalg.norm(delta_mano + 1e-6, axis=-1, keepdims=True)
        robot_dir = delta_robot / jnp.linalg.norm(delta_robot + 1e-6, axis=-1, keepdims=True)
        res_angle = (1.0 - (mano_dir * robot_dir).sum(axis=-1)) * off_diag * mano_mask

        return jnp.concatenate([res_position.flatten(), res_angle.flatten()]) * weights.local_alignment

    @jaxls.Cost.factory(name="GlobalAlignmentCost")
    def global_alignment_cost(vals: jaxls.VarValues, joint_var: jaxls.Var[jnp.ndarray]):
        return (mapped_link_positions(vals[joint_var]) - keypoints).flatten() * weights.global_alignment

    @jaxls.Cost.factory(name="PrevSmoothnessCost")
    def prev_smoothness_cost(vals: jaxls.VarValues, joint_var: jaxls.Var[jnp.ndarray]):
        return (vals[joint_var] - prev_cfg).flatten() * weights.joint_smoothness

    costs: list[jaxls.Cost] = [
        local_alignment_cost(joint_var, scale_var),
        global_alignment_cost(joint_var),
        prev_smoothness_cost(joint_var),
        pk.costs.rest_cost(joint_var, rest_pose=rest_cfg, weight=weights.rest),
        pk.costs.limit_cost(robot, joint_var, weight=weights.limit),
        pk.costs.self_collision_cost(
            robot,
            robot_coll=robot_coll,
            joint_var=joint_var,
            margin=weights.self_collision_margin,
            weight=weights.self_collision,
        ),
    ]
    costs.extend(
        pk.costs.world_collision_cost(
            robot,
            robot_coll,
            joint_var,
            geom,
            margin=weights.world_collision_margin,
            weight=weights.world_collision * mask,
        )
        for geom, mask in zip(world_coll, world_masks)
    )

    solution = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var, scale_var])
        .analyze()
        .solve(
            verbose=False,
            initial_vals=jaxls.VarValues.make(
                (joint_var.with_value(prev_cfg), scale_var.with_value(jnp.ones((n, n))))
            ),
            termination=jaxls.TerminationConfig(max_iterations=weights.max_iterations),
        )
    )
    return solution[joint_var]
