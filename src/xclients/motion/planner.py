from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk

from xclients.motion.embodiment import arm_home_cfg_and_mask, Embodiment
from xclients.motion.types import CostWeights, Targets
from xclients.motion.world import World


class OnlinePlanner:
    """Warm-started receding-horizon IK: targets -> q. Owns the previous-solution state."""

    def __init__(
        self,
        emb: Embodiment,
        weights: CostWeights = CostWeights(),
        len_traj: int = 5,
        dt: float = 0.1,
        world: World | None = None,
    ) -> None:
        self.emb = emb
        self.weights = weights
        self.len_traj = len_traj
        self.dt = dt
        world = World.default() if world is None else world
        self.world_coll = world.in_base_frame(emb.world_T_base)
        self.world_masks = world.link_masks(list(emb.robot.links.names))
        self.target_link_indices = jnp.array([emb.robot.links.names.index(n) for n in emb.target_links])
        home_cfg, home_mask = arm_home_cfg_and_mask(emb.robot)
        self.home_cfg = jnp.asarray(home_cfg)
        self.home_weight = jnp.asarray(weights.home * home_mask)
        self.sol_traj = emb.rest_cfg[None].repeat(len_traj, axis=0)

    def solve(self, targets: Targets) -> onp.ndarray:
        wxyz_xyz = onp.stack(
            [
                onp.concatenate([targets.poses[name].wxyz, targets.poses[name].position])
                for name in self.emb.target_links
            ],
            axis=0,
        )
        sol_traj, _sol_pos, _sol_wxyz = _solve_online_planning_jax(
            self.emb.robot,
            self.emb.robot_coll,
            self.world_coll,
            self.world_masks,
            jaxlie.SE3(jnp.asarray(wxyz_xyz)),
            self.target_link_indices,
            self.len_traj + 1,  # +1 for the start-anchor knot
            self.dt,
            jnp.asarray(self.sol_traj[0]),
            jnp.concatenate([self.sol_traj, self.sol_traj[-1:]], axis=0),
            self.home_cfg,
            self.home_weight,
            self.weights,
        )
        self.sol_traj = onp.array(sol_traj[1:])
        return self.sol_traj[0]

    def hold(self) -> onp.ndarray:
        """No target this tick: report the current head of the trajectory without advancing."""
        return self.sol_traj[0]

    def reset(self) -> None:
        self.sol_traj = self.emb.rest_cfg[None].repeat(self.len_traj, axis=0)


@jdc.jit
def _solve_online_planning_jax(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    world_masks: Sequence[jnp.ndarray],
    target_poses: jaxlie.SE3,
    target_links: jnp.ndarray,
    timesteps: jdc.Static[int],
    dt: float,
    start_cfg: jnp.ndarray,
    prev_sols: jnp.ndarray,
    home_cfg: jnp.ndarray,
    home_weight: jnp.ndarray,
    weights: jdc.Static[CostWeights],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    num_targets = len(target_links)

    def batched_rplus(pose: jaxlie.SE3, delta: jax.Array) -> jaxlie.SE3:
        return jax.vmap(jaxlie.manifold.rplus)(pose, delta.reshape(num_targets, -1))

    # Latent SE3 target trajectory, batched across target links.
    class BatchedSE3Var(  # pylint: disable=missing-class-docstring
        jaxls.Var[jaxlie.SE3],
        default_factory=lambda: jaxlie.SE3.identity((num_targets,)),
        retract_fn=batched_rplus,
        tangent_dim=jaxlie.SE3.tangent_dim * num_targets,
    ): ...

    traj_var = robot.joint_var_cls(jnp.arange(0, timesteps))
    traj_var_prev = robot.joint_var_cls(jnp.arange(0, timesteps - 1))
    traj_var_next = robot.joint_var_cls(jnp.arange(1, timesteps))
    pose_var = BatchedSE3Var(jnp.arange(0, timesteps))
    pose_var_prev = BatchedSE3Var(jnp.arange(0, timesteps - 1))
    pose_var_next = BatchedSE3Var(jnp.arange(1, timesteps))

    init_pose_vals = jaxlie.SE3(robot.forward_kinematics(prev_sols)[..., target_links, :])

    factors: list[jaxls.Cost] = []

    @jaxls.Cost.factory(name="SE3PoseMatchJointCost")
    def match_joint_to_pose_cost(
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[jnp.ndarray],
        pose_var: BatchedSE3Var,
    ):
        Ts_joint_world = robot.forward_kinematics(vals[joint_var])
        residual = ((jaxlie.SE3(Ts_joint_world[..., target_links, :])).inverse() @ vals[pose_var]).log()
        return residual.flatten() * weights.joint_pose_coupling

    @jaxls.Cost.factory(name="SE3SmoothnessCost")
    def pose_smoothness_cost(
        vals: jaxls.VarValues,
        pose_var: BatchedSE3Var,
        pose_var_prev: BatchedSE3Var,
    ):
        return (vals[pose_var].inverse() @ vals[pose_var_prev]).log().flatten() * weights.pose_smoothness

    @jaxls.Cost.factory(name="SE3PoseMatchCost")
    def pose_match_cost(vals: jaxls.VarValues, pose_var: BatchedSE3Var):
        return (
            (vals[pose_var].inverse() @ target_poses).log()
            * jnp.array([weights.pose_match_position] * 3 + [weights.pose_match_orientation] * 3)
        ).flatten()

    @jaxls.Cost.factory(name="MatchStartPoseCost")
    def match_start_pose_cost(vals: jaxls.VarValues, joint_var: jaxls.Var[jnp.ndarray]):
        return (vals[joint_var] - start_cfg).flatten() * weights.start_anchor

    factors.extend(
        [
            pose_match_cost(BatchedSE3Var(timesteps - 1)),
            pose_smoothness_cost(pose_var_next, pose_var_prev),
            match_start_pose_cost(robot.joint_var_cls(0)),
            match_joint_to_pose_cost(traj_var, pose_var),
            pk.costs.smoothness_cost(
                traj_var_prev,
                traj_var_next,
                weight=weights.traj_smoothness,
            ),
            pk.costs.limit_velocity_cost(
                jax.tree.map(lambda x: x[None], robot),
                traj_var_prev,
                traj_var_next,
                weight=weights.velocity_limit,
                dt=dt,
            ),
            pk.costs.limit_cost(
                jax.tree.map(lambda x: x[None], robot),
                traj_var,
                weight=weights.joint_limit,
            ),
            pk.costs.rest_cost(
                traj_var,
                jnp.array(traj_var.default_factory())[None],
                weight=weights.rest,
            ),
            # Home bias: arm joints only (per-dof weight is zero elsewhere), toward HOME_DXARM.
            pk.costs.rest_cost(
                traj_var,
                home_cfg[None],
                weight=home_weight[None],
            ),
            pk.costs.self_collision_cost(
                jax.tree.map(lambda x: x[None], robot),
                jax.tree.map(lambda x: x[None], robot_coll),
                traj_var,
                weight=weights.self_collision,
                margin=weights.self_collision_margin,
            ),
        ]
    )
    # One manipulability cost per target link: the cost factory would otherwise
    # broadcast a (num_targets,) index array against the (timesteps,) batch axis.
    factors.extend(
        [
            pk.costs.manipulability_cost(
                jax.tree.map(lambda x: x[None], robot),
                traj_var,
                weight=weights.manipulability,
                target_link_indices=target_links[i : i + 1],
            )
            for i in range(num_targets)
        ]
    )
    # Per-link 0/1 masks implement each world geom's ignore list (e.g. the mount
    # box exempts the base links bolted onto it).
    factors.extend(
        [
            pk.costs.world_collision_cost(
                jax.tree.map(lambda x: x[None], robot),
                jax.tree.map(lambda x: x[None], robot_coll),
                traj_var,
                jax.tree.map(lambda x: x[None], obs),
                weight=(weights.world_collision * mask)[None],
                margin=weights.world_collision_margin,
            )
            for obs, mask in zip(world_coll, world_masks)
        ]
    )

    solution = (
        jaxls.LeastSquaresProblem(factors, [traj_var, pose_var])
        .analyze()
        .solve(
            verbose=False,
            initial_vals=jaxls.VarValues.make((traj_var.with_value(prev_sols), pose_var.with_value(init_pose_vals))),
            termination=jaxls.TerminationConfig(max_iterations=weights.max_iterations),
        )
    )
    pose_traj = solution[pose_var]
    return solution[traj_var], pose_traj.translation(), pose_traj.rotation().wxyz
