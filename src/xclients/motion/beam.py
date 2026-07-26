"""Beam-search solve strategy (pyroki IK-Beam, Sec. III-C / Table III), reusable by any
warm-started LM solver in this package regardless of end-effector.

The idea is orthogonal to *what* is being solved: fan the warm-start out into a batch of
seeds, run a few fixed-count LM steps on all of them in parallel (``jax.vmap``), keep the
lowest-cost seeds, repeat, return the best. Each planner supplies its own ``solve_one``
(build-problem + fixed-iteration solve) and the joint value to perturb; the staging here is
shared. Fixed iteration counts (``early_termination=False`` in the callers) keep latency
deterministic, and the batch fans out over a leading axis so it costs little extra on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp


@dataclass
class BeamConfig:
    """IK-Beam schedule. One entry per stage: run ``lm[k]`` LM steps on ``batch[k]`` seeds,
    then keep the ``batch[k+1]`` lowest-cost for the next stage; the final stage returns the
    single best. Defaults reproduce the paper: 64 seeds x 6 steps -> keep 4 -> 10 steps.
    ``batch`` must be non-increasing and the same length as ``lm``.
    """

    lm: list[int] = field(default_factory=lambda: [6, 10])  # LM steps per stage
    batch: list[int] = field(default_factory=lambda: [64, 4])  # seeds entering each stage
    seed_noise: float = 0.1  # stddev (rad) of the joint-space perturbation; seed 0 is the exact warm-start
    lambda_initial: float = 10.0  # LM trust-region lambda seeding the first stage

    def __post_init__(self) -> None:
        if len(self.lm) != len(self.batch):
            raise ValueError(f"lm and batch must have one entry per stage; got {self.lm} and {self.batch}")
        if any(b <= 0 for b in self.batch) or any(s <= 0 for s in self.lm):
            raise ValueError("lm and batch entries must be positive")
        if any(self.batch[i + 1] > self.batch[i] for i in range(len(self.batch) - 1)):
            raise ValueError(f"batch sizes must be non-increasing across stages; got {self.batch}")


# solve_one(seed_value, lambda, n_steps) -> (solution_value, final_cost, final_lambda)
SolveOne = Callable[[jax.Array, jax.Array, int], tuple[jax.Array, jax.Array, jax.Array]]


def run_beam(
    warm_value: jax.Array,
    solve_one: SolveOne,
    lm: tuple[int, ...],
    batch: tuple[int, ...],
    seed_noise: float,
    lambda_initial: float,
    key: jax.Array,
) -> jax.Array:
    """Staged multi-seed LM over ``solve_one``. ``warm_value`` is the joint value to perturb
    into seeds (any shape; a config for single-step solvers, a trajectory for receding-horizon).
    Seed 0 is the exact warm-start. Returns the best solution value (same shape as warm_value).
    Call inside jit with ``lm``/``batch`` as static tuples so the stage loop unrolls."""
    noise = jax.random.normal(key, (batch[0],) + warm_value.shape) * seed_noise
    vals = (warm_value[None] + noise).at[0].set(warm_value)
    lambd = jnp.full((batch[0],), lambda_initial)
    vmapped = jax.vmap(solve_one, in_axes=(0, 0, None))

    sols = costs = None
    for k in range(len(lm)):
        sols, costs, lambdas = vmapped(vals, lambd, lm[k])
        if k + 1 < len(lm):
            keep = jnp.argsort(costs)[: batch[k + 1]]  # prune to the next stage's best seeds
            vals, lambd = sols[keep], lambdas[keep]
        else:
            return sols[jnp.argmin(costs)]
    return sols[jnp.argmin(costs)]  # unreachable (len(lm) >= 1); satisfies the type checker
