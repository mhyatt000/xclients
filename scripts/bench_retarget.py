"""Micro-benchmark for the retargeting solve in isolation (no camera/WiLoR/comms).

Answers: can the JAX keypoint-retarget solve sustain 30 Hz control, and is CPU or
GPU faster for it? Drives the ruka HandPlanner with large, fast-moving 21-point
MANO keypoints (the "many keypoints that move a lot" worst case) and times only
`retargeter -> planner.solve`, which is the piece the real pipeline offloads.

Pick the backend with the env var, e.g.
    JAX_PLATFORMS=cpu  python scripts/bench_retarget.py
    JAX_PLATFORMS=cuda python scripts/bench_retarget.py --arms 2 --concurrent
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jax"))

import numpy as np
import tyro

import jax

from xclients.motion.embodiment import (
    DXARM_L_POSITION,
    DXARM_L_WXYZ,
    DXARM_R_POSITION,
    DXARM_R_WXYZ,
    pose_to_matrix,
    xarm_gripper,
    xarm_ruka,
)
from xclients.motion.beam import BeamConfig
from xclients.motion.hand_planner import HandPlanner, RukaWeights
from xclients.motion.planner import OnlinePlanner
from xclients.motion.retargeter import GripperRetargeter, RukaRetargeter
from xclients.motion.types import CostWeights

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_arm_urdf(override: Path | None) -> Path:
    """The xArm7 base URDF. Gripper needs left_tip/right_tip links (only the
    retarget_helpers copy has them); ruka grafts its own hand so the root copy works too."""
    if override is not None:
        return Path(override).expanduser().resolve()
    for c in (
        REPO_ROOT / "retarget_helpers" / "hand" / "xarm" / "xarm7_standalone.urdf",
        Path("/data/projects/pyroki/examples/retarget_helpers/hand/xarm/xarm7_standalone.urdf"),
        REPO_ROOT / "xarm7_standalone.urdf",  # ruka-only fallback (lacks gripper tip links)
    ):
        if c.exists():
            return c
    raise FileNotFoundError("no xArm7 URDF found; pass --arm-urdf")


@dataclass
class BenchConfig:
    steps: int = 300  # warm (post-JIT) solves to time
    arms: int = 2  # 1 = one hand only; 2 = both hands (what the real rig runs)
    amp: float = 0.15  # translation oscillation amplitude (m) — "move a lot"
    curl: float = 0.9  # finger-curl oscillation amplitude (rad-ish) — reshape the hand each frame
    freq: float = 3.0  # oscillation cycles across the run (higher => bigger frame-to-frame jumps)
    concurrent: bool = False  # run the two arms in parallel threads (mirrors the LatestWorker layout)
    static: bool = False  # freeze the hand instead of moving it (isolates motion's cost)
    max_iters: int | None = None  # override LM iterations (RukaWeights/CostWeights); ignored by the beam
    ee: str = "ruka"  # "ruka" (HandPlanner) or "gripper" (OnlinePlanner)
    beam: bool = False  # solve with IK-Beam (multi-seed staged LM) instead of a single solve; works for either ee
    lm: list[int] = field(default_factory=lambda: [6, 10])  # beam: LM steps per stage
    batch: list[int] = field(default_factory=lambda: [64, 4])  # beam: seeds entering each stage
    seed_noise: float = 0.1  # beam: joint-space seed perturbation stddev (rad)
    arm_urdf: Path | None = None  # override the xArm7 base URDF (else auto-resolved)


def base_hand(center: np.ndarray) -> np.ndarray:
    """A plausible splayed 21-point MANO hand (world frame) centred at `center`."""
    kp = np.zeros((21, 3), dtype=np.float64)
    # Finger roots fan out in y; each finger extends along +x in 3 segments + tip.
    finger_y = {"thumb": -0.045, "index": -0.02, "middle": 0.0, "ring": 0.02, "pinky": 0.04}
    seg = {"thumb": 0.030, "index": 0.040, "middle": 0.045, "ring": 0.040, "pinky": 0.032}
    order = ["thumb", "index", "middle", "ring", "pinky"]
    idx = 1
    for f in order:
        y = finger_y[f]
        for j in range(4):  # joint1, joint2, joint3, tip
            kp[idx] = [0.03 + seg[f] * (j + 1), y, 0.0 if f != "thumb" else -0.02 * (j + 1)]
            idx += 1
    return kp + center


def curl_hand(kp: np.ndarray, amount: float) -> np.ndarray:
    """Bend the outer finger joints toward the palm to change the hand's shape."""
    out = kp.copy()
    # keypoints 3,4 / 7,8 / 11,12 / 15,16 / 19,20 are the outer segments+tips of each finger
    for tip in (4, 8, 12, 16, 20):
        for k in (tip - 1, tip):
            out[k, 0] -= amount * 0.03 * (k - (tip - 2))
            out[k, 2] -= amount * 0.05 * (k - (tip - 2))
    return out


def make_sequence(center: np.ndarray, cfg: BenchConfig, n: int) -> list[np.ndarray]:
    base = base_hand(center)
    seq = []
    for i in range(n):
        if cfg.static:
            seq.append(base)
            continue
        phase = 2.0 * np.pi * cfg.freq * i / max(n, 1)
        shift = np.array([cfg.amp * np.sin(phase), cfg.amp * np.cos(1.3 * phase), cfg.amp * np.sin(0.7 * phase)])
        curl = cfg.curl * 0.5 * (1.0 + np.sin(2.1 * phase))
        seq.append(curl_hand(base + shift, curl))
    return seq


@dataclass
class Arm:
    name: str
    retargeter: RukaRetargeter
    planner: HandPlanner
    seq: list[np.ndarray]

    def solve_step(self, i: int) -> None:
        targets = self.retargeter(self.seq[i % len(self.seq)])
        self.planner.solve(targets)  # returns np.ndarray -> forces device->host sync


def build_arms(cfg: BenchConfig) -> list[Arm]:
    arm_urdf = resolve_arm_urdf(cfg.arm_urdf)
    specs = [
        ("l", "left", pose_to_matrix(DXARM_L_POSITION, DXARM_L_WXYZ), np.array([0.35, -0.30, 0.45])),
        ("r", "right", pose_to_matrix(DXARM_R_POSITION, DXARM_R_WXYZ), np.array([0.35, 0.30, 0.45])),
    ][: cfg.arms]
    beam = BeamConfig(lm=list(cfg.lm), batch=list(cfg.batch), seed_noise=cfg.seed_noise) if cfg.beam else None
    tag = "beam" if cfg.beam else cfg.ee
    arms = []
    for suffix, side, world_T_base, center in specs:
        seq = make_sequence(center, cfg, cfg.steps + 5)
        if cfg.ee == "ruka":
            emb = xarm_ruka(side, world_T_base, arm_urdf_path=arm_urdf, coarse_hand_coll=True)
            weights = RukaWeights() if cfg.max_iters is None else RukaWeights(max_iterations=cfg.max_iters)
            planner = HandPlanner(emb, weights, beam=beam)
            arms.append(Arm(f"{tag}-{suffix}", RukaRetargeter(emb), planner, seq))
        elif cfg.ee == "gripper":
            emb = xarm_gripper(world_T_base, urdf_path=arm_urdf)
            weights = CostWeights() if cfg.max_iters is None else CostWeights(max_iterations=cfg.max_iters)
            planner = OnlinePlanner(emb, weights, beam=beam)
            arms.append(Arm(f"{tag}-{suffix}", GripperRetargeter(emb), planner, seq))
        else:
            raise ValueError(f"unknown --ee {cfg.ee!r}; expected 'ruka' or 'gripper'")
    return arms


def pct(xs: list[float], p: float) -> float:
    return sorted(xs)[min(len(xs) - 1, int(p / 100.0 * len(xs)))]


def report(label: str, per_step_ms: list[float], n_arms: int) -> None:
    mean = statistics.mean(per_step_ms)
    print(f"\n  {label}")
    print(f"    step latency (ms): mean {mean:6.2f} | p50 {pct(per_step_ms,50):6.2f} "
          f"| p90 {pct(per_step_ms,90):6.2f} | p99 {pct(per_step_ms,99):6.2f} | max {max(per_step_ms):6.2f}")
    print(f"    throughput: {1000.0/mean:6.1f} Hz control  ({n_arms} arm(s) per step)   "
          f"{'>=30Hz OK' if 1000.0/mean >= 30 else 'BELOW 30Hz'}")


def main(cfg: BenchConfig) -> None:
    dev = jax.devices()
    solver = f"{cfg.ee}/" + ("beam" if cfg.beam else "single")
    if cfg.beam:
        solver += f" lm={cfg.lm} batch={cfg.batch} seed_noise={cfg.seed_noise}"
    elif cfg.max_iters is not None:
        solver += f" max_iters={cfg.max_iters}"
    print(f"=== retarget solve benchmark | backend={dev[0].platform} devices={len(dev)} "
          f"| solver={solver} arms={cfg.arms} concurrent={cfg.concurrent} static={cfg.static} ===")
    arms = build_arms(cfg)

    t0 = time.perf_counter()
    for arm in arms:
        arm.solve_step(0)
        arm.planner.reset()
    print(f"  JIT compile (first solve, both arms): {time.perf_counter()-t0:6.2f} s")

    # A few untimed warm iters so the cache/allocator settle.
    for i in range(1, 5):
        for arm in arms:
            arm.solve_step(i)

    per_step_ms: list[float] = []
    if cfg.concurrent and len(arms) > 1:
        pool = ThreadPoolExecutor(max_workers=len(arms))
        for i in range(5, cfg.steps + 5):
            t = time.perf_counter()
            list(pool.map(lambda a: a.solve_step(i), arms))
            per_step_ms.append((time.perf_counter() - t) * 1000.0)
        pool.shutdown()
        report("both arms IN PARALLEL (threads)", per_step_ms, len(arms))
    else:
        for i in range(5, cfg.steps + 5):
            t = time.perf_counter()
            for arm in arms:
                arm.solve_step(i)
            per_step_ms.append((time.perf_counter() - t) * 1000.0)
        report(f"{len(arms)} arm(s) SEQUENTIAL", per_step_ms, len(arms))

    # Always also report the single-arm cost (the real pipeline solves each arm in its own worker).
    if len(arms) >= 1 and not (cfg.concurrent and len(arms) > 1):
        single = []
        for i in range(5, cfg.steps + 5):
            t = time.perf_counter()
            arms[0].solve_step(i)
            single.append((time.perf_counter() - t) * 1000.0)
        report("single arm only", single, 1)


if __name__ == "__main__":
    main(tyro.cli(BenchConfig))
