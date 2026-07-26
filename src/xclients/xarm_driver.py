"""Minimal xArm7 joint-position driver for the retarget stack.

Streams servo-mode joint targets to a UFACTORY xArm7 over IP. The boundary is
radians; obs["q"] is the planner's full config vector and only q[:7]
(joint1..joint7) is driven. With cfg.gripper, obs["aperture"] (thumb-index
distance, meters) maps to the normalized 0=closed/1=open hardware gripper; use
--no-gripper for ruka or a bare flange.

Standalone hardware test (lab arms: 192.168.1.231 / 192.168.1.238):

    # read-only state echo, no motion enable
    python -m xclients.xarm_driver --ip 192.168.1.231
    # +-2 deg sinusoid on all joints through the real step() path
    python -m xclients.xarm_driver --ip 192.168.1.231 --wiggle
    # one gripper close/open cycle
    python -m xclients.xarm_driver --ip 192.168.1.231 --test-gripper
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time

import numpy as np
from numpy.typing import NDArray
import tyro
from webpolicy.base_policy import BasePolicy

ARM_DOFS = 7
SERVO_MODE = 1
POSITION_MODE = 0
READY_STATE = 0
STOP_STATE = 4
GRIPPER_RAW_CLOSED = 0
GRIPPER_RAW_OPEN = 850
GRIPPER_SPEED = 5000
GRIPPER_EPSILON = 0.01


@dataclass
class XArmDriverConfig:
    ip: str  # controller address; no default so the wrong arm is never driven
    gripper: bool = True  # False for ruka or a bare flange
    clear: bool = False  # clean latched controller warnings/errors at connect
    max_step: float = math.radians(1.0)  # rad, per joint, per step() call
    max_vel: float = 1.0  # rad/s per joint; effective clamp is min(max_step, max_vel * dt)
    gripper_open: float = 0.08  # aperture (m) that maps to a fully open gripper


class XArmDriver(BasePolicy):
    """obs {"q": (dof,), "aperture": float | None} -> stream q[:7] (+ gripper) to the arm.

    Connecting is passive; motion enable + servo mode happen on the first
    step(), which also seeds the clamp origin from the measured joints so the
    arm walks toward the first target at the clamped rate instead of jumping.
    """

    def __init__(self, cfg: XArmDriverConfig) -> None:
        from xarm.wrapper import XArmAPI

        self.cfg = cfg
        self.arm = XArmAPI(cfg.ip, is_radian=True)
        if not self.arm.connected:
            raise ConnectionError(f"could not connect to xArm at {cfg.ip}")
        if cfg.clear:
            self._check(self.arm.clean_warn(), "clean_warn")
            self._check(self.arm.clean_error(), "clean_error")
        if int(getattr(self.arm, "error_code", 0)) != 0:
            raise RuntimeError(
                f"xArm at {cfg.ip} reports error_code={self.arm.error_code}; rerun with --clear or clear it on the controller"
            )
        self._armed = False
        self._stopped = False
        self._last: NDArray[np.float64] | None = None
        self._last_time = 0.0
        self._last_gripper: float | None = None
        logging.info("connected to xArm at %s", cfg.ip)

    def joint_pos(self) -> NDArray[np.float64]:
        # Synchronous query, not the .angles cache: the cache is [0]*7 until the
        # SDK's async report thread delivers its first state packet.
        code, angles = self.arm.get_servo_angle(is_radian=True)
        self._check(code, "get_servo_angle")
        q = np.asarray(angles, dtype=np.float64)
        if q.shape != (ARM_DOFS,):
            raise RuntimeError(f"xArm returned joint state with shape {q.shape}; expected ({ARM_DOFS},)")
        return q

    def step(self, obs: dict) -> dict:
        q = np.asarray(obs["q"], dtype=np.float64)
        if q.ndim != 1 or q.shape[0] < ARM_DOFS:
            raise ValueError(f"expected q with at least {ARM_DOFS} joints, got shape {q.shape}")
        if not np.all(np.isfinite(q[:ARM_DOFS])):
            raise ValueError("q contains non-finite values")
        self._ensure_armed()
        self._assert_ready()

        now = time.monotonic()
        limit = min(self.cfg.max_step, self.cfg.max_vel * (now - self._last_time))
        target = self._last + np.clip(q[:ARM_DOFS] - self._last, -limit, limit)
        self._check(self.arm.set_servo_angle_j(target.tolist(), is_radian=True), "set_servo_angle_j")
        self._last, self._last_time = target, now

        aperture = obs.get("aperture")
        if self.cfg.gripper and aperture is not None:
            self._command_gripper(float(np.clip(float(aperture) / self.cfg.gripper_open, 0.0, 1.0)))
        return {"joint_pos": target, "gripper": self._last_gripper}

    def reset(self, payload: dict | None = None) -> None:
        if self._armed:
            self._last = self.joint_pos()
            self._last_time = time.monotonic()

    def close(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            if self._armed and getattr(self.arm, "connected", False):
                for op, code in (
                    ("set_mode(position)", self.arm.set_mode(POSITION_MODE)),
                    ("set_state(stop)", self.arm.set_state(STOP_STATE)),
                ):
                    if code != 0:
                        logging.warning("xArm %s during close returned code %s", op, code)
        finally:
            self.arm.disconnect()

    def _ensure_armed(self) -> None:
        if self._armed:
            return
        self._check(self.arm.motion_enable(enable=True), "motion_enable")
        if self.cfg.gripper:
            self._check(self.arm.clean_gripper_error(), "clean_gripper_error")
            self._check(self.arm.set_gripper_enable(True), "set_gripper_enable")
            self._check(self.arm.set_gripper_mode(0), "set_gripper_mode")
            self._check(self.arm.set_gripper_speed(GRIPPER_SPEED), "set_gripper_speed")
        self._check(self.arm.set_mode(SERVO_MODE), "set_mode(servo)")
        self._check(self.arm.set_state(READY_STATE), "set_state(ready)")
        time.sleep(0.1)
        self._last = self.joint_pos()
        self._last_time = time.monotonic()
        self._armed = True

    def _command_gripper(self, normalized: float) -> None:
        if self._last_gripper is not None and abs(normalized - self._last_gripper) < GRIPPER_EPSILON:
            return
        raw = round(GRIPPER_RAW_CLOSED + normalized * (GRIPPER_RAW_OPEN - GRIPPER_RAW_CLOSED))
        self._check(self.arm.set_gripper_position(raw, wait=False), "set_gripper_position")
        self._last_gripper = normalized

    def _assert_ready(self) -> None:
        if self._stopped or not getattr(self.arm, "connected", False):
            raise ConnectionError(f"xArm at {self.cfg.ip} is disconnected")
        error_code = int(getattr(self.arm, "error_code", 0))
        if error_code != 0:
            raise RuntimeError(f"xArm controller error_code={error_code}")
        state = int(getattr(self.arm, "state", READY_STATE))
        if state >= STOP_STATE:
            raise RuntimeError(f"xArm is not ready for motion (state={state})")

    @staticmethod
    def _check(code: int, op: str) -> None:
        if code != 0:
            raise RuntimeError(f"xArm SDK call {op} failed with code {code}")


@dataclass
class MainConfig(XArmDriverConfig):
    """Standalone hardware test; the default is a read-only state echo (no motion enable)."""

    wiggle: bool = False  # stream a small all-joints sinusoid through step()
    test_gripper: bool = False  # one gripper close/open cycle
    seconds: float = 5.0  # wiggle duration
    hz: float = 30.0  # wiggle step() rate
    amp_deg: float = 2.0  # wiggle amplitude
    period: float = 2.0  # wiggle period, seconds


def echo(driver: XArmDriver, cfg: MainConfig) -> None:
    logging.info("read-only echo (no motion enable); ctrl-c to exit")
    while True:
        msg = "joints deg: %s" % np.array2string(np.rad2deg(driver.joint_pos()), precision=1)
        if cfg.gripper:
            code, raw = driver.arm.get_gripper_position()
            msg += f"  gripper raw: {raw if code == 0 else f'err {code}'}"
        logging.info(msg)
        time.sleep(0.2)


def wiggle(driver: XArmDriver, cfg: MainConfig) -> None:
    q0 = driver.joint_pos()
    logging.info("wiggling all joints +-%.1f deg for %.1fs at %.0f Hz", cfg.amp_deg, cfg.seconds, cfg.hz)
    start = time.monotonic()
    while (t := time.monotonic() - start) < cfg.seconds:
        q = q0 + math.radians(cfg.amp_deg) * math.sin(2.0 * math.pi * t / cfg.period)
        driver.step({"q": q, "aperture": None})
        time.sleep(1.0 / cfg.hz)
    driver.step({"q": q0, "aperture": None})


def gripper_cycle(driver: XArmDriver, cfg: MainConfig) -> None:
    q0 = driver.joint_pos()
    for label, aperture in (("close", 0.0), ("open", cfg.gripper_open)):
        logging.info("gripper %s", label)
        driver.step({"q": q0, "aperture": aperture})
        time.sleep(2.0)


def main(cfg: MainConfig) -> None:
    if cfg.test_gripper and not cfg.gripper:
        raise SystemExit("--test-gripper requires the gripper (drop --no-gripper)")
    driver = XArmDriver(cfg)
    try:
        if cfg.wiggle:
            wiggle(driver, cfg)
        if cfg.test_gripper:
            gripper_cycle(driver, cfg)
        if not (cfg.wiggle or cfg.test_gripper):
            echo(driver, cfg)
    except KeyboardInterrupt:
        pass
    finally:
        driver.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(MainConfig))
