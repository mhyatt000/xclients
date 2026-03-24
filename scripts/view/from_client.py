from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np
import rerun as rr
import tyro
from webpolicy.client import Client

from xclients.core.run.scene import RerunScene

logging.basicConfig(level=logging.INFO)

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
GRIPPER_JOINT_NAMES = (
    "drive_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)
VECTOR_ACTION_KEYS = ("action", "actions", "vector", "joint_action", "joint_actions", "proprio_single")
ARM_ACTION_KEYS = ("xarm_joints", "arm_joints", "joints", "joint_positions", "qpos", "q", "proprio_joints")
GRIPPER_ACTION_KEYS = ("xarm_gripper", "gripper", "grip", "gripper_position", "proprio_gripper")


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 8000
    host2: str | None = None
    port2: int | None = None
    route: Literal["dataset", "policy"] = "dataset"
    play_chunk: bool = True
    urdf: Path = Path("xarm7_standalone.urdf")
    app_id: str = "client_view"
    cams: list[Path] = field(default_factory=list)
    entity_path_prefix: str = "robot"
    transforms_path: str = "robot/transforms"
    spawn: bool = True
    rrd_path: Path | None = None
    limit: int | None = None
    dt: float = 0.05
    reset: bool = False
    action_key: str | None = None
    gripper_key: str | None = None

    def __post_init__(self) -> None:
        self.urdf = Path(self.urdf).expanduser().resolve()
        self.cams = [Path(path).expanduser().resolve() for path in self.cams]
        if self.rrd_path is not None:
            self.rrd_path = Path(self.rrd_path).expanduser().resolve()


def coerce_action_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim >= 2 and array.shape[-1] in {7, 8}:
        if array.ndim > 2:
            logging.warning("Action field %r has shape %s; flattening leading batch dimensions.", name, array.shape)
        return array.reshape(-1, array.shape[-1])
    return array.reshape(1, -1)


def coerce_gripper_value(value: Any, *, name: str) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if len(array) == 0:
        raise ValueError(f"Gripper field {name!r} is empty.")
    if len(array) > 1:
        logging.warning("Gripper field %r has shape %s; using the first value.", name, np.asarray(value).shape)
    return float(array[0])


def coerce_gripper_values(value: Any, *, name: str, count: int) -> np.ndarray:
    values = np.asarray(value, dtype=float).reshape(-1)
    if len(values) == 0:
        raise ValueError(f"Gripper field {name!r} is empty.")
    if len(values) == 1:
        return np.full(count, float(values[0]), dtype=float)
    if len(values) != count:
        raise ValueError(f"Gripper field {name!r} has {len(values)} values, expected 1 or {count}.")
    return values


def describe_action(action: Any) -> str:
    if isinstance(action, Mapping):
        return ", ".join(
            f"{key}:{np.asarray(value).shape if isinstance(value, np.ndarray | list | tuple) else type(value).__name__}"
            for key, value in action.items()
        )
    return f"{type(action).__name__}:{np.asarray(action).shape}"


def split_actions(
    values: np.ndarray, *, name: str, gripper: Any = None, gripper_name: str | None = None
) -> list[tuple[np.ndarray, float]]:
    if values.shape[1] == 8:
        return [(row[:7], float(row[7])) for row in values]

    if values.shape[1] != 7:
        raise ValueError(f"Action field {name!r} must have 7 or 8 values, got {values.shape[1]}.")

    if gripper is None or gripper_name is None:
        raise KeyError(f"Action field {name!r} has 7 arm joints but no gripper field was found.")

    gripper_values = coerce_gripper_values(gripper, name=gripper_name, count=len(values))
    return [(row, float(grip)) for row, grip in zip(values, gripper_values, strict=True)]


def extract_actions(action: Any, cfg: Config) -> list[tuple[np.ndarray, float]]:
    # Unwrap observation-mode response: {'observation': {...proprio_single, ...images}}
    if isinstance(action, Mapping) and list(action.keys()) == ["observation"]:
        obs = action["observation"]
        logging.debug("Unwrapping 'observation' key; inner keys: %s", list(obs.keys()))
        action = obs

    if not isinstance(action, Mapping):
        values = coerce_action_array(action, name="action")
        if values.shape[1] != 8:
            raise ValueError(f"Expected an 8-value action, got shape {np.asarray(action).shape}.")
        return [(row[:7], float(row[7])) for row in values]

    if cfg.action_key is not None:
        if cfg.action_key not in action:
            raise KeyError(
                f"Configured action key {cfg.action_key!r} is missing from action: {describe_action(action)}"
            )
        values = coerce_action_array(action[cfg.action_key], name=cfg.action_key)
        gripper_key = cfg.gripper_key or next((key for key in GRIPPER_ACTION_KEYS if key in action), None)
        if values.shape[1] == 7 and gripper_key is None:
            raise KeyError(
                f"Action field {cfg.action_key!r} has 7 arm joints but no gripper field was found in: {describe_action(action)}"
            )
        return split_actions(
            values,
            name=cfg.action_key,
            gripper=action.get(gripper_key) if gripper_key is not None else None,
            gripper_name=gripper_key,
        )

    if all(name in action for name in ARM_JOINT_NAMES):
        arm = np.column_stack([np.asarray(action[name], dtype=float).reshape(-1) for name in ARM_JOINT_NAMES])
        gripper_key = cfg.gripper_key or next((key for key in GRIPPER_ACTION_KEYS if key in action), None)
        if gripper_key is None:
            raise KeyError(f"Per-joint action is missing a gripper field: {describe_action(action)}")
        gripper_values = coerce_gripper_values(action[gripper_key], name=gripper_key, count=len(arm))
        return [(row, float(grip)) for row, grip in zip(arm, gripper_values, strict=True)]

    for key in (*VECTOR_ACTION_KEYS, *ARM_ACTION_KEYS):
        if key not in action:
            continue
        values = coerce_action_array(action[key], name=key)
        gripper_key = cfg.gripper_key or next((name for name in GRIPPER_ACTION_KEYS if name in action), None)
        if values.shape[1] == 7 and gripper_key is None:
            raise KeyError(f"Action field {key!r} has 7 arm joints but no gripper field: {describe_action(action)}")
        return split_actions(
            values,
            name=key,
            gripper=action.get(gripper_key) if gripper_key is not None else None,
            gripper_name=gripper_key,
        )

    raise ValueError(f"Could not infer a 7+1 xarm action from client output: {describe_action(action)}")


def gripper_joint_values(scene: RerunScene, gripper: float) -> dict[str, float]:
    if (gripper_joint := scene.joint_map.get("drive_joint")) is None:
        return {}

    value = float(gripper)
    if np.isfinite(gripper_joint.limit_lower):
        value = max(value, gripper_joint.limit_lower)
    if np.isfinite(gripper_joint.limit_upper):
        value = min(value, gripper_joint.limit_upper)

    return {name: value for name in GRIPPER_JOINT_NAMES if name in scene.joint_map}


def connect_client(host: str, port: int, *, name: str) -> Client:
    client = Client(host, port)
    metadata = client.get_server_metadata()
    logging.info("Connected to %s webpolicy server at %s:%d", name, host, port)
    if metadata:
        logging.info("%s server metadata: %s", name.capitalize(), metadata)
    return client


def get_action(dataset_client: Client, cfg: Config, policy_client: Client | None = None) -> Any:
    if cfg.route == "dataset":
        return dataset_client.step({})

    batch = dataset_client.step({})
    batch["ensemble"] = True
    if policy_client is None:
        raise ValueError("Policy route requires a second client.")
    return policy_client.step(batch)


def main(cfg: Config) -> None:
    client = connect_client(cfg.host, cfg.port, name="dataset")
    policy_client: Client | None = None
    if cfg.route == "policy":
        host2 = cfg.host2 or cfg.host
        port2 = cfg.port2 or cfg.port
        policy_client = connect_client(host2, port2, name="policy")
    if cfg.reset:
        client.reset()
        if policy_client is not None:
            policy_client.reset()

    scene = RerunScene(
        cfg.urdf,
        app_id=cfg.app_id,
        camera_ht_files=cfg.cams,
        entity_path_prefix=cfg.entity_path_prefix,
        transforms_path=cfg.transforms_path,
        spawn=cfg.spawn,
        rrd_path=cfg.rrd_path,
    )

    print("\n=== Config ===")
    print(f"  route: {cfg.route}")
    print(f"  host: {cfg.host}:{cfg.port}")
    print(f"  play_chunk: {cfg.play_chunk}")
    print(f"  dt: {cfg.dt}s")
    print(f"  limit: {cfg.limit}")
    print(f"  action_key: {cfg.action_key!r}  gripper_key: {cfg.gripper_key!r}")
    print(f"  URDF joints found: {scene.joint_names}")
    print()

    step = 0
    start = time.monotonic()
    while cfg.limit is None or step < cfg.limit:
        action = get_action(client, cfg, policy_client)

        # --- diagnostics: print first 3 raw responses ---
        if step < 3:
            print(f"[step {step}] raw action type: {type(action).__name__}")
            if isinstance(action, dict):
                for k, v in action.items():
                    arr = np.asarray(v) if isinstance(v, (list, tuple, np.ndarray)) else None
                    if arr is not None:
                        print(
                            f"  key={k!r}  shape={arr.shape}  dtype={arr.dtype}  first={arr.flat[0] if arr.size else 'empty'}"
                        )
                    else:
                        print(f"  key={k!r}  value={v!r}")
            else:
                arr = np.asarray(action)
                print(f"  shape={arr.shape}  dtype={arr.dtype}")

        actions = extract_actions(action, cfg)

        if step < 3:
            print(f"  -> extracted {len(actions)} (arm, gripper) pairs")
            for i, (arm, grip) in enumerate(actions[:3]):
                print(f"     [{i}] arm={np.round(arm, 4).tolist()}  gripper={grip:.4f}")

        if not cfg.play_chunk:
            actions = actions[:1]
        for arm_values, gripper in actions:
            if cfg.limit is not None and step >= cfg.limit:
                break
            if len(arm_values) != len(ARM_JOINT_NAMES):
                raise ValueError(f"Expected {len(ARM_JOINT_NAMES)} arm joints, got {len(arm_values)}")

            rr.set_time("time", duration=time.monotonic() - start)

            joint_values = {
                name: float(value)
                for name, value in zip(ARM_JOINT_NAMES, arm_values, strict=True)
                if name in scene.joint_map
            }
            joint_values.update(gripper_joint_values(scene, gripper))

            if step < 3:
                print(f"  -> logging joint_values: { {k: round(v, 4) for k, v in joint_values.items()} }")

            scene.log_joints(joint_values, step=step)
            step += 1

            if step % 100 == 0:
                elapsed = time.monotonic() - start
                print(f"[step {step}]  elapsed={elapsed:.1f}s  rate={step / elapsed:.1f} steps/s")

            if cfg.dt > 0.0:
                time.sleep(cfg.dt)


if __name__ == "__main__":
    main(tyro.cli(Config))
