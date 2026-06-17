"""Convert one bela zarr episode to a Foxglove protobuf MCAP."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import math
from pathlib import Path
import struct

import foxglove
from foxglove.channels import (
    CompressedImageChannel,
    JointStatesChannel,
    PoseChannel,
    RawImageChannel,
)
from foxglove.messages import (
    CompressedImage,
    JointState,
    JointStates,
    Pose,
    Quaternion,
    RawImage,
    Timestamp,
    Vector3,
)
import numpy as np
from tqdm import tqdm
import zarr

DEFAULT_EPISODE = Path(
    "~/bela_zarr/single/lift/2026-05-25_1932_episode_000001"
).expanduser()
RAW_IMAGE_ENCODING = "yuv422_yuy2"
FLOAT64_SCHEMA = foxglove.Schema(
    name="foxglove.Float64",
    encoding="protobuf",
    data=b"\x0a\x45\n\x18foxglove/Float64.proto\x12\x08foxglove\"\x1f\n\x07Float64\x12\x14\n\x05value\x18\x01 \x01(\x01R\x05value",
)


def stamp_from_ns(stamp_ns: int) -> Timestamp:
    sec, nsec = divmod(int(stamp_ns), 1_000_000_000)
    return Timestamp(sec, nsec)


def topic_groups(root: zarr.Group) -> list[tuple[str, zarr.Group]]:
    groups = []
    for name, group in root["topics"].groups():
        attrs = dict(group.attrs)
        topic = attrs.get("topic")
        if topic is None:
            raise ValueError(f"topic group {name!r} is missing a topic attribute")
        groups.append((str(topic), group))
    return sorted(groups)


def complete_len(group: zarr.Group, keys: Sequence[str]) -> int:
    return min(group[key].shape[0] for key in keys)


def warn_short_topic(topic: str, expected: int, actual: int) -> None:
    if actual < expected:
        skipped = expected - actual
        print(f"warning: {topic} has {actual}/{expected} complete samples; skipped {skipped}")


def write_raw_image(topic: str, group: zarr.Group) -> int:
    data = group["data"]
    stamps = group["stamp_ns"]
    count = complete_len(group, ("data", "stamp_ns"))
    warn_short_topic(topic, stamps.shape[0], count)

    if len(data.shape) != 4 or data.shape[-1] != 2:
        raise ValueError(f"{topic} expected shape (N, H, W, 2), got {data.shape}")

    height = int(data.shape[1])
    width = int(data.shape[2])
    step = width * 2
    channel = RawImageChannel(topic)

    for idx in tqdm(range(count), desc=topic, unit="msg"):
        stamp_ns = int(stamps[idx])
        frame = np.ascontiguousarray(data[idx])
        channel.log(
            RawImage(
                timestamp=stamp_from_ns(stamp_ns),
                frame_id="",
                width=width,
                height=height,
                encoding=RAW_IMAGE_ENCODING,
                step=step,
                data=frame.tobytes(),
            ),
            log_time=stamp_ns,
        )
    return count


def write_compressed_image(topic: str, group: zarr.Group) -> int:
    count = complete_len(group, ("data", "format", "stamp_ns"))
    channel = CompressedImageChannel(topic)

    for idx in tqdm(range(count), desc=topic, unit="msg"):
        stamp_ns = int(group["stamp_ns"][idx])
        channel.log(
            CompressedImage(
                timestamp=stamp_from_ns(stamp_ns),
                frame_id="",
                data=bytes(group["data"][idx]),
                format=str(group["format"][idx]),
            ),
            log_time=stamp_ns,
        )
    return count


def parse_joint_names(value: object) -> list[str]:
    raw = str(value)
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not all(isinstance(name, str) for name in parsed):
        raise ValueError(f"joint names must be a JSON string list, got {raw!r}")
    return parsed


def write_joint_states(topic: str, group: zarr.Group) -> int:
    keys = ("name_json", "position", "velocity", "effort", "stamp_ns")
    count = complete_len(group, keys)
    warn_short_topic(topic, group["stamp_ns"].shape[0], count)
    channel = JointStatesChannel(topic)

    for idx in tqdm(range(count), desc=topic, unit="msg"):
        stamp_ns = int(group["stamp_ns"][idx])
        names = parse_joint_names(group["name_json"][idx])
        positions = group["position"][idx].tolist()
        velocities = group["velocity"][idx].tolist()
        efforts = group["effort"][idx].tolist()
        joints = [
            JointState(
                name=name,
                position=float(position),
                velocity=float(velocity),
                effort=float(effort),
            )
            for name, position, velocity, effort in zip(
                names, positions, velocities, efforts, strict=True
            )
        ]
        channel.log(
            JointStates(timestamp=stamp_from_ns(stamp_ns), joints=joints),
            log_time=stamp_ns,
        )
    return count


def encode_float64(value: float) -> bytes:
    return b"\x09" + struct.pack("<d", value)


def write_gripper(topic: str, group: zarr.Group) -> int:
    count = complete_len(group, ("json", "stamp_ns"))
    channel = foxglove.Channel(
        topic,
        schema=FLOAT64_SCHEMA,
        message_encoding="protobuf",
    )

    for idx in tqdm(range(count), desc=topic, unit="msg"):
        stamp_ns = int(group["stamp_ns"][idx])
        payload = json.loads(str(group["json"][idx]))
        values = payload.get("data")
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(f"{topic} expected one gripper value, got {payload!r}")
        channel.log(encode_float64(float(values[0])), log_time=stamp_ns)
    return count


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return Quaternion(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy,
    )


def write_robot_poses(topic: str, group: zarr.Group) -> int:
    count = complete_len(group, ("json", "stamp_ns"))
    channel = PoseChannel(topic)

    for idx in tqdm(range(count), desc=topic, unit="msg"):
        stamp_ns = int(group["stamp_ns"][idx])
        payload = json.loads(str(group["json"][idx]))
        values = payload.get("pose")
        if not isinstance(values, list) or len(values) != 6:
            raise ValueError(
                f"{topic} expected pose [x, y, z, roll, pitch, yaw], got {payload!r}"
            )
        x, y, z, roll, pitch, yaw = [float(value) for value in values]
        channel.log(
            Pose(
                position=Vector3(x=x, y=y, z=z),
                orientation=rpy_to_quaternion(roll, pitch, yaw),
            ),
            log_time=stamp_ns,
        )
    return count


def convert_episode(episode: Path, output: Path, overwrite: bool) -> None:
    root = zarr.open_group(episode, mode="r")
    counts: dict[str, int] = {}

    with foxglove.open_mcap(str(output), allow_overwrite=overwrite):
        for topic, group in topic_groups(root):
            mode = str(group.attrs.get("mode", ""))
            if mode == "image":
                counts[topic] = write_raw_image(topic, group)
            elif mode == "compressed_image":
                counts[topic] = write_compressed_image(topic, group)
            elif mode == "joint_state":
                counts[topic] = write_joint_states(topic, group)
            elif mode == "json":
                if topic == "/xgym/gripper":
                    counts[topic] = write_gripper(topic, group)
                elif topic == "/xarm/robot_states":
                    counts[topic] = write_robot_poses(topic, group)
                else:
                    raise ValueError(f"{topic} has unsupported json payload")
            else:
                raise ValueError(f"{topic} has unsupported mode {mode!r}")

    print(f"wrote {output}")
    for topic, count in counts.items():
        print(f"  {topic}: {count} messages")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/Downloads"),
        help="Directory for the output MCAP. Defaults to ~/Downloads.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode = args.episode.expanduser()
    output = args.output_dir.expanduser() / f"{episode.name}.mcap"
    convert_episode(episode, output, args.overwrite)


if __name__ == "__main__":
    main()
