"""Convert bela zarr episodes to Foxglove protobuf MCAP files."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path

import cv2
import foxglove
from foxglove.channels import (
    JointStatesChannel,
    PoseChannel,
    RawImageChannel,
)
from foxglove.messages import (
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

from xclients.messages import Gripper

RAW_IMAGE_ENCODING = "yuv422_yuy2"
COLOR_RAW_ENCODING = "rgb8"


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


@dataclass(frozen=True, order=True)
class WriteEvent:
    stamp_ns: int
    topic: str
    sample_idx: int


@dataclass(frozen=True)
class TopicWriter:
    topic: str
    group: zarr.Group
    count: int
    write_sample: Callable[[int], None]


def prepare_raw_image(topic: str, group: zarr.Group) -> TopicWriter:
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

    def write_sample(idx: int) -> None:
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

    return TopicWriter(topic, group, count, write_sample)


def bytes_from_zarr(value: object) -> bytes:
    while isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value
    if isinstance(value, np.bytes_):
        return bytes(value)
    raise TypeError(f"expected bytes-like zarr value, got {type(value)!r}")


def raw_image_from_bytes(payload: bytes, format_name: str) -> tuple[np.ndarray, str]:
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError(
            f"could not decode compressed image payload with format {format_name!r}"
        )

    if decoded.ndim == 2:
        return np.ascontiguousarray(decoded), "mono8"
    if decoded.ndim == 3 and decoded.shape[2] == 3:
        frame = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(frame), COLOR_RAW_ENCODING
    if decoded.ndim == 3 and decoded.shape[2] == 4:
        frame = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
        return np.ascontiguousarray(frame), "rgba8"
    raise ValueError(f"unsupported decoded image shape {decoded.shape} for {format_name!r}")


def prepare_compressed_image(topic: str, group: zarr.Group) -> TopicWriter:
    count = complete_len(group, ("data", "format", "stamp_ns"))
    warn_short_topic(topic, group["stamp_ns"].shape[0], count)
    channel = RawImageChannel(topic)

    def write_sample(idx: int) -> None:
        stamp_ns = int(group["stamp_ns"][idx])
        frame, encoding = raw_image_from_bytes(
            bytes_from_zarr(group["data"][idx]),
            str(group["format"][idx]),
        )
        height, width = frame.shape[:2]
        channel.log(
            RawImage(
                timestamp=stamp_from_ns(stamp_ns),
                frame_id="",
                width=width,
                height=height,
                encoding=encoding,
                step=int(frame.strides[0]),
                data=frame.tobytes(),
            ),
            log_time=stamp_ns,
        )

    return TopicWriter(topic, group, count, write_sample)


def parse_joint_names(value: object) -> list[str]:
    raw = str(value)
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not all(isinstance(name, str) for name in parsed):
        raise ValueError(f"joint names must be a JSON string list, got {raw!r}")
    return parsed


def prepare_joint_states(topic: str, group: zarr.Group) -> TopicWriter:
    keys = ("name_json", "position", "velocity", "effort", "stamp_ns")
    count = complete_len(group, keys)
    warn_short_topic(topic, group["stamp_ns"].shape[0], count)
    channel = JointStatesChannel(topic)

    def write_sample(idx: int) -> None:
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

    return TopicWriter(topic, group, count, write_sample)


def prepare_gripper(topic: str, group: zarr.Group) -> TopicWriter:
    count = complete_len(group, ("json", "stamp_ns"))
    warn_short_topic(topic, group["stamp_ns"].shape[0], count)
    channel = foxglove.Channel(
        topic,
        schema=Gripper.schema,
        message_encoding="protobuf",
    )

    def write_sample(idx: int) -> None:
        stamp_ns = int(group["stamp_ns"][idx])
        payload = json.loads(str(group["json"][idx]))
        values = payload.get("data")
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(f"{topic} expected one gripper value, got {payload!r}")
        channel.log(Gripper.encode(stamp_ns, norm=float(values[0])), log_time=stamp_ns)

    return TopicWriter(topic, group, count, write_sample)


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


def prepare_robot_poses(topic: str, group: zarr.Group) -> TopicWriter:
    count = complete_len(group, ("json", "stamp_ns"))
    warn_short_topic(topic, group["stamp_ns"].shape[0], count)
    channel = PoseChannel(topic)

    def write_sample(idx: int) -> None:
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

    return TopicWriter(topic, group, count, write_sample)


def prepare_topic_writer(topic: str, group: zarr.Group) -> TopicWriter:
    mode = str(group.attrs.get("mode", ""))
    if mode == "image":
        return prepare_raw_image(topic, group)
    if mode == "compressed_image":
        return prepare_compressed_image(topic, group)
    if mode == "joint_state":
        return prepare_joint_states(topic, group)
    if mode == "json":
        if topic == "/xgym/gripper":
            return prepare_gripper(topic, group)
        if topic == "/xarm/robot_states":
            return prepare_robot_poses(topic, group)
        raise ValueError(f"{topic} has unsupported json payload")
    raise ValueError(f"{topic} has unsupported mode {mode!r}")


def write_events(writers: dict[str, TopicWriter]) -> None:
    events = [
        WriteEvent(int(writer.group["stamp_ns"][idx]), topic, idx)
        for topic, writer in writers.items()
        for idx in range(writer.count)
    ]
    events.sort()

    for event in tqdm(events, desc="messages", unit="msg"):
        writers[event.topic].write_sample(event.sample_idx)

def convert_episode(episode: Path, output: Path, overwrite: bool) -> None:
    root = zarr.open_group(episode, mode="r")

    with foxglove.open_mcap(str(output), allow_overwrite=overwrite):
        writers = {
            topic: prepare_topic_writer(topic, group)
            for topic, group in topic_groups(root)
        }
        write_events(writers)

    print(f"wrote {output}")
    for topic, writer in writers.items():
        print(f"  {topic}: {writer.count} messages")


def is_episode_dir(path: Path) -> bool:
    return path.is_dir() and (path / "zarr.json").is_file() and (path / "topics").is_dir()


def episode_dirs(input_dir: Path) -> list[Path]:
    if is_episode_dir(input_dir):
        return [input_dir]

    episodes = sorted(path for path in input_dir.iterdir() if is_episode_dir(path))
    if not episodes:
        raise ValueError(f"no zarr episode directories found under {input_dir}")
    return episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing zarr episodes, or one zarr episode directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output MCAP files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = episode_dirs(input_dir)
    print(f"found {len(episodes)} episode(s) under {input_dir}")
    for episode in episodes:
        output = output_dir / f"{episode.name}.mcap"
        convert_episode(episode, output, args.overwrite)


if __name__ == "__main__":
    main()
