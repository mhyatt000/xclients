"""Custom protobuf schema and serializer for gripper samples."""

from __future__ import annotations

import foxglove
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, timestamp_pb2

GRIPPER_PROTO = "xclients/gripper.proto"
GRIPPER_TYPE = "xclients.Gripper"


def _build_descriptor() -> descriptor_pb2.FileDescriptorProto:
    proto = descriptor_pb2.FileDescriptorProto()
    proto.name = GRIPPER_PROTO
    proto.package = "xclients"
    proto.syntax = "proto2"
    proto.dependency.append("google/protobuf/timestamp.proto")

    msg = proto.message_type.add()
    msg.name = "Gripper"

    field = msg.field.add()
    field.name = "timestamp"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REQUIRED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".google.protobuf.Timestamp"

    for number, name in ((2, "rad"), (3, "norm"), (4, "raw")):
        field = msg.field.add()
        field.name = name
        field.number = number
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE

    return proto


def _build_descriptor_set() -> bytes:
    descriptor_set = descriptor_pb2.FileDescriptorSet()
    descriptor_set.file.add().ParseFromString(timestamp_pb2.DESCRIPTOR.serialized_pb)
    descriptor_set.file.add().CopyFrom(_build_descriptor())
    return descriptor_set.SerializeToString()


def _build_message_class():
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(timestamp_pb2.DESCRIPTOR.serialized_pb)
    pool.Add(_build_descriptor())
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(GRIPPER_TYPE))


class Gripper:
    """Serializer for xclients.Gripper protobuf messages."""

    schema = foxglove.Schema(
        name=GRIPPER_TYPE,
        encoding="protobuf",
        data=_build_descriptor_set(),
    )
    message_cls = _build_message_class()

    @staticmethod
    def encode(stamp_ns: int, *, norm: float) -> bytes:
        msg = Gripper.message_cls(
            rad=norm * 0.85,
            norm=norm,
            raw=norm * 850.0,
        )
        msg.timestamp.seconds = int(stamp_ns) // 1_000_000_000
        msg.timestamp.nanos = int(stamp_ns) % 1_000_000_000
        return msg.SerializeToString()

    @staticmethod
    def decode(data: bytes):
        return Gripper.message_cls.FromString(data)
