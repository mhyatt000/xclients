from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import logging
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
from webpolicy.client import Client

from xclients.dream_dr.config import Config, Record


def safe_camera_id(camera: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", camera).strip("_")
    return safe or "camera"


@dataclass
class DreamCameraBatch:
    camera: str
    records: list[Record]
    masks: np.ndarray
    initial: dict[str, np.ndarray]


@dataclass
class ArrayRecordShard:
    index: int
    shard_id: str
    path: Path
    image_src: Any
    proprio_src: Any | None
    unpack_record: Any
    length: int

    def read_sample(self, local_index: int) -> dict:
        if self.proprio_src is None:
            return self.unpack_record(self.image_src[int(local_index)])
        image_record = self.unpack_record(self.image_src[int(local_index)])
        proprio_record = self.unpack_record(self.proprio_src[int(local_index)])
        return merge_image_proprio_records(image_record, proprio_record)

    def read_info_sample(self, local_index: int) -> dict:
        if self.proprio_src is None:
            return self.read_sample(local_index)
        proprio_record = self.unpack_record(self.proprio_src[int(local_index)])
        if "info" in proprio_record:
            return {"info": proprio_record["info"]}
        return proprio_record


def collect_dream_arrayrecord(cfg: Config) -> dict[str, np.ndarray]:
    if cfg.arrayrecord_dir is None:
        raise ValueError("arrayrecord_dir is required")

    cache = cfg.output_dir / "dream_arrayrecord_outputs.npz"
    if cache.exists() and not cfg.refresh_cache:
        return load_npz_dict(cache)

    payload: dict[str, Any] = {
        "type": "arrayrecord",
        "arrayrecord_dir": str(cfg.arrayrecord_dir),
        "path": str(cfg.arrayrecord_dir),
        "calibrate": True,
    }
    if cfg.arrayrecord_cameras:
        payload["cameras"] = list(cfg.arrayrecord_cameras)

    logging.info("Calling DREAM at %s:%s for arrayrecord %s", cfg.dream_host, cfg.dream_port, cfg.arrayrecord_dir)
    client = Client(cfg.dream_host, cfg.dream_port)
    try:
        out = client.step(payload)
    except RuntimeError as exc:
        if "KeyError: 'image'" not in str(exc):
            raise
        logging.warning("DREAM server does not accept arrayrecord payloads; falling back to local ArrayRecord loading")
        arrays = collect_dream_arrayrecord_compat(cfg, client)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **arrays)
        logging.info("Cached DREAM arrayrecord output to %s", cache)
        return arrays
    arrays = dream_response_to_arrays(out)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **arrays)
    logging.info("Cached DREAM arrayrecord output to %s", cache)
    return arrays


def iter_arrayrecord_shards(cfg: Config) -> list[ArrayRecordShard]:
    if cfg.arrayrecord_dir is None:
        raise ValueError("arrayrecord_dir is required")
    add_crossformer_path()
    from array_record.python.array_record_data_source import ArrayRecordDataSource
    from crossformer.data.arec.arec import unpack_record

    root = resolve_arrayrecord_branch(cfg.arrayrecord_dir)
    image_shards = sorted((root / "image").glob("*.arrayrecord"))
    proprio_shards = sorted((root / "proprio").glob("*.arrayrecord"))
    if image_shards and proprio_shards:
        if len(image_shards) != len(proprio_shards):
            raise ValueError(f"ArrayRecord image/proprio shard count mismatch: {len(image_shards)} != {len(proprio_shards)}")
        shards = []
        for index, (image_path, proprio_path) in enumerate(zip(image_shards, proprio_shards, strict=True)):
            image_src = ArrayRecordDataSource([str(image_path)])
            proprio_src = ArrayRecordDataSource([str(proprio_path)])
            if len(image_src) != len(proprio_src):
                raise ValueError(f"ArrayRecord shard length mismatch for {image_path.name}: {len(image_src)} != {len(proprio_src)}")
            shards.append(
                ArrayRecordShard(
                    index=index,
                    shard_id=image_path.stem,
                    path=image_path,
                    image_src=image_src,
                    proprio_src=proprio_src,
                    unpack_record=unpack_record,
                    length=len(image_src),
                )
            )
        return shards

    data_shards = sorted((root / "data").glob("*.arrayrecord")) or sorted(root.glob("*.arrayrecord"))
    if not data_shards:
        raise FileNotFoundError(f"No ArrayRecord shards found under {root}")
    shards = []
    for index, path in enumerate(data_shards):
        src = ArrayRecordDataSource([str(path)])
        shards.append(
            ArrayRecordShard(
                index=index,
                shard_id=path.stem,
                path=path,
                image_src=src,
                proprio_src=None,
                unpack_record=unpack_record,
                length=len(src),
            )
        )
    return shards


def selected_shard_indices(cfg: Config, shard: ArrayRecordShard, global_start: int) -> np.ndarray:
    global_indices = np.arange(global_start, global_start + shard.length, dtype=np.int64)
    if cfg.data_select != [-1]:
        selected = np.asarray(cfg.data_select, dtype=np.int64)
        keep = selected[(selected >= global_start) & (selected < global_start + shard.length)]
        local = keep - global_start
    else:
        local = np.arange(shard.length, dtype=np.int64)
    if cfg.max_records is not None:
        local = local[global_indices[local] < cfg.max_records]
    return local.astype(np.int64)


def discover_shard_cameras(cfg: Config, shard: ArrayRecordShard, local_indices: np.ndarray) -> list[str]:
    if cfg.arrayrecord_cameras:
        return list(cfg.arrayrecord_cameras)
    cameras: list[str] = []
    seen = set()
    for local_index in local_indices:
        for camera in camera_names(shard.read_info_sample(int(local_index))):
            if camera not in seen:
                seen.add(camera)
                cameras.append(camera)
    if not cameras and len(local_indices):
        cameras = camera_names(shard.read_sample(int(local_indices[0])))
    return cameras


def camera_available(sample: dict, camera: str) -> bool:
    try:
        resolve_camera_name(camera, camera_names(sample))
    except KeyError:
        return False
    return True


def collect_dream_arrayrecord_compat(cfg: Config, client: Client) -> dict[str, np.ndarray]:
    records = load_arrayrecord_samples(cfg)
    cameras = list(cfg.arrayrecord_cameras) or camera_names(records[0])
    if not cameras:
        cameras = ["camera"]

    all_images = []
    all_masks = []
    all_initial = []
    all_intrinsics = []
    all_stems = []
    joints = np.stack([sample_joints_deg(record) for record in records]).astype(np.float32)

    for camera in cameras:
        images = np.stack([sample_image(record, camera) for record in records]).astype(np.uint8)
        h, w = images.shape[1:3]
        intrinsics = np.repeat(default_intrinsics(h, w, cfg.arrayrecord_focal_px)[None], len(records), axis=0)
        logging.info("Calling DREAM for camera %s with %d ArrayRecord frames", camera, len(records))
        out = client.step(
            {
                "image": images,
                "type": "image",
                "q": joints,
                "K": intrinsics.astype(np.float32),
                "calibrate": True,
            }
        )
        dream = dream_response_to_arrays(out)
        if "mask" not in dream:
            raise KeyError(
                "DREAM response did not include mask. Start the DREAM server with mask returns enabled "
                "or use a checkpoint/config with a mask decoder."
            )
        initial = _required(dream, "calib_w2c", "w2c", "HT", "extrinsics")
        all_images.append(images)
        all_masks.append(np.asarray(dream["mask"], dtype=np.uint8))
        all_initial.append(np.asarray(initial, dtype=np.float32))
        all_intrinsics.append(intrinsics.astype(np.float32))
        all_stems.append(np.asarray([f"{camera}_{i:06d}" for i in range(len(records))]))

    return {
        "images": np.stack(all_images, axis=1),
        "masks": np.stack(all_masks, axis=1),
        "q": joints,
        "K": np.stack(all_intrinsics, axis=1),
        "calib_w2c": np.stack(all_initial, axis=0),
        "cameras": np.asarray(cameras),
        "stems": np.stack(all_stems, axis=1),
    }


def load_arrayrecord_samples(cfg: Config) -> list[dict]:
    if cfg.arrayrecord_dir is None:
        raise ValueError("arrayrecord_dir is required")
    add_crossformer_path()
    from array_record.python.array_record_data_source import ArrayRecordDataSource
    from crossformer.data.arec.arec import unpack_record

    root = resolve_arrayrecord_branch(cfg.arrayrecord_dir)
    image_shards = sorted((root / "image").glob("*.arrayrecord"))
    proprio_shards = sorted((root / "proprio").glob("*.arrayrecord"))
    if image_shards and proprio_shards:
        image_src = ArrayRecordDataSource([str(path) for path in image_shards])
        proprio_src = ArrayRecordDataSource([str(path) for path in proprio_shards])
        if len(image_src) != len(proprio_src):
            raise ValueError(f"ArrayRecord image/proprio length mismatch: {len(image_src)} != {len(proprio_src)}")
        n = len(image_src)
        indices = selected_indices(cfg, n)
        samples = []
        for index in indices:
            image_record = unpack_record(image_src[int(index)])
            proprio_record = unpack_record(proprio_src[int(index)])
            samples.append(merge_image_proprio_records(image_record, proprio_record))
        return samples

    data_shards = sorted((root / "data").glob("*.arrayrecord")) or sorted(root.glob("*.arrayrecord"))
    if not data_shards:
        raise FileNotFoundError(f"No ArrayRecord shards found under {root}")
    src = ArrayRecordDataSource([str(path) for path in data_shards])
    indices = selected_indices(cfg, len(src))
    return [unpack_record(src[int(index)]) for index in indices]


def merge_image_proprio_records(image_record: dict, proprio_record: dict) -> dict:
    out = dict(image_record)
    out["proprio"] = proprio_record.get("proprio", proprio_record)
    if "info" not in out and isinstance(proprio_record.get("info"), dict):
        out["info"] = proprio_record["info"]
    return out


def add_crossformer_path() -> None:
    path = Path("~/crossformer").expanduser()
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
    site_packages = path / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if site_packages.exists() and str(site_packages) not in sys.path:
        sys.path.append(str(site_packages))


def resolve_arrayrecord_branch(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "image").exists() or (path / "data").exists():
        return path
    main = path / "main"
    if (main / "image").exists() or (main / "data").exists():
        return main
    return path


def selected_indices(cfg: Config, n: int) -> np.ndarray:
    indices = np.arange(n, dtype=np.int64)
    if cfg.data_select != [-1]:
        indices = np.asarray(cfg.data_select, dtype=np.int64)
    if cfg.max_records is not None:
        indices = indices[: cfg.max_records]
    if len(indices) == 0:
        raise ValueError("ArrayRecord selection is empty")
    if np.any(indices < 0) or np.any(indices >= n):
        raise IndexError(f"ArrayRecord selection must be in [0, {n - 1}], got {indices.tolist()}")
    return indices


def camera_names(sample: dict) -> list[str]:
    image = sample.get("image")
    if isinstance(image, dict):
        return list(image)
    keys = sample.get("info", {}).get("image_keys", None)
    if keys is None:
        return []
    return [_decode_name(key) for key in keys]


def sample_image(sample: dict, camera: str) -> np.ndarray:
    image = sample["image"]
    if isinstance(image, dict):
        key = resolve_camera_name(camera, list(image))
        return np.asarray(image[key], dtype=np.uint8)

    arr = np.asarray(image)
    if arr.ndim == 4:
        keys = camera_names(sample)
        key = resolve_camera_name(camera, keys)
        if key in keys:
            return np.asarray(arr[keys.index(key)], dtype=np.uint8)
        raise KeyError(f"Camera {camera!r} not in image_keys {keys}")
    return np.asarray(arr, dtype=np.uint8)


def resolve_camera_name(camera: str, available: list[str]) -> str:
    if camera in available:
        return camera
    patterns = (
        f".{camera}.",
        f"/{camera}/",
        f"_{camera}_",
        f"{camera}.",
        f"{camera}/",
    )
    matches = [name for name in available if any(pattern in name for pattern in patterns)]
    if len(matches) == 1:
        return matches[0]
    suffix_matches = [name for name in available if name.endswith(camera)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    raise KeyError(f"Camera {camera!r} not found. Available cameras: {available}")


def sample_joints_deg(sample: dict) -> np.ndarray:
    proprio = sample.get("proprio", sample)
    joints = np.asarray(proprio["joints"], dtype=np.float32).reshape(-1)[:7]
    gripper = np.asarray(proprio.get("gripper", [0.0]), dtype=np.float32).reshape(-1)[:1]
    return np.concatenate([np.rad2deg(joints), gripper], axis=0).astype(np.float32)


def sample_joints_rad(sample: dict, include_gripper: bool = False) -> np.ndarray:
    proprio = sample.get("proprio", sample)
    joints = np.asarray(proprio["joints"], dtype=np.float32).reshape(-1)[:7]
    if not include_gripper:
        return joints

    drive = sample_gripper_drive_joint(sample)
    gripper = np.full(6, drive, dtype=np.float32)
    return np.concatenate([joints, gripper], axis=0).astype(np.float32)


def sample_gripper_drive_joint(sample: dict) -> np.float32:
    proprio = sample.get("proprio", sample)
    raw = float(np.asarray(proprio.get("gripper", [1.0]), dtype=np.float32).reshape(-1)[0])
    if 0.0 <= raw <= 1.0:
        return np.float32(np.clip(1.0 - raw, 0.0, 1.0) * 0.85)
    return np.float32(np.clip(raw, 0.0, 0.85))


def default_intrinsics(h: int, w: int, focal_px: float) -> np.ndarray:
    return np.array(
        [
            [focal_px, 0.0, w / 2.0],
            [0.0, focal_px, h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def dream_response_to_arrays(out: Any) -> dict[str, np.ndarray]:
    if isinstance(out, (str, Path)):
        return load_npz_dict(Path(out).expanduser().resolve())
    if isinstance(out, bytes):
        return load_npz_bytes(out)
    if not isinstance(out, dict):
        raise TypeError(f"Expected DREAM response dict/path/bytes, got {type(out).__name__}")

    for key in ("npz_path", "output_npz", "path"):
        value = out.get(key)
        if isinstance(value, (str, Path)):
            return load_npz_dict(Path(value).expanduser().resolve())

    for key in ("npz", "npz_bytes", "output"):
        value = out.get(key)
        if isinstance(value, bytes):
            return load_npz_bytes(value)

    return {key: np.asarray(value) for key, value in out.items() if _is_array_like(value)}


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_npz_bytes(blob: bytes) -> dict[str, np.ndarray]:
    with np.load(BytesIO(blob), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _is_array_like(value: Any) -> bool:
    if value is None or isinstance(value, (str, bytes, Path)):
        return False
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    return arr.dtype != object


def batches_from_dream_arrays(cfg: Config, arrays: dict[str, np.ndarray]) -> list[DreamCameraBatch]:
    cameras = _camera_names(cfg, arrays)
    images = _required(arrays, "image", "images")
    joints = _required(arrays, "q", "joints", "joint_states")
    masks = _required(arrays, "mask", "masks")
    intrinsics = _required(arrays, "K", "intrinsics", "camera_matrix")
    initial = _required(arrays, "calib_w2c", "w2c", "HT", "extrinsics")
    stems = _optional(arrays, "stem", "stems", "record", "records")

    batches = []
    for cam_index, camera in enumerate(cameras):
        cam_images = _select_camera_axis(images, cam_index, len(cameras), "image")
        cam_masks = _select_camera_axis(masks, cam_index, len(cameras), "mask")
        cam_intr = _select_intrinsics(intrinsics, cam_index, len(cameras), len(cam_images))
        cam_initial = _select_initial(initial, cam_index, len(cameras))
        cam_stems = _select_stems(stems, cam_index, len(cameras), len(cam_images), camera)
        cam_records = []
        for i, (image, joint, k, stem) in enumerate(zip(cam_images, joints[: len(cam_images)], cam_intr, cam_stems, strict=True)):
            cam_records.append(
                Record(
                    stem=str(stem),
                    path=cfg.arrayrecord_dir or cfg.output_dir / f"{camera}_{i:06d}",
                    image=np.asarray(image, dtype=np.uint8),
                    joints=np.asarray(joint, dtype=np.float32).reshape(-1)[:7],
                    intrinsics=np.asarray(k, dtype=np.float32),
                    w2c=None,
                )
            )
        batches.append(
            DreamCameraBatch(
                camera=camera,
                records=cam_records,
                masks=np.asarray(cam_masks, dtype=np.uint8),
                initial={"w2c": np.asarray(cam_initial, dtype=np.float32), "pose_source": np.asarray("dream_arrayrecord")},
            )
        )
    return batches


def _camera_names(cfg: Config, arrays: dict[str, np.ndarray]) -> list[str]:
    if cfg.arrayrecord_cameras:
        return list(cfg.arrayrecord_cameras)
    raw = _optional(arrays, "camera", "cameras", "cam", "cams", "camera_names")
    if raw is None:
        return ["camera"]
    names = np.asarray(raw).reshape(-1)
    return [_decode_name(name) for name in names]


def _decode_name(value: Any) -> str:
    if isinstance(value, bytes | np.bytes_):
        return value.decode().strip("\x00")
    return str(value).strip("\x00")


def _required(arrays: dict[str, np.ndarray], *keys: str) -> np.ndarray:
    value = _optional(arrays, *keys)
    if value is None:
        raise KeyError(f"DREAM output is missing one of {keys}. Available keys: {sorted(arrays)}")
    return np.asarray(value)


def _optional(arrays: dict[str, np.ndarray], *keys: str) -> np.ndarray | None:
    for key in keys:
        if key in arrays:
            return np.asarray(arrays[key])
    return None


def _select_camera_axis(arr: np.ndarray, cam_index: int, n_cameras: int, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if n_cameras == 1:
        if arr.ndim >= 4 and arr.shape[0] == 1:
            return arr[0]
        if arr.ndim >= 4 and arr.shape[1] == 1:
            return arr[:, 0]
        return arr
    if arr.ndim < 4:
        raise ValueError(f"Expected multi-camera {name} with at least 4 dims, got {arr.shape}")
    if arr.shape[0] == n_cameras:
        return arr[cam_index]
    if arr.shape[1] == n_cameras:
        return arr[:, cam_index]
    raise ValueError(f"Cannot find camera axis of length {n_cameras} in {name} shape {arr.shape}")


def _select_intrinsics(arr: np.ndarray, cam_index: int, n_cameras: int, n_records: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape == (3, 3):
        return np.repeat(arr[None], n_records, axis=0)
    if arr.ndim == 3:
        if arr.shape[0] == n_records:
            return arr
        if arr.shape[0] == n_cameras:
            return np.repeat(arr[cam_index][None], n_records, axis=0)
    if arr.ndim == 4:
        if arr.shape[0] == n_cameras:
            return arr[cam_index]
        if arr.shape[1] == n_cameras:
            return arr[:, cam_index]
    raise ValueError(f"Cannot align intrinsics shape {arr.shape} with {n_records} records and {n_cameras} cameras")


def _select_initial(arr: np.ndarray, cam_index: int, n_cameras: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape == (4, 4):
        return arr
    if arr.ndim == 3 and arr.shape[0] == n_cameras:
        return arr[cam_index]
    if arr.ndim == 4 and arr.shape[0] == n_cameras:
        return arr[cam_index]
    if arr.ndim == 4 and arr.shape[1] == n_cameras:
        return arr[:, cam_index]
    return arr


def _select_stems(stems: np.ndarray | None, cam_index: int, n_cameras: int, n_records: int, camera: str) -> list[str]:
    if stems is None:
        return [f"{camera}_{i:06d}" for i in range(n_records)]
    arr = np.asarray(stems)
    if arr.ndim == 2 and arr.shape[0] == n_cameras:
        arr = arr[cam_index]
    elif arr.ndim == 2 and arr.shape[1] == n_cameras:
        arr = arr[:, cam_index]
    arr = arr.reshape(-1)[:n_records]
    return [_decode_name(value) for value in arr]
