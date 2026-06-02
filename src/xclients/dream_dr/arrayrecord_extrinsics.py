from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from xclients.dream_dr.dream_arrayrecord import add_crossformer_path, resolve_camera_name, safe_camera_id


@dataclass
class MaterializeConfig:
    arrayrecord_dir: Path
    extrinsics_npz: Path
    output_arrayrecord_dir: Path | None = None
    output_version: str | None = None
    overwrite: bool = False
    dry_run: bool = False
    verify: bool = True
    default_missing: str = "identity"
    max_shards: int | None = None


@dataclass
class BranchLayout:
    source_root: Path
    source_branch: Path
    output_root: Path
    output_branch: Path


def materialize_arrayrecord_extrinsics(cfg: MaterializeConfig) -> None:
    add_crossformer_path()
    cfg.arrayrecord_dir = cfg.arrayrecord_dir.expanduser().resolve()
    cfg.extrinsics_npz = cfg.extrinsics_npz.expanduser().resolve()
    if cfg.output_arrayrecord_dir is not None:
        cfg.output_arrayrecord_dir = cfg.output_arrayrecord_dir.expanduser().resolve()

    layout = resolve_layout(cfg.arrayrecord_dir, cfg.output_arrayrecord_dir, cfg.output_version)
    extrinsics = load_extrinsics(cfg.extrinsics_npz)
    check_output(layout, cfg)

    image_shards = sorted((layout.source_branch / "image").glob("*.arrayrecord"))
    proprio_shards = sorted((layout.source_branch / "proprio").glob("*.arrayrecord"))
    data_shards = sorted((layout.source_branch / "data").glob("*.arrayrecord")) or sorted(
        layout.source_branch.glob("*.arrayrecord")
    )

    if image_shards and proprio_shards:
        write_split_shards(layout, limit_shards(image_shards, cfg), limit_shards(proprio_shards, cfg), extrinsics, cfg)
    elif data_shards:
        write_data_shards(layout, limit_shards(data_shards, cfg), extrinsics, cfg)
    else:
        raise FileNotFoundError(f"No ArrayRecord shards found under {layout.source_branch}")

    if cfg.dry_run:
        return

    copy_metadata(layout)
    if cfg.verify:
        verify_output(layout, extrinsics)
    logging.info("Wrote ArrayRecord with extrinsics to %s", layout.output_root)


def limit_shards(paths: list[Path], cfg: MaterializeConfig) -> list[Path]:
    if cfg.max_shards is None:
        return paths
    return paths[: max(0, int(cfg.max_shards))]


def resolve_layout(
    arrayrecord_dir: Path,
    output_arrayrecord_dir: Path | None,
    output_version: str | None,
) -> BranchLayout:
    if (arrayrecord_dir / "image").exists() or (arrayrecord_dir / "data").exists():
        source_branch = arrayrecord_dir
        source_root = arrayrecord_dir.parent
    elif (arrayrecord_dir / "main").exists():
        source_root = arrayrecord_dir
        source_branch = arrayrecord_dir / "main"
    else:
        raise FileNotFoundError(f"Could not find ArrayRecord branch under {arrayrecord_dir}")

    if output_arrayrecord_dir is None and output_version is None:
        raise ValueError("Pass either --output-arrayrecord-dir or --output-version.")
    if output_arrayrecord_dir is not None and output_version is not None:
        raise ValueError("Pass only one of --output-arrayrecord-dir or --output-version.")

    if output_version is not None:
        output_root = source_root.parent / output_version
        output_branch = output_root / source_branch.name
    elif output_arrayrecord_dir is not None and source_branch.name == "main":
        output_root = output_arrayrecord_dir
        output_branch = output_arrayrecord_dir / "main"
    elif output_arrayrecord_dir is not None:
        output_root = output_arrayrecord_dir.parent
        output_branch = output_arrayrecord_dir
    else:
        raise AssertionError("unreachable")

    if output_root == source_root or output_branch == source_branch:
        raise ValueError("Refusing to write over source ArrayRecord. Use a new output version or output directory.")

    return BranchLayout(
        source_root=source_root,
        source_branch=source_branch,
        output_root=output_root,
        output_branch=output_branch,
    )


def load_extrinsics(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        extrinsics = {key: np.asarray(data[key], dtype=np.float32) for key in data.files}
    bad = {key: value.shape for key, value in extrinsics.items() if value.shape != (4, 4)}
    if bad:
        raise ValueError(f"Expected all extrinsics to have shape (4, 4), got {bad}")
    return extrinsics


def check_output(layout: BranchLayout, cfg: MaterializeConfig) -> None:
    if cfg.dry_run:
        return
    if layout.output_branch.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"Output already exists: {layout.output_branch}. Pass --overwrite to replace it.")
        shutil.rmtree(layout.output_branch)
    layout.output_branch.mkdir(parents=True, exist_ok=True)


def write_split_shards(
    layout: BranchLayout,
    image_shards: list[Path],
    proprio_shards: list[Path],
    extrinsics: dict[str, np.ndarray],
    cfg: MaterializeConfig,
) -> None:
    from array_record.python.array_record_data_source import ArrayRecordDataSource
    from crossformer.data.arec.arec import pack_record, unpack_record

    if len(image_shards) != len(proprio_shards):
        raise ValueError(f"Image/proprio shard count mismatch: {len(image_shards)} != {len(proprio_shards)}")

    for image_path, proprio_path in zip(image_shards, proprio_shards, strict=True):
        shard_id = image_path.stem
        image_src = ArrayRecordDataSource([str(image_path)])
        proprio_src = ArrayRecordDataSource([str(proprio_path)])
        if len(image_src) != len(proprio_src):
            raise ValueError(f"Shard length mismatch for {shard_id}: {len(image_src)} != {len(proprio_src)}")

        first_image = unpack_record(image_src[0])
        first_proprio = unpack_record(proprio_src[0])
        first_camera_keys = decode_camera_keys(first_proprio.get("info", {}), first_image.get("image"))
        first_extrs, first_valid = extrinsics_for_cameras(
            shard_id,
            first_camera_keys,
            extrinsics,
            cfg.default_missing,
        )
        logging.info(
            "%s: records=%d cameras=%s extrinsics_valid=%s",
            shard_id,
            len(image_src),
            first_camera_keys,
            first_valid.tolist(),
        )
        if cfg.dry_run:
            continue

        extrinsics_cache = {tuple(first_camera_keys): (first_extrs, first_valid)}
        image_writer = open_writer(layout.output_branch / "image" / image_path.name, "group_size:1")
        proprio_writer = open_writer(layout.output_branch / "proprio" / proprio_path.name, "group_size:32")
        try:
            for index in range(len(image_src)):
                image_record = unpack_record(image_src[index])
                proprio_record = unpack_record(proprio_src[index])
                camera_keys = decode_camera_keys(proprio_record.get("info", {}), image_record.get("image"))
                extrs, valid = cached_extrinsics(
                    extrinsics_cache,
                    shard_id,
                    camera_keys,
                    extrinsics,
                    cfg.default_missing,
                )
                image_record = add_extrinsics(image_record, extrs, valid)
                image_writer.write(pack_record(image_record))
                proprio_writer.write(proprio_src[index])
        finally:
            image_writer.close()
            proprio_writer.close()


def write_data_shards(
    layout: BranchLayout,
    data_shards: list[Path],
    extrinsics: dict[str, np.ndarray],
    cfg: MaterializeConfig,
) -> None:
    from array_record.python.array_record_data_source import ArrayRecordDataSource
    from crossformer.data.arec.arec import pack_record, unpack_record

    for path in data_shards:
        shard_id = path.stem
        src = ArrayRecordDataSource([str(path)])
        first = unpack_record(src[0])
        first_camera_keys = decode_camera_keys(first.get("info", {}), first.get("image"))
        first_extrs, first_valid = extrinsics_for_cameras(shard_id, first_camera_keys, extrinsics, cfg.default_missing)
        logging.info(
            "%s: records=%d cameras=%s extrinsics_valid=%s",
            shard_id,
            len(src),
            first_camera_keys,
            first_valid.tolist(),
        )
        if cfg.dry_run:
            continue

        extrinsics_cache = {tuple(first_camera_keys): (first_extrs, first_valid)}
        writer = open_writer(layout.output_branch / "data" / path.name, "group_size:1")
        try:
            for index in range(len(src)):
                raw_record = unpack_record(src[index])
                camera_keys = decode_camera_keys(raw_record.get("info", {}), raw_record.get("image"))
                extrs, valid = cached_extrinsics(
                    extrinsics_cache,
                    shard_id,
                    camera_keys,
                    extrinsics,
                    cfg.default_missing,
                )
                record = add_extrinsics(raw_record, extrs, valid)
                writer.write(pack_record(record))
        finally:
            writer.close()


def open_writer(path: Path, options: str):
    from array_record.python.array_record_module import ArrayRecordWriter

    path.parent.mkdir(parents=True, exist_ok=True)
    return ArrayRecordWriter(str(path), options=options)


def decode_camera_keys(info: dict[str, Any], image: Any) -> list[str]:
    keys = info.get("image_keys")
    if keys is not None:
        return [decode_name(key) for key in np.asarray(keys).reshape(-1)]
    if isinstance(image, dict):
        return [decode_name(key) for key in image]
    image_arr = np.asarray(image)
    if image_arr.ndim == 4:
        return [str(index) for index in range(image_arr.shape[0])]
    return ["camera"]


def decode_name(value: Any) -> str:
    if isinstance(value, bytes | np.bytes_):
        return value.decode().strip("\x00")
    return str(value).strip("\x00")


def extrinsics_for_cameras(
    shard_id: str,
    camera_keys: list[str],
    extrinsics: dict[str, np.ndarray],
    default_missing: str,
) -> tuple[np.ndarray, np.ndarray]:
    extrs = []
    valid = []
    for camera in camera_keys:
        key = find_extrinsics_key(shard_id, camera, camera_keys, extrinsics)
        if key is None:
            extrs.append(missing_extrinsics(default_missing))
            valid.append(False)
        else:
            extrs.append(extrinsics[key])
            valid.append(True)
    return np.stack(extrs, axis=0).astype(np.float32), np.asarray(valid, dtype=bool)


def cached_extrinsics(
    cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]],
    shard_id: str,
    camera_keys: list[str],
    extrinsics: dict[str, np.ndarray],
    default_missing: str,
) -> tuple[np.ndarray, np.ndarray]:
    key = tuple(camera_keys)
    if key not in cache:
        cache[key] = extrinsics_for_cameras(shard_id, camera_keys, extrinsics, default_missing)
    return cache[key]


def find_extrinsics_key(
    shard_id: str,
    camera: str,
    camera_keys: list[str],
    extrinsics: dict[str, np.ndarray],
) -> str | None:
    shard_safe = safe_camera_id(shard_id)
    exact = f"{shard_safe}__{safe_camera_id(camera)}"
    if exact in extrinsics:
        return exact

    suffixes = [key.removeprefix(f"{shard_safe}__") for key in extrinsics if key.startswith(f"{shard_safe}__")]
    for suffix in suffixes:
        try:
            resolved = resolve_camera_name(suffix, camera_keys)
        except KeyError:
            continue
        if resolved == camera:
            return f"{shard_safe}__{suffix}"
    return None


def missing_extrinsics(default_missing: str) -> np.ndarray:
    if default_missing == "identity":
        return np.eye(4, dtype=np.float32)
    if default_missing == "zeros":
        return np.zeros((4, 4), dtype=np.float32)
    raise ValueError(f"Unsupported default_missing={default_missing!r}; expected 'identity' or 'zeros'")


def add_extrinsics(record: dict[str, Any], extrs: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    out = dict(record)
    obs = dict(out.get("obs", {}))
    obs["extrs"] = np.asarray(extrs, dtype=np.float32)
    out["obs"] = obs

    mask = dict(out.get("mask", {}))
    mask_obs = dict(mask.get("obs", {}))
    mask_obs["extrs"] = np.asarray(valid, dtype=bool)
    mask["obs"] = mask_obs
    out["mask"] = mask
    return out


def copy_metadata(layout: BranchLayout) -> None:
    for name in ("meta.json", "spec.json"):
        src = layout.source_branch / name
        if not src.exists():
            continue
        dst = layout.output_branch / name
        if name == "meta.json":
            write_meta(src, dst)
        else:
            write_spec(src, dst)


def write_meta(src: Path, dst: Path) -> None:
    meta = json.loads(src.read_text())
    writers = meta.get("writers", {})
    if "image" in writers:
        writers["image"] = update_writer_keys(writers["image"], ("obs", "mask"))
    elif "data" in writers:
        writers["data"] = update_writer_keys(writers["data"], ("obs", "mask"))
    meta["writers"] = writers
    meta["version"] = dst.parent.parent.name
    meta["schema_fingerprint"] = f"{meta.get('schema_fingerprint', 'unknown')}-extr"
    meta["extrinsics_materialized"] = True
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(meta, indent=2))


def update_writer_keys(writer: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(writer, list) and len(writer) == 2 and isinstance(writer[0], list):
        existing = list(writer[0])
        for key in keys:
            if key not in existing:
                existing.append(key)
        return [existing, writer[1]]
    if isinstance(writer, list):
        existing = list(writer)
        for key in keys:
            if key not in existing:
                existing.append(key)
        return existing
    return writer


def write_spec(src: Path, dst: Path) -> None:
    spec = json.loads(src.read_text())
    image_spec = first_image_spec(spec)
    k = int(image_spec[0][0]) if image_spec and image_spec[0] else 1
    addition = {
        "obs": {"extrs": [[k, 4, 4], "float32"]},
        "mask": {"obs": {"extrs": [[k], "bool"]}},
    }
    for section in (spec.get("data"), spec.get("writers", {}).get("image"), spec.get("writers", {}).get("data")):
        if isinstance(section, dict):
            deep_merge(section, addition)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(spec, indent=2))


def first_image_spec(spec: dict[str, Any]) -> Any | None:
    for section in (spec.get("writers", {}).get("image"), spec.get("writers", {}).get("data"), spec.get("data")):
        if isinstance(section, dict) and "image" in section:
            return section["image"]
    return None


def deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict):
            child = dst.setdefault(key, {})
            if isinstance(child, dict):
                deep_merge(child, value)
            else:
                dst[key] = value
        else:
            dst[key] = value


def verify_output(layout: BranchLayout, extrinsics: dict[str, np.ndarray]) -> None:
    from array_record.python.array_record_data_source import ArrayRecordDataSource
    from crossformer.data.arec.arec import unpack_record

    image_shards = sorted((layout.output_branch / "image").glob("*.arrayrecord"))
    data_shards = sorted((layout.output_branch / "data").glob("*.arrayrecord")) or sorted(
        layout.output_branch.glob("*.arrayrecord")
    )
    path = image_shards[0] if image_shards else data_shards[0]
    rec = unpack_record(ArrayRecordDataSource([str(path)])[0])
    extrs = np.asarray(rec["obs"]["extrs"])
    valid = np.asarray(rec["mask"]["obs"]["extrs"])
    if extrs.ndim != 3 or extrs.shape[-2:] != (4, 4):
        raise ValueError(f"Output obs.extrs has wrong shape: {extrs.shape}")
    if valid.shape != extrs.shape[:1]:
        raise ValueError(f"Output mask.obs.extrs shape {valid.shape} does not match extrs {extrs.shape}")
    if valid.any() and not any(np.allclose(extrs[index], value) for index in np.where(valid)[0] for value in extrinsics.values()):
        raise ValueError("Verified output has valid extrinsics, but none matched the provided npz.")
    logging.info("Verified %s: obs.extrs=%s mask.obs.extrs=%s", path, extrs.shape, valid.shape)


def main(cfg: MaterializeConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    materialize_arrayrecord_extrinsics(cfg)


def cli() -> None:
    import tyro

    main(tyro.cli(MaterializeConfig))


if __name__ == "__main__":
    cli()
