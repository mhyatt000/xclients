from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from xclients.dream_dr.config import Config, Record


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"Failed to write image to {path}")


def save_outputs(cfg: Config, records: list[Record], masks: np.ndarray, initial: dict, dr_out: dict | None) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.output_dir / "HT_initial.npy", np.asarray(initial["w2c"], dtype=np.float32))
    if cfg.call_dream:
        np.save(cfg.output_dir / "HT_dream.npy", np.asarray(initial["w2c"], dtype=np.float32))
    if dr_out is not None:
        np.save(cfg.output_dir / "HT_dr.npy", np.asarray(dr_out["HT"], dtype=np.float32))
        for key in ("overlays", "difference", "renders", "render_overlays"):
            if key not in dr_out:
                continue
            for record, image in zip(records, np.asarray(dr_out[key]), strict=True):
                write_image(cfg.output_dir / key / f"{record.stem}_{key}.png", image)

    metadata = {
        "records": [str(record.path) for record in records],
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(cfg).items()},
        "mask_shape": list(masks.shape),
        "initial_pose_keys": sorted(initial),
        "dr_keys": [] if dr_out is None else sorted(dr_out),
    }
    with (cfg.output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def print_inspect(records: list[Record], cfg: Config) -> None:
    print(f"records: {len(records)}")
    print(f"data_dir: {cfg.data_dir}")
    print(f"output_dir: {cfg.output_dir}")
    joints = np.stack([record.joints for record in records])
    if np.allclose(joints, 0.0):
        logging.warning("All loaded joint vectors are zero. DR renders will use the same robot pose for every record.")
    elif np.allclose(joints, joints[0]):
        logging.warning("All loaded joint vectors are identical. DR renders will not reflect per-record robot motion.")
    for record in records:
        print(
            f"{record.stem}: image={record.image.shape} joints={record.joints.shape} "
            f"K_fx_fy=({record.intrinsics[0, 0]:.3f}, {record.intrinsics[1, 1]:.3f}) "
            f"w2c={'yes' if record.w2c is not None else 'no'}"
        )
