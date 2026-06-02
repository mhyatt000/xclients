from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from xclients.dream_dr.config import Config, Record


def scale_intrinsics(k: np.ndarray, old_h: int, old_w: int, new_h: int, new_w: int) -> np.ndarray:
    out = np.asarray(k, dtype=np.float32).copy()
    out[0, :] *= new_w / float(old_w)
    out[1, :] *= new_h / float(old_h)
    return out


def model_image(data: dict[str, np.ndarray], size: int) -> tuple[np.ndarray, np.ndarray]:
    if "image_model" in data:
        image = np.asarray(data["image_model"])
        while image.ndim > 3 and image.shape[0] == 1:
            image = image[0]
    else:
        image = np.asarray(data["image"])

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB/BGR image shape (h, w, 3), got {image.shape}")

    image = image.astype(np.uint8, copy=False)
    h, w = image.shape[:2]
    if (h, w) == (size, size):
        return image, np.asarray(data["K"], dtype=np.float32)

    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized, scale_intrinsics(data["K"], h, w, size, size)


def load_records(cfg: Config) -> list[Record]:
    paths = sorted(cfg.data_dir.glob("*.npz"))
    if cfg.data_select != [-1]:
        selected = []
        for index in cfg.data_select:
            try:
                selected.append(paths[index])
            except IndexError as exc:
                raise IndexError(
                    f"data_select index {index} is out of range for {len(paths)} .npz files under {cfg.data_dir}"
                ) from exc
        paths = selected
    if cfg.max_records is not None:
        paths = paths[: cfg.max_records]
    if not paths:
        raise FileNotFoundError(f"No .npz records found under {cfg.data_dir}")

    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        image, k = model_image(arrays, cfg.image_size)
        records.append(
            Record(
                stem=path.stem,
                path=path,
                image=image,
                joints=np.asarray(arrays["joints"], dtype=np.float32).reshape(-1)[:7],
                intrinsics=k,
                w2c=np.asarray(arrays["w2c"], dtype=np.float32) if "w2c" in arrays else None,
            )
        )
    return records


def latest_npz(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {path}")
    return files[-1]


def load_first_array(path: Path, keys: tuple[str, ...]) -> np.ndarray | None:
    npz = latest_npz(path)
    loaded = np.load(npz, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        return np.asarray(loaded, dtype=np.float32)
    with loaded as data:
        for key in keys:
            if key in data.files:
                return np.asarray(data[key], dtype=np.float32)
        logging.warning("No keys %s found in %s. Available keys: %s", keys, npz, data.files)
    return None


def load_intrinsics_override(cfg: Config) -> np.ndarray | None:
    if cfg.intrinsics_path is None:
        return None
    if not cfg.intrinsics_path.exists():
        logging.warning("Intrinsics path does not exist: %s", cfg.intrinsics_path)
        return None
    intr = load_first_array(cfg.intrinsics_path, ("K", "intrinsics", "camera_matrix"))
    if intr is None:
        return None
    while intr.ndim > 2 and intr.shape[0] == 1:
        intr = intr[0]
    if intr.shape != (3, 3):
        raise ValueError(f"Expected intrinsics shape (3, 3), got {intr.shape} from {cfg.intrinsics_path}")
    return intr.astype(np.float32)


def apply_intrinsics_override(records: list[Record], intrinsics: np.ndarray | None) -> None:
    if intrinsics is None:
        return
    for record in records:
        record.intrinsics = intrinsics.copy()
