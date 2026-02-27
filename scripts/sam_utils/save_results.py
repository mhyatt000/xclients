"""Result saving pipeline for SAM-based segmentation."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime  import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


def ensure_dir(path: Path) -> Path:
    """Ensure that a directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_result_dir(base_dir: Path, frame_idx: int, timestamp: Optional[datetime] = None) -> Path:
    """Create a directory for saving results of a specific frame."""
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(base_dir / f"frame_{frame_idx:04d}_{ts}")


def write_image(path: Path, img: np.ndarray) -> None:
    """Write an image to disk."""
    saved = cv2.imwrite(str(path), img)
    if not saved:
        raise IOError(f"Failed to save image to {path}")


def squeeze_mask_to_hw(mask: np.ndarray) -> Optional[np.ndarray]:
    """Attempts to reduce mask to (H, W). Returns None if mask is empty."""
    if mask is None or mask.size == 0 or (mask.ndim ==4 and mask.shape[0] == 0):
        return None

    m = mask
    while m.ndim > 2:
        if m.shape[0] ==1:
            m = m[0]
        elif m.shape[-1] == 1:
            m = m[..., 0]
        else:
            m = m[0]
    return m


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a binary mask to an image, setting masked-out areas to black."""
    if mask.shape != image.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}")

    masked = image.copy()
    masked[mask == 0] = 0
    return masked


def mask_to_uint8_255(mask: np.ndarray) -> np.ndarray:
    """Convert a binary mask to uint8 format with values 0 and 255."""
    return (mask.astype(np.uint8) * 255)


def normalize_pointmap_to_u8(pointmap: np.ndarray) -> np.ndarray:
    """
    Normalize a pointmap to the range [0, 255] and convert to uint8.

    Standardizes using mean / std over all values
    Map [-1, 1] -> [0, 1] via (x + 1) / 2
    scale to [0, 255]
    """
    p = pointmap
    mu, std = p.mean(), p.std()
    p = (p - mu) / (std + 1e-8)  # Standardize
    p = (p + 1) / 2  # Map to [0, 1]
    p = (p * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    return p


def to_jsonable_output(
        output: dict[str, Any],
        *,
        result_dir: Path,
        ndarray_max_elems: int = 1000,
        ) -> dict[str, Any]:
    """Convert output dict to a JSON-serializable format, saving large arrays to disk."""
    data_to_save: dict[str, Any] = {}

    for key, value in output.items():
        if isinstance(value, np.ndarray):
            data_to_save[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "data": value.tolist() if value.size < ndarray_max_elems else "too large to save",
            }
        elif key == "mesh" and isinstance(value, dict):
            if "vertices" in value and "faces" in value:
                mesh_path = result_dir / "mesh.npz"
                np.savez(mesh_path, vertices=value["vertices"], faces=value["faces"])
                data_to_save[key] = "saved to mesh.npz"
            else:
                data_to_save[key] = str(value)
        else:
            data_to_save[key] = str(value)

    return data_to_save


@dataclass
class ResultSaver:
    """Utility class for saving SAM-based segmenetation results."""
    base_dir: Path = Path("results")

    def save_results(
            self,
            frame: np.ndarra,
            output: dict[str, Any],
            cfg: Any,
            *,
            frame_idx: int = 0,
            mask: Optional[np.ndarray] = None,
            ) -> None:
        ensure_dir(self.base_dir)
        result_dir = make_result_dir(self.base_dir, frame_idx)

        self._save_original(frame, result_dir)
        self._save_mask_assets(frame, mask, result_dir)
        self._save_pointmap(output, result_dir)
        self._save_pointmap_colors(output, result_dir)
        self._save_overlay(frame, output, result_dir)
        self._save_output_json(output, result_dir)

        logging.info(f"Results saved to {result_dir}")
        return result_dir


    def _save_original(self, frame: np.ndarray, result_dir: Path) -> None:
        """Save the original input frame."""
        write_image(result_dir / "original.png", frame)


    def _save_mask_assets(self, frame: np.ndarray, mask: Optional[np.ndarray], result_dir: Path) -> None:
        """Save the SAM3 mask and masked image."""
        if mask is None:
            return

        logging.info(f"Original mask shape: {getattr(mask, 'shape', None)}, dtype: {getattr(mask, 'dtype', None)}")

        mask_hw = squeeze_mask_to_hw(mask)
        if mask_hw is None:
            return

        logging.info(
            f"Processed mask shape: {mask_hw.shape}, dtype: {mask_hw.dtype}, "
            f"min: {np.min(mask_hw)}, max: {np.max(mask_hw)}"
        )

        mask_vis = mask_to_uint8_255(mask_hw)
        write_image(result_dir / "sam3_mask.png", mask_vis)

        masked_image = apply_mask(frame, mask_vis)
        write_image(result_dir / "masked_image.png", masked_image)

        logging.info("Saved SAM3 mask and masked image")


    def _save_pointmap(self, output: dict[str, Any], result_dir: Path) -> None:
        """Save the pointmap visualization."""
        if "pointmap" not in output:
            return

        p_u8 = normalize_pointmap_to_u8(output["pointmap"])
        # Your original did RGB->BGR conversion; keep it.
        write_image(result_dir / "pointmap.png", cv2.cvtColor(p_u8, cv2.COLOR_RGB2BGR))

    def _save_pointmap_colors(self, output: dict[str, Any], result_dir: Path) -> None:
        if "pointmap_colors" not in output:
            return

        pointmap_colors = output["pointmap_colors"]
        if not isinstance(pointmap_colors, np.ndarray):
            return

        logging.info(f"Pointmap colors shape: {pointmap_colors.shape}, dtype: {pointmap_colors.dtype}")

        vis = (pointmap_colors * 255).astype(np.uint8)
        bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        write_image(result_dir / "pointmap_colors.png", bgr)


    def _save_overlay(self, frame: np.ndarray, output: dict[str, Any], result_dir: Path) -> None:
        """Save an overlay of the mask on the original image."""
        overlay = frame.copy()
        y = 30

        def put(label: str, arr: Any, n: int, y_step: int = 30) -> None:
            nonlocal y

            a = np.asarray(arr)
            text = f"{label}: {a.flatten()[:n]}"
            logging.info(f"{label} shape: {getattr(a, 'shape', None)}, value: {a}")
            cv2.putText(overlay, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += y_step


        put("Rotation", output.get("rotation"), 4)
        print(f"Rotation: {output.get('rotation')}")
        put("6D Rot", output.get("6drotation_normalized"), 6)
        print(f"6D Rotation: {output.get('6drotation_normalized')}")
        put("Trans", output.get("translation"), 9999)
        print(f"Translation: {output.get('translation')}")
        put("Scale", output.get("scale"), 9999, y_step=0)
        print(f"Scale: {output.get('scale')}")

        write_image(result_dir / "overlay.png", overlay)


    def _save_output_json(self, output: dict[str, Any], result_dir: Path) -> None:
        """Save the output dict as a JSON file, converting non-serializable data."""
        jsonable_output = to_jsonable_output(output, result_dir=result_dir)
        with open(result_dir / "output.json", "w") as f:
            json.dump(jsonable_output, f, indent=2)













