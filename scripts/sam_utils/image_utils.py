"""Utility functions for image processing in SAM models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class CameraInput:
    """Configuration for camera input."""
    device: int = 0

def load_image(image_path: Path) -> np.ndarray:
    """Load image from file"""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = cv2.imread(str(image_path))

    logging.info(f"Loaded image with shape: {img.shape}")
    return img


def get_frame(cfg, cap: Optional[cv2.VideoCapture] = None) -> np.ndarray:
    """Get a frame from either camera or image file."""
    if isinstance(cfg.input_source, CameraInput):
        ret, frame = cap.read()
        return frame
    else:
        return load_image(cfg.input_source.image_path)
