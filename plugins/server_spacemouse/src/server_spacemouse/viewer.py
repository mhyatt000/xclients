"""Simple numpy-based viewer for 6-DoF input."""

from __future__ import annotations

import numpy as np


class Viewer:
    """Render a 256x256 visualization of translation inputs."""

    def __init__(self, size: int = 256) -> None:
        self.size = size
        self.base_radius = size / 4.0  # diameter is half image width

    def step(self, x: np.ndarray) -> np.ndarray:
        vec = np.clip(np.asarray(x, dtype=float).reshape(-1), -1.0, 1.0)
        if vec.shape[0] < 6:
            raise ValueError("Viewer.step expects a vector with at least 6 elements")

        tx, ty, tz = vec[:3]
        rx, ry = vec[3:5]
        ry = -ry
        tz = -tz

        img = np.full((self.size, self.size, 3), 255, dtype=np.uint8)

        # scale circle radius with z translation (±25%).
        radius = self.base_radius * (1.0 + 0.25 * tz)
        radius = max(1.0, min(radius, self.size / 2.0))

        # rx/ry specify a stretch direction and magnitude for the radius.
        stretch_vec = np.array([rx, ry], dtype=float)
        stretch_mag = float(np.linalg.norm(stretch_vec))
        stretch_mag_clipped = min(stretch_mag, 1.0)
        extra_radius = radius * 0.75 * stretch_mag_clipped
        max_radius = radius + extra_radius
        if stretch_mag > 1e-6:
            stretch_dir = stretch_vec / stretch_mag
        else:
            stretch_dir = np.array([0.0, 0.0], dtype=float)

        # move circle center with tx/ty up to 10% of image size.
        max_offset = 0.1 * self.size
        center_y = self.size / 2.0 - tx * max_offset  # +tx moves up
        center_x = self.size / 2.0 - ty * max_offset  # +ty moves left
        center_y = float(np.clip(center_y, max_radius, self.size - max_radius))
        center_x = float(np.clip(center_x, max_radius, self.size - max_radius))

        # draw filled circle with directional stretch (light green)
        yy, xx = np.ogrid[: self.size, : self.size]
        rel_x = xx - center_x
        rel_y = yy - center_y
        dist = np.sqrt(rel_x**2 + rel_y**2)

        if stretch_mag > 1e-6:
            with np.errstate(invalid="ignore"):
                zeros = np.zeros_like(dist)
                unit_x = np.divide(rel_x, dist, out=zeros.copy(), where=dist > 0)
                unit_y = np.divide(rel_y, dist, out=zeros.copy(), where=dist > 0)
            alignment = unit_x * stretch_dir[0] + unit_y * stretch_dir[1]
            alignment = np.clip(alignment, 0.0, 1.0)
            stretch_radius = radius + extra_radius * alignment
            mask = dist <= stretch_radius
        else:
            mask = dist <= radius

        img[mask] = np.array([144, 238, 144], dtype=np.uint8)

        return img
