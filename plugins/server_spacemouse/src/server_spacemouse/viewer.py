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

        img = np.full((self.size, self.size, 3), 255, dtype=np.uint8)

        # scale circle radius with z translation (±25%).
        radius = self.base_radius * (1.0 + 0.25 * tz)
        radius = max(1.0, min(radius, self.size / 2.0))

        # move circle center with rx/ry up to 10% of image size.
        max_offset = 0.1 * self.size
        center_y = self.size / 2.0 - rx * max_offset  # +rx moves up
        center_x = self.size / 2.0 - ry * max_offset  # +ry moves left
        center_y = float(np.clip(center_y, radius, self.size - radius))
        center_x = float(np.clip(center_x, radius, self.size - radius))

        # draw filled circle (light green)
        yy, xx = np.ogrid[: self.size, : self.size]
        circle_mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
        img[circle_mask] = np.array([144, 238, 144], dtype=np.uint8)

        # Draw red line showing translation direction (x up/down, y left/right)
        direction_norm = np.hypot(tx, ty)
        if direction_norm > 1e-6:
            dir_y = -tx / direction_norm
            dir_x = -ty / direction_norm
            end_y = center_y + dir_y * radius
            end_x = center_x + dir_x * radius
            steps = int(max(abs(end_y - center_y), abs(end_x - center_x))) + 1
            line_ys = np.linspace(center_y, end_y, steps)
            line_xs = np.linspace(center_x, end_x, steps)
            line_rows = np.clip(np.rint(line_ys).astype(int), 0, self.size - 1)
            line_cols = np.clip(np.rint(line_xs).astype(int), 0, self.size - 1)
            img[line_rows, line_cols] = np.array([255, 0, 0], dtype=np.uint8)

        return img
