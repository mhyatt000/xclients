"""Static world obstacles for the planner, expressed in world frame.

Each geom can exempt robot links from its collision cost (e.g. the mount box
overlaps the arm's own base links, which would otherwise be permanently "in
collision" and poison the solve).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import jaxlie
import numpy as np
from numpy.typing import NDArray
from pyroki.collision import Box, CollGeom, HalfSpace

IN = 0.0254

# The V-mount structure: links at or below link2 live inside/next to the box.
MOUNT_IGNORE_LINKS = ("world", "link_base", "link1", "link2")


@dataclass(frozen=True)
class WorldGeom:
    """One world-frame obstacle plus the robot links exempt from it."""

    geom: CollGeom
    ignore_links: tuple[str, ...] = ()


@dataclass(frozen=True)
class World:
    """Static world obstacles (world frame)."""

    geoms: tuple[WorldGeom, ...]

    @staticmethod
    def default() -> World:
        """Floor plane at z=0 plus the dual-arm mount box.

        Box: x in [0, 5in], y in [-1ft, 1ft], z in [0, 9in]. The mount box
        ignores every link up to link2 in the chain (those bolt onto it).
        """
        floor = HalfSpace.from_point_and_normal(np.zeros(3), np.array([0.0, 0.0, 1.0]))
        mount = Box.from_extent(
            [5 * IN, 24 * IN, 9 * IN],
            position=[2.5 * IN, 0.0, 4.5 * IN],
        )
        return World(
            geoms=(
                WorldGeom(floor),
                WorldGeom(mount, ignore_links=MOUNT_IGNORE_LINKS),
            )
        )

    def in_base_frame(self, world_T_base: NDArray[np.floating]) -> list[CollGeom]:
        """Geoms transformed into the robot base frame the planner solves in."""
        T = jaxlie.SE3.from_matrix(jnp.asarray(np.linalg.inv(world_T_base)))
        return [wg.geom.transform(T) for wg in self.geoms]

    def link_masks(self, link_names: list[str]) -> list[jnp.ndarray]:
        """Per-geom, per-link 0/1 weights implementing each geom's ignore list."""
        return [
            jnp.array([0.0 if name in wg.ignore_links else 1.0 for name in link_names])
            for wg in self.geoms
        ]
