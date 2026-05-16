from __future__ import annotations

import jax
import jax.numpy as jnp

_HILBERT_CUBE_CORNERS = jnp.array(
    [
        [0.0, 0.0, 0.0],  # black
        [1.0, 0.0, 0.0],  # red
        [1.0, 1.0, 0.0],  # yellow
        [0.0, 1.0, 0.0],  # green
        [0.0, 1.0, 1.0],  # cyan
        [0.0, 0.0, 1.0],  # blue
        [1.0, 0.0, 1.0],  # violet
        [1.0, 1.0, 1.0],  # white
    ],
    dtype=jnp.float32,
)


def _normalized_depth_to_hilbert_rgb(d: jax.Array | float) -> jax.Array:
    d = jnp.asarray(d, dtype=jnp.float32)
    x = jnp.clip(d, 0.0, 1.0) * (_HILBERT_CUBE_CORNERS.shape[0] - 1)
    i = jnp.minimum(x.astype(jnp.int32), _HILBERT_CUBE_CORNERS.shape[0] - 2)
    t = x - i.astype(jnp.float32)
    return (1.0 - t) * _HILBERT_CUBE_CORNERS[i] + t * _HILBERT_CUBE_CORNERS[i + 1]


def depth_to_hilbert_rgb(
    d: jax.Array | float,
    *,
    min_depth: float = 0.0,
    max_depth: float = 1.0,
) -> jax.Array:
    """Map depth to RGB along the Hilbert-style cube edge path."""
    d = jnp.asarray(d, dtype=jnp.float32)
    scale = jnp.maximum(jnp.asarray(max_depth - min_depth, dtype=jnp.float32), jnp.finfo(jnp.float32).eps)
    return _normalized_depth_to_hilbert_rgb((d - min_depth) / scale)


def apply_hilbert_colormap(
    depth: jax.Array,
    *,
    min_depth: float = 0.0,
    max_depth: float = 1.0,
    uint8: bool = False,
) -> jax.Array:
    """Normalize a depth image and map each pixel to the Hilbert-style RGB cube."""
    depth = jnp.asarray(depth, dtype=jnp.float32)
    cmap = lambda d: depth_to_hilbert_rgb(d, min_depth=min_depth, max_depth=max_depth)
    rgb = jax.vmap(jax.vmap(cmap))(depth)
    if uint8:
        return jnp.round(rgb * 255.0).astype(jnp.uint8)
    return rgb


__all__ = ["apply_hilbert_colormap", "depth_to_hilbert_rgb"]
