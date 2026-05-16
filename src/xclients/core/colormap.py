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

_DEPTH_LAMBDA = -3.0
_DEPTH_SCALE = 10.0 / 3.0


def _depth_to_cube_position(d: jax.Array | float) -> jax.Array:
    d = jnp.maximum(jnp.asarray(d, dtype=jnp.float32), 0.0)
    return 1.0 - (1.0 - d / (_DEPTH_LAMBDA * _DEPTH_SCALE)) ** (_DEPTH_LAMBDA + 1.0)


def _cube_position_to_hilbert_rgb(s: jax.Array | float) -> jax.Array:
    s = jnp.asarray(s, dtype=jnp.float32)
    x = jnp.clip(s, 0.0, 1.0) * (_HILBERT_CUBE_CORNERS.shape[0] - 1)
    i = jnp.minimum(x.astype(jnp.int32), _HILBERT_CUBE_CORNERS.shape[0] - 2)
    t = x - i.astype(jnp.float32)
    return (1.0 - t) * _HILBERT_CUBE_CORNERS[i] + t * _HILBERT_CUBE_CORNERS[i + 1]


def depth_to_hilbert_rgb(d: jax.Array | float) -> jax.Array:
    """Map metric depth to RGB along the Hilbert-style cube edge path."""
    return _cube_position_to_hilbert_rgb(_depth_to_cube_position(d))


def apply_hilbert_colormap(
    depth: jax.Array,
    *,
    uint8: bool = False,
) -> jax.Array:
    """Map a metric depth image to the Hilbert-style RGB cube."""
    depth = jnp.asarray(depth, dtype=jnp.float32)
    rgb = jax.vmap(jax.vmap(depth_to_hilbert_rgb))(depth)
    if uint8:
        return jnp.round(rgb * 255.0).astype(jnp.uint8)
    return rgb


__all__ = ["apply_hilbert_colormap", "depth_to_hilbert_rgb"]
