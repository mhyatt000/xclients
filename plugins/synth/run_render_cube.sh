#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="$(pwd)/.local_libs:${LD_LIBRARY_PATH:-}"
export OMNI_KIT_ACCEPT_EULA=yes
export UV_CACHE_DIR=/tmp/uv-cache

uv run python render_cube.py
