#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="$SCRIPT_DIR/.local_libs:${LD_LIBRARY_PATH:-}"
export OMNI_KIT_ACCEPT_EULA=yes
export UV_CACHE_DIR=/tmp/uv-cache

uv run python render_cube.py "$@"
