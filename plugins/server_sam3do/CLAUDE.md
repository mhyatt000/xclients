# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`server-sam3do` is a web policy server plugin that provides inference capabilities for SAM3D Objects (3D segmentation and 3D reconstruction from single images). It's part of the larger `xclients` monorepo and follows the webpolicy server plugin architecture.

The server:
- Accepts images and optional masks as input
- Runs SAM3D inference to generate 3D representations
- Returns gaussian splatting and/or mesh outputs
- Can be run with GPU acceleration on specified CUDA device

## Architecture

### Core Components

- **`main.py`**: CLI entry point using `tyro` for configuration. Starts a `webpolicy.server.Server` with the Policy implementation.
- **`src/server_sam3do/server.py`**: Contains:
  - `Policy(BasePolicy)`: Main inference logic implementing the webpolicy interface. Loads checkpoint from disk, runs inference on images/masks, returns 3D representation outputs.
  - `PolicyConfig`: Dataclass for policy-specific configuration (checkpoint path, compilation flags, device, seed, optional warmup image).
  - `Config`: Dataclass combining `PolicyConfig` with server settings (host, port).

### External Dependencies

- **webpolicy**: Server framework (installed from git: `https://github.com/mhyatt000/webpolicy`)
- **SAM3D Inference**: Loaded from hardcoded path `/home/nhogg/sam-3d-objects`. The code adds this to `sys.path` and imports `Inference`, `check_hydra_safety`, and filter lists.
- **ML Stack**: torch, torchvision, torchaudio, pytorch3d, open3d, xformers, etc.
- **Config Management**: Hydra via `omegaconf` (loads `pipeline.yaml` from checkpoint directory)

### Key Dependencies

- Uses `uv` for package management (see `pyproject.toml`)
- Configuration-driven via `tyro.cli()` which parses dataclass configs into CLI arguments
- Heavy dependencies on computer vision and 3D reconstruction libraries

## Development Commands

### Setup & Installation

```bash
# Install dependencies with uv
uv sync

# Activate the virtual environment (if using standard venv)
source .venv/bin/activate
```

### Running the Server

```bash
# Basic run (uses defaults from PolicyConfig/Config dataclasses)
python main.py

# Run with custom configuration via CLI (tyro-based)
python main.py --policy.checkpoint-dir /path/to/checkpoints --policy.device 0 --port 8003 --host 0.0.0.0

# Run with warmup image to preload model
python main.py --policy.warmup-path /path/to/warmup/image.png

# Enable model compilation for performance
python main.py --policy.compile true
```

### Configuration

All configuration flows through the `Config` dataclass. Common options:
- `policy.checkpoint_dir`: Path to model checkpoint with `pipeline.yaml`
- `policy.device`: CUDA device ID (None defaults to auto-detect)
- `policy.compile`: Enable torch model compilation
- `policy.seed`: Random seed for inference
- `policy.save_ply_path`: Optional path to save PLY files
- `policy.warmup_path`: Optional image to warm up the model on startup
- `port`: Server port (default 8003)
- `host`: Server host (default 0.0.0.0)

## Policy Implementation Details

The `Policy.step()` method:
1. Expects `obs` dict with `image` key (required) and optional `mask` key
2. Converts numpy/list inputs to numpy arrays
3. If no mask provided, creates full white mask (255)
4. Merges mask to RGBA using `inference.merge_mask_to_rgba()`
5. Runs inference with `torch.no_grad()` for efficiency
6. Returns dict with `success=True`, `has_gaussian_splatting`, `has_mesh`, and optional `ply_saved_to`

## Important Notes

### External Paths

The code assumes SAM3D inference code exists at `/home/nhogg/sam-3d-objects`. This is a hardcoded path that will need adjustment for different environments.

### Hydra Safety

The server runs `check_hydra_safety()` on the pipeline config with whitelist/blacklist filters before initialization.

### GPU Memory

This is a heavy inference workload. The `Policy.warmup()` option can help pre-allocate memory on startup. The device parameter allows specifying which GPU to use when multi-GPU is available.

### Model Format

The model checkpoint directory must contain a `pipeline.yaml` config file. The rendering engine is hardcoded to `pytorch3d`.

## Related Plugins

Other server plugins in the `xclients/plugins/` directory follow similar patterns:
- `server_sam3`: SAM3 segmentation
- `server_sam3db`: SAM3 database variant
- `server_roboreg`: Robot registration
- `server_hamer`: Hand pose estimation

These all inherit from `BasePolicy` and use the same `webpolicy.server.Server` architecture.
