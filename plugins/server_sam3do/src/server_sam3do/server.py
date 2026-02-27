from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging
import signal
from contextlib import contextmanager

import jax
import numpy as np
from omegaconf import OmegaConf
from rich import print
import torch
from webpolicy.base_policy import BasePolicy

from server_sam3do.inference import (
    BLACKLIST_FILTERS,
    check_hydra_safety,
    Inference,
    WHITELIST_FILTERS,
)

logging.basicConfig(level=logging.INFO)

class TimeoutException(Exception):
    """Raised when operation exceeds timeout"""
    pass

def timeout_handler(signum, frame):
    """Handler for timeout signal"""
    raise TimeoutException("Processing timeout exceeded.")

@contextmanager
def timeout_context(seconds):
    """context manager for enforcing a timeout on a block of code"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)

@dataclass
class PolicyConfig:
    ckpt: Path
    compile: bool = True
    seed: int = 42
    save_ply_path: str | None = None
    warmup_path: str | None = None  # TODO change to warmup:bool
    device: int | None = None

    def __post_init__(self):
        self.ckpt = Path(self.ckpt).expanduser().resolve()


@dataclass
class Config:
    port: int = 8003
    host: str = "0.0.0.0"
    timeout: float | None = None
    policy: PolicyConfig = field(default_factory=PolicyConfig)


class Policy(BasePolicy):
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        config_path = Path(cfg.ckpt) / "pipeline.yaml"
        config = OmegaConf.load(config_path)
        config.rendering_engine = "pytorch3d"
        config.compile_model = cfg.compile
        config.workspace_dir = str(Path(cfg.ckpt).parent)
        check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)

        self.inference = Inference(str(config_path), compile=cfg.compile)

        self.device = torch.device(
            cfg.device if cfg.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"sam3dobjects policy initialized on {self.device}")

        self.reset()

        # Warmup with zeros
        print("Starting warmup...")
        try:
            warmup_img = np.ones((480, 640, 3), dtype=np.uint8)
            warmup_mask = np.ones((480, 640), dtype=np.uint8)
            self.step({"image": warmup_img, "mask": warmup_mask})
            self.reset()
            print("Warmup complete")
        except Exception as e:
            print(f"Warmup failed: {e}")

        if cfg.warmup_path:
            import cv2

            warmup_img = cv2.imread(str(cfg.warmup_path))
            if warmup_img is not None:
                warmup_img = warmup_img[:, :, ::-1]
                h, w = warmup_img.shape[:2]
                mask = np.ones((h, w), dtype=np.uint8) * 255
                try:
                    self.step({"image": warmup_img, "mask": mask})
                    self.reset()
                    print("warmup from file complete")
                except Exception as e:
                    print(f"Warmup from file failed: {e}")


    def reset(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def _to_numpy(value, dtype=None) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        array = np.asarray(value)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    def _serialize_mesh(self, mesh) -> dict[str, np.ndarray]:
        if isinstance(mesh, dict) and "vertices" in mesh and "faces" in mesh:
            vertices = self._to_numpy(mesh["vertices"], np.float32)
            faces = self._to_numpy(mesh["faces"], np.int32)
            return {"vertices": vertices, "faces": faces}

        mesh_obj = None
        if isinstance(mesh, (list, tuple)) and len(mesh) > 0:
            mesh_obj = mesh[0]
        elif hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
            mesh_obj = mesh

        if mesh_obj is None or not hasattr(mesh_obj, "vertices") or not hasattr(mesh_obj, "faces"):
            raise TypeError(f"Unsupported mesh format for serialization: {type(mesh)}")

        vertices = self._to_numpy(mesh_obj.vertices, np.float32)
        faces = self._to_numpy(mesh_obj.faces, np.int32)
        return {"vertices": vertices, "faces": faces}

    def step(self, obs: dict) -> dict:
        """Process request with optional timeout"""
        # Extract timeout from payload, default to 30 seconds
        timeout = obs.get("timeout", 30.0)
        self.logger.info(f"Processing with timeout={timeout}s")
        
        try:
            with timeout_context(timeout):
                jax.tree.map(lambda x: np.array(x) if isinstance(x, list) else x, obs)
                image = obs["image"]
                mask = obs.get("mask")  # mask is optional

                if mask is None:
                    h, w = image.shape[:2]
                    mask = np.ones((h, w), dtype=np.uint8) * 255

                # image = self.inference.merge_mask_to_rgba(image, mask)

                with torch.no_grad():
                    result = self.inference(image, mask, seed=self.cfg.seed)

                if "gs" in result:
                    gs = result["gs"]
                    if self.cfg.save_ply_path:
                        gs.save_ply(self.cfg.save_ply_path)

                def spec(tree: dict):
                    return jax.tree.map(
                        lambda x: {"shape": x.shape, "dtype": x.dtype} if isinstance(x, torch.Tensor) else type(x), tree
                    )

                print(spec(result))
                print(result["gs"].aabb)

                for k in ["gaussian", "glb", "gs", "coords", "coords_original", "shape"]:
                    del result[k]
                if "mesh" in result:
                    result["mesh"] = self._serialize_mesh(result["mesh"])
                # tensor to numpy . cpu no gpu. detach
                return jax.tree.map(lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x, result)

        except TimeoutException:
            self.logger.error(f"Processing exceeded timeout of {timeout} seconds")
            return {"error": f"Processing timeout exceeded after {timeout}s"}
        except Exception as e:
            self.logger.error(f"Error during processing: {e}", exc_info=True)
            return {"error": str(e)}

class TryPolicy(BasePolicy):
    """Wraps a policy to handle exceptions gracefully"""
    def __init__(self, policy: BasePolicy):
        self.policy = policy
        self.logger = logging.getLogger(__name__)

    def step(self, obs: dict) -> dict:
        try:
            return self.policy.step(obs)
        except TimeoutException as e:
            self.logger.error(f"Tieout during processing: {e}")
            return {"error": str(e)}
        except Exception as e:
            self.logger.error(f"Error during processing: {e}", exc_info=True)
            return {"error": str(e)}

    def reset(self, *args, **kwargs) -> None:
        try:
            self.policy.reset(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error during reset: {e}", exc_info=True)



