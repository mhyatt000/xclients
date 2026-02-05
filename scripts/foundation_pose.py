from __future__ import annotations

import logging
from dataclasses import dataclass 
from pathlib import Path 

import tyro
from xclients.core.cfg import Config
from webpolicy.client import Client 


@dataclass
class FPConfig(Config):
    """
    COnfiguration for FoundationPose Client.
    Inherits --host, --port, --show from Config.
    """
    rgb_dir: Path | None = None
    mask_path: Path | None = None
    mesh_path: Path | None = None
    confidence: float = 0.25
    debug: bool = True 

    def __post_init__(self):
        # Validate paths
        if self.rgb_dir is None:
            raise ValueError("--rgb-dir is required.")
        if not self.rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {self.rgb_dir}")
        
        if self.mask_path is None:
            raise ValueError("--mask-path is required.")
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {self.mask_path}")
            
        if self.mesh_path is None:
            raise ValueError("--mesh-path is required.")
        if not self.mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {self.mesh_path}")

def main(cfg: FPConfig) -> None:
    client = Client(cfg.host, cfg.port)
    
    logging.info(f"Connecting to FoundationPose Server at {cfg.host}:{cfg.port}")
    logging.info(f"Mesh: {cfg.mesh_path.name}")
    logging.info(f"Data: {cfg.rgb_dir.name}")

    payload = {
        "type": "track",
        "rgb_dir": str(cfg.rgb_dir.resolve()),
        "mask_path": str(cfg.mask_path.resolve()),
        "mesh_path": str(cfg.mesh_path.resolve()),
        "confidence": cfg.confidence,
        "debug": cfg.debug,
    }

    try:
        #response = client.step(payload)
        while True:
            response = client.step({"type": "next_frame"})
            if response.get("status") == "finished":
                break
            print(f"Pose: {response.get('pose')}")
        
        if not response:
            logging.error("No response received from server.")
            return

        if response.get("status") == "success":
            print("\nTracking Initialized Successfully")
            print(f"Frames Found: {response.get('frames')}")
            print(f"Initial Pose: \n{response.get('initial_pose')}")
        elif response.get("status") == "error":
            logging.error(f"Server Error: {response.get('message')}")
        else:
            print(f"Unknown Response: {response}")

    except Exception as e:
        logging.error(f"Communication failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(FPConfig))
