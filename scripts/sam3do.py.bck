from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

import cv2
import numpy as np
from rich import print
import tyro
from webpolicy.client import Client

from xclients.core.cfg import Config, spec

class InputType(str, Enum):
    CAMERA = "camera"
    IMAGE = "image"

@dataclass
class SAMConfig(Config):
    """Config for SAM3 server"""
    host: str = "localhost"
    port: int = 8000
    prompt: str = "object"
    image_path: Path | None = None

@dataclass
class SAM3DoConfig(Config):
    """Config for SAM3Do server"""
    host: str = "localhost"
    port: int = 8003
    input_type: InputType = InputType.CAMERA
    camera_device: int = 0
    image_path: Path | None = None
    show: bool = True
    sam3: SAMConfig = field(default_factory=SAMConfig)
    timeout: float = 5.0

def load_image(image_path: Path) -> np.ndarray:
    """Load image from file"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return img

def get_frame(cfg: SAM3DoConfig) -> np.ndarray:
    """Get a frame from either camera or image file"""
    if cfg.input_type == InputType.CAMERA:
        cap = cv2.VideoCapture(cfg.camera_device)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame from camera {cfg.camera_device}")
        return frame
    else:  # InputType.IMAGE
        if cfg.image_path is None:
            raise ValueError("image_path must be provided when using IMAGE input type")
        return load_image(cfg.image_path)

def main(cfg: SAM3DoConfig) -> None:
    # Create clients
    sam3do_client = Client(cfg.host, cfg.port)
    sam3_client = Client(cfg.sam3.host, cfg.sam3.port)
    
    logging.info("Starting SAM3Do orchestration")
    
    while True:
        try:
            # Get frame from input source
            frame = get_frame(cfg)
            
            # Send to SAM3 first
            sam3_payload = {
                "image": frame,
                "type": "image",
                "text": cfg.sam3.prompt,
            }
            logging.info(f"Sending to SAM3: {spec(sam3_payload)}")
            sam3_out = sam3_client.step(sam3_payload)
            
            if not sam3_out:
                logging.error("Failed to get response from SAM3")
                continue
            
            logging.info(f"SAM3 output: {spec(sam3_out)}")
            
            # Send to SAM3Do with the frame and SAM3 output
            sam3do_payload = {
                "image": frame,
                "mask": frame[..., 0],
                "type": "image" ,
                "timeout": cfg.timeout,
            }
            logging.info(f"Sending to SAM3Do: {spec(sam3do_payload)}")
            sam3do_out = sam3do_client.step(sam3do_payload)
            
            if not sam3do_out:
                logging.error("Failed to read from SAM3Do")
                continue
            
            logging.info(f"SAM3Do output: {spec(sam3do_out)}")
            
            # Display if needed
            if cfg.show:
                cv2.imshow("Input Frame", frame)
                if cfg.input_type == InputType.IMAGE:
                    cv2.waitKey(0)  # Wait indefinitely for image input
                    break
                elif cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                
        except Exception as e:
            logging.error(f"Error: {e}")
            if cfg.input_type == InputType.IMAGE:
                break
            continue
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(SAM3DoConfig))
