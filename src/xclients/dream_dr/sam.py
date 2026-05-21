from __future__ import annotations

from dataclasses import asdict
import logging

import cv2
import numpy as np
import webpolicy.client as webpolicy_client
from webpolicy.client import Client

from xclients.dream_dr.config import Config, Record
from xclients.dream_dr.outputs import write_image


class RawWebPolicyClient:
    def __init__(self, host: str, port: int) -> None:
        self.uri = f"ws://{host}:{port}"
        logging.info("Waiting for server at %s...", self.uri)
        self.ws = webpolicy_client.websockets.sync.client.connect(self.uri, compression=None, max_size=None)
        self.packer = webpolicy_client.msgpack_numpy.Packer()
        self.metadata = webpolicy_client.msgpack_numpy.unpackb(self.ws.recv())

    def step(self, obs: dict) -> dict:
        self.ws.send(self.packer.pack(obs))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        unpacked = webpolicy_client.msgpack_numpy.unpackb(response)
        if isinstance(unpacked, dict) and "action" in unpacked:
            return unpacked["action"]
        return unpacked


def mask_from_sam(out: dict, shape: tuple[int, int]) -> np.ndarray:
    masks = out.get("masks")
    if masks is None:
        raise KeyError(f"SAM response has no masks key. Keys: {sorted(out)}")

    arr = np.asarray(masks)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3:
        mask = np.any(arr > 0, axis=0)
    elif arr.ndim == 2:
        mask = arr > 0
    else:
        raise ValueError(f"Unsupported SAM masks shape {arr.shape}")

    out_mask = mask.astype(np.uint8) * 255
    if out_mask.shape != shape:
        out_mask = cv2.resize(out_mask, shape[::-1], interpolation=cv2.INTER_NEAREST)
    return out_mask


def bool_mask(mask: np.ndarray) -> np.ndarray:
    return (mask / 255) > 0.5


def cv_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255
    if arr.max(initial=0) <= 1:
        return (arr * 255).astype(np.uint8)
    return arr.astype(np.uint8)


def collect_sam_masks(cfg: Config, records: list[Record]) -> np.ndarray:
    mask_dir = cfg.output_dir / "masks"
    masks = []
    missing = []
    for record in records:
        path = mask_dir / f"{record.stem}_mask.png"
        if path.exists() and not cfg.refresh_cache:
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Failed to read cached mask {path}")
            masks.append(mask)
        else:
            missing.append(record)

    if missing:
        client = (
            RawWebPolicyClient(cfg.sam_host, cfg.sam_port)
            if cfg.sam_raw_webpolicy
            else Client(cfg.sam_host, cfg.sam_port)
        )
        for record in missing:
            logging.info("Requesting SAM mask for %s", record.stem)
            arm_out = client.step(
                {
                    "type": "image",
                    "image": record.image,
                    "text": cfg.sam_prompt,
                    "confidence": cfg.sam_confidence,
                }
            )
            mask = mask_from_sam(arm_out, record.image.shape[:2])

            write_image(mask_dir / f"{record.stem}_mask.png", mask)

    if missing:
        return collect_sam_masks(Config(**(asdict(cfg) | {"refresh_cache": False})), records)
    return np.stack(masks).astype(np.uint8)
