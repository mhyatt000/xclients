from __future__ import annotations

from dataclasses import dataclass
import logging

import cv2
import numpy as np
import webpolicy.client as webpolicy_client
from webpolicy.client import Client

from xclients.dream_dr.config import Config, Record
from xclients.dream_dr.outputs import write_image


@dataclass
class SamCandidate:
    mask: np.ndarray
    confidence: float
    index: int
    model_score: float | None = None


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


def _score_values(out: dict) -> np.ndarray:
    for key in ("scores", "score", "iou_scores", "confidences", "confidence"):
        if key in out:
            return np.asarray(out[key], dtype=np.float32).reshape(-1)
    return np.array([], dtype=np.float32)


def _binary_mask(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw)
    if arr.dtype == np.bool_:
        return arr
    if arr.size == 0:
        return np.zeros(arr.shape, dtype=bool)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    threshold = 0.5 if lo >= 0.0 and hi <= 1.0 else 0.0
    return arr > threshold


def sam_candidates(out: dict, shape: tuple[int, int], confidence: float) -> list[SamCandidate]:
    masks = out.get("masks")
    if masks is None:
        raise KeyError(f"SAM response has no masks key. Keys: {sorted(out)}")

    arr = np.asarray(masks)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[-2:] == shape:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported SAM masks shape {arr.shape}")

    scores = _score_values(out)
    candidates = []
    for index, raw in enumerate(arr):
        mask = _binary_mask(raw).astype(np.uint8) * 255
        if mask.shape != shape:
            mask = cv2.resize(mask, shape[::-1], interpolation=cv2.INTER_NEAREST)
        score = float(scores[index]) if index < scores.size else None
        candidates.append(SamCandidate(mask=mask, confidence=confidence, index=index, model_score=score))
    return candidates


def mask_from_sam(out: dict, shape: tuple[int, int]) -> np.ndarray:
    candidates = sam_candidates(out, shape, confidence=0.0)
    if not candidates:
        return np.zeros(shape, dtype=np.uint8)
    return np.any([candidate.mask > 0 for candidate in candidates], axis=0).astype(np.uint8) * 255


def bool_mask(mask: np.ndarray) -> np.ndarray:
    return (mask / 255) > 0.5


def cv_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255
    if arr.max(initial=0) <= 1:
        return (arr * 255).astype(np.uint8)
    return arr.astype(np.uint8)


def _confidence_values(cfg: Config) -> list[float]:
    values = cfg.sam_confidence_candidates or [cfg.sam_confidence]
    return [float(value) for value in values]


def _request_candidates(client: RawWebPolicyClient | Client, cfg: Config, record: Record) -> list[SamCandidate]:
    candidates = []
    for confidence in _confidence_values(cfg):
        out = client.step(
            {
                "type": "image",
                "image": record.image,
                "text": cfg.sam_prompt,
                "confidence": confidence,
            }
        )
        candidates.extend(sam_candidates(out, record.image.shape[:2], confidence))
    return candidates


def _clean_mask(mask: np.ndarray, cfg: Config) -> np.ndarray:
    out = mask > 0
    if cfg.sam_min_component_area > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), connectivity=8)
        keep = np.zeros_like(out)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= cfg.sam_min_component_area:
                keep |= labels == label
        out = keep
    if cfg.sam_close_kernel_size > 0:
        kernel = np.ones((cfg.sam_close_kernel_size, cfg.sam_close_kernel_size), dtype=np.uint8)
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    return out.astype(np.uint8) * 255


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    region = mask > 0
    if np.any(region):
        overlay = np.zeros_like(out)
        overlay[:, :] = color
        out[region] = cv2.addWeighted(out[region], 0.45, overlay[region], 0.55, 0)
    return out


def _raster_priors(cfg: Config, records: list[Record]) -> dict[str, np.ndarray]:
    posed = [record for record in records if record.w2c is not None]
    if not posed:
        return {}

    try:
        import torch

        from xclients.dream_dr.roboreg import ensure_plugin_src, repo_root

        ensure_plugin_src()
        from server_roboreg.common import HydraConfig
        from server_roboreg.dr import render_cv_w2c
        from server_roboreg.render import Renderer, RendererConfig

        h, w = posed[0].image.shape[:2]
        hcfg = HydraConfig(
            urdf=cfg.urdf_path or repo_root() / "plugins/server_roboreg/xarm7_standalone.urdf",
            root_link_name=cfg.root_link_name,
            end_link_name=cfg.end_link_name,
            collision_meshes=cfg.collision_meshes,
        )
        renderer = Renderer(
            hcfg,
            RendererConfig(batch_size=len(posed)),
            height=h,
            width=w,
            intr=np.stack([record.intrinsics for record in posed]).astype(np.float32)[0],
        )
        joints = torch.tensor(np.stack([record.joints for record in posed]), dtype=torch.float32, device=renderer.device)
        w2c = torch.tensor(np.stack([record.w2c for record in posed]), dtype=torch.float32, device=renderer.device)
        intr = torch.tensor(
            np.stack([record.intrinsics for record in posed]).astype(np.float32),
            dtype=torch.float32,
            device=renderer.device,
        )
        render = render_cv_w2c(renderer, joints, w2c, intr, h, w).detach().cpu().numpy()[..., 0]
        return {record.stem: (mask > 0.5).astype(np.uint8) * 255 for record, mask in zip(posed, render, strict=True)}
    except Exception:
        logging.exception("Failed to render SAM raster priors; falling back to SAM-only candidate scoring")
        return {}


def _candidate_score(candidate: SamCandidate, prior: np.ndarray | None, cfg: Config) -> float:
    mask = candidate.mask > 0
    area_ratio = float(mask.mean())
    if area_ratio <= 0.0:
        return -1.0
    if area_ratio < cfg.sam_min_area_ratio or area_ratio > cfg.sam_max_area_ratio:
        return -0.5 * area_ratio

    model_score = candidate.model_score if candidate.model_score is not None else 0.0
    if prior is None or not np.any(prior > 0):
        return area_ratio + 0.05 * model_score

    prior_mask = prior > 0
    intersection = float(np.logical_and(mask, prior_mask).sum())
    union = float(np.logical_or(mask, prior_mask).sum())
    iou = intersection / union if union > 0 else 0.0
    prior_area = float(prior_mask.sum())
    mask_area = float(mask.sum())
    ratio = mask_area / prior_area if prior_area > 0 else 0.0
    area_penalty = min(ratio, 1.0 / ratio) if ratio > 0.0 else 0.0
    return iou * area_penalty + 0.05 * model_score


def _write_candidate_qa(
    cfg: Config,
    record: Record,
    candidates: list[SamCandidate],
    chosen: SamCandidate,
    mask: np.ndarray,
    prior: np.ndarray | None,
) -> None:
    qa_dir = cfg.output_dir / "masks_qa" / record.stem
    for candidate in candidates:
        stem = f"c{candidate.confidence:g}_{candidate.index:02d}"
        write_image(qa_dir / f"{stem}_mask.png", candidate.mask)
        write_image(qa_dir / f"{stem}_overlay.png", _overlay(record.image, candidate.mask, (0, 255, 255)))

    if prior is not None:
        write_image(qa_dir / "raster_prior.png", prior)
        write_image(qa_dir / "raster_prior_overlay.png", _overlay(record.image, prior, (255, 0, 255)))

    write_image(qa_dir / "chosen_mask.png", mask)
    write_image(qa_dir / "chosen_overlay.png", _overlay(record.image, mask, (0, 255, 0)))
    logging.info(
        "Selected SAM mask for %s confidence=%.3f candidate=%d model_score=%s area_ratio=%.4f",
        record.stem,
        chosen.confidence,
        chosen.index,
        "none" if chosen.model_score is None else f"{chosen.model_score:.4f}",
        float((mask > 0).mean()),
    )


def _select_mask(cfg: Config, record: Record, candidates: list[SamCandidate], prior: np.ndarray | None) -> np.ndarray:
    if not candidates:
        raise ValueError(f"SAM returned no mask candidates for {record.stem}")

    scored = [(_candidate_score(candidate, prior, cfg), candidate) for candidate in candidates]
    _, chosen = max(scored, key=lambda item: item[0])
    mask = _clean_mask(chosen.mask, cfg)
    _write_candidate_qa(cfg, record, candidates, chosen, mask, prior)
    return mask


def collect_sam_masks(cfg: Config, records: list[Record]) -> np.ndarray:
    masks, _ = collect_sam_masks_with_records(cfg, records)
    return masks


def collect_sam_masks_with_records(cfg: Config, records: list[Record]) -> tuple[np.ndarray, list[Record]]:
    mask_dir = cfg.output_dir / "masks"
    masks = []
    kept_records = []
    missing = []
    for record in records:
        path = mask_dir / f"{record.stem}_mask.png"
        if path.exists() and not cfg.refresh_cache:
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Failed to read cached mask {path}")
            masks.append(mask)
            kept_records.append(record)
        else:
            missing.append(record)

    if missing:
        client = (
            RawWebPolicyClient(cfg.sam_host, cfg.sam_port)
            if cfg.sam_raw_webpolicy
            else Client(cfg.sam_host, cfg.sam_port)
        )
        priors = _raster_priors(cfg, missing)
        for record in missing:
            logging.info("Requesting SAM mask for %s", record.stem)
            candidates = _request_candidates(client, cfg, record)
            if not candidates:
                logging.warning("SAM returned no mask candidates for %s; excluding this frame from DR", record.stem)
                continue
            mask = _select_mask(cfg, record, candidates, priors.get(record.stem))

            write_image(mask_dir / f"{record.stem}_mask.png", mask)
            masks.append(mask)
            kept_records.append(record)

    if not masks:
        raise ValueError("SAM produced no usable masks for any selected records")
    return np.stack(masks).astype(np.uint8), kept_records
