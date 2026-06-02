from __future__ import annotations

import logging

import numpy as np
import torch
from webpolicy.client import Client

from xclients.dream_dr.config import Config, Record
from xclients.dream_dr.records import load_first_array
from xclients.dream_dr.roboreg import ensure_plugin_src, repo_root


def dream_joints(joints: np.ndarray, units: str) -> np.ndarray:
    if units == "deg":
        return np.rad2deg(joints).astype(np.float32)
    if units == "rad":
        return joints.astype(np.float32)
    raise ValueError(f"Unsupported dream_joint_units={units}")


def collect_dream_pose(cfg: Config, records: list[Record], masks: np.ndarray) -> dict:
    cache = cfg.output_dir / "dream_outputs.npz"
    if cache.exists() and not cfg.refresh_cache:
        with np.load(cache, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    client = Client(cfg.dream_host, cfg.dream_port)
    images = np.stack([record.image for record in records]).astype(np.uint8)
    joints = np.stack([dream_joints(record.joints, cfg.dream_joint_units) for record in records])
    intrinsics = np.stack([record.intrinsics for record in records]).astype(np.float32)

    out = client.step({"image": images, "q": joints, "K": intrinsics, "mask": masks})
    if "w2c" not in out:
        raise KeyError(f"Dream response has no w2c key. Keys: {sorted(out)}")

    wanted = {
        key: np.asarray(value)
        for key, value in out.items()
        if key
        in {
            "w2c",
            "K",
            "pnp_success",
            "pnp_reproj_px",
            "mask_iou",
            "mask_iou_reject",
            "dr_success",
            "dr_loss_init",
            "dr_loss_final",
            "dr_cxy_delta",
        }
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **wanted)
    return wanted


def collect_initial_pose(cfg: Config, records: list[Record], masks: np.ndarray) -> dict:
    if cfg.call_dream:
        out = collect_dream_pose(cfg, records, masks)
        out["pose_source"] = np.asarray("dream")
        return out

    if cfg.extrinsics_path is not None:
        ht = load_first_array(cfg.extrinsics_path, ("w2c", "HT", "extrinsics"))
        if ht is None:
            raise ValueError(f"No w2c/HT/extrinsics found in {cfg.extrinsics_path}")
        return {"w2c": apply_w2c_adjustments(cfg, ht), "pose_source": np.asarray(str(cfg.extrinsics_path))}

    record, w2c = select_record_w2c(cfg, records, masks)
    if w2c is not None:
        logging.info("Using static initial w2c from record %s", record.path)
        return {"w2c": w2c, "pose_source": np.asarray(str(record.path))}
    raise ValueError("No record has w2c. Pass --extrinsics-path or use --call-dream to get initial extrinsics.")


def apply_w2c_adjustments(_cfg: Config, ht: np.ndarray) -> np.ndarray:
    return np.asarray(ht, dtype=np.float32).copy()


def select_record_w2c(cfg: Config, records: list[Record], masks: np.ndarray) -> tuple[Record, np.ndarray | None]:
    candidates = [(i, record, record.w2c) for i, record in enumerate(records) if record.w2c is not None]
    if not candidates:
        return records[0], None

    if cfg.record_w2c_index is not None:
        if cfg.record_w2c_index < 0 or cfg.record_w2c_index >= len(records):
            raise ValueError(f"record_w2c_index must be in [0, {len(records) - 1}], got {cfg.record_w2c_index}")
        record = records[cfg.record_w2c_index]
        if record.w2c is None:
            raise ValueError(f"Record {record.path} has no w2c")
        return record, apply_w2c_adjustments(cfg, record.w2c)

    if len(candidates) == 1:
        _, record, w2c = candidates[0]
        return record, apply_w2c_adjustments(cfg, w2c)

    try:
        return score_record_w2c(cfg, records, masks, candidates)
    except Exception:
        logging.exception("Failed to score record w2c candidates; falling back to first record w2c")
        _, record, w2c = candidates[0]
        return record, w2c


def score_record_w2c(
    cfg: Config,
    records: list[Record],
    masks: np.ndarray,
    candidates: list[tuple[int, Record, np.ndarray]],
) -> tuple[Record, np.ndarray]:
    """
    Optionally score record extrinsics by iou overlap with sam masks
    Best overlap is used as initial guess for RR
    """
    ensure_plugin_src()

    from server_roboreg.common import HydraConfig
    from server_roboreg.dr import render_cv_w2c
    from server_roboreg.render import Renderer, RendererConfig

    bundled_urdf = repo_root() / "plugins/server_roboreg/xarm7_standalone.urdf"
    hcfg = HydraConfig(
        ros_package=cfg.ros_package,
        xacro_path=cfg.xacro_path,
        urdf=cfg.urdf_path or bundled_urdf,
        root_link_name=cfg.root_link_name,
        end_link_name=cfg.end_link_name,
        collision_meshes=cfg.collision_meshes,
    )
    renderer = Renderer(
        hcfg,
        RendererConfig(batch_size=len(records)),
        height=masks[0].shape[0],
        width=masks[0].shape[1],
        intr=np.stack([record.intrinsics for record in records]).astype(np.float32)[0],
    )
    joints = torch.tensor(np.stack([record.joints for record in records]), dtype=torch.float32, device=renderer.device)
    mask_bin = masks > 0
    intr = torch.tensor(
        np.stack([record.intrinsics for record in records]).astype(np.float32)[0],
        dtype=torch.float32,
        device=renderer.device,
    )

    best_score = -1.0
    best_record = candidates[0][1]
    best_w2c = candidates[0][2]
    for index, record, w2c in candidates:
        adjusted_w2c = apply_w2c_adjustments(cfg, w2c)
        ht = np.repeat(adjusted_w2c[None], len(records), axis=0)
        render = render_cv_w2c(
            renderer,
            joints,
            torch.tensor(ht, dtype=torch.float32, device=renderer.device),
            intr,
            masks[0].shape[0],
            masks[0].shape[1],
        )
        # render: torch size B,W,H,C=1
        render_bin = render.detach().cpu().numpy()[..., 0] > 0.5  # np B,W,H
        intersection = np.logical_and(render_bin, mask_bin).sum()
        union = np.logical_or(render_bin, mask_bin).sum()
        render_area = render_bin.sum()
        mask_area = mask_bin.sum()
        area_ratio = render_area / float(mask_area) if mask_area > 0 else 0.0
        area_penalty = min(area_ratio, 1.0 / area_ratio) if area_ratio > 0.0 else 0.0
        iou = float(intersection / union) if union > 0 else 0.0
        score = iou * area_penalty
        logging.info(
            "record w2c candidate %d score=%.6f iou=%.6f render_area=%d mask_area=%d area_ratio=%.3f",
            index,
            score,
            iou,
            render_area,
            mask_area,
            area_ratio,
        )
        if score > best_score or (score == best_score and render_area > 0):
            best_score = score
            best_record = record
            best_w2c = adjusted_w2c
    return best_record, best_w2c


def assert_dream_pose(out: dict, n: int) -> np.ndarray:
    ht = np.asarray(out["w2c"], dtype=np.float32)
    if ht.shape != (4, 4) and ht.shape != (n, 4, 4):
        raise ValueError(f"Expected w2c shape (4, 4) or ({n}, 4, 4), got {ht.shape}")
    axes = (0, 1) if ht.shape == (4, 4) else (1, 2)
    bad = np.flatnonzero(~np.isfinite(ht).all(axis=axes))
    if bad.size:
        raise ValueError(f"Dream produced non-finite w2c for record indices {bad.tolist()}")
    return ht


def warn_w2c_consistency(records: list[Record]) -> None:
    poses = [(record, record.w2c) for record in records if record.w2c is not None]
    if len(poses) < 2:
        return
    base = poses[0][1]
    trans_deltas = []
    angle_deltas = []
    for _, pose in poses[1:]:
        delta = pose @ np.linalg.inv(base)
        trans_deltas.append(float(np.linalg.norm(delta[:3, 3])))
        cos_angle = np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        angle_deltas.append(float(np.rad2deg(np.arccos(cos_angle))))

    max_trans = max(trans_deltas, default=0.0)
    max_angle = max(angle_deltas, default=0.0)
    if max_trans > 0.05 or max_angle > 5.0:
        logging.warning(
            "Stored record w2c poses are inconsistent for a static camera: "
            "max relative translation %.3f m, max relative rotation %.1f deg. "
            "Treat them as per-frame DREAM/PnP estimates, not calibrated extrinsics.",
            max_trans,
            max_angle,
        )
