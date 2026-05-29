from __future__ import annotations

from dataclasses import replace
import logging
import re

import cv2
import numpy as np
from webpolicy.client import Client

from xclients.dream_dr.config import Config, Record
from xclients.dream_dr.dream_arrayrecord import (
    camera_available,
    default_intrinsics,
    discover_shard_cameras,
    dream_response_to_arrays,
    iter_arrayrecord_shards,
    sample_image,
    sample_joints_deg,
    sample_joints_rad,
    selected_shard_indices,
)
from xclients.dream_dr.outputs import print_inspect, save_outputs, write_image
from xclients.dream_dr.pose import assert_dream_pose, collect_initial_pose, warn_w2c_consistency
from xclients.dream_dr.records import apply_intrinsics_override, load_intrinsics_override, load_records
from xclients.dream_dr.roboreg import run_dr
from xclients.dream_dr.sam import collect_sam_masks, collect_sam_masks_with_records


def main(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO)
    if cfg.arrayrecord_dir is not None:
        main_arrayrecord(cfg)
        return

    records = load_records(cfg)
    apply_intrinsics_override(records, load_intrinsics_override(cfg))
    print_inspect(records, cfg)
    warn_w2c_consistency(records)
    if cfg.inspect:
        return

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        write_image(cfg.output_dir / "images" / f"{record.stem}_image.png", record.image)

    masks = collect_sam_masks(cfg, records)
    initial = collect_initial_pose(cfg, records, masks)
    ht = assert_dream_pose(initial, len(records))

    dr_out = run_dr(cfg, records, masks, ht) if cfg.run_dr else None
    if dr_out is None:
        logging.info("DR output: None")
    else:
        logging.info("DR output generated %s", type(dr_out).__name__)
    save_outputs(cfg, records, masks, initial, dr_out)
    logging.info("Wrote outputs to %s", cfg.output_dir)


def main_arrayrecord(cfg: Config) -> None:
    ht_by_camera = {}
    client = None
    global_start = 0
    wrote_outputs = False

    for shard in iter_arrayrecord_shards(cfg):
        local_indices = selected_shard_indices(cfg, shard, global_start)
        if len(local_indices) == 0:
            global_start += shard.length
            continue

        cameras = discover_shard_cameras(cfg, shard, local_indices)
        if cfg.inspect:
            print(
                f"{shard.shard_id}: records={shard.length} selected={len(local_indices)} "
                f"global_start={global_start} cameras={cameras}"
            )
            global_start += shard.length
            continue
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        wrote_outputs = True
        logging.info(
            "Processing ArrayRecord shard %s (%d selected records, cameras=%s)",
            shard.shard_id,
            len(local_indices),
            cameras,
        )
        for camera in cameras:
            if client is None:
                client = Client(cfg.dream_host, cfg.dream_port)
            result = collect_shard_camera_dream(cfg, client, shard, global_start, local_indices, camera)
            if result is None:
                logging.info("Skipping camera %s in shard %s: no matching records", camera, shard.shard_id)
                continue

            records, dream_masks, initial = result
            dr_records, _ = select_dr_records(cfg, records, dream_masks)
            camera_id = safe_camera_id(camera)
            shard_id = safe_camera_id(shard.shard_id)
            out_key = f"{shard_id}__{camera_id}"
            cam_cfg = replace(cfg, output_dir=cfg.output_dir / "shards" / shard_id / camera_id)
            cam_cfg.output_dir.mkdir(parents=True, exist_ok=True)
            for record in dr_records:
                write_image(cam_cfg.output_dir / "images" / f"{record.stem}_image.png", record.image)
            dr_masks, dr_records = collect_sam_masks_with_records(cam_cfg, dr_records)
            logging.info("Using %d SAM-masked frames for roboreg DR", len(dr_records))

            ht = assert_dream_pose(initial, len(dr_records))
            dr_out = run_dr(cam_cfg, dr_records, dr_masks, ht) if cfg.run_dr else None
            save_outputs(cam_cfg, dr_records, dr_masks, initial, dr_out)
            ht_by_camera[out_key] = ht if dr_out is None else dr_out["HT"]

        global_start += shard.length

    if ht_by_camera:
        np.savez_compressed(
            cfg.output_dir / "HT_dr_cameras.npz",
            **{camera: np.asarray(ht, dtype=np.float32) for camera, ht in ht_by_camera.items()},
        )
    if wrote_outputs:
        logging.info("Wrote arrayrecord outputs to %s", cfg.output_dir)


def safe_camera_id(camera: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", camera).strip("_")
    return safe or "camera"


def select_dr_records(cfg: Config, records: list[Record], masks: np.ndarray) -> tuple[list[Record], np.ndarray]:
    max_records = int(cfg.arrayrecord_dr_max_records)
    if max_records <= 0 or len(records) <= max_records:
        return records, masks
    indices = np.linspace(0, len(records) - 1, max_records, dtype=np.int64)
    logging.info("Using %d/%d frames for roboreg DR", len(indices), len(records))
    return [records[int(index)] for index in indices], masks[indices]


def collect_shard_camera_dream(
    cfg: Config,
    client: Client,
    shard,
    global_start: int,
    local_indices: np.ndarray,
    camera: str,
) -> tuple[list[Record], np.ndarray, dict] | None:
    records = []
    masks = []
    calib_candidates = []
    frame_candidates = []
    batch_size = max(1, int(cfg.arrayrecord_batch_size))

    for start in range(0, len(local_indices), batch_size):
        batch_indices = local_indices[start : start + batch_size]
        batch_samples = [shard.read_sample(int(local_index)) for local_index in batch_indices]
        available = [
            (local_index, sample)
            for local_index, sample in zip(batch_indices, batch_samples, strict=True)
            if camera_available(sample, camera)
        ]
        if not available:
            continue

        images = np.stack([sample_image(sample, camera) for _, sample in available]).astype(np.uint8)
        dream_joints = np.stack([sample_joints_deg(sample) for _, sample in available]).astype(np.float32)
        dr_joints = np.stack([sample_joints_rad(sample) for _, sample in available]).astype(np.float32)
        if start == 0:
            logging.info("Using degree joints for DREAM and radian joints for roboreg rendering")
        h, w = images.shape[1:3]
        intrinsics = np.repeat(default_intrinsics(h, w, cfg.arrayrecord_focal_px)[None], len(available), axis=0)
        logging.info(
            "Calling DREAM for shard=%s camera=%s batch=%d:%d records=%d",
            shard.shard_id,
            camera,
            start,
            start + len(batch_indices),
            len(available),
        )
        out = dream_response_to_arrays(
            client.step(
                {
                    "image": images,
                    "type": "image",
                    "q": dream_joints,
                    "K": intrinsics.astype(np.float32),
                    "calibrate": True,
                }
            )
        )
        if "mask" not in out:
            raise KeyError(
                "DREAM response did not include mask. Start DREAM with mask returns enabled "
                "or use a checkpoint/config with a mask decoder."
            )

        batch_masks = normalize_mask_batch(out["mask"])
        if batch_masks.shape[0] != len(available):
            raise ValueError(f"DREAM returned {batch_masks.shape[0]} masks for {len(available)} records")
        batch_masks = resize_mask_batch(batch_masks, images.shape[1], images.shape[2])
        masks.extend(batch_masks)
        for row, ((local_index, _sample), image, joint, intr) in enumerate(
            zip(available, images, dr_joints, intrinsics, strict=True)
        ):
            global_index = global_start + int(local_index)
            records.append(
                Record(
                    stem=f"{safe_camera_id(shard.shard_id)}_{safe_camera_id(camera)}_{global_index:06d}",
                    path=shard.path,
                    image=image,
                    joints=joint,
                    intrinsics=intr,
                    w2c=None,
                )
            )
            if "w2c" in out:
                w2c = np.asarray(out["w2c"], dtype=np.float32)
                if w2c.shape == (len(available), 4, 4) and np.isfinite(w2c[row]).all():
                    frame_candidates.append(w2c[row])

        for key in ("calib_w2c", "HT", "extrinsics"):
            if key not in out:
                continue
            candidate = np.asarray(out[key], dtype=np.float32)
            if candidate.shape == (4, 4) and np.isfinite(candidate).all():
                calib_candidates.append(candidate)
            elif candidate.ndim == 3 and candidate.shape[-2:] == (4, 4):
                calib_candidates.extend([pose for pose in candidate if np.isfinite(pose).all()])
            break

    if not records:
        return None
    initial_candidates = calib_candidates or frame_candidates
    if not initial_candidates:
        raise ValueError(f"DREAM produced no finite initial pose for shard={shard.shard_id} camera={camera}")

    initial_ht = average_w2c(initial_candidates)
    return records, np.stack(masks).astype(np.uint8), {"w2c": initial_ht, "pose_source": np.asarray("dream_arrayrecord")}


def average_w2c(poses: list[np.ndarray]) -> np.ndarray:
    if len(poses) == 1:
        return np.asarray(poses[0], dtype=np.float32)
    from scipy.spatial.transform import Rotation

    arr = np.stack([np.asarray(pose, dtype=np.float64) for pose in poses])
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = Rotation.from_matrix(arr[:, :3, :3]).mean().as_matrix().astype(np.float32)
    out[:3, 3] = np.median(arr[:, :3, 3], axis=0).astype(np.float32)
    return out


def normalize_mask_batch(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 2:
        arr = arr[None]
    elif arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected DREAM mask batch shape (B,H,W), (B,H,W,1), or (B,1,H,W), got {arr.shape}")

    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.size and float(np.nanmax(arr)) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def resize_mask_batch(masks: np.ndarray, height: int, width: int) -> np.ndarray:
    arr = np.asarray(masks, dtype=np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"Expected mask batch shape (B,H,W), got {arr.shape}")
    if arr.shape[1:3] == (height, width):
        return arr
    return np.stack(
        [cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) for mask in arr],
        axis=0,
    ).astype(np.uint8)
