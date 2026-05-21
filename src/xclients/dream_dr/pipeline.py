from __future__ import annotations

import logging

from xclients.dream_dr.config import Config
from xclients.dream_dr.outputs import print_inspect, save_outputs, write_image
from xclients.dream_dr.pose import assert_dream_pose, collect_initial_pose, warn_w2c_consistency
from xclients.dream_dr.records import apply_intrinsics_override, load_intrinsics_override, load_records
from xclients.dream_dr.roboreg import run_dr
from xclients.dream_dr.sam import collect_sam_masks


def main(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO)
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
    print(dr_out)
    save_outputs(cfg, records, masks, initial, dr_out)
    logging.info("Wrote outputs to %s", cfg.output_dir)
