from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from xclients.dream_dr.config import Config, Record


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_plugin_src() -> None:
    script_dir = repo_root() / "scripts"
    sys.path = [path for path in sys.path if Path(path or ".").resolve() != script_dir]

    module = sys.modules.get("roboreg")
    module_file = Path(getattr(module, "__file__", "")).resolve() if module is not None else None
    if module_file == script_dir / "roboreg.py":
        sys.modules.pop("roboreg", None)

    plugin_src = repo_root() / "plugins/server_roboreg/src"
    if str(plugin_src) not in sys.path:
        sys.path.insert(0, str(plugin_src))


def run_dr(cfg: Config, records: list[Record], masks: np.ndarray, ht: np.ndarray) -> dict:
    ensure_plugin_src()
    from server_roboreg.common import DRConfig, HydraConfig, REGISTRATION_MODE
    from server_roboreg.dr import DR

    bundled_urdf = repo_root() / "plugins/server_roboreg/xarm7_standalone.urdf"
    hcfg = HydraConfig(
        ros_package=cfg.ros_package,
        xacro_path=cfg.xacro_path,
        urdf=cfg.urdf_path or bundled_urdf,
        root_link_name=cfg.root_link_name,
        end_link_name=cfg.end_link_name,
        collision_meshes=cfg.collision_meshes,
    )
    hcfg.dr = DRConfig(
        optimizer=cfg.dr_optimizer,
        lr=cfg.dr_lr,
        max_iterations=cfg.dr_max_iterations,
        step_size=cfg.dr_step_size,
        gamma=cfg.dr_gamma,
        mode=REGISTRATION_MODE(cfg.dr_mode),
    )

    payload = {
        "images": np.stack([record.image for record in records]).astype(np.uint8),
        "joints": np.stack([record.joints for record in records]).astype(np.float32),
        "mask": masks.astype(np.uint8),
        "intrinsics": np.stack([record.intrinsics for record in records]).astype(np.float32)[0],
        "HT": ht.astype(np.float32),
        "ht_is_cv_w2c": True,
    }
    return DR(hcfg.dr, hcfg).step(payload)
