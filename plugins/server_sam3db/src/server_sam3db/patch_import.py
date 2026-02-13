from __future__ import annotations

from pathlib import Path

import pyrootutils

# follow sam3db demo.py style
demo = Path(__file__).parents[4] / "external/sam3db"
root = pyrootutils.setup_root(
    search_from=demo,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

# why did authors not use pyproject.toml ??

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.build_detector import HumanDetector
from tools.build_fov_estimator import FOVEstimator
from tools.build_sam import HumanSegmentor
from tools.vis_utils import visualize_sample_together

__all__ = [
    "FOVEstimator",
    "HumanDetector",
    "HumanSegmentor",
    "SAM3DBodyEstimator",
    "load_sam_3d_body",
    "visualize_sample_together",
]
