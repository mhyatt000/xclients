import numpy as np
import cv2
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]  # adjust if needed
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external"))
sys.path.insert(0, str(ROOT / "external" / "sam-3d-body"))
sys.path.insert(0, str(ROOT / "external" / "webpolicy" / "src"))

from plugins.server_sam3db.server import Sam3dBodyPolicy

os.environ["SAM3DB_ROOT"] = "/data/home/lliao2/sam3db/sam-3d-body"

policy = Sam3dBodyPolicy()

# ---- load episode ----
data = np.load("/data/home/lliao2/25dec10/ep0.npz")

view = "over"          # try: over / side / low
frames = data[view]    # (T, H, W, 3)

out_dir = Path("sam3db_npz_test") / view
out_dir.mkdir(parents=True, exist_ok=True)

# ---- run a few frames ----
for i in range(5):  # start with small number
    img = frames[i]

    payload = {
        "image": img
    }

    out = policy.step(payload, render=True)

    if out["render"] is not None:
        cv2.imwrite(str(out_dir / f"frame_{i:03d}.jpg"), out["render"])
        print(f"Saved frame {i}")
    else:
        print(f"No output for frame {i}")