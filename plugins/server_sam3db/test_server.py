# test_server.py
import os
import cv2
from pathlib import Path
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external"))
sys.path.insert(0, str(ROOT / "external" / "webpolicy" / "src"))

from plugins.server_sam3db.server import Sam3dBodyPolicy

os.environ["SAM3DB_ROOT"] = "/data/home/lliao2/sam3db/sam-3d-body"

policy = Sam3dBodyPolicy()

img = cv2.imread("/data/home/lliao2/xclients/plugins/server_sam3db/test.jpg")

payload = {
    "image": img
}

out = policy.step(payload, render=True)

if out["render"] is not None:
    cv2.imwrite("result.jpg", out["render"])
    print("Saved result.jpg")
else:
    print("No render output")