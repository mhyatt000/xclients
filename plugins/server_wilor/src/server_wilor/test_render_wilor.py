import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from plugins.server_wilor.main import WilorPolicy, WilorConfig

cfg = WilorConfig()         
policy = WilorPolicy(cfg)






img = cv2.imread("external/wilor/demo_img/test1.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

out = policy.step({"image": img_rgb})

if out is None:
    print("❌ No hand detected")
    exit(0)

preds = out["wilor_preds"]
 
preds["pred_right"] = np.array([[out["is_right"]]], dtype=np.float32)

crop_info = out.get("crop_info", None)

overlay = policy.render_result(
    img_rgb,
    preds,
    crop_info,
)

overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
cv2.imwrite("RENDER_OK.jpg", overlay_bgr)

print("✅ Render finished, saved RENDER_OK.jpg")
