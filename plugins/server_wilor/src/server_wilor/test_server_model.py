# plugins/server_wilor/src/server_wilor/test_render_wilor.py

import cv2
import os
from server import WilorModel

TEST_IMAGE = "/data/home/lliao2/xclients/external/wilor/demo_img/test1.jpg"


def main():
    print("[TEST] Initializing WilorModel...")
    model = WilorModel()
    print("[OK] WilorModel initialized")

    print("[TEST] Loading image...")
    assert os.path.exists(TEST_IMAGE), "Test image not found"
    image = cv2.imread(TEST_IMAGE)
    assert image is not None, "cv2.imread failed"
    print("[OK] Image loaded")

    print("[TEST] Running step(render=True)...")
    result = model.step({"image": image}, render=True)

    assert result is not None, "step() returned None"
    assert "render" in result, "render not found in result"
    assert result["render"] is not None, "render is None"

    cv2.imwrite("RENDER_OK.jpg", result["render"])
    print("✅ Render finished, saved RENDER_OK.jpg")

    print("\n=== RENDER TEST PASSED ===")


if __name__ == "__main__":
    main()
