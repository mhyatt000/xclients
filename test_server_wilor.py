from plugins.server_wilor.main import load
import numpy as np

# load plugin
policy = load()
print("Loaded:", type(policy))

# create dummy image
dummy = np.zeros((480, 640, 3), dtype=np.uint8)

# test step()
result = policy.step({"image": dummy})
print("Result:", result)

# test with a real image
import cv2

img_path = "/home/lliao2/WiLoR/demo_img/test1.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("\nTesting with REAL image:", img_path)
real_result = policy.step({"image": img})
print("Real Image Result:", real_result)