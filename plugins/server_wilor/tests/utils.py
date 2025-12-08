import pytest
import requests
import numpy as np
import cv2


def fetch_image(url: str = "https://raw.githubusercontent.com/warmshao/WiLoR-mini/9b93aaf/assets/hand.png"):
    """Fetch an image from a raw GitHub URL into a NumPy array.

    Raises pytest fail on HTTP errors or decode errors.
    """

    try:
        resp = requests.get(url, headers={"User-Agent": "pytest-fetch"}, timeout=10)
    except Exception as e:
        pytest.fail(f"Failed to request URL {url}: {e}")

    if resp.status_code != 200:
        pytest.fail(f"HTTP {resp.status_code} loading image: {url}")

    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        pytest.fail(f"OpenCV failed to decode image at {url}")

    return img

