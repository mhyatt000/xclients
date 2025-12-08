import numpy as np
from server_wilor.demo import demo
from tests.utils import fetch_image


def test_demo_matches_reference():
    # Run your function
    img = fetch_image()
    demo(img)
