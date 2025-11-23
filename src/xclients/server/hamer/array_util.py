from collections import deque

import numpy as np


def stack_and_pad(history: deque, num_obs: int):
    """
    Converts a list of observation dictionaries (`history`) into a single observation dictionary
    by stacking the values. Adds a padding mask to the observation that denotes which timesteps
    represent padding based on the number of observations seen so far (`num_obs`).
    """
    horizon = len(history)
    full_obs = {k: np.stack([dic[k] for dic in history]) for k in history[0]}
    pad_length = horizon - min(num_obs, horizon)
    timestep_pad_mask = np.ones(horizon)
    timestep_pad_mask[:pad_length] = 0
    full_obs["timestep_pad_mask"] = timestep_pad_mask
    return full_obs


def keyp2bbox(keyp):
    """get bounding box from keypoints
    the boxes are returned in the format [x1, y1, x2, y2]
    """
    valid = keyp[:, 2] > 0.5
    if sum(valid) > 3:  # if more than 3 keypoints have confidence > 0.5?
        bbox = [
            keyp[valid, 0].min(),
            keyp[valid, 1].min(),
            keyp[valid, 0].max(),
            keyp[valid, 1].max(),
        ]
        return bbox
    return None
