from __future__ import annotations

import numpy as np

# change from ROS FLU (Forward, Left, Up) to opencv RDF (Right, Down, Forward)

FLU2RDF = np.array(
    [
        [0, 0, 1, 0],
        [-1, 0, 0, 0],  # -
        [0, -1, 0, 0],  # -
        [0, 0, 0, 1],
    ]
)

RDF2FLU = np.linalg.inv(FLU2RDF)

in2m = 0.0254
