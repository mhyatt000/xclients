"""Single-view PnP hand lift over recorded episodes, visualised in rerun.

Formerly multi-view triangulation. Cameras now move between episodes, so instead
of triangulating across cameras we recover each hand *per camera* with
``cv2.solvePnP`` via ``xclients.triangulate.lift_hand_pnp`` (see that module for
the math). This script is the visual check: it plays recorded ``ep*.npz`` frames
through the wilor server and, for each frame, logs to rerun:

  - the camera image,
  - detected 2D joints (RED) from wilor,
  - the lifted 3D reprojected back to 2D (GREEN)  -> GREEN on RED == correct,
  - the lifted 3D hand (per camera, in that camera's own frame).

The per-frame reprojection error is also printed, so it is informative even with
no viewer.

Run order:
  1) start the wilor server (plugins/server_wilor/server.py), e.g. --port 8084
  2) python scripts/triangulate.py --ep /path/to/ep22.npz --port 8084
       --ep may be a single ep*.npz OR a directory containing several.
       headless (no display)? add  --rrd out.rrd  and open it in Rerun later.
"""

from __future__ import annotations
from dataclasses import dataclass
import tyro

import argparse
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
from webpolicy.client import Client
from xclients.core.cfg import Config

from xclients.triangulate import lift_hand_pnp, project_points

# MANO bone connectivity (wrist = 0), for drawing the 3D skeleton.
MANO_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
]


def _set_frame(t: int) -> None:
    """Set the rerun timeline cursor (compatible across rerun versions)."""
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("frame", t)
    else:
        rr.set_time("frame", sequence=t)


def episode_files(ep_arg: str) -> list[Path]:
    """--ep may be a single ep*.npz file or a directory of them."""
    p = Path(ep_arg)
    return sorted(p.glob("ep*.npz")) if p.is_dir() else [p]


def play(path: Path, client: Client, K: np.ndarray, stride: int) -> None:
    data = dict(np.load(path).items())
    cams = [k for k in data if np.asarray(data[k]).ndim == 4]  # camera-name -> [T, H, W, 3]
    T = len(data[cams[0]])
    print(f"episode {path.name}: {T} frames, cameras={cams}")

    for t in range(0, T, stride):
        _set_frame(t)
        for cam in cams:
            bgr = np.asarray(data[cam][t])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # wilor expects RGB

            out = client.step({"image": rgb})
            hand = (out.get("hands") or [None])[0]

            rr.log(f"cam/{cam}/image", rr.Image(bgr, color_model="BGR").compress(jpeg_quality=75))
            if not hand or "keypoints_2d" not in hand:
                continue  # no hand detected in this view this frame (expected sometimes)

            kp2d = np.asarray(hand["keypoints_2d"], dtype=np.float64)
            kp3d_rel = np.asarray(hand["keypoints_3d"], dtype=np.float64)

            kp3d_cam, R, tvec = lift_hand_pnp(kp2d, kp3d_rel, K)
            reproj = project_points(kp3d_cam, K)
            err = float(np.linalg.norm(reproj - kp2d, axis=1).mean())
            print(f"  t={t:4d} [{cam:4s}] reproj err {err:6.2f} px   depth {tvec[2]:.2f} m")

            # 2D overlay: GREEN (reprojected) should land on RED (detected)
            rr.log(f"cam/{cam}/image/detected", rr.Points2D(kp2d, colors=[255, 0, 0], radii=4))
            rr.log(f"cam/{cam}/image/reprojected", rr.Points2D(reproj, colors=[0, 255, 0], radii=2))
            # the lifted 3D hand, in this camera's own frame
            rr.log(f"cam/{cam}/hand3d", rr.Points3D(kp3d_cam, radii=0.004))
            rr.log(f"cam/{cam}/hand3d/bones", rr.LineStrips3D(kp3d_cam[np.array(MANO_PAIRS)]))


@dataclass
class MyConfig(Config):
    ep:bool = True # an ep*.npz file OR a directory of them
    stride:int = 5 # use every Nth frame
    rrd:str|None = None # save rerun log here instead of opening a viewer

def main(cfg:MyConfig) -> None:
    print('main.')
    print(cfg)

    if cfg.rrd:
        rr.init("triangulate")
        rr.save(cfg.rrd)
    else:
        rr.init("triangulate", spawn=True)
    # camera-frame convention: x-right, y-down, z-forward (OpenCV / what PnP uses)
    rr.log("/", rr.ViewCoordinates.RDF, static=True)

    H, W = 480, 640
    K = np.array([[515.0, 0.0, W / 2], [0.0, 515.0, H / 2], [0.0, 0.0, 1.0]])

    client = Client(host=cfg.host, port=cfg.port)  # waits until the server is up
    for path in episode_files(cfg.ep):
        play(path, client, K, cfg.stride)


if __name__ == "__main__":
    main(tyro.cli(MyConfig))
