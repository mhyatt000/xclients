from __future__ import annotations

from dataclasses import dataclass, field
import datetime

# from evdev import ecodes, InputDevice
import logging
from pathlib import Path

import cv2
import jax
import numpy as np
from rich import print
from tqdm import tqdm
import tyro
from webpolicy.client import Client

from xclients.core.cfg import Config, spec
from xclients.triangulate import lift_hand_pnp

log = logging.getLogger(__name__)

NJOINTS = 21  # MANO / WiLoR hand


def waitq(ms: int) -> bool:
    return (_key := cv2.waitKey(ms)) == ord("q")


def display(cfg: Config):
    eps = list(cfg.dir.glob("ep*.npz"))
    wait = int(1000 / cfg.fps)  # convert fps to ms
    firsts = []
    for ep in tqdm(eps, leave=False):
        try:
            data = np.load(ep)
        except Exception as e:
            print(f"Error loading {ep}: {e}")
            ep.unlink()  # remove the file
            continue
        data = dict(data.items())

        raise NotImplementedError("fix me")
        # n = len(data[list(data.keys())[0]])

        # pprint(spec(data))
        for i in tqdm(range(n), leave=False):
            steps = {k: v[i] for k, v in data.items()}
            all_imgs = np.concatenate(list(steps.values()), axis=1)
            if i == 0:
                firsts.append(all_imgs)
            cv2.imshow("frame", recolor(all_imgs))
            if cfg.show_first_only:
                break

            if waitq(wait):
                break
        if waitq(wait):
            break

    _f = 255 - np.array(firsts)[..., -1:].std(0).astype(np.uint8)  # [...,-1:]
    _f = 255 - _f
    _f = np.clip(_f**1.4, 0, 255)
    # normalize the view of the std img
    # _f = (_f - _f.mean()) /(_f.std() + 1e-6)
    # _f = _f  *(_f.std() + 1e-6)
    cv2.imshow("std", recolor(_f.astype(np.uint8)))
    # save the std image
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = str(f"std_{now}.png")
    cv2.imwrite(out, _f)
    cv2.waitKey(0)


@dataclass
class Episode:
    path: Path
    data: dict = field(init=False)

    def from_npz(path: Path) -> Episode:
        data = np.load(path)
        data = dict(data.items())
        ep = Episode(path)
        ep.data = data
        return ep

    def __len__(self):
        # assume all data entries have the same length
        first = next(iter(self.data))
        return len(self.data[first])

    @property
    def shapes(self):
        return {k: v.shape for k, v in self.data.items()}

    def __getitem__(self, i: int):
        return {k: v[i] for k, v in self.data.items()}

    def get(self, key: str):
        return self.data[key]

    def __rich_repr__(self):
        r = {"path": self.path, "shapes": self.shapes}
        yield from r.items()


@dataclass
class Prediction:
    pass


@dataclass
class WilorP(Prediction):
    betas: np.ndarray
    pose: np.ndarray

    global_orient: np.ndarray
    cam: np.ndarray
    cam_t_full: np.ndarray

    kp2d: np.ndarray
    kp3d: np.ndarray
    vertices: np.ndarray
    scaled_focal_length: float


@dataclass
class ManoP(Prediction):
    bbox: np.ndarray
    is_right: float
    wilor: WilorP

    def from_prediction(d: dict) -> ManoP:
        w = jax.tree.map(lambda x: x[0] if isinstance(x, np.ndarray) and x.ndim > 1 else x, d["wilor_preds"])
        w = {k.replace("pred_", ""): v for k, v in w.items()}
        w = {k.replace("keypoints_", "kp"): v for k, v in w.items()}
        w["pose"] = w.pop("hand_pose")

        mp = ManoP(bbox=np.array(d["hand_bbox"]), is_right=d["is_right"], wilor=WilorP(**w))


@dataclass
class PrepConfig(Config):
    dir: Path  # data directory
    fps: int = 30
    show_first_only: bool = False


def make_intrinsics(w: int = 640, h: int = 480, fx: float = 515.0, fy: float = 515.0) -> np.ndarray:
    """Pinhole intrinsics K with a centered principal point. PnP needs no extrinsics."""
    cx, cy = w / 2, h / 2
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def extract_kp3d_step(step: dict, *, client: Client, K: np.ndarray) -> dict:
    """Per-camera single-view hand lift for one timestep.

    step: {cam_name: frame[H, W, 3] uint8 RGB}. Runs WiLoR per view and places
    each hand into that camera's own frame via lift_hand_pnp (no cross-view
    fusing). Returns step augmented with per-camera kp3d_cam / kp2d / visible;
    no-hand views are zero-filled and flagged invisible rather than dropped.
    """
    kp3d_cam, kp2d_all, visible = {}, {}, {}
    for cam, img in step.items():
        out = client.step({"image": img})  # recordings are RGB; wilor expects RGB
        hand = (out.get("hands") or [None])[0]
        if hand and "keypoints_2d" in hand:
            kp2d = np.asarray(hand["keypoints_2d"], dtype=np.float64)  # [21, 2]
            kp3d_rel = np.asarray(hand["keypoints_3d"], dtype=np.float64)  # [21, 3] wrist-rel
            placed, _R, _t = lift_hand_pnp(kp2d, kp3d_rel, K)  # [21, 3] in cam frame
            kp3d_cam[cam], kp2d_all[cam], visible[cam] = placed, kp2d, True
        else:
            # no hand this cam/frame: keep the frame, mask it, fill placeholders
            kp3d_cam[cam] = np.zeros((NJOINTS, 3))
            kp2d_all[cam] = np.zeros((NJOINTS, 2))
            visible[cam] = False
    return dict(step) | {"kp3d_cam": kp3d_cam, "kp2d": kp2d_all, "visible": visible}


def main(cfg: PrepConfig):
    # Per-camera single-view PnP. Cameras move between episodes, so instead of
    # triangulating across views we lift each camera independently with
    # lift_hand_pnp. Output keeps all cameras grouped per frame (no fusing): one
    # hand per camera in that camera's own frame, plus a per-camera visibility
    # flag so no-hand frames are masked rather than dropped.
    client = Client(host=cfg.host, port=cfg.port)
    eps = list(cfg.dir.glob("ep*.npz"))
    K = make_intrinsics(w=640, h=480)

    for ep_path in tqdm(eps):
        ep = Episode.from_npz(ep_path)
        steps = [extract_kp3d_step(dict(s), client=client, K=K) for s in tqdm(ep, leave=False)]
        episode = jax.tree.map(lambda *x: np.stack(x, axis=0), *steps)
        print(spec(episode))
        yield episode


if __name__ == "__main__":
    main(tyro.cli(PrepConfig))
