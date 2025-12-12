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

import xclients
from xclients.core import tf as xctf
from xclients.core.cfg import Config, spec
from xclients.triangulate import batch_triangulate

log = logging.getLogger(__name__)


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


def load_extr(name: str):
    p = xclients.ROOT / "data/cam" / name / "HT.npz"
    x = np.load(p)
    x = {k: x[k] for k in x}["HT"]
    x = xctf.RDF2FLU @ np.linalg.inv(x)
    return x


def main(cfg: PrepConfig):
    # display(cfg)

    client = Client(host=cfg.host, port=cfg.port)
    f = list(cfg.dir.glob("ep*.npz"))

    h, w = 480, 640
    cx, cy = w / 2, h / 2
    fx, fy = 515.0, 515.0
    intr = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    np.eye(4)[:3, :]  # dummy

    ep = Episode.from_npz(f[0])
    cameras = {
        k: {
            "intrinsics": intr,
            "extrinsics": load_extr(k),
            "height": h,
            "width": w,
        }
        for k in ep.data
    }

    P = {k: cameras[k]["intrinsics"] @ cameras[k]["extrinsics"][:3, :] for k in ep.data}

    for p in tqdm(f):
        ep = Episode.from_npz(p)
        _ep = []

        for s in tqdm(ep, leave=False):
            # print(spec(s))
            topayload = lambda x: {"image": x}
            outs = {k: client.step(topayload(v)) for k, v in s.items()}
            # s = s | out

            # print(spec(outs))
            k2ds = {k: v.get("wilor_preds", {}).get("pred_keypoints_2d") for k, v in outs.items()}

            if sum([p is not None for p in k2ds.values()]) >= 2:
                p = np.stack([P[k] for k in cameras if k2ds[k] is not None], axis=0)
                k2ds = np.array([k2ds[k] for k in cameras if k2ds[k] is not None]).reshape(len(p), -1, 2)
                # pin 100% confidence
                k2ds = np.concatenate([k2ds, np.ones((*k2ds.shape[:-1], 1))], axis=-1)
                # print(p.shape, k2ds.shape)

                k3ds = batch_triangulate(k2ds, p, min_views=2)
            else:
                # warning
                log.warning("Not enough views with 2D keypoints detected, skipping triangulation.")
                continue

            s = s | {"k3ds": k3ds}
            # print(spec(s))
            _ep.append(s)
        _ep = jax.tree.map(lambda *x: np.stack(x, axis=0), *_ep)
        print(spec(_ep))
        yield _ep


if __name__ == "__main__":
    main(tyro.cli(PrepConfig))
