from __future__ import annotations

from dataclasses import dataclass
import datetime
from enum import Enum
import logging
from pathlib import Path
import threading
import time

import cv2
from evdev import ecodes, InputDevice
import jax
import numpy as np
from rich.pretty import pprint
from tqdm import tqdm
import tyro

logger = logging.getLogger(__name__)


CAM_MAP = {
    "over": 0,
    "side": 2,
    "low": 4,
}


class MyCamera:
    def __repr__(self) -> str:
        return f"MyCamera(device_id={self.id})"

    def __init__(self, cam: int | cv2.VideoCapture, fps=30):
        self.id = cam
        self.cam = cv2.VideoCapture(cam) if isinstance(cam, int) else cam
        self.thread = None

        self.fps = fps
        self.freq = 5  # hz
        self.dt = 1 / self.fps

        self.img = None
        _, self.img = self.cam.read()
        self.start()
        time.sleep(0.1)

    def start(self):
        self._recording = True

        def _record():
            while self._recording:
                tick = time.time()

                ret, img = self.cam.read()
                if ret:
                    self.img = img

                toc = time.time()
                elapsed = toc - tick
                time.sleep(max(0, self.dt - elapsed))

        self.thread = threading.Thread(target=_record, daemon=True)
        self.thread.start()

    def stop(self):
        self._recording = False
        if self.thread is not None:
            self.thread.join()  # Wait for the thread to finish
            del self.thread

    def read(self):
        if self.img is None:
            return False, None
            raise RuntimeError("Camera not started or no image available.")
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        return True, img


class FootPedalRunner:
    def __init__(
        self,
        path="/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd",
        callback=None,
    ):
        self.device = InputDevice(path)
        self.device.grab()

        # Start a background thread to continuously read events
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()
        self.value = np.array([0, 0, 0])
        self.hz = 50

        self.pmap = {
            ecodes.KEY_A: 0,
            ecodes.KEY_B: 1,
            ecodes.KEY_C: 2,
        }

        self.callback = []

    def read(self):
        """
        Continuously reads events from the foot pedal device
        as 3-element array whenever a pedal's state changes.
        """

        for event in self.device.read_loop():
            if event.type == ecodes.EV_KEY and event.code in self.pmap:
                p = self.pmap[event.code]
                new = event.value  # 0=release, 1=press, 2=hold/repeat

                if changed := (self.value[p] != new):
                    self.value[p] = new

                    if self.callback:
                        self.callback(self.value)


class Mode(str, Enum):
    COLLECT = "collect"
    PLAY = "play"


@dataclass
class Config:
    dir: str  # path to the directory where the data will be saved

    episodes: int = 100  # number of episodes to record
    seconds: int = 15  # max seconds per episode
    fps: int = 50  # fps of data (not of the cameras)

    viz: bool = False  # show the camera images while recording
    cammap: bool = False  # assert that you checked the cam map with camera.py
    mode: Mode = Mode.COLLECT
    show_first_only: bool = False  # show only the first frame of each episode

    # ===
    # task: str = input("Task: ").lower()
    # base_dir: str = Path("~/data").expanduser()
    # time: str = time.strftime("%Y%m%d-%H%M%S")
    # env_name: str = f"xgym-mano-{task}-{time}"
    # data_dir: str = base_dir / env_name
    # nsteps: int = 300
    # nepisodes: int = 100
    # >>> 3bda70f (gym)

    def __post_init__(self):
        self.dir = Path(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        if not self.cammap:
            logger.error("Please check the camera mapping with camera.py before running this script.")

    @property
    def nsteps(self):
        return int(self.seconds * self.fps)


def spec(arr):
    return jax.tree.map(lambda x: x.shape, arr)


def _flush(episode: dict[list], ep: int, cfg: Config):
    episode = {k: np.array(v) for k, v in episode.items()}
    pprint(spec(episode))
    # quit()
    # episode = {CAM_MAP[k]: v for k, v in episode.items() if k in CAM_MAP}
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = str(cfg.dir / f"ep{ep}_{now}")


def flush(episode: dict, ep: int, cfg: RunCFG):
    print("what", len(episode))
    # episode = {CAM_MAP[k]: v for k, v in episode.items() if k in CAM_MAP}
    out = str(cfg.dir / f"ep{ep}")

    print(episode)
    np.savez(out, **episode)
    # cu.save_frames(v, str(cfg.dir / f"ep{ep}_{k}"), ext="mp4", fps=30)


def recolor(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def wait_for_pedal(pedal: FootPedalRunner, cams: dict[int, MyCamera], show: bool):
    pprint("press pedal to start recording")

    def border(img):
        """makes a red border around the image"""
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 255))
        return img

    while True:
        imgs = {k: cam.read() for k, cam in cams.items()}
        imgs = {k: v[1] for k, v in imgs.items() if v[0]}
        all_imgs = np.concatenate(list(imgs.values()), axis=1)

        if show:
            cv2.imshow("frame", border(recolor(all_imgs)))
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        p = pedal.value
        if p[0] == 1:
            pedal.value[0] = 0
            break


def main(cfg: Config):
    if cfg.mode == Mode.PLAY:
        eps = list(cfg.dir.glob("ep*.npz"))
        wait = int(1000 / cfg.fps)  # convert fps to ms
        firsts = []
        for ep in tqdm(eps, leave=False):
            print(ep)
            try:
                data = np.load(ep)
            except Exception as e:
                print(f"Error loading {ep}: {e}")
                ep.unlink()  # remove the file
                continue
            data = dict(data.items())

            print(data.keys())
            n = len(data[list(data.keys())[0]])

            # pprint(spec(data))
            for i in tqdm(range(n), leave=False):
                steps = {k: v[i] for k, v in data.items()}
                all_imgs = np.concatenate(list(steps.values()), axis=1)
                if i == 0:
                    firsts.append(all_imgs)
                cv2.imshow("frame", recolor(all_imgs))
                if cfg.show_first_only:
                    break

                if (key := cv2.waitKey(wait)) == ord("q"):
                    break
            if (key := cv2.waitKey(wait)) == ord("q"):
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
        quit()

    fps = cfg.fps
    dt = 1 / fps

    cams = {k: MyCamera(v) for k, v in CAM_MAP.items()}
    # cams = {k: cv2.VideoCapture(k) for k in CAM_MAP}
    print(cams)
    pedal = FootPedalRunner()
    time.sleep(2)

    wait_for_pedal(pedal, cams, True)

    print(cams)
    for ep in tqdm(range(cfg.episodes), leave=False):
        frames = {k: [] for k in cams}
        for step in tqdm(range(cfg.nsteps), leave=False):
            tic = time.time()

            imgs = {k: cam.read()[1] for k, cam in cams.items()}

            # mycamera is already in RGB format
            # imgs = {k:cv2.cvtColor(v, cv2.COLOR_RGB2BGR) for k,v in imgs.items()}
            # imgs = {k: cu.square(f) for k, f in imgs.items()}
            # imgs = {k: cv2.resize(f, (224, 224)) for k, f in imgs.items()}

            for k, v in imgs.items():
                frames[k].append(v)

            if cfg.viz:
                all_imgs = np.concatenate(list(imgs.values()), axis=1)
                cv2.imshow("frame", recolor(all_imgs))
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break

            if pedal.value[0] == 1:
                pedal.value[0] = 0
                time.sleep(0.1)
                break

            toc = time.time()
            elapsed = toc - tic
            time.sleep(max(0, dt - elapsed))
        flush(frames, ep, cfg)
        wait_for_pedal(pedal, cams, True)


if __name__ == "__main__":
    main(tyro.cli(Config))
