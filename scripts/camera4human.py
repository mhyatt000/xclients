from __future__ import annotations

from dataclasses import dataclass
import datetime
from enum import Enum
import logging
from pathlib import Path
import sys
import termios
import threading
import time
import tty

import cv2
from evdev import ecodes, InputDevice
import jax
import numpy as np
from rich.pretty import pprint
from tqdm import tqdm
import tyro

logger = logging.getLogger(__name__)


CAM_MAP = {
    "side": 0,
    "low": 12,
    "left": 6,
    "brio": 2
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
        if not self.cam.isOpened():
            logger.error(f"Camera {cam} failed to open (device not accessible).")
            return
        ret, self.img = self.cam.read()
        if not ret or self.img is None:
            logger.error(f"Camera {cam} opened but failed to read first frame.")
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
    ):
        self.value = np.array([0, 0, 0])
        self.hz = 50
        self.callback = None

        self.pmap = {
            ecodes.KEY_A: 0,
            ecodes.KEY_B: 1,
            ecodes.KEY_C: 2,
        }

        # Try to connect to USB foot pedal; fall back to keyboard only
        self.device = None
        try:
            self.device = InputDevice(path)
            self.device.grab()
            threading.Thread(target=self._read_pedal, daemon=True).start()
            logger.info("Foot pedal connected.")
        except Exception as e:
            logger.warning(f"Foot pedal not found ({e}), using keyboard fallback.")

        threading.Thread(target=self._read_keyboard, daemon=True).start()

    def _read_pedal(self):
        for event in self.device.read_loop():
            if event.type == ecodes.EV_KEY and event.code in self.pmap:
                p = self.pmap[event.code]
                new = event.value  # 0=release, 1=press, 2=hold/repeat
                if new == 2:
                    continue
                if self.value[p] != new:
                    self.value[p] = new
                    if self.callback:
                        self.callback(self.value)

    def _read_keyboard(self):
        """Press 'p' to trigger pedal 0 (same as foot pedal press)."""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == "s":
                    self.value[0] = 1
                    if self.callback:
                        self.callback(self.value)
                elif ch in ("q", "\x03"):  # q or Ctrl+C
                    sys.exit(0)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


class Mode(str, Enum):
    COLLECT = "collect"
    PLAY = "play"


@dataclass
class Config:
    dir: str  # path to the directory where the data will be saved

    episodes: int = 20  # number of episodes to record
    fps: int = 50  # fps of data (not of the cameras)

    cammap: bool = False  # assert that you checked the cam map with camera.py
    mode: Mode = Mode.COLLECT
    show_first_only: bool = False  # show only the first frame of each episode

    def __post_init__(self):
        self.dir = Path(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        if not self.cammap:
            logger.error("Please check the camera mapping with camera.py before running this script.")


def spec(arr):
    return jax.tree.map(lambda x: x.shape, arr)


def flush(episode: dict, ep: int, cfg: RunCFG):
    out = str(cfg.dir / f"ep{ep}")

    np.savez(out, **episode)


def recolor(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def hstack(imgs: list[np.ndarray]) -> np.ndarray:
    """Horizontally stack images, resizing all to the same height as the first."""
    h = imgs[0].shape[0]
    resized = []
    for img in imgs:
        if img.shape[0] != h:
            w = int(img.shape[1] * h / img.shape[0])
            img = cv2.resize(img, (w, h))
        resized.append(img)
    return np.concatenate(resized, axis=1)


def wait_for_pedal(pedal: FootPedalRunner, cams: dict[int, MyCamera], show: bool):

    def border(img, color):
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=color)
        return img

    while True:
        imgs = {k: cam.read() for k, cam in cams.items()}
        imgs = {k: v[1] for k, v in imgs.items() if v[0]}
        all_imgs = hstack(list(imgs.values()))

        if show:
            cv2.imshow("frame", border(recolor(all_imgs), (0, 255, 0)))  # green = ready
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                sys.exit(0)
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

            for i in tqdm(range(n), leave=False):
                steps = {k: v[i] for k, v in data.items()}
                all_imgs = hstack(list(steps.values()))
                if i == 0:
                    firsts.append(all_imgs)
                cv2.imshow("frame", recolor(all_imgs))
                if cfg.show_first_only:
                    break

                if (key := cv2.waitKey(wait) & 0xFF) == ord("q"):
                    break
                all_imgs = hstack(list(steps.values()))
                if i == 0:
                    firsts.append(all_imgs)
                cv2.imshow("frame", recolor(all_imgs))
                if cfg.show_first_only:
                    break

                if (key := cv2.waitKey(wait) & 0xFF) == ord("q"):
                    break
            if (key := cv2.waitKey(wait) & 0xFF) == ord("q"):
                break

        _f = 255 - np.array(firsts)[..., -1:].std(0).astype(np.uint8)  # [...,-1:]
        _f = 255 - _f
        _f = np.clip(_f**1.4, 0, 255)

        # normalize the view of the std img
        cv2.imshow("std", recolor(_f.astype(np.uint8)))

        # save the std image
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = str(f"std_{now}.png")
        cv2.imwrite(out, _f)
        cv2.waitKey(0)
        quit()

    fps = cfg.fps
    dt = 1 / fps

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    cams = {k: MyCamera(v) for k, v in CAM_MAP.items()}
    print(cams)
    pedal = FootPedalRunner()
    time.sleep(2)

    wait_for_pedal(pedal, cams, True)

    print(cams)
    pprint("Press leftmost pedal or s to start and end recording")
    for ep in tqdm(range(cfg.episodes), leave=False):
        frames = {k: [] for k in cams}
        while pedal.value[0] == 0:
            tic = time.time()

            imgs = {k: cam.read()[1] for k, cam in cams.items()}

            for k, v in imgs.items():
                frames[k].append(v)

            all_imgs = hstack(list(imgs.values()))
            all_imgs = cv2.copyMakeBorder(all_imgs, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))  # red = recording
            cv2.imshow("frame", recolor(all_imgs))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                sys.exit(0)

            toc = time.time()
            elapsed = toc - tic
            time.sleep(max(0, dt - elapsed))

        pedal.value[0] = 0
        flush(frames, ep, cfg)
        wait_for_pedal(pedal, cams, True)


if __name__ == "__main__":
    main(tyro.cli(Config))
