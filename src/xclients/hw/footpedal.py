"""Foot pedal reader: step() -> which of the 3 pedals are on.

No ROS. A background thread reads the evdev device; step() returns the latest
per-pedal state and calls any hooks registered at init with that state. Default
is press-to-toggle per pedal; --deadman makes a pedal on only while held.

Standalone test:

    python -m xclients.hw.footpedal
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import logging
import threading
import time

from evdev import ecodes, InputDevice
import tyro
from webpolicy.base_policy import BasePolicy

PEDALS = {ecodes.KEY_A: "a", ecodes.KEY_B: "b", ecodes.KEY_C: "c"}
KEY_DOWN = 1  # evdev event.value: 0 release, 1 press, 2 autorepeat


@dataclass
class FootPedalConfig:
    path: str = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
    deadman: bool = False  # pedal on only while held, instead of press-to-toggle


class FootPedal(BasePolicy):
    """obs ignored -> {"a": bool, "b": bool, "c": bool}; hooks run on every step()."""

    def __init__(self, cfg: FootPedalConfig | None = None, hooks: Sequence[Callable[[dict], None]] = ()) -> None:
        self.cfg = cfg or FootPedalConfig()
        self.hooks = list(hooks)
        self.state = dict.fromkeys(PEDALS.values(), False)
        self._device = InputDevice(self.cfg.path)
        self._device.grab()  # keep pedal keystrokes out of the terminal
        threading.Thread(target=self._read, daemon=True, name="footpedal").start()

    def _read(self) -> None:
        for event in self._device.read_loop():
            if event.type != ecodes.EV_KEY or event.code not in PEDALS:
                continue
            pedal = PEDALS[event.code]
            if self.cfg.deadman:
                self.state[pedal] = event.value != 0
            elif event.value == KEY_DOWN:  # toggle on key-down only; release/autorepeat don't re-fire
                self.state[pedal] = not self.state[pedal]
            else:
                continue
            logging.info("pedal %s -> %s", pedal, self.state[pedal])

    def step(self, obs: dict) -> dict:
        state = dict(self.state)
        for hook in self.hooks:
            hook(state)
        return state

    def reset(self, payload: dict | None = None) -> None:
        self.state = dict.fromkeys(PEDALS.values(), False)

    def close(self) -> None:
        self._device.close()


def main(cfg: FootPedalConfig) -> None:
    live = {"live": False}
    pedal = FootPedal(cfg, hooks=[lambda s: live.update(live=s["b"])])
    logging.info("echoing pedal state (hook maps pedal b -> live); ctrl-c to exit")
    last = None
    try:
        while True:
            state = pedal.step({})
            if state != last:
                logging.info("step: %s live=%s", state, live["live"])
                last = state
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        pedal.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(FootPedalConfig))
