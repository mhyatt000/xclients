from __future__ import annotations

from dataclasses import dataclass
import logging
import select
import sys
import termios
import tty
from typing import Any

import tyro
from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8080
    loop: bool = False  # whether to debug in loop, right here right now


class KeyboardPolicy(BasePolicy):
    def __init__(self) -> None:
        super().__init__()

        self.fd: int | None = None
        self.term_state: list[Any] | None = None
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.term_state = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

    @property
    def enabled(self) -> bool:
        return self.fd is not None

    def step(self, _obs: dict | None = None) -> dict[str, bool]:
        while True:
            pressed: dict[str, bool] = {}
            key = self._read_key()
            if key is None:
                return pressed
            pressed[key] = True
            print(pressed)
            return pressed

    def reset(self, *_args: object, **_kwargs: object) -> None:
        return None

    def close(self) -> None:
        if self.fd is None or self.term_state is None:
            return
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.term_state)
        self.fd = None
        self.term_state = None

    def _read_key(self) -> str | None:
        if self.fd is None:
            return None

        readable, _, _ = select.select([self.fd], [], [], 0.0)
        if not readable:
            return None

        ch = sys.stdin.read(1)
        if not ch:
            return None

        # [ or ] from up/down/left/right
        if ch in "[]":
            return ch + sys.stdin.read(1)
        if ch != "\x1b":
            return ch.lower()

        # is the following needed??

        seq = ""
        for _ in range(2):
            readable, _, _ = select.select([self.fd], [], [], 0.0)
            if not readable:
                break
            seq += sys.stdin.read(1)
        # print(seq)

        return {
            "[A": "up",
            "[B": "down",
            "[C": "right",
            "[D": "left",
        }.get(seq)


def main(cfg: Config) -> None:
    policy = KeyboardPolicy()
    try:
        if not policy.enabled:
            raise RuntimeError("stdin is not a TTY. Run in an interactive terminal.")

        if cfg.loop:
            while True:
                obs = policy.step()

        Server(policy, cfg.host, cfg.port).serve()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        policy.close()


if __name__ == "__main__":
    main(tyro.cli(Config))
