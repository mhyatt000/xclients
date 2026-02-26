from __future__ import annotations

from dataclasses import dataclass
import select
import sys
import termios
import tty
from typing import Any

import tyro
from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8080


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
        pressed: dict[str, bool] = {}
        while True:
            key = self._read_key()
            if key is None:
                return pressed
            pressed[key] = True

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

        if ch != "\x1b":
            return ch.lower()

        seq = ""
        for _ in range(2):
            readable, _, _ = select.select([self.fd], [], [], 0.0)
            if not readable:
                break
            seq += sys.stdin.read(1)

        return {
            "[A": "up",
            "[B": "down",
            "[C": "right",
            "[D": "left",
        }.get(seq)


def main(cfg: Config) -> None:
    policy = KeyboardPolicy()
    if not policy.enabled:
        raise RuntimeError("stdin is not a TTY. Run in an interactive terminal.")
    try:
        Server(policy, cfg.host, cfg.port).serve()
    finally:
        policy.close()


if __name__ == "__main__":
    main(tyro.cli(Config))
