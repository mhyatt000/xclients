from __future__ import annotations

import os

from rich import print
from server_sam3do.server import Config, Policy
import tyro
from webpolicy.server import Server


def main(cfg: Config):
    print(cfg)
    if cfg.policy.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.policy.device)

    policy = Policy(cfg.policy)
    server = Server(
        policy,
        host=cfg.host,
        port=cfg.port,
        metadata=None,
    )
    print(f"serving sam3dobjects on {cfg.host}:{cfg.port}")
    server.serve()


if __name__ == "__main__":
    main(tyro.cli(Config))
