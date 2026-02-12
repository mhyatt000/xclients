import os
import sys

import tyro
from rich import print
from server import Config, Policy
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
