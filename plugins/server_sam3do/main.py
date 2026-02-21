from __future__ import annotations

import os

from rich import print
import logging
import tyro
from server_sam3do.server import Policy,  Config, TryPolicy
from webpolicy.server import Server

def main(cfg: Config):
    policy = Policy(cfg.policy)
    wrapped_policy = TryPolicy(policy)
    server = Server(wrapped_policy, cfg.host, cfg.port)
    logging.info(f"Starting server on {cfg.host}:{cfg.port} with policy {cfg.policy}")
    server.serve()

if __name__ == "__main__":
    main(tyro.cli(Config))
