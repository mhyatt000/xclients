from __future__ import annotations

import logging

from server_roboreg.server import RegConfig, RegistrationPolicy
import tyro
from webpolicy.server import Server


def main(cfg: RegConfig):
    logging.basicConfig(level=logging.INFO)
    policy = RegistrationPolicy(cfg)
    server = Server(policy, cfg.host, cfg.port)
    server.serve()


if __name__ == "__main__":
    main(tyro.cli(RegConfig))
