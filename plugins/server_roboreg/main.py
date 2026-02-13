from __future__ import annotations

from server_roboreg.common import HydraConfig
from server_roboreg.hydra import Hydra
import tyro
from webpolicy.server import Server


def main(cfg: HydraConfig):
    policy = Hydra(cfg)
    server = Server(policy, cfg.host, cfg.port)
    server.serve()


if __name__ == "__main__":
    main(tyro.cli(HydraConfig))
