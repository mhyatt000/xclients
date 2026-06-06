from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from rich import print
from server_roboreg.common import HydraConfig
from server_roboreg.dr import DR
from server_roboreg.hydra import Hydra
from webpolicy.base_policy import BasePolicy


@dataclass
class RegConfig:
    host: str = "0.0.0.0"
    port: int = 8021

    hydra: HydraConfig = field(default_factory=HydraConfig)
    dr_end_link_name: str = "__all__"
    mode: Literal["icp", "dr", "both"] = "both"
    agg: Literal["online", "batch"] = "batch"


class RegistrationPolicy(BasePolicy):
    def __init__(self, cfg: RegConfig):
        print(cfg)
        self.cfg = cfg
        self.icp = Hydra(cfg.hydra) if cfg.mode in {"icp", "both"} else None

        self.dr_hydra = replace(cfg.hydra, end_link_name=cfg.dr_end_link_name)
        self.dr = DR(self.dr_hydra.dr, self.dr_hydra) if cfg.mode in {"dr", "both"} else None

    def step(self, obs: dict) -> dict:
        if self.cfg.agg == "online":
            raise NotImplementedError("Online aggregation is not implemented yet.")

        if self.cfg.mode == "icp":
            return {"HT": self._run_icp(obs)}
        if self.cfg.mode == "dr":
            return self.dr.step(obs)
        if self.cfg.mode == "both":
            ht = self._run_icp(obs)
            dr_obs = self._with_ht(obs, ht)
            return {"HT": self._run_dr(dr_obs)}

        raise ValueError(f"Unsupported registration mode: {self.cfg.mode}")

    def _run_icp(self, obs: dict):
        if self.icp is None:
            raise RuntimeError("ICP is not configured for this registration mode.")
        return self.icp._run_hydra(obs)

    def _run_dr(self, obs: dict):
        if self.dr is None:
            raise RuntimeError("DR is not configured for this registration mode.")
        out = self.dr.step(obs)
        return out["HT"]

    def _with_ht(self, obs: dict, ht) -> dict:
        return obs | {"HT": ht}
