from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import viser
from viser.extras import ViserUrdf
import viser.transforms as vtf
import yourdfpy

from xclients.core.tf import FLU2RDF

REPO_ROOT = Path(__file__).resolve().parents[2]
XARM7_URDF = REPO_ROOT / "xarm7_standalone.urdf"

DXARM_L_POSITION = (0.0, -0.09687363, 0.14767363)
DXARM_L_WXYZ = (0.92387953, 0.38268343, 0.0, 0.0)
DXARM_R_POSITION = (0.0, 0.09687363, 0.14767363)
DXARM_R_WXYZ = (0.92387953, -0.38268343, 0.0, 0.0)


@dataclass
class UrdfAsset:
    path: Path
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def default_assets() -> dict[str, UrdfAsset]:
    return {
        "dxarm-l": UrdfAsset(XARM7_URDF, DXARM_L_POSITION, DXARM_L_WXYZ),
        "dxarm-r": UrdfAsset(XARM7_URDF, DXARM_R_POSITION, DXARM_R_WXYZ),
    }


class ViserWebUI:
    """Viser scene that displays URDF assets at fixed root poses.

    Joint configurations are pushed per-asset via `step({name: cfg})`;
    assets not mentioned keep their last configuration.
    """

    def __init__(self, assets: dict[str, UrdfAsset] | None = None, port: int = 8080) -> None:
        self.assets = default_assets() if assets is None else assets
        self.server = viser.ViserServer(port=port)
        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        self._urdf_vis: dict[str, ViserUrdf] = {}
        self._cfgs: dict[str, NDArray[np.float64]] = {}
        self._cameras: dict[str, viser.CameraFrustumHandle] = {}

        for name, asset in self.assets.items():
            root = f"/{name}"
            self.server.scene.add_frame(
                root,
                position=np.asarray(asset.position, dtype=np.float64),
                wxyz=np.asarray(asset.wxyz, dtype=np.float64),
                show_axes=False,
            )
            urdf = yourdfpy.URDF.load(asset.path)
            vis = ViserUrdf(self.server, urdf, root_node_name=root)
            self._urdf_vis[name] = vis
            self._cfgs[name] = np.zeros(urdf.num_actuated_joints)
            vis.update_cfg(self._cfgs[name])

    def add_camera(
        self,
        name: str,
        world_from_cam_flu: NDArray[np.floating],
        fov: float,
        aspect: float,
        scale: float = 0.15,
        image: NDArray[np.uint8] | None = None,
        jpeg_quality: int = 70,
    ) -> viser.CameraFrustumHandle:
        """Add a camera frustum posed by a camera-FLU -> world 4x4 (repo HT convention)."""
        world_from_cam_rdf = np.asarray(world_from_cam_flu, dtype=np.float64) @ FLU2RDF
        handle = self.server.scene.add_camera_frustum(
            f"/{name}",
            fov=fov,
            aspect=aspect,
            scale=scale,
            position=world_from_cam_rdf[:3, 3],
            wxyz=vtf.SO3.from_matrix(world_from_cam_rdf[:3, :3]).wxyz,
            image=image,
            format="jpeg",
            jpeg_quality=jpeg_quality,
        )
        self._cameras[name] = handle
        return handle

    def update_camera_image(self, name: str, image: NDArray[np.uint8]) -> None:
        if name not in self._cameras:
            raise KeyError(f"Unknown camera {name!r}; have {sorted(self._cameras)}")
        self._cameras[name].image = image

    def step(self, cfgs: dict[str, NDArray[np.floating]] | None = None) -> None:
        with self.server.atomic():
            for name, cfg in (cfgs or {}).items():
                if name not in self._urdf_vis:
                    raise KeyError(f"Unknown asset {name!r}; have {sorted(self._urdf_vis)}")
                cfg = np.asarray(cfg, dtype=np.float64)
                self._cfgs[name] = cfg
                self._urdf_vis[name].update_cfg(cfg)
