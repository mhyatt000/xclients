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

HAND_EDGES = np.array(
    [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ],
    dtype=np.int32,
)
HAND_COLORS = {
    "left": (0, 255, 0),
    "right": (255, 0, 255),
}
DEFAULT_HAND_COLOR = (0, 180, 255)

# Keep in sync with xclients.motion.embodiment: dxarm-l is the arm on the operator's left (+Y).
DXARM_L_POSITION = (0.0, 0.09687363, 0.14767363)
DXARM_L_WXYZ = (0.92387953, -0.38268343, 0.0, 0.0)
DXARM_R_POSITION = (0.0, -0.09687363, 0.14767363)
DXARM_R_WXYZ = (0.92387953, 0.38268343, 0.0, 0.0)


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
        self._hand_points: dict[str, viser.PointCloudHandle] = {}
        self._hand_bones: dict[str, viser.LineSegmentsHandle] = {}

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

    def update_hand(
        self,
        name: str,
        kp3d: NDArray[np.floating],
        color: tuple[int, int, int] = (0, 255, 0),
        point_size: float = 0.008,
    ) -> None:
        """Plot 21 world-frame hand keypoints (and bones) under /hands/<name>."""
        kp3d = np.asarray(kp3d, dtype=np.float32)
        if kp3d.shape != (21, 3):
            raise ValueError(f"Expected hand keypoints with shape (21, 3), got {kp3d.shape}")
        segments = kp3d[HAND_EDGES]
        if name not in self._hand_points:
            self._hand_points[name] = self.server.scene.add_point_cloud(
                f"/hands/{name}/kp3d",
                points=kp3d,
                colors=np.tile(np.array(color, dtype=np.uint8), (21, 1)),
                point_size=point_size,
                point_shape="circle",
            )
            self._hand_bones[name] = self.server.scene.add_line_segments(
                f"/hands/{name}/bones",
                points=segments,
                colors=np.array(color, dtype=np.uint8),
                line_width=3.0,
            )
        else:
            with self.server.atomic():
                self._hand_points[name].points = kp3d
                self._hand_bones[name].points = segments

    def update_hands(self, kp3ds: dict[str, NDArray[np.floating]]) -> None:
        """Plot named hands (e.g. 'left'/'right'); hides hands no longer present."""
        for name, kp3d in kp3ds.items():
            self.update_hand(name, kp3d, color=HAND_COLORS.get(name, DEFAULT_HAND_COLOR))
        with self.server.atomic():
            for name, points in self._hand_points.items():
                visible = name in kp3ds
                points.visible = visible
                self._hand_bones[name].visible = visible

    def step(self, cfgs: dict[str, NDArray[np.floating]] | None = None) -> None:
        with self.server.atomic():
            for name, cfg in (cfgs or {}).items():
                if name not in self._urdf_vis:
                    raise KeyError(f"Unknown asset {name!r}; have {sorted(self._urdf_vis)}")
                cfg = np.asarray(cfg, dtype=np.float64)
                self._cfgs[name] = cfg
                self._urdf_vis[name].update_cfg(cfg)
