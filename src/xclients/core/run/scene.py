from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.urdf as rru

from xclients.core import tf as xctf
from xclients.core.run.blueprint import init_blueprint
from xclients.core.run.fustrum import log_fustrum


@dataclass
class CameraCalibration:
    name: str
    ht_path: Path
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    width: int
    height: int


class RerunScene:
    def __init__(
        self,
        urdf: Path,
        *,
        app_id: str = "rerun_scene",
        entity_path_prefix: str = "robot",
        transforms_path: str | None = None,
        joint_values_path: str | None = None,
        world_path: Path = Path("world"),
        cameras: Sequence[str | int] | None = None,
        camera_ht_files: Sequence[Path] | None = None,
        camera_resolution: tuple[int, int] = (640, 480),
        camera_focal_length: tuple[float, float] = (515.0, 515.0),
        spawn: bool = True,
        rrd_path: Path | None = None,
        view_coordinates: rr.ViewCoordinates = rr.ViewCoordinates.FLU,
    ) -> None:
        self.log = logging.getLogger(__name__)
        self.urdf = urdf.expanduser().resolve()
        self.app_id = app_id
        self.entity_path_prefix = entity_path_prefix.strip("/")
        self.transforms_path = transforms_path or f"{self.entity_path_prefix}/transforms"
        self.joint_values_path = joint_values_path or f"{self.entity_path_prefix}/joint_values"
        self.world_path = world_path
        self.cameras = list(cameras or [])
        self.camera_ht_files = [Path(path).expanduser().resolve() for path in camera_ht_files or []]
        self.camera_resolution = camera_resolution
        self.camera_focal_length = camera_focal_length
        self.spawn = spawn
        self.rrd_path = rrd_path
        self.view_coordinates = view_coordinates

        rr.init(self.app_id)
        if self.rrd_path is not None:
            rr.save(self.rrd_path)
        if self.spawn:
            try:
                rr.spawn()
            except RuntimeError as exc:
                self.log.warning("Failed to spawn rerun viewer: %s", exc)

        rr.log("/", self.view_coordinates, static=True)
        rr.log(str(self.world_path), rr.CoordinateFrame(frame=str(self.world_path)), static=True)
        rr.log(str(self.world_path), self.view_coordinates, static=True)
        rr.log(
            str(self.world_path),
            rr.Transform3D(
                translation=[0.0, 0.0, 0.0],
                quaternion=[0.0, 0.0, 0.0, 1.0],
                relation=rr.TransformRelation.ChildFromParent,
            ),
            static=True,
        )
        rr.log_file_from_path(self.urdf, entity_path_prefix=self.entity_path_prefix, static=True)

        self.urdf_tree = rru.UrdfTree.from_file_path(self.urdf, entity_path_prefix=self.entity_path_prefix)
        self.joints = [
            joint for joint in self.urdf_tree.joints() if joint.joint_type in {"revolute", "continuous", "prismatic"}
        ]
        self.joint_map = {joint.name: joint for joint in self.joints}
        self.joint_names = list(self.joint_map)
        self.joint_values_rad = {name: self._clamp_joint_value(joint, 0.0) for name, joint in self.joint_map.items()}

        self.camera_calibrations = self._load_camera_calibrations(self.camera_ht_files)
        if self.camera_calibrations:
            self.set_cameras(list(self.camera_calibrations))
            self.log_camera_poses(self.camera_calibrations)
        elif self.cameras:
            init_blueprint(self.cameras)

    def set_cameras(self, cameras: Sequence[str | int]) -> None:
        self.cameras = list(cameras)
        if self.cameras:
            init_blueprint(self.cameras)

    def log_camera_poses(self, cameras: Mapping[str | int, dict], *, inv: bool = False) -> None:
        if cameras and not self.cameras:
            self.set_cameras(list(cameras))
        log_fustrum(dict(cameras), root=self.world_path, inv=inv)

    def log_camera_images(
        self,
        frames: Mapping[str | int, np.ndarray],
        *,
        jpeg_quality: int = 75,
        static: bool = False,
    ) -> None:
        for cam, frame in frames.items():
            entity_path = f"{self.world_path}/cam/{cam}/image"
            rr.log(entity_path, rr.CoordinateFrame(frame=self._image_plane_frame(cam)), static=True)
            rr.log(
                entity_path,
                rr.Image(frame, color_model="BGR").compress(jpeg_quality=jpeg_quality),
                static=static,
            )

    def log_points2d(
        self,
        camera: str | int,
        points: np.ndarray,
        *,
        colors: np.ndarray | None = None,
        radii: float | np.ndarray | None = None,
        labels: Sequence[str] | None = None,
        path: str | None = None,
        parent_frame: str | None = None,
        static: bool = False,
    ) -> None:
        entity_path = path or f"{self.world_path}/cam/{camera}/kp2d"
        rr.log(entity_path, rr.CoordinateFrame(frame=self._image_plane_frame(camera)), static=True)
        rr.log(
            entity_path,
            rr.Points2D(
                points,
                colors=colors,
                radii=radii,
                labels=labels,
            ),
            static=static,
        )

    def log_points3d(
        self,
        points: np.ndarray,
        *,
        colors: np.ndarray | None = None,
        radii: float | np.ndarray | None = None,
        labels: Sequence[str] | None = None,
        path: str | None = None,
        parent_frame: str | None = None,
        static: bool = False,
    ) -> None:
        entity_path = path or f"{self.world_path}/scene/points"
        rr.log(entity_path, rr.CoordinateFrame(frame=parent_frame or str(self.world_path)), static=True)
        rr.log(
            entity_path,
            rr.Points3D(
                points,
                colors=colors,
                radii=radii,
                labels=labels,
            ),
            static=static,
        )

    def log_joints(
        self,
        values: Mapping[str, float] | Sequence[float],
        *,
        step: int,
        timeline: str = "step",
        degrees: bool = False,
    ) -> None:
        self._update_joint_values(values, degrees=degrees)
        rr.send_columns(
            self.transforms_path,
            indexes=[rr.TimeColumn(timeline, sequence=[step] * len(self.joints))],
            columns=self._joint_transform_columns(),
        )

        rr.set_time(timeline, sequence=step)
        for name in self.joint_names:
            rr.log(
                f"{self.joint_values_path}/{name}",
                rr.Scalars([float(np.rad2deg(self.joint_values_rad[name]))]),
            )

    def _load_camera_calibrations(self, paths: Sequence[Path]) -> dict[str, dict]:
        calibrations = {}
        width, height = self.camera_resolution
        fx, fy = self.camera_focal_length
        intrinsics = np.array(
            [
                [fx, 0.0, width / 2.0],
                [0.0, fy, height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )

        for path in paths:
            data = np.load(path)
            ht = {k: data[k] for k in data.files}["HT"]
            calibration = CameraCalibration(
                name=path.parent.name,
                ht_path=path,
                intrinsics=intrinsics.copy(),
                extrinsics=xctf.RDF2FLU @ np.linalg.inv(ht),
                width=width,
                height=height,
            )
            calibrations[calibration.name] = {
                "intrinsics": calibration.intrinsics,
                "extrinsics": calibration.extrinsics,
                "width": calibration.width,
                "height": calibration.height,
            }

        return calibrations

    def _image_plane_frame(self, camera: str | int) -> str:
        return f"{self.world_path}/cam/{camera}/image_plane"

    def _joint_transform_columns(self):
        translations = []
        quaternions = []
        parent_frames = []
        child_frames = []

        for name in self.joint_names:
            transform = self.joint_map[name].compute_transform(self.joint_values_rad[name], clamp=True)
            translations.append(self._to_pylist(transform.translation))
            quaternions.append(self._to_pylist(transform.quaternion))
            parent_frames.append(self._to_pylist(transform.parent_frame))
            child_frames.append(self._to_pylist(transform.child_frame))

        return rr.Transform3D.columns(
            translation=translations,
            quaternion=quaternions,
            parent_frame=parent_frames,
            child_frame=child_frames,
        )

    def _update_joint_values(self, values: Mapping[str, float] | Sequence[float], *, degrees: bool) -> None:
        if isinstance(values, Mapping):
            for name, value in values.items():
                joint = self.joint_map.get(name)
                if joint is None:
                    raise KeyError(f"Unknown joint: {name}")
                self.joint_values_rad[name] = self._coerce_joint_value(joint, value, degrees=degrees)
            return

        if len(values) != len(self.joint_names):
            raise ValueError(f"Expected {len(self.joint_names)} joint values, got {len(values)}")

        for name, value in zip(self.joint_names, values, strict=True):
            self.joint_values_rad[name] = self._coerce_joint_value(self.joint_map[name], value, degrees=degrees)

    def _coerce_joint_value(self, joint: rru.UrdfJoint, value: float, *, degrees: bool) -> float:
        value_rad = float(np.deg2rad(value) if degrees else value)
        return self._clamp_joint_value(joint, value_rad)

    @staticmethod
    def _clamp_joint_value(joint: rru.UrdfJoint, value: float) -> float:
        if np.isfinite(joint.limit_lower):
            value = max(value, joint.limit_lower)
        if np.isfinite(joint.limit_upper):
            value = min(value, joint.limit_upper)
        return float(value)

    @staticmethod
    def _to_pylist(batch: object):
        return batch.as_arrow_array().to_pylist()[0]
