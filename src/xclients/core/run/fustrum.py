from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R


def log_fustrum(cameras: dict[np.array], root: Path, inv: bool = False):
    for k, cam in cameras.items():
        name = f"{k}"
        print(name)

        extrinsic = cam["extrinsics"]  # 3x4
        extrinsic_3x3 = extrinsic[:3, :3]
        quat_xyzw = R.from_matrix(extrinsic_3x3).as_quat()  # xyzw
        # quat_xyzw = image.qvec[[1, 2, 3, 0]].astype(np.float32)  # COLMAP uses wxyz quaternions
        t = cam["extrinsics"][:3, 3].astype(np.float32)

        rr.log(
            f"{root / 'cam' / name}",
            rr.Transform3D(
                translation=t,
                quaternion=quat_xyzw,
                relation=rr.TransformRelation.ChildFromParent if not inv else rr.TransformRelation.ParentFromChild,
            ),
            static=True,
        )
        rr.log(name, rr.ViewCoordinates.RDF, static=True)  # X=Right, Y=Down, Z=Forward

        height, width = cam["height"], cam["width"]
        fx = cam["intrinsics"][0, 0]
        fy = cam["intrinsics"][1, 1]
        cx = cam["intrinsics"][0, 2]
        cy = cam["intrinsics"][1, 2]

        rr.log(
            f"{root / 'cam' / name}",
            rr.Pinhole(
                resolution=[width, height],
                focal_length=[fx, fy],
                principal_point=[cx, cy],
            ),
        )

        # rr.log(f"world/cam/{k}/image", rr.Image(cam['frame'], color_model="BGR").compress(jpeg_quality=75), static=False)
