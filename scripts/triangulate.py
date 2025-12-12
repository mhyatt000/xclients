from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
from einops import rearrange
from jaxtyping import Float, Int
import numpy as np
from numpy import ndarray

# import pycolmap
import rerun as rr
from rich import print
import tyro
from webpolicy.client import Client

from xclients.core import tf as xctf
from xclients.core.cfg import Config
from xclients.core.run import blueprint
from xclients.core.run.fustrum import log_fustrum
from xclients.core.run.rerun_urdf import ez_load_urdf

np.set_printoptions(suppress=True, precision=2)


joints = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

joint_pairs = [
    (5, 7),
    (7, 9),  # left shoulder → left elbow → left wrist
    (6, 8),
    (8, 10),  # right shoulder → right elbow → right wrist
    (5, 6),  # shoulders
    (11, 12),  # hips
    (5, 11),
    (6, 12),  # torso links
    (11, 13),
    (13, 15),  # left hip → knee → ankle
    (12, 14),
    (14, 16),  # right hip → knee → ankle
    (1, 3),
    (2, 4),  # face: eye → ear
    (0, 1),
    (0, 2),  # nose → eyes
]


# MANO joint order
# 21 keypoints
mano_joints = {
    0: "Wrist",
    1: "Thumb_MCP",
    2: "Thumb_PIP",
    3: "Thumb_DIP",
    4: "Thumb_Tip",
    5: "Index_MCP",
    6: "Index_PIP",
    7: "Index_DIP",
    8: "Index_Tip",
    9: "Middle_MCP",
    10: "Middle_PIP",
    11: "Middle_DIP",
    12: "Middle_Tip",
    13: "Ring_MCP",
    14: "Ring_PIP",
    15: "Ring_DIP",
    16: "Ring_Tip",
    17: "Pinky_MCP",
    18: "Pinky_PIP",
    19: "Pinky_DIP",
    20: "Pinky_Tip",
}
mano_joint_pairs = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Pinky
]

colors = {
    "Wrist": [1.0, 1.0, 0.0],
    "MCP_PIP": [1.0, 0.0, 0.0],
    "PIP_DIP": [0.0, 1.0, 0.0],
    "DIP_Tip": [0.0, 0.0, 1.0],
}


def batch_triangulate(
    keypoints_2d: Float[ndarray, "nViews nJoints 3"],
    projection_matrices: Float[ndarray, "nViews 3 4"],
    min_views: int = 2,
) -> Float[ndarray, "nJoints 4"]:
    """Triangulate joints from multi-view correspondences with linear least squares.

    Args:
        keypoints_2d: Homogeneous 2D joints ``[n_views, n_joints, 3]``.
        projection_matrices: Camera projection matrices ``[n_views, 3, 4]``.
        min_views: Minimum number of valid views required to triangulate a joint.

    Returns:
        Homogeneous 3D joints ``[n_joints, 4]`` with aggregated confidence.
    """
    num_joints: int = keypoints_2d.shape[1]

    # Count views where each joint is visible
    visibility_count: Int[ndarray, nJoints] = (keypoints_2d[:, :, -1] > 0).sum(axis=0)
    valid_joints = np.where(visibility_count >= min_views)[0]

    # Filter keypoints by valid joints
    filtered_keypoints: Float[ndarray, "nViews nJoints 3"] = keypoints_2d[:, valid_joints]
    conf3d = filtered_keypoints[:, :, -1].sum(axis=0) / visibility_count[valid_joints]

    P0: Float[ndarray, "1 nViews 4"] = projection_matrices[None, :, 0, :]
    P1: Float[ndarray, "1 nViews 4"] = projection_matrices[None, :, 1, :]
    P2: Float[ndarray, "1 nViews 4"] = projection_matrices[None, :, 2, :]

    # x-coords homogenous
    u: Float[ndarray, "nJoints nViews 1"] = rearrange(filtered_keypoints[..., 0], "c j -> j c 1")
    uP2: Float[ndarray, "nJoints nViews 4"] = u * P2

    # y-coords homogenous
    v: Float[ndarray, "nJoints nViews 1"] = rearrange(filtered_keypoints[..., 1], "c j -> j c 1")
    vP2: Float[ndarray, "nJoints nViews 4"] = v * P2

    confidences: Float[ndarray, "nJoints nViews 1"] = rearrange(filtered_keypoints[..., 2], "c j -> j c 1")

    Au: Float[ndarray, "nJoints nViews 4"] = confidences * (uP2 - P0)
    Av: Float[ndarray, "nJoints nViews 4"] = confidences * (vP2 - P1)
    A: Float[ndarray, "nJoints _ 4"] = np.hstack([Au, Av])

    # Solve using SVD
    _, _, Vh = np.linalg.svd(A)
    triangulated_points = Vh[:, -1, :]
    triangulated_points /= triangulated_points[:, 3, None]

    # Construct result
    result: Float[ndarray, "nJoints 4"] = np.zeros((num_joints, 4))
    # convert from homogenous to euclidean and add confidence
    result[valid_joints, :3] = triangulated_points[:, :3]
    result[valid_joints, 3] = conf3d

    return result


def _get_cam_from_world(image: pycolmap.Image) -> pycolmap.Rigid3d:
    """Handle both property-style and method-style cam_from_world."""
    attr = image.cam_from_world
    return attr() if callable(attr) else attr


def projection_matrix_from_image(
    reconstruction: pycolmap.Reconstruction,
    image_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        reconstruction: pycolmap.Reconstruction loaded from model_dir.
        image_name: Either full filename or stem ('low', 'side', etc.).

    Returns:
        P_3x4: 3x4 projection matrix K [R | t]
        T_cw_4x4: 4x4 world→camera transform
        K_3x3: intrinsics
    """
    # Find the pycolmap.Image by name or stem
    img = None
    for im in reconstruction.images.values():
        stem = Path(im.name).stem
        if im.name == image_name or stem == image_name:
            img = im
            break
    if img is None:
        raise ValueError(f"No image with name/stem '{image_name}' in reconstruction")

    cam = reconstruction.cameras[img.camera_id]

    height, width = cam.height, cam.width

    # world → camera pose
    T_cw_3x4 = img.cam_from_world()  # Rigid3d object
    T_cw_3x4 = np.asarray(T_cw_3x4.matrix())
    R_cw = T_cw_3x4[:, :3]
    t_cw = T_cw_3x4[:, 3:4]  # [3,1]

    # OpenCV-style rvec/tvec, if you want them:
    # rvec, _ = cv2.Rodrigues(R_cw)
    # tvec = t_cw

    # intrinsics
    K = np.asarray(cam.calibration_matrix())  # [3,3]

    # 3x4 projection matrix K [R | t]
    P_3x4 = K @ np.hstack([R_cw, t_cw])  # [3,4]

    # optional 4x4 homogeneous
    T_cw_4x4 = np.eye(4)
    T_cw_4x4[:3, :4] = T_cw_3x4

    return {  # projection matrix K [R | t]
        "P": P_3x4,
        "T": T_cw_4x4,
        "K": K,
        "r_cw": R_cw,
        "t_cw": t_cw,
        "height": height,
        "width": width,
    }
    return P_3x4, T_cw_4x4, K


def open_captures(cfg) -> dict[str, cv2.VideoCapture]:
    return {
        "low": cv2.VideoCapture(4),
        "side": cv2.VideoCapture(2),
        "over": cv2.VideoCapture(0),
    }


def grab_keypoints_once(
    caps: dict[str, cv2.VideoCapture],
    clients: dict[str, Client],
    view_order: Sequence[str],
) -> np.ndarray:
    """
    Returns:
        keypoints_2d: [nViews, nJoints, 3] (u, v, conf)
    """
    kp_per_view: list[np.ndarray] = []

    outs = {}
    for name in view_order:
        cap = caps[name]
        client = clients[name]

        ok, frame = cap.read()
        if not ok:
            print(f"Failed to read from camera '{name}'")
            outs[name] = {}
            continue

        # run your detector
        out = client.step({"img": frame})
        if not out:
            print(f"Failed to get keypoints from camera '{name}'")
            outs[name] = {}
            continue

        print(out.keys())
        if "pred_keypoints_2d" in out:
            print(out["pred_keypoints_2d"].shape)
            m = max(0, len(out["pred_keypoints_2d"]) - 1)  # which hand pred
            kp2dn = out["pred_keypoints_2d"][m]  # norm

            kp2d = kp2dn.copy()
            center, size = out["box_center"][m], out["box_size"][m]  # 2, 1

            # -1 to 1  => 0 to width/height
            # * [frame.shape[1], frame.shape[0]]  # to pixel coords
            # add center
            kp2d[:, 0] = (kp2dn[:, 0] * size) + center[0]
            kp2d[:, 1] = (kp2dn[:, 1] * size) + center[1]

            conf = np.ones_like(kp2d[:, :1]) * 1.0  # dummy confidence
            kpc = np.concatenate([kp2d, conf], axis=-1)
            out = {"xyc": kpc}
            outs[name] = out

        else:
            out["xyc"] = np.concatenate([out["xy"][0], np.expand_dims(out["conf"][0], axis=-1)], axis=-1)
            outs[name] = out

        kp_per_view.append(out["xyc"])

    # stack into [nViews, nJoints, 3]
    keypoints_2d = np.stack(kp_per_view, axis=0) if kp_per_view else None
    return outs, keypoints_2d


@dataclass
class MyConfig(Config):
    urdf: Path  # path to robot urdf
    thresh: float = 0.5  # min confidence to keep 3d point

    def __post_init__(self):
        self.urdf = self.urdf.expanduser().resolve()


def main(cfg: MyConfig) -> None:
    # Load COLMAP ; build P matx
    # reconstruction = pycolmap.Reconstruction(str(cfg.colmap_path))
    # points3dbin = read.read_points3D_binary(cfg.colmap_path / "points3D.bin")
    # pcolor = [p.rgb for p in points3dbin.values()]

    # points3dbin = {id: point for id, point in points3dbin.items() if np.linalg.norm(point.xyz) < 25}

    view_order = ["low", "side", "over"]
    # view_order = ["side", "over"]
    cameras = {}
    """
    P_list = []
    for name in view_order:
        info = projection_matrix_from_image(reconstruction, name)
        P_3x4, _T_cw_4x4, _K = info["P"], info["T"], info["K"]

        print(info)
        camera = PinholeParameters(
            name=name,
            extrinsics=Extrinsics(
                cam_R_world=info["r_cw"],
                cam_t_world=info["t_cw"].reshape(-1),
            ),
            intrinsics=Intrinsics.from_matrix(info["K"]),
        )
        camera.intrinsics.height = info["height"]
        camera.intrinsics.width = info["width"]
        cameras[name] = camera

        P_list.append(P_3x4)
    projection_matrices = np.stack(P_list, axis=0)  # [3, 3, 4]
    """

    caps = open_captures(cfg)

    h, w = 480, 640
    cx, cy = w / 2, h / 2
    fx, fy = 515.0, 515.0

    intr = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    np.eye(4)[:3, :]  # dummy

    def load_extr(name: str):
        p = Path("data/cam") / name / "HT.npz"
        x = np.load(p)
        x = {k: x[k] for k in x}["HT"]
        x = xctf.RDF2FLU @ np.linalg.inv(x)
        return x

    cameras = {
        k: {
            "intrinsics": intr,
            "extrinsics": load_extr(k),
            "height": h,
            "width": w,
        }
        for k in view_order
    }

    print(cameras)
    print(caps)
    P = {k: cameras[k]["intrinsics"] @ cameras[k]["extrinsics"][:3, :] for k in view_order}

    rr.init("triangulate", spawn=True)
    blueprint.init_blueprint(list(caps.keys()))
    rr.log("/", rr.ViewCoordinates.FLU, static=True)

    """
    for name, cam in cameras.items():
        log_pinhole(
            cam,
            Path("world") / name,
        )
    """

    root = Path("world")
    print(cameras)
    log_fustrum(cameras, root)
    client = Client(cfg.host, cfg.port)

    print(caps)

    # to_robot = -np.array([4.5, 6.7, 10.3])  # adjust to your robot base
    # to_robot_mat = np.eye(4)
    # to_robot_mat[:3, 3] = to_robot
    # FLU2RDF = xctf.FLU2RDF @ to_robot_mat

    # rr.log(
    # "/world",
    # rr.Transform3D(
    # translation=xctf.FLU2RDF[:3, 3],
    # mat3x3=xctf.FLU2RDF[:3, :3],
    # ),
    # static=True,
    # )

    """
    rr.log(
        "/world/scene/points",
        rr.Points3D(
            # [point.xyz for point in scene.values()],
            [p.xyz for p in points3dbin.values()],
            colors=pcolor,
            radii=0.2,
        ),
        static=True,
    )
    """

    ez_load_urdf(cfg.urdf)

    def perspective_transform(p, p3d):
        """P: [3,4], p3d: [nPoints, 3]
        h denotes homogeneous coordinates
        """
        print(p.shape, p3d.shape)
        nPoints = p3d.shape[0]
        p3dh = np.hstack([p3d, np.ones((nPoints, 1))])  # [nPoints, 4]
        p2dh = (p @ p3dh.T).T  # [nPoints, 3]
        p2d = p2dh[:, :2] / p2dh[:, 2:3]
        return p2d

    while True:
        frames = {k: cap.read() for k, cap in caps.items()}
        frames = {k: f[1] for k, f in frames.items() if f[0]}
        outs = {k: client.step({"image": f}) for k, f in frames.items()}

        from xclients.core.cfg import spec

        print(spec(outs))
        k2ds = {k: v.get("wilor_preds", {}).get("pred_keypoints_2d") for k, v in outs.items()}

        # k2ds = {k: v.get('wilor_preds',{}).get('pred_vertices') for k,v in outs.items()}
        # k2ds = {k: None if v is None  else  v[0]+outs[k]['wilor_preds']['pred_cam'] for k,v in k2ds.items()}
        # k2ds = {k: None if v is None  else perspective_transform(P[k], v.reshape(-1,3)) for k,v in k2ds.items()}

        # print(k2ds)

        for k, f in frames.items():
            rr.log(
                f"world/cam/{k}/image",
                rr.Image(f, color_model="BGR").compress(jpeg_quality=75),
                static=False,
            )

            k2d = k2ds[k]
            if k2d is None:
                continue
            rr.log(
                f"world/cam/{k}/kp2d",
                rr.Points2D(
                    k2d,
                    colors=np.tile(np.array([[1.0, 0.0, 0.0]]), (k2d.shape[0], 1)),
                    radii=np.ones((k2d.shape[0], 1)) * 3,
                ),
            )

        if sum([p is not None for p in k2ds.values()]) >= 2:
            p = np.stack([P[k] for k in caps if k2ds[k] is not None], axis=0)
            k2ds = np.array([k2ds[k] for k in caps if k2ds[k] is not None]).reshape(len(p), -1, 2)
            # 100% confidence
            k2ds = np.concatenate([k2ds, np.ones((*k2ds.shape[:-1], 1))], axis=-1)
            print(p.shape, k2ds.shape)

            k3ds = batch_triangulate(k2ds, p, min_views=2)
            # print(k3ds)

            rr.log(
                "world/kp3d",
                rr.Points3D(
                    k3ds[:, :3],
                    colors=np.tile(np.array([[1.0, 0.0, 0.0]]), (k3ds.shape[0], 1)),
                    # labels=labels,
                    radii=np.ones(k3ds.shape[0]) * 0.0025,
                ),
            )

            # segments are [a,b], [b,c], ... from kp3d for points in mano joint order
            segments = [[mano_joint_pairs[i][0], mano_joint_pairs[i][1]] for i in range(len(mano_joint_pairs))]
            segments = k3ds[..., :3][segments]  # [nSegments, 2, 3]
            colors_ls = []
            for i in range(len(mano_joint_pairs)):
                j1, j2 = mano_joint_pairs[i]
                if j1 == 0:
                    colors_ls.append(colors["Wrist"])
                elif j2 % 4 == 0:
                    colors_ls.append(colors["MCP_PIP"])
                elif j2 % 4 == 3:
                    colors_ls.append(colors["DIP_Tip"])
                else:
                    colors_ls.append(colors["PIP_DIP"])

            colors_ls = np.array(colors_ls)

            print(segments.shape)
            print(colors_ls.shape)
            # segments = segments.reshape(-1, 3)  # flatten to line strip
            rr.log(  # the lines
                "world/lp3d",
                rr.LineStrips3D(
                    segments,
                    colors=np.array(colors_ls),
                ),
            )

    # VIS = O3DPointStepper()
    while True:
        outs, k2d = grab_keypoints_once(caps, clients, view_order)

        # k2d: [nViews, nJoints, 3]

        """
        for xy, conf in zip(out['xy'][0], out['conf'][0]) : # keypoints
            if conf < 0.3:
                continue
            x, y = int(xy[0]), int(xy[1])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        for j1, j2 in joint_pairs:
            x1, y1 = int(out['xy'][0][j1][0]), int(out['xy'][0][j1][1])
            x2, y2 = int(out['xy'][0][j2][0]), int(out['xy'][0][j2][1])
            c1, c2 = out['conf'][0][j1], out['conf'][0][j2]
            if c1 < 0.3 or c2 < 0.3:
                continue
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        """

        # print(k2d)
        # scene = {id:point for id, point in scene.items() if point.rgb.any()}

        if k2d is not None and len(k2d) > 1:
            conf = k2d[..., -1].mean(axis=0)
            k2d[..., -1] = 1
            # print(k2d)
            print(k2d.shape)
            joints_3d_h = batch_triangulate(k2d, projection_matrices, min_views=2)
            # joints_3d_h: [nJoints, 4] => (X, Y, Z, conf3d)

            # select only confident points
            # labels=[joints[i] for i in range(joints_3d_h.shape[0])], # 17 only
            mask = conf >= cfg.thresh
            joints_3d_h = joints_3d_h[mask]
            # labels = [joints[i] for i in range(joints_3d_h.shape[0]) if mask[i]]

            joints_3d_h[:, :3] = joints_3d_h[:, :3]  # /100  # in mm

            print("Triangulated joints (XYZ + conf):")

            rr.log(
                "world/kp3d",
                rr.Points3D(
                    joints_3d_h[:, :3],
                    colors=np.tile(np.array([[1.0, 0.0, 0.0]]), (joints_3d_h.shape[0], 1)),
                    # labels=labels,
                    radii=np.ones((joints_3d_h.shape[0], 1)) * 0.25,
                ),
            )
            rr.log(  # the lines
                "world/kp3d/lines",
                rr.LineStrips3D(
                    joints_3d_h[:, :3],
                ),
            )

        for _n, k in enumerate(cameras):
            cap, cam = caps[k], cameras[k]
            ret, frame = cap.read()
            if not ret:
                continue

            rr.log(
                f"world/cam/{cam.name}/image",
                rr.Image(frame, color_model="BGR").compress(jpeg_quality=75),
            )

            xyc = outs[k].get("xyc")
            if xyc is None:
                continue
            mask2d = xyc[:, 2] >= cfg.thresh
            xyc = xyc[mask2d]
            rr.log(
                f"world/cam/{cam.name}/kp2d",
                rr.Points2D(
                    xyc[:, :2],
                    colors=np.tile(np.array([[1.0, 0.0, 0.0]]), (xyc.shape[0], 1)),
                    radii=np.ones((xyc.shape[0], 1)) * 3,
                ),
            )

        # print(joints_3d_h)
        # VIS.step(joints_3d_h/100)

    for cap in caps.values():
        cap.release()


if __name__ == "__main__":
    main(tyro.cli(MyConfig))
