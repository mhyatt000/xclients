from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import rerun as rr
import tyro
from einops import rearrange
from jaxtyping import Float, Int
from numpy import ndarray
from rich import print
from server_tri import read
from server_tri.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from server_tri.rerun_util import blueprint
from server_tri.rerun_util.log import log_pinhole
from webpolicy.client import Client

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


def create_clients(cfg: Config) -> dict[str, Client]:
    c = Client(cfg.host, cfg.port)
    return {"low": c, "side": Client(cfg.host, 8004), "over": c}


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
class Config:
    host: str  # yolo host
    port: int  # yolo port

    colmap_path: Path  # map to colmap dir ie: .../sparse/0
    urdf_path: Path  # path to robot urdf
    thresh: float = 0.5  # min confidence to keep 3d point

    def __post_init__(self):
        self.colmap_path = self.colmap_path.expanduser().resolve()
        self.urdf_path = self.urdf_path.expanduser().resolve()


def log_fustrum(cameras: dict[str, PinholeParameters], root: Path):
    for cam in cameras.values():
        # path = Path(cam.name)
        name = f"{cam.name}"
        # COLMAP's camera transform is "camera from world"

        from scipy.spatial.transform import Rotation as R

        quat_xyzw = R.from_matrix(cam.extrinsics.cam_R_world).as_quat()  # xyzw
        # quat_xyzw = image.qvec[[1, 2, 3, 0]].astype(np.float32)  # COLMAP uses wxyz quaternions
        t = cam.extrinsics.cam_t_world  # / 100

        # t = t * in2m

        rr.log(
            f"{root / 'cam' / name}",
            rr.Transform3D(
                translation=t,
                quaternion=quat_xyzw,
                relation=rr.TransformRelation.ChildFromParent,
            ),
            static=True,
        )
        rr.log(name, rr.ViewCoordinates.RDF, static=True)  # X=Right, Y=Down, Z=Forward

        # camera = cameras[image.camera_id]
        # camera = convert_simple_radial_to_pinhole(camera)
        # assert camera.model == "PINHOLE"
        rr.log(
            f"{root / 'cam' / name}",
            rr.Pinhole(
                resolution=[cam.intrinsics.width, cam.intrinsics.height],
                focal_length=cam.intrinsics.f,
                principal_point=cam.intrinsics.c,
            ),
        )

        # bgr = cv2.imread(str(root / image.name))
        # rr.log("camera/image/keypoints", rr.Points2D(visible_xys, colors=[34, 138, 167]))


def main(cfg: Config) -> None:
    # Load COLMAP ; build P matx
    reconstruction = pycolmap.Reconstruction(str(cfg.colmap_path))

    points3dbin = read.read_points3D_binary(cfg.colmap_path / "points3D.bin")
    pcolor = [p.rgb for p in points3dbin.values()]

    points3dbin = {id: point for id, point in points3dbin.items() if np.linalg.norm(point.xyz) < 25}

    view_order = ["low", "side", "over"]
    # view_order = ["side", "over"]
    cameras = {}
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

    caps = open_captures(cfg)
    print(cameras)
    print(caps)

    rr.init("triangulate", spawn=True)
    blueprint.init(list(cameras.values()))
    rr.log("/", rr.ViewCoordinates.FLU, static=True)

    for name, cam in cameras.items():
        log_pinhole(
            cam,
            Path("world") / name,
        )

    root = Path("world")
    log_fustrum(cameras, root)
    clients = create_clients(cfg)

    print(projection_matrices)
    print(caps)
    print(clients)

    in2m = 0.0254
    FLU2RDF = np.array(
        [
            [0, 0, 1, 0],
            [-1, 0, 0, 0],  # -
            [0, -1, 0, 0],  # -
            [0, 0, 0, 1],
        ]
    )
    to_robot = -np.array([4.5, 6.7, 10.3])  # adjust to your robot base
    to_robot_mat = np.eye(4)
    to_robot_mat[:3, 3] = to_robot
    FLU2RDF = FLU2RDF @ to_robot_mat

    rr.log(
        "/world",
        rr.Transform3D(
            translation=FLU2RDF[:3, 3],
            mat3x3=FLU2RDF[:3, :3],
        ),
        static=True,
    )

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

    from yourdfpy import URDF

    urdf = URDF.load(cfg.urdf_path)

    # orig, _ = urdf.scene.graph.get("camera_link")
    # scale = trimesh.transformations.scale_matrix(0.00254)
    # urdf.scene.graph.update(frame_to="camera_link", matrix=orig.dot(scale))

    scaled = urdf.scene.scaled(1 / in2m)
    # scaled = urdf.scene.scaled(1.0)

    from server_tri import rerun_urdf

    rerun_urdf.log_scene(scene=scaled, node=urdf.base_link, path="/robot/urdf", static=True)

    # from lxml import etree
    # elem = robot._to_xml(None,'.')
    # xml = etree.tostring(elem,encoding="utf-8")

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
            rr.log(f"world/cam/{cam.name}/image", rr.Image(frame, color_model="BGR").compress(jpeg_quality=75))

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
    main(tyro.cli(Config))
