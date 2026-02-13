from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import os
import os.path as osp
from pathlib import Path
import time

import cv2
from flax.traverse_util import flatten_dict
from hamer.configs import CACHE_DIR_HAMER
from hamer.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD, ViTDetDataset
from hamer.models import download_models, load_hamer
from hamer.utils import recursive_to, SkeletonRenderer
from hamer.utils.geometry import perspective_projection
from hamer.utils.render_openpose import render_openpose
from hamer.utils.renderer import cam_crop_to_full, Renderer
import imageio
import jax
import numpy as np
from PIL import Image
import torch

from server_hamer.array_util import keyp2bbox
from server_hamer.log import logger
from server_hamer.vitpose_model import ViTPoseModel

# from manotorch.manolayer import ManoLayer, MANOOutput

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def timing(label=None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            name = label or fn.__name__
            start = time.time()
            result = fn(*args, **kwargs)
            end = time.time()
            print(f"TIMING | {name}: {(end - start) * 1000:.2f} ms")
            return result

        return wrapper

    return decorator


@timing()
def infer(img, detector, vitpose, device, model, model_cfg, renderer: Renderer, cfg):
    img_path = "demo.jpg"
    # img_cv2, img = load_and_preprocess_image(img_path)
    img_cv2 = img.copy()[:, :, ::-1]  # RGB to BGR

    logger.info("### 1. Detect humans with ViTDet")
    pred_bboxes, pred_scores = detect_humans(detector, img_cv2)

    logger.warn(f"Detected {len(pred_bboxes)} people but only using the first one")
    pred_bboxes = pred_bboxes[:1]
    pred_scores = pred_scores[:1]

    if not cfg.fast:
        print(f"Detected {len(pred_bboxes)} people")
        print(pred_bboxes)
        print(pred_scores)

        logger.info("### 2. Detect coarse keypoints with ViTPose")

    pred_bboxes = pred_bboxes.cpu().numpy()
    pred_scores = pred_scores.cpu().numpy()
    poses = timing()(vitpose.predict_pose)(
        img,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    if not cfg.fast:
        viz = vitpose.visualize_pose_results(
            img,
            poses,
            kpt_score_threshold=0.5,
            vis_dot_radius=2,
            vis_line_thickness=1,
        )

        print(poses[0].keys())

        print(f"keypoints shape: {poses[0]['keypoints'].shape}")
        print(f"keypoints mean: {poses[0]['keypoints'].mean()}")

        logger.info("### 3. Extract hand boxes from ViTPose poses")

    bboxes, is_right = extract_hand_keypoints(poses)
    # print(bboxes, is_right)

    if len(bboxes) == 0:
        print("No hands detected")
        return

    if not cfg.fast:
        viz = viz.copy()
        # plot the bboxes on the image and name them right or left
        for bbox, right in zip(bboxes, is_right, strict=False):
            cv2.putText(
                viz,
                "Right" if right else "Left",
                (int(bbox[0]), int(bbox[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0) if right else (0, 0, 255),
                1,
            )
            cv2.rectangle(
                viz,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (255, 0, 0),
                2,
            )
        # save image
        # cv2.imwrite(f"pose_{i}.jpg", cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

    logger.info("### 4. get MANO parameters from HaMeR")

    OUT, _front = run_hand_reconstruction(
        model_cfg,
        img_cv2,
        bboxes,
        is_right,
        device,
        model,
        renderer,
        img_path,
        cfg,
    )

    print(OUT.keys)
    return OUT  # , front


@timing()
def load_and_preprocess_image(img_path):
    """uses cv2 to read image from path and return both BGR and RGB format"""
    img_cv2 = cv2.imread(str(img_path))
    img = img_cv2.copy()[:, :, ::-1]  # RGB to BGR
    return img_cv2, img


@timing()
def detect_humans(detector, img_cv2):
    """detect humans in the image using detectron2 model (ViTDet)
    for all the detections, select only the person class with score > 0.5
    """

    out = detector(img_cv2)
    instances = out["instances"]
    valid_idx = (instances.pred_classes == 0) & (instances.scores > 0.5)
    pred_bboxes = instances.pred_boxes.tensor[valid_idx]
    pred_scores = instances.scores[valid_idx]
    return pred_bboxes, pred_scores


@timing()
def extract_hand_keypoints(poses):
    """the last 42 keypoints are for hands"""

    bboxes, is_right = [], []
    for vitposes in poses:
        left = vitposes["keypoints"][-42:-21]
        right = vitposes["keypoints"][-21:]

        bbox = keyp2bbox(left)
        if bbox is not None:
            bboxes.append(bbox)
            is_right.append(0)

        bbox = keyp2bbox(right)
        if bbox is not None:
            bboxes.append(bbox)
            is_right.append(1)

    return np.stack(bboxes) if bboxes else [], np.stack(is_right) if is_right else []


class Store:
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.data: dict[str, list] = {key: [] for key in keys}

        for key in keys:
            setattr(self, key, self.data[key])

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k].append(v)

    def clear(self):
        for key in self.keys:
            self.data[key].clear()


@timing()
def run_hand_reconstruction(
    model_cfg,
    img_cv2,
    bboxes,
    is_right,
    device,
    model,
    renderer: Renderer,
    img_path,
    args,
):
    """predict mano hand mesh on one image from coarse cropped hand bbox"""

    ### syntax for loading the predictions in batches
    # we arent loading anything new here just img, hand bbox, is_right

    # @timing()
    def mk_dataset():
        dataset = ViTDetDataset(model_cfg, img_cv2, bboxes, is_right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        return dataset, dataloader

    _dataset, dataloader = mk_dataset()

    # S = Store(["verts", "cam_t", "right"])
    OUT = Store(
        [
            "img",
            "personid",
            "box_center",
            "box_size",
            "img_size",
            "right",
            "pred_cam",
            "pred_mano_params",
            "pred_cam_t",
            "focal_length",
            "pred_keypoints_3d",
            "pred_vertices",
            "pred_keypoints_2d",
            "pred_cam_t_full",
            "img_size",
            "scaled_focal_length",
        ]
    )

    _out = []
    for batch in dataloader:  # for every box in the frame...
        if not args.fast:
            print(batch["img"].shape)

        # model fwd ≈ 43ms
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = timing(label="hamer")(model)(batch)

            if not args.fast:
                print(batch.keys())
                print(out.keys())
                print(out["pred_mano_params"].keys())
                print({k: v.shape for k, v in out["pred_mano_params"].items()})

        OUT.add(**batch, **out)
        # _out.append(batch, out)

        """
        cfg = {
            "DATA_DIR": "_DATA/data/",
            "MODEL_PATH": "./_DATA/data/mano",
            "GENDER": "neutral",
            "NUM_HAND_JOINTS": 15,
            "MEAN_PARAMS": "./_DATA/data/mano_mean_params.npz",
            "CREATE_BODY_POSE": False,
        }
        cfg = {k.lower(): v for k, v in cfg.items()}
        from hamer.utils.geometry import rot6d_to_rotmat, aa_to_rotmat

        mano = MANO(**cfg)

        batch_size = batch["img"].shape[0]
        m = {
            # "global_orient": torch.randn(batch_size, 1, 3, 3),
            "global_orient": out["pred_mano_params"]["global_orient"].detach().cpu(),
            "hand_pose": aa_to_rotmat(torch.zeros(batch_size*15, 3)).reshape(batch_size, 15, 3, 3),
            # "betas": torch.randn(batch_size, 10),
            "betas": out["pred_mano_params"]["betas"].detach().cpu(),
        }

        # from manotorch
        # mano = MANO( rot_mode="axisang", use_pca=False, side="right" if batch["right"][i] else "left", center_idx=9, flat_hand_mean=False, model_path="_DATA/data/mano",)

        m = mano(**m, pose2rot=False)
        out["pred_vertices"] = m.vertices
        """

        pred_cam_t_full, img_size, scaled_focal_length = process_batch(batch, out, model_cfg, force_focal=args.fx)

        OUT.add(pred_cam_t_full=pred_cam_t_full, img_size=img_size, scaled_focal_length=scaled_focal_length)

        # render_hand_view(
        # renderer, batch, out, pred_cam_t_full, img_size, img_path, S, args
        # )

    # THIS IS THE IMPORTANT PART
    # if args.full_frame and len(S.verts) > 0:
    # front = render_front_view(
    # S, renderer, img_size, img_cv2, img_path, scaled_focal_length, args
    # )
    front = None

    return OUT, front


# @timing()
def process_batch(batch, out, model_cfg, force_focal=None):
    """process batch to get full frame camera and scale"""

    multiplier = 2 * batch["right"] - 1
    pred_cam = out["pred_cam"]
    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    img_size = batch["img_size"].float()

    cam_fxy = model_cfg.EXTRA.FOCAL_LENGTH
    if force_focal is not None:
        cam_fxy = force_focal
    # cam_fxy = 522
    # cam_fxy = 250 # trial & error ... seems to be roughly 1/2 the true fx
    # cam_fxy = 800
    scaled_focal_length = cam_fxy / model_cfg.MODEL.IMAGE_SIZE * img_size.max()

    pred_cam_t_full = (
        cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
    )
    return pred_cam_t_full, img_size, scaled_focal_length


# @timing()
def render_hand_view(renderer: Renderer, batch, out, pred_cam_t_full, img_size, img_path, S: Store, args):
    """render hand view and save mesh if needed"""

    batch_size = batch["img"].shape[0]
    for n in range(batch_size):
        img_fname, _ = os.path.splitext(os.path.basename(img_path))
        person_id = int(batch["personid"][n])

        mean, std = DEFAULT_MEAN[:, None, None] / 255, DEFAULT_STD[:, None, None] / 255
        -torch.ones_like(batch["img"][n]).cpu() / (std) + (mean)
        input_patch = batch["img"][n].cpu() * (std) + (mean)
        input_patch = input_patch.permute(1, 2, 0).numpy()

        vert = out["pred_vertices"][n].detach().cpu().numpy()
        camt = out["pred_cam_t"][n].detach().cpu().numpy()
        img = batch["img"][n]

        if not args.fast:

            def render(vert, camt, img, side_view=False):
                return renderer(
                    vert,
                    camt,
                    img,
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(-1, 1, 1),
                    side_view=side_view,
                )

            regression_img = render(vert, camt, img)

            if args.side_view:
                side_img = render(vert, camt, img, side_view=True)
                np.concatenate([input_patch, regression_img, side_img], axis=1)
            else:
                np.concatenate([input_patch, regression_img], axis=1)

            # cv2.imwrite(
            # os.path.join(args.out_folder, f"{img_fname}_{person_id}.png"),
            # 255 * final_img[:, :, ::-1],
            # )

        verts = out["pred_vertices"][n].detach().cpu().numpy()
        is_right_value = batch["right"][n].cpu().numpy()
        verts[:, 0] = (2 * is_right_value - 1) * verts[:, 0]
        cam_t = pred_cam_t_full[n]

        if not args.fast and args.save_mesh:
            mpath = os.path.join(args.out_folder, f"{img_fname}_{person_id}.obj")
            save_mesh(verts, cam_t, renderer, is_right_value, mpath)

        # return dict(verts=verts, cam_t=cam_t, right=is_right_value)
        S.add(verts=verts, cam_t=cam_t, right=is_right_value)


@timing()
def save_mesh(verts, cam_t, renderer: Renderer, is_right, mpath):
    tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_BLUE, is_right=is_right)
    tmesh.export(mpath)


@timing()
def render_front_view(
    S: Store,
    renderer: Renderer,
    img_size,
    img_cv2,
    img_path,
    scaled_focal_length,
    args,
):
    misc_args = {
        "mesh_base_color": LIGHT_BLUE,
        "scene_bg_color": (-1, 1, 1),
        "focal_length": scaled_focal_length,
    }
    cam_view = renderer.render_rgba_multiple(
        S.verts,
        cam_t=S.cam_t,
        render_res=img_size[0],
        is_right=S.right,
        **misc_args,
    )

    input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
    input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
    input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

    _img_fname, _ = os.path.splitext(os.path.basename(img_path))
    # cv2.imwrite(
    # os.path.join(args.out_folder, f"{_img_fname}_all.jpg"),
    # 255 * input_img_overlay[:, :, ::-1],
    # )

    front = 255 * input_img_overlay[:, :, ::-1]
    return front


def init_vitdet():
    """
    Initialize the ViTDet model with a pretrained checkpoint.
    TODO fix download of model_final_f05665.pkl from hub

    in $HOME /
    .torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl
    """

    from detectron2.config import LazyConfig
    import hamer
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

    cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    return detector


def init_regnety():
    from detectron2 import model_zoo
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

    detectron2_cfg = model_zoo.get_config("new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True)
    detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
    detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    return detector


def init_detector(args):
    if args.body_detector == "vitdet":
        return init_vitdet()
    elif args.body_detector == "regnety":
        return init_regnety()
    else:
        raise ValueError(f"Unknown body detector: {args.body_detector}")


@timing()
def cam_full_to_crop(cam_full, box, img_size, focal_length=5000.0):
    """
    Converts camera parameters from the full image coordinate system to the cropped image coordinate system.

    Args:
        cam_full (torch.Tensor): Camera parameters in the full image coordinate system of shape (N, 3),
                                 where each row is (s_full, tx_full, ty_full).
        box (torch.Tensor): Crop parameters of shape (N, 4), xywh
                                    where each row is (crop_x0, crop_y0, crop_w, crop_h).
        img_size (torch.Tensor): Full image sizes of shape (N, 2), where each row is (img_w, img_h).
        focal_length (float, optional): The focal length of the camera. Defaults to 5000.0.

    Returns:
        torch.Tensor: Camera translations (tx, ty, tz) in the cropped coordinate system of shape (N, 3).
    """
    tx_full, ty_full, tz_full = cam_full[:, 0], cam_full[:, 1], cam_full[:, 2]
    x, y, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    img_w, img_h = img_size[:, 0], img_size[:, 1]

    tz = 2 * focal_length / tz_full

    # Compute translations in X and Y for the cropped coordinate system
    tx = tx_full - (2 * (x + w / 2 - img_w / 2)) / tz
    ty = ty_full - (2 * (y + h / 2 - img_h / 2)) / tz

    # Stack the translations into a single tensor
    crop_cam = torch.stack([tx, ty, tz], dim=-1)
    return crop_cam


@timing()
def plot_cropped(img, out):
    points = out["pred_keypoints_3d"] + out["pred_cam_t_full"][:, None]
    n = 1
    points = points[n]
    cam_full = out["pred_cam"][n]

    # xywh
    box = torch.Tensor([img.shape[1] / 2.0, 0, img.shape[1] / 2.0, img.shape[0] / 2.0])
    img_size = torch.tensor([img.shape[1], img.shape[0]], dtype=torch.float)

    cam_crop = cam_full_to_crop(cam_full[None], box[None], img_size[None])[0]
    cam_diff = cam_crop - cam_full

    print(cam_full)
    print(cam_crop)
    print(cam_diff)

    x, y, w, h = box.tolist()
    cropped_image = Image.fromarray(img).crop((x, y, x + w, y + h))
    cropped_image.save(f"cropped_{0}.jpg")

    points = points - cam_diff
    # Adjust for scale differences
    focal_length = 5000.0
    scale_full = 2 * focal_length / (cam_full[2] + 1e-9)  # Avoid division by zero
    scale_crop = 2 * focal_length / (cam_crop[2] + 1e-9)  # could be [:,2] for batch
    print(scale_crop / scale_full)
    points_3d_scaled = points * (scale_crop / scale_full)

    points_2d = perspective_projection(
        points=points_3d_scaled[None],
        translation=torch.zeros_like(cam_crop)[None],
        focal_length=torch.full((1, 2), focal_length),
        camera_center=torch.tensor([img.shape[1] / 2 - w / 2, img.shape[0] - h / 2], dtype=torch.float).reshape(1, 2),
    )[0]  # remove batch dim

    # Add confidence = 1
    points_2d = torch.cat([points_2d, torch.ones((points_2d.shape[0], 1))], dim=1)
    cropped_image = render_openpose(np.array(cropped_image), np.array(points_2d))
    Image.fromarray(cropped_image).save(f"cropped_{0}.jpg")

    return cropped_image


@dataclass
class Config:
    pass


def main(cfg: Config):
    args = cfg
    print(args)

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    print(model_cfg)

    # Setup HaMeR model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    detector = init_detector()  # Load detector
    vitpose = ViTPoseModel(device)  # keypoint detector
    renderer = Renderer(model_cfg, faces=model.mano.faces)  # Setup the renderer
    SkeletonRenderer(model_cfg)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)
    # Get all demo images ends with .jpg or .png
    # img_paths = [ img for end in args.file_type for img in Path(args.img_folder).glob(end) ]

    def spec(arr):
        return jax.tree.map(lambda x: (type(x), x.shape), arr)

    def is_leaf(x):
        return isinstance(x, (list, tuple))

    frames = []

    vpath = osp.expanduser("~/output.mp4")
    reader = imageio.get_reader(vpath)
    for i, img in enumerate(reader):
        try:
            out, front = infer(i, img, detector, vitpose, device, model, model_cfg, renderer)
            out = jax.tree.map(lambda x: x[0], out.data, is_leaf=is_leaf)  # everything is wrapped in list

            out = flatten_dict(out)

            def clean(x):
                return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

            out = jax.tree.map(clean, out)

            print(spec(out))

            np.savez(f"pose_{i}.npz", **out)
            continue

        except Exception as e:
            print(e)
            continue

        frames.append(front.copy())


if __name__ == "__main__":
    main()
