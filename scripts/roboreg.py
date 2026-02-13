from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Literal

from common import Config, spec
import cv2
import numpy as np
from PIL import Image
import rerun as rr
from rich import print
from sam3 import SAMConfig
import tyro
from webpolicy.client import Client

from xclients.core.run import rerun_urdf
from xclients.core.run.blueprint import init_blueprint
from xclients.core.run.fustrum import log_fustrum

logging.basicConfig(level=logging.INFO)
np.set_printoptions(precision=3, suppress=True)


@dataclass
class RoboregConfig:
    da: Config
    sam: SAMConfig
    roboreg: Config

    cams: list[int] = field(default_factory=lambda: [0])
    show: bool = True
    downscale: int = 1
    urdf: Path = Path("data/robot.urdf")

    do: Literal["calibrate", "collect", "display"] = "display"
    ht: Path = Path("cam")


def main(cfg: RoboregConfig) -> None:
    from xarm.wrapper import XArmAPI

    arm = XArmAPI("192.168.1.231", is_radian=True)
    arm.connect()
    arm.motion_enable(True)
    print(arm.angles)

    da = Client(cfg.da.host, cfg.da.port)
    sam = Client(cfg.sam.host, cfg.sam.port)
    roboreg = Client(cfg.roboreg.host, cfg.roboreg.port)

    rr.init("roboreg", spawn=True)
    rr.log("/", rr.ViewCoordinates.FLU, static=True)
    init_blueprint(cfg.cams)

    caps = {cam: cv2.VideoCapture(cam) for cam in cfg.cams}

    from yourdfpy import URDF

    urdf = URDF.load(cfg.urdf)
    import trimesh

    trimesh.transformations.scale_matrix(1)
    scaled = urdf.scene.scaled(1.001)
    rerun_urdf.log_scene(scene=scaled, node=urdf.base_link, path="/robot/urdf", static=True)

    logging.info("Displaying frames from camera 0; press 'q' to quit.")
    begin, _uncalibrated = True, True
    files = Path(".").glob("roboreg_*.npz")
    roboreg.step({})

    while True:
        frames = {cam: caps[cam].read() for cam in cfg.cams}
        frames = {cam: frame for cam, (ret, frame) in frames.items() if ret}
        frames = list(frames.values())

        for k, frame in zip(cfg.cams, frames, strict=False):
            rr.log(
                f"world/cam/{k}/image",
                rr.Image(frame, color_model="BGR").compress(jpeg_quality=75),
                static=False,
            )

        frame = frames[0]

        if cfg.do == "collect":
            cv2.imshow("frame", frame)
            if _key := cv2.waitKey(100) & 0xFF == ord("y"):
                print("saving npz...")
                from datetime import datetime as dt

                t = dt.now().strftime("%Y%m%d_%H%M%S")
                payload = {
                    "images": frame,
                    # points=points,
                    # colors=colors,
                    "joints": arm.angles,
                    # mask=mask,
                    # depth=np.array(resized_pred),
                    # intrinsics=intrinsics,
                    # extrinsics=da_out['extrinsics'][0],
                }
                np.savez(f"roboreg_{t}.npz", **payload)
            if _key := cv2.waitKey(100) & 0xFF == ord("q"):
                quit()
            continue

        HT, overlays = None, None
        datas = []
        if cfg.do == "calibrate":
            # uncalibrated=False

            for file in files:
                print(file)
                data = np.load(file)
                data = {k: data[k] for k in data.files}

                frame, _joints = data["images"], data["joints"]

                sam_payload = {
                    "image": frame,
                    "type": "image",
                    "text": cfg.sam.prompt,
                    "confidence": cfg.sam.confidence,
                }
                sam_out = sam.step(sam_payload)

                mask = sam_out["masks"].sum(0)[0].astype(np.uint8)
                sam_payload["text"] = "end effector"
                sam_payload["confidence"] = 0.4
                sam_out = sam.step(sam_payload)
                ee_mask = sam_out["masks"].sum(0)[0].astype(np.uint8)

                print(ee_mask.max(), ee_mask.min())
                mask = np.logical_and(mask, np.logical_not(ee_mask)).astype(np.uint8)

                data["mask"] = mask
                mask.reshape(-1)

                width, height = frame.shape[1], frame.shape[0]
                fx, fy = 515.0, 515.0  # hardcode for realsense
                cx, cy = width / 2, height / 2  # hardcode for realsense
                _intr = np.array(
                    [
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1],
                    ]
                )

                da_payload = {
                    "image": [frame],  # frames are live from camera
                    # 'intrinsics': np.array([_intr]*len(frames))
                }
                da_out = da.step(da_payload)
                data["depth"] = np.array(Image.fromarray(da_out["depth"][0]).resize((width, height), Image.NEAREST))
                print(spec(data))

                data["intrinsics"] = _intr
                points, colors = da_out["points"], da_out["colors"][:, ::-1]  # BGR to RGB
                datas.append(data)

                cv2.imshow("mask", np.concatenate([mask * 255, ee_mask * 255], axis=1))
                # cv2.imshow("frame", frame)
                cv2.waitKey(1)

                print(data["depth"].shape)

                """
                # pred = depth_anything.infer_image(image, height)
                # Resize depth prediction to match the original image size
                pred = da_out['depth'][0]
                width, height = frame.shape[1], frame.shape[0]
                print(pred.shape, (height, width))
                resized_pred = Image.fromarray(pred).resize((width, height), Image.NEAREST)

                intrinsics = da_out['intrinsics'][0]
                fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                cx, cy = intrinsics[0, 2], intrinsics[1, 2]

                # Generate mesh grid and calculate point cloud coordinates
                x, y = np.meshgrid(np.arange(width), np.arange(height))
                x = (x - cx / 2) / fx
                y = (y - cy / 2) / fy
                z = np.array(resized_pred)
                points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
                colors = (np.array(frame).reshape(-1, 3) / 255.0)[..., ::-1]  # BGR to RGB
                """

                reg_out = roboreg.step(data)

            reg_out = roboreg.step({})
            HT = reg_out.get("HT")
            overlays = reg_out.get("overlays", overlays)
            print({k: type(v) for k, v in reg_out.items()})
            print(overlays)
            print(HT)

            if HT is not None and HT.shape == (4, 4):
                np.savez("HT.npz", HT=HT)

        HT = np.load(cfg.ht / "HT.npz", allow_pickle=True)
        HT = {k: HT[k] for k in HT.files}["HT"]
        print("ht", HT)
        # HT = np.array(
        # [[-0.529 , 0.82 , -0.219 , 0.708],
        # [-0.657 ,-0.559 ,-0.505 , 0.547],
        # [-0.536 ,-0.124 , 0.835 , 0.758],
        # [-0. ,   -0. ,   -0. ,    1.,   ]],
        # )

        if overlays is not None:
            n = len(overlays)
            sqnrt = int(np.ceil(np.sqrt(n)))  # number of overlays per row/column

            def squarecrop(img: np.array) -> np.array:
                h, w = img.shape[:2]
                size = max(h, w)
                new_img = np.zeros((size, size, 3), dtype=img.dtype)
                new_img[(size - h) // 2 : (size - h) // 2 + h, (size - w) // 2 : (size - w) // 2 + w] = img
                return new_img

            # # where mask is true increase red channel by 50%
            # over[..., 2] = np.where(datas[i]['mask'] > 0, np.clip(over[..., 2] + 0.5 * 255, 0, 255), over[..., 2])
            overlays = list(overlays)
            for i, o in enumerate(overlays):
                over = o.astype(np.uint8)
                over[..., 2] = np.where(datas[i]["mask"] > 0, np.clip(over[..., 2] + 0.5 * 255, 0, 255), over[..., 2])
                overlays[i] = over

            overlays = [squarecrop(overlay) for overlay in overlays]
            # pad overlays to make n a perfect square
            while len(overlays) < sqnrt * sqnrt:
                overlays.append(np.zeros_like(overlays[0]))
            # resize overlays to 256x256
            # overlays = [cv2.resize(overlay, (256, 256)) for overlay in overlays]
            # arrange overlays into grid
            overlays = [np.hstack(overlays[i * sqnrt : (i + 1) * sqnrt]) for i in range(sqnrt)]
            overlays = np.vstack(overlays)
            cv2.imshow("overlays", overlays.astype(np.uint8))

            # for i, overlay in enumerate(overlays):
            # over = overlay.astype(np.uint8)
            # cv2.imshow(f"overlay_{i}", over)
            # cv2.imshow(f"overlay_{i}", overlay)
            # cv2.imshow(f"mask_{i}", datas[i]['mask'] * 255)
            cv2.waitKey(1)

        if cfg.do in ["display", "calibrate"]:
            width, height = frame.shape[1], frame.shape[0]
            fx, fy = 515.0, 515.0  # hardcode for realsense
            cx, cy = width / 2, height / 2  # hardcode for realsense
            _intr = np.array(
                [
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1],
                ]
            )

            da_payload = {"image": frames, "intrinsics": np.array([_intr] * len(frames))}
            da_out = da.step(da_payload)
            if False:
                depth = np.array(Image.fromarray(da_out["depth"][0]).resize((width, height), Image.NEAREST))

                x, y = np.meshgrid(np.arange(width), np.arange(height))
                x = (x - cx) / fx
                y = (y - cy) / fy
                z = np.array(depth)
                points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
                colors = (np.array(frame).reshape(-1, 3) / 255.0)[..., ::-1]  # BGR to RGB
            else:
                points, colors = da_out["points"], da_out["colors"][:, ::-1]  # BGR to RGB

        print(spec(da_out))
        for i in range(len(da_out["depth"])):
            d = da_out["depth"][i]
            cmap = 255.0
            d = ((d - np.min(d)) / (np.max(d) - np.min(d)) * cmap).astype(np.uint8)
            d = np.array([d, d, d]).transpose(1, 2, 0)
            rr.log(
                f"world/cam/{cfg.cams[i]}/depth",
                rr.Image(d, color_model="RGB").compress(jpeg_quality=75),
                static=False,
            )

        # print(da_out["extrinsics"])
        # print(da_out["intrinsics"])
        print()

        # points, colors = da_out["points"], da_out["colors"][:, ::-1]  # BGR to RGB

        FLU2RDF = np.array(
            [
                [0, 0, 1, 0],
                [-1, 0, 0, 0],  # -
                [0, -1, 0, 0],  # -
                [0, 0, 0, 1],
            ]
        )
        FLU2RDF = HT @ FLU2RDF
        rr.log(
            "world",
            rr.Transform3D(
                translation=FLU2RDF[:3, 3],
                mat3x3=FLU2RDF[:3, :3],
            ),
            static=True,
        )

        extr = {k: da_out["extrinsics"][i] for i, k in enumerate(cfg.cams)}
        intr = {k: _intr for i, k in enumerate(cfg.cams)}
        info = {
            k: {
                "intrinsics": intr[k],
                "extrinsics": extr[k],
                "width": frames[i].shape[1],
                "height": frames[i].shape[0],
                "frame": frames[i],
            }
            for i, k in enumerate(cfg.cams)
        }

        if begin:
            log_fustrum(info, root=Path("world"))

        # print(len(mask_flat), len(points), len(colors))

        # increase blue 50% where mask is >0
        # colors = np.where(mask_flat[:, None]>0, np.clip(colors + np.array([0, 0, 0.8]), 0, 1), colors)
        # colors = colors[mask_flat > 0]
        # points = points[mask_flat > 0]

        maxd = 1.5
        colors = colors[np.linalg.norm(points, axis=1) < float(maxd)]
        points = points[np.linalg.norm(points, axis=1) < float(maxd)]

        rr.log(
            "world/scene/points",
            rr.Points3D(
                points,
                colors=colors,
                radii=0.002,
            ),
            # static=True,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        begin = False

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(tyro.cli(RoboregConfig))
