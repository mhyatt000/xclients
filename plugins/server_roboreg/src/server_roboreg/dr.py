import importlib
import os

import cv2
import jax
import numpy as np
import pytorch_kinematics as pk
import rich
import rich.progress
import torch
import tyro
from rich import print
from roboreg.losses import soft_dice_loss
from roboreg.util import mask_distance_transform, mask_exponential_decay, overlay_mask
from roboreg.util.factories import create_robot_scene

from server_roboreg.common import REGISTRATION_MODE, DRConfig, HydraConfig
from server_roboreg.render import Renderer, RendererConfig


class DR:
    def __init__(self, cfg: DRConfig, hcfg: HydraConfig):
        self.cfg, self.hcfg = cfg, hcfg

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["MAX_JOBS"] = str(cfg.max_jobs)  # limit number of concurrent jobs
        mode = cfg.mode

        self.modefn = (
            mask_distance_transform
            if mode == REGISTRATION_MODE.DISTANCE_FUNCTION
            else mask_exponential_decay
        )
        self.r = None

    def validate(self, payload: dict):
        images, joints, masks = (payload["depth"], payload["joints"], payload["mask"])
        images = [img.astype(np.float32) for img in images]
        images = [(img - img.min()) / (img.max() - img.min() + 1e-8) for img in images]
        images = [np.stack([img] * 3, axis=-1).astype(np.float32) for img in images]
        joints = [np.array(joint).astype(np.float32) for joint in joints]
        masks = [mask.astype(np.uint8) for mask in masks]

        if not all(mask.dtype == np.uint8 for mask in masks):
            raise ValueError("Masks must be of type np.uint8.")
        if not all(np.all(mask >= 0) and np.all(mask <= 255) for mask in masks):
            raise ValueError("Masks must be in the range [0, 255].")
        if not all(
            mask.shape[:2] == image.shape[:2] for mask, image in zip(masks, images, strict=False)
        ):
            raise ValueError("Mask and image shapes do not match.")
        if not all(mask.ndim == 2 for mask in masks):
            raise ValueError("Masks must be 2D.")
        if not all(image.ndim == 3 for image in images):
            raise ValueError("Images must be 3D.")
        if not all(image.shape[-1] == 3 for image in images):
            raise ValueError("Images must have 3 channels")
        return images, joints, masks

    def step(self, payload: dict) -> dict:
        images, joints, masks = self.validate(payload)

        extrinsics = payload.get("HT")
        print("extrinsics", extrinsics)

        b, h, w = payload["depth"].shape
        if self.r is None:
            self.r = Renderer(
                self.hcfg,
                RendererConfig(
                    batch_size=len(joints),
                ),
                intr=payload["intrinsics"],
                # no extrinsics provided, initialize with identity
                # then optimize with the grad extrinsics
                extr=None,  # extrinsics,
                height=h,
                width=w,
            )

        joints = torch.tensor(np.array(payload["joints"]), dtype=torch.float32, device=self.device)
        if self.cfg.mode == REGISTRATION_MODE.DISTANCE_FUNCTION:
            targets = [mask_distance_transform(mask) for mask in masks]
        elif self.cfg.mode == REGISTRATION_MODE.SEGMENTATION:
            targets = [mask_exponential_decay(mask) for mask in masks]
        else:
            raise ValueError("Invalid registration mode.")
        targets = torch.tensor(
            np.array(targets), dtype=torch.float32, device=self.device
        ).unsqueeze(-1)

        # load extrinsics estimate
        extrinsics = torch.tensor(extrinsics, dtype=torch.float32, device=self.device)
        extrinsics_inv = torch.linalg.inv(extrinsics)

        # enable gradient tracking and instantiate optimizer
        extrinsics_9d_inv = pk.matrix44_to_se3_9d(extrinsics_inv)
        extrinsics_9d_inv.requires_grad = True
        optimizer = getattr(importlib.import_module("torch.optim"), self.cfg.optimizer)(
            [extrinsics_9d_inv], lr=self.cfg.lr
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.cfg.step_size, gamma=self.cfg.gamma
        )
        best_extrinsics = extrinsics
        best_extrinsics_inv = extrinsics_inv
        best_loss = float("inf")

        for iteration in rich.progress.track(
            range(1, self.cfg.max_iterations + 1), "Optimizing..."
        ):
            if not extrinsics_9d_inv.requires_grad:
                raise ValueError("Extrinsics require gradients.")
            if not torch.is_grad_enabled():
                raise ValueError("Gradients must be enabled.")
            extrinsics_inv = pk.se3_9d_to_matrix44(extrinsics_9d_inv)
            self.r.scene.robot.configure(joints, extrinsics_inv)
            renders = {
                "camera": self.r.scene.observe_from("camera"),
            }
            if self.cfg.mode == REGISTRATION_MODE.DISTANCE_FUNCTION:
                loss = torch.nn.functional.mse_loss(targets, renders["camera"])
            elif self.cfg.mode == REGISTRATION_MODE.SEGMENTATION:
                loss = soft_dice_loss(targets, renders["camera"]).mean()
            else:
                raise ValueError("Invalid registration mode.")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            rich.print(
                f"Step [{iteration} / {self.cfg.max_iterations}], loss: {np.round(loss.item(), 3)}, best loss: {np.round(best_loss, 3)}, lr: {scheduler.get_last_lr().pop()}"
            )

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_extrinsics_inv = extrinsics_inv.detach().clone()
                best_extrinsics = torch.linalg.inv(best_extrinsics_inv)

        # render final results and save extrinsics
        with torch.no_grad():
            self.r.scene.robot.configure(joints, best_extrinsics_inv)
            renders = self.r.scene.observe_from("camera").squeeze(-1)

        outs = []
        for i, render in enumerate(renders):
            render = render.cpu().numpy()

            im = images[i]  # im = np.stack([images[i]]*3, axis=-1).astype(np.uint8)
            im = (im - im.min()) / (im.max() - im.min()) * 255.0
            im = im.astype(np.uint8)

            rmask = (render * 255.0).astype(np.uint8)
            print(im.shape, im.dtype, rmask.shape, rmask.dtype)
            print(im.min(), im.max(), rmask.min(), rmask.max())
            print(im.sum(), rmask.sum())
            overlay = overlay_mask(im, rmask, self.r.color, scale=1.0)
            difference = np.abs(render - masks[i].astype(np.float32) / 255.0)
            difference = overlay_mask(
                im,
                (difference * 255.0).astype(np.uint8),
                mode="r",
                scale=1.0,
            )

            out = {
                "overlays": overlay,
                # 'difference': (difference* 255.0).astype(np.uint8),
                "difference": difference,
            }
            outs.append(out)

        outs = jax.tree.map(lambda *x: np.stack(x), *outs)
        outs = outs | {"HT": best_extrinsics.cpu().numpy()}
        return outs


def mono(
    camera,
    images,
    joint_states,
    masks,
) -> None:
    from server_roboreg.old_cli.rr_mono_dr import args_factory

    args = args_factory()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["MAX_JOBS"] = str(args.max_jobs)  # limit number of concurrent jobs
    mode = REGISTRATION_MODE(args.mode)

    # pre-process data
    joint_states = torch.tensor(np.array(joint_states), dtype=torch.float32, device=device)
    if mode == REGISTRATION_MODE.DISTANCE_FUNCTION:
        targets = [mask_distance_transform(mask) for mask in masks]
    elif mode == REGISTRATION_MODE.SEGMENTATION:
        targets = [mask_exponential_decay(mask) for mask in masks]
    else:
        raise ValueError("Invalid registration mode.")
    targets = torch.tensor(np.array(targets), dtype=torch.float32, device=device).unsqueeze(-1)

    # instantiate camera with default identity extrinsics because we optimize for robot pose instead

    # instantiate robot scene
    scene = create_robot_scene(
        batch_size=joint_states.shape[0],
        ros_package=args.ros_package,
        xacro_path=args.xacro_path,
        root_link_name=args.root_link_name,
        end_link_name=args.end_link_name,
        collision=args.collision_meshes,
        cameras=camera,
        device=device,
    )

    # load extrinsics estimate
    extrinsics = torch.tensor(np.load(args.extrinsics_file), dtype=torch.float32, device=device)
    extrinsics_inv = torch.linalg.inv(extrinsics)

    # enable gradient tracking and instantiate optimizer
    extrinsics_9d_inv = pk.matrix44_to_se3_9d(extrinsics_inv)
    extrinsics_9d_inv.requires_grad = True
    optimizer = getattr(importlib.import_module("torch.optim"), args.optimizer)(
        [extrinsics_9d_inv], lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    best_extrinsics = extrinsics
    best_extrinsics_inv = extrinsics_inv
    best_loss = float("inf")

    for iteration in rich.progress.track(range(1, args.max_iterations + 1), "Optimizing..."):
        if not extrinsics_9d_inv.requires_grad:
            raise ValueError("Extrinsics require gradients.")
        if not torch.is_grad_enabled():
            raise ValueError("Gradients must be enabled.")
        extrinsics_inv = pk.se3_9d_to_matrix44(extrinsics_9d_inv)
        scene.robot.configure(joint_states, extrinsics_inv)
        renders = {
            "camera": scene.observe_from("camera"),
        }
        if mode == REGISTRATION_MODE.DISTANCE_FUNCTION:
            loss = torch.nn.functional.mse_loss(targets, renders["camera"])
        elif mode == REGISTRATION_MODE.SEGMENTATION:
            loss = soft_dice_loss(targets, renders["camera"]).mean()
        else:
            raise ValueError("Invalid registration mode.")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        rich.print(
            f"Step [{iteration} / {args.max_iterations}], loss: {np.round(loss.item(), 3)}, best loss: {np.round(best_loss, 3)}, lr: {scheduler.get_last_lr().pop()}"
        )

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_extrinsics_inv = extrinsics_inv.detach().clone()
            best_extrinsics = torch.linalg.inv(best_extrinsics_inv)

        # display optimization progress
        if args.display_progress:
            render = renders["camera"][0].squeeze().detach().cpu().numpy()
            image = images[0]
            render_overlay = overlay_mask(
                image,
                (render * 255.0).astype(np.uint8),
                scale=1.0,
            )
            # difference left / right render / mask
            difference = (
                cv2.cvtColor(
                    np.abs(render - masks[0].astype(np.float32) / 255.0),
                    cv2.COLOR_GRAY2BGR,
                )
                * 255.0
            ).astype(np.uint8)
            # overlay segmentation mask
            segmentation_overlay = overlay_mask(
                image,
                masks[0],
                mode="b",
                scale=1.0,
            )
            cv2.imshow(
                "left to right: render overlay, difference, segmentation overlay",
                cv2.resize(
                    np.hstack(
                        [
                            render_overlay,
                            difference,
                            segmentation_overlay,
                        ]
                    ),
                    (0, 0),
                    fx=0.5,
                    fy=0.5,
                ),
            )
            cv2.waitKey(1)

    # render final results and save extrinsics
    with torch.no_grad():
        scene.robot.configure(joint_states, best_extrinsics_inv)
        renders = scene.observe_from("camera")

    for i, render in enumerate(renders):
        render = render.squeeze().cpu().numpy()
        overlay = overlay_mask(images[i], (render * 255.0).astype(np.uint8), scale=1.0)
        difference = np.abs(render - masks[i].astype(np.float32) / 255.0)

        cv2.imwrite(os.path.join(args.path, f"dr_overlay_{i}.png"), overlay)
        cv2.imwrite(
            os.path.join(args.path, f"dr_difference_{i}.png"),
            (difference * 255.0).astype(np.uint8),
        )

    np.save(
        os.path.join(args.path, args.output_file),
        best_extrinsics.cpu().numpy(),
    )


def main(cfg: DRConfig):
    print(cfg)


if __name__ == "__main__":
    main(tyro.cli(DRConfig))
