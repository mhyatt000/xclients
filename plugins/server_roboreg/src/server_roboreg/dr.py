from __future__ import annotations

import importlib
import logging
import os

import cv2
import jax
import numpy as np
import pytorch_kinematics as pk
from roboreg.losses import soft_dice_loss
from roboreg.util import mask_distance_transform, mask_exponential_decay, overlay_mask
import torch
import tyro

from server_roboreg.common import DRConfig, HydraConfig, REGISTRATION_MODE
from server_roboreg.logging_utils import (
    log_dr_batch_stats,
    log_dr_complete,
    log_dr_output_sample,
    log_dr_payload,
    log_dr_setup,
    log_dr_step,
    optimization_progress,
    print_dr_batch_stats,
    print_dr_step,
)
from server_roboreg.render import Renderer, RendererConfig


def opencv_projection(intr: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if intr.ndim == 2:
        projection = torch.zeros(4, 4, dtype=intr.dtype, device=intr.device)
    elif intr.ndim == 3:
        projection = torch.zeros((intr.shape[0], 4, 4), dtype=intr.dtype, device=intr.device)
    else:
        raise ValueError(f"Expected intrinsics with shape (3, 3) or (B, 3, 3), got {tuple(intr.shape)}")

    znear, zfar = 0.01, 10.0
    projection[..., 0, 0] = 2.0 * intr[..., 0, 0] / width
    projection[..., 1, 1] = 2.0 * intr[..., 1, 1] / height
    projection[..., 0, 2] = 1.0 - 2.0 * intr[..., 0, 2] / width
    projection[..., 1, 2] = 2.0 * intr[..., 1, 2] / height - 1.0
    projection[..., 2, 2] = -(zfar + znear) / (zfar - znear)
    projection[..., 2, 3] = -2.0 * zfar * znear / (zfar - znear)
    projection[..., 3, 2] = -1.0
    return projection


def render_cv_w2c(
    renderer: Renderer,
    joints: torch.Tensor,
    w2c: torch.Tensor,
    intr: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    renderer.scene.robot.configure(joints)
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=w2c.dtype, device=w2c.device))
    mvp = opencv_projection(intr, width, height) @ (flip @ w2c)
    observed_vertices = torch.matmul(renderer.scene.robot.configured_vertices, mvp.transpose(-1, -2))
    faces = renderer.scene.robot.mesh_container.faces
    render = renderer.scene.renderer.constant_color(
        observed_vertices,
        faces,
        renderer.scene.cameras[renderer.camera_name].resolution,
    )
    return torch.flip(render, dims=[1])


class DR:
    def __init__(self, cfg: DRConfig, hcfg: HydraConfig):
        self.cfg, self.hcfg = cfg, hcfg

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["MAX_JOBS"] = str(cfg.max_jobs)  # limit number of concurrent jobs
        mode = cfg.mode

        self.modefn = mask_distance_transform if mode == REGISTRATION_MODE.DISTANCE_FUNCTION else mask_exponential_decay
        self.r = None

    @staticmethod
    def _as_batch(value: object, name: str) -> np.ndarray:
        arr = np.asarray(value)
        if arr.ndim == 2:
            arr = arr[None]
        if arr.ndim == 3 and name in {"image", "images"} and arr.shape[-1] in (1, 3, 4):
            arr = arr[None]
        if arr.ndim < 3:
            raise ValueError(f"Expected batched {name}, got shape {arr.shape}")
        return arr

    @staticmethod
    def _normalize_image(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected image with shape (h, w, 3), got {image.shape}")

        image = image.astype(np.float32)
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return np.zeros_like(image, dtype=np.float32)
        lo = float(finite.min())
        hi = float(finite.max())
        if hi > lo:
            image = (image - lo) / (hi - lo)
        elif hi > 1.0:
            image = image / 255.0
        return np.clip(image, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _distance_target(mask: np.ndarray) -> np.ndarray:
        target = mask_distance_transform(mask).astype(np.float32)
        max_value = float(target.max())
        if max_value > 0.0:
            target /= max_value
        return target

    def _opencv_projection(self, intr: torch.Tensor, width: int, height: int) -> torch.Tensor:
        return opencv_projection(intr, width, height)

    def _render_cv_w2c(
        self,
        joints: torch.Tensor,
        w2c: torch.Tensor,
        intr: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        return render_cv_w2c(self.r, joints, w2c, intr, height, width)

    def validate(self, payload: dict):
        if "images" in payload:
            images_raw = self._as_batch(payload["images"], "images")
        elif "image" in payload:
            images_raw = self._as_batch(payload["image"], "image")
        elif "depth" in payload:
            images_raw = self._as_batch(payload["depth"], "depth")
        else:
            images_raw = self._as_batch(payload["mask"], "mask")

        images = [self._normalize_image(img) for img in images_raw]
        joints = [np.array(joint).astype(np.float32) for joint in payload["joints"]]
        masks = [mask.astype(np.uint8) for mask in self._as_batch(payload["mask"], "mask")]

        if not all(mask.ndim == 2 for mask in masks):
            raise ValueError("Masks must be 2D.")

        if not all(mask.dtype == np.uint8 for mask in masks):
            raise ValueError("Masks must be of type np.uint8.")
        if not all(np.all(mask >= 0) and np.all(mask <= 255) for mask in masks):
            raise ValueError("Masks must be in the range [0, 255].")
        if not all(mask.shape[:2] == image.shape[:2] for mask, image in zip(masks, images, strict=False)):
            raise ValueError("Mask and image shapes do not match.")
        if not all(image.ndim == 3 for image in images):
            raise ValueError("Images must be 3D.")
        if not all(image.shape[-1] == 3 for image in images):
            raise ValueError("Images must have 3 channels")
        if len(images) != len(joints) or len(images) != len(masks):
            raise ValueError(f"Batch mismatch: images={len(images)} joints={len(joints)} masks={len(masks)}")
        return images, joints, masks

    def step(self, payload: dict) -> dict:
        images, joints, masks = self.validate(payload)

        extrinsics = payload.get("HT")
        if extrinsics is None:
            raise KeyError("DR payload must include an initial HT transform.")

        h, w = masks[0].shape[:2]
        log_dr_payload(
            len(images),
            images[0].shape,
            masks[0].shape,
            np.asarray(payload["joints"]).shape,
            np.asarray(extrinsics).shape,
        )
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
            targets = [self._distance_target(mask) for mask in masks]
        elif self.cfg.mode == REGISTRATION_MODE.SEGMENTATION:
            targets = [mask_exponential_decay(mask) for mask in masks]
        else:
            raise ValueError("Invalid registration mode.")
        targets = torch.tensor(np.array(targets), dtype=torch.float32, device=self.device).unsqueeze(-1)

        # load extrinsics estimate
        extrinsics = torch.tensor(extrinsics, dtype=torch.float32, device=self.device)
        optimize_cv_w2c = bool(payload.get("ht_is_cv_w2c", False))
        optimize_root_transform = bool(payload.get("ht_is_root", False))
        root_transform = extrinsics if optimize_cv_w2c or optimize_root_transform else torch.linalg.inv(extrinsics)
        intr = torch.tensor(payload["intrinsics"], dtype=torch.float32, device=self.device)
        if intr.shape[-2:] != (3, 3):
            raise ValueError(f"Expected intrinsics with shape (3, 3) or (B, 3, 3), got {tuple(intr.shape)}")
        if intr.ndim == 3 and intr.shape[0] != len(joints):
            raise ValueError(f"Batched intrinsics count {intr.shape[0]} does not match batch size {len(joints)}")
        log_dr_setup(
            self.cfg.mode.value,
            self.cfg.optimizer,
            self.cfg.lr,
            self.cfg.max_iterations,
            optimize_cv_w2c,
            optimize_root_transform,
            self.cfg.optimize_intrinsics,
            intr.detach().cpu().numpy(),
        )

        # enable gradient tracking and instantiate optimizer
        root_transform_9d = pk.matrix44_to_se3_9d(root_transform)
        root_transform_9d.requires_grad = True
        intr.requires_grad = self.cfg.optimize_intrinsics

        optim_params = [root_transform_9d]
        if self.cfg.optimize_intrinsics:
            optim_params.append(intr)
        optimizer = getattr(importlib.import_module("torch.optim"), self.cfg.optimizer)(
            optim_params,
            lr=self.cfg.lr,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.step_size, gamma=self.cfg.gamma)
        best_extrinsics = extrinsics
        best_root_transform = root_transform
        best_intr = intr.detach().clone()
        best_loss = float("inf")
        steps_without_improvement = 0
        early_stop_patience = max(0, int(self.cfg.early_stop_patience))
        early_stop_min_delta = max(0.0, float(self.cfg.early_stop_min_delta))

        for iteration in optimization_progress(range(1, self.cfg.max_iterations + 1)):
            if not root_transform_9d.requires_grad:
                raise ValueError("Extrinsics require gradients.")
            if not torch.is_grad_enabled():
                raise ValueError("Gradients must be enabled.")
            root_transform = pk.se3_9d_to_matrix44(root_transform_9d)
            if optimize_cv_w2c:
                renders = {"camera": self._render_cv_w2c(joints, root_transform, intr, h, w)}
            else:
                self.r.scene.robot.configure(joints, root_transform)
                renders = {
                    "camera": self.r.scene.observe_from("camera"),
                }

            if self.cfg.mode == REGISTRATION_MODE.DISTANCE_FUNCTION:
                loss = torch.nn.functional.mse_loss(targets, renders["camera"])
            elif self.cfg.mode == REGISTRATION_MODE.SEGMENTATION:
                loss = soft_dice_loss(targets, renders["camera"]).mean()
            else:
                raise ValueError("Invalid registration mode.")

            if iteration == 1 or iteration == self.cfg.max_iterations or iteration % self.cfg.step_size == 0:
                print_dr_batch_stats(iteration, targets, renders["camera"])
                log_dr_batch_stats(iteration, targets, renders["camera"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

            print_dr_step(iteration, self.cfg.max_iterations, loss.item(), best_loss, lr)
            log_dr_step(
                iteration,
                self.cfg.max_iterations,
                loss.item(),
                best_loss,
                lr,
            )

            loss_value = loss.item()
            if best_loss - loss_value > early_stop_min_delta:
                best_loss = loss_value
                best_root_transform = root_transform.detach().clone()
                best_intr = intr.detach().clone()
                best_extrinsics = (
                    best_root_transform
                    if optimize_cv_w2c or optimize_root_transform
                    else torch.linalg.inv(best_root_transform)
                )
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            if early_stop_patience and steps_without_improvement >= early_stop_patience:
                logging.info(
                    "Stopping DR early at iteration %d/%d: no loss improvement > %.3g for %d steps",
                    iteration,
                    self.cfg.max_iterations,
                    early_stop_min_delta,
                    early_stop_patience,
                )
                break

        # render final results and save extrinsics
        with torch.no_grad():
            if optimize_cv_w2c:
                renders = self._render_cv_w2c(joints, best_root_transform, best_intr, h, w).squeeze(-1)
            else:
                self.r.scene.robot.configure(joints, best_root_transform)
                renders = self.r.scene.observe_from("camera").squeeze(-1)

        outs = []
        for i, render in enumerate(renders):
            render = render.cpu().numpy()

            im = images[i]  # im = np.stack([images[i]]*3, axis=-1).astype(np.uint8)
            im = (im - im.min()) / (im.max() - im.min() + 1e-8) * 255.0
            im = im.astype(np.uint8)

            rmask = (render * 255.0).astype(np.uint8)
            rimg = np.stack([rmask, rmask, rmask], axis=-1)
            log_dr_output_sample(
                i,
                int(im.min()),
                int(im.max()),
                int(rmask.min()),
                int(rmask.max()),
                float(rmask.mean() / 255.0),
            )
            overlay = overlay_mask(im, rmask, self.r.color, scale=1.0)
            rast_overlay = overlay_mask(rimg, masks[i], mode="g", scale=1.0)
            difference = np.abs(render - masks[i].astype(np.float32) / 255.0)
            difference = overlay_mask(
                im,
                (difference * 255.0).astype(np.uint8),
                mode="r",
                scale=1.0,
            )

            out = {
                "overlays": overlay,
                "renders": rmask,
                "render_overlays": rast_overlay,
                # 'difference': (difference* 255.0).astype(np.uint8),
                "difference": difference,
            }
            outs.append(out)

        outs = jax.tree.map(lambda *x: np.stack(x), *outs)
        outs = outs | {"HT": best_extrinsics.cpu().numpy()}
        log_dr_complete(best_loss, outs["HT"])
        return outs


def mono(
    camera,
    images,
    joint_states,
    masks,
) -> None:
    from roboreg.util.factories import create_robot_scene

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
    optimizer = getattr(importlib.import_module("torch.optim"), args.optimizer)([extrinsics_9d_inv], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    best_extrinsics = extrinsics
    best_extrinsics_inv = extrinsics_inv
    best_loss = float("inf")

    for iteration in optimization_progress(range(1, args.max_iterations + 1)):
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
        lr = scheduler.get_last_lr()[0]

        print_dr_step(iteration, args.max_iterations, loss.item(), best_loss, lr)

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
