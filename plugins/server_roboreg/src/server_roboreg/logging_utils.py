from __future__ import annotations

from collections.abc import Iterable, Iterator
import logging
from pathlib import Path

import numpy as np
import rich
import rich.progress
import torch


def optimization_progress(iterations: Iterable[int]) -> Iterator[int]:
    return rich.progress.track(iterations, "Optimizing...")


def log_renderer_init(
    device: str,
    batch_size: int,
    height: int,
    width: int,
    urdf: Path | None,
) -> None:
    logging.info(
        "Initializing Renderer device=%s batch_size=%d resolution=(%d, %d) urdf=%s",
        device,
        batch_size,
        height,
        width,
        urdf or None,
    )


def log_renderer_urdf(urdf: Path) -> None:
    logging.info("Creating robot scene from local URDF: %s", urdf)


def log_renderer_ros_scene(ros_package: str, xacro_path: str) -> None:
    logging.info("Creating robot scene from ROS package=%s xacro=%s", ros_package, xacro_path)


def log_renderer_step(images_shape: tuple[int, ...], joints_shape: tuple[int, ...]) -> None:
    logging.info("Renderer.step images_shape=%s joints_shape=%s", images_shape, joints_shape)


def log_renderer_sample(
    index: int, image_shape: tuple[int, ...], render_shape: tuple[int, ...], render_mean: float
) -> None:
    logging.info(
        "Renderer.step sample=%d image_shape=%s render_shape=%s render_mean=%.6f",
        index,
        image_shape,
        render_shape,
        render_mean,
    )


def log_dr_payload(
    batch_size: int,
    image_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
    joints_shape: tuple[int, ...],
    ht_shape: tuple[int, ...],
) -> None:
    logging.info(
        "DR payload batch=%d image_shape=%s mask_shape=%s joints_shape=%s ht_shape=%s",
        batch_size,
        image_shape,
        mask_shape,
        joints_shape,
        ht_shape,
    )


def log_dr_setup(
    mode: str,
    optimizer: str,
    lr: float,
    max_iterations: int,
    optimize_cv_w2c: bool,
    optimize_root_transform: bool,
    optimize_intrinsics: bool,
    intrinsics: np.ndarray,
) -> None:
    logging.info(
        "DR setup mode=%s optimizer=%s lr=%.6g iterations=%d optimize_cv_w2c=%s ht_is_root=%s "
        "optimize_intrinsics=%s intrinsics=%s",
        mode,
        optimizer,
        lr,
        max_iterations,
        optimize_cv_w2c,
        optimize_root_transform,
        optimize_intrinsics,
        intrinsics,
    )


def _batch_stats(targets: torch.Tensor, renders: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = targets.detach().squeeze(-1)
    render = renders.detach().squeeze(-1)
    target_bin = target > 0.0
    render_bin = render > 0.5
    inter = (target_bin & render_bin).sum(dim=(1, 2)).float()
    union = (target_bin | render_bin).sum(dim=(1, 2)).float()
    iou = torch.where(union > 0.0, inter / union, torch.zeros_like(union))
    return target.mean(dim=(1, 2)), render.mean(dim=(1, 2)), iou


def print_dr_batch_stats(iteration: int, targets: torch.Tensor, renders: torch.Tensor) -> None:
    for i, (tm, rm, miou) in enumerate(zip(*_batch_stats(targets, renders), strict=True)):
        rich.print(
            f"Step {iteration} sample {i}: target_mean={tm.item():.4f}, "
            f"render_mean={rm.item():.4f}, mask_iou={miou.item():.4f}"
        )


def log_dr_batch_stats(iteration: int, targets: torch.Tensor, renders: torch.Tensor) -> None:
    for i, (tm, rm, miou) in enumerate(zip(*_batch_stats(targets, renders), strict=True)):
        logging.info(
            "DR step=%d sample=%d target_mean=%.6f render_mean=%.6f mask_iou=%.6f",
            iteration,
            i,
            tm.item(),
            rm.item(),
            miou.item(),
        )


def print_dr_step(iteration: int, max_iterations: int, loss: float, best_loss: float, lr: float) -> None:
    rich.print(
        f"Step [{iteration} / {max_iterations}], loss: {np.round(loss, 3)}, "
        f"best loss: {np.round(best_loss, 3)}, lr: {lr}"
    )


def log_dr_step(iteration: int, max_iterations: int, loss: float, best_loss: float, lr: float) -> None:
    logging.info(
        "DR step=%d/%d loss=%.6f best_loss=%.6f lr=%.6g",
        iteration,
        max_iterations,
        loss,
        best_loss,
        lr,
    )


def log_dr_output_sample(
    index: int,
    image_min: int,
    image_max: int,
    render_min: int,
    render_max: int,
    mask_mean: float,
) -> None:
    logging.info(
        "DR output sample=%d image_minmax=(%d,%d) render_minmax=(%d,%d) mask_mean=%.6f",
        index,
        image_min,
        image_max,
        render_min,
        render_max,
        mask_mean,
    )


def log_dr_complete(best_loss: float, ht: np.ndarray) -> None:
    logging.info("DR complete best_loss=%.6f HT=%s", best_loss, ht)
