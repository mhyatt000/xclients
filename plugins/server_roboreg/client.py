from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from server_roboreg.common import HydraConfig
from server_roboreg.render import Renderer, RendererConfig
import torch
import tyro
from webpolicy.client import Client

HEIGHT = 480
WIDTH = 640


@dataclass
class ClientConfig:
    host: str = "127.0.0.1"
    port: int = 8022
    urdf: Path = Path("xarm7_standalone.urdf")
    xyz_noise: float = 0.01
    quat_noise: float = 0.005
    seed: int = 7
    n: int = 2


def normalize_quat(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    trace = np.trace(r)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        return normalize_quat(
            np.array(
                [
                    0.25 * s,
                    (r[2, 1] - r[1, 2]) / s,
                    (r[0, 2] - r[2, 0]) / s,
                    (r[1, 0] - r[0, 1]) / s,
                ],
                dtype=np.float32,
            )
        )

    i = int(np.argmax(np.diag(r)))
    if i == 0:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [((r[2, 1] - r[1, 2]) / s), 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif i == 1:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [((r[0, 2] - r[2, 0]) / s), (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [((r[1, 0] - r[0, 1]) / s), (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    return normalize_quat(np.array(q, dtype=np.float32))


def xyzquat_to_ht(xyz: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    ht = np.eye(4, dtype=np.float32)
    ht[:3, :3] = quat_to_matrix(quat_wxyz)
    ht[:3, 3] = xyz
    return ht


def look_at_camera_link_pose(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    left = np.cross(world_up, forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)

    ht = np.eye(4, dtype=np.float32)
    ht[:3, :3] = np.stack([forward, left, up], axis=1)
    ht[:3, 3] = eye
    return ht


def noisy_initial_ht(true_ht: np.ndarray, xyz_noise: float, quat_noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xyz = true_ht[:3, 3] + rng.normal(0.0, xyz_noise, size=3).astype(np.float32)
    quat = matrix_to_quat(true_ht[:3, :3])
    quat = normalize_quat(quat + rng.normal(0.0, quat_noise, size=4).astype(np.float32))
    return xyzquat_to_ht(xyz, quat)


def make_intrinsics() -> np.ndarray:
    return np.array(
        [
            [515.0, 0.0, WIDTH / 2],
            [0.0, 515.0, HEIGHT / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def make_joints(n: int) -> np.ndarray:
    base = np.array([0.0, -0.55, 0.0, 0.75, 0.0, 1.05, 0.0], dtype=np.float32)
    delta = np.array([0.15, -0.10, 0.10, 0.10, -0.10, -0.10, 0.12], dtype=np.float32)
    if n <= 0:
        raise ValueError(f"Expected n > 0, got {n}.")
    if n == 1:
        return base[None]
    alpha = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    return base + alpha * delta


def render_masks(extr: np.ndarray, intr: np.ndarray, joints: np.ndarray, urdf: Path) -> np.ndarray:
    cfg = HydraConfig(urdf=urdf, end_link_name="link7")
    renderer = Renderer(
        cfg,
        RendererConfig(batch_size=len(joints)),
        height=HEIGHT,
        width=WIDTH,
        intr=intr,
        extr=extr,
    )

    joint_t = torch.as_tensor(joints, dtype=torch.float32, device=renderer.device)
    renderer.scene.robot.configure(joint_t)
    masks = renderer.scene.observe_from("camera").squeeze(-1).detach().cpu().numpy()
    return (masks > 0.5).astype(np.uint8) * 255


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_b = pred > 0
    gt_b = gt > 0
    intersection = np.logical_and(pred_b, gt_b).reshape(pred.shape[0], -1).sum(axis=1)
    union = np.logical_or(pred_b, gt_b).reshape(pred.shape[0], -1).sum(axis=1)
    return intersection / np.maximum(union, 1)


def main() -> None:
    cfg = tyro.cli(ClientConfig)

    intr = make_intrinsics()
    joints = make_joints(cfg.n)
    depth = np.zeros((cfg.n, HEIGHT, WIDTH), dtype=np.float32)

    true_ht = look_at_camera_link_pose(
        eye=np.array([0.75, -0.45, 0.55], dtype=np.float32),
        target=np.array([0.25, 0.0, 0.25], dtype=np.float32),
    )
    init_ht = noisy_initial_ht(true_ht, cfg.xyz_noise, cfg.quat_noise, cfg.seed)

    masks = render_masks(true_ht, intr, joints, cfg.urdf)
    if not np.any(masks):
        raise RuntimeError("Rendered masks are empty. Adjust the true camera pose or joint seed.")

    payload = {
        "depth": depth,
        "joints": joints,
        "mask": masks,
        "intrinsics": intr,
        "HT": init_ht,
    }

    print("true HT:\n", true_ht)
    print("initial HT:\n", init_ht)
    print("mask pixels:", masks.reshape(masks.shape[0], -1).sum(axis=1) // 255)

    client = Client(cfg.host, cfg.port)
    out = client.step(payload)
    solved_ht = out["HT"]
    initial_masks = render_masks(init_ht, intr, joints, cfg.urdf)
    solved_masks = render_masks(solved_ht, intr, joints, cfg.urdf)

    print("server keys:", out.keys())
    print("initial IoU:", mask_iou(initial_masks, masks))
    print("solved IoU:", mask_iou(solved_masks, masks))
    print("solved HT:\n", solved_ht.round(2))


if __name__ == "__main__":
    main()
