from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="renders/cube_views")
    parser.add_argument("--output-dir", default="renders/smoke_test")
    return parser.parse_args()


def project_world_to_pixel(world_xyz: list[float], world_to_camera: list[list[float]], k: list[list[float]]) -> tuple[float, float] | None:
    world = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0], dtype=np.float64)
    w2c = np.array(world_to_camera, dtype=np.float64)
    cam = world @ w2c
    depth = -cam[2]
    if depth <= 1e-6:
        return None
    k_mat = np.array(k, dtype=np.float64)
    norm = np.array([cam[0] / depth, cam[1] / depth, 1.0], dtype=np.float64)
    pix = k_mat @ norm
    x = float(pix[0])
    y = float(2.0 * k_mat[1, 2] - pix[1])
    return x, y


def draw_circle(img: np.ndarray, xy: tuple[float, float], color: tuple[int, int, int]) -> None:
    center = (int(round(xy[0])), int(round(xy[1])))
    cv2.circle(img, center, 2, color, thickness=1, lineType=cv2.LINE_AA)


def draw_cross(img: np.ndarray, xy: tuple[float, float], color: tuple[int, int, int]) -> None:
    center = (int(round(xy[0])), int(round(xy[1])))
    size = 3
    cv2.line(img, (center[0] - size, center[1] - size), (center[0] + size, center[1] + size), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, (center[0] - size, center[1] + size), (center[0] + size, center[1] - size), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, (center[0] - size, center[1] - size), (center[0] + size, center[1] + size), color, 1, cv2.LINE_AA)
    cv2.line(img, (center[0] - size, center[1] + size), (center[0] + size, center[1] - size), color, 1, cv2.LINE_AA)


def draw_label(img: np.ndarray, text: str, xy: tuple[float, float], color: tuple[int, int, int]) -> None:
    origin = (int(round(xy[0] + 8)), int(round(xy[1] - 8)))
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_world_frame(image: np.ndarray, world_to_camera: list[list[float]], k: list[list[float]]) -> None:
    origin = project_world_to_pixel([0.0, 0.0, 0.0], world_to_camera, k)
    x_tip = project_world_to_pixel([0.15, 0.0, 0.0], world_to_camera, k)
    y_tip = project_world_to_pixel([0.0, 0.15, 0.0], world_to_camera, k)
    z_tip = project_world_to_pixel([0.0, 0.0, 0.15], world_to_camera, k)

    if origin is None:
        return

    origin_px = (int(round(origin[0])), int(round(origin[1])))
    cv2.circle(image, origin_px, 6, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, origin_px, 4, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
    draw_label(image, "O", origin, (0, 0, 255))

    axes = [
        ("X", x_tip, (0, 0, 255)),
        ("Y", y_tip, (0, 255, 0)),
        ("Z", z_tip, (255, 0, 0)),
    ]
    for label, tip, color in axes:
        if tip is None:
            continue
        tip_px = (int(round(tip[0])), int(round(tip[1])))
        cv2.line(image, origin_px, tip_px, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.line(image, origin_px, tip_px, color, 2, cv2.LINE_AA)
        cv2.circle(image, tip_px, 4, color, thickness=-1, lineType=cv2.LINE_AA)
        draw_label(image, label, tip, color)


def overlay_one(image_path: Path, json_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read {image_path}")
    payload = json.loads(json_path.read_text())
    k = payload["camera"]["intrinsics"]["K"]
    world_to_camera = payload["camera"]["extrinsics"]["world_to_camera"]

    draw_world_frame(image, world_to_camera, k)

    for keypoint in payload.get("keypoints", []):
        label = keypoint["name"]
        pixel_xy = keypoint.get("pixel_xy")
        world_xyz = keypoint.get("world_xyz")
        visible = bool(keypoint.get("visible", False))

        if pixel_xy is not None:
            draw_circle(image, (float(pixel_xy[0]), float(pixel_xy[1])), (0, 255, 255) if visible else (0, 128, 255))
            draw_label(image, label, (float(pixel_xy[0]), float(pixel_xy[1])), (0, 255, 255) if visible else (0, 128, 255))

        if world_xyz is not None:
            reproj = project_world_to_pixel(world_xyz, world_to_camera, k)
            if reproj is not None:
                draw_cross(image, reproj, (255, 0, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write {output_path}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(input_dir.glob("view_*.png"))
    if not images:
        raise RuntimeError(f"No input images found in {input_dir}")

    for image_path in images:
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            continue
        output_path = output_dir / image_path.name
        overlay_one(image_path, json_path, output_path)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
