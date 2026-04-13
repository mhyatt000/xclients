from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import isaacsim
from isaacsim.simulation_app import SimulationApp


KIT_EXPERIENCE = Path(isaacsim.__file__).resolve().parent / "kit/apps/omni.app.hydra.kit"
IMAGE_SIZE = 512

APP = SimulationApp(
    {
        "headless": True,
        "width": IMAGE_SIZE,
        "height": IMAGE_SIZE,
        "renderer": "RayTracedLighting",
        "multi_gpu": False,
    },
    experience=os.fspath(KIT_EXPERIENCE),
)

import omni.kit.app
import omni.kit.commands
import omni.usd
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux


def pump(frames: int = 1) -> None:
    for _ in range(frames):
        APP.update()


def create_stage() -> tuple[Usd.Stage, Sdf.Path]:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    cube.CreateSizeAttr(140.0)
    cube.AddTranslateOp().Set((0.0, 0.0, 70.0))
    UsdGeom.XformCommonAPI(cube).SetRotate((18.0, 28.0, 12.0))
    cube_prim = cube.GetPrim()
    cube_prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set(
        [Gf.Vec3f(0.95, 0.15, 0.1)]
    )

    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    ground.CreatePointsAttr(
        [
            (-500.0, -500.0, 0.0),
            (500.0, -500.0, 0.0),
            (500.0, 500.0, 0.0),
            (-500.0, 500.0, 0.0),
        ]
    )
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.GetPrim().CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set(
        [Gf.Vec3f(0.6, 0.62, 0.65)]
    )

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1500.0)

    sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    sun.CreateIntensityAttr(4000.0)
    UsdGeom.XformCommonAPI(sun).SetRotate((315.0, 0.0, 0.0))

    stage.Save()
    return stage, cube.GetPath()


def create_camera(stage: Usd.Stage, eye: tuple[float, float, float], target: tuple[float, float, float], name: str) -> Sdf.Path:
    camera = UsdGeom.Camera.Define(stage, f"/World/{name}")
    camera.CreateFocalLengthAttr(35.0)
    camera.CreateFocusDistanceAttr(700.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(1.0, 10000.0))

    m = Gf.Matrix4d(1.0)
    m.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 0.0, 1.0))
    camera.MakeMatrixXform().Set(m.GetInverse())
    return camera.GetPath()


def frame_cube(viewport, camera_path: Sdf.Path, cube_path: Sdf.Path) -> None:
    viewport.camera_path = camera_path
    pump(8)
    omni.kit.commands.execute(
        "FramePrimsCommand",
        prim_to_move=str(camera_path),
        prims_to_frame=[str(cube_path)],
        time_code=Usd.TimeCode.Default(),
        aspect_ratio=1.0,
        zoom=0.55,
    )
    pump(8)


def wait_for_capture(path: Path) -> None:
    for _ in range(240):
        pump(1)
        if path.exists() and path.stat().st_size > 0:
            return
    raise RuntimeError(f"Timed out waiting for {path}")


def image_stats(path: Path) -> tuple[int, int, float]:
    image = Image.open(path).convert("RGB")
    extrema = image.getextrema()
    mins = [mn for mn, _ in extrema]
    maxs = [mx for _, mx in extrema]
    luminance = list(image.convert("L").getdata(band=0))
    mean = sum(luminance) / len(luminance)
    return min(mins), max(maxs), mean


def orbit_positions(radius: float, height: float, count: int) -> list[tuple[float, float, float]]:
    positions = []
    for i in range(count):
        theta = (2.0 * math.pi * i) / count
        positions.append((radius * math.cos(theta), radius * math.sin(theta), height))
    return positions


def main() -> None:
    out_dir = Path("renders/cube_views").resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {out_dir}")

    stage, cube_path = create_stage()
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active viewport available")

    pump(24)
    camera_positions = orbit_positions(radius=520.0, height=260.0, count=5)
    target = (0.0, 0.0, 70.0)

    for i, eye in enumerate(camera_positions):
        image_path = out_dir / f"view_{i}.png"
        camera_path = create_camera(stage, eye, target, f"Camera_{i}")
        frame_cube(viewport, camera_path, cube_path)
        if image_path.exists():
            image_path.unlink()
        capture_viewport_to_file(viewport, os.fspath(image_path))
        wait_for_capture(image_path)
        mn, mx, mean = image_stats(image_path)
        print(f"view_{i}: min={mn} max={mx} mean={mean:.2f}")
        if mx == 0:
            raise RuntimeError(f"{image_path.name} is fully black")
        pump(4)

    print(f"Wrote renders to {out_dir}")
    stage = None
    viewport = None
    omni.usd.get_context().close_stage()
    APP.close()


if __name__ == "__main__":
    main()
