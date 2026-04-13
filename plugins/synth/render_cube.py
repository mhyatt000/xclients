from __future__ import annotations

import math
import os
import random
import shutil
import colorsys
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
import omni.usd
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from PIL import Image, ImageStat
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade


VIEW_COUNT = 5


def pump(frames: int = 1) -> None:
    for _ in range(frames):
        APP.update()


def make_material(stage: Usd.Stage, path: str) -> tuple[UsdShade.Material, UsdShade.Shader]:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.7, 0.7, 0.7))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material, shader


def bind_material(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI(prim).Bind(material)


def create_stage() -> tuple[Usd.Stage, dict[str, object]]:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    UsdGeom.Xform.Define(stage, "/World/Looks")

    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    cube.CreateSizeAttr(140.0)
    cube.AddTranslateOp().Set((0.0, 0.0, 70.0))
    cube_api = UsdGeom.XformCommonAPI(cube)

    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    ground.CreatePointsAttr(
        [
            (-900.0, -900.0, 0.0),
            (900.0, -900.0, 0.0),
            (900.0, 900.0, 0.0),
            (-900.0, 900.0, 0.0),
        ]
    )
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    backdrop = UsdGeom.Mesh.Define(stage, "/World/Backdrop")
    backdrop.CreatePointsAttr(
        [
            (-1200.0, -1000.0, 0.0),
            (1200.0, -1000.0, 0.0),
            (1200.0, -1000.0, 1400.0),
            (-1200.0, -1000.0, 1400.0),
        ]
    )
    backdrop.CreateFaceVertexCountsAttr([4])
    backdrop.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

    cube_mat, cube_shader = make_material(stage, "/World/Looks/CubeMat")
    ground_mat, ground_shader = make_material(stage, "/World/Looks/GroundMat")
    backdrop_mat, backdrop_shader = make_material(stage, "/World/Looks/BackdropMat")
    bind_material(cube.GetPrim(), cube_mat)
    bind_material(ground.GetPrim(), ground_mat)
    bind_material(backdrop.GetPrim(), backdrop_mat)

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1500.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    sun.CreateIntensityAttr(4000.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    sun_api = UsdGeom.XformCommonAPI(sun)
    sun_api.SetRotate((315.0, 0.0, 0.0))

    stage.Save()
    scene = {
        "cube": cube,
        "cube_api": cube_api,
        "cube_path": cube.GetPath(),
        "cube_shader": cube_shader,
        "ground_shader": ground_shader,
        "backdrop_shader": backdrop_shader,
        "dome": dome,
        "sun": sun,
        "sun_api": sun_api,
    }
    return stage, scene


def create_camera(stage: Usd.Stage, eye: tuple[float, float, float], target: tuple[float, float, float], name: str) -> Sdf.Path:
    camera = UsdGeom.Camera.Define(stage, f"/World/{name}")
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateFocusDistanceAttr(700.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(1.0, 10000.0))

    m = Gf.Matrix4d(1.0)
    m.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 0.0, 1.0))
    camera.MakeMatrixXform().Set(m.GetInverse())
    return camera.GetPath()


def frame_cube(viewport, camera_path: Sdf.Path, cube_path: Sdf.Path) -> None:
    viewport.camera_path = camera_path
    pump(16)


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
    mean = ImageStat.Stat(image.convert("L")).mean[0]
    return min(mins), max(maxs), mean


def vec3(color: tuple[float, float, float]) -> Gf.Vec3f:
    return Gf.Vec3f(*color)


def rand_color(rng: random.Random, lo: float = 0.15, hi: float = 0.95) -> tuple[float, float, float]:
    return tuple(rng.uniform(lo, hi) for _ in range(3))


def rand_vivid_color(rng: random.Random) -> tuple[float, float, float]:
    hue = rng.random()
    sat = rng.uniform(0.55, 0.9)
    val = rng.uniform(0.72, 0.98)
    return colorsys.hsv_to_rgb(hue, sat, val)


def rand_muted_color(rng: random.Random) -> tuple[float, float, float]:
    hue = rng.random()
    sat = rng.uniform(0.12, 0.35)
    val = rng.uniform(0.35, 0.72)
    return colorsys.hsv_to_rgb(hue, sat, val)


def apply_domain_randomization(scene: dict[str, object], rng: random.Random, view_index: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    cube_api: UsdGeom.XformCommonAPI = scene["cube_api"]  # type: ignore[assignment]
    cube_shader: UsdShade.Shader = scene["cube_shader"]  # type: ignore[assignment]
    ground_shader: UsdShade.Shader = scene["ground_shader"]  # type: ignore[assignment]
    backdrop_shader: UsdShade.Shader = scene["backdrop_shader"]  # type: ignore[assignment]
    dome: UsdLux.DomeLight = scene["dome"]  # type: ignore[assignment]
    sun: UsdLux.DistantLight = scene["sun"]  # type: ignore[assignment]
    sun_api: UsdGeom.XformCommonAPI = scene["sun_api"]  # type: ignore[assignment]

    cube_translate = (
        rng.uniform(-45.0, 45.0),
        rng.uniform(-35.0, 35.0),
        rng.uniform(70.0, 105.0),
    )
    cube_rotate = (
        rng.uniform(-20.0, 35.0),
        rng.uniform(5.0, 55.0),
        rng.uniform(-35.0, 35.0),
    )
    cube_scale = rng.uniform(0.8, 1.35)
    cube_api.SetTranslate(cube_translate)
    cube_api.SetRotate(cube_rotate)
    cube_api.SetScale((cube_scale, cube_scale, cube_scale))

    cube_color = rand_vivid_color(rng)
    ground_color = rand_muted_color(rng)
    backdrop_color = rand_muted_color(rng)
    cube_shader.GetInput("diffuseColor").Set(vec3(cube_color))
    cube_shader.GetInput("roughness").Set(rng.uniform(0.08, 0.55))
    cube_shader.GetInput("metallic").Set(rng.uniform(0.0, 0.15))
    ground_shader.GetInput("diffuseColor").Set(vec3(ground_color))
    ground_shader.GetInput("roughness").Set(rng.uniform(0.4, 0.95))
    backdrop_shader.GetInput("diffuseColor").Set(vec3(backdrop_color))
    backdrop_shader.GetInput("roughness").Set(rng.uniform(0.5, 0.95))

    dome.GetIntensityAttr().Set(rng.uniform(350.0, 1200.0))
    dome.GetColorAttr().Set(vec3(rand_color(rng, 0.75, 1.0)))
    sun.GetIntensityAttr().Set(rng.uniform(5000.0, 14000.0))
    sun.GetColorAttr().Set(vec3(rand_color(rng, 0.7, 1.0)))
    sun_api.SetRotate((rng.uniform(230.0, 330.0), rng.uniform(-25.0, 25.0), 0.0))

    azimuth = (360.0 * view_index / VIEW_COUNT) + rng.uniform(-18.0, 18.0)
    radius = rng.uniform(480.0, 620.0)
    height = rng.uniform(180.0, 280.0)
    theta = math.radians(azimuth)
    eye = (radius * math.cos(theta), radius * math.sin(theta), height)
    target = cube_translate
    return eye, target


def main() -> None:
    out_dir = Path("renders/cube_views").resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {out_dir}")

    stage, scene = create_stage()
    cube_path: Sdf.Path = scene["cube_path"]  # type: ignore[assignment]
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active viewport available")

    pump(24)
    rng = random.Random(20260413)

    for i in range(VIEW_COUNT):
        image_path = out_dir / f"view_{i}.png"
        eye, target = apply_domain_randomization(scene, rng, i)
        camera_path = create_camera(stage, eye, target, f"Camera_{i}")
        frame_cube(viewport, camera_path, cube_path)
        if image_path.exists():
            image_path.unlink()
        capture_viewport_to_file(viewport, os.fspath(image_path))
        wait_for_capture(image_path)
        mn, mx, mean = image_stats(image_path)
        print(f"view_{i}: eye={tuple(round(v, 1) for v in eye)} target={tuple(round(v, 1) for v in target)} min={mn} max={mx} mean={mean:.2f}")
        if mx == 0 or mean < 8.0:
            raise RuntimeError(f"{image_path.name} is fully black")
        pump(4)

    print(f"Wrote renders to {out_dir}")
    stage = None
    viewport = None
    omni.usd.get_context().close_stage()
    APP.close()


if __name__ == "__main__":
    main()
