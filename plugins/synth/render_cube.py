from __future__ import annotations

import argparse
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
ROBOT_LINK_PATHS = [
    "Geometry/world/link_base/link1",
    "Geometry/world/link_base/link1/link2",
    "Geometry/world/link_base/link1/link2/link3",
    "Geometry/world/link_base/link1/link2/link3/link4",
    "Geometry/world/link_base/link1/link2/link3/link4/link5",
    "Geometry/world/link_base/link1/link2/link3/link4/link5/link6",
    "Geometry/world/link_base/link1/link2/link3/link4/link5/link6/link7",
]
ROBOT_GRIPPER_PATHS = [
    "Geometry/world/link_base/link1/link2/link3/link4/link5/link6/link7/link_eef/xarm_gripper_base_link/left_outer_knuckle",
    "Geometry/world/link_base/link1/link2/link3/link4/link5/link6/link7/link_eef/xarm_gripper_base_link/left_outer_knuckle/left_finger",
    "Geometry/world/link_base/link1/link2/link3/link4/link5/link6/link7/link_eef/xarm_gripper_base_link/left_inner_knuckle",
    "Geometry/world/link_base/link1/link2/link3/link4/link5/link6/link7/link_eef/xarm_gripper_base_link/right_outer_knuckle",
    "Geometry/world/link_base/link1/link2/link3/link4/link5/link6/link7/link_eef/xarm_gripper_base_link/right_outer_knuckle/right_finger",
    "Geometry/world/link_base/link1/link2/link3/link4/link5/link6/link7/link_eef/xarm_gripper_base_link/right_inner_knuckle",
]
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


def bind_material_stronger(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI(prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )


def make_ground_and_backdrop(stage: Usd.Stage) -> tuple[UsdShade.Shader, UsdShade.Shader]:
    UsdGeom.Xform.Define(stage, "/World/Looks")

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
    left_wall = UsdGeom.Mesh.Define(stage, "/World/LeftWall")
    left_wall.CreatePointsAttr(
        [
            (-900.0, -1000.0, 0.0),
            (-900.0, 900.0, 0.0),
            (-900.0, 900.0, 1200.0),
            (-900.0, -1000.0, 1200.0),
        ]
    )
    left_wall.CreateFaceVertexCountsAttr([4])
    left_wall.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    right_wall = UsdGeom.Mesh.Define(stage, "/World/RightWall")
    right_wall.CreatePointsAttr(
        [
            (900.0, 900.0, 0.0),
            (900.0, -1000.0, 0.0),
            (900.0, -1000.0, 1200.0),
            (900.0, 900.0, 1200.0),
        ]
    )
    right_wall.CreateFaceVertexCountsAttr([4])
    right_wall.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ceiling = UsdGeom.Mesh.Define(stage, "/World/Ceiling")
    ceiling.CreatePointsAttr(
        [
            (-900.0, -1000.0, 1200.0),
            (900.0, -1000.0, 1200.0),
            (900.0, 900.0, 1200.0),
            (-900.0, 900.0, 1200.0),
        ]
    )
    ceiling.CreateFaceVertexCountsAttr([4])
    ceiling.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

    ground_mat, ground_shader = make_material(stage, "/World/Looks/GroundMat")
    backdrop_mat, backdrop_shader = make_material(stage, "/World/Looks/BackdropMat")
    wall_mat, wall_shader = make_material(stage, "/World/Looks/WallMat")
    bind_material(ground.GetPrim(), ground_mat)
    bind_material(backdrop.GetPrim(), backdrop_mat)
    bind_material(left_wall.GetPrim(), wall_mat)
    bind_material(right_wall.GetPrim(), wall_mat)
    bind_material(ceiling.GetPrim(), wall_mat)
    return ground_shader, backdrop_shader, wall_shader


def make_lights(stage: Usd.Stage) -> tuple[UsdLux.DomeLight, UsdLux.DistantLight, UsdGeom.XformCommonAPI, UsdLux.DistantLight]:
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1500.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    sun.CreateIntensityAttr(4000.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    sun_api = UsdGeom.XformCommonAPI(sun)
    sun_api.SetRotate((315.0, 0.0, 0.0))
    fill = UsdLux.DistantLight.Define(stage, "/World/FillLight")
    fill.CreateIntensityAttr(1800.0)
    fill.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    UsdGeom.XformCommonAPI(fill).SetRotate((40.0, 30.0, 0.0))
    return dome, sun, sun_api, fill


def create_cube_stage() -> tuple[Usd.Stage, dict[str, object]]:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    cube.CreateSizeAttr(140.0)
    cube.AddTranslateOp().Set((0.0, 0.0, 70.0))
    cube_api = UsdGeom.XformCommonAPI(cube)

    cube_mat, cube_shader = make_material(stage, "/World/Looks/CubeMat")
    bind_material(cube.GetPrim(), cube_mat)
    ground_shader, backdrop_shader, wall_shader = make_ground_and_backdrop(stage)
    dome, sun, sun_api, fill = make_lights(stage)

    stage.Save()
    scene = {
        "mode": "cube",
        "cube": cube,
        "cube_api": cube_api,
        "subject_path": cube.GetPath(),
        "subject_shader": cube_shader,
        "ground_shader": ground_shader,
        "backdrop_shader": backdrop_shader,
        "wall_shader": wall_shader,
        "dome": dome,
        "sun": sun,
        "sun_api": sun_api,
        "fill": fill,
    }
    return stage, scene


def subject_bbox(stage: Usd.Stage, prim_path: Sdf.Path) -> Gf.Range3d:
    prim = stage.GetPrimAtPath(prim_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bound = cache.ComputeWorldBound(prim)
    return Gf.Range3d(bound.ComputeAlignedBox().GetMin(), bound.ComputeAlignedBox().GetMax())


def create_asset_stage(usd_path: Path) -> tuple[Usd.Stage, dict[str, object]]:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    asset = UsdGeom.Xform.Define(stage, "/World/Subject")
    asset.GetPrim().GetReferences().AddReference(os.fspath(usd_path))
    asset_api = UsdGeom.XformCommonAPI(asset)
    asset_api.SetTranslate((0.0, 0.0, 0.0))
    asset_api.SetRotate((0.0, 0.0, 0.0))
    asset_api.SetScale((1.0, 1.0, 1.0))

    ground_shader, backdrop_shader, wall_shader = make_ground_and_backdrop(stage)
    dome, sun, sun_api, fill = make_lights(stage)
    pump(8)

    link_orients: list[tuple[UsdGeom.XformOp, Gf.Quatf, str]] = []
    for rel_path in ROBOT_LINK_PATHS + ROBOT_GRIPPER_PATHS:
        prim = stage.GetPrimAtPath(f"{asset.GetPath().pathString}/{rel_path}")
        if not prim.IsValid():
            continue
        xformable = UsdGeom.Xformable(prim)
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                link_orients.append((op, op.Get(), rel_path))
                break

    robot_mat, robot_shader = make_material(stage, "/World/Looks/RobotMat")
    bind_material_stronger(asset.GetPrim(), robot_mat)

    scene = {
        "mode": "asset",
        "asset_api": asset_api,
        "subject_path": asset.GetPath(),
        "robot_link_orients": link_orients,
        "subject_shader": robot_shader,
        "ground_shader": ground_shader,
        "backdrop_shader": backdrop_shader,
        "wall_shader": wall_shader,
        "dome": dome,
        "sun": sun,
        "sun_api": sun_api,
        "fill": fill,
    }
    stage.Save()
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


def rand_room_white(rng: random.Random, floor: bool = False) -> tuple[float, float, float]:
    hue = rng.uniform(0.0, 1.0)
    sat = rng.uniform(0.0, 0.08 if not floor else 0.12)
    val = rng.uniform(0.88, 0.99) if not floor else rng.uniform(0.72, 0.92)
    return colorsys.hsv_to_rgb(hue, sat, val)


def rand_neutral_metal(rng: random.Random) -> tuple[float, float, float]:
    base = rng.uniform(0.45, 0.92)
    tint = rng.uniform(-0.06, 0.06)
    return (
        max(0.2, min(1.0, base + tint)),
        max(0.2, min(1.0, base)),
        max(0.2, min(1.0, base - tint)),
    )


def rotate_about_z(base: Gf.Quatf, degrees: float) -> Gf.Quatf:
    spin = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), degrees).GetQuat()
    spin_f = Gf.Quatf(float(spin.GetReal()), Gf.Vec3f(*spin.GetImaginary()))
    return base * spin_f


def apply_domain_randomization(scene: dict[str, object], rng: random.Random, view_index: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if scene["mode"] != "cube":
        raise RuntimeError("Cube randomization called for non-cube scene")
    cube_api: UsdGeom.XformCommonAPI = scene["cube_api"]  # type: ignore[assignment]
    cube_shader: UsdShade.Shader = scene["subject_shader"]  # type: ignore[assignment]
    ground_shader: UsdShade.Shader = scene["ground_shader"]  # type: ignore[assignment]
    backdrop_shader: UsdShade.Shader = scene["backdrop_shader"]  # type: ignore[assignment]
    wall_shader: UsdShade.Shader = scene["wall_shader"]  # type: ignore[assignment]
    dome: UsdLux.DomeLight = scene["dome"]  # type: ignore[assignment]
    sun: UsdLux.DistantLight = scene["sun"]  # type: ignore[assignment]
    sun_api: UsdGeom.XformCommonAPI = scene["sun_api"]  # type: ignore[assignment]
    fill: UsdLux.DistantLight = scene["fill"]  # type: ignore[assignment]

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
    ground_color = rand_room_white(rng, floor=True)
    backdrop_color = rand_room_white(rng)
    wall_color = rand_room_white(rng)
    cube_shader.GetInput("diffuseColor").Set(vec3(cube_color))
    cube_shader.GetInput("roughness").Set(rng.uniform(0.08, 0.55))
    cube_shader.GetInput("metallic").Set(rng.uniform(0.0, 0.15))
    ground_shader.GetInput("diffuseColor").Set(vec3(ground_color))
    ground_shader.GetInput("roughness").Set(rng.uniform(0.3, 0.7))
    backdrop_shader.GetInput("diffuseColor").Set(vec3(backdrop_color))
    backdrop_shader.GetInput("roughness").Set(rng.uniform(0.45, 0.8))
    wall_shader.GetInput("diffuseColor").Set(vec3(wall_color))
    wall_shader.GetInput("roughness").Set(rng.uniform(0.35, 0.7))

    dome.GetIntensityAttr().Set(rng.uniform(700.0, 2200.0))
    dome.GetColorAttr().Set(vec3(rand_room_white(rng)))
    sun.GetIntensityAttr().Set(rng.uniform(9000.0, 22000.0))
    sun.GetColorAttr().Set(vec3(rand_room_white(rng)))
    sun_api.SetRotate((rng.uniform(230.0, 330.0), rng.uniform(-25.0, 25.0), 0.0))
    fill.GetIntensityAttr().Set(rng.uniform(2500.0, 9000.0))
    fill.GetColorAttr().Set(vec3(rand_room_white(rng)))

    azimuth = (360.0 * view_index / VIEW_COUNT) + rng.uniform(-18.0, 18.0)
    radius = rng.uniform(480.0, 620.0)
    height = rng.uniform(180.0, 280.0)
    theta = math.radians(azimuth)
    eye = (radius * math.cos(theta), radius * math.sin(theta), height)
    target = cube_translate
    return eye, target


def apply_robot_randomization(scene: dict[str, object], rng: random.Random, view_index: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    asset_api: UsdGeom.XformCommonAPI = scene["asset_api"]  # type: ignore[assignment]
    ground_shader: UsdShade.Shader = scene["ground_shader"]  # type: ignore[assignment]
    backdrop_shader: UsdShade.Shader = scene["backdrop_shader"]  # type: ignore[assignment]
    wall_shader: UsdShade.Shader = scene["wall_shader"]  # type: ignore[assignment]
    dome: UsdLux.DomeLight = scene["dome"]  # type: ignore[assignment]
    sun: UsdLux.DistantLight = scene["sun"]  # type: ignore[assignment]
    sun_api: UsdGeom.XformCommonAPI = scene["sun_api"]  # type: ignore[assignment]
    fill: UsdLux.DistantLight = scene["fill"]  # type: ignore[assignment]
    link_orients: list[tuple[UsdGeom.XformOp, Gf.Quatf, str]] = scene["robot_link_orients"]  # type: ignore[assignment]
    robot_shader: UsdShade.Shader = scene["subject_shader"]  # type: ignore[assignment]

    asset_api.SetTranslate((0.0, 0.0, 0.0))
    asset_api.SetRotate((0.0, 0.0, 0.0))
    asset_api.SetScale((1.0, 1.0, 1.0))

    arm_limits = [170.0, 110.0, 165.0, 120.0, 165.0, 110.0, 175.0]
    for index, (op, base_quat, _) in enumerate(link_orients[:7]):
        op.Set(rotate_about_z(base_quat, rng.uniform(-arm_limits[index], arm_limits[index])))

    grip_angle = rng.uniform(0.0, 28.0)
    for op, base_quat, rel_path in link_orients[7:]:
        sign = -1.0 if "right" in rel_path else 1.0
        op.Set(rotate_about_z(base_quat, sign * grip_angle))

    robot_shader.GetInput("diffuseColor").Set(vec3(rand_neutral_metal(rng) if rng.random() < 0.5 else rand_vivid_color(rng)))
    robot_shader.GetInput("roughness").Set(rng.uniform(0.12, 0.55))
    robot_shader.GetInput("metallic").Set(rng.uniform(0.0, 0.85))

    ground_shader.GetInput("diffuseColor").Set(vec3(rand_room_white(rng, floor=True)))
    ground_shader.GetInput("roughness").Set(rng.uniform(0.28, 0.68))
    backdrop_shader.GetInput("diffuseColor").Set(vec3(rand_room_white(rng)))
    backdrop_shader.GetInput("roughness").Set(rng.uniform(0.38, 0.72))
    wall_shader.GetInput("diffuseColor").Set(vec3(rand_room_white(rng)))
    wall_shader.GetInput("roughness").Set(rng.uniform(0.3, 0.68))

    dome.GetIntensityAttr().Set(rng.uniform(800.0, 2400.0))
    dome.GetColorAttr().Set(vec3(rand_room_white(rng)))
    sun.GetIntensityAttr().Set(rng.uniform(9000.0, 24000.0))
    sun.GetColorAttr().Set(vec3(rand_room_white(rng)))
    sun_api.SetRotate((rng.uniform(220.0, 330.0), rng.uniform(-25.0, 25.0), 0.0))
    fill.GetIntensityAttr().Set(rng.uniform(3000.0, 10000.0))
    fill.GetColorAttr().Set(vec3(rand_room_white(rng)))

    azimuth = (360.0 * view_index / VIEW_COUNT) + rng.uniform(-24.0, 24.0)
    radius = rng.uniform(1.4, 2.4)
    height = rng.uniform(0.5, 1.3)
    theta = math.radians(azimuth)
    eye = (radius * math.cos(theta), radius * math.sin(theta), height)
    target = (0.0, 0.0, 0.0)
    return eye, target


def orbit_camera_for_bbox(bbox: Gf.Range3d, view_index: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    center = (bbox.GetMin() + bbox.GetMax()) / 2.0
    size = bbox.GetSize()
    radius = max(size[0], size[1], size[2]) * 2.8
    height = center[2] + max(size[2] * 1.25, radius * 0.35)
    theta = math.radians((360.0 * view_index) / VIEW_COUNT)
    eye = (
        center[0] + radius * math.cos(theta),
        center[1] + radius * math.sin(theta),
        height,
    )
    target = (center[0], center[1], center[2])
    return eye, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd_path", nargs="?", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path("renders/cube_views").resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {out_dir}")

    usd_path = Path(args.usd_path).resolve() if args.usd_path else None
    if usd_path is not None:
        if not usd_path.exists():
            raise FileNotFoundError(usd_path)
        stage, scene = create_asset_stage(usd_path)
    else:
        stage, scene = create_cube_stage()

    subject_path: Sdf.Path = scene["subject_path"]  # type: ignore[assignment]
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active viewport available")

    pump(32)
    rng = random.Random(20260413)
    bbox = subject_bbox(stage, subject_path)

    for i in range(VIEW_COUNT):
        image_path = out_dir / f"view_{i}.png"
        if scene["mode"] == "cube":
            eye, target = apply_domain_randomization(scene, rng, i)
            bbox = subject_bbox(stage, subject_path)
        else:
            eye, target = apply_robot_randomization(scene, rng, i)
            bbox = subject_bbox(stage, subject_path)
        camera_path = create_camera(stage, eye, target, f"Camera_{i}")
        frame_cube(viewport, camera_path, subject_path)
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
