from __future__ import annotations

import json
import math
import os
import random
import shutil
import colorsys
from dataclasses import dataclass
from pathlib import Path

import tyro

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


VIEW_COUNT = 50
ASSET_ROOM_X = 4.5
ASSET_ROOM_Y_NEG = -4.5
ASSET_ROOM_Y_POS = 4.5
ASSET_ROOM_Z_MIN = -3.0
ASSET_ROOM_Z_MAX = 4.5
ASSET_ROOM_MARGIN = 0.08
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
ROBOT_KEYPOINT_PATHS = [
    ("base", "Geometry/world/link_base"),
    ("joint1", "Physics/joint1"),
    ("joint2", "Physics/joint2"),
    ("joint3", "Physics/joint3"),
    ("joint4", "Physics/joint4"),
    ("joint5", "Physics/joint5"),
    ("joint6", "Physics/joint6"),
    ("joint7", "Physics/joint7"),
    ("eef", "Physics/joint_eef"),
    ("tcp", "Physics/joint_tcp"),
    ("gripper_drive", "Physics/drive_joint"),
    ("gripper_left_finger", "Physics/left_finger_joint"),
    ("gripper_left_inner", "Physics/left_inner_knuckle_joint"),
    ("gripper_right_outer", "Physics/right_outer_knuckle_joint"),
    ("gripper_right_finger", "Physics/right_finger_joint"),
    ("gripper_right_inner", "Physics/right_inner_knuckle_joint"),
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


def make_ground_and_backdrop(
    stage: Usd.Stage,
    *,
    x_half: float = 900.0,
    y_min: float = -1000.0,
    y_max: float = 900.0,
    z_min: float = 0.0,
    z_max: float = 1200.0,
    add_front_wall: bool = False,
) -> tuple[UsdShade.Shader, UsdShade.Shader, UsdShade.Shader]:
    UsdGeom.Xform.Define(stage, "/World/Looks")

    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    ground.CreatePointsAttr(
        [
            (-x_half, y_min, z_min),
            (x_half, y_min, z_min),
            (x_half, y_max, z_min),
            (-x_half, y_max, z_min),
        ]
    )
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateDoubleSidedAttr(True)
    backdrop = UsdGeom.Mesh.Define(stage, "/World/Backdrop")
    backdrop.CreatePointsAttr(
        [
            (-x_half, y_min, z_min),
            (x_half, y_min, z_min),
            (x_half, y_min, z_max),
            (-x_half, y_min, z_max),
        ]
    )
    backdrop.CreateFaceVertexCountsAttr([4])
    backdrop.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    backdrop.CreateDoubleSidedAttr(True)
    left_wall = UsdGeom.Mesh.Define(stage, "/World/LeftWall")
    left_wall.CreatePointsAttr(
        [
            (-x_half, y_min, z_min),
            (-x_half, y_max, z_min),
            (-x_half, y_max, z_max),
            (-x_half, y_min, z_max),
        ]
    )
    left_wall.CreateFaceVertexCountsAttr([4])
    left_wall.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    left_wall.CreateDoubleSidedAttr(True)
    right_wall = UsdGeom.Mesh.Define(stage, "/World/RightWall")
    right_wall.CreatePointsAttr(
        [
            (x_half, y_max, z_min),
            (x_half, y_min, z_min),
            (x_half, y_min, z_max),
            (x_half, y_max, z_max),
        ]
    )
    right_wall.CreateFaceVertexCountsAttr([4])
    right_wall.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    right_wall.CreateDoubleSidedAttr(True)
    ceiling = UsdGeom.Mesh.Define(stage, "/World/Ceiling")
    ceiling.CreatePointsAttr(
        [
            (-x_half, y_min, z_max),
            (x_half, y_min, z_max),
            (x_half, y_max, z_max),
            (-x_half, y_max, z_max),
        ]
    )
    ceiling.CreateFaceVertexCountsAttr([4])
    ceiling.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ceiling.CreateDoubleSidedAttr(True)
    front_wall = None
    if add_front_wall:
        front_wall = UsdGeom.Mesh.Define(stage, "/World/FrontWall")
        front_wall.CreatePointsAttr(
            [
                (-x_half, y_max, z_min),
                (x_half, y_max, z_min),
                (x_half, y_max, z_max),
                (-x_half, y_max, z_max),
            ]
        )
        front_wall.CreateFaceVertexCountsAttr([4])
        front_wall.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        front_wall.CreateDoubleSidedAttr(True)

    ground_mat, ground_shader = make_material(stage, "/World/Looks/GroundMat")
    backdrop_mat, backdrop_shader = make_material(stage, "/World/Looks/BackdropMat")
    wall_mat, wall_shader = make_material(stage, "/World/Looks/WallMat")
    bind_material(ground.GetPrim(), ground_mat)
    bind_material(backdrop.GetPrim(), backdrop_mat)
    bind_material(left_wall.GetPrim(), wall_mat)
    bind_material(right_wall.GetPrim(), wall_mat)
    bind_material(ceiling.GetPrim(), wall_mat)
    if front_wall is not None:
        bind_material(front_wall.GetPrim(), wall_mat)
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

    ground_shader, backdrop_shader, wall_shader = make_ground_and_backdrop(
        stage,
        x_half=ASSET_ROOM_X,
        y_min=ASSET_ROOM_Y_NEG,
        y_max=ASSET_ROOM_Y_POS,
        z_min=ASSET_ROOM_Z_MIN,
        z_max=ASSET_ROOM_Z_MAX,
        add_front_wall=True,
    )
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


CAMERA_APERTURE = 20.955
FOCAL_LENGTH_BASE = 24.0
FOCAL_LENGTH_JITTER = 0.15


def create_camera(
    stage: Usd.Stage,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    name: str,
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    focal_length: float = FOCAL_LENGTH_BASE,
) -> Sdf.Path:
    camera = UsdGeom.Camera.Define(stage, f"/World/{name}")
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureAttr(CAMERA_APERTURE)
    camera.CreateVerticalApertureAttr(CAMERA_APERTURE)
    camera.CreateFocusDistanceAttr(700.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(1e-4, 10000.0))
    camera.MakeMatrixXform().Set(camera_xform(eye, target, rpy_deg))
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


def vec3d_tuple(vec: Gf.Vec3d) -> list[float]:
    return [float(vec[0]), float(vec[1]), float(vec[2])]


def mat4_list(mat: Gf.Matrix4d) -> list[list[float]]:
    return [[float(mat[r][c]) for c in range(4)] for r in range(4)]


def camera_xform(eye: tuple[float, float, float], target: tuple[float, float, float], rpy_deg: tuple[float, float, float]) -> Gf.Matrix4d:
    look = Gf.Matrix4d(1.0)
    look.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 0.0, 1.0))
    camera_to_world = look.GetInverse()
    roll, pitch, yaw = rpy_deg
    rot = (
        Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), roll)
        * Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), pitch)
        * Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), yaw)
    )
    offset = Gf.Matrix4d(1.0)
    offset.SetRotateOnly(rot)
    return camera_to_world * offset


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


def rand_broad_color(rng: random.Random, floor: bool = False) -> tuple[float, float, float]:
    hue = rng.random()
    sat = rng.uniform(0.15, 0.95 if not floor else 0.75)
    val = rng.uniform(0.28, 0.98 if not floor else 0.85)
    return colorsys.hsv_to_rgb(hue, sat, val)


def rand_scene_color(rng: random.Random, floor: bool = False) -> tuple[float, float, float]:
    if rng.random() < 0.10:
        return rand_room_white(rng, floor=floor)
    return rand_broad_color(rng, floor=floor)


def rand_neutral_robot(rng: random.Random) -> tuple[float, float, float]:
    hue = rng.random()
    sat = rng.uniform(0.0, 0.18)
    val = rng.uniform(0.55, 0.98)
    return colorsys.hsv_to_rgb(hue, sat, val)


def perturb_color(
    rng: random.Random,
    base: tuple[float, float, float],
    hue_jitter: float = 0.06,
    sat_jitter: float = 0.15,
    val_jitter: float = 0.18,
) -> tuple[float, float, float]:
    h, s, v = colorsys.rgb_to_hsv(*base)
    h = (h + rng.uniform(-hue_jitter, hue_jitter)) % 1.0
    s = max(0.0, min(1.0, s + rng.uniform(-sat_jitter, sat_jitter)))
    v = max(0.0, min(1.0, v + rng.uniform(-val_jitter, val_jitter)))
    return colorsys.hsv_to_rgb(h, s, v)


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
    ground_color = rand_scene_color(rng, floor=True)
    backdrop_color = rand_scene_color(rng)
    wall_color = rand_scene_color(rng)
    cube_shader.GetInput("diffuseColor").Set(vec3(cube_color))
    cube_shader.GetInput("roughness").Set(rng.uniform(0.08, 0.55))
    cube_shader.GetInput("metallic").Set(rng.uniform(0.0, 0.15))
    ground_shader.GetInput("diffuseColor").Set(vec3(ground_color))
    ground_shader.GetInput("roughness").Set(rng.uniform(0.3, 0.7))
    backdrop_shader.GetInput("diffuseColor").Set(vec3(backdrop_color))
    backdrop_shader.GetInput("roughness").Set(rng.uniform(0.45, 0.8))
    wall_shader.GetInput("diffuseColor").Set(vec3(wall_color))
    wall_shader.GetInput("roughness").Set(rng.uniform(0.35, 0.7))

    dome.GetIntensityAttr().Set(rng.uniform(250.0, 3200.0))
    dome.GetColorAttr().Set(vec3(rand_scene_color(rng)))
    sun.GetIntensityAttr().Set(rng.uniform(4000.0, 24000.0))
    sun.GetColorAttr().Set(vec3(rand_scene_color(rng)))
    sun_api.SetRotate((rng.uniform(230.0, 330.0), rng.uniform(-25.0, 25.0), 0.0))
    fill.GetIntensityAttr().Set(rng.uniform(1200.0, 12000.0))
    fill.GetColorAttr().Set(vec3(rand_scene_color(rng)))

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
    arm_angles: dict[str, float] = {}
    for index, (op, base_quat, _) in enumerate(link_orients[:7]):
        angle = rng.uniform(-arm_limits[index], arm_limits[index])
        op.Set(rotate_about_z(base_quat, angle))
        arm_angles[f"joint{index + 1}"] = angle

    grip_angle = rng.uniform(0.0, 28.0)
    for op, base_quat, rel_path in link_orients[7:]:
        sign = -1.0 if "right" in rel_path else 1.0
        op.Set(rotate_about_z(base_quat, sign * grip_angle))
    arm_angles["gripper_angle"] = grip_angle

    wall_color = rand_scene_color(rng)
    if rng.random() < 0.5:
        robot_color = rand_neutral_robot(rng)
    else:
        robot_color = perturb_color(rng, wall_color)
    robot_shader.GetInput("diffuseColor").Set(vec3(robot_color))
    robot_shader.GetInput("roughness").Set(rng.uniform(0.12, 0.55))
    robot_shader.GetInput("metallic").Set(rng.uniform(0.0, 0.85))

    ground_shader.GetInput("diffuseColor").Set(vec3(rand_scene_color(rng, floor=True)))
    ground_shader.GetInput("roughness").Set(rng.uniform(0.2, 0.82))
    backdrop_shader.GetInput("diffuseColor").Set(vec3(rand_scene_color(rng)))
    backdrop_shader.GetInput("roughness").Set(rng.uniform(0.25, 0.82))
    wall_shader.GetInput("diffuseColor").Set(vec3(wall_color))
    wall_shader.GetInput("roughness").Set(rng.uniform(0.2, 0.8))

    dome_on = rng.random() < 0.7
    dome.GetIntensityAttr().Set(rng.uniform(500.0, 3000.0) if dome_on else 0.0)
    dome.GetColorAttr().Set(vec3(rand_scene_color(rng)))
    sun.GetIntensityAttr().Set(rng.uniform(3500.0, 26000.0))
    sun.GetColorAttr().Set(vec3(rand_scene_color(rng)))
    sun_api.SetRotate((rng.uniform(220.0, 330.0), rng.uniform(-25.0, 25.0), 0.0))
    fill.GetIntensityAttr().Set(rng.uniform(0.0, 12000.0))
    fill.GetColorAttr().Set(vec3(rand_scene_color(rng)))

    azimuth = (360.0 * view_index / VIEW_COUNT) + rng.uniform(-24.0, 24.0)
    radius = rng.uniform(1.4, 2.4)
    height = rng.uniform(0.5, 1.3)
    theta = math.radians(azimuth)
    eye = (radius * math.cos(theta), radius * math.sin(theta), height)
    target = (0.0, 0.0, 0.0)
    scene["robot_joint_values"] = arm_angles
    return eye, target


def camera_calibration(stage: Usd.Stage, camera_path: Sdf.Path) -> dict[str, object]:
    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    focal = float(camera.GetFocalLengthAttr().Get())
    horiz_ap = float(camera.GetHorizontalApertureAttr().Get())
    vert_ap = float(camera.GetVerticalApertureAttr().Get())
    fx = IMAGE_SIZE * focal / horiz_ap
    fy = IMAGE_SIZE * focal / vert_ap
    cx = IMAGE_SIZE / 2.0
    cy = IMAGE_SIZE / 2.0
    xform_cache = UsdGeom.XformCache()
    camera_to_world = xform_cache.GetLocalToWorldTransform(camera.GetPrim())
    world_to_camera = camera_to_world.GetInverse()
    return {
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE,
            "K": [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
        },
        "extrinsics": {
            "camera_to_world": mat4_list(camera_to_world),
            "world_to_camera": mat4_list(world_to_camera),
        },
    }


def project_point(world_to_camera: Gf.Matrix4d, point_world: Gf.Vec3d, intr: dict[str, object]) -> dict[str, object]:
    cam = world_to_camera.Transform(point_world)
    depth = -float(cam[2])
    if depth <= 1e-6:
        return {
            "camera_xyz": vec3d_tuple(cam),
            "pixel_xy": None,
            "visible": False,
        }
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    x = fx * (float(cam[0]) / depth) + cx
    y = cy - fy * (float(cam[1]) / depth)
    visible = 0.0 <= x < IMAGE_SIZE and 0.0 <= y < IMAGE_SIZE
    return {
        "camera_xyz": vec3d_tuple(cam),
        "pixel_xy": [x, y],
        "visible": visible,
    }


def joint_world_point(xform_cache: UsdGeom.XformCache, joint_prim: Usd.Prim) -> Gf.Vec3d | None:
    body0_targets = joint_prim.GetRelationship("physics:body0").GetTargets()
    local_pos0_attr = joint_prim.GetAttribute("physics:localPos0")
    if not body0_targets or not local_pos0_attr.IsValid():
        return None
    body0 = joint_prim.GetStage().GetPrimAtPath(body0_targets[0])
    if not body0.IsValid():
        return None
    local_pos0 = local_pos0_attr.Get()
    if local_pos0 is None:
        return None
    return xform_cache.GetLocalToWorldTransform(body0).Transform(Gf.Vec3d(*local_pos0))


def robot_world_points(stage: Usd.Stage, subject_path: Sdf.Path) -> list[tuple[str, Gf.Vec3d]]:
    xform_cache = UsdGeom.XformCache()
    points: list[tuple[str, Gf.Vec3d]] = []
    for name, rel_path in ROBOT_KEYPOINT_PATHS:
        prim = stage.GetPrimAtPath(f"{subject_path.pathString}/{rel_path}")
        if not prim.IsValid():
            continue
        if rel_path.startswith("Physics/"):
            point = joint_world_point(xform_cache, prim)
        else:
            point = xform_cache.GetLocalToWorldTransform(prim).ExtractTranslation()
        if point is not None:
            points.append((name, point))
    return points


def segment_distance(a0: Gf.Vec3d, a1: Gf.Vec3d, b0: Gf.Vec3d, b1: Gf.Vec3d) -> float:
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    a = float(Gf.Dot(u, u))
    b = float(Gf.Dot(u, v))
    c = float(Gf.Dot(v, v))
    d = float(Gf.Dot(u, w))
    e = float(Gf.Dot(v, w))
    den = a * c - b * b
    eps = 1e-8
    if den < eps:
        s = 0.0
        t = max(0.0, min(1.0, e / c if c > eps else 0.0))
    else:
        s = max(0.0, min(1.0, (b * e - c * d) / den))
        t = max(0.0, min(1.0, (a * e - b * d) / den))
    pa = a0 + s * u
    pb = b0 + t * v
    return float((pa - pb).GetLength())


def point_segment_distance(p: Gf.Vec3d, a: Gf.Vec3d, b: Gf.Vec3d) -> float:
    ab = b - a
    denom = float(Gf.Dot(ab, ab))
    if denom <= 1e-8:
        return float((p - a).GetLength())
    t = max(0.0, min(1.0, float(Gf.Dot(p - a, ab)) / denom))
    proj = a + t * ab
    return float((p - proj).GetLength())


def robot_segments(points: list[tuple[str, Gf.Vec3d]]) -> list[tuple[str, str, Gf.Vec3d, Gf.Vec3d]]:
    by_name = {name: point for name, point in points}
    edges = [
        ("base", "joint1"),
        ("joint1", "joint3"),
        ("joint3", "joint4"),
        ("joint4", "joint5"),
        ("joint5", "joint7"),
        ("joint7", "tcp"),
    ]
    out = []
    for a, b in edges:
        if a in by_name and b in by_name:
            out.append((a, b, by_name[a], by_name[b]))
    return out


def robot_penetrates_room(bbox: Gf.Range3d) -> bool:
    mn = bbox.GetMin()
    mx = bbox.GetMax()
    return any(
        [
            float(mn[0]) < -ASSET_ROOM_X + ASSET_ROOM_MARGIN,
            float(mx[0]) > ASSET_ROOM_X - ASSET_ROOM_MARGIN,
            float(mn[1]) < ASSET_ROOM_Y_NEG + ASSET_ROOM_MARGIN,
            float(mx[1]) > ASSET_ROOM_Y_POS - ASSET_ROOM_MARGIN,
            float(mx[2]) > ASSET_ROOM_Z_MAX - ASSET_ROOM_MARGIN,
        ]
    )


def robot_self_collision(points: list[tuple[str, Gf.Vec3d]]) -> bool:
    segments = robot_segments(points)
    if len(segments) < 4:
        return False
    interesting_pairs = {
        ("base", "joint1", "joint4", "joint5"),
        ("base", "joint1", "joint5", "joint7"),
        ("base", "joint1", "joint7", "tcp"),
        ("joint1", "joint3", "joint5", "joint7"),
        ("joint1", "joint3", "joint7", "tcp"),
        ("joint3", "joint4", "joint7", "tcp"),
    }
    for i, (a0_name, a1_name, a0, a1) in enumerate(segments):
        for j in range(i + 1, len(segments)):
            b0_name, b1_name, b0, b1 = segments[j]
            if (a0_name, a1_name, b0_name, b1_name) not in interesting_pairs:
                continue
            if segment_distance(a0, a1, b0, b1) < 0.02:
                return True
    return False


def camera_collides_with_robot(eye: tuple[float, float, float], bbox: Gf.Range3d, points: list[tuple[str, Gf.Vec3d]]) -> bool:
    eye_vec = Gf.Vec3d(*eye)
    mn = bbox.GetMin()
    mx = bbox.GetMax()
    if (
        float(mn[0]) - 0.12 <= eye[0] <= float(mx[0]) + 0.12
        and float(mn[1]) - 0.12 <= eye[1] <= float(mx[1]) + 0.12
        and float(mn[2]) - 0.12 <= eye[2] <= float(mx[2]) + 0.12
    ):
        return True
    for _, point in points:
        if float((eye_vec - point).GetLength()) < 0.22:
            return True
    for _, _, a, b in robot_segments(points):
        if point_segment_distance(eye_vec, a, b) < 0.16:
            return True
    return False


def sample_robot_camera(
    stage: Usd.Stage,
    subject_path: Sdf.Path,
    rng: random.Random,
    view_index: int,
    focal_length: float = FOCAL_LENGTH_BASE,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], int]:
    points = robot_world_points(stage, subject_path)
    target_name, target_point = rng.choice(points)
    azimuth = (360.0 * view_index / VIEW_COUNT) + rng.uniform(-55.0, 55.0)
    radius = rng.uniform(0.65, 2.4)
    height = rng.uniform(-0.9, 1.55)
    theta = math.radians(azimuth)
    eye = (radius * math.cos(theta), radius * math.sin(theta), height)
    target = (float(target_point[0]), float(target_point[1]), float(target_point[2]))
    rpy_deg = (
        rng.uniform(-40.0, 40.0),
        rng.uniform(-24.0, 24.0),
        rng.uniform(-24.0, 24.0),
    )
    fx_fy = IMAGE_SIZE * focal_length / CAMERA_APERTURE
    intr = {"fx": fx_fy, "fy": fx_fy, "cx": IMAGE_SIZE / 2.0, "cy": IMAGE_SIZE / 2.0}
    world_to_camera = camera_xform(eye, target, rpy_deg).GetInverse()
    visible = 0
    for _, point in points:
        if project_point(world_to_camera, point, intr)["visible"]:
            visible += 1
    return eye, target, rpy_deg, visible


def robot_keypoint_payload(stage: Usd.Stage, subject_path: Sdf.Path, camera_path: Sdf.Path, scene: dict[str, object]) -> dict[str, object]:
    calibration = camera_calibration(stage, camera_path)
    intr = calibration["intrinsics"]
    world_to_camera = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(camera_path)).GetInverse()

    keypoints = []
    for name, point_world in robot_world_points(stage, subject_path):
        projection = project_point(world_to_camera, point_world, intr)
        keypoints.append(
            {
                "name": name,
                "world_xyz": vec3d_tuple(point_world),
                "camera_xyz": projection["camera_xyz"],
                "pixel_xy": projection["pixel_xy"],
                "visible": projection["visible"],
            }
        )

    return {
        "joints": scene.get("robot_joint_values", {}),
        "keypoints": keypoints,
        "camera": calibration,
    }


def write_sidecar(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))


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


def tag_semantics(stage: Usd.Stage, prim_path: str, label: str) -> None:
    try:
        from pxr import Semantics  # type: ignore
    except Exception:
        return
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
    sem.CreateSemanticTypeAttr("class")
    sem.CreateSemanticDataAttr(label)


def apply_scene_semantics(stage: Usd.Stage, scene: dict[str, object]) -> None:
    if scene["mode"] == "asset":
        subject_path = scene["subject_path"]  # type: ignore[assignment]
        tag_semantics(stage, f"{subject_path}", "robot")
    else:
        tag_semantics(stage, "/World/Cube", "cube")
    tag_semantics(stage, "/World/Ground", "ground")
    tag_semantics(stage, "/World/Backdrop", "backdrop")
    tag_semantics(stage, "/World/LeftWall", "wall")
    tag_semantics(stage, "/World/RightWall", "wall")
    tag_semantics(stage, "/World/Ceiling", "wall")


@dataclass
class Config:
    """Config for rendering cube / robot views."""

    usd_path: Path | None = None  # optional USD asset to load instead of the default cube scene
    seed: int = 20260413  # RNG seed for deterministic domain randomization
    num_views: int = VIEW_COUNT  # number of views to render
    output_dir: Path = Path("renders/cube_views")  # directory where renders and sidecars are written
    no_clean: bool = False  # keep existing output_dir contents instead of wiping it before rendering


def main(cfg: Config) -> None:
    global VIEW_COUNT
    VIEW_COUNT = cfg.num_views

    out_dir = cfg.output_dir.resolve()
    if out_dir.exists() and not cfg.no_clean:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {out_dir} (seed={cfg.seed}, num_views={VIEW_COUNT})")

    usd_path = cfg.usd_path.resolve() if cfg.usd_path else None
    if usd_path is not None:
        if not usd_path.exists():
            raise FileNotFoundError(usd_path)
        stage, scene = create_asset_stage(usd_path)
    else:
        stage, scene = create_cube_stage()

    subject_path: Sdf.Path = scene["subject_path"]  # type: ignore[assignment]
    apply_scene_semantics(stage, scene)

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active viewport available")

    pump(32)
    rng = random.Random(cfg.seed)
    bbox = subject_bbox(stage, subject_path)

    total_rejects = {"room": 0, "self": 0, "camera": 0, "pose": 0, "dark": 0}

    for i in range(VIEW_COUNT):
        image_path = out_dir / f"view_{i}.png"
        json_path = out_dir / f"view_{i}.json"
        accepted = False
        reject_counts = {"room": 0, "self": 0, "camera": 0, "pose": 0, "dark": 0}
        focal_length = FOCAL_LENGTH_BASE
        for attempt in range(40):
            rpy_deg = (0.0, 0.0, 0.0)
            if scene["mode"] == "cube":
                eye, target = apply_domain_randomization(scene, rng, i)
                bbox = subject_bbox(stage, subject_path)
            else:
                focal_length = FOCAL_LENGTH_BASE * rng.uniform(
                    1.0 - FOCAL_LENGTH_JITTER, 1.0 + FOCAL_LENGTH_JITTER
                )
                pose_ok = False
                for _ in range(24):
                    apply_robot_randomization(scene, rng, i)
                    bbox = subject_bbox(stage, subject_path)
                    points = robot_world_points(stage, subject_path)
                    if robot_penetrates_room(bbox):
                        reject_counts["room"] += 1
                        continue
                    if robot_self_collision(points):
                        reject_counts["self"] += 1
                        continue
                    eye, target, rpy_deg, visible_count = sample_robot_camera(
                        stage,
                        subject_path,
                        rng,
                        i,
                        focal_length=focal_length,
                    )
                    if camera_collides_with_robot(eye, bbox, points):
                        reject_counts["camera"] += 1
                        continue
                    if visible_count >= 1:
                        scene["visible_keypoint_count"] = visible_count
                        scene["camera_rpy_deg"] = list(rpy_deg)
                        pose_ok = True
                        break
                if not pose_ok:
                    reject_counts["pose"] += 1
                    continue
            camera_path = create_camera(
                stage, eye, target, f"Camera_{i}", rpy_deg=rpy_deg, focal_length=focal_length
            )
            frame_cube(viewport, camera_path, subject_path)
            if image_path.exists():
                image_path.unlink()
            if json_path.exists():
                json_path.unlink()
            capture_viewport_to_file(viewport, os.fspath(image_path))
            wait_for_capture(image_path)
            mn, mx, mean = image_stats(image_path)
            if mx > 0 and mean >= 8.0:
                accepted = True
                break
            reject_counts["dark"] += 1
        for key, val in reject_counts.items():
            total_rejects[key] += val
        if not accepted:
            raise RuntimeError(f"Failed to render a usable frame for view_{i}: {reject_counts}")
        sidecar = {
            "image_file": image_path.name,
            "scene_mode": scene["mode"],
            "camera_target_world": [float(target[0]), float(target[1]), float(target[2])],
            "camera_rpy_deg": [float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2])],
            "camera_focal_length": float(focal_length),
            "reject_counts": reject_counts,
        }
        if scene["mode"] == "asset":
            sidecar.update(robot_keypoint_payload(stage, subject_path, camera_path, scene))
            sidecar["visible_keypoint_count"] = int(scene.get("visible_keypoint_count", 0))
        else:
            sidecar["joints"] = {}
            sidecar["keypoints"] = []
            sidecar["camera"] = camera_calibration(stage, camera_path)
        write_sidecar(json_path, sidecar)
        print(
            f"view_{i}: eye={tuple(round(v, 1) for v in eye)} "
            f"target={tuple(round(v, 1) for v in target)} "
            f"min={mn} max={mx} mean={mean:.2f} rejects={reject_counts}"
        )

    print(f"Wrote renders to {out_dir}")
    print(f"Total rejects across {VIEW_COUNT} views: {total_rejects}")
    stage = None
    viewport = None
    omni.usd.get_context().close_stage()
    APP.close()


if __name__ == "__main__":
    main(tyro.cli(Config))
