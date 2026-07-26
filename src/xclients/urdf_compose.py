"""Compose URDF models at load time — prune, attach, place — without rewriting sources.

Sources stay untouched on disk; rigs are assembled in memory from the pieces
(e.g. bare xArm7 + ruka hand + world placement) and can be saved to
``urdf/generated/`` for inspection. Mesh paths are absolutized on load so a
composed model mixing files from different directories still resolves.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import yourdfpy
from yourdfpy.urdf import Collision, Geometry, Joint, Link, Robot, Sphere

log = logging.getLogger(__name__)


def load_model(path: str | Path) -> Robot:
    """Load a URDF as a bare model (no meshes, no scene graph) with mesh paths absolutized."""
    path = Path(path).expanduser().resolve()
    urdf = yourdfpy.URDF.load(
        path,
        build_scene_graph=False,
        load_meshes=False,
        load_collision_meshes=False,
    )
    _absolutize_mesh_paths(urdf.robot, path.parent)
    return urdf.robot


def sort_joints_topologically(robot: Robot) -> None:
    """Reorder joints root-down (stable BFS), in place.

    Keeps consumers that map q by document order (yourdfpy/ViserUrdf) consistent
    with pyroki, which topologically re-sorts internally when the file isn't.
    """
    by_parent: dict[str, list[Joint]] = {}
    for j in robot.joints:
        by_parent.setdefault(j.parent, []).append(j)
    ordered: list[Joint] = []
    frontier = [root_link(robot).name]
    while frontier:
        for j in by_parent.get(frontier.pop(0), []):
            ordered.append(j)
            frontier.append(j.child)
    if len(ordered) != len(robot.joints):
        raise ValueError("Joint graph is not a single tree; cannot sort topologically")
    robot.joints = ordered


def save(robot: Robot, path: str | Path) -> Path:
    """Write a composed model to disk (for inspection or downstream loading).

    Joints are topologically sorted (in place) before writing.
    """
    sort_joints_topologically(robot)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    urdf = yourdfpy.URDF(robot=robot, build_scene_graph=False, load_meshes=False)
    tree = urdf.write_xml()
    # yourdfpy 0.0.60's _write_joint drops <mimic> (its _write_mimic is never
    # called); re-inject so coupled joints (e.g. xArm gripper) stay coupled.
    mimics = {j.name: j.mimic for j in robot.joints if j.mimic is not None}
    if mimics:
        for elem in tree.getroot().iter("joint"):
            mimic = mimics.get(elem.get("name"))
            if mimic is not None:
                urdf._write_mimic(elem, mimic)
    tree.write(str(path), xml_declaration=True, pretty_print=True)
    return path


def load_urdf(path: str | Path, load_collision_meshes: bool = True) -> yourdfpy.URDF:
    """Fully load a (generated) URDF for pyroki / viser: scene graph + meshes."""
    return yourdfpy.URDF.load(
        str(path),
        build_scene_graph=True,
        load_meshes=True,
        load_collision_meshes=load_collision_meshes,
    )


def root_link(robot: Robot) -> Link:
    children = {j.child for j in robot.joints}
    roots = [link for link in robot.links if link.name not in children]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root link, found {[r.name for r in roots]}")
    return roots[0]


def prune_subtree(robot: Robot, joint_name: str) -> None:
    """Remove a joint and every link/joint below it (e.g. cut `gripper_fix` off the xArm)."""
    by_parent: dict[str, list[Joint]] = {}
    for j in robot.joints:
        by_parent.setdefault(j.parent, []).append(j)

    start = next((j for j in robot.joints if j.name == joint_name), None)
    if start is None:
        raise ValueError(f"Joint {joint_name!r} not found in {robot.name!r}")

    dead_joints = {start.name}
    dead_links = {start.child}
    frontier = [start.child]
    while frontier:
        for j in by_parent.get(frontier.pop(), []):
            dead_joints.add(j.name)
            dead_links.add(j.child)
            frontier.append(j.child)

    robot.joints = [j for j in robot.joints if j.name not in dead_joints]
    robot.links = [link for link in robot.links if link.name not in dead_links]
    robot.transmission = [
        t
        for t in (robot.transmission or [])
        if not any(tj.name in dead_joints for tj in getattr(t, "joints", []) or [])
    ]


def attach(
    parent: Robot,
    child: Robot,
    parent_link: str,
    T_parent_child: NDArray[np.float64],
    prefix: str,
) -> None:
    """Graft `child` onto `parent` at `parent_link` via a fixed joint with origin T.

    Child link/joint/material names (and mimic references) get `prefix` to stay
    collision-free. `child` is not mutated; `parent` is extended in place.
    Child transmission/gazebo blocks are dropped (irrelevant for kinematics).
    """
    if parent_link not in {link.name for link in parent.links}:
        raise ValueError(f"Parent link {parent_link!r} not in {parent.name!r}")

    child = copy.deepcopy(child)
    child_root = root_link(child).name

    for link in child.links:
        link.name = prefix + link.name
        for v in link.visuals:
            if v.material is not None and v.material.name:
                v.material.name = prefix + v.material.name
    for j in child.joints:
        j.name = prefix + j.name
        j.parent = prefix + j.parent
        j.child = prefix + j.child
        if j.mimic is not None:
            j.mimic.joint = prefix + j.mimic.joint
    for m in child.materials or []:
        m.name = prefix + m.name

    taken = {link.name for link in parent.links} | {j.name for j in parent.joints}
    added = {link.name for link in child.links} | {j.name for j in child.joints}
    if clash := taken & added:
        raise ValueError(f"Name clash after prefixing with {prefix!r}: {sorted(clash)[:5]}")

    parent.links.extend(child.links)
    parent.joints.extend(child.joints)
    parent.materials = list(parent.materials or []) + list(child.materials or [])
    parent.joints.append(
        Joint(
            name=f"{prefix}mount",
            type="fixed",
            parent=parent_link,
            child=prefix + child_root,
            origin=np.asarray(T_parent_child, dtype=np.float64),
        )
    )


def add_collision_sphere(
    robot: Robot,
    link_name: str,
    center: NDArray[np.floating],
    radius: float,
    name: str = "coll_proxy",
) -> None:
    """Add a sphere <collision> primitive to a link (e.g. a coarse whole-hand proxy).

    pyroki fits each link's collision capsule over ALL of the link's collision
    geometry, so this inflates the link's capsule to cover the sphere.
    """
    link = next((l for l in robot.links if l.name == link_name), None)
    if link is None:
        raise ValueError(f"Link {link_name!r} not found in {robot.name!r}")
    origin = np.eye(4)
    origin[:3, 3] = np.asarray(center, dtype=np.float64)
    link.collisions.append(
        Collision(name=name, origin=origin, geometry=Geometry(sphere=Sphere(radius=float(radius))))
    )


def place(robot: Robot, T_world_base: NDArray[np.float64], world_link: str = "world") -> None:
    """Bake the rig's world pose in as a fixed world→root joint.

    After this, pyroki FK/costs operate directly in world frame, so world-frame
    targets (e.g. MANO keypoints) need no per-step transform. Reuses an existing
    `world` root joint if the source URDF has one (the xArm does).
    """
    T = np.asarray(T_world_base, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected (4, 4) transform, got {T.shape}")

    root = root_link(robot)
    if root.name == world_link:
        joints = [j for j in robot.joints if j.parent == world_link]
        if len(joints) != 1 or joints[0].type != "fixed":
            raise ValueError(f"Existing {world_link!r} root must have exactly one fixed child joint")
        joints[0].origin = T
        return

    if world_link in {link.name for link in robot.links}:
        raise ValueError(f"Link {world_link!r} exists but is not the root")
    robot.links.insert(0, Link(name=world_link))
    robot.joints.append(
        Joint(name=f"{world_link}_joint", type="fixed", parent=world_link, child=root.name, origin=T)
    )


def _absolutize_mesh_paths(robot: Robot, base_dir: Path) -> None:
    for link in robot.links:
        for v in list(link.visuals) + list(link.collisions):
            mesh = getattr(v.geometry, "mesh", None) if v.geometry is not None else None
            if mesh is None or not mesh.filename:
                continue
            resolved = _resolve_mesh(mesh.filename, base_dir)
            if resolved is None:
                log.warning("Could not resolve mesh %r relative to %s", mesh.filename, base_dir)
            else:
                mesh.filename = str(resolved)


def _resolve_mesh(filename: str, base_dir: Path) -> Path | None:
    if Path(filename).is_absolute():
        return Path(filename) if Path(filename).exists() else None
    candidates = []
    if filename.startswith("package://"):
        tail = Path(filename[len("package://") :])
        # `package://pkg/x.stl` — the "package" dir may be base_dir itself (ruka
        # uses `package://assets/x.stl` for meshes sitting directly in base_dir).
        candidates += [base_dir / tail, base_dir / Path(*tail.parts[1:])]
    else:
        candidates.append(base_dir / filename)
    return next((c.resolve() for c in candidates if c.exists()), None)
