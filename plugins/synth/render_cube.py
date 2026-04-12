from __future__ import annotations

import os
from pathlib import Path

import isaacsim
from isaacsim.simulation_app import SimulationApp


KIT_EXPERIENCE = Path(isaacsim.__file__).resolve().parent / "kit/apps/omni.app.empty.kit"

APP = SimulationApp(
    {
        "headless": True,
        "width": 512,
        "height": 512,
        "renderer": "RealTimePathTracing",
        "extra_args": [
            "--enable",
            "omni.kit.loop-isaac",
            "--enable",
            "omni.kit.usd.layers",
            "--enable",
            "omni.usd",
        ],
    },
    experience=os.fspath(KIT_EXPERIENCE),
)

import carb.settings
import omni.kit.app

manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("omni.usd", True)
manager.set_extension_enabled_immediate("omni.replicator.core", True)

import omni.replicator.core as rep
import omni.usd


def main() -> None:
    out_dir = Path("renders/cube_views").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    omni.usd.get_context().new_stage()
    rep.orchestrator.set_capture_on_play(False)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(parent="/World", intensity=1200, name="DomeLight")
    rep.functional.create.plane(parent="/World", position=(0, 0, -0.5), scale=(4, 4, 1), name="Ground")
    cube = rep.functional.create.cube(parent="/World", position=(0, 0, 0), scale=(1, 1, 1), name="Cube")
    rep.functional.modify.semantics(cube, {"class": "cube"}, mode="add")

    camera_positions = [
        (3.0, 3.0, 2.0),
        (-3.0, 3.0, 2.0),
        (3.0, -3.0, 2.0),
        (-3.0, -3.0, 2.0),
        (0.0, 0.0, 4.0),
    ]

    render_products = []
    for i, position in enumerate(camera_positions):
        camera = rep.functional.create.camera(
            parent="/World",
            name=f"Camera_{i}",
            position=position,
            look_at=(0, 0, 0),
        )
        render_products.append(rep.create.render_product(camera, (512, 512), name=f"view_{i}"))

    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=os.fspath(out_dir))
    writer = rep.writers.get("BasicWriter")
    writer.initialize(backend=backend, rgb=True)
    writer.attach(render_products)

    rep.orchestrator.step()
    rep.orchestrator.wait_until_complete()

    writer.detach()
    for render_product in render_products:
        render_product.destroy()

    print(f"Wrote renders to {out_dir}")
    APP.close()


if __name__ == "__main__":
    main()
