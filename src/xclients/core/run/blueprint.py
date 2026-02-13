from __future__ import annotations

from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb


def create_blueprint(cameras: list, parent: Path = Path("world")) -> rrb.Blueprint:
    # get only n cameras evenly spaced
    # n_cameras = len(cameras)
    # step = n_cameras // 4
    # cameras = cameras[::step][:4]

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(),
            rrb.Vertical(
                contents=[
                    rrb.Spatial2DView(
                        origin=f"{parent}/cam/{cam}/",
                        contents=[
                            "+ $origin/**",
                        ],
                    )
                    for cam in cameras
                ]
            ),
            column_shares=[4, 1],
        ),
        collapse_panels=True,
    )
    return blueprint


def init_blueprint(cameras: list):
    blueprint = create_blueprint(cameras=cameras)
    rr.send_blueprint(blueprint)
