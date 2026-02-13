from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
from rich import print
import tyro
from webpolicy.client import Client

from xclients.core.cfg import Config, spec
from xclients.core.run import ez_load_urdf
from xclients.core.run.blueprint import init_blueprint
from xclients.core.run.fustrum import log_fustrum
from xclients.core.tf import FLU2RDF

logging.basicConfig(level=logging.INFO)
np.set_printoptions(precision=3, suppress=True)


@dataclass
class DA3Config(Config):
    cams: list[int] = field(default_factory=lambda: [0])
    show: bool = True
    downscale: int = 1
    urdf: Path = Path("data/robot.urdf")


def main(cfg: DA3Config) -> None:
    client = Client(cfg.host, cfg.port)

    rr.init("DA3", spawn=True)
    rr.log("/", rr.ViewCoordinates.FLU, static=True)
    init_blueprint(cfg.cams)

    caps = {cam: cv2.VideoCapture(cam) for cam in cfg.cams}

    ez_load_urdf(cfg.urdf)

    logging.info("Displaying frames from camera 0; press 'q' to quit.")
    begin = True
    while True:
        frames = {cam: caps[cam].read() for cam in cfg.cams}
        frames = {cam: frame for cam, (ret, frame) in frames.items() if ret}
        if cfg.downscale > 1:
            frames = {
                cam: cv2.resize(
                    frame,
                    (
                        frame.shape[1] // cfg.downscale,
                        frame.shape[0] // cfg.downscale,
                    ),
                    interpolation=cv2.INTER_AREA,
                )
                for cam, frame in frames.items()
            }

        print(spec(frames))
        frames = list(frames.values())

        fx, fy = 515.0, 515.0
        w, h = frames[0].shape[1], frames[0].shape[0]
        cx, cy = w / 2.0, h / 2.0
        intr = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

        payload = {
            "image": frames,
            "intrinsics": np.array([intr for _ in cfg.cams]),
        }

        out = client.step(payload)
        if not out:
            logging.error("Failed to read frame from camera 0")
            continue

        print(spec(out))
        d = np.concatenate(out["depth"])
        cmap = 255.0
        d = ((d - np.min(d)) / (np.max(d) - np.min(d)) * cmap).astype(np.uint8)

        print(out["extrinsics"])
        print(out["intrinsics"])
        print()

        points, colors = out["points"], out["colors"][:, ::-1]  # BGR to RGB

        HT = np.array(
            [
                [-0.529, 0.82, -0.219, 0.708],
                [-0.657, -0.559, -0.505, 0.547],
                [-0.536, -0.124, 0.835, 0.758],
                [
                    -0.0,
                    -0.0,
                    -0.0,
                    1.0,
                ],
            ],
        )

        flu2rdf = HT @ FLU2RDF
        rr.log(
            "world",
            rr.Transform3D(
                translation=flu2rdf[:3, 3],
                mat3x3=flu2rdf[:3, :3],
            ),
            static=True,
        )

        extr = {k: out["extrinsics"][i] for i, k in enumerate(cfg.cams)}
        intr = {k: out["intrinsics"][i] for i, k in enumerate(cfg.cams)}
        info = {
            k: {
                "intrinsics": intr[k],
                "extrinsics": extr[k],
                "width": frames[i].shape[1],
                "height": frames[i].shape[0],
                "frame": frames[i],
            }
            for i, k in enumerate(cfg.cams)
        }

        if begin:
            log_fustrum(info, root=Path("world"))

        # maxd = 1.0
        # colors = colors[np.linalg.norm(points, axis=1) < 1.0]
        # points = points[np.linalg.norm(points, axis=1) < 1.0]

        vp3d = rr.Points3D(points, colors=colors, radii=0.002)
        rr.log("world/scene/points", vp3d)  # static=True

        for k, cam in info.items():
            rr.log(
                f"world/cam/{k}/image",
                rr.Image(cam["frame"], color_model="BGR").compress(jpeg_quality=75),
                static=False,
            )

        if cfg.show:
            cv2.imshow("Camera", d)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(tyro.cli(DA3Config))
