from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from roboreg.differentiable import Robot
from roboreg.hydra_icp import hydra_centroid_alignment, hydra_robust_icp
from roboreg.io import URDFParser
from roboreg.util import (
    clean_xyz,
    compute_vertex_normals,
    depth_to_xyz,
    from_homogeneous,
    generate_ht_optical,
    mask_extract_extended_boundary,
    to_homogeneous,
)
from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server


@dataclass
class HydraConfig:
    host: str = "0.0.0.0"
    port: int = 8021

    ros_package: str = "xarm_description"
    xacro_path: str = "urdf/xarm_device.urdf.xacro"
    urdf: Path = Path()
    root_link_name: str = "link_base"
    end_link_name: str = "link7"
    collision_meshes: bool = False
    depth_conversion_factor: float = 1.0
    z_min: float = 0.01
    z_max: float = 2.0
    number_of_points: int = 5000
    max_distance: float = 0.1
    outer_max_iter: int = 50
    inner_max_iter: int = 10
    output_file: str = "HT_hydra_robust.npy"
    no_boundary: bool = False
    dilation_kernel_size: int = 3
    erosion_kernel_size: int = 10


class Hydra(BasePolicy):
    def __init__(self, cfg: HydraConfig):
        self.cfg = cfg
        self.hist = []

        # parse URDF once
        parser = URDFParser()
        parser.from_ros_xacro(
            ros_package=cfg.ros_package,
            xacro_path=cfg.xacro_path,
        )

        root = cfg.root_link_name
        end = cfg.end_link_name

        if root == "":
            root = parser.link_names_with_meshes(collision=cfg.collision_meshes)[0]
        if end == "":
            end = parser.link_names_with_meshes(collision=cfg.collision_meshes)[-1]

        self.urdf_parser = parser
        self.root_link_name = root
        self.end_link_name = end

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------
    # STEP
    # --------------------------------------------------------
    def step(self, payload: dict) -> dict:
        """
        payload = {
            'joints': (J,),
            'intrinsics': (3,3),
            'depth': (H,W),
            'mask': (H,W)
        }
        """

        # add incoming pose
        self.hist.append(payload)

        # need >= 3 frames
        if len(self.hist) < 3:
            return {"hist": len(self.hist), "required": 3}

        # run optimization
        HT = self._run_hydra(self.hist)

        # reset for next batch
        self.hist = []

        return {"HT": HT}

    # --------------------------------------------------------
    # CORE HYDRA PIPELINE
    # --------------------------------------------------------
    def _run_hydra(self, hist):
        cfg = self.cfg
        device = self.device

        # unpack all payloads
        joints_list = [p["joints"] for p in hist]
        intr_list = [p["intrinsics"] for p in hist]
        depth_list = [p["depth"] for p in hist]
        mask_list = [p["mask"] for p in hist]

        # assume intrinsics same for all
        intrinsics = torch.tensor(intr_list[0], dtype=torch.float32, device=device)

        # shapes
        H, W = depth_list[0].shape
        batch = len(hist)

        # ----------------------------------------------------
        # Robot FK
        # ----------------------------------------------------
        robot = Robot(
            urdf_parser=self.urdf_parser,
            root_link_name=self.root_link_name,
            end_link_name=self.end_link_name,
            collision=cfg.collision_meshes,
            batch_size=batch,
        )

        joint_states = torch.tensor(np.array(joints_list), dtype=torch.float32, device=device)
        robot.configure(joint_states)

        # get mesh vertices + normals
        mesh_vertices = from_homogeneous(robot.configured_vertices)
        mesh_vertices = [mesh_vertices[i].contiguous() for i in range(batch)]

        mesh_normals = [
            compute_vertex_normals(vertices=mesh_vertices[i], faces=robot.faces)
            for i in range(batch)
        ]

        # ----------------------------------------------------
        # Depth → XYZ
        # ----------------------------------------------------
        depths = torch.tensor(np.array(depth_list), dtype=torch.float32, device=device)

        xyzs = depth_to_xyz(
            depth=depths,
            intrinsics=intrinsics,
            z_min=cfg.z_min,
            z_max=cfg.z_max,
            conversion_factor=cfg.depth_conversion_factor,
        )  # B x H x W x 3

        # flatten
        xyzs = xyzs.view(batch, H * W, 3)
        xyzs = to_homogeneous(xyzs)

        ht_optical = generate_ht_optical(batch, dtype=torch.float32, device=device)
        xyzs = torch.matmul(xyzs, ht_optical.transpose(-1, -2))
        xyzs = from_homogeneous(xyzs)  # B x N x 3

        # reshape back to (H,W,3)
        xyzs = xyzs.view(batch, H, W, 3)
        xyzs = [xyz.cpu().numpy() for xyz in xyzs]

        # ----------------------------------------------------
        # Clean masked points
        # ----------------------------------------------------
        observed_vertices = []
        for xyz, mask in zip(xyzs, mask_list, strict=False):
            if cfg.no_boundary:
                m = mask
            else:
                m = mask_extract_extended_boundary(
                    mask,
                    dilation_kernel=np.ones([cfg.dilation_kernel_size] * 2),
                    erosion_kernel=np.ones([cfg.erosion_kernel_size] * 2),
                )

            cleaned = clean_xyz(xyz=xyz, mask=m)
            observed_vertices.append(torch.tensor(cleaned, dtype=torch.float32, device=device))

        # ----------------------------------------------------
        # Downsample robot mesh
        # ----------------------------------------------------
        for i in range(batch):
            idx = torch.randperm(mesh_vertices[i].shape[0])[: cfg.number_of_points]
            mesh_vertices[i] = mesh_vertices[i][idx]
            mesh_normals[i] = mesh_normals[i][idx]

        # ----------------------------------------------------
        # Hydra ICP
        # ----------------------------------------------------
        HT_init = hydra_centroid_alignment(observed_vertices, mesh_vertices)
        HT = hydra_robust_icp(
            HT_init,
            observed_vertices,
            mesh_vertices,
            mesh_normals,
            max_distance=cfg.max_distance,
            outer_max_iter=cfg.outer_max_iter,
            inner_max_iter=cfg.inner_max_iter,
        )

        return HT.cpu().numpy()


def main(cfg: HydraConfig):
    policy = Hydra(cfg)
    server = Server(policy, cfg.host, cfg.port)
    server.serve()


if __name__ == "__main__":
    main(tyro.cli(HydraConfig))
