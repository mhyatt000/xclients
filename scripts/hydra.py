import webpolicy
from rich.pretty import pprint
from rich import print

import open3d as o3d
import numpy as np


import open3d as o3d
import numpy as np


from xclients.gui.viewer import O3DPointCloudViewer
import time
from functools import wraps
from webpolicy.deploy.client import WebsocketClientPolicy as Client

from dataclasses import dataclass
import tyro
import cv2
import numpy as np

from pathlib import Path


def show(img: np.ndarray, title: str = 'Image'):

    if False:
# convert from min,max to mean0-std1
        img = (img - np.mean(img)) / np.std(img)
# convert from min,mix to 0-1
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
# add color 
        img = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_COOL)

    pprint((img.shape, img.dtype, img.min(), img.max()))
    cv2.imshow('Selfie', img)

    if key:= cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        quit()

def color_points(points):
    """ returns a color array based on z value 
    interpolate between blue (low) min to red (high) max
    """
    z = points[:,2]
    zmin, zmax = z.min(), z.max()
    znorm = (z - zmin) / (zmax - zmin + 1e-8)
    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    colors[:,0] = (znorm * 255).astype(np.uint8) # R
    colors[:,2] = ((1 - znorm) * 255).astype(np.uint8) # B
    return colors

@dataclass
class Config:
    do_opt: bool = False


def timeit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = f(*args, **kwargs)
        end = time.perf_counter()
        print(f"Function {f.__name__} took {end - start:.4f} seconds | {1/(end - start):.2f} FPS")
        return result
    return wrapper

def icp_refine(src_model, tgt_cloud, init_T, voxel=0.01, max_corr=0.03):
    src = src_model.voxel_down_sample(voxel)
    tgt = tgt_cloud.voxel_down_sample(voxel)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3*voxel, max_nn=50)) 

    # for corr in np.linspace(max_corr*5, max_corr/10, 3):
    loss = o3d.pipelines.registration.TukeyLoss(k=max_corr)  # robust
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500)
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, max_corr, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
        criteria=crit,
    )
    # tighten and repeat
    for d in [0.02, 0.015, 0.01]:
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, d, reg.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
            criteria=crit,
        )
    return reg.transformation, reg.fitness, reg.inlier_rmse


def main(cfg: Config):

    payload = np.load('roboreg-payload.npz', allow_pickle=True)
    payload = {k:payload[k] for k in payload.files}
    payload['mesh_vertices'] = list(payload['mesh_vertices'].item().values())
    payload['observed_vertices'] = list(payload['observed_vertices'].item().values())
    HT = payload['HT']

    # take 0
    from rich.progress import track
    for i in track(range(0,len(payload['mesh_vertices']),10)):
        stop, step = len(payload['mesh_vertices']), len(payload['mesh_vertices'])//5
        stop, step = i+1, 1
        mesh_v = np.vstack(payload['mesh_vertices'][i:stop: step])
        obs_v = np.vstack(payload['observed_vertices'][i:stop: step])

        # sparsify
        # mesh_v , obs_v = mesh_v[::3], obs_v[::3]

        # one is green and one is pink
        colors_mesh = np.tile(np.array([[1.0, 0.0, 1.0]]), (mesh_v.shape[0], 1))
        colors_obs = np.tile(np.array([[0.0, 1.0, 0.0]]), (obs_v.shape[0], 1))

        if cfg.do_opt:
            t, fitness, rmse = icp_refine(
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh_v)),
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obs_v)),
                HT,
                voxel=0.01,
                max_corr=0.05,
            )
            print(fitness, rmse)
            # quit()
        else:
            t = HT

        print(t)
        print()
        t = np.linalg.inv(HT)
        print(t)

        mesh_v = np.matmul(t[:3, :3], mesh_v.T).T + t[:3, 3]
        # obs_v = np.matmul(HT[:3, :3], obs_v.T).T + HT[:3, 3]

        # random or your own Nx3 points
        mpcd = o3d.geometry.PointCloud()
        mpcd.points = o3d.utility.Vector3dVector(mesh_v)
        mpcd.colors = o3d.utility.Vector3dVector(colors_mesh)
        opcd = o3d.geometry.PointCloud()
        opcd.points = o3d.utility.Vector3dVector(obs_v)
        opcd.colors = o3d.utility.Vector3dVector(colors_obs)

        o3d.visualization.draw_geometries([mpcd, opcd])

    quit()

    out = None
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            continue

        # resize downsample n times
        n = cfg.resize
        _h, _w = frame.shape[:2]
        h , w = _h // n, _w // n
        frame = cv2.resize(frame, (w,h))

        infer = timeit(client.infer)
        # out = infer({'img': frame, 'max_depth': cfg.maxd, 'relative': cfg.relative})
        # out={}

        cv2.imshow('Selfie', frame)
        if out is None or cv2.waitKey(1) & 0xFF == ord('n'): # next
            out = infer({})
        else:
            continue

        # p = points[i]
        # xyz = p.reshape(-1, 3)
        # i+=1
        # print(p)

        xyz = out['pcd']['xyz'] 
        xyz =   xyz.reshape(-1, 3)
        """
        view_tf = -np.eye(3)
        view_tf[0,0] = 1
        xyz = xyz @ view_tf

        zm, zs = xyz[:,-1].mean(), xyz[:,-1].std()
        # filter for only points within 1 stddev
        mask = (xyz[:,-1] > zm - zs) & (xyz[:,-1] < zm + zs)
        xyz = xyz[mask]
        # filter out points very close together
        dists = np.linalg.norm(xyz[:,None,:] - xyz[None,:,:], axis=-1)
        min_dist = 0.01
        close_points = (dists < min_dist).sum(axis=-1) > 1
        xyz = xyz[~close_points]
        """

        print(xyz.shape)
        if not np.isfinite(xyz).all() or len(xyz) == 0:
            print("Non-finite points detected")
            xyz = xyz[np.isfinite(xyz).all(axis=-1)]

        # quit()

        print(color_points(xyz).shape)

        viewer.update(xyz, rgb=color_points(xyz))
        viewer.win.post_redraw()
        print(viewer._latest_xyz.shape)
        viewer.app.run_one_tick()
        # input()

        depth, cmap = out.get('depth'), out.get('cmap')
        if depth is None:
            continue

        pprint((depth.min(), depth.max(), depth.mean()))

        cmap = cv2.resize(cmap, (_w, _h))
        if cfg.extreme:
            cmap = cmap/255
            cmap = cmap**2 
            cmap = (cmap * 255).astype(np.uint8)

        show(cmap, title="Webcam Feed")


if __name__ == "__main__":
    main(tyro.cli(Config))
