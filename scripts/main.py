import webpolicy
from rich.pretty import pprint

from viewer import O3DPointCloudViewer
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

@dataclass
class Config:

    host: str
    port: int = 8001

    cam: int = 0
    maxd: float = 20.0 # max depth
    relative: bool = False # relative colors
    resize: int = 5 # downsample n times
    extreme: bool = False # use extreme color


def timeit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = f(*args, **kwargs)
        end = time.perf_counter()
        print(f"Function {f.__name__} took {end - start:.4f} seconds | {1/(end - start):.2f} FPS")
        return result
    return wrapper

def main(cfg: Config):
    print("Hello from dav-policy!")

    cam = cv2.VideoCapture(cfg.cam)
    client = Client(host=cfg.host, port=cfg.port)
    viewer = O3DPointCloudViewer(target_hz=30, point_size=5.0)
    # viewer.app.run()

    """
    depths = np.load('/Users/matthewhyatt/outputs/xgym.camera.side_depths.npz')
    depths = depths[depths.files[0]]

    _, height,width = depths.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - width / 2) / 470.4
    y = (y - height / 2) / 470.4
    z = np.array(depths)
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1) # .reshape(-1, 3)
    i = 0
    pprint(depths.shape)
    """

    out = None
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            continue

        print('a')
        # resize downsample n times
        n = cfg.resize
        _h, _w = frame.shape[:2]
        h , w = _h // n, _w // n
        frame = cv2.resize(frame, (w,h))

        infer = timeit(client.infer)
        # out = infer({'img': frame, 'max_depth': cfg.maxd, 'relative': cfg.relative})
        # out={}

        cv2.imshow('Selfie', frame)
        # if out is None or cv2.waitKey(1) & 0xFF == ord('n'): # next
            # out = infer({})

        # p = points[i]
        # xyz = p.reshape(-1, 3)
        # i+=1
        # print(p)

        # xyz = out['pcd']['xyz'] 
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

        viewer.update(xyz)
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
