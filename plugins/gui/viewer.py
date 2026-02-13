# pip install "open3d[gui]"
from __future__ import annotations

import threading
import time

import numpy as np
import open3d as o3d

# os.environ.setdefault("OPEN3D_RENDERING_ENGINE", "OpenGL")  # Intel mac


class O3DPointCloudViewer:
    """Threaded, nonblocking point cloud viewer. Call update(xyz, rgb) from any thread."""

    def __init__(self, title="PointCloud Viewer", target_hz=30.0, point_size=2.0):
        self._title = title
        self._dt_ms = int(1000 / target_hz)
        self._lock = threading.Lock()
        self._latest_xyz = None
        self._latest_rgb = None
        self._running = True
        self._point_size = point_size

        app = o3d.visualization.gui.Application.instance
        app.initialize()
        print("prewin")
        win = app.create_window(self._title, 1280, 720)
        print("prescene")
        scene = o3d.visualization.gui.SceneWidget()
        scene.scene = o3d.visualization.rendering.Open3DScene(win.renderer)
        win.add_child(scene)

        print("win/scene")

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = self._point_size

        pcd = o3d.geometry.PointCloud()
        scene.scene.add_geometry("pcd", pcd, mat)
        scene.scene.show_axes(True)
        scene.setup_camera(60.0, o3d.geometry.AxisAlignedBoundingBox([-1, -1, -1], [1, 1, 1]), [0, 0, 0])
        print("cam")

        self.app = app
        self.win = win
        self.scene = scene
        self.mat = mat
        self.pcd = pcd

        def on_layout(_):
            scene.frame = win.content_rect

        win.set_on_layout(on_layout)

        self._thread = threading.Thread(target=self.app.run, daemon=True)

    def look(self):
        xyz = self._latest_xyz
        center = xyz.mean(0) * 0
        h = 0.5
        center[2] += h
        radius = 5
        speed = 0.1
        t = time.time()
        if False:  # y
            eye = center + radius * np.array([np.cos(t * 0.5), -0.5, np.sin(t * 0.5)])  # flip Y
            self.scene.scene.camera.look_at(center, eye, [0, -1, 0])
        if False:  # z
            eye = center + radius * np.array([0.3, np.cos(0.5 * t), np.sin(0.5 * t)])
            self.scene.scene.camera.look_at(center, eye, [1, 0, 0])

        # x
        eye = center + radius * np.array([np.cos(speed * t), np.sin(speed * t), 0.2])
        self.scene.scene.camera.look_at(center, eye, [0, 0, 1])

    def update(self, xyz: np.ndarray, rgb: np.ndarray | None = None):
        self._latest_xyz = np.asarray(xyz, np.float32)
        if rgb is not None:
            c = np.asarray(rgb, np.float32)
            if c.max() > 1:  # allow 0–255
                c /= 255.0
            self._latest_rgb = c
        else:
            self._latest_rgb = None

        xyz, rgb = self._latest_xyz, self._latest_rgb
        if xyz is not None:
            self.pcd.points = o3d.utility.Vector3dVector(xyz)
            # if rgb is not None:
            # self.pcd.colors = o3d.utility.Vector3dVector(rgb)
            # else:
            self.pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(xyz))
            self.scene.scene.remove_geometry("pcd")
            self.scene.scene.add_geometry("pcd", self.pcd, self.mat)

    def close(self):
        self._running = False


# -------------------------------------------------------------------


# Example integration into your inference loop
def main(client, cam):
    viewer = O3DPointCloudViewer(target_hz=30)
    try:
        while True:
            # get frame from camera and run inference
            ret, frame = cam.read()
            if not ret:
                break
            out = client.infer({"img": frame})
            pcd_xyz = out["pcd.xyz"]  # shape (N,3)
            pcd_rgb = out["pcd.color"]  # shape (N,3) in 0–255 or 0–1
            viewer.update(pcd_xyz, pcd_rgb)
            time.sleep(1 / 30)  # pacing
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
