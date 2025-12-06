import open3d as o3d


class O3DPointStepper:
    def __init__(self):
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        self.pcd = o3d.geometry.PointCloud()
        pts = np.random.random((17, 3))
        colors = pts
        self.pcd.points = o3d.utility.Vector3dVector(pts)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        self.vis.add_geometry(self.pcd)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        # self.vis.draw_geometries([pc, frame])
        self.vis.add_geometry(frame)

        self.lines = o3d.geometry.LineSet()
        self.vis.add_geometry(self.lines)

        # source,target = self.prepare_data()
        # self.vis.add_geometry(source)

    def prepare_data(self):
        pcd_data = o3d.data.DemoICPPointClouds()
        source_raw = o3d.io.read_point_cloud(pcd_data.paths[0])
        target_raw = o3d.io.read_point_cloud(pcd_data.paths[1])
        source = source_raw.voxel_down_sample(voxel_size=0.02)
        target = target_raw.voxel_down_sample(voxel_size=0.02)

        trans = [
            [0.862, 0.011, -0.507, 0.0],
            [-0.139, 0.967, -0.215, 0.7],
            [0.487, 0.255, 0.835, -1.4],
            [0.0, 0.0, 0.0, 1.0],
        ]
        source.transform(trans)
        flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        source.transform(flip_transform)
        target.transform(flip_transform)
        return source, target

    def step(self, arr: np.ndarray):
        pts = arr[:, :3]
        conf = arr[:, 3]

        # random float
        # pts = np.random.random(pts.shape)
        colors = np.stack([conf, np.zeros_like(conf), np.zeros_like(conf)], axis=-1)
        colors = np.ones_like(pts) * conf.reshape(conf.shape[0], 1)

        print(pts, colors)
        self.pcd.points = o3d.utility.Vector3dVector(pts)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        self.lines.points = o3d.utility.Vector3dVector(pts)
        self.lines.lines = o3d.utility.Vector2iVector(joint_pairs)
        self.lines.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in joint_pairs])

        self.vis.update_geometry(self.pcd)
        self.vis.update_geometry(self.lines)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()
