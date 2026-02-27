#!/usr/bin/env python3
"""
Interactive SAM3D Mesh Transformation Debugger
Creates a Gradio UI to visualize and debug 3D transformations
"""
import gradio as gr
import numpy as np
import cv2
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import sys

# Add xclients to path
sys.path.insert(0, str(Path.home() / "xclients" / "src"))

def load_latest_sam3d_result():
    """Load the most recent SAM3D result"""
    results_dir = Path.home() / "xclients" / "scripts" / "sam3do_results"
    if not results_dir.exists():
        return None, None, None, None, None, None, None, None
    
    result_dirs = sorted(results_dir.glob("result_*"))
    if not result_dirs:
        return None, None, None, None, None, None, None, None
    
    latest = result_dirs[-1]
    
    # Load JSON
    with open(latest / "output.json") as f:
        data = json.load(f)
    
    # Load images
    original = cv2.imread(str(latest / "original.png"))
    mask = cv2.imread(str(latest / "sam3_mask.png"), cv2.IMREAD_GRAYSCALE)
    
    # Extract values
    translation = np.array(data["translation"]["data"]).flatten()
    rotation = np.array(data["rotation"]["data"]).flatten()
    scale = float(np.array(data["scale"]["data"]).flatten()[0])
    
    if "intrinsics" in data:
        intrinsics = np.array(data["intrinsics"]["data"])
    else:
        intrinsics = None
    
    # Load mesh if available
    vertices = None
    faces = None
    
    # Check for mesh.npz file first (new format)
    mesh_npz_path = latest / "mesh.npz"
    if mesh_npz_path.exists():
        mesh_data = np.load(mesh_npz_path)
        vertices = mesh_data["vertices"].astype(np.float32)
        faces = mesh_data["faces"].astype(np.int32)
    elif "mesh" in data and isinstance(data["mesh"], dict):
        # Fall back to JSON format (old format)
        mesh_data = data["mesh"]
        if "vertices" in mesh_data:
            verts_data = mesh_data["vertices"]
            if isinstance(verts_data, dict) and "data" in verts_data:
                vertices = np.array(verts_data["data"], dtype=np.float32)
            elif isinstance(verts_data, list):
                vertices = np.array(verts_data, dtype=np.float32)
        
        if "faces" in mesh_data:
            faces_data = mesh_data["faces"]
            if isinstance(faces_data, dict) and "data" in faces_data:
                faces = np.array(faces_data["data"], dtype=np.int32)
            elif isinstance(faces_data, list):
                faces = np.array(faces_data, dtype=np.int32)
    
    return original, mask, translation, rotation, scale, intrinsics, vertices, faces

def quaternion_to_euler(quat_wxyz):
    """Convert quaternion [w,x,y,z] to Euler angles [roll, pitch, yaw] in degrees"""
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    r = R.from_quat(quat_xyzw)
    euler_rad = r.as_euler('xyz')
    return np.degrees(euler_rad)

def euler_to_quaternion(euler_deg):
    """Convert Euler angles [roll, pitch, yaw] in degrees to quaternion [w,x,y,z]"""
    euler_rad = np.radians(euler_deg)
    r = R.from_euler('xyz', euler_rad)
    quat_xyzw = r.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

def create_axis_mesh(length=0.1):
    """Create colored axis lines for visualization"""
    axes = {
        'X': (np.array([[0,0,0], [length,0,0]]), (255, 0, 0)),  # Red
        'Y': (np.array([[0,0,0], [0,length,0]]), (0, 255, 0)),  # Green
        'Z': (np.array([[0,0,0], [0,0,length]]), (0, 0, 255)),  # Blue
    }
    return axes

def apply_transform(vertices, translation, rotation_quat, scale, 
                   flip_x, flip_y, flip_z,
                   coord_system_mode,
                   rotation_mode):
    """
    Apply transformation with various options
    
    coord_system_mode:
      - "none": No coordinate flip
      - "flip_xy": Flip X and Y (PyTorch3D -> OpenCV)
      - "flip_all": Flip all axes
    
    rotation_mode:
      - "normal": Apply rotation as-is
      - "transformed": Transform rotation for coord system
      - "inverse": Use inverse rotation
    """
    # Convert quaternion to rotation matrix
    quat_xyzw = [rotation_quat[1], rotation_quat[2], rotation_quat[3], rotation_quat[0]]
    rot = R.from_quat(quat_xyzw)
    rot_matrix = rot.as_matrix()
    
    # Apply rotation mode
    if rotation_mode == "transformed" and coord_system_mode == "flip_xy":
        coord_flip = np.diag([-1, -1, 1])
        rot_matrix = coord_flip @ rot_matrix @ coord_flip.T
    elif rotation_mode == "inverse":
        rot_matrix = rot_matrix.T
    
    # Step 1: Scale
    v = vertices * scale
    
    # Step 2: Rotate
    v = v @ rot_matrix.T
    
    # Step 3: Manual flips
    if flip_x:
        v[:, 0] *= -1
    if flip_y:
        v[:, 1] *= -1
    if flip_z:
        v[:, 2] *= -1
    
    # Step 4: Coordinate system mode
    if coord_system_mode == "flip_xy":
        v[:, 0] *= -1
        v[:, 1] *= -1
        t = translation.copy()
        t[0] *= -1
        t[1] *= -1
    elif coord_system_mode == "flip_all":
        v *= -1
        t = -translation
    else:
        t = translation.copy()
    
    # Step 5: Translate
    v = v + t
    
    return v

def project_to_image(vertices_3d, intrinsics, img_shape):
    """Project 3D vertices to 2D image coordinates"""
    h, w = img_shape[:2]
    
    if intrinsics is None:
        # Default intrinsics
        fx = fy = 500
        cx, cy = w/2, h/2
    else:
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Denormalize if needed
        if fx < 10:
            fx *= w
            fy *= h
            cx *= w
            cy *= h
    
    # Project
    x_2d = (vertices_3d[:, 0] / vertices_3d[:, 2]) * fx + cx
    y_2d = (vertices_3d[:, 1] / vertices_3d[:, 2]) * fy + cy
    
    return np.stack([x_2d, y_2d], axis=1)

def visualize_transform(
    # Transform parameters
    tx, ty, tz,
    roll, pitch, yaw,
    scale,
    # Flip options
    flip_x, flip_y, flip_z,
    # Mode options
    coord_system_mode,
    rotation_mode,
    # View options
    show_axes,
    show_bbox,
    show_origin,
):
    """Main visualization function - uses simple cube"""
    
    # Load data
    original, mask, trans_orig, rot_orig, scale_orig, intrinsics, _, _ = load_latest_sam3d_result()
    
    if original is None:
        return None, "No SAM3D results found!"
    
    # Use current slider values
    translation = np.array([tx, ty, tz])
    rotation_quat = euler_to_quaternion([roll, pitch, yaw])
    
    # Create simple cube for visualization
    size = 0.1
    cube_vertices = np.array([
        [-size, -size, -size], [size, -size, -size],
        [size, size, -size], [-size, size, -size],
        [-size, -size, size], [size, -size, size],
        [size, size, size], [-size, size, size],
    ])
    
    # Create axes
    axes = create_axis_mesh(length=0.15)
    
    # Transform cube
    cube_transformed = apply_transform(
        cube_vertices, translation, rotation_quat, scale,
        flip_x, flip_y, flip_z,
        coord_system_mode, rotation_mode
    )
    
    # Transform axes
    axes_transformed = {}
    for name, (pts, color) in axes.items():
        pts_t = apply_transform(
            pts, translation, rotation_quat, scale,
            flip_x, flip_y, flip_z,
            coord_system_mode, rotation_mode
        )
        axes_transformed[name] = (pts_t, color)
    
    # Get image dimensions
    h, w = original.shape[:2]
    
    # Setup intrinsics
    if intrinsics is None:
        fx = fy = 500
        cx, cy = w/2, h/2
    else:
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Denormalize if needed
        if fx < 10:
            fx *= w
            fy *= h
            cx *= w
            cy *= h
    
    # Project to image
    cube_2d = project_to_image(cube_transformed, intrinsics, original.shape)
    
    # Draw on image
    vis = original.copy()
    
    # Draw axes
    if show_axes:
        for name, (pts_3d, color) in axes_transformed.items():
            pts_2d = project_to_image(pts_3d, intrinsics, original.shape)
            if pts_2d.shape[0] >= 2:
                pt1 = tuple(pts_2d[0].astype(int))
                pt2 = tuple(pts_2d[1].astype(int))
                cv2.arrowedLine(vis, pt1, pt2, color, 3, tipLength=0.3)
                cv2.putText(vis, name, pt2, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Draw bbox
    if show_bbox:
        # Draw cube edges
        edges = [
            (0,1), (1,2), (2,3), (3,0),  # Back face
            (4,5), (5,6), (6,7), (7,4),  # Front face
            (0,4), (1,5), (2,6), (3,7),  # Connecting edges
        ]
        for i, j in edges:
            pt1 = tuple(cube_2d[i].astype(int))
            pt2 = tuple(cube_2d[j].astype(int))
            cv2.line(vis, pt1, pt2, (0, 255, 255), 2)
    
    # Draw origin
    if show_origin:
        origin_3d = apply_transform(
            np.array([[0, 0, 0]]), translation, rotation_quat, 1.0,
            flip_x, flip_y, flip_z,
            coord_system_mode, rotation_mode
        )
        origin_2d = project_to_image(origin_3d, intrinsics, original.shape)
        if origin_2d.shape[0] > 0:
            pt = tuple(origin_2d[0].astype(int))
            cv2.circle(vis, pt, 8, (255, 0, 255), -1)
            cv2.putText(vis, "Origin", (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    # Draw mask overlay
    if mask is not None:
        mask_color = np.zeros_like(vis)
        mask_color[mask > 128] = [255, 255, 0]  # Yellow
        vis = cv2.addWeighted(vis, 0.7, mask_color, 0.3, 0)
    
    # Generate diagnostics
    rot_matrix = R.from_quat([rotation_quat[1], rotation_quat[2], rotation_quat[3], rotation_quat[0]]).as_matrix()
    det = np.linalg.det(rot_matrix)
    is_orthonormal = np.allclose(rot_matrix.T @ rot_matrix, np.eye(3), atol=1e-6)
    
    # Transform origin and basis vectors
    origin = translation
    ex = rot_matrix[:, 0] * scale
    ey = rot_matrix[:, 1] * scale
    ez = rot_matrix[:, 2] * scale
    
    # Calculate effective cube size
    effective_size = size * scale * 2  # *2 because size is half-extent
    
    info = f"""
**Cube Visualization:**
- Size: {effective_size:.3f}m ({size*2:.3f}m base × {scale:.3f} scale)
- Cube center (2D): {cube_2d.mean(axis=0).astype(int)}
- Image center: [{w//2}, {h//2}]

**Transform:**
- Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]
- Distance: {np.linalg.norm(translation):.3f}m
- Rotation (Euler): [{roll:.1f}°, {pitch:.1f}°, {yaw:.1f}°]
- Scale: {scale:.4f}

**Validation:**
- Det(R): {det:.6f} {'✓' if abs(det - 1.0) < 0.01 else '✗ NOT VALID'}
- Orthonormal: {'✓' if is_orthonormal else '✗'}

**Transformed Basis:**
- Origin: [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]
- +X axis: [{ex[0]:.3f}, {ex[1]:.3f}, {ex[2]:.3f}]
- +Y axis: [{ey[0]:.3f}, {ey[1]:.3f}, {ey[2]:.3f}]
- +Z axis: [{ez[0]:.3f}, {ez[1]:.3f}, {ez[2]:.3f}]

**Intrinsics:**
- fx={fx:.1f}, fy={fy:.1f}
- cx={cx:.1f}, cy={cy:.1f}

**Modes:**
- Coord system: {coord_system_mode}
- Rotation mode: {rotation_mode}
- Manual flips: X={flip_x}, Y={flip_y}, Z={flip_z}

**Original SAM3D Values:**
- Trans: [{trans_orig[0]:.3f}, {trans_orig[1]:.3f}, {trans_orig[2]:.3f}]
- Rot: [{rot_orig[0]:.3f}, {rot_orig[1]:.3f}, {rot_orig[2]:.3f}, {rot_orig[3]:.3f}]
- Scale: {scale_orig:.4f}
"""
    
    return vis, info

def reset_to_original():
    """Load original SAM3D values"""
    _, _, translation, rotation, scale, _, _, _ = load_latest_sam3d_result()
    if translation is not None:
        euler = quaternion_to_euler(rotation)
        return (
            float(translation[0]), float(translation[1]), float(translation[2]),
            float(euler[0]), float(euler[1]), float(euler[2]),
            float(scale)
        )
    return 0, 0, 1, 0, 0, 0, 1

# Create Gradio interface
with gr.Blocks(title="SAM3D Transform Debugger") as app:
    gr.Markdown("# 🔧 SAM3D Mesh Transformation Debugger")
    gr.Markdown("Interactive tool to debug 3D pose transformations and coordinate system conversions")
    
    with gr.Row():
        with gr.Column(scale=2):
            output_image = gr.Image(label="Transformed Visualization", type="numpy")
        
        with gr.Column(scale=1):
            info_box = gr.Markdown("Load data to see diagnostics...")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📍 Translation")
            tx = gr.Slider(-2, 2, value=0.195, step=0.01, label="X (right)")
            ty = gr.Slider(-2, 2, value=-0.148, step=0.01, label="Y (up in PyTorch3D)")
            tz = gr.Slider(0, 5, value=1.02, step=0.01, label="Z (depth)")
        
        with gr.Column():
            gr.Markdown("### 🔄 Rotation (Euler Angles)")
            roll = gr.Slider(-180, 180, value=0, step=1, label="Roll (X-axis)")
            pitch = gr.Slider(-180, 180, value=0, step=1, label="Pitch (Y-axis)")
            yaw = gr.Slider(-180, 180, value=0, step=1, label="Yaw (Z-axis)")
        
        with gr.Column():
            gr.Markdown("### 📏 Scale")
            scale = gr.Slider(0.01, 2, value=0.167, step=0.01, label="Scale")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔀 Manual Flips")
            flip_x = gr.Checkbox(label="Flip X", value=False)
            flip_y = gr.Checkbox(label="Flip Y", value=False)
            flip_z = gr.Checkbox(label="Flip Z", value=False)
        
        with gr.Column():
            gr.Markdown("### 🌐 Coordinate System")
            coord_mode = gr.Radio(
                ["none", "flip_xy", "flip_all"],
                label="Coordinate Transform",
                value="flip_xy",
                info="none=identity, flip_xy=PyTorch3D→OpenCV, flip_all=mirror"
            )
        
        with gr.Column():
            gr.Markdown("### 🔄 Rotation Mode")
            rot_mode = gr.Radio(
                ["normal", "transformed", "inverse"],
                label="Rotation Handling",
                value="transformed",
                info="normal=as-is, transformed=coord-aware, inverse=R^T"
            )
    
    with gr.Row():
        gr.Markdown("### 👁️ Visualization Options")
        show_axes = gr.Checkbox(label="Show Axes (RGB=XYZ)", value=True)
        show_bbox = gr.Checkbox(label="Show BBox", value=True)
        show_origin = gr.Checkbox(label="Show Origin", value=True)
    
    with gr.Row():
        update_btn = gr.Button("🔄 Update Visualization", variant="primary")
        reset_btn = gr.Button("↩️ Reset to SAM3D Values")
    
    # Connect buttons
    inputs = [
        tx, ty, tz, roll, pitch, yaw, scale,
        flip_x, flip_y, flip_z,
        coord_mode, rot_mode,
        show_axes, show_bbox, show_origin
    ]
    
    update_btn.click(visualize_transform, inputs=inputs, outputs=[output_image, info_box])
    
    reset_btn.click(
        reset_to_original,
        outputs=[tx, ty, tz, roll, pitch, yaw, scale]
    )
    
    gr.Markdown("""
    ### 💡 Tips:
    1. **Start with Reset** to load your SAM3D output
    2. **Check Diagnostics** - Det(R) should be ~1.0, axes should be orthonormal
    3. **Toggle modes** to see which coordinate convention works
    4. **Watch the axes** - RGB lines show XYZ orientation
    5. **Yellow overlay** is your SAM mask - mesh should align with it
    
    ### 🐛 Common Issues:
    - **Det(R) ≠ 1.0**: Quaternion order wrong or reflection in rotation
    - **Axes point weird directions**: Coordinate system mismatch
    - **Position OK but wrong angle**: Need "transformed" rotation mode
    - **Upside down**: Try different Y flip combinations
    """)

if __name__ == "__main__":
    app.launch(share=True, server_name="0.0.0.0", server_port=7860)
