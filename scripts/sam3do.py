import jax
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging
from typing import Dict, Union, Optional
import time

import json
from datetime import datetime
import numpy as np
import cv2
import numpy as np
from rich import print
import tyro
from webpolicy.client import Client

from xclients.renderer import Renderer
from xclients.core.cfg import Config, spec
from sam_utils.image_utils import load_image, get_frame
from sam_utils.save_results import ResultSaver
from sam3_utils.render_pipeline import Sam3DoRenderPipeline

def spec(tree) -> dict:
    return jax.tree.map(lambda x: (x.shape, x.dtype), tree)

@dataclass
class CameraInput:
    """Configuration for camera input"""
    device: int = 0

@dataclass
class ImageInput:
    """Configuration for image file input"""
    image_path: Path

@dataclass
class SAMConfig(Config):
    """Config for SAM3 server"""
    host: str = "localhost"
    port: int = 8000
    prompt: str = "object"

@dataclass
class SAM3DoConfig(Config):
    """Config for SAM3Do server"""
    input_source: Union[CameraInput, ImageInput] = tyro.MISSING
    host: str = "localhost"
    port: int = 8003
    show: bool = False
    timeout: float = 60.0  # Timeout for server processing
    sam3: SAMConfig = field(default_factory=SAMConfig)

def _render_sam3do_output_legacy(renderer, frame, sam3do_out, results_dir, frame_idx, cfg):
    """Render SAM3Do mesh output on frame"""
    try:
        # Extract mesh data
        if "mesh" not in sam3do_out:
            logging.warning("No mesh in SAM3Do output, skipping rendering")
            return
        
        mesh_data = sam3do_out["mesh"]
        logging.info(f"Mesh type: {type(mesh_data)}")
        
        # Handle mesh as dict (from server) or as object (from direct inference)
        if isinstance(mesh_data, dict):
            vertices = np.array(mesh_data["vertices"], dtype=np.float32)
            faces = np.array(mesh_data["faces"], dtype=np.int32)
        elif isinstance(mesh_data, list):
            if len(mesh_data) == 0:
                logging.warning("Mesh list is empty")
                return
            mesh_obj = mesh_data[0]
            if hasattr(mesh_obj, 'vertices') and hasattr(mesh_obj, 'faces'):
                vertices = mesh_obj.vertices
                faces = mesh_obj.faces
                if hasattr(vertices, 'cpu'):
                    vertices = vertices.cpu().numpy().astype(np.float32)
                else:
                    vertices = np.array(vertices, dtype=np.float32)
                if hasattr(faces, 'cpu'):
                    faces = faces.cpu().numpy().astype(np.int32)
                else:
                    faces = np.array(faces, dtype=np.int32)
            else:
                logging.error(f"Mesh object doesn't have vertices/faces. Available: {dir(mesh_obj)}")
                return
        else:
            logging.error(f"Unexpected mesh format: {type(mesh_data)}")
            return
        
        # Set renderer faces
        renderer.faces = faces
        
        # Extract transformation parameters
        translation = np.array(sam3do_out["translation"]).flatten().astype(np.float32)
        scale_val = float(np.array(sam3do_out["scale"]).flatten()[0])
        rotation_quat = np.array(sam3do_out["rotation"]).flatten().astype(np.float32)  # [w, x, y, z]
        
        # Check if intrinsics are available and use them
        camera_center = None
        if "intrinsics" in sam3do_out:
            intrinsics = np.array(sam3do_out["intrinsics"])
            logging.info(f"Intrinsics shape: {intrinsics.shape}, value:\n{intrinsics}")
            if intrinsics.ndim == 2 and intrinsics.shape[0] >= 2 and intrinsics.shape[1] >= 3:
                fx = float(intrinsics[0, 0])
                fy = float(intrinsics[1, 1])
                cx = float(intrinsics[0, 2])
                cy = float(intrinsics[1, 2])
                
                # Check if intrinsics are normalized (values < 10 are likely normalized)
                if fx < 10 and fy < 10:
                    # Denormalize: multiply by image dimensions
                    h, w = frame.shape[:2]
                    fx = fx * w
                    fy = fy * h
                    cx = cx * w
                    cy = cy * h
                    logging.info(f"Intrinsics were normalized, denormalized to pixels")
                
                renderer.focal_length = (fx + fy) / 2.0
                camera_center = [cx, cy]
                logging.info(f"Using intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        else:
            logging.warning("No intrinsics in output!")
        
        logging.info(f"Raw transform - translation: {translation}, scale: {scale_val}, rotation: {rotation_quat}")
        logging.info(f"Mesh: vertices {vertices.shape}, faces {faces.shape}, min: {vertices.min(axis=0)}, max: {vertices.max(axis=0)}")
        logging.info(f"Renderer focal_length: {renderer.focal_length}, camera_center: {camera_center}")
        
        # Convert quaternion to rotation matrix
        from scipy.spatial.transform import Rotation as R
        # SAM3D uses [w, x, y, z], scipy uses [x, y, z, w]
        quat_scipy = np.array([rotation_quat[1], rotation_quat[2], rotation_quat[3], rotation_quat[0]])
        rot_matrix = R.from_quat(quat_scipy).as_matrix()
        
        # Define coordinate system flip: PyTorch3D (X-right, Y-up) -> OpenCV (X-right, Y-down)
        # We need to flip X and Y axes
        coord_flip = np.array([
            [-1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0,  1]
        ], dtype=np.float32)
        
        # Transform rotation to new coordinate system: R_opencv = Flip @ R_pytorch3d @ Flip^T
        rot_matrix_opencv = coord_flip @ rot_matrix @ coord_flip.T
        
        # Apply transformations to vertices: scale, rotate (in OpenCV frame), then translate
        vertices_transformed = vertices * scale_val
        vertices_transformed = vertices_transformed @ rot_matrix_opencv.T  # Apply rotation in OpenCV frame
        
        # Flip vertices from PyTorch3D to OpenCV
        vertices_transformed[:, 0] *= -1  # Flip X
        vertices_transformed[:, 1] *= -1  # Flip Y
        
        # Flip translation from PyTorch3D to OpenCV
        translation_opencv = translation.copy()
        translation_opencv[0] *= -1  # Flip X
        translation_opencv[1] *= -1  # Flip Y
        
        vertices_transformed = vertices_transformed + translation_opencv  # Apply translation
        
        # The renderer expects camera_translation (cam_t), not mesh position
        # Set cam_t to zero since we've already transformed vertices
        cam_t = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        logging.info(f"Transformed vertices - min: {vertices_transformed.min(axis=0)}, max: {vertices_transformed.max(axis=0)}")
        
        # Render mesh on original frame
        rendered_img = renderer(
            vertices=vertices_transformed,
            cam_t=cam_t,
            image=frame,
            mesh_base_color=(0.2, 0.5, 1.0),  # Blue
            scene_bg_color=(0, 0, 0),
            return_rgba=False,
            camera_center=camera_center,  # Use intrinsics principal point
        )
        
        # Also render standalone RGBA view for debugging
        rendered_rgba = renderer.render_rgba(
            vertices=vertices_transformed,
            cam_t=np.array([0.0, 0.0, 3.0]),  # Default camera distance for standalone view
            rot_angle=0,
            mesh_base_color=(0.2, 0.5, 1.0),
            render_res=[640, 480],
        )
        
        # Add visualization overlays
        rendered_img_vis = (rendered_img * 255).astype(np.uint8).copy()
        
        # Draw coordinate axes at mesh origin
        origin_3d = translation_opencv
        axes_length = scale_val * 0.5  # Half the scale for visibility
        axes_3d = {
            'X': np.array([[0,0,0], [axes_length, 0, 0]]),  # Red
            'Y': np.array([[0,0,0], [0, axes_length, 0]]),  # Green  
            'Z': np.array([[0,0,0], [0, 0, axes_length]]),  # Blue
        }
        axes_colors = {'X': (0, 0, 255), 'Y': (0, 255, 0), 'Z': (255, 0, 0)}  # BGR format
        
        # Project axes to 2D
        h, w = frame.shape[:2]
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        for axis_name, axis_pts in axes_3d.items():
            # Transform axis points
            pts_transformed = axis_pts @ rot_matrix_opencv.T
            pts_transformed[:, 0] *= -1  # Flip X
            pts_transformed[:, 1] *= -1  # Flip Y
            pts_transformed += translation_opencv
            
            # Project to 2D
            pts_2d = []
            for pt in pts_transformed:
                if pt[2] > 0:  # In front of camera
                    uv = K @ pt
                    uv = uv[:2] / uv[2]
                    pts_2d.append((int(uv[0]), int(uv[1])))
            
            if len(pts_2d) == 2:
                cv2.line(rendered_img_vis, pts_2d[0], pts_2d[1], axes_colors[axis_name], 3)
                cv2.circle(rendered_img_vis, pts_2d[1], 5, axes_colors[axis_name], -1)
        
        # Calculate and overlay mesh dimensions
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        mesh_dims = bbox_max - bbox_min
        mesh_dims_scaled = mesh_dims * scale_val  # In meters
        
        # Add text overlay with dimensions
        text_lines = [
            f"Dims: {mesh_dims_scaled[0]:.3f}m x {mesh_dims_scaled[1]:.3f}m x {mesh_dims_scaled[2]:.3f}m",
            f"Scale: {scale_val:.3f}",
            f"Translation: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]",
            f"Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"
        ]
        
        y_offset = 30
        for i, line in enumerate(text_lines):
            cv2.putText(rendered_img_vis, line, (10, y_offset + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Save results
        results_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(results_dir / "rendered_on_frame.png"), rendered_img_vis)
        cv2.imwrite(str(results_dir / "rendered_rgba.png"), (rendered_rgba * 255).astype(np.uint8))
        
        logging.info(f"Saved rendered results to {results_dir}")
        
        if cfg.show:
            cv2.imshow("SAM3Do Rendered", (rendered_img * 255).astype(np.uint8))
            cv2.imshow("SAM3Do RGBA", (rendered_rgba * 255).astype(np.uint8))
    
    except Exception as e:
        logging.error(f"Rendering failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()


RENDER_PIPELINE = Sam3DoRenderPipeline()


def render_sam3do_output(renderer, frame, sam3do_out, result_dir, frame_idx, cfg):
    RENDER_PIPELINE.render(
        renderer=renderer,
        frame=frame,
        sam3do_out=sam3do_out,
        result_dir=result_dir,
        show=cfg.show,
    )

def main(cfg: SAM3DoConfig) -> None:
    """Create clients with extended timeout and run inference loop"""
    import cv2
    from datetime import datetime
    
    sam3do_client = Client(cfg.host, cfg.port)
    sam3_client = Client(cfg.sam3.host, cfg.sam3.port)

    # Initialize renderer ONCE before the loop
    # Use a more appropriate focal length based on image size
    # Typical webcam has FOV ~60-70 degrees, for 640px width: f ≈ width / (2 * tan(FOV/2))
    # For FOV=60°: f ≈ 640 / (2 * tan(30°)) ≈ 554
    focal_length = 515
    renderer = Renderer(focal_length=focal_length)
    logging.info(f"Initialized renderer with focal_length={focal_length}")

    cap: Optional[cv2.VideoCapture] = None
    if isinstance(cfg.input_source, CameraInput):
        cap = cv2.VideoCapture(cfg.input_source.device)

    frame_idx = 0
    logging.info("Starting SAM3Do orchestration")
    
    while True:
        # Get frame from input source
        frame = get_frame(cfg, cap)
        
        # Send to SAM3 first
        sam3_payload = {
            "image": frame,
            "type": "image",
            "text": cfg.sam3.prompt,
        }
        logging.info(f"Sending to SAM3...")
        sam3_out = sam3_client.step(sam3_payload)
        
#        if not sam3_out:
#            logging.error("No output received from SAM3 server")
#            continue
        
        logging.info(f"Received output from SAM3: {sam3_out.keys()}")
        
        mask = sam3_out['masks']
        mask = mask.sum(axis=(0, 1)).astype(np.bool_)  # Fixed: np.bool_ instead of np.bool
        
        # Send to SAM3Do with timeout parameter
        sam3do_payload = {
            "image": frame,
            "mask": mask,
            "type": "image",
            "timeout": cfg.timeout,
        }
        logging.info(f"Sending to SAM3Do with server timeout={cfg.timeout}s...")
        sam3do_out = sam3do_client.step(sam3do_payload)
        
        if not sam3do_out:
            logging.error("No output received from SAM3Do server")
            continue
        
        logging.info(f"Received output from SAM3Do: {sam3do_out.keys()}")
        if "error" in sam3do_out:
            logging.error(f"SAM3Do server returned error: {sam3do_out['error']}")
            continue

        # Check if mesh is in output and what it contains
        mesh_data = sam3do_out.get("mesh")
        if mesh_data is None:
            logging.error(
                "No mesh key in SAM3Do output; rendered_on_frame.png will not be produced. "
                "SAM3Do keys: %s",
                sorted(sam3do_out.keys()),
            )
        else:
            logging.info(f"Mesh type: {type(mesh_data)}")
            if isinstance(mesh_data, list):
                logging.info(f"Mesh list length: {len(mesh_data)}")
                if len(mesh_data) > 0:
                    logging.info(f"First mesh item type: {type(mesh_data[0])}")
    
        # Save result
        saver = ResultSaver(base_dir=Path("sam3do_results"))
        result_dir = saver.save_results(frame, sam3do_out, cfg, frame_idx=frame_idx, mask=mask)
        if mesh_data is None:
            (result_dir / "render_error.txt").write_text(
                "Missing 'mesh' in SAM3Do output. Restart server_sam3do to pick up latest code "
                "and verify you are connected to the SAM3Do server port (default 8003), not SAM3."
            )
        
        # Log the results
        for key, value in sam3do_out.items():
            if isinstance(value, dict) and 'shape' in value:
                logging.info(f"  {key}: shape={value['shape']}, dtype={value['dtype']}")
            elif isinstance(value, np.ndarray):
                logging.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, (int, float, str)):
                logging.info(f"  {key}: {value}")
            elif isinstance(value, list):
                logging.info(f"  {key}: list of {len(value)} items")
            else:
                logging.info(f"  {key}: {type(value)}")

        # Render the output - FIX: pass renderer instance, not the class
        render_sam3do_output(renderer, frame, sam3do_out, result_dir, frame_idx, cfg)
        
        # Display if needed
        if cfg.show:
            cv2.imshow("Input Frame", frame)
            if isinstance(cfg.input_source, ImageInput):
                cv2.waitKey(0)
                break
            elif cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        frame_idx += 1

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(SAM3DoConfig))
