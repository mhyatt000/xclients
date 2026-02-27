#!/usr/bin/env python3
"""
Quick test script to try different coordinate transformations for SAM3D mesh alignment.
Run this to generate multiple renderings with different transformation options.
"""
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import sys

def apply_transform_option(vertices, rotation_quat, translation, scale_val, option):
    """
    Apply different transformation options:
    0: No flip (PyTorch3D as-is)
    1: Flip X,Y of vertices and translation (current)
    2: Flip only translation
    3: Flip only vertices
    4: Flip X,Y,Z
    5: Negate rotation then flip X,Y
    6: Invert rotation then flip X,Y
    """
    # Convert quaternion to rotation matrix
    quat_scipy = np.array([rotation_quat[1], rotation_quat[2], rotation_quat[3], rotation_quat[0]])
    rot_matrix = R.from_quat(quat_scipy).as_matrix()
    
    vertices_scaled = vertices * scale_val
    
    if option == 0:
        # No flip
        vertices_out = vertices_scaled @ rot_matrix.T
        translation_out = translation.copy()
    
    elif option == 1:
        # Current implementation: Flip X,Y of both
        vertices_rotated = vertices_scaled @ rot_matrix.T
        vertices_out = vertices_rotated.copy()
        vertices_out[:, 0] *= -1
        vertices_out[:, 1] *= -1
        translation_out = translation.copy()
        translation_out[0] *= -1
        translation_out[1] *= -1
    
    elif option == 2:
        # Flip only translation
        vertices_out = vertices_scaled @ rot_matrix.T
        translation_out = translation.copy()
        translation_out[0] *= -1
        translation_out[1] *= -1
    
    elif option == 3:
        # Flip only vertices
        vertices_rotated = vertices_scaled @ rot_matrix.T
        vertices_out = vertices_rotated.copy()
        vertices_out[:, 0] *= -1
        vertices_out[:, 1] *= -1
        translation_out = translation.copy()
    
    elif option == 4:
        # Flip all three axes
        vertices_rotated = vertices_scaled @ rot_matrix.T
        vertices_out = -vertices_rotated
        translation_out = -translation
    
    elif option == 5:
        # Negate quaternion (flip rotation direction) then flip X,Y
        quat_scipy_neg = -quat_scipy
        rot_matrix_neg = R.from_quat(quat_scipy_neg).as_matrix()
        vertices_rotated = vertices_scaled @ rot_matrix_neg.T
        vertices_out = vertices_rotated.copy()
        vertices_out[:, 0] *= -1
        vertices_out[:, 1] *= -1
        translation_out = translation.copy()
        translation_out[0] *= -1
        translation_out[1] *= -1
    
    elif option == 6:
        # Invert rotation then flip X,Y
        rot_matrix_inv = rot_matrix.T
        vertices_rotated = vertices_scaled @ rot_matrix_inv.T
        vertices_out = vertices_rotated.copy()
        vertices_out[:, 0] *= -1
        vertices_out[:, 1] *= -1
        translation_out = translation.copy()
        translation_out[0] *= -1
        translation_out[1] *= -1
    
    return vertices_out, translation_out

# Load the most recent result
results_dir = Path("sam3do_results")
if not results_dir.exists():
    print("No sam3do_results directory found!")
    sys.exit(1)

# Find most recent result directory
result_dirs = sorted(results_dir.glob("result_*"))
if not result_dirs:
    print("No results found!")
    sys.exit(1)

latest_dir = result_dirs[-1]
print(f"Using results from: {latest_dir}")

# This script is just a template - you'll need to integrate it with your actual rendering code
print("\nTo test different transformations, modify render_sam3do_output() in sam3do.py")
print("and change the transformation option number (0-6) in the code.")
print("\nOptions:")
print("  0: No flip (PyTorch3D as-is)")
print("  1: Flip X,Y of vertices and translation (current)")
print("  2: Flip only translation")
print("  3: Flip only vertices")
print("  4: Flip all axes")
print("  5: Negate quaternion then flip X,Y")
print("  6: Invert rotation then flip X,Y")
