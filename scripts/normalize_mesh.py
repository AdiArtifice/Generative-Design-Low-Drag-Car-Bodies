#!/usr/bin/env python
"""
Mesh Normalization Script (Open3D-Accelerated)
----------------------------------------------
This script loads the raw STL meshes using Open3D, translates them so their bounding box
center is at the origin (0, 0, 0), scales them uniformly so their largest
dimension equals 1.0 (unit bounding box scaling) using NumPy, and exports the resulting
normalized STL files.

Usage:
    python scripts/normalize_mesh.py [--input INPUT_DIR] [--output OUTPUT_DIR] [--center-method {bounds,centroid}]
"""

import os
import sys
import gc
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from dotenv import load_dotenv

# Load configuration from .env file
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Center and scale STL meshes using Open3D.")
    
    # Default paths from environment variables, falling back to sensible project defaults
    default_input = os.getenv("RAW_STL_DIR", "raw_stl/fastback smooth wheel with covers")
    default_output = os.getenv("NORMALIZED_STL_DIR", "normalized/fastback_smooth_wheelcovers")
    
    parser.add_argument(
        "--input", 
        type=str, 
        default=default_input,
        help=f"Directory containing raw STL files (default: {default_input})"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=default_output,
        help=f"Directory to save normalized STL files (default: {default_output})"
    )
    parser.add_argument(
        "--center-method",
        type=str,
        choices=["bounds", "centroid"],
        default="bounds",
        help="Method to center the mesh: 'bounds' (bounding box center) or 'centroid' (center of mass) (default: bounds)"
    )
    
    return parser.parse_args()

def normalize_mesh(file_path: Path, output_dir: Path, center_method: str) -> bool:
    """
    Centers and scales a single mesh using Open3D and NumPy, then exports it.
    
    Parameters:
        file_path (Path): Path to the input STL file.
        output_dir (Path): Directory where the normalized STL will be saved.
        center_method (str): "bounds" or "centroid" for centering logic.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Load mesh using Open3D
        mesh = o3d.io.read_triangle_mesh(str(file_path))
        
        if not mesh.has_triangles():
            raise ValueError(f"File loaded is not a valid triangle mesh: {file_path.name}")
            
        # Get vertices as numpy array
        vertices = np.asarray(mesh.vertices)
        if len(vertices) == 0:
            raise ValueError("Mesh has 0 vertices.")
            
        # 1. Centering
        if center_method == "bounds":
            # Center of the bounding box
            min_box = vertices.min(axis=0)
            max_box = vertices.max(axis=0)
            center = (min_box + max_box) / 2.0
        else:
            # Centroid (average of vertices)
            center = vertices.mean(axis=0)
            
        vertices = vertices - center
        
        # 2. Scaling (uniform scaling to fit largest dimension to 1.0)
        min_box = vertices.min(axis=0)
        max_box = vertices.max(axis=0)
        extents = max_box - min_box
        max_extent = max(extents)
        if max_extent == 0:
            raise ValueError("Mesh extents are invalid or zero.")
            
        scale_factor = 1.0 / max_extent
        vertices = vertices * scale_factor
        
        # Update mesh vertices
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        
        # Recompute normals before exporting (critical for STL format in Open3D)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        
        # 3. Export
        output_filename = f"{file_path.stem}_norm.stl"
        output_path = output_dir / output_filename
        
        # Export mesh (binary STL format is more compact and loads much faster)
        o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=False)
        
        # Explicit cleanup to keep memory footprint tiny
        del mesh
        return True
        
    except Exception as e:
        print(f"  [Error] Failed to normalize {file_path.name}: {str(e)}", flush=True)
        return False

def main():
    args = parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    print("=" * 60)
    print("        Mesh Preprocessing Pipeline: Normalization (Open3D)")
    print("=" * 60)
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Centering method: {args.center_method}")
    print("-" * 60)
    
    if not input_dir.exists():
        print(f"[Error] Input directory '{input_dir}' does not exist.")
        sys.exit(1)
        
    # Find all STL files
    stl_files = sorted(list(input_dir.glob("*.stl")))
    total_files = len(stl_files)
    
    if total_files == 0:
        print(f"[Warning] No STL files found in {input_dir}")
        sys.exit(0)
        
    print(f"Found {total_files} STL files to normalize. Starting normalization...")
    
    # Ensure output folder exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful_count = 0
    for idx, file_path in enumerate(stl_files, start=1):
        print(f"[{idx}/{total_files}] Normalizing {file_path.name}...", end="", flush=True)
        success = normalize_mesh(file_path, output_dir, args.center_method)
        if success:
            successful_count += 1
            print(" Done.", flush=True)
        # Clean up memory immediately
        gc.collect()
        
    print("-" * 60)
    print("Normalization Completed!")
    print(f"Successfully normalized: {successful_count}/{total_files} meshes")
    print(f"Output saved to         : {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
