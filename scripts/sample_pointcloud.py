#!/usr/bin/env python
"""
Point Cloud Sampling Script
---------------------------
This script loads the normalized STL meshes, uniformly samples a dense point cloud 
from the triangle surfaces, downsamples it to exactly N points using Farthest Point 
Sampling (FPS) for optimal spatial coverage, and exports the resulting point clouds
as binary PLY files (retaining coordinates and normal vectors).

Usage:
    python scripts/sample_pointcloud.py [--input INPUT_DIR] [--output OUTPUT_DIR] [--num-points NUM_POINTS]
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.sampling import hybrid_fps_curvature_sampling

def parse_args():
    parser = argparse.ArgumentParser(description="Convert STL meshes to point clouds via FPS or Hybrid Curvature Sampling.")
    
    # Default paths and parameters from environment variables, falling back to sensible defaults
    default_input = os.getenv("NORMALIZED_STL_DIR", "normalized/fastback_smooth_wheelcovers")
    default_output = os.getenv("POINTCLOUD_DIR", "pointclouds/fastback_smooth_wheelcovers")
    default_num_points = int(os.getenv("NUM_POINTS_PC", 50000))
    
    parser.add_argument(
        "--input", 
        type=str, 
        default=default_input,
        help=f"Directory containing normalized STL files (default: {default_input})"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=default_output,
        help=f"Directory to save PLY point cloud files (default: {default_output})"
    )
    parser.add_argument(
        "--num-points", 
        type=int, 
        default=default_num_points,
        help=f"Exact number of points to sample (default: {default_num_points})"
    )
    parser.add_argument(
        "--use-fps",
        action="store_true",
        help="Use Farthest Point Sampling (FPS) downsampling for optimal spatial coverage"
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        choices=["uniform", "fps", "hybrid"],
        default="hybrid",
        help="Sampling strategy: 'uniform', 'fps', or 'hybrid' (75% FPS + 25% Curvature) (default: 'hybrid')"
    )
    parser.add_argument(
        "--fps-ratio",
        type=float,
        default=0.75,
        help="Ratio of FPS points when using hybrid strategy (default: 0.75)"
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=20,
        help="k-NN parameter for local surface curvature calculation (default: 20)"
    )
    
    return parser.parse_args()

def sample_mesh_to_pc(file_path: Path, output_dir: Path, num_points: int, 
                       sampling_strategy: str = "hybrid", fps_ratio: float = 0.75, knn_k: int = 20) -> bool:
    """
    Loads a single mesh, performs point cloud sampling using the specified strategy,
    and writes the output to PLY format.
    
    Parameters:
        file_path (Path): Path to the input normalized STL file.
        output_dir (Path): Directory to save the PLY point cloud.
        num_points (int): Number of target points.
        sampling_strategy (str): 'uniform', 'fps', or 'hybrid'.
        fps_ratio (float): Ratio of FPS points in hybrid mode.
        knn_k (int): k-NN parameter for curvature estimation.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # 1. Load mesh using Open3D
        mesh = o3d.io.read_triangle_mesh(str(file_path))
        
        if not mesh.has_triangles():
            raise ValueError("Mesh has no triangles or failed to load correctly.")
            
        # Recompute vertex and face normals to ensure clean outward-oriented normals
        mesh.compute_vertex_normals()
        
        # 2. Sampling
        if sampling_strategy == "hybrid":
            # Sample dense points first (e.g. 50k points or 10x target points)
            dense_count = max(50000, num_points * 10)
            dense_pcd = mesh.sample_points_uniformly(number_of_points=dense_count)
            if not dense_pcd.has_points():
                raise ValueError("Failed to uniformly sample dense points from mesh surface.")
            
            pts = np.asarray(dense_pcd.points, dtype=np.float32)
            nms = np.asarray(dense_pcd.normals, dtype=np.float32)
            
            features = hybrid_fps_curvature_sampling(
                points=pts,
                normals=nms,
                total_points=num_points,
                fps_ratio=fps_ratio,
                k_neighbors=knn_k
            )
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(features[:, :3])
            pcd.normals = o3d.utility.Vector3dVector(features[:, 3:])
            del dense_pcd
            
        elif sampling_strategy == "fps":
            dense_pcd = mesh.sample_points_uniformly(number_of_points=max(num_points * 2, 10000))
            if not dense_pcd.has_points():
                raise ValueError("Failed to uniformly sample dense points from mesh surface.")
            pcd = dense_pcd.farthest_point_down_sample(num_points)
            del dense_pcd
        else:
            # Direct uniform sampling
            pcd = mesh.sample_points_uniformly(number_of_points=num_points)
            
        if not pcd.has_points() or len(pcd.points) != num_points:
            raise ValueError(f"Sampling failed to yield exactly {num_points} points (got {len(pcd.points)}).")
            
        # 3. Export to PLY
        output_filename = f"{file_path.stem.replace('_norm', '')}_pc.ply"
        output_path = output_dir / output_filename
        
        # Save as binary PLY (includes coordinates x,y,z and normals nx,ny,nz)
        o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
        
        # Explicitly clear objects to prevent RAM leak
        del mesh
        del pcd
        return True
        
    except Exception as e:
        print(f"  [Error] Failed to sample {file_path.name}: {str(e)}", flush=True)
        return False

def main():
    args = parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    num_points = args.num_points
    
    # Resolve strategy from --use-fps flag if set explicitly
    strategy = args.sampling_strategy
    if args.use_fps and strategy == "hybrid":
        strategy = "fps"
    
    print("=" * 60)
    print("                Mesh Preprocessing Pipeline: PC Sampling")
    print("=" * 60)
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target points   : {num_points}")
    print(f"Strategy        : {strategy} (FPS ratio: {args.fps_ratio}, k-NN: {args.knn_k})")
    print("-" * 60)
    
    if not input_dir.exists():
        print(f"[Error] Input directory '{input_dir}' does not exist.")
        sys.exit(1)
        
    stl_files = sorted(list(input_dir.glob("*.stl")))
    total_files = len(stl_files)
    
    if total_files == 0:
        print(f"[Warning] No STL files found in {input_dir}")
        sys.exit(0)
        
    print(f"Found {total_files} meshes to sample. Starting sampling...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    o3d.utility.random.seed(42)
    
    successful_count = 0
    for idx, file_path in enumerate(stl_files, start=1):
        print(f"[{idx}/{total_files}] Sampling {file_path.name}...", end="", flush=True)
        success = sample_mesh_to_pc(
            file_path=file_path, 
            output_dir=output_dir, 
            num_points=num_points, 
            sampling_strategy=strategy,
            fps_ratio=args.fps_ratio,
            knn_k=args.knn_k
        )
        if success:
            successful_count += 1
            print(" Done.", flush=True)
        gc.collect()
        
    print("-" * 60)
    print("Point Cloud Sampling Completed!")
    print(f"Successfully sampled: {successful_count}/{total_files} meshes")
    print(f"Output saved to      : {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
