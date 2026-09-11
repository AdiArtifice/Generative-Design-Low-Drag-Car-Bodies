#!/usr/bin/env python
"""
Offline Hybrid Point Cloud Downsampling Script
-----------------------------------------------
Pre-computes the 75% FPS + 25% Curvature Hybrid Sampling offline across all
point cloud PLY files in the dataset, converting 50,000-point clouds to exact
2,048-point hybrid representation.

Runs in parallel across multiple CPU workers for maximum throughput.

Usage:
    python scripts/preprocess_pointclouds_hybrid.py [--pc-dir pointclouds] [--num-points 2048] [--workers 8]
"""

import os
import sys
import gc
import time
import argparse
import concurrent.futures
from pathlib import Path
import numpy as np
import open3d as o3d
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.sampling import hybrid_fps_curvature_sampling

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel offline hybrid point cloud downsampling.")
    parser.add_argument("--pc-dir", type=str, default="pointclouds", help="Root directory containing point cloud PLY files")
    parser.add_argument("--num-points", type=int, default=2048, help="Target number of points (default: 2048)")
    parser.add_argument("--fps-ratio", type=float, default=0.75, help="Ratio of FPS points (default: 0.75)")
    parser.add_argument("--knn-k", type=int, default=20, help="k-NN neighbors for curvature computation (default: 20)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes (default: 8)")
    return parser.parse_args()

def process_single_ply(ply_path: Path, target_points: int = 2048, fps_ratio: float = 0.75, knn_k: int = 20) -> bool:
    """
    Loads a single 50k PLY file, downsamples via hybrid FPS + curvature sampling to target_points,
    and overwrites the file in-place with the 2048-point cloud.
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if not pcd.has_points():
            return False
            
        points = np.asarray(pcd.points, dtype=np.float32)
        
        # If already downsampled to target points, skip
        if len(points) == target_points:
            return True
            
        if pcd.has_normals():
            normals = np.asarray(pcd.normals, dtype=np.float32)
        else:
            pcd.estimate_normals()
            normals = np.asarray(pcd.normals, dtype=np.float32)
            
        features = hybrid_fps_curvature_sampling(
            points=points,
            normals=normals,
            total_points=target_points,
            fps_ratio=fps_ratio,
            k_neighbors=knn_k
        )
        
        pcd_down = o3d.geometry.PointCloud()
        pcd_down.points = o3d.utility.Vector3dVector(features[:, :3])
        pcd_down.normals = o3d.utility.Vector3dVector(features[:, 3:])
        
        o3d.io.write_point_cloud(str(ply_path), pcd_down, write_ascii=False)
        return True
    except Exception as e:
        print(f"Error processing {ply_path.name}: {e}", flush=True)
        return False

def main():
    args = parse_args()
    pc_root = Path(args.pc_dir)
    
    print("=" * 60)
    print("      Offline Hybrid Point Cloud Downsampler (75% FPS + 25% Curvature)")
    print("=" * 60)
    print(f"Point Cloud Directory: {pc_root}")
    print(f"Target Points        : {args.num_points}")
    print(f"FPS Ratio            : {args.fps_ratio}")
    print(f"k-NN Neighbors       : {args.knn_k}")
    print(f"Workers              : {args.workers}")
    print("-" * 60)
    
    ply_files = sorted(list(pc_root.glob("*/*.ply")))
    if not ply_files:
        ply_files = sorted(list(pc_root.glob("*.ply")))
        
    total_files = len(ply_files)
    print(f"Found {total_files} point cloud PLY files to process.")
    if total_files == 0:
        print("No files to process. Exiting.")
        return
        
    start_time = time.time()
    successful = 0
    
    if args.workers <= 1:
        for idx, f in enumerate(ply_files, 1):
            ok = process_single_ply(f, args.num_points, args.fps_ratio, args.knn_k)
            if ok:
                successful += 1
            if idx % 50 == 0 or idx == total_files:
                elapsed = time.time() - start_time
                print(f"[{idx}/{total_files}] Downsampled {successful} files ({elapsed:.1f}s)...", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_file = {
                executor.submit(process_single_ply, f, args.num_points, args.fps_ratio, args.knn_k): f 
                for f in ply_files
            }
            
            for idx, future in enumerate(concurrent.futures.as_completed(future_to_file), 1):
                try:
                    if future.result():
                        successful += 1
                except Exception as e:
                    print(f"Worker error on {future_to_file[future].name}: {e}", flush=True)
                    
                if idx % 100 == 0 or idx == total_files:
                    elapsed = time.time() - start_time
                    rate = idx / elapsed if elapsed > 0 else 0
                    print(f"[{idx}/{total_files}] Downsampled {successful} files ({rate:.1f} files/sec, {elapsed:.1f}s elapsed)...", flush=True)
                    
    total_elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Completed in {total_elapsed:.1f}s ({total_elapsed/60:.2f} min).")
    print(f"Successfully downsampled: {successful}/{total_files} point clouds to {args.num_points} points.")
    print("=" * 60)

if __name__ == "__main__":
    main()
