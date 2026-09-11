#!/usr/bin/env python
"""
Safe Local Hybrid Point Cloud Downsampling Script
---------------------------------------------------
Reads from the ORIGINAL 50,000-point PLY files in pointclouds/ and writes
the hybrid-downsampled 2,048-point files into a SEPARATE directory
(pointclouds_hybrid/) preserving the subdirectory structure.

** DOES NOT OVERWRITE ORIGINAL FILES **

Usage:
    python scripts/downsample_local_safe.py [--input-dir pointclouds] [--output-dir pointclouds_hybrid] [--workers 4]
"""

import os
import sys
import time
import argparse
import concurrent.futures
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.sampling import hybrid_fps_curvature_sampling


def parse_args():
    parser = argparse.ArgumentParser(description="Safe local hybrid point cloud downsampling.")
    parser.add_argument("--input-dir", type=str, default="pointclouds",
                        help="Source directory with original 50k-point PLY files")
    parser.add_argument("--output-dir", type=str, default="pointclouds_hybrid",
                        help="Output directory for downsampled 2048-point PLY files")
    parser.add_argument("--num-points", type=int, default=2048,
                        help="Target number of points (default: 2048)")
    parser.add_argument("--fps-ratio", type=float, default=0.75,
                        help="Ratio of FPS points (default: 0.75)")
    parser.add_argument("--knn-k", type=int, default=20,
                        help="k-NN neighbors for curvature computation (default: 20)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    return parser.parse_args()


def process_single_ply(args_tuple):
    """
    Loads a single 50k PLY file, downsamples via hybrid FPS + curvature sampling,
    and writes the result to the output path (separate from input).
    """
    input_path, output_path, target_points, fps_ratio, knn_k = args_tuple
    try:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(input_path))
        if not pcd.has_points():
            return False, str(input_path), "No points found"

        points = np.asarray(pcd.points, dtype=np.float32)

        # If already at target points, just copy directly
        if len(points) == target_points:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
            return True, str(input_path), "already_correct_size"

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

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        o3d.io.write_point_cloud(str(output_path), pcd_down, write_ascii=False)
        return True, str(input_path), "ok"
    except Exception as e:
        return False, str(input_path), str(e)


def main():
    args = parse_args()
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    print("=" * 60)
    print("  Safe Local Hybrid Point Cloud Downsampler")
    print("  (75% FPS + 25% Curvature -> Separate Output Directory)")
    print("=" * 60)
    print(f"Input Directory  : {input_root.resolve()}")
    print(f"Output Directory : {output_root.resolve()}")
    print(f"Target Points    : {args.num_points}")
    print(f"FPS Ratio        : {args.fps_ratio}")
    print(f"k-NN Neighbors   : {args.knn_k}")
    print(f"Workers          : {args.workers}")
    print("-" * 60)

    # Discover all PLY files
    ply_files = sorted(list(input_root.glob("*/*.ply")))
    if not ply_files:
        ply_files = sorted(list(input_root.glob("*.ply")))

    total_files = len(ply_files)
    print(f"Found {total_files} point cloud PLY files to process.")
    if total_files == 0:
        print("No files to process. Exiting.")
        return

    # Build input->output path pairs, preserving subdirectory structure
    task_args = []
    for ply_path in ply_files:
        rel_path = ply_path.relative_to(input_root)
        out_path = output_root / rel_path
        task_args.append((str(ply_path), str(out_path), args.num_points, args.fps_ratio, args.knn_k))

    start_time = time.time()
    successful = 0
    failed = 0
    errors = []

    if args.workers <= 1:
        for idx, ta in enumerate(task_args, 1):
            ok, fpath, msg = process_single_ply(ta)
            if ok:
                successful += 1
            else:
                failed += 1
                errors.append((fpath, msg))
            if idx % 50 == 0 or idx == total_files:
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                print(f"[{idx}/{total_files}] OK={successful} FAIL={failed} ({rate:.1f} files/sec, {elapsed:.1f}s)", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_single_ply, ta) for ta in task_args]

            for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    ok, fpath, msg = future.result()
                    if ok:
                        successful += 1
                    else:
                        failed += 1
                        errors.append((fpath, msg))
                except Exception as e:
                    failed += 1
                    errors.append(("unknown", str(e)))

                if idx % 100 == 0 or idx == total_files:
                    elapsed = time.time() - start_time
                    rate = idx / elapsed if elapsed > 0 else 0
                    print(f"[{idx}/{total_files}] OK={successful} FAIL={failed} ({rate:.1f} files/sec, {elapsed:.1f}s)", flush=True)

    total_elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Completed in {total_elapsed:.1f}s ({total_elapsed/60:.2f} min).")
    print(f"Successfully downsampled: {successful}/{total_files}")
    print(f"Failed: {failed}/{total_files}")
    if errors:
        print("Errors:")
        for fpath, msg in errors[:10]:
            print(f"  {fpath}: {msg}")
    print("=" * 60)


if __name__ == "__main__":
    main()
