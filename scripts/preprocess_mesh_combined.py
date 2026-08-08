#!/usr/bin/env python
"""
Combined Mesh Preprocessing Script (Open3D-Accelerated & Parallelized)
----------------------------------------------------------------------
Processes a directory of raw STL files by running normalization, point cloud
sampling, and occupancy generation in a single pass per file, minimizing file I/O overhead.
Runs in parallel across all CPU cores for maximum throughput.

Usage:
    python scripts/preprocess_mesh_combined.py --raw-dir raw_stl/F_S_WWS_WM --norm-dir normalized/F_S_WWS_WM --pc-dir pointclouds/F_S_WWS_WM --occ-dir occupancy/F_S_WWS_WM --workers 4
"""

import os
import sys
import gc
import time
import argparse
import concurrent.futures
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

# Load configuration from .env file
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel combined mesh preprocessing pipeline.")
    parser.add_argument("--raw-dir", type=str, required=True, help="Directory containing raw STL files")
    parser.add_argument("--norm-dir", type=str, required=True, help="Directory to save normalized STL files")
    parser.add_argument("--pc-dir", type=str, required=True, help="Directory to save PLY point cloud files")
    parser.add_argument("--occ-dir", type=str, required=True, help="Directory to save occupancy NPZ files")
    parser.add_argument("--num-points-pc", type=int, default=50000, help="Number of point cloud points to sample")
    parser.add_argument("--num-points-occ", type=int, default=2048, help="Number of occupancy query points to sample")
    parser.add_argument("--sigma-occ", type=float, default=0.015, help="Surface perturbation standard deviation for occupancy")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes/threads")
    parser.add_argument("--save-norm", action="store_true", help="Save the normalized STL file to disk")
    return parser.parse_args()

def process_single_file(file_path: Path, norm_dir: Path, pc_dir: Path, occ_dir: Path, 
                        num_points_pc: int, num_points_occ: int, sigma_occ: float, save_norm: bool = False) -> bool:
    """
    Main worker function that preprocesses a single raw mesh file through the entire pipeline:
    1. Load Raw Mesh -> 2. Normalize Vertices -> 3. Export STL (optional) -> 4. Sample PC -> 5. Sample Occupancy
    """
    try:
        # Import Open3D locally to prevent multiprocessing deadlock on Windows
        import open3d as o3d
        
        # Step 1: Load mesh
        mesh = o3d.io.read_triangle_mesh(str(file_path))
        if not mesh.has_triangles():
            raise ValueError(f"Mesh has no triangles or failed to load: {file_path.name}")
            
        vertices = np.asarray(mesh.vertices)
        if len(vertices) == 0:
            raise ValueError(f"Mesh has 0 vertices: {file_path.name}")
            
        # Step 2: Normalization (centering and scaling)
        # Bounding box center centering
        min_box = vertices.min(axis=0)
        max_box = vertices.max(axis=0)
        center = (min_box + max_box) / 2.0
        vertices = vertices - center
        
        # Scale to unit box
        extents = max_box - min_box
        max_extent = max(extents)
        if max_extent == 0:
            raise ValueError(f"Mesh extents are zero: {file_path.name}")
        scale_factor = 1.0 / max_extent
        vertices = vertices * scale_factor
        
        # Update vertices in-place
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        
        # Step 3: Export Normalized STL (optional)
        if save_norm:
            # Normals must be computed for STL writer in Open3D
            mesh.compute_triangle_normals()
            mesh.compute_vertex_normals()
            norm_path = norm_dir / f"{file_path.stem}_norm.stl"
            o3d.io.write_triangle_mesh(str(norm_path), mesh, write_ascii=False)
        
        # Step 4: Sample Point Cloud from the normalized mesh
        pcd = mesh.sample_points_uniformly(number_of_points=num_points_pc)
        if not pcd.has_points():
            raise ValueError(f"Failed to sample points from mesh surface: {file_path.name}")
            
        pc_path = pc_dir / f"{file_path.stem}_pc.ply"
        o3d.io.write_point_cloud(str(pc_path), pcd, write_ascii=False)
        
        # Step 5: Sample Occupancy Query Points
        num_half = num_points_occ // 2
        
        # Bounding box of normalized mesh
        norm_min = vertices.min(axis=0)
        norm_max = vertices.max(axis=0)
        
        # Uniform sampling inside bounding box
        uniform_points = np.random.uniform(norm_min, norm_max, size=(num_half, 3))
        
        # Surface perturbation sampling - Optimized: reuse pcd points from step 4!
        surface_points_all = np.asarray(pcd.points)
        idx = np.random.choice(len(surface_points_all), size=num_half, replace=(len(surface_points_all) < num_half))
        surface_points = surface_points_all[idx]
        noise = np.random.normal(scale=sigma_occ, size=surface_points.shape)
        perturbed_points = surface_points + noise
        
        del pcd
        
        query_points = np.vstack([uniform_points, perturbed_points]).astype(np.float32)
        
        # Compute occupancy labels using RaycastingScene
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(t_mesh)
        
        query_tensor = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
        occupancy_tensor = scene.compute_occupancy(query_tensor)
        occupancy = occupancy_tensor.numpy().astype(np.float32)
        
        occ_path = occ_dir / f"{file_path.stem}_occ.npz"
        np.savez_compressed(
            str(occ_path),
            query_points=query_points,
            occupancy=occupancy
        )
        
        # Explicit memory cleanup
        del mesh, t_mesh, scene, query_tensor, occupancy_tensor
        return True
        
    except Exception as e:
        print(f"\n  [Error] Failed to process {file_path.name}: {str(e)}", flush=True)
        return False

def main():
    args = parse_args()
    
    raw_dir = Path(args.raw_dir)
    norm_dir = Path(args.norm_dir)
    pc_dir = Path(args.pc_dir)
    occ_dir = Path(args.occ_dir)
    
    print("=" * 60)
    print("   Combined Parallel Mesh Preprocessing Pipeline (Open3D)")
    print("=" * 60)
    print(f"Raw input dir    : {raw_dir}")
    print(f"Normalized output: {norm_dir} (Save: {args.save_norm})")
    print(f"Point Cloud output: {pc_dir}")
    print(f"Occupancy output : {occ_dir}")
    print(f"Workers          : {args.workers}")
    print("-" * 60)
    
    if not raw_dir.exists():
        print(f"[Error] Raw directory '{raw_dir}' does not exist.")
        sys.exit(1)
        
    stl_files = sorted(list(raw_dir.glob("*.stl")))
    total_files = len(stl_files)
    
    if total_files == 0:
        print(f"[Warning] No STL files found in {raw_dir}")
        sys.exit(0)
        
    norm_dir.mkdir(parents=True, exist_ok=True)
    pc_dir.mkdir(parents=True, exist_ok=True)
    occ_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seeds for determinism
    np.random.seed(42)
    # Import Open3D locally to seed it
    import open3d as o3d
    o3d.utility.random.seed(42)
    
    start_time = time.time()
    successful_count = 0
    
    if args.workers <= 1:
        for idx, file_path in enumerate(stl_files, start=1):
            success = process_single_file(
                file_path=file_path,
                norm_dir=norm_dir,
                pc_dir=pc_dir,
                occ_dir=occ_dir,
                num_points_pc=args.num_points_pc,
                num_points_occ=args.num_points_occ,
                sigma_occ=args.sigma_occ,
                save_norm=args.save_norm
            )
            if success:
                successful_count += 1
            print(f"[{idx}/{total_files}] Processed {file_path.name}: {'[OK]' if success else '[FAIL]'}", flush=True)
            if idx % 10 == 0:
                gc.collect()
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for file_path in stl_files:
                future = executor.submit(
                    process_single_file,
                    file_path=file_path,
                    norm_dir=norm_dir,
                    pc_dir=pc_dir,
                    occ_dir=occ_dir,
                    num_points_pc=args.num_points_pc,
                    num_points_occ=args.num_points_occ,
                    sigma_occ=args.sigma_occ,
                    save_norm=args.save_norm
                )
                futures[future] = file_path
                
            for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                file_path = futures[future]
                success = future.result()
                if success:
                    successful_count += 1
                    
                # print progress dynamically
                print(f"[{idx}/{total_files}] Processed {file_path.name}: {'[OK]' if success else '[FAIL]'}", flush=True)
                
                # Clean up memory occasionally
                if idx % 10 == 0:
                    gc.collect()
                
    elapsed = time.time() - start_time
    print(f"\n{'-' * 60}")
    print(f"Combined processing completed in {elapsed:.2f} seconds!")
    print(f"Successfully processed: {successful_count}/{total_files} meshes")
    print(f"Average time per mesh : {elapsed / total_files:.2f} seconds")
    print("=" * 60)

if __name__ == "__main__":
    main()
