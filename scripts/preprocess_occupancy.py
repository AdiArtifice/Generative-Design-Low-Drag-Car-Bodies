#!/usr/bin/env python
"""
Occupancy Preprocessing Script (Open3D-Accelerated)
----------------------------------------------------
This script loads each normalized STL mesh using Open3D, samples 2,048 query points
(50% uniform in bounding box, 50% perturbed near the surface), computes whether they
are inside (1) or outside (0) the mesh using Open3D's fast RaycastingScene,
and exports the data to compressed .npz files.

Usage:
    python scripts/preprocess_occupancy.py [--metadata metadata/metadata.csv] [--output occupancy]
"""

import os
import sys
import gc
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import open3d as o3d
from dotenv import load_dotenv

# Load configuration from .env file
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess watertight meshes to calculate occupancy targets.")
    parser.add_argument(
        "--metadata",
        type=str,
        default="metadata/metadata.csv",
        help="Path to the master metadata CSV (default: metadata/metadata.csv)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Directory containing normalized STL files to process (if provided, bypasses metadata.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="occupancy",
        help="Root directory to save occupancy .npz files (default: occupancy)"
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=2048,
        help="Number of query points to sample per mesh (default: 2048)"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.015,
        help="Standard deviation of Gaussian noise added to surface points (default: 0.015)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to the first N meshes (useful for quick local tests)"
    )
    return parser.parse_args()

def process_mesh_occupancy(mesh_path: Path, output_path: Path, num_points: int, sigma: float) -> bool:
    """
    Samples query points and computes inside/outside labels for a single mesh using Open3D.
    
    Parameters:
        mesh_path (Path): Path to the input normalized STL file.
        output_path (Path): Path where the .npz file will be saved.
        num_points (int): Total number of query points to sample.
        sigma (float): Noise standard deviation for surface clustering.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Load mesh using Open3D
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        
        if len(mesh.vertices) == 0:
            raise ValueError(f"Mesh has 0 vertices: {mesh_path.name}")
            
        num_half = num_points // 2
        
        # 1. Sample uniform points inside the bounding box
        vertices = np.asarray(mesh.vertices)
        min_box = vertices.min(axis=0)
        max_box = vertices.max(axis=0)
        uniform_points = np.random.uniform(min_box, max_box, size=(num_half, 3))
        
        # 2. Sample points on the surface and perturb them
        pcd = mesh.sample_points_uniformly(number_of_points=num_half)
        surface_points = np.asarray(pcd.points)
        noise = np.random.normal(scale=sigma, size=surface_points.shape)
        perturbed_points = surface_points + noise
        
        # Combine query points
        query_points = np.vstack([uniform_points, perturbed_points]).astype(np.float32) # shape: [num_points, 3]
        
        # 3. Determine occupancy (1 for inside, 0 for outside) using RaycastingScene
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(t_mesh)
        
        query_tensor = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
        occupancy_tensor = scene.compute_occupancy(query_tensor)
        occupancy = occupancy_tensor.numpy().astype(np.float32) # shape: [num_points]
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save compressed npz file
        np.savez_compressed(
            str(output_path),
            query_points=query_points,
            occupancy=occupancy
        )
        
        # Clean up
        del mesh, t_mesh, scene, query_tensor, occupancy_tensor
        return True
        
    except Exception as e:
        print(f"  [Error] Failed to process occupancy for {mesh_path.name}: {str(e)}", flush=True)
        return False

def main():
    args = parse_args()
    
    metadata_path = Path(args.metadata)
    output_root = Path(args.output)
    
    print("=" * 60)
    print("        Occupancy Preprocessing Pipeline (Open3D-Accelerated)")
    print("=" * 60)
    print(f"Metadata file     : {metadata_path}")
    print(f"Output directory  : {output_root}")
    print(f"Points per mesh   : {args.num_points}")
    print(f"Surface noise (std): {args.sigma}")
    if args.limit:
        print(f"Limit processing  : First {args.limit} meshes")
    print("-" * 60)
    
    if args.input:
        input_root = Path(args.input)
        if not input_root.exists():
            print(f"[Error] Input directory '{input_root}' does not exist.")
            sys.exit(1)
            
        mesh_paths_all = list(input_root.glob("**/*.stl"))
        if args.limit:
            mesh_paths_all = mesh_paths_all[:args.limit]
            
        total_files = len(mesh_paths_all)
        print(f"Processing {total_files} meshes from input directory...")
        
        successful_count = 0
        for idx, mesh_path in enumerate(mesh_paths_all):
            # Same relative logic, but assuming input_root is something like normalized/F_S_WWC_WM
            try:
                rel_to_norm = mesh_path.relative_to(input_root.parent)
            except ValueError:
                rel_to_norm = Path(mesh_path.name)
            
            output_file_path = output_root / rel_to_norm.parent / f"{mesh_path.stem}_occ.npz"
            
            print(f"[{idx+1}/{total_files}] Processing {mesh_path.name} -> {output_file_path.name}...", end="", flush=True)
            success = process_mesh_occupancy(
                mesh_path=mesh_path,
                output_path=output_file_path,
                num_points=args.num_points,
                sigma=args.sigma
            )
            
            if success:
                successful_count += 1
                print(" Done.", flush=True)
                
            if idx % 10 == 0:
                gc.collect()

    else:
        if not metadata_path.exists():
            print(f"[Error] Metadata file '{metadata_path}' does not exist.")
            sys.exit(1)
            
        # Read metadata CSV
        df = pd.read_csv(metadata_path)
        
        if args.limit:
            df = df.head(args.limit)
            
        total_files = len(df)
        print(f"Processing {total_files} meshes from metadata...")
        
        successful_count = 0
        for idx, row in df.iterrows():
            mesh_rel_path = row["normalized_stl_path"]
            mesh_path = Path(mesh_rel_path)
            
            if not mesh_path.exists():
                print(f"[{idx+1}/{total_files}] [Error] File not found: {mesh_path}")
                continue
                
            # Structure the output file path to match relative structure under occupancy/
            rel_to_norm = mesh_path.relative_to("normalized")
            output_file_path = output_root / rel_to_norm.parent / f"{rel_to_norm.stem}_occ.npz"
            
            print(f"[{idx+1}/{total_files}] Processing {mesh_path.name} -> {output_file_path.name}...", end="", flush=True)
            success = process_mesh_occupancy(
                mesh_path=mesh_path,
                output_path=output_file_path,
                num_points=args.num_points,
                sigma=args.sigma
            )
            
            if success:
                successful_count += 1
                print(" Done.", flush=True)
                
            # Collect garbage to keep RAM clean
            if idx % 10 == 0:
                gc.collect()
            
    print("-" * 60)
    print("Occupancy Preprocessing Completed!")
    print(f"Successfully processed: {successful_count}/{total_files} meshes")
    print(f"Output saved to         : {output_root}")
    print("=" * 60)

if __name__ == "__main__":
    main()

