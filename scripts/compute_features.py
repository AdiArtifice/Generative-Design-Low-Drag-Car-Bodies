#!/usr/bin/env python
"""
Feature Engineering Script
--------------------------
This script calculates geometric features from point cloud meshes.

Features Computed:
    - Bounding Box Dimensions: Length (X), Width (Y), Height (Z)
    - Bounding Box Volume (L * W * H)
    - Convex Hull Volume (volume of the 3D tightly wrapped convex hull)
    - Normalized Frontal Area (2D silhouette projection on YZ-plane; Note: True physical Frontal Area
      is merged from DrivAerNetPlusPlus_CarDesign_Areas.csv in scripts/link_metadata.py)

Usage:
    python scripts/compute_features.py --input pointclouds/F_S_WWC_WM
"""

import os
import sys
import gc
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import trimesh
from dotenv import load_dotenv

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Compute aerodynamic geometric features.")
    
    default_input = os.getenv("POINTCLOUD_DIR", "pointclouds/F_S_WWC_WM")
    default_output = os.getenv("METADATA_DIR", "metadata") + "/computed_features.csv"
    
    parser.add_argument("--input", type=str, default=default_input, help="Point Cloud input dir (.ply)")
    parser.add_argument("--output", type=str, default=default_output, help="Output CSV path")
    parser.add_argument("--resolution", type=float, default=0.005, help="Frontal area grid resolution in meters (default: 5mm)")
    
    return parser.parse_args()

def compute_frontal_area_vectorized(mesh: trimesh.Trimesh, resolution: float) -> float:
    """
    Computes the silhouette frontal area by projecting all vertices onto the YZ-plane
    (longitudinal axis = X) and binning them using np.histogram2d.
    """
    if len(mesh.vertices) == 0:
        return 0.0
        
    y_coords = mesh.vertices[:, 1]
    z_coords = mesh.vertices[:, 2]
    
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    min_z, max_z = np.min(z_coords), np.max(z_coords)
    
    # Number of bins based on resolution
    bins_y = int(np.ceil((max_y - min_y) / resolution))
    bins_z = int(np.ceil((max_z - min_z) / resolution))
    
    if bins_y <= 0 or bins_z <= 0:
        return 0.0
        
    # Vectorized 2D histogram
    H, _, _ = np.histogram2d(y_coords, z_coords, bins=[bins_y, bins_z])
    
    filled_cells = np.count_nonzero(H)
    cell_area = resolution * resolution
    
    return float(filled_cells * cell_area)

def extract_features(file_path: Path, resolution: float) -> dict:
    """
    Extracts geometric features from a raw mesh.
    """
    record = {
        "id": file_path.stem.replace("_pc", ""),  # Ensure raw ID format
        "filename": file_path.name,
        "status": "error"
    }
    
    try:
        # Load point cloud using trimesh
        mesh = trimesh.load(str(file_path), process=False)
        
        # In trimesh, PointCloud objects also have .extents and .convex_hull
        extents = mesh.extents
        length, width, height = extents[0], extents[1], extents[2]
        
        record["length_x"] = float(length)
        record["width_y"] = float(width)
        record["height_z"] = float(height)
        
        # 2. Bounding Box Volume
        record["bbox_volume"] = float(length * width * height)
        
        # 3. Convex Hull Volume
        convex_volume = mesh.convex_hull.volume
        record["convex_hull_volume"] = float(convex_volume)
        
        # 4. Frontal Area (Silhouette Projection)
        frontal_area = compute_frontal_area_vectorized(mesh, resolution)
        record["frontal_area"] = float(frontal_area)
        
        record["status"] = "success"
        
        # Memory management
        del mesh
        
    except Exception as e:
        print(f"\n  [Error] Failed to process {file_path.name}: {str(e)}", flush=True)
        record["error_msg"] = str(e)
        
    return record

def main():
    args = parse_args()
    
    input_dir = Path(args.input)
    output_csv = Path(args.output)
    
    print("=" * 60)
    print("                Mesh Preprocessing: Feature Engineering")
    print("=" * 60)
    print(f"Input directory : {input_dir}")
    print(f"Output CSV path : {output_csv}")
    print(f"Grid Resolution : {args.resolution} m")
    print("-" * 60)
    
    if not input_dir.exists():
        print(f"[Error] Input directory '{input_dir}' does not exist.")
        sys.exit(1)
        
    # Find all PLY files recursively to support multiple configuration subfolders
    stl_files = sorted(list(input_dir.glob("**/*.ply")))
    total_files = len(stl_files)
    
    if total_files == 0:
        print(f"[Warning] No PLY files found in {input_dir}")
        sys.exit(0)
        
    print(f"Found {total_files} Point Cloud meshes. Computing features in parallel...")
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial

    worker_fn = partial(extract_features, resolution=args.resolution)
    records = []
    
    max_workers = min(os.cpu_count() or 4, 16)
    print(f"Using {max_workers} worker processes...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker_fn, f): f for f in stl_files}
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            records.append(res)
            completed += 1
            if completed % 100 == 0 or completed == total_files:
                print(f"Progress: [{completed}/{total_files}] files processed...", flush=True)

    # Sort records by id to keep deterministic order
    records.sort(key=lambda r: r.get("id", ""))
    
    # Export report
    df_features = pd.DataFrame(records)
    df_features.to_csv(output_csv, index=False)
    
    print("-" * 60)
    print("Feature Engineering Completed!")
    print(f"Computed features exported to: {output_csv}")
    print("=" * 60)

if __name__ == "__main__":
    main()
