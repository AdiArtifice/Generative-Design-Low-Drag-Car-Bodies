#!/usr/bin/env python
"""
Mesh Inspection Script
----------------------
This script scans the raw STL meshes, validates their topological and geometric 
integrity (watertightness, manifoldness, vertex/face count, bounding box bounds), 
attempts minor repairs (fixing face normals), and outputs a comprehensive 
CSV inspection report to the metadata directory.

Usage:
    python scripts/inspect_meshes.py [--input INPUT_DIR] [--output OUTPUT_CSV]
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import trimesh
from dotenv import load_dotenv

# Load configuration from .env file
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect and validate STL meshes.")
    
    # Default paths from environment variables, falling back to sensible project defaults
    default_input = os.getenv("RAW_STL_DIR", "raw_stl/fastback smooth wheel with covers")
    default_metadata_dir = os.getenv("METADATA_DIR", "metadata")
    default_output = os.path.join(default_metadata_dir, "mesh_inspection_report.csv")
    
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
        help=f"Path to save output CSV report (default: {default_output})"
    )
    
    return parser.parse_args()

def inspect_mesh(file_path: Path) -> dict:
    """
    Loads and inspects a single STL mesh, reporting key geometric and topological features.
    
    Parameters:
        file_path (Path): Path to the STL file.
        
    Returns:
        dict: Inspection records/statistics for this mesh.
    """
    record = {
        "id": file_path.stem,
        "filename": file_path.name,
        "is_watertight": False,
        "is_manifold": False,
        "num_vertices": 0,
        "num_faces": 0,
        "bounds_min_x": 0.0,
        "bounds_min_y": 0.0,
        "bounds_min_z": 0.0,
        "bounds_max_x": 0.0,
        "bounds_max_y": 0.0,
        "bounds_max_z": 0.0,
        "extent_x": 0.0,
        "extent_y": 0.0,
        "extent_z": 0.0,
        "winding_consistent": False,
        "repair_status": "none",
        "volume": 0.0,
        "status": "success"
    }
    
    try:
        # Load mesh. process=False preserves raw vertices/faces without auto-merging or healing.
        mesh = trimesh.load_mesh(str(file_path), process=False)
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"File loaded is not a valid Trimesh object: {type(mesh)}")
            
        record["num_vertices"] = len(mesh.vertices)
        record["num_faces"] = len(mesh.faces)
        
        # Geometrical bounding box checks
        if mesh.bounds is not None:
            bounds_min, bounds_max = mesh.bounds
            record["bounds_min_x"] = float(bounds_min[0])
            record["bounds_min_y"] = float(bounds_min[1])
            record["bounds_min_z"] = float(bounds_min[2])
            record["bounds_max_x"] = float(bounds_max[0])
            record["bounds_max_y"] = float(bounds_max[1])
            record["bounds_max_z"] = float(bounds_max[2])
            
        if mesh.extents is not None:
            record["extent_x"] = float(mesh.extents[0])
            record["extent_y"] = float(mesh.extents[1])
            record["extent_z"] = float(mesh.extents[2])
            
        # Topological checks
        record["is_watertight"] = bool(mesh.is_watertight)
        record["is_manifold"] = bool(mesh.is_volume)  # If it has non-zero closed volume
        
        # Winding check and normal repair
        winding_ok = bool(mesh.is_winding_consistent)
        record["winding_consistent"] = winding_ok
        
        if not winding_ok:
            print(f"  [!] Winding inconsistent in {file_path.name}. Attempting normal repair...", flush=True)
            mesh.fix_normals()
            if mesh.is_winding_consistent:
                record["winding_consistent"] = True
                record["repair_status"] = "fixed_normals"
                print(f"  [+] Winding repaired successfully for {file_path.name}.", flush=True)
            else:
                record["repair_status"] = "failed_repair"
                print(f"  [-] Normal repair failed for {file_path.name}.", flush=True)
                
        # Volume (only valid if watertight)
        if record["is_watertight"]:
            record["volume"] = float(mesh.volume)
            
    except Exception as e:
        print(f"  [Error] Failed to process {file_path.name}: {str(e)}", flush=True)
        record["status"] = "error"
        record["repair_status"] = f"error: {str(e)}"
        
    return record

def main():
    args = parse_args()
    
    input_dir = Path(args.input)
    output_csv = Path(args.output)
    
    print("=" * 60)
    print("                Mesh Preprocessing Pipeline: Inspection")
    print("=" * 60)
    print(f"Input directory : {input_dir}")
    print(f"Output CSV path : {output_csv}")
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
        
    print(f"Found {total_files} STL files to inspect. Starting inspection...")
    
    # Ensure metadata parent folder exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    import gc
    records = []
    for idx, file_path in enumerate(stl_files, start=1):
        print(f"[{idx}/{total_files}] Inspecting {file_path.name}...", flush=True)
        res = inspect_mesh(file_path)
        records.append(res)
        gc.collect()
    # Compile report and export
    df_report = pd.DataFrame(records)
    if output_csv.exists():
        df_report.to_csv(output_csv, mode='a', header=False, index=False)
    else:
        df_report.to_csv(output_csv, index=False)
    
    print("-" * 60)
    print("Inspection Completed Successfully!")
    print(f"Report exported to: {output_csv}")
    print("-" * 60)
    
    # Print brief summary statistics
    successful_runs = df_report[df_report["status"] == "success"]
    watertight_count = successful_runs["is_watertight"].sum()
    inconsistent_count = df_report[df_report["repair_status"] == "fixed_normals"].count()["id"]
    
    print(f"Total processed     : {total_files}")
    print(f"Successful loads    : {len(successful_runs)}")
    print(f"Watertight meshes   : {watertight_count}")
    print(f"Normals auto-fixed  : {inconsistent_count}")
    print("=" * 60)

if __name__ == "__main__":
    main()
