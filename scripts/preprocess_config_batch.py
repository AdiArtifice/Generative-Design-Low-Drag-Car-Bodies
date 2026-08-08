#!/usr/bin/env python
"""
Batch Preprocessing Pipeline Orchestrator (Chunked & Storage-Safe)
-----------------------------------------------------------------
Orchestrates the end-to-end geometry and metadata preprocessing workflow for a specific
car configuration (e.g. E_S_WWC_WM, N_S_WW_WM) while respecting strict disk limits.

It processes the files in chunks:
  1. Inspects raw STL meshes directly from source (G: drive) to avoid local copy overhead.
  2. For each chunk of meshes (default 50):
     a. Copies raw STLs to a local temp folder.
     b. Normalizes meshes.
     c. Samples point clouds (.ply).
     d. Computes occupancy grids (.npz).
     e. Immediately deletes the local raw and normalized STLs for that chunk.
  3. Computes physical features recursively on all generated point clouds.
  4. Updates the master metadata.
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Automated single-config batch preprocessing pipeline (Chunked).")
    parser.add_argument("--config", type=str, required=True, help="Config name (e.g., E_S_WWC_WM, N_S_WW_WM)")
    parser.add_argument("--raw-source", type=str, required=True, help="Directory containing raw STL files to process (e.g. on G: drive)")
    parser.add_argument("--chunk-size", type=int, default=50, help="Number of files to process in a single chunk to prevent disk overflow")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers for processing")
    return parser.parse_args()

def run_command(cmd_args):
    cmd_str = " ".join(cmd_args)
    print(f"\n[EXEC] {cmd_str}", flush=True)
    res = subprocess.run(cmd_args, text=True)
    if res.returncode != 0:
        print(f"[Error] Command failed with exit code {res.returncode}: {cmd_str}")
        sys.exit(res.returncode)

def main():
    args = parse_args()
    config_name = args.config
    raw_src_dir = Path(args.raw_source)
    chunk_size = args.chunk_size
    
    if not raw_src_dir.exists():
        print(f"[Error] Raw source directory '{raw_src_dir}' does not exist.")
        sys.exit(1)
        
    stl_files = sorted(list(raw_src_dir.glob("*.stl")))
    total_files = len(stl_files)
    if total_files == 0:
        print(f"[Error] No STL files found in raw source directory: {raw_src_dir}")
        sys.exit(1)
        
    print("=" * 60)
    print(f"       Batch Preprocessing Orchestrator (Chunked): '{config_name}'")
    print(f"       Source: {raw_src_dir} ({total_files} files)")
    print(f"       Chunk Size: {chunk_size} files")
    print("=" * 60)
    
    # Target directories
    raw_target_dir = Path(f"raw_stl/{config_name}")
    norm_target_dir = Path(f"normalized/{config_name}")
    pc_target_dir = Path(f"pointclouds/{config_name}")
    occ_target_dir = Path(f"occupancy/{config_name}")
    
    # Ensure pointclouds and occupancy directories exist
    pc_target_dir.mkdir(parents=True, exist_ok=True)
    occ_target_dir.mkdir(parents=True, exist_ok=True)
    
    # Delete old local inspection report if it exists to start fresh for this config
    report_file = Path("metadata/mesh_inspection_report.csv")
    if report_file.exists():
        report_file.unlink()
        print("Removed old mesh inspection report.")
    
    # Process in chunks
    num_chunks = (total_files + chunk_size - 1) // chunk_size
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_files)
        chunk_files = stl_files[start_idx:end_idx]
        
        print(f"\n============================================================")
        print(f"  Processing Chunk {chunk_idx + 1}/{num_chunks} (Files {start_idx + 1} to {end_idx})")
        print(f"============================================================")
        
        # A. Copy chunk files to local temp raw_stl
        raw_target_dir.mkdir(parents=True, exist_ok=True)
        print(f"Copying {len(chunk_files)} raw STLs to local disk...", end="", flush=True)
        for stl in chunk_files:
            shutil.copy2(stl, raw_target_dir / stl.name)
        print(" [OK]", flush=True)
        
        # B. Run Combined Processing (Normalize, PC, and Occupancy in Parallel)
        print("\n--- Step 1: Combined Parallel Processing ---")
        run_command([
            sys.executable, "scripts/preprocess_mesh_combined.py",
            "--raw-dir", str(raw_target_dir),
            "--norm-dir", str(norm_target_dir),
            "--pc-dir", str(pc_target_dir),
            "--occ-dir", str(occ_target_dir),
            "--workers", str(args.num_workers)
        ])
        
        # E. Cleanup Chunk raw and normalized STL files to save space
        print("\n--- Step 5: Cleaning up Chunk STL Files ---")
        if raw_target_dir.exists():
            shutil.rmtree(raw_target_dir)
            print(f"  Deleted temp raw STLs: {raw_target_dir}")
        if norm_target_dir.exists():
            shutil.rmtree(norm_target_dir)
            print(f"  Deleted temp normalized STLs: {norm_target_dir}")
            
    # Step 6: Compute Physical Features recursively for all point clouds
    print("\n--- Step 6: Computing Physical Features ---")
    run_command([sys.executable, "scripts/compute_features.py", "--input", "pointclouds"])
    
    # Step 7: Update Master Metadata
    print("\n--- Step 7: Updating Master Metadata ---")
    run_command([sys.executable, "scripts/link_metadata.py"])
    
    print("\n" + "=" * 60)
    print(f"Successfully finished preprocessing batch '{config_name}' in chunks!")
    print("=" * 60)

if __name__ == "__main__":
    main()
