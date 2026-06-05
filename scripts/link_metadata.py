#!/usr/bin/env python
"""
Metadata Linking Script
-----------------------
Fuses physical geometric features, CFD aerodynamic scores, and DrivAerNet 
parametric design control variables into a single unified master dataset (metadata.csv).

Key Output Features:
    - Inner join across three data sources with ID sanitization.
    - Drag Area index computation: drag_area = Cd * frontal_area.
    - Path injection for PyTorch Dataloader.
    - Deterministic 80/10/10 train/val/test split.
    - Target scale summary statistics saved to JSON.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Link aerodynamic and geometric parameters into master metadata.")
    
    default_features = "metadata/computed_features.csv"
    default_drag = "../selected_subset.csv"
    default_excel = "../DrivAerNet_ParametricData (2).xlsx"
    default_output_csv = "metadata/metadata.csv"
    default_output_json = "metadata/target_scales.json"
    
    parser.add_argument("--features-csv", type=str, default=default_features, help="Computed physical features CSV")
    parser.add_argument("--drag-csv", type=str, default=default_drag, help="CFD drag values CSV")
    parser.add_argument("--excel-file", type=str, default=default_excel, help="DrivAerNet parametric xlsx sheet")
    parser.add_argument("--output-csv", type=str, default=default_output_csv, help="Consolidated master CSV output path")
    parser.add_argument("--output-json", type=str, default=default_output_json, help="Summary statistics JSON output path")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    features_csv = Path(args.features_csv)
    drag_csv = Path(args.drag_csv)
    excel_file = Path(args.excel_file)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    
    print("=" * 60)
    print("                Mesh Preprocessing: Metadata Linking")
    print("=" * 60)
    print(f"Features CSV      : {features_csv}")
    print(f"Drag CSV          : {drag_csv}")
    print(f"Excel Sheet       : {excel_file}")
    print(f"Output Master CSV : {output_csv}")
    print(f"Output Statistics : {output_json}")
    print("-" * 60)
    
    # 1. Validation of inputs
    if not features_csv.exists():
        print(f"[Error] Features file '{features_csv}' does not exist. Run scripts/compute_features.py first.")
        sys.exit(1)
    if not drag_csv.exists():
        print(f"[Error] Drag values file '{drag_csv}' does not exist.")
        sys.exit(1)
    if not excel_file.exists():
        print(f"[Error] Parametric Excel sheet '{excel_file}' does not exist.")
        sys.exit(1)
        
    # 2. Load Data Sources
    print("Loading data sources...", flush=True)
    df_features = pd.read_csv(features_csv)
    df_drag = pd.read_csv(drag_csv)
    df_excel = pd.read_excel(excel_file)
    
    # 3. ID Sanitization and Normalization
    # Trim whitespaces, enforce uppercase, and remove suffixes
    df_features["id_clean"] = df_features["id"].astype(str).str.strip().str.upper().str.replace("_NORM", "")
    df_drag["id_clean"] = df_drag["ID"].astype(str).str.strip().str.upper().str.replace("_NORM", "")
    df_excel["id_clean"] = df_excel["Experiment"].astype(str).str.strip().str.upper().str.replace("_NORM", "")
    
    # 4. Multi-table Merging
    print("Performing multi-table joins...", flush=True)
    # Merge Features & Drag Values
    df_merged = pd.merge(df_features, df_drag, on="id_clean", suffixes=("", "_drag"))
    
    # Merge with Parametric Excel sheet
    df_final = pd.merge(df_merged, df_excel, on="id_clean", suffixes=("", "_excel"))
    
    # Assertion to verify exact matching of all 100 meshes
    record_count = len(df_final)
    print(f"Successfully joined dataset. Merged record count: {record_count}")
    assert record_count == 100, f"[Assertion Error] Expected exactly 100 records, but got {record_count}!"
    
    # 5. Feature Engineering
    print("Performing feature engineering...", flush=True)
    # Target values: CD can be Average Cd or Drag_Value (they are identical)
    df_final["cd"] = df_final["Drag_Value"]
    
    # Drag Area Index: Cd * Frontal Area (essential physical metric)
    df_final["drag_area"] = df_final["cd"] * df_final["frontal_area"]
    
    # Configuration code extraction (e.g. F_S_WWC_WM)
    df_final["config"] = df_final["id_clean"].apply(lambda x: "_".join(x.split("_")[:-1]))
    
    # 6. Inject relative paths for PyTorch dataloaders
    df_final["raw_stl_path"] = df_final["id_clean"].apply(lambda x: f"raw_stl/fastback smooth wheel with covers/{x}.stl")
    df_final["normalized_stl_path"] = df_final["id_clean"].apply(lambda x: f"normalized/fastback_smooth_wheelcovers/{x}_norm.stl")
    df_final["pointcloud_path"] = df_final["id_clean"].apply(lambda x: f"pointclouds/fastback_smooth_wheelcovers/{x}_pc.ply")
    
    # 7. Deterministic Train/Val/Test Split (80/10/10)
    print("Executing deterministic train/val/test splits (80/10/10)...", flush=True)
    # Seed for determinism
    np.random.seed(42)
    # Shuffle indices
    shuffled_indices = np.random.permutation(len(df_final))
    
    # Split assignment
    splits = []
    for idx in range(len(df_final)):
        shuffled_pos = np.where(shuffled_indices == idx)[0][0]
        if shuffled_pos < 80:
            splits.append("train")
        elif shuffled_pos < 90:
            splits.append("val")
        else:
            splits.append("test")
            
    df_final["split"] = splits
    
    # Clean up column headers (discard raw duplicate ID columns from joins)
    # Keep standard neat id and config columns
    df_final["id"] = df_final["id_clean"]
    
    # Select and order final clean columns
    param_cols = [
        'B_Ramp_Angle', 'B_Diffusor_Angle', 'B_Trunklid_Angle', 'C_Side_Mirrors_Rotation', 
        'D_Rear_Window_Inclination', 'D_Winscreen_Inclination', 'C_Side_Mirrors_Translate_X', 
        'C_Side_Mirrors_Translate_Z', 'D_Winscreen_Length', 'D_Rear_Window_Length', 
        'E_A_B_C_Pillar_Thickness', 'G_Trunklid_Curvature', 'G_Trunklid_Length', 
        'H_Front_Bumper_Curvature', 'H_Front_Bumper_Length', 'F_Door_Handles_Thickness', 
        'F_Door_Handles_Z_Position', 'E_Fenders_Arch_Offset', 'A_Car_Length', 
        'F_Door_Handles_X_Position', 'A_Car_Width', 'A_Car_Roof_Height', 'A_Car_Green_House_Angle'
    ]
    
    base_cols = [
        "id", "config", "split", "cd", "drag_area", 
        "Average Cl", "Average Cl_f", "Average Cl_r",
        "frontal_area", "convex_hull_volume", "bbox_volume", 
        "length_x", "width_y", "height_z",
        "raw_stl_path", "normalized_stl_path", "pointcloud_path"
    ]
    
    final_cols = base_cols + param_cols
    df_output = df_final[final_cols].copy()
    
    # Rename excel lift column headers for elegance
    df_output.rename(columns={
        "Average Cl": "cl",
        "Average Cl_f": "cl_f",
        "Average Cl_r": "cl_r"
    }, inplace=True)
    
    # 8. Export master CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_csv, index=False)
    print(f"Master metadata exported successfully to: {output_csv}", flush=True)
    
    # 9. Compute target summary statistics
    print("Computing target summary scales...", flush=True)
    target_stats = {}
    for col in ["cd", "drag_area", "cl", "cl_f", "cl_r", "frontal_area", "convex_hull_volume"]:
        target_stats[col] = {
            "mean": float(df_output[col].mean()),
            "std": float(df_output[col].std()),
            "min": float(df_output[col].min()),
            "max": float(df_output[col].max()),
            "median": float(df_output[col].median())
        }
        
    with open(output_json, "w") as f:
        json.dump(target_stats, f, indent=4)
    print(f"Summary targets JSON exported successfully to: {output_json}", flush=True)
    
    print("-" * 60)
    print("Metadata Linking Phase Completed successfully!")
    print(f"Train samples : {len(df_output[df_output['split'] == 'train'])}")
    print(f"Val samples   : {len(df_output[df_output['split'] == 'val'])}")
    print(f"Test samples  : {len(df_output[df_output['split'] == 'test'])}")
    print("=" * 60)

if __name__ == "__main__":
    main()
