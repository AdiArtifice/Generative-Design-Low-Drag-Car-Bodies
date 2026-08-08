#!/usr/bin/env python
"""
Surgical Patch Script: Process Missing F_S_WWC_WM Meshes
---------------------------------------------------------
This script identifies and processes ONLY the missing meshes for F_S_WWC_WM.
It handles three separate gaps:
  1. 55 meshes missing from normalized/ and pointclouds/ (zero-byte normalized files)
  2. ~503 meshes missing from occupancy/ (interrupted batch run)

Strategy:
  - For the 55 missing meshes: copy raw STL from G:\ -> normalize -> sample PC -> compute occ -> delete raw+norm
  - For the ~448 meshes that have valid normalized STLs but no occupancy: compute occ from existing normalized
  - Each raw STL is deleted immediately after its outputs are generated (~150MB peak disk usage)
  - At the end: run compute_features and link_metadata on the full 692-car dataset

Usage:
    python scripts/patch_missing_meshes.py
"""

import os
import sys
import gc
import shutil
from pathlib import Path
import numpy as np

# --------------- Configuration ---------------
CONFIG = "F_S_WWC_WM"
G_DRIVE_SRC = Path(r"G:\.shortcut-targets-by-id\1WOsw0v1GPcX8lMXMErBMlQYwLrQKF3pQ\Main Project Resource\3D meshes of EV cars\F_S_WWC_WM")
NORM_DIR = Path(f"normalized/{CONFIG}")
PC_DIR = Path(f"pointclouds/{CONFIG}")
OCC_DIR = Path(f"occupancy/{CONFIG}")
NUM_PC_POINTS = 50000
NUM_OCC_POINTS = 2048
OCC_SIGMA = 0.015

def get_missing_ids():
    """Identify exactly which mesh IDs are missing from each output directory."""
    # All IDs available on G:\ drive (ground truth: 692)
    g_drive_ids = set(p.stem for p in G_DRIVE_SRC.glob("*.stl"))
    # All valid normalized IDs (non-zero-byte)
    norm_valid = set(p.stem.replace("_norm", "") for p in NORM_DIR.glob("*.stl") if p.stat().st_size > 0)
    # Zero-byte normalized IDs (need full reprocessing from raw STL)
    norm_zero = set(p.stem.replace("_norm", "") for p in NORM_DIR.glob("*.stl") if p.stat().st_size == 0)
    # Existing point cloud IDs
    pc_ids = set(p.stem.replace("_pc", "") for p in PC_DIR.glob("*.ply"))
    # Existing occupancy IDs
    occ_ids = set(p.stem.replace("_norm_occ", "") for p in OCC_DIR.glob("*.npz"))
    
    # IDs that need FULL reprocessing: zero-byte OR completely absent locally
    locally_present = norm_valid | norm_zero | pc_ids
    completely_absent = g_drive_ids - locally_present
    full_reprocess = sorted(norm_zero | completely_absent)
    # IDs that have valid normalized STL but missing occupancy
    occ_only = sorted(norm_valid - occ_ids - norm_zero)
    
    return full_reprocess, occ_only, norm_valid, pc_ids, occ_ids


def normalize_single_mesh(raw_path, norm_path):
    """Normalize a single mesh: center + scale to unit bounding box."""
    import trimesh
    try:
        mesh = trimesh.load_mesh(str(raw_path), process=False)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
            return False
        center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-center)
        scale_factor = 1.0 / max(mesh.extents)
        mesh.apply_scale(scale_factor)
        mesh.export(str(norm_path), file_type="stl")
        del mesh
        return True
    except Exception as e:
        print(f"    [Error] Normalize failed: {e}")
        return False


def sample_pointcloud_single(norm_path, pc_path):
    """Sample a point cloud from a single normalized mesh."""
    import open3d as o3d
    try:
        mesh = o3d.io.read_triangle_mesh(str(norm_path))
        if not mesh.has_triangles():
            return False
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=NUM_PC_POINTS)
        if not pcd.has_points() or len(pcd.points) != NUM_PC_POINTS:
            return False
        o3d.io.write_point_cloud(str(pc_path), pcd, write_ascii=False)
        del mesh, pcd
        return True
    except Exception as e:
        print(f"    [Error] PC sampling failed: {e}")
        return False


def compute_occupancy_single(norm_path, occ_path):
    """Compute occupancy grid for a single normalized mesh."""
    import open3d as o3d
    try:
        mesh = o3d.io.read_triangle_mesh(str(norm_path))
        if len(mesh.vertices) == 0:
            return False
        num_half = NUM_OCC_POINTS // 2
        vertices = np.asarray(mesh.vertices)
        min_box = vertices.min(axis=0)
        max_box = vertices.max(axis=0)
        uniform_points = np.random.uniform(min_box, max_box, size=(num_half, 3))
        pcd = mesh.sample_points_uniformly(number_of_points=num_half)
        surface_points = np.asarray(pcd.points)
        noise = np.random.normal(scale=OCC_SIGMA, size=surface_points.shape)
        perturbed_points = surface_points + noise
        query_points = np.vstack([uniform_points, perturbed_points]).astype(np.float32)
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(t_mesh)
        query_tensor = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
        occupancy = scene.compute_occupancy(query_tensor).numpy().astype(np.float32)
        occ_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(occ_path), query_points=query_points, occupancy=occupancy)
        del mesh, t_mesh, scene, query_tensor
        return True
    except Exception as e:
        print(f"    [Error] Occupancy failed: {e}")
        return False


def main():
    np.random.seed(42)
    
    print("=" * 60)
    print("  Surgical Patch: Completing F_S_WWC_WM Missing Meshes")
    print("=" * 60)
    
    # --- Phase 0: Audit ---
    full_reprocess, occ_only, norm_valid, pc_ids, occ_ids = get_missing_ids()
    print(f"\nAudit Results:")
    print(f"  Valid normalized STLs : {len(norm_valid)}")
    print(f"  Zero-byte (need full) : {len(full_reprocess)}")
    print(f"  Point clouds present  : {len(pc_ids)}")
    print(f"  Occupancy present     : {len(occ_ids)}")
    print(f"  Need occupancy only   : {len(occ_only)}")
    print(f"  G:\\ drive accessible  : {G_DRIVE_SRC.exists()}")
    
    if not G_DRIVE_SRC.exists():
        print("[FATAL] G:\\ drive source directory not accessible. Aborting.")
        sys.exit(1)
    
    # --- Phase 1: Full reprocessing of 55 missing meshes ---
    print(f"\n{'='*60}")
    print(f"  Phase 1: Full Reprocessing ({len(full_reprocess)} meshes)")
    print(f"  Strategy: Copy 1 raw STL -> Normalize -> PC -> OCC -> Delete")
    print(f"{'='*60}")
    
    # First, delete the 55 zero-byte normalized files
    deleted_count = 0
    for mesh_id in full_reprocess:
        zero_file = NORM_DIR / f"{mesh_id}_norm.stl"
        if zero_file.exists() and zero_file.stat().st_size == 0:
            zero_file.unlink()
            deleted_count += 1
    print(f"  Cleaned up {deleted_count} zero-byte normalized files.\n")
    
    phase1_success = 0
    phase1_fail = 0
    for idx, mesh_id in enumerate(full_reprocess, 1):
        raw_name = f"{mesh_id}.stl"
        norm_name = f"{mesh_id}_norm.stl"
        pc_name = f"{mesh_id}_pc.ply"
        occ_name = f"{mesh_id}_norm_occ.npz"
        
        raw_src = G_DRIVE_SRC / raw_name
        raw_local = Path(f"raw_stl/{CONFIG}") / raw_name
        norm_path = NORM_DIR / norm_name
        pc_path = PC_DIR / pc_name
        occ_path = OCC_DIR / occ_name
        
        print(f"  [{idx}/{len(full_reprocess)}] {mesh_id}:", end="", flush=True)
        
        # Step A: Copy raw STL from G:\ drive
        if not raw_src.exists():
            print(f" [SKIP] Not found on G:\\ drive")
            phase1_fail += 1
            continue
        raw_local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_src, raw_local)
        print(" copied", end="", flush=True)
        
        # Step B: Normalize
        ok = normalize_single_mesh(raw_local, norm_path)
        if not ok:
            print(" [FAIL norm]")
            raw_local.unlink(missing_ok=True)
            phase1_fail += 1
            continue
        print(" -> normed", end="", flush=True)
        
        # Step C: Sample point cloud
        ok = sample_pointcloud_single(norm_path, pc_path)
        if not ok:
            print(" [FAIL pc]")
            raw_local.unlink(missing_ok=True)
            phase1_fail += 1
            continue
        print(" -> pc", end="", flush=True)
        
        # Step D: Compute occupancy
        ok = compute_occupancy_single(norm_path, occ_path)
        if not ok:
            print(" [FAIL occ]")
            raw_local.unlink(missing_ok=True)
            phase1_fail += 1
            continue
        print(" -> occ", end="", flush=True)
        
        # Step E: Delete raw STL immediately (save ~75MB per file)
        raw_local.unlink(missing_ok=True)
        print(" -> cleaned [OK]", flush=True)
        
        phase1_success += 1
        gc.collect()
    
    # Clean up raw_stl directory if empty
    raw_dir = Path(f"raw_stl/{CONFIG}")
    if raw_dir.exists() and len(list(raw_dir.glob("*"))) == 0:
        raw_dir.rmdir()
    
    print(f"\n  Phase 1 Complete: {phase1_success}/{len(full_reprocess)} succeeded, {phase1_fail} failed")
    
    # --- Phase 2: Compute occupancy for meshes that have valid normalized STLs ---
    # Re-audit after Phase 1
    _, occ_only_updated, _, _, _ = get_missing_ids()
    
    print(f"\n{'='*60}")
    print(f"  Phase 2: Occupancy-Only ({len(occ_only_updated)} meshes)")
    print(f"  Strategy: Read existing normalized STL -> Compute OCC")
    print(f"{'='*60}")
    
    phase2_success = 0
    phase2_fail = 0
    for idx, mesh_id in enumerate(occ_only_updated, 1):
        norm_path = NORM_DIR / f"{mesh_id}_norm.stl"
        occ_path = OCC_DIR / f"{mesh_id}_norm_occ.npz"
        
        if not norm_path.exists() or norm_path.stat().st_size == 0:
            phase2_fail += 1
            continue
        
        if idx % 50 == 1 or idx == len(occ_only_updated):
            print(f"  [{idx}/{len(occ_only_updated)}] Processing {mesh_id}...", end="", flush=True)
        
        ok = compute_occupancy_single(norm_path, occ_path)
        if ok:
            phase2_success += 1
            if idx % 50 == 1 or idx == len(occ_only_updated):
                print(" [OK]", flush=True)
        else:
            phase2_fail += 1
            if idx % 50 == 1 or idx == len(occ_only_updated):
                print(" [FAIL]", flush=True)
        
        if idx % 10 == 0:
            gc.collect()
    
    print(f"\n  Phase 2 Complete: {phase2_success}/{len(occ_only_updated)} succeeded, {phase2_fail} failed")
    
    # --- Phase 3: Final Audit ---
    print(f"\n{'='*60}")
    print(f"  Phase 3: Final Audit")
    print(f"{'='*60}")
    
    norm_final = len([p for p in NORM_DIR.glob("*.stl") if p.stat().st_size > 0])
    pc_final = len(list(PC_DIR.glob("*.ply")))
    occ_final = len(list(OCC_DIR.glob("*.npz")))
    
    print(f"  Normalized (valid) : {norm_final} / 692")
    print(f"  Point Clouds       : {pc_final} / 692")
    print(f"  Occupancy Grids    : {occ_final} / 692")
    
    all_good = (norm_final == 692 and pc_final == 692 and occ_final == 692)
    
    if not all_good:
        print(f"\n  [WARNING] Not all 692 meshes are complete. Review above counts.")
    else:
        print(f"\n  [SUCCESS] All 692 meshes are fully processed!")
    
    # --- Phase 4: Compute features and link metadata ---
    print(f"\n{'='*60}")
    print(f"  Phase 4: Compute Features & Link Metadata")
    print(f"{'='*60}")
    
    import subprocess
    
    print("  Running compute_features.py on full pointclouds...", flush=True)
    res = subprocess.run([sys.executable, "scripts/compute_features.py", "--input", str(PC_DIR)], text=True)
    if res.returncode != 0:
        print(f"  [Error] compute_features.py failed with exit code {res.returncode}")
    else:
        print("  compute_features.py completed [OK]")
    
    print("  Running link_metadata.py...", flush=True)
    res = subprocess.run([sys.executable, "scripts/link_metadata.py"], text=True)
    if res.returncode != 0:
        print(f"  [Error] link_metadata.py failed with exit code {res.returncode}")
    else:
        print("  link_metadata.py completed [OK]")
    
    print(f"\n{'='*60}")
    print(f"  Patch Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
