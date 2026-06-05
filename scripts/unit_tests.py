#!/usr/bin/env python
"""
Unit Tests for Preprocessing Pipeline
-------------------------------------
This file contains unit tests to validate the geometry centering, scaling, 
and file parsing functions within the pipeline.

Usage:
    pytest scripts/unit_tests.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pytest
import trimesh
import pandas as pd

# Add the workspace root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_geometry_centering_bounds():
    """
    Test that centering a mesh moves its bounding box center to exactly (0, 0, 0).
    """
    # Create a box with arbitrary size (2, 4, 6) offset from origin (5, 10, 15)
    offset = np.array([5.0, 10.0, 15.0])
    box = trimesh.primitives.Box(extents=[2.0, 4.0, 6.0])
    box.apply_translation(offset)
    
    # Bounding box center before centering
    initial_center = box.bounds.mean(axis=0)
    assert np.allclose(initial_center, offset), f"Expected center close to {offset}, got {initial_center}"
    
    # Center the box using bounds center
    center = box.bounds.mean(axis=0)
    box.apply_translation(-center)
    
    # Bounding box center after centering
    final_center = box.bounds.mean(axis=0)
    assert np.allclose(final_center, [0.0, 0.0, 0.0], atol=1e-7), f"Expected origin center, got {final_center}"

def test_geometry_scaling():
    """
    Test that scaling a mesh uniformly limits its largest extent dimension to exactly 1.0.
    """
    # Create a box with arbitrary size (3.0, 9.0, 2.0) centered at origin
    box = trimesh.primitives.Box(extents=[3.0, 9.0, 2.0])
    
    # Max extent before scaling should be 9.0
    assert np.allclose(max(box.extents), 9.0), f"Expected max extent 9.0, got {max(box.extents)}"
    
    # Scale box uniformly
    scale_factor = 1.0 / max(box.extents)
    box.apply_scale(scale_factor)
    
    # Max extent after scaling must be exactly 1.0
    assert np.allclose(max(box.extents), 1.0, atol=1e-7), f"Expected max extent 1.0, got {max(box.extents)}"
    # Bounding box limits should be within [-0.5, 0.5] along the largest axis
    assert np.allclose(box.bounds[0, 1], -0.5, atol=1e-7), f"Expected min bound along Y to be -0.5, got {box.bounds[0, 1]}"
    assert np.allclose(box.bounds[1, 1], 0.5, atol=1e-7), f"Expected max bound along Y to be 0.5, got {box.bounds[1, 1]}"

def test_inspection_report_file():
    """
    Test that the inspection report CSV was created and contains valid records.
    """
    report_path = Path("metadata/mesh_inspection_report.csv")
    
    # This test will run if the inspection script has finished. If not, it will be skipped.
    if not report_path.exists():
        pytest.skip("Inspection report CSV not generated yet.")
        
    df = pd.read_csv(report_path)
    
    # Should have 100 entries for the 100 meshes
    assert len(df) == 100, f"Expected 100 entries in inspection report, got {len(df)}"
    
    # Essential columns must exist
    required_cols = {
        "id", "filename", "is_watertight", "is_manifold", 
        "num_vertices", "num_faces", "extent_x", "extent_y", "extent_z", "status"
    }
    assert required_cols.issubset(df.columns), f"Missing required columns in CSV: {required_cols - set(df.columns)}"
    
    # All rows must report 'success' status
    assert (df["status"] == "success").all(), "Some STL meshes failed loading or processing"

def test_pointcloud_count():
    """
    Test that the generated point cloud PLY files contain exactly the requested number of points (50,000).
    """
    import open3d as o3d
    
    # Check if at least one PLY file exists (usually the first one F_S_WWC_WM_001_pc.ply)
    ply_path = Path("pointclouds/fastback_smooth_wheelcovers/F_S_WWC_WM_001_pc.ply")
    if not ply_path.exists():
        pytest.skip("Point cloud file F_S_WWC_WM_001_pc.ply not generated yet.")
        
    pcd = o3d.io.read_point_cloud(str(ply_path))
    assert len(pcd.points) == 50000, f"Expected 50000 points, got {len(pcd.points)}"
    assert pcd.has_normals(), "Point cloud PLY file is missing surface normal vectors."

def test_pointcloud_determinism():
    """
    Test that the point cloud sampling is mathematically deterministic when seeded.
    """
    import open3d as o3d
    
    # Create a simple synthetic sphere mesh using Open3D
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh.compute_vertex_normals()
    
    # Run first sampling with seed 42
    np.random.seed(42)
    o3d.utility.random.seed(42)
    pcd1 = mesh.sample_points_uniformly(number_of_points=1000)
    points1 = np.asarray(pcd1.points)
    normals1 = np.asarray(pcd1.normals)
    
    # Run second sampling with same seed 42
    np.random.seed(42)
    o3d.utility.random.seed(42)
    pcd2 = mesh.sample_points_uniformly(number_of_points=1000)
    points2 = np.asarray(pcd2.points)
    normals2 = np.asarray(pcd2.normals)
    
    # Assert coordinates and normals are exactly identical
    assert np.allclose(points1, points2, atol=1e-8), "Sampling coordinates are non-deterministic."
    assert np.allclose(normals1, normals2, atol=1e-8), "Sampling normals are non-deterministic."

def test_computed_features():
    """
    Test that computed features CSV was successfully generated and contains physically sound values.
    """
    features_path = Path("metadata/computed_features.csv")
    if not features_path.exists():
        pytest.skip("Computed features CSV not generated yet.")
        
    df = pd.read_csv(features_path)
    
    assert len(df) == 100, f"Expected 100 entries in computed features, got {len(df)}"
    
    required_cols = {"id", "filename", "frontal_area", "convex_hull_volume", "bbox_volume", "length_x", "width_y", "height_z"}
    assert required_cols.issubset(df.columns), f"Missing columns in features CSV: {required_cols - set(df.columns)}"
    
    # Physical plausibility assertions
    assert (df["frontal_area"] > 1.0).all() and (df["frontal_area"] < 3.0).all(), "Frontal area values are physically implausible!"
    assert (df["length_x"] > 4.0).all() and (df["length_x"] < 6.0).all(), "Vehicle length values are physically implausible!"
    assert (df["width_y"] > 1.5).all() and (df["width_y"] < 2.5).all(), "Vehicle width values are physically implausible!"

def test_master_metadata():
    """
    Test that the consolidated master metadata and target scales were successfully generated with correct splits and column mappings.
    """
    metadata_path = Path("metadata/metadata.csv")
    scales_path = Path("metadata/target_scales.json")
    
    if not metadata_path.exists() or not scales_path.exists():
        pytest.skip("Master metadata files not fully generated yet.")
        
    df = pd.read_csv(metadata_path)
    
    # 1. Total records check
    assert len(df) == 100, f"Expected exactly 100 records in master metadata, got {len(df)}"
    
    # 2. Key column mapping checks
    essential_cols = [
        "id", "config", "split", "cd", "drag_area", "cl", "cl_f", "cl_r",
        "frontal_area", "convex_hull_volume", "bbox_volume", 
        "length_x", "width_y", "height_z",
        "raw_stl_path", "normalized_stl_path", "pointcloud_path"
    ]
    for col in essential_cols:
        assert col in df.columns, f"Missing essential column '{col}' in master metadata!"
        
    # 3. Deterministic train/val/test split assertion (80/10/10)
    split_counts = df["split"].value_counts().to_dict()
    assert split_counts.get("train", 0) == 80, f"Expected 80 train samples, got {split_counts.get('train', 0)}"
    assert split_counts.get("val", 0) == 10, f"Expected 10 val samples, got {split_counts.get('val', 0)}"
    assert split_counts.get("test", 0) == 10, f"Expected 10 test samples, got {split_counts.get('test', 0)}"
    
    # 4. JSON scales scaling checks
    import json
    with open(scales_path, "r") as f:
        scales = json.load(f)
        
    assert "cd" in scales, "cd statistics missing in target_scales.json"
    assert "drag_area" in scales, "drag_area statistics missing in target_scales.json"
    assert "frontal_area" in scales, "frontal_area statistics missing in target_scales.json"
    
    assert scales["cd"]["min"] == df["cd"].min(), "cd scale min mismatch!"
    assert scales["cd"]["max"] == df["cd"].max(), "cd scale max mismatch!"

def test_pytorch_dataset():
    """
    Test that the PyTorch dataset returns correct shapes and applies Z-score target normalization.
    """
    from src.dataset import VehiclePointCloudDataset
    import torch
    
    metadata_path = Path("metadata/metadata.csv")
    scales_path = Path("metadata/target_scales.json")
    
    if not metadata_path.exists() or not scales_path.exists():
        pytest.skip("Metadata or scales file missing.")
        
    dataset = VehiclePointCloudDataset(
        csv_path=str(metadata_path),
        scales_path=str(scales_path),
        split="val",
        num_points=1024,
        normalize_targets=True
    )
    
    # 1. Split size validation
    assert len(dataset) == 10, f"Expected 10 validation samples, got {len(dataset)}"
    
    # 2. Shape validation
    features, targets = dataset[0]
    assert features.shape == (6, 1024), f"Expected shape (6, 1024), got {features.shape}"
    assert isinstance(features, torch.Tensor), "Features should be returned as PyTorch Tensors"
    
    # 3. Target validation
    for key in ["cd", "drag_area", "cd_raw", "drag_area_raw"]:
        assert key in targets, f"Missing key '{key}' in target dict!"
        assert isinstance(targets[key], torch.Tensor), f"Target '{key}' should be a Tensor"
        
    # 4. Check normalization math
    import json
    with open(scales_path, "r") as f:
        scales = json.load(f)
        
    raw_cd = targets["cd_raw"].item()
    norm_cd = targets["cd"].item()
    
    expected_norm_cd = (raw_cd - scales["cd"]["mean"]) / scales["cd"]["std"]
    assert np.allclose(norm_cd, expected_norm_cd, atol=1e-5), "Normalized Cd math mismatch!"


def test_vae_forward_shapes():
    """
    Test that the VAE Encoder, Decoder, and full network produce correct shapes.
    """
    import torch
    from src.models.vae import PointNetVAE
    
    batch_size = 4
    channels = 6
    num_points = 2048
    latent_dim = 128
    
    dummy_input = torch.rand(batch_size, channels, num_points)
    
    model = PointNetVAE(in_channels=channels, latent_dim=latent_dim, num_points=num_points)
    
    recon_x, mu, logvar = model(dummy_input)
    
    assert recon_x.shape == (batch_size, 3, num_points), f"Expected recon shape {(batch_size, 3, num_points)}, got {recon_x.shape}"
    assert mu.shape == (batch_size, latent_dim), f"Expected mu shape {(batch_size, latent_dim)}, got {mu.shape}"
    assert logvar.shape == (batch_size, latent_dim), f"Expected logvar shape {(batch_size, latent_dim)}, got {logvar.shape}"


def test_chamfer_distance_loss():
    """
    Test that the custom Chamfer Distance calculation functions correctly.
    """
    import torch
    from src.models.vae import chamfer_distance
    
    # Generate two identical point clouds (offset a bit for testing non-zero value)
    p1 = torch.ones(2, 3, 100)
    p2 = torch.ones(2, 3, 100)
    
    # Same point clouds should yield exactly 0.0 distance
    loss_self = chamfer_distance(p1, p2)
    assert np.allclose(loss_self.item(), 0.0, atol=1e-5), f"Expected 0.0 distance for identical clouds, got {loss_self.item()}"
    
    # Displaced point cloud: p3 = p1 + 1.0. Distance should be exactly 6.0 (squared offset of 3.0 per point, summed symmetrically)
    p3 = p1 + 1.0
    loss_displaced = chamfer_distance(p1, p3)
    assert np.allclose(loss_displaced.item(), 6.0, atol=1e-5), f"Expected 6.0 distance, got {loss_displaced.item()}"


def test_pytorch_occupancy_dataset():
    """
    Test that the VehicleOccupancyDataset loads data correctly, downsamples properly,
    and returns valid tensor types and shapes.
    """
    from src.dataset import VehicleOccupancyDataset
    import torch
    
    metadata_path = Path("metadata/metadata.csv")
    scales_path = Path("metadata/target_scales.json")
    
    if not metadata_path.exists() or not scales_path.exists():
        pytest.skip("Metadata or scales file missing.")
        
    dataset = VehicleOccupancyDataset(
        csv_path=str(metadata_path),
        scales_path=str(scales_path),
        split="val",
        num_points=1024,
        num_query_points=512,
        normalize_targets=True
    )
    
    # Check length
    assert len(dataset) == 10, f"Expected 10 validation samples, got {len(dataset)}"
    
    # Retrieve single item. In val split, first entry is index 0.
    # Since background preprocessing is running, we make sure it has at least one processed file.
    # We will use index 0 which corresponds to F_S_WWC_WM_001.
    try:
        features, query_pts, occupancy, targets = dataset[0]
        
        # Verify shapes
        assert features.shape == (6, 1024), f"Expected PC features shape (6, 1024), got {features.shape}"
        assert query_pts.shape == (512, 3), f"Expected query points shape (512, 3), got {query_pts.shape}"
        assert occupancy.shape == (512,), f"Expected occupancy shape (512,), got {occupancy.shape}"
        
        assert isinstance(features, torch.Tensor)
        assert isinstance(query_pts, torch.Tensor)
        assert isinstance(occupancy, torch.Tensor)
        
        # Values checks
        assert (occupancy >= 0.0).all() and (occupancy <= 1.0).all(), "Occupancy labels must be 0 or 1"
        assert (query_pts >= -0.6).all() and (query_pts <= 0.6).all(), "Query points coordinates out of expected boundaries"
    except FileNotFoundError:
        pytest.skip("Preprocessing files are not fully generated for validation set index 0.")


def test_triplane_vae_forward_shapes():
    """
    Test that the TriplaneVAE modules (PointNetEncoder, TriplaneDecoder, OccupancyMLP)
    and the integrated TriplaneVAE produce correct output shapes.
    """
    import torch
    from src.models.triplane import TriplaneVAE
    
    batch_size = 2
    in_channels = 6
    num_pc_points = 512
    num_query_pts = 128
    latent_dim = 256
    
    dummy_pc = torch.rand(batch_size, in_channels, num_pc_points)
    dummy_query = torch.rand(batch_size, num_query_pts, 3) - 0.5 # Centered within [-0.5, 0.5]
    
    model = TriplaneVAE(
        in_channels=in_channels,
        latent_dim=latent_dim,
        plane_channels=8,
        plane_resolution=32
    )
    
    logits, mu, logvar = model(dummy_pc, dummy_query)
    
    assert logits.shape == (batch_size, num_query_pts), f"Expected logits shape {(batch_size, num_query_pts)}, got {logits.shape}"
    assert mu.shape == (batch_size, latent_dim), f"Expected mu shape {(batch_size, latent_dim)}, got {mu.shape}"
    assert logvar.shape == (batch_size, latent_dim), f"Expected logvar shape {(batch_size, latent_dim)}, got {logvar.shape}"




