#!/usr/bin/env python
"""
Implicit Occupancy Field Mesh Reconstruction (Phase 4, Step 2)
--------------------------------------------------------------
Loads a trained Triplane VAE model and a vehicle point cloud, evaluates the learned
occupancy field on a dense 3D grid, and extracts a watertight STL mesh using Marching Cubes.

Usage:
    python scripts/generate_mesh.py --model_path models/triplane_vae_smoke.pth --grid_res 64
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from skimage.measure import marching_cubes

# Import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.triplane import TriplaneVAE

def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct watertight mesh from point cloud using Triplane VAE.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/triplane_vae_smoke.pth",
        help="Path to the trained Triplane VAE model checkpoint"
    )
    parser.add_argument(
        "--pc_path",
        type=str,
        default="pointclouds/fastback_smooth_wheelcovers/F_S_WWC_WM_001_pc.ply",
        help="Path to the input point cloud PLY file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="reconstructed_car.stl",
        help="Path to save the reconstructed watertight STL mesh"
    )
    parser.add_argument(
        "--grid_res",
        type=int,
        default=64,
        help="Resolution of the 3D querying grid (default: 64, i.e., 64x64x64)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Occupancy probability threshold for surface extraction (default: 0.5)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16384,
        help="Batch size for querying grid coordinates (default: 16384)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("        Watertight Mesh Reconstruction Pipeline")
    print("=" * 60)
    print(f"Model path        : {args.model_path}")
    print(f"Point cloud path  : {args.pc_path}")
    print(f"Output STL path   : {args.output_path}")
    print(f"Grid resolution   : {args.grid_res}x{args.grid_res}x{args.grid_res}")
    print(f"Surface threshold : {args.threshold}")
    print("-" * 60)
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")
    
    # 2. Initialize and load model
    if not os.path.exists(args.model_path):
        print(f"[Error] Model checkpoint not found at: {args.model_path}")
        sys.exit(1)
        
    model = TriplaneVAE(in_channels=6, latent_dim=256, plane_channels=16, plane_resolution=64).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
    
    # 3. Load and format point cloud input
    if not os.path.exists(args.pc_path):
        print(f"[Error] Point cloud file not found at: {args.pc_path}")
        sys.exit(1)
        
    pcd = trimesh.load(args.pc_path)
    raw_data = pcd.metadata["_ply_raw"]["vertex"]["data"]
    points = np.stack([raw_data['x'], raw_data['y'], raw_data['z']], axis=-1).astype(np.float32)
    normals = np.stack([raw_data['nx'], raw_data['ny'], raw_data['nz']], axis=-1).astype(np.float32)
    
    # Downsample / sample exactly 2048 points
    num_points = len(points)
    target_num = 2048
    if num_points >= target_num:
        indices = np.random.choice(num_points, target_num, replace=False)
    else:
        indices = np.random.choice(num_points, target_num, replace=True)
        
    points = points[indices]
    normals = normals[indices]
    
    # Concatenate shape [1, 6, 2048]
    features = np.concatenate([points, normals], axis=1) # [2048, 6]
    features_tensor = torch.tensor(features, dtype=torch.float32).t().unsqueeze(0).to(device) # [1, 6, 2048]
    print(f"Loaded point cloud: {num_points} points downsampled to {target_num}.")
    
    # 4. Generate dense grid coordinates
    # Mesh boundaries are centered at (0,0,0) and normalized within [-0.5, 0.5].
    # We query from -0.6 to 0.6 to capture the boundaries cleanly.
    min_val, max_val = -0.6, 0.6
    x = np.linspace(min_val, max_val, args.grid_res, dtype=np.float32)
    y = np.linspace(min_val, max_val, args.grid_res, dtype=np.float32)
    z = np.linspace(min_val, max_val, args.grid_res, dtype=np.float32)
    
    # Create grid mesh
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    grid_coords = np.stack([xv, yv, zv], axis=-1) # shape: [grid_res, grid_res, grid_res, 3]
    flat_coords = grid_coords.reshape(-1, 3) # shape: [N_total, 3]
    total_grid_points = len(flat_coords)
    
    print(f"Generated dense query grid of {total_grid_points:,} points.")
    
    # 5. Model Inference (forward pass)
    with torch.no_grad():
        # Encode point cloud
        mu, logvar = model.encoder(features_tensor)
        z_latent = mu # Deterministic reconstruction (mean)
        
        # Decode latent to planes
        plane_xy, plane_xz, plane_yz = model.decoder(z_latent)
        
        print("Evaluating implicit occupancy field...", end="", flush=True)
        
        # Query occupancy in batches
        occupancies = []
        for i in range(0, total_grid_points, args.batch_size):
            batch_coords = flat_coords[i : i + args.batch_size]
            batch_tensor = torch.tensor(batch_coords, dtype=torch.float32).unsqueeze(0).to(device) # [1, Batch, 3]
            
            # Project to planes
            grid_xy = (batch_tensor[..., [0, 1]] * 2.0).unsqueeze(2) # [1, Batch, 1, 2]
            grid_xz = (batch_tensor[..., [0, 2]] * 2.0).unsqueeze(2) # [1, Batch, 1, 2]
            grid_yz = (batch_tensor[..., [1, 2]] * 2.0).unsqueeze(2) # [1, Batch, 1, 2]
            
            feat_xy = F.grid_sample(plane_xy, grid_xy, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(-1)
            feat_xz = F.grid_sample(plane_xz, grid_xz, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(-1)
            feat_yz = F.grid_sample(plane_yz, grid_yz, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(-1)
            
            # Compute occupancy logits
            logits = model.occupancy_mlp(feat_xy, feat_xz, feat_yz, batch_tensor) # [1, Batch]
            probs = torch.sigmoid(logits).squeeze(0) # [Batch]
            
            occupancies.append(probs.cpu().numpy())
            
        print(" Done.", flush=True)
        
    # Combine results
    occupancy_flat = np.concatenate(occupancies, axis=0) # [N_total]
    occupancy_grid = occupancy_flat.reshape(args.grid_res, args.grid_res, args.grid_res)
    
    # Apply bounding box mask to eliminate extrapolation artifacts outside the vehicle envelope
    # Normalized cars are centered at (0,0,0) with X in [-0.5, 0.5], Y in [-0.25, 0.25], Z in [-0.18, 0.18]
    mask_x = (x >= -0.5) & (x <= 0.5)
    mask_y = (y >= -0.25) & (y <= 0.25)
    mask_z = (z >= -0.18) & (z <= 0.18)
    mask_3d = mask_x[:, None, None] & mask_y[None, :, None] & mask_z[None, None, :]
    occupancy_grid[~mask_3d] = 0.0
    
    print(f"Occupancy values stats (masked): Min={occupancy_grid.min():.4f}, Max={occupancy_grid.max():.4f}, Mean={occupancy_grid.mean():.4f}")
    
    # 6. Apply Marching Cubes
    # We check if there's a valid isosurface
    if occupancy_grid.max() < args.threshold:
        print(f"[Warning] Maximum occupancy value {occupancy_grid.max():.4f} is less than threshold {args.threshold}.")
        print("No surface can be extracted. Try training the model longer or lowering the threshold.")
        sys.exit(1)
        
    print(f"Running Marching Cubes at threshold level = {args.threshold}...")
    try:
        # Pad the occupancy grid with zeros on all sides to ensure a closed, watertight mesh
        padded_grid = np.pad(occupancy_grid, pad_width=1, mode='constant', constant_values=0.0)
        
        # Marching cubes output vertices coordinates relative to index space (0 to grid_res+1)
        verts, faces, normals, values = marching_cubes(
            volume=padded_grid,
            level=args.threshold
        )
        
        # Map vertices from index space back to physical coordinate space [-0.6, 0.6]
        # Shifting by 1.0 unit to account for the zero padding on each side
        spacing = (max_val - min_val) / (args.grid_res - 1)
        verts_physical = (verts - 1.0) * spacing + min_val
        
        # 7. Create Trimesh object and export
        mesh = trimesh.Trimesh(vertices=verts_physical, faces=faces, vertex_normals=normals)
        
        # Ensure correct normal orientation (positive volume)
        if mesh.volume < 0:
            mesh.invert()
            
        # Save STL file
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        mesh.export(args.output_path)
        
        print("-" * 60)
        print("Mesh Reconstruction Stats:")
        print(f"  - Output file path : {args.output_path}")
        print(f"  - Number of Verts  : {len(verts_physical)}")
        print(f"  - Number of Faces  : {len(faces)}")
        print(f"  - Is Watertight    : {mesh.is_watertight}")
        print(f"  - Is Manifold      : {mesh.is_volume}")
        print(f"  - Volume           : {mesh.volume:.6f}")
        print("=" * 60)
        print("Mesh Generation Completed Successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"[Error] Marching Cubes or Mesh Export failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
