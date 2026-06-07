#!/usr/bin/env python
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from skimage.measure import marching_cubes

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.triplane import TriplaneVAE
from src.models.latent_regressor import LatentDragRegressor
from src.dataset import VehiclePointCloudDataset

def extract_mesh(vae, z, output_path, device, grid_res=64, threshold=0.5):
    # Generates dense grid coordinates
    min_val, max_val = -0.6, 0.6
    x = np.linspace(min_val, max_val, grid_res, dtype=np.float32)
    y = np.linspace(min_val, max_val, grid_res, dtype=np.float32)
    z_coords = np.linspace(min_val, max_val, grid_res, dtype=np.float32)
    
    xv, yv, zv = np.meshgrid(x, y, z_coords, indexing='ij')
    grid_coords = np.stack([xv, yv, zv], axis=-1)
    flat_coords = grid_coords.reshape(-1, 3)
    
    # Inference
    with torch.no_grad():
        plane_xy, plane_xz, plane_yz = vae.decoder(z)
        
        batch_size = 16384
        occupancies = []
        for i in range(0, len(flat_coords), batch_size):
            batch_coords = flat_coords[i : i + batch_size]
            batch_tensor = torch.tensor(batch_coords, dtype=torch.float32).unsqueeze(0).to(device)
            
            grid_xy = (batch_tensor[..., [0, 1]] * 2.0).unsqueeze(2)
            grid_xz = (batch_tensor[..., [0, 2]] * 2.0).unsqueeze(2)
            grid_yz = (batch_tensor[..., [1, 2]] * 2.0).unsqueeze(2)
            
            feat_xy = F.grid_sample(plane_xy, grid_xy, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(-1)
            feat_xz = F.grid_sample(plane_xz, grid_xz, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(-1)
            feat_yz = F.grid_sample(plane_yz, grid_yz, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(-1)
            
            logits = vae.occupancy_mlp(feat_xy, feat_xz, feat_yz, batch_tensor)
            probs = torch.sigmoid(logits).squeeze(0)
            occupancies.append(probs.cpu().numpy())
            
    occupancy_flat = np.concatenate(occupancies, axis=0)
    occupancy_grid = occupancy_flat.reshape(grid_res, grid_res, grid_res)
    
    # Mask out-of-bounds
    mask_x = (x >= -0.5) & (x <= 0.5)
    mask_y = (y >= -0.25) & (y <= 0.25)
    mask_z = (z_coords >= -0.18) & (z_coords <= 0.18)
    mask_3d = mask_x[:, None, None] & mask_y[None, :, None] & mask_z[None, None, :]
    occupancy_grid[~mask_3d] = 0.0
    
    if occupancy_grid.max() < threshold:
        print(f"[Warning] Max occupancy {occupancy_grid.max():.4f} is less than threshold.")
        return False
        
    padded_grid = np.pad(occupancy_grid, pad_width=1, mode='constant', constant_values=0.0)
    verts, faces, normals, values = marching_cubes(volume=padded_grid, level=threshold)
    
    spacing = (max_val - min_val) / (grid_res - 1)
    verts_physical = (verts - 1.0) * spacing + min_val
    
    mesh = trimesh.Trimesh(vertices=verts_physical, faces=faces, vertex_normals=normals)
    if mesh.volume < 0:
        mesh.invert()
        
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    mesh.export(output_path)
    return True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def optimize(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")
    
    # 1. Load VAE
    print(f"Loading Triplane VAE from {args.vae_path}...")
    vae = TriplaneVAE(in_channels=6, latent_dim=256, plane_channels=16, plane_resolution=64).to(device)
    if not os.path.exists(args.vae_path):
        print(f"Error: {args.vae_path} not found.")
        sys.exit(1)
    vae.load_state_dict(torch.load(args.vae_path, map_location=device))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
        
    # 2. Load Latent Regressor
    print(f"Loading Latent Drag Regressor from {args.regressor_path}...")
    regressor = LatentDragRegressor(latent_dim=256).to(device)
    if not os.path.exists(args.regressor_path):
        # Fallback to smoke model if default file not found and smoke is available
        if args.regressor_path == "models/latent_regressor_best.pth" and os.path.exists("models/latent_regressor_smoke.pth"):
            print("Warning: Using smoke model as latent_regressor_best.pth is not found.")
            args.regressor_path = "models/latent_regressor_smoke.pth"
        else:
            print(f"Error: {args.regressor_path} not found.")
            sys.exit(1)
            
    regressor.load_state_dict(torch.load(args.regressor_path, map_location=device))
    regressor.eval()
    for param in regressor.parameters():
        param.requires_grad = False
        
    # 3. Load baseline car
    print("Loading dataset to find a baseline high-drag car...")
    dataset = VehiclePointCloudDataset(
        csv_path="metadata/metadata.csv",
        scales_path="metadata/target_scales.json",
        split="test",
        num_points=2048,
        normalize_targets=False
    )
    
    # Find highest drag car in test set
    dataset_idx = dataset.df['drag_area'].argmax()
    row = dataset.df.iloc[dataset_idx]
    baseline_id = row['id']
    
    print(f"Selected baseline car: {baseline_id} with original drag_area = {row['drag_area']:.4f} m^2")
    
    features, targets = dataset[dataset_idx]
    features = features.unsqueeze(0).to(device)
    
    # Extract baseline latent vector (explicitly detached & cloned)
    with torch.no_grad():
        z_initial, _ = vae.encoder(features)
        z_initial = z_initial.detach().clone()
        
    # Initial prediction
    with torch.no_grad():
        initial_pred_drag = regressor(z_initial).item()
        
    print(f"Baseline Predicted Drag Area: {initial_pred_drag:.4f} m^2")
    
    # 4. Optimization Loop
    z_opt = torch.nn.Parameter(z_initial.clone())
    optimizer = torch.optim.Adam([z_opt], lr=args.lr)
    
    os.makedirs("optimization_output", exist_ok=True)
    
    # Export step 0 (baseline)
    print("Exporting initial mesh (Step 0)...")
    success = extract_mesh(vae, z_opt, "optimization_output/optimized_car_step_0.stl", device)
    if not success:
        print("[Warning] Initial mesh reconstruction failed.")
    
    print("\nStarting Latent Space Optimization...")
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        
        pred_drag = regressor(z_opt)
        similarity_penalty = torch.norm(z_opt - z_initial, p=2)
        
        # Loss formula
        loss = pred_drag + args.lambda_reg * similarity_penalty
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0 or step == 1:
            print(f"Step {step:03d} | Loss: {loss.item():.4f} | Drag: {pred_drag.item():.4f} m^2 | Penalty: {similarity_penalty.item():.4f}")
            
        if step % 50 == 0 or step == args.steps:
            output_path = f"optimization_output/optimized_car_step_{step}.stl"
            print(f"  -> Exporting intermediate mesh: {output_path}")
            success = extract_mesh(vae, z_opt, output_path, device, grid_res=64, threshold=0.5)
            if not success:
                print(f"[Warning] Mesh reconstruction failed at step {step}.")
            
    # Calculate final reduction in raw physical units
    with torch.no_grad():
        final_pred_drag = regressor(z_opt).item()
        reduction = (initial_pred_drag - final_pred_drag) / initial_pred_drag * 100 if initial_pred_drag != 0 else 0
        
    print("\nOptimization Complete!")
    print(f"Final Predicted Drag Area: {final_pred_drag:.4f} m^2 (Baseline: {initial_pred_drag:.4f} m^2)")
    print(f"Theoretical Drag Reduction: {reduction:.2f}%")
    print(f"Check the 'optimization_output' folder for STL files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250, help="Number of optimization steps")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for Adam optimizer")
    parser.add_argument("--lambda_reg", type=float, default=0.1, help="L2 penalty weight to preserve core structure")
    parser.add_argument("--vae_path", type=str, default="models/triplane_vae_best.pth", help="Path to pre-trained VAE weights")
    parser.add_argument("--regressor_path", type=str, default="models/latent_regressor_best.pth", help="Path to trained regressor weights")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    args = parser.parse_args()
    
    optimize(args)
