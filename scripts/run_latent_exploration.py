#!/usr/bin/env python
"""
Latent Space Exploration Script (Phase 4, Step 1)
-------------------------------------------------
Identifies the highest and lowest drag vehicles in the test set,
performs linear latent interpolation (morphing) between them using the VAE,
and saves the side-by-side 3D visualization to metadata/vae_morphing_interpolation.png.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import VehiclePointCloudDataset
from src.models.vae import PointNetVAE

def main():
    print("--- Starting VAE Latent Space Interpolation ---")
    
    # 1. Load Metadata
    metadata_path = "metadata/metadata.csv"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
    metadata = pd.read_csv(metadata_path)
    test_cars = metadata[metadata["split"] == "test"]
    print(f"Total cars in test set: {len(test_cars)}")
    
    if len(test_cars) < 2:
        # Fallback to train if test is empty (should not happen)
        print("Warning: Too few cars in test set, falling back to train split.")
        test_cars = metadata[metadata["split"] == "train"]
        
    # Find lowest and highest drag cars based on drag_area
    low_drag_row = test_cars.loc[test_cars["drag_area"].idxmin()]
    high_drag_row = test_cars.loc[test_cars["drag_area"].idxmax()]
    
    print(f"\nLow-Drag Car Reference:")
    print(f"  ID:         {low_drag_row['id']}")
    print(f"  Cd:         {low_drag_row['cd']:.4f}")
    print(f"  Drag Area:  {low_drag_row['drag_area']:.4f}")
    
    print(f"\nHigh-Drag Car Reference:")
    print(f"  ID:         {high_drag_row['id']}")
    print(f"  Cd:         {high_drag_row['cd']:.4f}")
    print(f"  Drag Area:  {high_drag_row['drag_area']:.4f}")
    
    # 2. Setup Device & Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device.type.upper()}")
    
    # Instantiate dataset for loading targets
    dataset = VehiclePointCloudDataset(
        csv_path=metadata_path,
        scales_path="metadata/target_scales.json",
        split="test",
        num_points=2048,
        normalize_targets=False
    )
    
    # Load VAE model
    vae = PointNetVAE(in_channels=6, latent_dim=128, num_points=2048).to(device)
    model_path = "models/vae_best.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained VAE weights not found at {model_path}. Run training script first.")
        
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()
    print("PointNet-VAE successfully loaded!")
    
    # 3. Extract and Encode Target Cars
    low_drag_idx = dataset.df[dataset.df["id"] == low_drag_row["id"]].index[0]
    high_drag_idx = dataset.df[dataset.df["id"] == high_drag_row["id"]].index[0]
    
    low_features, _ = dataset[low_drag_idx]
    high_features, _ = dataset[high_drag_idx]
    
    # Shape inputs to [1, 6, 2048]
    low_features = low_features.unsqueeze(0).to(device)
    high_features = high_features.unsqueeze(0).to(device)
    
    with torch.no_grad():
        mu_low, _ = vae.encoder(low_features)
        mu_high, _ = vae.encoder(high_features)
        
    print(f"Encoded low-drag car to latent vector mu of shape: {mu_low.shape}")
    print(f"Encoded high-drag car to latent vector mu of shape: {mu_high.shape}")
    
    # 4. Perform Linear Latent Space Interpolation
    num_steps = 5
    alphas = np.linspace(0.0, 1.0, num_steps)
    reconstructed_clouds = []
    
    print(f"\nPerforming interpolation over {num_steps} steps...")
    with torch.no_grad():
        for alpha in alphas:
            z_interp = (1.0 - alpha) * mu_high + alpha * mu_low
            recon_points = vae.decoder(z_interp)
            
            # Reshape output to [2048, 3] for Matplotlib plotting
            points_np = recon_points.squeeze(0).transpose(0, 1).cpu().numpy()
            reconstructed_clouds.append(points_np)
            
    # 5. Save the 3D Scatter Plot Visualization
    print("Generating 3D scatter plots...")
    fig = plt.figure(figsize=(20, 6))
    titles = [f"Alpha={a:.2f}\n({int((1-a)*100)}% High / {int(a*100)}% Low)" for a in alphas]
    
    for idx, points in enumerate(reconstructed_clouds):
        ax = fig.add_subplot(1, num_steps, idx + 1, projection='3d')
        
        # Plot points and color them by Z axis value to represent depth
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        ax.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=0.6)
        
        ax.set_title(titles[idx], fontsize=12)
        
        # Set bounding bounds to keep scale consistent
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(-0.6, 0.6)
        
        # Side profile view of the car body
        ax.view_init(elev=20, azim=-60)
        ax.axis('off')
        
    plt.suptitle("Generative 3D Shape Morphing (High-Drag to Low-Drag Car Body)", fontsize=16, y=0.98)
    plt.tight_layout()
    
    os.makedirs("metadata", exist_ok=True)
    plot_path = "metadata/vae_morphing_interpolation.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Interpolation plot saved successfully to: {plot_path}")
    print("--- Latent Space Exploration Complete ---")

if __name__ == "__main__":
    main()
