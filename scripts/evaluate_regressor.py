#!/usr/bin/env python
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is in system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import VehiclePointCloudDataset
from src.models.triplane import TriplaneVAE
from src.models.latent_regressor import LatentDragRegressor

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")
    
    # 1. Load Dataset
    print("Loading validation dataset...")
    val_dataset = VehiclePointCloudDataset(
        csv_path="metadata/metadata.csv",
        scales_path="metadata/target_scales.json",
        split="val",
        num_points=2048,
        normalize_targets=False
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 2. Load Models
    vae_path = "models/triplane_vae_best_80.pth"
    regressor_path = "models/latent_regressor_best_80.pth"
    
    print(f"Loading Triplane VAE from {vae_path}...")
    vae = TriplaneVAE(in_channels=6, latent_dim=256, plane_channels=16, plane_resolution=64).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()
    
    print(f"Loading Latent Drag Regressor from {regressor_path}...")
    regressor = LatentDragRegressor(latent_dim=256).to(device)
    regressor.load_state_dict(torch.load(regressor_path, map_location=device))
    regressor.eval()
    
    # 3. Evaluation Loop
    y_true = []
    y_pred = []
    
    print("Running inference on validation set...")
    with torch.no_grad():
        for point_clouds, targets in val_loader:
            point_clouds = point_clouds.to(device)
            target_drags = targets["drag_area"].numpy()
            
            mu, _ = vae.encoder(point_clouds)
            preds = regressor(mu).squeeze(1).cpu().numpy()
            
            y_true.extend(target_drags)
            y_pred.extend(preds)
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 4. Calculate Metrics
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # R2 Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Pearson Correlation
    pearson = np.corrcoef(y_true, y_pred)[0, 1]
    
    print("\n" + "="*50)
    print("         REGRESSOR VALIDATION PERFORMANCE")
    print("="*50)
    print(f"Validation Samples : {len(y_true)}")
    print(f"Mean Squared Error (MSE)   : {mse:.8f}")
    print(f"Mean Absolute Error (MAE)  : {mae:.6f} m^2")
    print(f"Mean Absolute % Error (MAPE): {mape:.2f}%")
    print(f"R^2 Score (Variance explained) : {r2:.4f}")
    print(f"Pearson Correlation (r)    : {pearson:.4f}")
    print("="*50)

if __name__ == "__main__":
    evaluate()
