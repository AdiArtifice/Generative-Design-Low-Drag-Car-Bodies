#!/usr/bin/env python
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is in system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import VehiclePointCloudDataset
from src.models.triplane import TriplaneVAE
from src.models.latent_regressor import LatentDragRegressor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")
    
    # 1. Load Datasets
    print("Loading datasets...")
    train_dataset = VehiclePointCloudDataset(
        csv_path="metadata/metadata.csv",
        scales_path="metadata/target_scales.json",
        split="train",
        num_points=2048,
        normalize_targets=False
    )
    val_dataset = VehiclePointCloudDataset(
        csv_path="metadata/metadata.csv",
        scales_path="metadata/target_scales.json",
        split="val",
        num_points=2048,
        normalize_targets=False
    )
    
    if args.smoke_test:
        print("Smoke-test mode: limiting dataset size.")
        train_dataset.df = train_dataset.df.iloc[:4]
        val_dataset.df = val_dataset.df.iloc[:2]
        args.epochs = 2
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 2. Load Pre-trained VAE
    print(f"Loading pre-trained Triplane VAE from {args.vae_path}...")
    vae = TriplaneVAE(in_channels=6, latent_dim=256, plane_channels=16, plane_resolution=64).to(device)
    if not os.path.exists(args.vae_path):
        raise FileNotFoundError(f"Missing {args.vae_path}. Train the Triplane VAE first.")
    
    vae.load_state_dict(torch.load(args.vae_path, map_location=device))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
        
    # 3. Initialize Latent Drag Regressor
    model = LatentDragRegressor(latent_dim=256).to(device)
    criterion = nn.MSELoss()
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 4. Training Loop
    best_val_loss = float('inf')
    early_stop_patience = 20
    epochs_no_improve = 0
    os.makedirs("models", exist_ok=True)
    
    print("\nStarting Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for point_clouds, targets in train_loader:
            point_clouds = point_clouds.to(device)
            target_drags = targets["drag_area"].unsqueeze(1).to(device) # [B, 1]
            
            # Extract latent vector (mu) deterministically
            with torch.no_grad():
                mu, _ = vae.encoder(point_clouds)
                
            optimizer.zero_grad()
            preds = model(mu)
            loss = criterion(preds, target_drags)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * point_clouds.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for point_clouds, targets in val_loader:
                point_clouds = point_clouds.to(device)
                target_drags = targets["drag_area"].unsqueeze(1).to(device)
                
                mu, _ = vae.encoder(point_clouds)
                preds = model(mu)
                loss = criterion(preds, target_drags)
                
                val_loss += loss.item() * point_clouds.size(0)
                
        val_loss /= len(val_dataset)
        
        # Step LR scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | LR: {current_lr:.6f}")
        
        # Save best model and handle early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_path = f"models/latent_regressor_smoke{suffix}.pth" if args.smoke_test else f"models/latent_regressor_best{suffix}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best regressor model checkpoint to {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
            
    print(f"Training Complete. Best Val MSE: {best_val_loss:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--vae_path", type=str, default="models/triplane_vae_best.pth", help="Path to pre-trained VAE weights")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output regressor weights")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick test with tiny dataset")
    args = parser.parse_args()
    
    train(args)
