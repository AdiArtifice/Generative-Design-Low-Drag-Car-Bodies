#!/usr/bin/env python
"""
PointNet-VAE Training Script (Phase 4, Step 1)
---------------------------------------------
Trains the 3D PointNet Variational Autoencoder to learn a continuous latent space of vehicle shapes.
Uses Chamfer Distance for reconstruction and KL Divergence for latent regularization.

Usage:
    python scripts/train_vae.py --epochs 30 --latent_dim 128 --kl_weight 0.001
"""

import os
import argparse
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import VehiclePointCloudDataset
from src.models.vae import PointNetVAE, chamfer_distance

def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D PointNet VAE")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_points", type=int, default=2048, help="Number of points to sample per mesh")
    parser.add_argument("--latent_dim", type=int, default=128, help="Dimension of latent space")
    parser.add_argument("--kl_weight", type=float, default=0.001, help="Weight factor for KL Divergence loss")
    return parser.parse_args()

def train_epoch(model, dataloader, optimizer, device, kl_weight):
    model.train()
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0
    
    for batch_idx, (features, _) in enumerate(dataloader):
        features = features.to(device)
        input_coords = features[:, :3, :] # Extract input [x, y, z] coordinates
        
        optimizer.zero_grad()
        
        recon_x, mu, logvar = model(features)
        
        # Calculate losses
        recon_loss = chamfer_distance(input_coords, recon_x)
        # KL loss: -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) per sample, then average over batch
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        total_loss = recon_loss + kl_weight * kl_loss
        
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item() * features.size(0)
        running_recon += recon_loss.item() * features.size(0)
        running_kl += kl_loss.item() * features.size(0)
        
    N = len(dataloader.dataset)
    return running_loss / N, running_recon / N, running_kl / N

def evaluate(model, dataloader, device, kl_weight):
    model.eval()
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0
    
    with torch.no_grad():
        for features, _ in dataloader:
            features = features.to(device)
            input_coords = features[:, :3, :]
            
            recon_x, mu, logvar = model(features)
            
            recon_loss = chamfer_distance(input_coords, recon_x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            total_loss = recon_loss + kl_weight * kl_loss
            
            running_loss += total_loss.item() * features.size(0)
            running_recon += recon_loss.item() * features.size(0)
            running_kl += kl_loss.item() * features.size(0)
            
    N = len(dataloader.dataset)
    return running_loss / N, running_recon / N, running_kl / N

def main():
    args = parse_args()
    print("--- Starting PointNet VAE Training ---")
    print(f"Config: Epochs={args.epochs}, Batch Size={args.batch_size}, Points={args.num_points}, "
          f"Latent Dim={args.latent_dim}, KL Weight={args.kl_weight}")
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")
    
    # 1. Datasets & Dataloaders
    print("Loading datasets...")
    train_dataset = VehiclePointCloudDataset(split="train", num_points=args.num_points, normalize_targets=False)
    val_dataset = VehiclePointCloudDataset(split="val", num_points=args.num_points, normalize_targets=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Initialize Model and Optimizer
    model = PointNetVAE(in_channels=6, latent_dim=args.latent_dim, num_points=args.num_points).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Decay learning rate as training progresses
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    os.makedirs("models", exist_ok=True)
    best_model_path = "models/vae_best.pth"
    
    # 3. Training Loop
    history = {
        "train_loss": [], "val_loss": [],
        "train_recon": [], "val_recon": [],
        "train_kl": [], "val_kl": []
    }
    best_val_recon = float('inf')
    
    for epoch in range(args.epochs):
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device, args.kl_weight)
        val_loss, val_recon, val_kl = evaluate(model, val_loader, device, args.kl_weight)
        
        scheduler.step()
        
        # Log metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_recon"].append(train_recon)
        history["val_recon"].append(val_recon)
        history["train_kl"].append(train_kl)
        history["val_kl"].append(val_kl)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.2f}) | "
              f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.2f})")
              
        # Checkpoint based on Reconstruction Quality (Chamfer Distance)
        if val_recon < best_val_recon:
            best_val_recon = val_recon
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best reconstruction model (Val Recon: {best_val_recon:.4f})")
            
    # 4. Save Learning Curves
    os.makedirs("metadata", exist_ok=True)
    plt.figure(figsize=(15, 5))
    
    # Total loss
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Total VAE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Reconstruction loss (Chamfer Distance)
    plt.subplot(1, 3, 2)
    plt.plot(history["train_recon"], label="Train Recon (Chamfer)")
    plt.plot(history["val_recon"], label="Val Recon (Chamfer)")
    plt.title("Reconstruction Loss (Chamfer)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # KL Divergence loss
    plt.subplot(1, 3, 3)
    plt.plot(history["train_kl"], label="Train KL")
    plt.plot(history["val_kl"], label="Val KL")
    plt.title("KL Divergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plot_path = "metadata/vae_training.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"\nTraining curves saved to: {plot_path}")
    print("--- VAE Training Complete ---")

if __name__ == "__main__":
    main()
