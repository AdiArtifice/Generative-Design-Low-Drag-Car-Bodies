#!/usr/bin/env python
"""
Triplane VAE Training Script (Phase 4, Step 2)
----------------------------------------------
Trains the Triplane VAE to learn a continuous latent shape space and occupancy field.
Optimized using BCE (reconstruction) + KL divergence (regularization) loss.
Supports a `--smoke-test` mode to run locally on CPU with minimal samples.

Usage:
    python scripts/train_triplane.py --epochs 20 --batch_size 4 --beta 0.01
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import VehicleOccupancyDataset
from src.models.triplane import TriplaneVAE

def parse_args():
    parser = argparse.ArgumentParser(description="Train Triplane VAE for Implicit Occupancy Fields")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent space dimension z")
    parser.add_argument("--plane_channels", type=int, default=16, help="Channels per triplane")
    parser.add_argument("--plane_res", type=int, default=64, help="Resolution of each triplane")
    parser.add_argument("--beta", type=float, default=0.005, help="KL Divergence weight hyperparameter")
    parser.add_argument("--num_points", type=int, default=2048, help="Number of points in input cloud")
    parser.add_argument("--num_query_points", type=int, default=2048, help="Number of query points to load")
    parser.add_argument("--smoke_test", action="store_true", help="Run a fast local test on CPU")
    return parser.parse_args()

def train_epoch(model, dataloader, optimizer, device, beta):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_acc = 0.0
    num_samples = 0
    
    for batch_idx, (pc, query_pts, occupancy, _) in enumerate(dataloader):
        pc = pc.to(device)
        query_pts = query_pts.to(device)
        occupancy = occupancy.to(device) # [B, N_q]
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, mu, logvar = model(pc, query_pts) # logits: [B, N_q]
        
        # 1. Reconstruction Loss: Binary Cross Entropy
        recon_loss = F.binary_cross_entropy_with_logits(logits, occupancy, reduction='mean')
        
        # 2. KL Divergence: sum over latent dimensions, mean over batch
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        
        # Combined Loss
        loss = recon_loss + beta * kl_loss
        
        loss.backward()
        optimizer.step()
        
        # Compute accuracy (inside/outside classification accuracy)
        preds = (logits > 0).float()
        acc = (preds == occupancy).float().mean()
        
        batch_size = pc.size(0)
        total_loss += loss.item() * batch_size
        total_recon += recon_loss.item() * batch_size
        total_kl += kl_loss.item() * batch_size
        total_acc += acc.item() * batch_size
        num_samples += batch_size
        
    return total_loss / num_samples, total_recon / num_samples, total_kl / num_samples, total_acc / num_samples

def validate(model, dataloader, device, beta):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_acc = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for pc, query_pts, occupancy, _ in dataloader:
            pc = pc.to(device)
            query_pts = query_pts.to(device)
            occupancy = occupancy.to(device)
            
            logits, mu, logvar = model(pc, query_pts)
            
            recon_loss = F.binary_cross_entropy_with_logits(logits, occupancy, reduction='mean')
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
            loss = recon_loss + beta * kl_loss
            
            preds = (logits > 0).float()
            acc = (preds == occupancy).float().mean()
            
            batch_size = pc.size(0)
            total_loss += loss.item() * batch_size
            total_recon += recon_loss.item() * batch_size
            total_kl += kl_loss.item() * batch_size
            total_acc += acc.item() * batch_size
            num_samples += batch_size
            
    return total_loss / num_samples, total_recon / num_samples, total_kl / num_samples, total_acc / num_samples

def main():
    args = parse_args()
    
    if args.smoke_test:
        print("=" * 60)
        print("                 SMOKE-TEST MODE ACTIVE")
        print("=" * 60)
        args.epochs = 2
        args.batch_size = 2
        args.num_points = 512
        args.num_query_points = 512
        
    print(f"Config: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr}, Beta={args.beta}")
    print(f"Model: LatentDim={args.latent_dim}, PlaneChannels={args.plane_channels}, PlaneRes={args.plane_res}")
    
    # Setup Device
    if args.smoke_test:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")
    
    # 1. Initialize Datasets
    print("Loading datasets...")
    train_dataset = VehicleOccupancyDataset(
        split="train", 
        num_points=args.num_points, 
        num_query_points=args.num_query_points,
        normalize_targets=True
    )
    val_dataset = VehicleOccupancyDataset(
        split="val", 
        num_points=args.num_points, 
        num_query_points=args.num_query_points,
        normalize_targets=True
    )
    
    # If in smoke test mode, restrict dataframe rows to first 2 samples
    # to avoid loading files that haven't been preprocessed yet.
    if args.smoke_test:
        train_dataset.df = train_dataset.df.head(2).reset_index(drop=True)
        val_dataset.df = val_dataset.df.head(2).reset_index(drop=True)
        print(f"Smoke-test: limited dataset to {len(train_dataset)} samples.")
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Initialize Model, Optimizer
    model = TriplaneVAE(
        in_channels=6,
        latent_dim=args.latent_dim,
        plane_channels=args.plane_channels,
        plane_resolution=args.plane_res
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    os.makedirs("models", exist_ok=True)
    checkpoint_path = "models/triplane_vae_best.pth"
    
    # 3. Training Loop
    history = {
        "train_loss": [], "val_loss": [],
        "train_recon": [], "val_recon": [],
        "train_kl": [], "val_kl": [],
        "train_acc": [], "val_acc": []
    }
    
    best_val_loss = float('inf')
    
    print("\nStarting training loop...")
    for epoch in range(args.epochs):
        train_loss, train_recon, train_kl, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            beta=args.beta
        )
        
        val_loss, val_recon, val_kl, val_acc = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            beta=args.beta
        )
        
        scheduler.step()
        
        # Save metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_recon"].append(train_recon)
        history["val_recon"].append(val_recon)
        history["train_kl"].append(train_kl)
        history["val_kl"].append(val_kl)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}, Acc: {train_acc:.4f}) | "
              f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f}, Acc: {val_acc:.4f})")
              
        if val_loss < best_val_loss and not args.smoke_test:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved new best model checkpoint to {checkpoint_path}")
            
    # Save the last model in case of smoke test
    if args.smoke_test:
        torch.save(model.state_dict(), "models/triplane_vae_smoke.pth")
        print("  --> Saved smoke-test model checkpoint to models/triplane_vae_smoke.pth")
        
    # 4. Plot curves
    os.makedirs("metadata", exist_ok=True)
    plt.figure(figsize=(15, 5))
    
    # Loss Curve
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Total Loss (BCE + KL)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Recon vs KL Curve
    plt.subplot(1, 3, 2)
    plt.plot(history["train_recon"], label="Train Recon (BCE)")
    plt.plot(history["val_recon"], label="Val Recon (BCE)")
    plt.plot(history["train_kl"], label="Train KL", linestyle="--")
    plt.plot(history["val_kl"], label="Val KL", linestyle="--")
    plt.title("Recon vs KL Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Component")
    plt.legend()
    
    # Accuracy Curve
    plt.subplot(1, 3, 3)
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.title("Boundary Classification Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plot_name = "triplane_training_smoke.png" if args.smoke_test else "triplane_training.png"
    plot_path = f"metadata/{plot_name}"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"\nTraining curves saved to: {plot_path}")
    
    # Save training history
    history_name = "triplane_history_smoke.json" if args.smoke_test else "triplane_history.json"
    history_path = f"metadata/{history_name}"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training history metrics saved to: {history_path}")
    print("--- Training Script Completed Successfully ---")

if __name__ == "__main__":
    main()
