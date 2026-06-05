#!/usr/bin/env python
"""
PointNet Training Script (Phase 3)
----------------------------------
Trains the 3D PointNet regressor to predict aerodynamic metrics (drag_area) directly from point clouds.
Includes CPU optimizations, model checkpointing, and final evaluation against the tabular baseline.

Usage:
    python scripts/train_pointnet.py --epochs 30 --target drag_area --num_points 2048
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# Import custom modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import VehiclePointCloudDataset
from src.models.pointnet import PointNetRegressor

def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D PointNet Regressor")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_points", type=int, default=2048, help="Number of points to sample per mesh")
    parser.add_argument("--target", type=str, default="drag_area", choices=["cd", "drag_area"], help="Target to predict")
    return parser.parse_args()

def train(model, dataloader, criterion, optimizer, device, target_name):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    for batch_idx, (features, targets) in enumerate(dataloader):
        features = features.to(device)
        # We predict the normalized targets
        y = targets[target_name].to(device)
        
        optimizer.zero_grad()
        preds = model(features)
        
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * features.size(0)
        
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(y.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_r2 = r2_score(all_targets, all_preds)
    return epoch_loss, epoch_r2

def evaluate(model, dataloader, criterion, device, target_name):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            y = targets[target_name].to(device)
            
            preds = model(features)
            loss = criterion(preds, y)
            
            running_loss += loss.item() * features.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_r2 = r2_score(all_targets, all_preds)
    epoch_mae = mean_absolute_error(all_targets, all_preds)
    return epoch_loss, epoch_r2, epoch_mae, all_preds, all_targets

def main():
    args = parse_args()
    print(f"--- Starting PointNet Training for target: {args.target} ---")
    print(f"Config: Epochs={args.epochs}, Batch Size={args.batch_size}, Points={args.num_points}, LR={args.lr}")
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")
    
    # 1. Initialize Datasets and Dataloaders
    print("Loading datasets...")
    train_dataset = VehiclePointCloudDataset(split="train", num_points=args.num_points, normalize_targets=True)
    val_dataset = VehiclePointCloudDataset(split="val", num_points=args.num_points, normalize_targets=True)
    test_dataset = VehiclePointCloudDataset(split="test", num_points=args.num_points, normalize_targets=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Initialize Model, Loss, Optimizer
    model = PointNetRegressor(in_channels=6).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Simple learning rate scheduler to drop LR by half every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    os.makedirs("models", exist_ok=True)
    best_model_path = f"models/pointnet_best_{args.target}.pth"
    
    # 3. Training Loop
    history = {"train_loss": [], "val_loss": [], "train_r2": [], "val_r2": []}
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss, train_r2 = train(model, train_loader, criterion, optimizer, device, args.target)
        val_loss, val_r2, _, _, _ = evaluate(model, val_loader, criterion, device, args.target)
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_r2"].append(train_r2)
        history["val_r2"].append(val_r2)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} R2: {train_r2:.4f} | "
              f"Val Loss: {val_loss:.4f} R2: {val_r2:.4f}")
              
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best model (Val Loss: {best_val_loss:.4f})")
            
    # 4. Final Evaluation on Test Set
    print("\n--- Final Evaluation ---")
    # Load best model for testing
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_loss, test_r2, test_mae, test_preds, test_targets = evaluate(model, test_loader, criterion, device, args.target)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test R2:   {test_r2:.4f}")
    print(f"Test MAE:  {test_mae:.4f} (Normalized Scale)")
    
    # Read Baseline metrics for comparison if available
    baseline_path = "metadata/tabular_baseline_metrics.json"
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            baseline = json.load(f)
            rf_r2 = baseline.get(args.target, {}).get("RandomForest", {}).get("test_r2", "N/A")
            gb_r2 = baseline.get(args.target, {}).get("GradientBoosting", {}).get("test_r2", "N/A")
            print(f"\n[Comparison vs Tabular Baseline Test R2]")
            print(f"  Random Forest:     {rf_r2:.4f}")
            print(f"  Gradient Boosting: {gb_r2:.4f}")
            print(f"  3D PointNet:       {test_r2:.4f}")
            
    # 5. Plot Learning Curves
    os.makedirs("metadata", exist_ok=True)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("MSE Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["train_r2"], label="Train R2")
    plt.plot(history["val_r2"], label="Val R2")
    plt.axhline(y=test_r2, color='r', linestyle='--', label=f"Final Test R2 ({test_r2:.2f})")
    plt.title("R² Score over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.legend()
    
    plot_path = f"metadata/pointnet_training_{args.target}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"\nTraining curves saved to: {plot_path}")
    print("--- Training Complete ---")

if __name__ == "__main__":
    main()
