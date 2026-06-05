#!/usr/bin/env python
"""
Tabular Baseline Benchmark (Phase 1)
------------------------------------
This script trains Random Forest and Gradient Boosting regressors on the 23 design parameters
and 6 computed geometric features from metadata.csv to predict Cd and Drag Area.
It respects the pre-allocated Train/Val/Test splits, computes R2 and MAE,
and exports feature importance plots.

Usage:
    python scripts/tabular_baseline.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def main():
    # 1. Load consolidated metadata
    metadata_path = "metadata/metadata.csv"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Consolidated metadata not found at {metadata_path}. Please run link_metadata.py first.")
    
    df = pd.read_csv(metadata_path)
    print(f"Loaded master metadata with {len(df)} samples.")
    
    # 2. Identify features and targets
    # Geometric features
    geo_features = ["frontal_area", "convex_hull_volume", "bbox_volume", "length_x", "width_y", "height_z"]
    
    # Design parameters (columns starting with prefix codes A_, B_, C_, D_, E_, F_, G_, H_)
    design_prefixes = ("A_", "B_", "C_", "D_", "E_", "F_", "G_", "H_")
    design_features = [col for col in df.columns if col.startswith(design_prefixes) and col not in ["A_Car_Length", "A_Car_Width"]]
    # Note: df has shape parameters like A_Car_Length, A_Car_Width, A_Car_Roof_Height, etc. 
    # Let's verify and keep all columns that start with prefix and are numeric
    design_features = [col for col in df.columns if col.startswith(design_prefixes) and pd.api.types.is_numeric_dtype(df[col])]
    
    features = geo_features + design_features
    print(f"Using {len(features)} features: {len(geo_features)} geometric + {len(design_features)} shape parameters.")
    print("Features:", features)
    
    targets = ["cd", "drag_area"]
    
    # 3. Split dataset based on pre-allocated split column
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    
    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    X_train, y_train = train_df[features], train_df[targets]
    X_val, y_val = val_df[features], val_df[targets]
    X_test, y_test = test_df[features], test_df[targets]
    
    # 4. Train and evaluate models
    results = {}
    
    models = {
        "RandomForest": lambda: RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        "GradientBoosting": lambda: GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    }
    
    # Create directory for output plots if not exists
    os.makedirs("metadata", exist_ok=True)
    
    fig, axes = plt.subplots(len(targets), len(models), figsize=(14, 10))
    fig.suptitle("Feature Importance Benchmark (Tabular Baseline)", fontsize=16, fontweight='bold')
    
    for t_idx, target in enumerate(targets):
        results[target] = {}
        for m_idx, (model_name, model_fn) in enumerate(models.items()):
            print(f"\n--- Training {model_name} for target: {target} ---")
            
            # Fit model
            model = model_fn()
            model.fit(X_train, y_train[target])
            
            # Predictions
            pred_train = model.predict(X_train)
            pred_val = model.predict(X_val)
            pred_test = model.predict(X_test)
            
            # Metrics
            metrics = {
                "train_r2": float(r2_score(y_train[target], pred_train)),
                "train_mae": float(mean_absolute_error(y_train[target], pred_train)),
                "val_r2": float(r2_score(y_val[target], pred_val)),
                "val_mae": float(mean_absolute_error(y_val[target], pred_val)),
                "test_r2": float(r2_score(y_test[target], pred_test)),
                "test_mae": float(mean_absolute_error(y_test[target], pred_test))
            }
            results[target][model_name] = metrics
            
            print(f"Train R2: {metrics['train_r2']:.4f} | MAE: {metrics['train_mae']:.6f}")
            print(f"Val   R2: {metrics['val_r2']:.4f} | MAE: {metrics['val_mae']:.6f}")
            print(f"Test  R2: {metrics['test_r2']:.4f} | MAE: {metrics['test_mae']:.6f}")
            
            # Plot Feature Importance
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Select top 12 features for visibility
            top_n = min(12, len(features))
            top_indices = indices[:top_n]
            top_importances = importances[top_indices]
            top_names = [features[i] for i in top_indices]
            
            ax = axes[t_idx, m_idx]
            y_pos = np.arange(top_n)
            ax.barh(y_pos, top_importances[::-1], align='center', color='royalblue', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_names[::-1], fontsize=9)
            ax.set_xlabel('Relative Importance')
            ax.set_title(f"{model_name} - {target.upper()}")
            
    plt.tight_layout()
    plot_path = "metadata/feature_importance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nFeature importance plots saved to: {plot_path}")
    
    # 5. Export metrics to json
    metrics_path = "metadata/tabular_baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Baseline metrics exported to: {metrics_path}")

if __name__ == "__main__":
    main()
