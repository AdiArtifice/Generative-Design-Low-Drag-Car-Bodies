#!/bin/bash
set -e
# train_regressor_cloud.sh
# Cloud execution script for Latent Drag Regressor Training

echo "=================================================="
echo " Latent Drag Regressor Full Training (Phase 5) "
echo "=================================================="

STASH_ROOT=$(pwd)
PROJECT_DIR="${STASH_ROOT}/aerodesign"

cd ${PROJECT_DIR}
echo "[1/2] Installing dependencies..."
python3 -m pip install --no-cache-dir -r requirements_cloud.txt

echo "[2/2] Running Full Regressor Training..."
python3 scripts/train_latent_regressor.py \
    --vae_path models/triplane_vae_best_80.pth \
    --output_suffix 80 \
    --epochs 100

echo "=================================================="
echo " Regressor Training Completed Successfully! "
echo "=================================================="
