#!/bin/bash
set -e

echo "=================================================="
echo " Triplane VAE 1-Epoch Test Run (Phase 4) "
echo "=================================================="

STASH_ROOT=$(pwd)
PROJECT_DIR="${STASH_ROOT}/aerodesign"

cd ${PROJECT_DIR}
echo "[1/2] Installing dependencies..."
python3 -m pip install --no-cache-dir -r requirements_cloud.txt

echo "[2/2] Running 1-Epoch VAE Training test..."
python3 scripts/train_triplane.py --epochs 1 --batch_size 4 --lr 1e-3 --beta 0.005

echo "=================================================="
echo " VAE Training Test Completed Successfully! "
echo "=================================================="
