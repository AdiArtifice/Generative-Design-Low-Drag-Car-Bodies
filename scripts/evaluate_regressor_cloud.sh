#!/bin/bash
set -e
# evaluate_regressor_cloud.sh
# Cloud execution script for evaluating Latent Drag Regressor

echo "=================================================="
echo " Latent Drag Regressor Full Cloud Evaluation "
echo "=================================================="

STASH_ROOT=$(pwd)
PROJECT_DIR="${STASH_ROOT}/aerodesign"

cd ${PROJECT_DIR}
echo "[1/2] Installing dependencies..."
python3 -m pip install --no-cache-dir -r requirements_cloud.txt

echo "[2/2] Running Full Validation Evaluation..."
python3 scripts/evaluate_regressor.py

echo "=================================================="
echo " Evaluation Completed Successfully! "
echo "=================================================="
