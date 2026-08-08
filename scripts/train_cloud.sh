#!/bin/bash
set -e # Exit on any error

echo "=================================================="
echo " Starting C-VAE Training Job on Camber Cloud"
echo "=================================================="

# 1. Setup environment
cd "$(dirname "$0")/.."
echo "Current directory: $(pwd)"

USER_SITE=$(python3 -c "import site; print(site.getusersitepackages())")
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$USER_SITE:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "Python version: $(python3 --version)"
echo "User site-packages: $USER_SITE"

echo "Installing requirements..."
python3 -m pip install -r requirements_cloud.txt

# 2. Run C-VAE Training
echo "Starting Triplane C-VAE Training..."
python3 -u scripts/train_triplane.py \
    --epochs 200 \
    --batch_size 64 \
    --num_classes 3 \
    --embed_dim 16

echo "=================================================="
echo " Training Job Complete! "
echo "=================================================="
