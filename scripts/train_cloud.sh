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

# 2. Cache dataset to local high-speed SSD (/tmp) to eliminate FUSE network latency
echo "Caching dataset to /tmp for ultra-fast I/O..."
mkdir -p /tmp/pointclouds /tmp/occupancy
cp -r pointclouds/* /tmp/pointclouds/ &
cp -r occupancy/* /tmp/occupancy/ &
wait
echo "Local dataset caching complete!"

# Ensure models directory has full write permissions
mkdir -p models
chmod 777 models || true

# 3. Run C-VAE Training
echo "Starting Triplane C-VAE Training (128x128)..."
python3 -u scripts/train_triplane.py \
    --epochs 200 \
    --batch_size 64 \
    --plane_res 128 \
    --num_classes 3 \
    --embed_dim 16 \
    --lr 1e-3 \
    --pc_dir /tmp/pointclouds \
    --occupancy_dir /tmp/occupancy \
    --num_workers 2

echo "=================================================="
echo " Training Job Complete! "
echo "=================================================="
