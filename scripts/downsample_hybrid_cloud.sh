#!/bin/bash
set -e

echo "=================================================="
echo " Starting Offline Hybrid PC Downsampling on Cloud "
echo "=================================================="

cd "$(dirname "$0")/.."
USER_SITE=$(python3 -c "import site; print(site.getusersitepackages())")
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$USER_SITE:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "Installing requirements..."
python3 -m pip install -r requirements_cloud.txt

echo "Running Parallel Hybrid Downsampling (50k -> 2048)..."
python3 -u scripts/preprocess_pointclouds_hybrid.py --num-points 2048 --workers 8

echo "=================================================="
echo " Offline Hybrid Downsampling Completed! "
echo "=================================================="
