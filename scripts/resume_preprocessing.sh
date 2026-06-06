#!/bin/bash
set -e # Exit on any error

echo "=================================================="
echo " Resuming Cloud Preprocessing Job (F_S_WWC_WM) "
echo "=================================================="

# The stash root is the current directory
# If executed from stash://adityabehera28502187/, this will be the root.
STASH_ROOT=$(pwd)
PROJECT_DIR="${STASH_ROOT}/aerodesign"

cd ${PROJECT_DIR}
echo "[1/3] Installing missing dependencies..."
pip install -r requirements_cloud.txt

echo "[2/3] Sampling point clouds (50,000 points)..."
python scripts/sample_pointcloud.py --input ${PROJECT_DIR}/normalized/F_S_WWC_WM --output ${PROJECT_DIR}/pointclouds/F_S_WWC_WM --num-points 50000

echo "[3/3] Purging raw STL files to free up storage..."
rm -rf ${PROJECT_DIR}/raw_stl/extracted/

echo "=================================================="
echo " Resume Job Successfully Completed! "
echo "=================================================="
