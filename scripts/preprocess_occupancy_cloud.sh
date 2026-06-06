#!/bin/bash
set -e # Exit on any error

echo "=================================================="
echo " Cloud Occupancy Grid Preprocessing (F_S_WWC_WM) "
echo "=================================================="

# The stash root is the current directory
STASH_ROOT=$(pwd)
PROJECT_DIR="${STASH_ROOT}/aerodesign"

cd ${PROJECT_DIR}
echo "[1/3] Installing dependencies..."
pip install -r requirements_cloud.txt

echo "[2/3] Generating Occupancy Grids (Open3D)..."
# Pass --input to bypass metadata.csv and process all 692 normalized STLs directly
python scripts/preprocess_occupancy.py --input ${PROJECT_DIR}/normalized/F_S_WWC_WM --output ${PROJECT_DIR}/occupancy/F_S_WWC_WM --num-points 2048

echo "[3/3] Purging Normalized STL files to free up ~48 GB of storage..."
rm -rf ${PROJECT_DIR}/normalized/

echo "=================================================="
echo " Occupancy Job Successfully Completed! "
echo "=================================================="
