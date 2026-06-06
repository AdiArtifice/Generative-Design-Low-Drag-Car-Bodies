#!/bin/bash
set -e # Exit on any error

echo "=================================================="
echo " Starting Cloud Preprocessing Job (F_S_WWC_WM) "
echo "=================================================="

# Print current directory to debug
pwd
ls -la

# The stash root is the current directory
STASH_ROOT=$(pwd)
PROJECT_DIR="${STASH_ROOT}/aerodesign"

# 1. Setup environment
cd ${PROJECT_DIR}
echo "[1/5] Installing dependencies..."
pip install -r requirements_cloud.txt

# Create necessary directories inside the writeable project folder
mkdir -p ${PROJECT_DIR}/raw_stl/extracted
mkdir -p ${PROJECT_DIR}/normalized/F_S_WWC_WM
mkdir -p ${PROJECT_DIR}/pointclouds/F_S_WWC_WM

# 2. Unzip the dataset to the writeable directory
echo "[2/5] Unzipping dataset..."
unzip -q -o ${STASH_ROOT}/drivaer_dataset/raw_stl/F_S_WWC_WM.zip -d ${PROJECT_DIR}/raw_stl/extracted/

# Find the exact folder dynamically
RAW_STL_DIR=$(find ${PROJECT_DIR}/raw_stl/extracted/ -type d -name "F_S_WWC_WM" | head -n 1)

if [ -z "$RAW_STL_DIR" ]; then
    echo "Error: Could not find F_S_WWC_WM directory after extraction!"
    exit 1
fi
echo "Found raw STLs at: $RAW_STL_DIR"

# 3. Normalize Meshes
echo "[3/5] Normalizing meshes..."
python scripts/normalize_mesh.py --input "$RAW_STL_DIR" --output ${PROJECT_DIR}/normalized/F_S_WWC_WM

# 4. Sample Point Clouds (50,000 points)
echo "[4/5] Sampling point clouds..."
python scripts/sample_pointcloud.py --input ${PROJECT_DIR}/normalized/F_S_WWC_WM --output ${PROJECT_DIR}/pointclouds/F_S_WWC_WM --num-points 50000

# 5. Process & Purge
echo "[5/5] Purging raw STL files to free up storage..."
rm -rf ${PROJECT_DIR}/raw_stl/extracted/

echo "=================================================="
echo " Preprocessing Job Successfully Completed! "
echo "=================================================="
