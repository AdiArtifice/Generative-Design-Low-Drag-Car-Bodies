#!/bin/bash
set -e

echo "=================================================="
echo " Cloud Master Metadata Generation (Phase 3) "
echo "=================================================="

STASH_ROOT=$(pwd)
PROJECT_DIR="${STASH_ROOT}/aerodesign"
EXCEL_FILE="${STASH_ROOT}/DrivAerNet_ParametricData (2).xlsx"

cd ${PROJECT_DIR}
echo "[1/3] Installing dependencies..."
pip install -r requirements_cloud.txt
pip install openpyxl  # Required for pandas to read excel files

echo "[2/3] Extracting Point Cloud features..."
python scripts/compute_features.py --input pointclouds/F_S_WWC_WM --output metadata/computed_features.csv

echo "[3/3] Fusing CFD drag values from Excel and linking master metadata..."
python scripts/link_metadata.py --features-csv metadata/computed_features.csv --excel-file "${EXCEL_FILE}" --output-csv metadata/metadata.csv --output-json metadata/target_scales.json

echo "=================================================="
echo " Metadata Generation Completed Successfully! "
echo "=================================================="
