#!/bin/bash
# Sync local codebase & pre-downsampled dataset to NEW Camber Cloud Account Stash
# Account: nidhithakur24 (stash://nidhithakur24/aerodesign)
set -e

# New account API Key
export CAMBER_API_KEY="a28c9e9787456ea8edacffd7f4ad412debf4b08c"

STASH_ROOT="stash://nidhithakur24/aerodesign"
CAMBER_CMD="camber stash cp"

echo "=================================================="
echo " Syncing to NEW Camber Account (nidhithakur24)"
echo " Target: $STASH_ROOT"
echo "=================================================="

# 1. Sync source code
echo "[1/6] Syncing src/ ..."
$CAMBER_CMD -r src $STASH_ROOT/src

# 2. Sync scripts
echo "[2/6] Syncing scripts/ ..."
$CAMBER_CMD -r scripts $STASH_ROOT/scripts

# 3. Sync metadata
echo "[3/6] Syncing metadata/ ..."
$CAMBER_CMD -r metadata $STASH_ROOT/metadata

# 4. Sync requirements & config
echo "[4/6] Syncing requirements & config ..."
$CAMBER_CMD requirements_cloud.txt $STASH_ROOT/requirements_cloud.txt
$CAMBER_CMD .env $STASH_ROOT/.env

# 5. Sync pre-downsampled hybrid point clouds (2048-point files)
#    Upload pointclouds_hybrid/ as pointclouds/ on the stash
#    (metadata.csv references pointclouds/<subdir>/<file>.ply)
echo "[5/6] Syncing hybrid point clouds (pointclouds_hybrid/ -> pointclouds/) ..."
for subdir in pointclouds_hybrid/*/; do
    subdir_name=$(basename "$subdir")
    echo "  Uploading $subdir_name ..."
    $CAMBER_CMD -r "pointclouds_hybrid/$subdir_name" "$STASH_ROOT/pointclouds/$subdir_name"
done

# 6. Sync occupancy data
echo "[6/6] Syncing occupancy/ ..."
for subdir in occupancy/*/; do
    subdir_name=$(basename "$subdir")
    echo "  Uploading $subdir_name ..."
    $CAMBER_CMD -r "occupancy/$subdir_name" "$STASH_ROOT/occupancy/$subdir_name"
done

# 7. Create empty models directory
echo "Creating models/ directory in stash..."
mkdir -p /tmp/empty_models
touch /tmp/empty_models/.gitkeep
$CAMBER_CMD /tmp/empty_models/.gitkeep $STASH_ROOT/models/.gitkeep

echo "=================================================="
echo " Sync to NEW Camber Account Complete!"
echo "=================================================="
