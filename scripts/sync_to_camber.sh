#!/bin/bash
# Sync local codebase to Camber Cloud Stash
set -e

# API Key directly exported
export CAMBER_API_KEY="040b1ec4afff843bddae8cc4665d5aa22007f003"

STASH_ROOT="stash://adityabehera28502187/aerodesign"
CAMBER_CMD="$HOME/.camber/bin/camber stash cp"

echo "=================================================="
echo " Syncing Codebase to Camber Stash"
echo "=================================================="

# Sync required directories and files
$CAMBER_CMD -r src $STASH_ROOT/src
$CAMBER_CMD -r scripts $STASH_ROOT/scripts
$CAMBER_CMD -r metadata $STASH_ROOT/metadata
$CAMBER_CMD requirements_cloud.txt $STASH_ROOT/requirements_cloud.txt
$CAMBER_CMD .env $STASH_ROOT/.env

echo "Sync completed successfully!"
