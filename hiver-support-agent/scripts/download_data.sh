#!/usr/bin/env bash
# Downloads the Kaggle "Customer Support on Twitter" dataset.
# Requires a free Kaggle account + API token (kaggle.json). No cost.
#
# Setup (one-time):
#   1. Create a free account at kaggle.com
#   2. Go to kaggle.com/settings/account -> "Create New Token"
#      This downloads kaggle.json
#   3. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
#      chmod 600 ~/.kaggle/kaggle.json
#   4. pip install kaggle
#
# Then run this script from the repo root:
#   bash scripts/download_data.sh

set -euo pipefail

DEST_DIR="$(dirname "$0")/../data/raw"
mkdir -p "$DEST_DIR"

echo "Downloading thoughtvector/customer-support-on-twitter to $DEST_DIR ..."
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p "$DEST_DIR" --unzip

echo "Done. Expect $DEST_DIR/twcs.csv (~2.6GB uncompressed)."
echo "Next: python -m src.data_prep"
