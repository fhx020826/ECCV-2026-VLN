#!/bin/bash
# ============================================================
# Download ETPNav data, weights, and R2R-CE dataset
# Run AFTER placing download_mp.py in scripts/data_download/
#
# Usage: bash scripts/data_download/download_all.sh
# ============================================================

set -e

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
DATA_DIR="$ROOT_DIR/ETPNav/data"
PRETRAINED_DIR="$ROOT_DIR/ETPNav/pretrained"

pip install gdown -q

echo "=== [1/4] Downloading etp_ckpt.zip (datasets + pretrained + finetuned weights) ==="
# From README: https://drive.google.com/file/d/1MWR_Cf4m9HEl_3z8a5VfZeyUWIUTfIYr/view
python -m gdown "1MWR_Cf4m9HEl_3z8a5VfZeyUWIUTfIYr" \
    -O "$ROOT_DIR/ETPNav/etp_ckpt.zip"

echo "=== [2/4] Extracting etp_ckpt.zip ==="
cd "$ROOT_DIR/ETPNav" && unzip -o etp_ckpt.zip && rm etp_ckpt.zip
echo "Extraction done."

echo "=== [3/4] Downloading Waypoint Predictor (R2R-CE, hfov90) ==="
# From README: https://drive.google.com/file/d/1goXbgLP2om9LsEQZ5XvB0UpGK4A5SGJC/view
python -m gdown "1goXbgLP2om9LsEQZ5XvB0UpGK4A5SGJC" \
    -O "$DATA_DIR/wp_pred/check_cwp_bestdist_hfov90"

echo "=== [4/4] Downloading Precomputed Image Features (for pretrain) ==="
# From README: https://drive.google.com/file/d/1D3Gd9jqRfF-NjlxDAQG_qwxTIakZlrWd/view
python -m gdown "1D3Gd9jqRfF-NjlxDAQG_qwxTIakZlrWd" \
    -O "$ROOT_DIR/ETPNav/pretrain_src/precomputed_features.zip"
cd "$ROOT_DIR/ETPNav/pretrain_src" && unzip -o precomputed_features.zip && rm precomputed_features.zip

echo "=== All downloads complete! ==="
echo "Data structure:"
echo "  ETPNav/data/datasets/      - R2R-CE annotations"
echo "  ETPNav/pretrained/ETP/     - pretrained + finetuned weights"
echo "  ETPNav/data/wp_pred/       - waypoint predictor checkpoint"
echo ""
echo "Next step: place download_mp.py and run sbatch scripts/slurm/download_mp3d.sbatch"
