#!/bin/bash
#SBATCH --job-name=mp3d_download
#SBATCH --output=logs/mp3d_download_%j.out
#SBATCH --error=logs/mp3d_download_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu

# ============================================================
# Download MP3D Habitat scenes (.glb) for VLN-CE / ETPNav
# Prerequisites: obtain download_mp.py from Matterport3D
#   https://niessner.github.io/Matterport/
# Usage: sbatch scripts/slurm/download_mp3d.sbatch
# ============================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname $(dirname $SCRIPT_DIR))"
DATA_DIR="$ROOT_DIR/ETPNav/data/scene_datasets/mp3d"

mkdir -p "$DATA_DIR"
echo "[$(date)] Starting MP3D Habitat scene download to $DATA_DIR"

# The download_mp.py script must be placed in scripts/data_download/
DOWNLOAD_SCRIPT="$SCRIPT_DIR/../data_download/download_mp.py"

if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
    echo "ERROR: download_mp.py not found at $DOWNLOAD_SCRIPT"
    echo "Please place the script from http://kaldir.vc.cit.tum.de/matterport/download_mp.py there first."
    exit 1
fi

# Download only the Habitat-format .glb files (much smaller than full MP3D)
# This uses --type .glb which downloads only what habitat-sim needs
python "$DOWNLOAD_SCRIPT" \
    --task habitat \
    -o "$DATA_DIR" 2>&1

echo "[$(date)] MP3D download finished. Verifying scene count..."
SCENE_COUNT=$(find "$DATA_DIR" -name "*.glb" | wc -l)
echo "Found $SCENE_COUNT .glb scene files (expected 90)"
