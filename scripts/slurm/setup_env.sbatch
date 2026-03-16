#!/bin/bash
#SBATCH --job-name=setup_vlnce_env
#SBATCH --output=logs/setup_vlnce_%j.out
#SBATCH --error=logs/setup_vlnce_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu

# Run the environment setup on a compute node (avoids login node limits)
set -e

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
bash "$ROOT_DIR/environments/setup_vlnce.sh" 2>&1

echo "Environment setup complete."
