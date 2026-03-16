#!/bin/bash
#SBATCH --job-name=etp_pretrain
#SBATCH --output=logs/etp_pretrain_%j.out
#SBATCH --error=logs/etp_pretrain_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:2

# ============================================================
# ETPNav Backbone Pretraining (MLM + SAP, 100K steps)
# Init from LXMERT, trains GlocalTextPathCMTPreTraining
# Prerequisites:
#   - pretrain_src/datasets/pretrained/LXMERT/model_LXRT.pth
#   - pretrain_src/img_features/CLIP-ViT-B-32-views-habitat.hdf5
#   - pretrain_src/img_features/ddppo_resnet50_depth_features.hdf5
#   - pretrain_src/datasets/R2R/annotations/pretrain_map/*.jsonl
# Usage: sbatch scripts/slurm/pretrain_etp.sbatch
# ============================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname $(dirname $SCRIPT_DIR))"
ETP_DIR="$ROOT_DIR/ETPNav"

cd "$ETP_DIR"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate vlnce

export HTTPS_PROXY=http://127.0.0.1:17897
export PYTHONPATH="$ETP_DIR:$ETP_DIR/pretrain_src:$PYTHONPATH"

OUTDIR="pretrained/ETP/mlm.sap_r2r"
mkdir -p "$OUTDIR/ckpts"

echo "[$(date)] Starting ETPNav pretraining on $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --node_rank=0 \
    --master_port=29501 \
    pretrain_src/pretrain_src/train_r2r.py \
    --world_size 2 \
    --vlnbert cmt \
    --model_config pretrain_src/run_pt/r2r_model_config_dep.json \
    --config pretrain_src/run_pt/r2r_pretrain_habitat.json \
    --output_dir "$OUTDIR" \
    --init_pretrained lxmert \
    2>&1

echo "[$(date)] Pretraining complete."
