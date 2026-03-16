#!/bin/bash
#SBATCH --job-name=etp_finetune
#SBATCH --output=logs/etp_finetune_%j.out
#SBATCH --error=logs/etp_finetune_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:2

# ============================================================
# ETPNav VLN-CE Finetuning (IL, 15K iters)
# Prerequisites:
#   - pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt
#   - data/datasets/R2R_VLNCE_v1-3_preprocessed_BERTidx/
#   - data/wp_pred/check_cwp_bestdist_hfov90
#   - data/scene_datasets/mp3d/   (MP3D .glb files)
# Usage: sbatch scripts/slurm/finetune_etp.sbatch
# ============================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname $(dirname $SCRIPT_DIR))"
ETP_DIR="$ROOT_DIR/ETPNav"

cd "$ETP_DIR"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate vlnce

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export PYTHONPATH="$ETP_DIR:$PYTHONPATH"

EXP_NAME="release_r2r_$(date +%Y%m%d_%H%M%S)"
mkdir -p "data/logs/checkpoints/$EXP_NAME"

echo "[$(date)] Starting ETPNav VLN-CE finetuning on $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"
echo "Experiment: $EXP_NAME"

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29502 \
    run.py \
    --exp_name "$EXP_NAME" \
    --run-type train \
    --exp-config run_r2r/iter_train.yaml \
    SIMULATOR_GPU_IDS [0,1] \
    TORCH_GPU_IDS [0,1] \
    GPU_NUMBERS 2 \
    NUM_ENVIRONMENTS 8 \
    IL.iters 15000 \
    IL.lr 1e-5 \
    IL.log_every 200 \
    IL.ml_weight 1.0 \
    IL.sample_ratio 0.75 \
    IL.decay_interval 3000 \
    IL.load_from_ckpt False \
    IL.is_requeue True \
    IL.waypoint_aug True \
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
    MODEL.pretrained_path pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt \
    2>&1

echo "[$(date)] Finetuning complete."
