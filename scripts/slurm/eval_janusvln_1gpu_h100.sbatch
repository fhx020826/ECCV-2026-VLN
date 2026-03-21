#!/bin/bash
#SBATCH --job-name=janus_eval_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/home/hxfeng/PAM-FlowNav/logs/eval_janusvln_unseen_%j.out
#SBATCH --error=/home/hxfeng/PAM-FlowNav/logs/eval_janusvln_unseen_%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pamflownav

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
unset ALL_PROXY
# Do NOT load cuda module for inference: torch 2.8.0+cu128 bundles its own
# CUDA 12.8 runtime; loading cuda/12.4.1 would override paths and break Triton

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

cd /home/hxfeng/PAM-FlowNav/JanusVLN

CHECKPOINT="/home/hxfeng/PAM-FlowNav/data/checkpoints/misstl/JanusVLN_Extra"
CONFIG="config/vln_r2r.yaml"
OUTPUT_PATH="/home/hxfeng/PAM-FlowNav/logs/eval_extra_unseen_$(date +%Y%m%d_%H%M)"
MASTER_PORT=$((RANDOM % 101 + 20000))

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "CHECKPOINT: $CHECKPOINT"
echo "OUTPUT_PATH: $OUTPUT_PATH"

mkdir -p $OUTPUT_PATH

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
    src/evaluation.py \
    --model_path $CHECKPOINT \
    --habitat_config_path $CONFIG \
    --output_path $OUTPUT_PATH
