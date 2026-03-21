#!/bin/bash
#SBATCH --job-name=janus_eval_unseen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/home/hxfeng/PAM-FlowNav/logs/eval_janusvln_unseen_%j.out
#SBATCH --error=/home/hxfeng/PAM-FlowNav/logs/eval_janusvln_unseen_%j.err

# 环境初始化
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pamflownav

# 关键：CUDA lib 路径（防止 TensorRT 库覆盖）
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 关键：关闭死代理
unset ALL_PROXY

# 关键：CUDA 模块加载（确保 nvcc 可用，flash-attn 等编译用）
. /usr/share/modules/init/bash
module use --append /home/share/modules/modulefiles
module load cuda/12.4.1

# Habitat 日志静默
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

cd /home/hxfeng/PAM-FlowNav/JanusVLN

CHECKPOINT="/home/hxfeng/PAM-FlowNav/data/checkpoints/misstl/JanusVLN_Extra"
CONFIG="config/vln_r2r.yaml"
OUTPUT_PATH="/home/hxfeng/PAM-FlowNav/logs/eval_extra_unseen_$(date +%Y%m%d_%H%M)"
MASTER_PORT=$((RANDOM % 101 + 20000))

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CHECKPOINT: $CHECKPOINT"
echo "CONFIG: $CONFIG"
echo "OUTPUT_PATH: $OUTPUT_PATH"

mkdir -p $OUTPUT_PATH

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT \
    src/evaluation.py \
    --model_path $CHECKPOINT \
    --habitat_config_path $CONFIG \
    --output_path $OUTPUT_PATH
