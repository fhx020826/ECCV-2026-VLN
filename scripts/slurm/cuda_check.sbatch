#!/bin/bash
#SBATCH --job-name=cuda-check
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=00:03:00
#SBATCH --output=/home/hxfeng/PAM-FlowNav/logs/cuda_check_%j.out
#SBATCH --error=/home/hxfeng/PAM-FlowNav/logs/cuda_check_%j.err

echo "=== nvidia-smi ==="
nvidia-smi

echo ""
echo "=== CUDA version ==="
nvcc --version 2>/dev/null || echo "nvcc not found in PATH"

echo ""
echo "=== CUDA lib paths ==="
find /usr/local -maxdepth 3 -name "version*.txt" 2>/dev/null | xargs -I{} sh -c 'echo "{}:"; cat {}'
ls /usr/local/ | grep -i cuda

echo ""
echo "=== GCC version ==="
gcc --version | head -1

echo ""
echo "=== Python ==="
which python3
python3 --version
