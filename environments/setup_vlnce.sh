#!/bin/bash
# ============================================================
# Setup VLN-CE conda environment for ETPNav
# Python 3.8 + habitat-sim 0.1.7 + PyTorch 1.9.1
#
# Prerequisites:
#   - Miniconda3 installed at ~/miniconda3
#
# Usage:
#   bash environments/setup_vlnce.sh
# ============================================================

set -e

CONDA_BASE="$HOME/miniconda3"
ENV_NAME="vlnce"

# Initialize conda for this script
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "=== Creating conda environment: $ENV_NAME (Python 3.8) ==="
conda create -n "$ENV_NAME" python=3.8 -y

conda activate "$ENV_NAME"

echo "=== Installing habitat-sim 0.1.7 (headless) ==="
conda install -c aihabitat -c conda-forge \
    habitat-sim=0.1.7 headless -y

echo "=== Installing habitat-lab 0.1.7 ==="
pip install git+https://github.com/facebookresearch/habitat-lab.git@v0.1.7 \
    --no-build-isolation

echo "=== Installing PyTorch 1.9.1 + CUDA 11.1 ==="
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

echo "=== Installing CLIP ==="
pip install git+https://github.com/openai/CLIP.git

echo "=== Installing remaining dependencies ==="
pip install \
    gym==0.21.0 \
    numpy==1.23.5 \
    einops \
    networkx \
    tqdm \
    tensorboard \
    fastdtw \
    jsonlines \
    timm==0.5.4 \
    transformers==4.12.5 \
    sentencepiece \
    scikit-learn \
    scipy \
    Pillow

echo "=== Verifying installation ==="
python -c "import habitat_sim; print('habitat_sim:', habitat_sim.__version__)"
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import clip; print('CLIP: OK')"

echo "=== Done! Activate with: conda activate $ENV_NAME ==="
