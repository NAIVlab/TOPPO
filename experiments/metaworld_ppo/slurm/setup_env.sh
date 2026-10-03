#!/bin/bash
# ==============================================================================
# One-time environment setup for HPC cluster
# Usage: bash slurm/setup_env.sh
# Safe to re-run — skips steps that are already done.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_HELPER="${SCRIPT_DIR}/conda_init.sh"

# === Configuration (edit these for your cluster) ===
PYTHON_VERSION="3.12"
ENV_NAME="mtrl_env"
# PyTorch CUDA wheel URL — change to match your cluster's CUDA version
# CUDA 12.8: https://download.pytorch.org/whl/cu128
# CUDA 12.4: https://download.pytorch.org/whl/cu124
# CUDA 12.1: https://download.pytorch.org/whl/cu121
# CUDA 11.8: https://download.pytorch.org/whl/cu118
PYTORCH_CUDA_URL="https://download.pytorch.org/whl/cu124"
# Optional torch version pin; empty = newest on the index above.  The pin must
# exist on that index: e.g. "==2.6.0", the newest torch on cu124 (424 of the
# 822 Meta-World PPO runs in the public W&B project that carry a
# requirements.txt used 2.6.0+cu124), or
# "==2.10.0" together with the cu128 URL (the headline groups used
# 2.10.0+cu128).  The versions behind the runs are listed at the top of
# requirements.txt.
TORCH_VERSION_SPEC=""

# === Step 1: Load modules (uncomment/edit for your cluster) ===
# module load cuda/12.4
# module load anaconda3

# === Step 2: Find or install conda ===
_activate_conda() {
    if [ -f "${CONDA_HELPER}" ]; then
        # shellcheck disable=SC1091
        source "${CONDA_HELPER}"
        if init_slurm_conda_shell; then
            return 0
        fi
    fi
    return 1
}

if ! _activate_conda; then
    echo "Conda not found. Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    export CONDA_ROOT="$HOME/miniconda3"
    if ! _activate_conda; then
        echo "ERROR: Miniconda installed but conda shell initialization still failed." >&2
        exit 1
    fi
    echo "Miniconda installed."
fi
echo "Conda: $(conda --version)"

# === Step 3: Create conda environment (skip if exists) ===
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists, skipping creation."
else
    echo "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"

# === Step 4: Install PyTorch ===
echo "Installing PyTorch (CUDA: ${PYTORCH_CUDA_URL})"
pip install "torch${TORCH_VERSION_SPEC}" --index-url "${PYTORCH_CUDA_URL}"

# === Step 5: Install project dependencies ===
echo "Installing dependencies from ${PROJECT_DIR}/requirements.txt"
pip install -r "${PROJECT_DIR}/requirements.txt"

# === Step 6: Verify installation ===
echo ""
echo "=== Verifying installation ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import metaworld
print('MetaWorld: OK')
import tianshou
print('Tianshou: OK')
import gymnasium
print('Gymnasium: OK')
print('All dependencies installed successfully!')
"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Edit the #SBATCH lines in slurm/*.sbatch to set your partition name"
echo "  2. Run 'bash slurm/launch_baselines.sh --dry-run' to preview jobs"
echo "  3. Run 'bash slurm/launch_baselines.sh' to submit the 20 baseline runs"
