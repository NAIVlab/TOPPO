#!/usr/bin/env bash
# Train, then evaluate, one (group, seed) of the AMP campaign on one GPU.
#
#   bash scripts/run_arm.sh <ASSEMBLED_TREE> <wandb_group> <seed>
#
# <ASSEMBLED_TREE> is the directory setup.sh created. The commands are the ones
# `python campaign.py train|eval <group> <seed>` prints; the checkpoint goes to
# <tree>/output/<group>_s<seed>/model.pt and the evaluation record to
# <tree>/results/eval/<group>_s<seed>.json.
#
# One process on cuda:0 (the first visible GPU): the SAC and ARS code paths
# assert a single process. Select the GPU with CUDA_VISIBLE_DEVICES.
#
# Env: PYTHON (default: python); MIN_GPU_MIB (default 20000) refuses smaller GPUs
# such as MIG slices, on which 4096 environments do not fit (training peaks at
# about 14-17 GB on the reference hardware); SKIP_TRAIN=1 only evaluates.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TREE="$(cd "${1:?usage: run_arm.sh <tree> <group> <seed>}" && pwd)"
GROUP="${2:?usage: run_arm.sh <tree> <group> <seed>}"
SEED="${3:?usage: run_arm.sh <tree> <group> <seed>}"
PYTHON="${PYTHON:-python}"
MIN_GPU_MIB="${MIN_GPU_MIB:-20000}"

if command -v nvidia-smi >/dev/null 2>&1; then
    FIRST_GPU="${CUDA_VISIBLE_DEVICES:-0}"
    FIRST_GPU="${FIRST_GPU%%,*}"
    MIB="$(nvidia-smi --id="$FIRST_GPU" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
           | head -1 | tr -d ' ' || true)"
    if ! [[ "$MIB" =~ ^[0-9]+$ ]]; then
        echo "warning: could not read the memory of GPU $FIRST_GPU; skipping the size check" >&2
    elif [ "$MIB" -lt "$MIN_GPU_MIB" ]; then
        echo "error: GPU $FIRST_GPU has ${MIB} MiB < MIN_GPU_MIB=${MIN_GPU_MIB} (MIG slice?)" >&2
        exit 1
    fi
fi

TRAIN_CMD="$("$PYTHON" "$HERE/campaign.py" train "$GROUP" "$SEED" --python "$PYTHON")"
EVAL_CMD="$("$PYTHON" "$HERE/campaign.py" eval "$GROUP" "$SEED" --python "$PYTHON")"

cd "$TREE"
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    echo "=== [$(date +%H:%M:%S)] train $GROUP s$SEED"
    echo "$TRAIN_CMD"
    eval "$TRAIN_CMD"
fi
echo "=== [$(date +%H:%M:%S)] eval  $GROUP s$SEED"
echo "$EVAL_CMD"
eval "$EVAL_CMD"
echo "=== [$(date +%H:%M:%S)] done: results/eval/${GROUP}_s${SEED}.json"
