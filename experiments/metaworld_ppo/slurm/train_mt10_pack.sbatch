#!/bin/bash
# ==============================================================================
# SLURM job: packed MT10 run (2 seeds × 1 GPU) for the baseline launcher.
# Mirrors train.sbatch but launches 2 python children on the allocated GPU
# (analogous to the MT10 array sbatches under slurm/).  MT50 keeps using
# train.sbatch (1 seed per GPU).
#
# Usage (normally via slurm/launch_baselines.sh, which also creates run_logs/;
# when submitting by hand, run `mkdir -p run_logs` first, from
# experiments/metaworld_ppo/):
#   sbatch --export=ALL,CONFIG_FILE=...,SEED_A=0,SEED_B=1,
#                   EXP_NAME_A=...,EXP_NAME_B=... slurm/train_mt10_pack.sbatch
#
# Caller passes EXTRA_ARGS / WANDB_TAGS via env (sbatch --export=ALL).
# EXTRA_ARGS is appended to the python command line of BOTH children.
# ==============================================================================

#SBATCH --job-name=mt10_pack
#SBATCH --output=run_logs/%x_%j.out
#SBATCH --error=run_logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpua100|gpuv100x|gpuv100
#SBATCH --time=48:00:00
# Deprioritize MT10 vs MT50 (which has no --nice) so MT50 tasks win when
# fairshare + cluster capacity is contested — same convention as the MT10
# array sbatches.
#SBATCH --nice=100
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append

# === Project / environment paths ===
# TOPPO_ROOT, if set, must be <clone>/experiments/metaworld_ppo (the directory
# holding train.py and slurm/), not the repository root.
PROJECT_ROOT="${TOPPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN_LOG_DIR="${PROJECT_ROOT}/run_logs"
TRAIN_LOG_DIR="${PROJECT_ROOT}/logs/baseline"

# === Activate conda environment ===
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/slurm/conda_init.sh"
activate_slurm_conda_env mtrl_env
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

# === Required parameters ===
CONFIG_FILE="${CONFIG_FILE:?CONFIG_FILE required}"
SEED_A="${SEED_A:?SEED_A required}"
SEED_B="${SEED_B:?SEED_B required}"
EXP_NAME_A="${EXP_NAME_A:?EXP_NAME_A required}"
EXP_NAME_B="${EXP_NAME_B:?EXP_NAME_B required}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# === MuJoCo headless rendering ===
export MUJOCO_GL=egl
unset DISPLAY

# === WandB ===
# Credentials are NEVER stored in this repository. Provide the key by
# exporting WANDB_API_KEY, or by placing it in ~/.wandb_api_key (mode 600).
if [ -z "${WANDB_API_KEY:-}" ] && [ -f ~/.wandb_api_key ]; then
    export WANDB_API_KEY="$(<~/.wandb_api_key)"
fi

# === Temporary directory ===
export TMPDIR="${PROJECT_ROOT}/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$RUN_LOG_DIR" "$TRAIN_LOG_DIR"

cd "$PROJECT_ROOT"

# === Job info ===
echo "=========================================="
echo "SLURM Job ID:    $SLURM_JOB_ID"
echo "Node:            $(hostname)"
echo "GPU:             $CUDA_VISIBLE_DEVICES"
echo "Config:          $CONFIG_FILE"
echo "Seeds (packed):  $SEED_A, $SEED_B"
echo "Exp A / B:       $EXP_NAME_A / $EXP_NAME_B"
echo "WandB tags:      ${WANDB_TAGS:-<unset>}"
echo "Extra args:      ${EXTRA_ARGS:-<none>}"
echo "Start time:      $(date)"
echo "=========================================="

# === Launch 2 training runs on the same GPU ===
TMPDIR_A="${TMPDIR}/a"
TMPDIR_B="${TMPDIR}/b"
mkdir -p "$TMPDIR_A" "$TMPDIR_B"

# Forward SIGUSR1 to both python children so they snapshot latest.pth and
# exit 0 cleanly; the GOT_SIGNAL flag drives the requeue decision below.
GOT_SIGNAL=0
forward_usr1() {
    echo "[sbatch] Received SIGUSR1 at $(date) — forwarding to PIDs $PID_A $PID_B"
    GOT_SIGNAL=1
    [ -n "$PID_A" ] && kill -USR1 "$PID_A" 2>/dev/null || true
    [ -n "$PID_B" ] && kill -USR1 "$PID_B" 2>/dev/null || true
}
trap forward_usr1 USR1

# `--logging.save-interval 200000` = one MT10 epoch.  Like train.sbatch, this
# template adds no LayerNorm or PopArt flag of its own: the arm comes from the
# YAML plus EXTRA_ARGS, which carries `--ppo.use-popart --network.ln-critic
# --logging.wandb-group ...` from the launcher; unquoted expansion
# word-splits it into tokens.
(
    export TMPDIR="$TMPDIR_A"
    "$PYTHON_BIN" "${PROJECT_ROOT}/train.py" \
        --config "$CONFIG_FILE" \
        --seed "$SEED_A" \
        --device cuda \
        --exp-name "$EXP_NAME_A" \
        --logging.test-interval 2 \
        --logging.no-save-best \
        --logging.save-interval 200000 \
        --logging.log-dir "$TRAIN_LOG_DIR" \
        $EXTRA_ARGS
) > "${RUN_LOG_DIR}/${EXP_NAME_A}.log" 2>&1 &
PID_A=$!

(
    export TMPDIR="$TMPDIR_B"
    "$PYTHON_BIN" "${PROJECT_ROOT}/train.py" \
        --config "$CONFIG_FILE" \
        --seed "$SEED_B" \
        --device cuda \
        --exp-name "$EXP_NAME_B" \
        --logging.test-interval 2 \
        --logging.no-save-best \
        --logging.save-interval 200000 \
        --logging.log-dir "$TRAIN_LOG_DIR" \
        $EXTRA_ARGS
) > "${RUN_LOG_DIR}/${EXP_NAME_B}.log" 2>&1 &
PID_B=$!

echo "Launched seed $SEED_A (pid $PID_A) and seed $SEED_B (pid $PID_B)."

# `wait` is interrupted by SIGUSR1 (trap runs, then wait returns non-zero).
# Re-wait until the PID really exits so we capture the true exit code.
wait $PID_A 2>/dev/null; EXIT_A=$?
while kill -0 $PID_A 2>/dev/null; do wait $PID_A 2>/dev/null; EXIT_A=$?; done
wait $PID_B 2>/dev/null; EXIT_B=$?
while kill -0 $PID_B 2>/dev/null; do wait $PID_B 2>/dev/null; EXIT_B=$?; done

echo "=========================================="
echo "Seed $SEED_A exit code: $EXIT_A"
echo "Seed $SEED_B exit code: $EXIT_B"
echo "GOT_SIGNAL:      $GOT_SIGNAL"
echo "End time:        $(date)"
echo "=========================================="

if [ "$GOT_SIGNAL" -eq 1 ]; then
    if [ $EXIT_A -eq 0 ] && [ $EXIT_B -eq 0 ]; then
        echo "[sbatch] Graceful shutdown succeeded — requeuing ${SLURM_JOB_ID}"
        scontrol requeue "${SLURM_JOB_ID}"
        exit 0
    else
        echo "[sbatch] Signal received BUT graceful save failed (seed_A=$EXIT_A, seed_B=$EXIT_B) — NOT requeuing; inspect $RUN_LOG_DIR/${EXP_NAME_A}.log and ${EXP_NAME_B}.log"
        exit 1
    fi
fi

if [ $EXIT_A -ne 0 ] || [ $EXIT_B -ne 0 ]; then
    exit 1
fi
