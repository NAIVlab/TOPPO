#!/bin/bash
# ==============================================================================
# SLURM job script for MTRL baseline training (MT10 / MT50)
# Usage, from experiments/metaworld_ppo/ (SLURM does not create the --output
# directory, so make run_logs/ before the first submission):
#   mkdir -p run_logs
#   sbatch --export=ALL,CONFIG_FILE=...,SEED=0,EXP_NAME=... slurm/train.sbatch
#
# One seed of a docs/RUNS.md group: set CONFIG_FILE to the row's config,
# SEED=<seed> (it defaults to 0, so set it for every seed) and
# EXP_NAME=<group>_s<seed>, and put the row's overrides plus
# `--logging.wandb-group <group>` in EXTRA_ARGS (export it in your shell;
# --export=ALL forwards it).  The template adds no method, normalizer or
# LayerNorm flag of its own, so the row's config and overrides alone decide
# the arm; the flags it does add (--device cuda, --logging.test-interval 2,
# which every paper config already sets, and the log-dir / checkpoint
# settings) do not change what is trained.  The configs/ppo_*_toppo_*.yaml
# files already carry their canonical wandb_group, so the MT50 headline,
# seed 0, is just:
#   sbatch --export=ALL,CONFIG_FILE=configs/ppo_mt50_toppo_lnc_rfv2.yaml,SEED=0,EXP_NAME=mt50v2_ppo_apc_cfg1_npaw_lnc_vc1_400x3_s0 slurm/train.sbatch
# If a PCGrad-actor job dies with exit 136, add MTRL_PCGRAD_CPU=1 to --export
# (docs/env_vars.md).
#
# Supports SIGUSR1 graceful-checkpoint + scontrol requeue: SLURM fires
# SIGUSR1 3600s before walltime; the trap forwards it to the srun child,
# train.py writes latest.pth at the next epoch boundary and exits 0, this
# script then calls `scontrol requeue` so the job restarts and resumes.
# Non-zero exit after the signal → no requeue (surfaces real bugs).
# ==============================================================================

#SBATCH --job-name=mtrl
#SBATCH --output=run_logs/%x_%j.out
#SBATCH --error=run_logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpua100|gpuv100x|gpuv100
#SBATCH --time=144:00:00
# 3600s signal lead: safe for MT10 (~6min/epoch → ~10 epochs' headroom)
# and MT50 (~30min/epoch → ~2 epochs' headroom).  train.py snapshots
# latest.pth at the next epoch boundary and exits 0.
#SBATCH --signal=B:USR1@3600
#SBATCH --requeue
#SBATCH --open-mode=append

# ==============================================================================
# CONFIGURE THESE FOR YOUR CLUSTER:
#   1. Change --partition to your GPU partition name
#   2. Uncomment/edit module loads below
#   3. Adjust --time if needed.  Typical wall time of one 100M-step MT50
#      seed, evaluation on, on a V100: default combiner ~18-26 h (single
#      seeds up to ~45 h) and FairGrad / PCGrad / CAGrad combiners ~30-57 h
#      (single seeds up to ~80 h); the headline config 23-24 h on an RTX 3090
#      (31-32 h when preempted and resumed) and ~14 h on an H200 NVL.
#      A run longer than --time saves and requeues (below), so --time need
#      not cover the whole run.
# ==============================================================================

# === Module loads (edit for your cluster) ===
# module load cuda/12.4
# module load anaconda3

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

# === Parameters (passed via --export or defaults) ===
CONFIG_FILE="${CONFIG_FILE:-configs/ppo_mt50_default.yaml}"
SEED="${SEED:-0}"
EXP_NAME="${EXP_NAME:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# One-epoch save interval inferred from benchmark (MT10 step_per_epoch=200000,
# MT50=1000000).  Matches the fairgrad sweep's cadence so graceful exit
# loses at most ~1 epoch of progress.
if [[ "$CONFIG_FILE" == *mt10* ]]; then
    SAVE_INTERVAL=200000
else
    SAVE_INTERVAL=1000000
fi

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

# === Working directory ===
cd "$PROJECT_ROOT"

# === Build command ===
# --device cuda: override hardcoded device in YAML configs
# Neither critic LayerNorm nor PopArt is hardcoded here: both come from the
# chosen YAML (every shipped config states `network.ln_critic` and
# `ppo.use_popart` explicitly) or from an explicit flag in EXTRA_ARGS
# (`--network.ln-critic` / `--network.no-ln-critic`, `--ppo.use-popart` /
# `--ppo.no-use-popart`).  Launchers that pin a rung pass the flags
# themselves -- see slurm/launch_baselines.sh (LN-c and PopArt on).
# SLURM sets CUDA_VISIBLE_DEVICES automatically with --gres=gpu:1
CMD="${PYTHON_BIN} ${PROJECT_ROOT}/train.py --config ${CONFIG_FILE} --seed ${SEED} --device cuda --logging.log-dir ${TRAIN_LOG_DIR} --logging.test-interval 2 --logging.no-save-best --logging.save-interval ${SAVE_INTERVAL}"
if [ -n "$EXP_NAME" ]; then
    CMD="$CMD --exp-name ${EXP_NAME}"
fi
if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD ${EXTRA_ARGS}"
fi

# === Job info ===
echo "=========================================="
echo "SLURM Job ID:    $SLURM_JOB_ID"
echo "Node:            $(hostname)"
echo "GPU:             $CUDA_VISIBLE_DEVICES"
echo "Working dir:     $(pwd)"
echo "Config:          $CONFIG_FILE"
echo "Seed:            $SEED"
echo "Save interval:   $SAVE_INTERVAL"
echo "Python:          $PYTHON_BIN"
echo "TMPDIR:          $TMPDIR"
echo "Train log dir:   $TRAIN_LOG_DIR"
echo "SLURM run logs:  $RUN_LOG_DIR"
echo "WandB:           ${WANDB_API_KEY:+key set (${#WANDB_API_KEY} chars)}${WANDB_API_KEY:+, }mode=${WANDB_MODE:-online}${WANDB_TAGS:+, tags=$WANDB_TAGS}"
echo "Extra args:      ${EXTRA_ARGS:-<none>}"
echo "Command:         $CMD"
echo "Start time:      $(date)"
echo "=========================================="

# === Signal trap → graceful save + requeue ===
# SLURM delivers SIGUSR1 to the batch shell (B: prefix in --signal); the
# trap forwards it to the srun child so python can snapshot latest.pth
# and exit 0.  The GOT_SIGNAL flag drives the requeue decision below.
GOT_SIGNAL=0
forward_usr1() {
    echo "[sbatch] Received SIGUSR1 at $(date) — forwarding to srun PID $SRUN_PID"
    GOT_SIGNAL=1
    [ -n "$SRUN_PID" ] && kill -USR1 "$SRUN_PID" 2>/dev/null || true
}
trap forward_usr1 USR1

# === Run training ===
srun $CMD &
SRUN_PID=$!

# `wait` is interrupted by SIGUSR1 (trap runs, then wait returns non-zero).
# Re-wait until the PID really exits so we capture the true exit code.
wait $SRUN_PID 2>/dev/null; EXIT_CODE=$?
while kill -0 $SRUN_PID 2>/dev/null; do wait $SRUN_PID 2>/dev/null; EXIT_CODE=$?; done

echo "=========================================="
echo "srun exit code:  $EXIT_CODE"
echo "GOT_SIGNAL:      $GOT_SIGNAL"
echo "End time:        $(date)"
echo "=========================================="

if [ "$GOT_SIGNAL" -eq 1 ]; then
    # Only requeue when srun exited cleanly.  A non-zero exit after the
    # signal means the graceful save itself raised (disk full,
    # checkpoint-format mismatch, etc.) — requeuing would mask a real
    # failure and the next re-entry would resume from a stale/incomplete
    # latest.pth.
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[sbatch] Graceful shutdown succeeded — requeuing ${SLURM_JOB_ID}"
        scontrol requeue "${SLURM_JOB_ID}"
        exit 0
    else
        echo "[sbatch] Signal received BUT graceful save failed (exit=$EXIT_CODE) — NOT requeuing; inspect $RUN_LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.{out,err}"
        exit 1
    fi
fi

exit $EXIT_CODE
