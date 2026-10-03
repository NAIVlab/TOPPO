#!/bin/bash
# ==============================================================================
# SLURM array job: MT50 method sweep
#   Array 0..99 = 10 configs × 10 seeds
#   1 run per array task (MT50's 50 env workers saturate CPU; no packing).
#
# v2.13+ per-side design: actor_method / critic_method are independently
# overridden from a single-method base YAML. No global `scope` field.
# Every run has `--network.ln-critic` enabled and PopArt explicitly DISABLED
# via `--ppo.no-use-popart`.  This is the LN-c-only rung (LN = C,
# PopArt = no; wandb groups `hpc_ppo_mt{10,50}_*_lnc`, tag `hpc_ppo_lnc_v1`).
# The flag is mandatory: the base YAMLs configs/ppo_mt{10,50}_{default,pcgrad,
# cagrad,fairgrad}.yaml ship `ppo.use_popart: true`, so without it every run in
# this sweep would silently be a PopArt run under a non-PopArt group name.
#
# The group names this template emits, `hpc_ppo_<bench>_<method>_<side>_lnc`
# with tag `hpc_ppo_lnc_v1`, are not docs/RUNS.md names and appear on no
# published run.  The paper's LN-c-only rows (mt{10,50}v2_ppo_ad_cd_nnone_lnc_vc1_400x3,
# mt{10,50}v2_ppo_ad_cfg1_nnone_lnc_vc1_400x3) were launched from
# configs/ppo_mt{10,50}_default.yaml as hpc_ppo_mt{10,50}_ad_{cd,cfg}_lnc (as
# their recorded logging.wandb_group shows), not by this template.  To
# reproduce a RUNS.md row, use slurm/train.sbatch with that row's config and
# overrides.  GROUP below is passed as
# --logging.wandb-group.  Submit through slurm/launch_method_sweep.sh, which
# creates run_logs/ first (SLURM does not create the --output directory).
# ==============================================================================

#SBATCH --job-name=hpc_ppo_mt50_sweep_lnc
#SBATCH --output=run_logs/%x_%A_%a.out
#SBATCH --error=run_logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# 16 cpus for 50 env workers = 0.32 core/worker — oversubscribed but a
# compromise with queue time. Fits all three node types: A100 50%/node,
# V100x 44%/node, V100 57%/node (each taking ~2 GPUs' worth of CPU budget).
#SBATCH --constraint=gpua100|gpuv100x|gpuv100
#SBATCH --time=96:00:00
#SBATCH --array=0-99%30

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

# === Decode array task id → (method, side, seed) ===
# 10 configs indexed 0..9 (keep in lockstep with launch_method_sweep.sh):
#   0: default / both
#   1-3: pcgrad   {both, actor, critic}
#   4-6: cagrad   {both, actor, critic}
#   7-9: fairgrad {both, actor, critic}
METHODS=(default pcgrad pcgrad pcgrad cagrad cagrad cagrad fairgrad fairgrad fairgrad)
SIDES=(both both actor critic both actor critic both actor critic)

CFG_IDX=$((SLURM_ARRAY_TASK_ID / 10))
SEED=$((SLURM_ARRAY_TASK_ID % 10))
METHOD=${METHODS[$CFG_IDX]}
SIDE=${SIDES[$CFG_IDX]}

GROUP="hpc_ppo_mt50_${METHOD}_${SIDE}_lnc"
EXP="hpc_ppo_mt50_${METHOD}_${SIDE}_lnc_s${SEED}"
CONFIG="configs/ppo_mt50_${METHOD}.yaml"

# Per-side overrides: each base YAML sets both sides to METHOD; we override
# the "off" side to default when SIDE != both.  SIDE=both needs no override.
case "$SIDE" in
    both)   GRAD_OVERRIDE=() ;;
    actor)  GRAD_OVERRIDE=(--gradient.critic-method default) ;;
    critic) GRAD_OVERRIDE=(--gradient.actor-method default) ;;
esac

# === MuJoCo headless rendering ===
export MUJOCO_GL=egl
unset DISPLAY

# === WandB ===
# Credentials are NEVER stored in this repository. Provide the key by
# exporting WANDB_API_KEY, or by placing it in ~/.wandb_api_key (mode 600).
if [ -z "${WANDB_API_KEY:-}" ] && [ -f ~/.wandb_api_key ]; then
    export WANDB_API_KEY="$(<~/.wandb_api_key)"
fi
export WANDB_TAGS="hpc_ppo_lnc_v1,mt50"

# === Temporary directory ===
export TMPDIR="${PROJECT_ROOT}/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$RUN_LOG_DIR" "$TRAIN_LOG_DIR"

cd "$PROJECT_ROOT"

# === Job info ===
echo "=========================================="
echo "Array job:       $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID"
echo "Node:            $(hostname)"
echo "GPU:             $CUDA_VISIBLE_DEVICES"
echo "Config:          $CONFIG"
echo "Method / Side:   $METHOD / $SIDE"
echo "Grad override:   ${GRAD_OVERRIDE[*]:-<none>}"
echo "Seed:            $SEED"
echo "Group:           $GROUP"
echo "Exp name:        $EXP"
echo "WandB tags:      $WANDB_TAGS"
echo "Start time:      $(date)"
echo "=========================================="

srun "$PYTHON_BIN" "${PROJECT_ROOT}/train.py" \
    --config "$CONFIG" \
    --seed "$SEED" \
    --device cuda \
    "${GRAD_OVERRIDE[@]}" \
    --network.ln-critic \
    --ppo.no-use-popart \
    --exp-name "$EXP" \
    --logging.wandb-group "$GROUP" \
    --logging.test-interval 2 \
    --logging.no-save-best \
    --logging.log-dir "$TRAIN_LOG_DIR"

echo "=========================================="
echo "End time:        $(date)"
echo "=========================================="
