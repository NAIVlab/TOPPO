#!/bin/bash
# ==============================================================================
# SLURM array job: MT50 FairGrad + PopArt + LN-critic sweep
#   Array 0..29 = 3 configs × 10 seeds
#   1 run per array task (MT50's 50 env workers saturate CPU; no packing).
#
# Matrix: actor ∈ {default, fairgrad}, critic = fairgrad (configs 0-1).
# Config 2 is an MT50-only deep variant of (fairgrad, fairgrad) with
# [400,400,400,400] actor/critic — asks whether fairgrad-on-actor benefits
# from extra capacity when paired with plain-fairgrad critic.  Every run
# has `--ppo.use-popart` and `--network.ln-critic` enabled.
#
# No launcher ships for this array.  Submit it from experiments/metaworld_ppo/
# (SLURM does not create the --output directory):
#   mkdir -p run_logs && sbatch slurm/train_mt50_fairgrad_array.sbatch
#
# The `hpc_ppo_mt50_a*_c*_popart_lnc` group names are the internal names this
# sweep was launched under, not canonical docs/RUNS.md names; GROUP below is
# passed as --logging.wandb-group.
# ==============================================================================

#SBATCH --job-name=hpc_ppo_mt50_fgsweep_popart_lnc
#SBATCH --output=run_logs/%x_%A_%a.out
#SBATCH --error=run_logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# 24 cpus for 50 env workers (~0.5 core/worker) — oversubscribed but a
# compromise with queue time.
#SBATCH --constraint=gpua100|gpuv100x|gpuv100
#SBATCH --time=96:00:00
#SBATCH --array=0-29
# Signal-driven graceful checkpoint: SLURM sends SIGUSR1 to the batch
# script 3600s before walltime (~2 epochs' headroom at MT50's ~30min/epoch).
# The trap forwards the signal to the srun child (python train.py);
# train.py snapshots latest.pth at the next epoch boundary and exits 0;
# this script then calls `scontrol requeue` so the array task restarts
# and resumes from latest.pth.
#SBATCH --signal=B:USR1@3600
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

# === Decode array task id → (actor, critic, seed) ===
# 3 configs indexed 0..2:
#   0: default  / fairgrad
#   1: fairgrad / fairgrad
#   2: fairgrad / fairgrad        (deep [400,400,400,400])
ACTORS=(default fairgrad fairgrad)
CRITICS=(fairgrad fairgrad fairgrad)

CFG_IDX=$((SLURM_ARRAY_TASK_ID / 10))
SEED=$((SLURM_ARRAY_TASK_ID % 10))
ACTOR=${ACTORS[$CFG_IDX]}
CRITIC=${CRITICS[$CFG_IDX]}

# Config 2 swaps the [400,400,400] defaults for a deeper [400,400,400,400]
# actor/critic; configs 0-1 inherit the YAML's [400,400,400].
NET_OVERRIDE=()
GROUP_SUFFIX=""
if [ "$CFG_IDX" -ge 2 ]; then
    NET_OVERRIDE=(--network.actor 400 400 400 400 --network.critic 400 400 400 400)
    GROUP_SUFFIX="_deep"
fi

# Short codes: default→d, fairgrad→fg
short() {
    case "$1" in
        default) echo "d" ;;
        fairgrad) echo "fg" ;;
    esac
}
A_SHORT=$(short "$ACTOR")
C_SHORT=$(short "$CRITIC")

GROUP="hpc_ppo_mt50_a${A_SHORT}_c${C_SHORT}${GROUP_SUFFIX}_popart_lnc"
EXP="${GROUP}_s${SEED}"
# Base YAML carries fairgrad α parameters; methods are explicitly
# overridden via CLI regardless of the YAML's own settings.
CONFIG="configs/ppo_mt50_fairgrad.yaml"

# === MuJoCo headless rendering ===
export MUJOCO_GL=egl
unset DISPLAY

# === WandB ===
# Credentials are NEVER stored in this repository. Provide the key by
# exporting WANDB_API_KEY, or by placing it in ~/.wandb_api_key (mode 600).
if [ -z "${WANDB_API_KEY:-}" ] && [ -f ~/.wandb_api_key ]; then
    export WANDB_API_KEY="$(<~/.wandb_api_key)"
fi
export WANDB_TAGS="hpc_ppo_popart_fgsweep_lnc_v1,mt50"

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
echo "Actor method:    $ACTOR"
echo "Critic method:   $CRITIC"
echo "Seed:            $SEED"
echo "Group:           $GROUP"
echo "Exp name:        $EXP"
echo "WandB tags:      $WANDB_TAGS"
echo "Start time:      $(date)"
echo "=========================================="

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

# --logging.save-interval = one epoch (1000000 env steps) → rolling
# latest.pth refreshes every epoch; graceful exit at most loses ~1 epoch.
srun "$PYTHON_BIN" "${PROJECT_ROOT}/train.py" \
    --config "$CONFIG" \
    --seed "$SEED" \
    --device cuda \
    --gradient.actor-method "$ACTOR" \
    --gradient.critic-method "$CRITIC" \
    --ppo.use-popart \
    --network.ln-critic \
    "${NET_OVERRIDE[@]}" \
    --exp-name "$EXP" \
    --logging.wandb-group "$GROUP" \
    --logging.test-interval 2 \
    --logging.no-save-best \
    --logging.save-interval 1000000 \
    --logging.log-dir "$TRAIN_LOG_DIR" &
SRUN_PID=$!

wait $SRUN_PID 2>/dev/null; EXIT_CODE=$?
while kill -0 $SRUN_PID 2>/dev/null; do wait $SRUN_PID 2>/dev/null; EXIT_CODE=$?; done

echo "=========================================="
echo "srun exit code:  $EXIT_CODE"
echo "GOT_SIGNAL:      $GOT_SIGNAL"
echo "End time:        $(date)"
echo "=========================================="

if [ "$GOT_SIGNAL" -eq 1 ]; then
    # Only requeue when the srun child exited cleanly.  A non-zero exit
    # after the signal means the graceful save raised (disk full,
    # checkpoint-format mismatch, etc.) — requeuing that silently would
    # mask a real failure, and the next re-entry would resume from a
    # stale/incomplete latest.pth without anyone noticing.
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[sbatch] Graceful shutdown succeeded — requeuing array task ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        exit 0
    else
        echo "[sbatch] Signal received BUT graceful save failed (exit=$EXIT_CODE) — NOT requeuing; inspect $RUN_LOG_DIR/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
        exit 1
    fi
fi

exit $EXIT_CODE
