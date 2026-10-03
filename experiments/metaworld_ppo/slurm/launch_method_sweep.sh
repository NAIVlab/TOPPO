#!/bin/bash
# ==============================================================================
# Method-sweep launcher — submits two SLURM arrays covering 200 runs
# (critic LayerNorm `--network.ln-critic` on every run, PopArt explicitly OFF
# via `--ppo.no-use-popart` in both array sbatch files — this sweep is the
# LN-c-only rung, tag `hpc_ppo_lnc_v1`, groups `*_lnc`):
#   MT10: 50 array tasks × 2 runs/task = 100 runs (10 configs × 10 seeds)
#   MT50: 100 array tasks × 1 run/task = 100 runs (10 configs × 10 seeds)
#
# Matrix (10 configs per benchmark):
#   1 × default                (actor=default, critic=default)
#   3 × method × 3 sides       (both, actor-only, critic-only) for
#       methods ∈ {pcgrad, cagrad, fairgrad}
#
# "side" semantics (v2.13+ per-side design — no global scope):
#   both   — actor_method = critic_method = <method> (from YAML)
#   actor  — actor_method = <method>, critic_method = default
#   critic — actor_method = default, critic_method = <method>
#
# Group names: the names this sweep emits, `hpc_ppo_<bench>_<method>_<side>_lnc`
# with tag `hpc_ppo_lnc_v1`, are not docs/RUNS.md names and appear on no
# published run.  The paper's LN-c-only rows
# (mt{10,50}v2_ppo_ad_cd_nnone_lnc_vc1_400x3,
# mt{10,50}v2_ppo_ad_cfg1_nnone_lnc_vc1_400x3) were launched from
# configs/ppo_mt{10,50}_default.yaml as hpc_ppo_mt{10,50}_ad_{cd,cfg}_lnc (as
# their recorded logging.wandb_group shows), not by this sweep.  To reproduce
# a RUNS.md row, use slurm/train.sbatch with that row's config and overrides.
# The group is passed to train.py as --logging.wandb-group in the array
# sbatch files.
#
# Usage:
#   bash slurm/launch_method_sweep.sh              # submit MT10 + MT50
#   bash slurm/launch_method_sweep.sh mt10         # MT10 only
#   bash slurm/launch_method_sweep.sh mt50         # MT50 only
#   bash slurm/launch_method_sweep.sh --dry-run    # preview, no sbatch calls
# ==============================================================================

set -e

DRY_RUN=false
FILTER=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        mt10|MT10) FILTER="mt10" ;;
        mt50|MT50) FILTER="mt50" ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MT10_SBATCH="${SCRIPT_DIR}/train_mt10_array.sbatch"
MT50_SBATCH="${SCRIPT_DIR}/train_mt50_array.sbatch"

# The sbatch files write their logs to the RELATIVE path run_logs/... because
# SLURM does not expand shell variables inside #SBATCH directives.  We submit
# with --chdir so that path resolves here, and mkdir first because SLURM does
# not auto-create --output parents.  TOPPO_ROOT, if set, must be
# <clone>/experiments/metaworld_ppo (the directory holding train.py), not the
# repository root.
HPC_PROJECT_ROOT="${TOPPO_ROOT:-$PROJECT_DIR}"
HPC_RUN_LOG_DIR="${HPC_PROJECT_ROOT}/run_logs"
HPC_TRAIN_LOG_DIR="${HPC_PROJECT_ROOT}/logs/baseline"
HPC_TMP_DIR="${HPC_PROJECT_ROOT}/tmp"

# Keep these two arrays in lockstep with the same tables in
# train_mt10_array.sbatch and train_mt50_array.sbatch (CFG_IDX 0..9).
METHODS=(default pcgrad pcgrad pcgrad cagrad cagrad cagrad fairgrad fairgrad fairgrad)
SIDES=(both both actor critic both actor critic both actor critic)

preview_bench() {
    local bench=$1
    echo "=== ${bench^^} (10 configs × 10 seeds = 100 runs) ==="
    for i in "${!METHODS[@]}"; do
        local group="hpc_ppo_${bench}_${METHODS[$i]}_${SIDES[$i]}_lnc"
        echo "  $group  (seeds 0..9)"
    done
    echo ""
}

if [ "$DRY_RUN" = true ]; then
    echo "=== DRY RUN MODE (no jobs will be submitted) ==="
    echo ""
    if [ -z "$FILTER" ] || [ "$FILTER" = "mt10" ]; then
        preview_bench mt10
        echo "[DRY RUN] sbatch $MT10_SBATCH   (array 0-49%30, packed 2 runs/task, ~48h)"
        echo ""
    fi
    if [ -z "$FILTER" ] || [ "$FILTER" = "mt50" ]; then
        preview_bench mt50
        echo "[DRY RUN] sbatch $MT50_SBATCH   (array 0-99%30, 1 run/task, ~96h)"
        echo ""
    fi
    echo "=== Total runs that would be submitted: "
    if [ -z "$FILTER" ]; then
        echo "MT10: 100, MT50: 100, total 200"
    elif [ "$FILTER" = "mt10" ]; then
        echo "MT10: 100"
    else
        echo "MT50: 100"
    fi
    echo ""
    echo "Would pre-create (skipped in dry-run):"
    echo "  $HPC_RUN_LOG_DIR"
    echo "  $HPC_TRAIN_LOG_DIR"
    echo "  $HPC_TMP_DIR"
    exit 0
fi

# Pre-create directories that SLURM / the training script will write into.
# Must run BEFORE sbatch because SLURM --output parent dir is resolved at
# submit time, and MTRLLogger's log-dir is created only after the job starts.
echo "Pre-creating HPC log directories under ${HPC_PROJECT_ROOT}:"
for d in "$HPC_RUN_LOG_DIR" "$HPC_TRAIN_LOG_DIR" "$HPC_TMP_DIR"; do
    if [ ! -d "$d" ]; then
        mkdir -p "$d" && echo "  created $d"
    else
        echo "  exists  $d"
    fi
done
echo ""

SUBMITTED=()
# Submit MT50 first so it ages ahead of MT10 in the queue (which also
# carries --nice=100 in its sbatch for a belt-and-suspenders bias).
if [ -z "$FILTER" ] || [ "$FILTER" = "mt50" ]; then
    echo "Submitting MT50 array..."
    MT50_ID=$(sbatch --parsable --chdir="$HPC_PROJECT_ROOT" "$MT50_SBATCH")
    echo "  MT50 array job ID: $MT50_ID  (100 tasks × 1 run = 100 runs)"
    SUBMITTED+=("mt50=$MT50_ID")
fi
if [ -z "$FILTER" ] || [ "$FILTER" = "mt10" ]; then
    echo "Submitting MT10 array..."
    MT10_ID=$(sbatch --parsable --chdir="$HPC_PROJECT_ROOT" "$MT10_SBATCH")
    echo "  MT10 array job ID: $MT10_ID  (50 tasks × 2 runs = 100 runs)"
    SUBMITTED+=("mt10=$MT10_ID")
fi

echo ""
echo "=== Submitted: ${SUBMITTED[*]} ==="
echo "Monitor:   squeue -u \$USER"
echo "Cancel:    scancel <job_id>  or  scancel -u \$USER"
echo "Logs:      ${HPC_RUN_LOG_DIR}/"
echo "WandB:     tag 'hpc_ppo_lnc_v1' filters all 200 runs (+ 'mt10' / 'mt50' splits)"
