#!/bin/bash
# ==============================================================================
# Launch MT10 + MT50 default-combiner MTPPO baselines (LN-c + PopArt rung, not
# the `vanilla` rung 0): actor=critic=default × 10 seeds.
# 20 runs total (10 mt10 + 10 mt50) in 15 jobs (MT10 packs 2 seeds per job).
#
# Method: default (group label `ad_cd` for symmetry with fairgrad sweep's
# a<x>_c<y> codes).  PopArt + critic LayerNorm are toggled on for every run
# via `--ppo.use-popart --network.ln-critic`; base YAML is configs/ppo_<bench>_default.yaml.
# Both flags are passed explicitly and are NOT redundant bookkeeping: they pin
# this sweep to the PopArt + LN-c rung (LN = C / PopArt = yes) independently of
# what the base YAML happens to default to.  Do not remove them.
#
# Group names: `ppo_<bench>_ad_cd_popart_lnc` is the internal name this sweep
# was launched under, not a canonical docs/RUNS.md name.  RUNS.md lists the
# same config with no overrides as mt{10,50}v2_ppo_ad_cd_npaw_lnc_vc1_400x3; to
# log under that name, edit `group` and the exp names below (`group` is
# passed as --logging.wandb-group).
#
# Usage:
#   bash slurm/launch_baselines.sh             # submit all jobs
#   bash slurm/launch_baselines.sh --dry-run   # preview commands only
#   bash slurm/launch_baselines.sh mt10        # MT10 only
#   bash slurm/launch_baselines.sh mt50        # MT50 only
#   bash slurm/launch_baselines.sh mt10 --dry-run
#
# For the pcgrad / cagrad / fairgrad per-side sweep (LN-c only, PopArt off),
# see launch_method_sweep.sh.
# ==============================================================================

set -e

DRY_RUN=false
FILTER=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        mt10|MT10) FILTER="mt10" ;;
        mt50|MT50) FILTER="mt50" ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    echo "=== DRY RUN MODE (no jobs will be submitted) ==="
    echo ""
fi

# === Project paths ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
# MT50 → 1 seed/GPU via train.sbatch.  MT10 → 2 seeds/GPU via the packed
# sbatch (one job per pair, parameterized through SEED_A/SEED_B).
SBATCH_MT50="${SCRIPT_DIR}/train.sbatch"
SBATCH_MT10_PACK="${SCRIPT_DIR}/train_mt10_pack.sbatch"

# === Experiment matrix ===
# METHODS = YAML suffix (configs/ppo_<bench>_<method>.yaml).
# LABELS  = group / exp / job-name infix; `ad_cd` marks (actor=default,
# critic=default) for symmetry with the fairgrad sweep's a<x>_c<y> codes.
BENCHMARKS=(mt10 mt50)
METHODS=(default)
LABELS=(ad_cd)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# Base WandB tag for every run launched by this script (mt10/mt50 sub-tag
# appended per-run).  Both train.sbatch (MT50) and train_mt10_pack.sbatch
# inherit WANDB_TAGS via --export=ALL.
# `lnc` = critic LayerNorm enabled via --network.ln-critic.
BASE_TAG="hpc_ppo_popart_lnc_v1"

# === Create output directory ===
mkdir -p "${PROJECT_DIR}/run_logs"

# === Submit jobs ===
TOTAL=0
for bench in "${BENCHMARKS[@]}"; do
    # Filter by benchmark if specified
    if [ -n "$FILTER" ] && [ "$FILTER" != "$bench" ]; then
        continue
    fi

    for i in "${!METHODS[@]}"; do
        method=${METHODS[$i]}
        label=${LABELS[$i]}
        config="configs/ppo_${bench}_${method}.yaml"

        # Verify config exists
        if [ ! -f "${PROJECT_DIR}/${config}" ]; then
            echo "WARNING: Config not found: ${config}, skipping"
            continue
        fi

        group="ppo_${bench}_${label}_popart_lnc"
        # Export EXTRA_ARGS / WANDB_TAGS into the parent shell env so
        # sbatch --export=ALL propagates them to the job.  Doing it this way
        # (rather than inlining into --export=...) sidesteps sbatch's
        # comma-delimited parser choking on spaces in the value.
        # `--network.ln-critic` enables pre-activation LayerNorm in the critic
        # MLP hidden layers (v2.19); on for every run in this sweep.
        export EXTRA_ARGS="--ppo.use-popart --network.ln-critic --logging.wandb-group ${group}"
        export WANDB_TAGS="${BASE_TAG},${bench}"

        if [ "$bench" = "mt10" ]; then
            # Pack 2 seeds per GPU: 5 sbatch jobs per (config, label) covering
            # seeds 0..9.  Each job has --time=48h, --nice=100 (yields to MT50).
            for pair in 0 1 2 3 4; do
                seed_a=$((pair * 2))
                seed_b=$((pair * 2 + 1))
                exp_a="ppo_${bench}_${label}_popart_lnc_s${seed_a}"
                exp_b="ppo_${bench}_${label}_popart_lnc_s${seed_b}"
                job_name="${bench}_${label}_popart_lnc_s${seed_a}-${seed_b}"

                if [ "$DRY_RUN" = true ]; then
                    echo "[DRY RUN] sbatch --job-name=${job_name} (mt10 pack) ... CONFIG=${config} SEED_A=${seed_a} SEED_B=${seed_b} EXP_NAME_A=${exp_a} EXP_NAME_B=${exp_b} EXTRA_ARGS='${EXTRA_ARGS}' WANDB_TAGS=${WANDB_TAGS}"
                else
                    echo -n "Submitting ${job_name}... "
                    sbatch \
                        --job-name="${job_name}" \
                        --chdir="${PROJECT_DIR}" \
                        --export=ALL,PROJECT_DIR="${PROJECT_DIR}",CONFIG_FILE="${config}",SEED_A="${seed_a}",SEED_B="${seed_b}",EXP_NAME_A="${exp_a}",EXP_NAME_B="${exp_b}" \
                        "${SBATCH_MT10_PACK}"
                fi
                TOTAL=$((TOTAL + 2))
            done
        else
            # MT50: 1 seed per GPU (50 env workers already saturate CPU).
            for seed in "${SEEDS[@]}"; do
                job_name="${bench}_${label}_popart_lnc_s${seed}"
                exp_name="ppo_${bench}_${label}_popart_lnc_s${seed}"

                if [ "$DRY_RUN" = true ]; then
                    echo "[DRY RUN] sbatch --job-name=${job_name} ... CONFIG=${config} SEED=${seed} EXTRA_ARGS='${EXTRA_ARGS}' WANDB_TAGS=${WANDB_TAGS}"
                else
                    echo -n "Submitting ${job_name}... "
                    sbatch \
                        --job-name="${job_name}" \
                        --chdir="${PROJECT_DIR}" \
                        --export=ALL,PROJECT_DIR="${PROJECT_DIR}",CONFIG_FILE="${config}",SEED="${seed}",EXP_NAME="${exp_name}" \
                        "${SBATCH_MT50}"
                fi
                TOTAL=$((TOTAL + 1))
            done
        fi
    done
done

echo ""
echo "=== Total: ${TOTAL} runs ==="
if [ "$DRY_RUN" = false ]; then
    echo "Use 'squeue -u \$USER' to check job status"
    echo "Use 'scancel <job_id>' to cancel a specific job"
    echo "Logs will be in: ${PROJECT_DIR}/run_logs/"
    echo "WandB tag '${BASE_TAG}' filters all runs"
fi
