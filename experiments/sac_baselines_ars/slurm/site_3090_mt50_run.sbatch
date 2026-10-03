#!/bin/bash
#SBATCH --job-name=mt50_sac
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out
# RTX-3090 adaptation of slurm/site_a100/mt50_run.sbatch (same protocol, arms
# and naming; see its header for the wandb naming and the env-var knobs):
#   - no node-local scratch mount here: TMPDIR -> node-local /tmp; checkpoints -> shared FS
#     (resume=auto continues after preemption; --requeue resubmits); no offload
#   - conda: BW_CONDA_ROOT (default ${CONDA_ROOT}), env BW_CONDA_ENV (default ars_mw3)
#   - mtsac_fullstack = mtsac_cb + actor_method=pcgrad (the paper's full
#     four-module stack ported to MTSAC; PCGrad-a path is the upstream ARS
#     implementation, see learner_sac_mt.py)
# Usage: sbatch slurm/site_3090_mt50_run.sbatch <arm> [seed] [extra k=v...]
#   arm: ars | ars_ln | ars_ln1024 | ars_lora | mtsac | mtsac_cb | mtsac_fullstack
set -u
ARM="${1:?arm required: ars|ars_ln|ars_ln1024|ars_lora|mtsac|mtsac_cb|mtsac_fullstack}"; S="${2:-0}"
shift 2 2>/dev/null || shift $# 2>/dev/null || true
BW_CONDA_ROOT="${BW_CONDA_ROOT:-${CONDA_ROOT:?set CONDA_ROOT (or BW_CONDA_ROOT) to your conda install}}"
BW_CONDA_ENV="${BW_CONDA_ENV:-ars_mw3}"
MAX_STEPS="${MAX_STEPS:-2000000}"
NAME_SUFFIX="${NAME_SUFFIX:-}"
source "${BW_CONDA_ROOT}/etc/profile.d/conda.sh" \
  || { echo "[site_3090_mt50_run] no conda under BW_CONDA_ROOT=$BW_CONDA_ROOT" >&2; exit 2; }
conda activate "$BW_CONDA_ENV" \
  || { echo "[site_3090_mt50_run] no conda env '$BW_CONDA_ENV'; see slurm/site_a100/setup_env_mw3.sh" >&2; exit 2; }
export PYTHONNOUSERSITE=1
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.wandb_api_key" ]; then
  export WANDB_API_KEY="$(<"$HOME/.wandb_api_key")"
fi
if [ -n "${WANDB_ENTITY:-}" ]; then export WANDB_ENTITY; fi  # optional; see the A100 sbatch
export MUJOCO_GL=egl; unset DISPLAY
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"

python -c "import jax; d=jax.devices(); assert d and d[0].platform=='gpu', f'JAX has no GPU backend ({d})'" \
  || { echo "[site_3090_mt50_run] JAX GPU check FAILED"; exit 3; }

MOD=train_ars
case "$ARM" in
  ars)        AA="use_ars=True hidden_dim=400" ;;
  ars_ln)     AA="use_ars=True hidden_dim=400 critic_layernorm=True critic_init_layernorm=True" ;;
  ars_ln1024) AA="use_ars=True hidden_dim=1024 critic_layernorm=True critic_init_layernorm=True" ;;
  mtsac)      AA="use_ars=False hidden_dim=400" ;;
  mtsac_cb)   AA="use_ars=False hidden_dim=400 critic_layernorm=True critic_method=fairgrad use_popart=True popart_method=ema" ;;
  mtsac_fullstack) AA="use_ars=False hidden_dim=400 critic_layernorm=True critic_method=fairgrad use_popart=True popart_method=ema actor_method=pcgrad" ;;
  ars_lora)   MOD=train_ars_lora
              AA="hidden_dim=400 critic_layernorm=True critic_init_layernorm=True rank=16 threshold=0.65" ;;
  *) echo "bad arm: '$ARM'" >&2; exit 2 ;;
esac
name="sac_mt50_${ARM}_s${S}${NAME_SUFFIX}"
group="sac_mt50_${ARM}${NAME_SUFFIX}"
ckpt_dir="checkpoints/${group}/${name}"
mkdir -p "$ckpt_dir"

echo "host=$(hostname) arm=$ARM seed=$S name=$name max_steps=$MAX_STEPS extra=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# shellcheck disable=SC2086
python -u run_experiments.py --main_module_name "$MOD" --seeds "$S" \
  --config_args domain=metaworld3 env_name=MT50 activation=tanh n_layer=4 \
    batch_size=100 replay_buffer_size=500000 n_reset=6 \
    eval_interval=20000 eval_distribution=det rf_version=v2 \
    max_steps="$MAX_STEPS" checkpoint_interval=100000 resume=auto checkpoint_dir="$ckpt_dir" \
    $AA \
    project="${WANDB_PROJECT:-toppo-repro}" wandb_name="$name" \
    wandb_group="$group" wandb_tags="${WANDB_TAGS:-mt50,sac}" "$@"
