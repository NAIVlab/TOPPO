#!/bin/bash
#SBATCH --job-name=mt50_sac
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1,lscratch:80
#SBATCH --cpus-per-task=16
#SBATCH --mem=60g
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
# ARS-family / critic-balancing MT50 (MetaWorld 3.0.0, "metaworld3" domain) on
# an A100 Slurm site.  Site conventions:
# job shape calibrated on an RTX 3090 with ~12 cores: ~57 it/s
# busy, MaxRSS 48 G.  The site's A100 nodes are 64 CPUs / 4 GPUs -> the enforced
# per-GPU allotment is 16 CPUs (= 8 physical cores; MORE PENDS FOREVER with
# no error — documented site behavior), so expect ~40-60 it/s and
# ~17-24 h/run (ars_ln1024 ~28-36 h) inside the 3-day walltime.
# --mem=60g (48 G measured peak) packs 4 jobs per 243 GB node cleanly.
#
# Usage (from the ARS repo root; `mkdir -p logs` once first):
#   sbatch [--job-name=<name>] slurm/site_a100/mt50_run.sbatch <arm> [seed] [extra k=v...]
#   arm: ars | ars_ln | ars_ln1024 | ars_lora | mtsac | mtsac_cb | mtsac_fullstack
#   For the 10-seed grid use slurm/site_a100/launch_mt50_grid.sh.
#
# wandb: project ${WANDB_PROJECT:-toppo-repro}, name
# sac_mt50_<arm>_s<seed>${NAME_SUFFIX}, group = name minus _s<seed>, tags
# mt50,sac.  The public W&B runs carry the mt50v2_sac_* group names of the
# root docs/RUNS.md instead (see the arm table in README.md); append
# wandb_group=<that group> as an extra k=v to log under it.  WANDB_MODE
# defaults to online, which needs outbound access from the compute nodes;
# set WANDB_MODE=offline to log locally and `wandb sync` from a login node.
# Needs WANDB_API_KEY in the job env; when it is unset, ~/.wandb_api_key is
# read instead.
#
# Env-var knobs (all optional):
#   BW_CONDA_ROOT (default ${CLUSTER_SCRATCH:-/scratch/$USER}/miniforge3)
#   BW_CONDA_ENV  (default ars_mw3 — build with slurm/site_a100/setup_env_mw3.sh)
#   MAX_STEPS     (default 2000000 loop iterations = 2M/task = 100M total)
#   NAME_SUFFIX   (default empty; use it to disambiguate concurrent waves)
#   NODE_SCRATCH  (node-local scratch root for TMPDIR and async checkpoints;
#                  default /tmp)
#   WANDB_ENTITY, WANDB_PROJECT, WANDB_MODE, WANDB_TAGS
#   CHECKPOINT_REMOTE  rclone remote, e.g. myremote:toppo/checkpoints.  Setting
#                 it turns checkpoint offload on (CHECKPOINT_OFFLOAD then
#                 defaults to 1, else to 0; OFFLOAD=1 without a remote aborts).
#   CHECKPOINT_ASYNC (default 1: slurm/site_a100/checkpoint_worker.py uploads
#                 in the background), CHECKPOINT_WORKER_INTERVAL (default 30 s),
#                 CHECKPOINT_RCLONE_TIMEOUT (default 21600 s)
#   CLUSTER_HOME  (offload only: if set, rclone reads
#                 ${CLUSTER_HOME}/.config/rclone/rclone.conf)
#
# CHECKPOINT/RESUME: resume=auto — resubmitting the same command continues
# from checkpoints/<group>/<name>/latest (learner + replay buffer + ARS scale
# + wandb run id; ars_lora also restores its LoRA sidecar state).  Rolling
# latest+prev with the filled f32 buffer is ~40 GB/run — see the quota math
# for your site before launching many seeds, or offload them.
set -uo pipefail
ARM="${1:?arm required: ars|ars_ln|ars_ln1024|ars_lora|mtsac|mtsac_cb|mtsac_fullstack}"; S="${2:-0}"
shift 2 2>/dev/null || shift $# 2>/dev/null || true

BW_CONDA_ROOT="${BW_CONDA_ROOT:-${CLUSTER_SCRATCH:-/scratch/$USER}/miniforge3}"
BW_CONDA_ENV="${BW_CONDA_ENV:-ars_mw3}"
MAX_STEPS="${MAX_STEPS:-2000000}"
NAME_SUFFIX="${NAME_SUFFIX:-}"

source "$BW_CONDA_ROOT/etc/profile.d/conda.sh" \
  || { echo "[mt50_run] no conda under BW_CONDA_ROOT=$BW_CONDA_ROOT; set BW_CONDA_ROOT" >&2; exit 2; }
conda activate "$BW_CONDA_ENV" \
  || { echo "[mt50_run] no conda env '$BW_CONDA_ENV'; build it with slurm/site_a100/setup_env_mw3.sh" >&2; exit 2; }
export CHECKPOINT_REMOTE="${CHECKPOINT_REMOTE:-}"
if [ -n "$CHECKPOINT_REMOTE" ]; then offload_default=1; else offload_default=0; fi
export CHECKPOINT_OFFLOAD="${CHECKPOINT_OFFLOAD:-$offload_default}"
if [ "$CHECKPOINT_OFFLOAD" != 0 ] && [ -z "$CHECKPOINT_REMOTE" ]; then
  echo "[mt50_run] CHECKPOINT_OFFLOAD=$CHECKPOINT_OFFLOAD needs CHECKPOINT_REMOTE (an rclone remote, e.g. myremote:toppo/checkpoints); or set CHECKPOINT_OFFLOAD=0" >&2
  exit 2
fi
export CHECKPOINT_ASYNC="${CHECKPOINT_ASYNC:-1}"
if [ "$CHECKPOINT_OFFLOAD" != 0 ] && [ -n "${CLUSTER_HOME:-}" ]; then
  export RCLONE_CONFIG="${CLUSTER_HOME}/.config/rclone/rclone.conf"
fi
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1   # user-site packages must not shadow the env (site requirement)
# jax cuda12_pip keeps its CUDA libs in the env's nvidia/*/lib dirs — build the
# path from the ACTIVE env instead of hardcoding a username.
NV_LIBS=$(python - <<'PY'
import glob, os, sys
base = os.path.join(sys.prefix, "lib",
                    f"python{sys.version_info.major}.{sys.version_info.minor}",
                    "site-packages", "nvidia")
print(":".join(sorted(glob.glob(os.path.join(base, "*", "lib")))))
PY
)
export LD_LIBRARY_PATH="$NV_LIBS:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$PWD:${PYTHONPATH:-}"
# Online: curves stream live, no sync step.  Override with WANDB_MODE=offline
# if the compute nodes have no outbound access (then: wandb sync --sync-all
# wandb/ from a login node).
export WANDB_MODE="${WANDB_MODE:-online}"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.wandb_api_key" ]; then
  export WANDB_API_KEY="$(<"$HOME/.wandb_api_key")"
fi
# Optional WANDB_ENTITY pins the entity AT INIT: offline runs otherwise
# inherit the default entity of whichever login later runs `wandb sync`.
if [ -n "${WANDB_ENTITY:-}" ]; then export WANDB_ENTITY; fi
export MUJOCO_GL=egl; unset DISPLAY
# 50 lockstep env workers: keep per-process BLAS single-threaded
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="${NODE_SCRATCH:-/tmp}/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"

python -c "import jax; d=jax.devices(); assert d and d[0].platform=='gpu', f'JAX has no GPU backend ({d}) — cudnn must be 8.9.7.29, see setup_env_mw3.sh'" \
  || { echo "[mt50_run] JAX GPU check FAILED — aborting before wasting walltime"; exit 3; }

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
group="sac_mt50_${ARM}${NAME_SUFFIX}"   # NOT ${name%_s*}: that would eat the suffix
# Offload + resume wiring.  BOTH modes restore from the remote when the local
# checkpoint is absent (sync-mode offload DELETES local copies after upload,
# so without this a resubmit would silently restart from scratch).  Without
# offload, ckpt_dir stays "" and the trainer uses checkpoints/<group>/<name>.
ckpt_dir=""
worker_pid=""
if [ "$CHECKPOINT_OFFLOAD" != 0 ] && [ -n "$CHECKPOINT_REMOTE" ]; then
  if [ "$CHECKPOINT_ASYNC" != 0 ]; then
    ckpt_dir="${NODE_SCRATCH:-/tmp}/${SLURM_JOB_ID}/checkpoints/${group}/${name}"
  else
    ckpt_dir="checkpoints/${group}/${name}"
  fi
  mkdir -p "$ckpt_dir"
  if [ ! -d "$ckpt_dir/latest" ] && command -v rclone >/dev/null 2>&1; then
    rclone copy "${CHECKPOINT_REMOTE%/}/${group}/${name}/latest" "$ckpt_dir/latest" --transfers 1 --checkers 2 --retries 3 || true
    # a failed/absent remote can leave a torn dir; resume=auto also verifies
    # completeness, but drop obvious junk here
    [ -f "$ckpt_dir/latest/meta.json" ] || rm -rf "$ckpt_dir/latest"
  fi
  if [ "$CHECKPOINT_ASYNC" != 0 ]; then
    python "$SLURM_SUBMIT_DIR/slurm/site_a100/checkpoint_worker.py" "$ckpt_dir" "$CHECKPOINT_REMOTE" --interval "${CHECKPOINT_WORKER_INTERVAL:-30}" &
    worker_pid=$!
  fi
fi

echo "host=$(hostname) arm=$ARM seed=$S name=$name max_steps=$MAX_STEPS extra=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# Protocol: MetaWorld 3.0.0 V2 reward, H=500,
# eval = deterministic, all 50 benchmark goals/task, every 1M total env steps
# (mtrl test/mean_win_rate + test_win_rate/<task> keys, x-axis = total steps).
# shellcheck disable=SC2086
python -u run_experiments.py --main_module_name "$MOD" --seeds "$S" \
  --config_args domain=metaworld3 env_name=MT50 activation=tanh n_layer=4 \
    batch_size=100 replay_buffer_size=500000 n_reset=6 \
    eval_interval=20000 eval_distribution=det rf_version=v2 \
    max_steps="$MAX_STEPS" checkpoint_interval=100000 resume=auto checkpoint_dir="$ckpt_dir" \
    $AA \
    project="${WANDB_PROJECT:-toppo-repro}" wandb_name="$name" \
    wandb_group="$group" wandb_tags="${WANDB_TAGS:-mt50,sac}" "$@"
status=$?
if [ -n "$worker_pid" ]; then
  kill "$worker_pid" 2>/dev/null || true
  wait "$worker_pid" 2>/dev/null || true
  python "$SLURM_SUBMIT_DIR/slurm/site_a100/checkpoint_worker.py" "$ckpt_dir" "$CHECKPOINT_REMOTE" --once
fi
exit "$status"
