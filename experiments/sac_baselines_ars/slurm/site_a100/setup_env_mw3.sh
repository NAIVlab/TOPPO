#!/bin/bash
# Build the ars_mw3 conda env on an A100 Slurm site (login node — it has the proxy).
# MetaWorld-3.0.0 / JAX stack for the MT50 baseline runs.
#
#   bash slurm/site_a100/setup_env_mw3.sh
#
# Site conventions: miniforge under
# ${CLUSTER_SCRATCH:-/scratch/$USER} (NOT /home, which is quota-limited), pip --no-user, PYTHONNOUSERSITE=1.
#
# PYTHON MUST BE 3.10: python 3.11+ rejects upstream ARS's mutable dataclass
# defaults (value_net.py) at import time.
# THE cudnn TRAP: jaxlib 0.4.24+cuda12.cudnn89 needs
# nvidia-cudnn-cu12==8.9.7.29; pip happily resolves 9.x instead, and JAX then
# falls back to CPU with ONE quiet log line (observed: 1.6 it/s vs 57).
# The sbatch hard-fails on this, but pin it here so it never happens.
set -euo pipefail
BW_CONDA_ROOT="${BW_CONDA_ROOT:-${CLUSTER_SCRATCH:-/scratch/$USER}/miniforge3}"
source "$BW_CONDA_ROOT/etc/profile.d/conda.sh"
export PYTHONNOUSERSITE=1

# WARNING: re-running DELETES and rebuilds ars_mw3 — do NOT run while jobs
# using this env are queued or running.
if command -v squeue >/dev/null 2>&1 && [ -n "$(squeue -u "$USER" -h 2>/dev/null | head -1)" ]; then
  echo "[setup] NOTE: you have jobs in the queue — aborting the env rebuild." >&2
  echo "[setup] Re-run after they finish, or FORCE_ENV_REBUILD=1 to override." >&2
  [ "${FORCE_ENV_REBUILD:-0}" = 1 ] || exit 2
fi
conda env remove -n ars_mw3 -y >/dev/null 2>&1 || true
conda create -n ars_mw3 python=3.10 -y
conda activate ars_mw3
python --version

# Run these pip lines IN ORDER — later lines pin down transitive deps the
# earlier lines float (the jax line first pulls numpy 2.x; line 3 lands the
# jax-0.4.24-compatible 1.26.4).  gymnasium/mujoco pinned to the exact
# versions of the validated wave.
python -m pip install --no-user --no-cache-dir "jax[cuda12_pip]==0.4.24" "nvidia-cudnn-cu12==8.9.7.29" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -m pip install --no-user --no-cache-dir flax==0.7.5 optax==0.1.7 chex==0.1.86 \
    orbax-checkpoint==0.4.8 tensorflow-probability==0.23.0
python -m pip install --no-user --no-cache-dir numpy==1.26.4 metaworld==3.0.0 \
    gymnasium==1.3.0 mujoco==3.10.0 wandb==0.17.0 \
    pyrallis==0.3.1 tqdm scipy packaging
python -m pip check

echo "[verify] imports + one real env step (CPU is fine on the login node):"
python - <<'EOF'
import jax, flax, numpy
print("jax", jax.__version__, "| flax", flax.__version__, "| numpy", numpy.__version__)
import tensorflow_probability.substrates.jax as tfp  # noqa
import gymnasium, metaworld, mujoco
print("gymnasium", gymnasium.__version__, "| mujoco", mujoco.__version__)
# importlib.metadata, NOT pkg_resources: setuptools >= 83 (what a fresh
# conda-forge py3.10 ships TODAY) removed pkg_resources entirely.
from importlib.metadata import version as _v
v = _v("nvidia-cudnn-cu12")
assert v.startswith("8.9"), f"cudnn pin broken: {v}"
b = metaworld.MT10(seed=0)
names = sorted(b.train_classes.keys())
env = b.train_classes[names[0]](reward_function_version="v2")
env.set_task([t for t in b.train_tasks if t.env_name == names[0]][0])
obs, _ = env.reset(seed=0)
env.step(env.action_space.sample())
print("metaworld MT env step OK | cudnn", v)
print("ALL OK")
EOF
echo "[setup] DONE — env ars_mw3 ready"
