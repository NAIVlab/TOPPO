# Environments

Use **one environment per family**. The ARS baselines need a different Python version, so they
cannot share an environment with the others. The Meta-World and AMP families both use Python 3.12,
but the AMP code pins numpy, MuJoCo and the Newton simulator stack to the versions its runs used,
which differ from those of the Meta-World runs; keeping them apart reproduces each family's
recorded stack.

| Family | Python | Notes |
|---|---|---|
| `experiments/metaworld_ppo` | 3.12 | Meta-World 3.0.0, MuJoCo, tianshou |
| `experiments/sac_baselines_ars` | **exactly 3.10** | 3.11+ rejects upstream's mutable dataclass defaults in `value_net.py`. JAX-based. |
| `experiments/amp_humanoid` | 3.12 | MimicKit + Newton 1.3.0 (MuJoCo-Warp), torch 2.10.0+cu128; one NVIDIA GPU with ≥ 20 GB. Pins and install order in its `requirements.txt`. |

## Meta-World PPO

```bash
cd experiments/metaworld_ppo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MUJOCO_GL=egl        # headless; without it MuJoCo tries to open a display
unset DISPLAY
```

The tested versions (as W&B recorded them for 822 of the 823 Meta-World PPO runs, and the exact
pins of the headline groups) are listed at the top of `experiments/metaworld_ppo/requirements.txt`. On a cluster,
`slurm/setup_env.sh` builds the conda environment; set `TORCH_VERSION_SPEC` at the top of that
script to pin torch from the CUDA wheel index.

MT50 runs 50 MuJoCo worker processes. Give it 16–24 CPU cores and ~18 GB RAM; the GPU
requirement is small (the headline runs peaked at 1.65 GB on an RTX 3090 and 2.16 GB on an H200
NVL). **This is a CPU-bound workload** — wall-clock scales with core count, not GPU class, which
is why timing comparisons must state the host.

## ARS baselines

See `experiments/sac_baselines_ars/slurm/site_a100/setup_env_mw3.sh`. Watch for the
`nvidia-cudnn-cu12` pin: a mismatched version makes JAX fall back to CPU **silently**, turning a
multi-hour run into a multi-day one with no error.

## AMP humanoid

See `experiments/amp_humanoid/README.md` (Install). It installs torch from the CUDA 12.8 wheel
index, then the pins of `experiments/amp_humanoid/requirements.txt` (the versions recorded in the
evaluation records: newton 1.3.0, warp-lang 1.15.0, mujoco 3.10.0, mujoco-warp 3.10.0.1), and
`setup.sh` assembles MimicKit `2ed1e6c` with our patch and files. Do not install `newton[sim]`,
which pins an older MuJoCo. The analysis scripts there need only Python with numpy, scipy, PyYAML
and matplotlib, no GPU and no simulator.

## BLAS threading

The env workers are separate processes and the networks are small, so extra BLAS threads buy
nothing and oversubscribe the node. The ARS launchers, and
`experiments/sac_baselines_ars/scripts/cluster_env.sh` for a seed run by hand, set
`OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to 1 (50 lockstep env workers per
job). The PPO launchers set no cap; if you add one, keep it at or below the allocated core count
(the paper's timing study set all three to its 16 allocated cores).
