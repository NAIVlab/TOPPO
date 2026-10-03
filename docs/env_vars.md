# Runtime environment variables

## Meta-World PPO (`experiments/metaworld_ppo`)

The PPO Slurm templates also take their job parameters (config, seed, run name, extra flags)
through `sbatch --export`; each template's header lists them.

| Variable | Effect |
|---|---|
| `MTRL_PCGRAD_CPU=1` | Crash workaround for PCGrad-actor configurations, including the TOPPO headline stack, on some GPUs. The CUDA `torch.dot` inside the PCGrad projection raises SIGFPE (exit 136) on certain nodes; this runs the identical arithmetic on CPU. Set it if you see exit 136; no shipped launcher sets it for you. |
| `MTRL_SOLVER_BENCH=<path.csv>` | Off by default (unset = no extra work, no file opened). When set, each FairGrad solve also runs a timed scipy `least_squares` reference on the same fp64 Gram (x0 = 1/K, bounds (1e-12, inf)) and discards the result, so training is unchanged. One CSV row is appended per solve: solve-only Newton and scipy times, Newton's own iterations/convergence/residual, and separate `fallback_*` columns. It re-implements the instrument behind `benchmarks/solver/solverbench_rfv1.csv`; for the exact regeneration command (expect 1,984 rows), see `benchmarks/solver/realgram_results.md`. Before a rerun, delete the old CSV (the hook appends) and the run's `logs/<exp_name>/` (train.py would resume from its checkpoints). |
| `SPIKE_ALPHA` | α used by `benchmarks/solver/fairgrad_spike_scenarios_a1.py` (default 1.0). |
| `WANDB_API_KEY` | Your key. Never stored in this repository. `train.py` reads it from the environment; the Slurm templates (PPO and ARS) fall back to `~/.wandb_api_key`. |
| `WANDB_ENTITY`, `WANDB_PROJECT` | **Training** logs to the config's `project_name` (override with `--project-name`) and does not read `$WANDB_PROJECT`. Every shipped config, and the code default, sets `project_name` to `toppo-repro`, a scratch name, so a reproduction never logs into the published project, `liyp1220-uci/TOPPO`, by default. The entity is your W&B default unless `$WANDB_ENTITY` is set. `$WANDB_MODE` has no effect on PPO training: `train.py` always passes the config's `wandb_mode` to `wandb.init`, and that argument takes precedence; use `--logging.wandb-mode offline` or `--logging.no-use-wandb` instead. **The analysis scripts** (`metrics.py`, `export_wandb.py`, `check_runs_index.py --wandb`) share one rule (`resolve_source`): they read `liyp1220-uci/TOPPO`, the published runs, unless `--entity`, `--project` or `$WANDB_PROJECT` is given. A project named by `--project`, else `$WANDB_PROJECT` (any name, `TOPPO` included), is read under `--entity`, else `$WANDB_ENTITY`, else your W&B default entity; `--entity` alone reads `<entity>/TOPPO`. `$WANDB_ENTITY` alone does not move the source. To read the runs you trained with the shipped configs, pass `--entity <you> --project toppo-repro` (or set `WANDB_PROJECT=toppo-repro`, which reads it under `$WANDB_ENTITY` or your default entity). |
| `MUJOCO_GL=egl` | Headless MuJoCo rendering. |
| `TOPPO_ROOT` | Must point to `<clone>/experiments/metaworld_ppo` (the directory holding `train.py` and `slurm/`), not the repository root. Used by the PPO Slurm templates in place of the original absolute paths; when unset they use the submit directory. |
| `CONDA_ROOT`, `CONDA_ROOT_ALT` | Conda install for the PPO Slurm templates (`slurm/conda_init.sh`, which also tries `~/mambaforge`, `~/miniconda3`, `~/anaconda3` and `conda` on `PATH`). `CONDA_ROOT_ALT` is an optional parent directory searched for those three installs. `CONDA_ROOT` is also the default `BW_CONDA_ROOT` of the ARS template `slurm/site_3090_mt50_run.sbatch`, which aborts with a message if neither is set. |
| `OMP_NUM_THREADS` etc. | See below and `docs/environments.md`. |

## ARS / SAC baselines (`experiments/sac_baselines_ars`)

Read by `slurm/site_a100/mt50_run.sbatch` and `slurm/site_3090_mt50_run.sbatch` unless noted;
all optional. The A100 template's header lists them as well.

| Variable | Effect |
|---|---|
| `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_MODE`, `WANDB_TAGS` | `WANDB_ENTITY` is exported only when set. `WANDB_PROJECT` defaults to `toppo-repro` (as does the trainers' `project` key), so a re-run does not log into the published project by default; `WANDB_TAGS` defaults to `mt50,sac`, and `WANDB_MODE` to `online` (A100 template). The ARS trainers pass no mode to `wandb.init`, so `WANDB_MODE=offline` applies to them. |
| `BW_CONDA_ROOT`, `BW_CONDA_ENV` | Conda install and environment (default env `ars_mw3`). The root defaults to `${CLUSTER_SCRATCH:-/scratch/$USER}/miniforge3` in the A100 template and `slurm/site_a100/setup_env_mw3.sh`, and to `${CONDA_ROOT}` in the 3090 template. |
| `CLUSTER_SCRATCH` | Scratch root for that A100 default. |
| `MAX_STEPS` | Training-loop iterations (default 2000000; each steps all 50 task envs, so 100M env steps). |
| `NAME_SUFFIX` | Appended to run and group names, to keep concurrent waves apart. |
| `NODE_SCRATCH` | A100 template: node-local scratch root for `TMPDIR` and async checkpoints (default `/tmp`). |
| `SBATCH_EXTRA` | `scripts/launch_mt50.sh`: extra `sbatch` flags (partition, QOS, account). |
| `FORCE_ENV_REBUILD=1` | `slurm/site_a100/setup_env_mw3.sh`: rebuild `ars_mw3` even while you have jobs queued (otherwise it aborts). |
| `CHECKPOINT_REMOTE` | A100 template: rclone remote for checkpoint offload; setting it turns offload on. |
| `CHECKPOINT_OFFLOAD` | Default 1 if `CHECKPOINT_REMOTE` is set, else 0; 1 without a remote aborts. |
| `CHECKPOINT_ASYNC` | 1 = upload from a background worker (template default 1; trainer default 0). |
| `CHECKPOINT_WORKER_INTERVAL`, `CHECKPOINT_RCLONE_TIMEOUT` | Worker poll interval (default 30 s) and rclone timeout (default 21600 s). |
| `CLUSTER_HOME` | Offload only: if set, rclone reads `${CLUSTER_HOME}/.config/rclone/rclone.conf`. |
| `MUJOCO_BUILD_PARALLELISM`, `MUJOCO_BUILD_TIMEOUT`, `MUJOCO_BUILD_VERBOSE=1` | `wrappers/vec_env.py`, batched MuJoCo model warm-up: workers built at once (default 4), per-worker timeout (default 60 s), and a progress line per worker. |

## AMP humanoid (`experiments/amp_humanoid`)

All optional. The AMP commands do not log to W&B (training uses `--logger txt`).

| Variable | Effect |
|---|---|
| `PYTHON` | Interpreter used by `setup.sh`, `scripts/run_arm.sh` and `scripts/time_arm.sh` (default `python`). |
| `MIMICKIT_URL` | `setup.sh`: where to clone MimicKit from (default the GitHub repository); commit `2ed1e6c` is checked out either way, and every file is checked against `MANIFEST.sha256`. |
| `CUDA_VISIBLE_DEVICES` | Selects the GPU; the commands run one process on `cuda:0`, the first visible GPU. |
| `MIN_GPU_MIB` | `scripts/run_arm.sh`: refuse a GPU with less memory than this (default 20000 MiB). |
| `SKIP_TRAIN=1` | `scripts/run_arm.sh`: evaluate an existing checkpoint without training. |

## Thread caps

Only the ARS launchers and `experiments/sac_baselines_ars/scripts/cluster_env.sh` (the setup for
running an ARS seed by hand) set `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` (50
lockstep env workers per job). No PPO launcher sets a thread cap.

## A note on the SIGFPE

An earlier diagnosis blamed OpenBLAS thread-pool sizing for exit 136. That attribution was wrong.
Faulthandler traces put the crash in the CUDA `torch.dot` inside the PCGrad projection; it
reproduces identically at 8 and 32 threads, and the FairGrad solve is unaffected. A thread cap
does not fix it; `MTRL_PCGRAD_CPU=1` does.
