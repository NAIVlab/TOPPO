# Meta-World multi-task PPO

The TOPPO method and every Meta-World experiment in the paper. Multi-task PPO on
[Meta-World](https://meta-world.github.io/) MT10 / MT50, built on Tianshou and PyTorch, with
critic-side interventions for task imbalance.

## The method

Per-task value gradients differ in scale far more than advantage-normalized policy gradients
(about 497× vs 4.1× across tasks in the paper's MT10 diagnostics). The paper hypothesizes that
this critic imbalance contributes to tail-task failure, a hypothesis its correlational diagnostics
support but do not establish as a causal chain. Actor-only interventions leave the critic's
immediate update geometry unchanged, so TOPPO conditions the critic:

| Module | What it does |
|---|---|
| **LN-c** | pre-activation LayerNorm in the critic only (not the actor) |
| **PopArt** | per-task value-target normalization |
| **FairGrad (alpha=1)** | alpha-fairness aggregation of per-task critic gradients, so every task contributes equally |
| **PCGrad** | applied to the *actor*, whose advantage-normalized gradients differ far less in scale; it projects away conflicting directions |

FairGrad's alpha-fairness KKT system `G x = x^(-1/alpha)` is solved by a y-space projected Newton
method (`mtrl/gradients/fairgrad.py`), which is what makes it affordable at K=50 tasks.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.12
pip install -r requirements.txt
export MUJOCO_GL=egl && unset DISPLAY               # headless
```

The header of `requirements.txt` lists the package versions behind the paper runs (the headline
groups used torch 2.10.0+cu128, mujoco 3.7.0, numpy 2.4.4, gymnasium 1.2.3); pin those to match
the headline environment exactly.

Tests (CPU only; no GPU or W&B needed):

```bash
pip install -r requirements-dev.txt && python -m pytest tests
```

The PopArt smoke test trains one tiny epoch (about 2 min on CPU) in a temporary directory;
deselect it with `-k "not smoke"`.

## Run

```bash
# smoke test: two tiny epochs, no evaluation, no W&B
# (delete logs/smoke/ before re-running it: training auto-resumes from that directory)
python train.py --config configs/ppo_mt10_default.yaml --seed 0 --device cuda \
    --exp-name smoke --training.epoch 2 --training.step-per-epoch 10000 \
    --training.step-per-collect 5000 --training.batch-size 1000 \
    --training.repeat-per-collect 1 --logging.test-interval 0 --logging.no-use-wandb

# one seed of a published run group (logs to W&B project toppo-repro, see Analysis)
python train.py --config configs/ppo_mt50_toppo_lnc_rfv2.yaml --seed 0 --device cuda \
    --exp-name mt50v2_ppo_apc_cfg1_npaw_lnc_vc1_400x3_s0 \
    --logging.wandb-group mt50v2_ppo_apc_cfg1_npaw_lnc_vc1_400x3
```

> **Always pass `--exp-name`.** Each config ships a fixed `exp_name` that does not depend on the
> seed, and training auto-resumes from any checkpoint under `logs/<exp_name>/` without checking
> the seed — so without it, seed 1 silently continues seed 0's run. `../../docs/RUNS.md` gives the
> canonical group name for every published row.

Any config field can be overridden on the command line with dotted flags
(`--ppo.vf-coef 0.001`, `--training.epoch 100`, `--logging.wandb-group my_run`).

> **If a PCGrad-actor run dies with exit 136**, set `MTRL_PCGRAD_CPU=1`. On some GPUs the CUDA
> `torch.dot` inside the PCGrad projection raises SIGFPE; the env var runs identical arithmetic on
> CPU. No shipped launcher sets it for you. See `../../docs/env_vars.md`.

Runs are reproducible statistically (as seed-aggregate statistics over 10 seeds: mean ± std, or
IQM ± std on V1, as the paper reports them), not bit for bit: `--seed` seeds
Python/NumPy/PyTorch and `metaworld.MT*(seed=)` fixes each task's goal set, but the order in which
training episodes sample goals comes from Meta-World's unseeded env RNG (as in the published runs);
evaluation goals are iterated deterministically.

To render MP4 rollouts from a checkpoint you trained:

```bash
python render.py --checkpoint logs/<exp_name>/checkpoints/latest.pth \
    [--cameras corner topview] [--num-episodes 1] [--seed 0]
```

It reads the config from `logs/<exp_name>/config.yaml`, writes
`videos/<exp_name>/<checkpoint-stem>/<task>_<camera>_ep<i>.mp4`, and needs `MUJOCO_GL=egl` when
headless. No trained checkpoints are released.

## Config naming

`ppo_<benchmark>_<variant>[_lnc][_rfv{1,2}].yaml`

| Token | Meaning |
|---|---|
| `mt10` / `mt50` | benchmark |
| `default` / `pcgrad` / `cagrad` / `fairgrad` | gradient combiner |
| `toppo` | the full stack: LN-c + PopArt (Welford) + FairGrad(α=1) critic + PCGrad actor |
| `vanilla` | rung 0 of the ladder: no LayerNorm, no normalizer |
| `popart` | PopArt value-target normalization |
| `lnc` | LayerNorm on the critic |
| `rfv1` / `rfv2` | Meta-World reward version (V1 or V2) -- these behave very differently, see below |
| `rewnorm` / `retscale` / `hlgauss` | alternative critic normalizers compared against PopArt |
| `novclip` | PPO value clipping disabled (required by `hlgauss`, which has no cross-entropy analog) |
| `fullstack` | PCGrad-actor + FairGrad-critic + LN-c together |

**The token list is not exhaustive, and a missing token does not mean a disabled
feature.** `ppo_mt{10,50}_{default,pcgrad,cagrad,fairgrad}.yaml` carry no `_lnc`,
`popart` or `rfv` token, yet all eight enable LN-c and PopArt and run V2 rewards --
they are the `+LN-c +PopArt` rung with the combiner varied, not a bare baseline.
Read the YAML, or look the arm up in [`../../docs/RUNS.md`](../../docs/RUNS.md),
which lists the effective settings for every run group in the paper.

`ppo_mt50_ad_cfg_popart_lnc_rfv{1,2}.yaml` are the Welford "LN-c + PopArt + FG-c" configs (TOPPO
without the PCGrad actor); the paper's no-normalizer rows of that arm use them with
`--ppo.no-use-popart`. `configs/post_submission/` holds only EMA-PopArt arms explored after the
paper was submitted, including the actor-aggregation sweep that the camera-ready reports as a
separate EMA cohort (Figure 5, Appendix E.6). They are kept separate deliberately: the paper
headline uses the **Welford** (non-EMA) PopArt, and mixing the two families has caused a real
reporting error before. Every paper config evaluates every 2 epochs (`test_interval: 2`), so the
Slurm templates' `--logging.test-interval 2` is redundant but harmless.

## V1 vs V2 rewards

Meta-World's two reward versions are not interchangeable and the module ranking changes between
them. On V1 the per-task value-target scales span ~5 orders of magnitude, so normalization is
load-bearing; on V2 they span ~500x, and most interventions become statistically
indistinguishable. Always state which regime a number came from.

## Layout

```
train.py                 entry point
render.py                render MP4 rollouts from a checkpoint you trained
mtrl/
  algos/multitask_ppo.py multi-task PPO: per-side combiners, PopArt, HL-Gauss, return scaling
  gradients/             fairgrad (KKT + y-space Newton solver), pcgrad, cagrad, per-side dispatch
  models/                actor/critic networks, PopArt module, HL-Gauss categorical head
  envs/                  Meta-World wrappers, reward-version switch, reward normalization
  utils/                 config schema, logging, checkpointing
configs/                 experiment configs (see naming above)
analysis/                aggregation: worst-k, IQM, paired tests, W&B export, RUNS.md checker
benchmarks/solver/       FairGrad solver benchmarks + raw output (real-Gram CSV, appendix
                         solver table, scale-invariance figure data)
tests/                   unit tests for the combiners, PopArt, sampling, and the normalizers
requirements-dev.txt     pytest, for tests/
slurm/                   Slurm templates for multi-seed runs (site-specific; edit before use)
```

## Analysis

`analysis/` computes the paper's final-checkpoint statistics (mean, IQM, worst-k, paired tests)
from local run directories or from W&B, plus matched-progress scores across seeds
(`matched_step.py`), per-task normalizer scales from checkpoints (`mechanism.py`,
`mechanism_agg.py`) and the `docs/RUNS.md` checker. It does not produce the paper's figures, the
best/peak/late-checkpoint statistics, the best-checkpoint steps, the crossing steps or the
parameter counts. Those were built outside this release; the figures and checkpoint-history
statistics come from the per-task evaluation histories, which are public on W&B.

**From local runs.** `collect_csv.py` writes one JSON record per run to **stdout**, so redirect
it. It infers the task count from the CSV header, so the same command works on MT10 and MT50.
`aggregate.py` recovers `(group, seed)` from the run directory name, which is why runs must be
launched with `--exp-name <group>_s<seed>` (see `../../docs/RUNS.md`). Its `--require-step`
defaults to the full budget of each record's suite (100M for 50 tasks, 20M for 10):

```bash
# MT50
python analysis/collect_csv.py --logdir logs/ > runs.jsonl
python analysis/aggregate.py --jsonl runs.jsonl --groups <group> ... --worst-k 10 20

# MT10 — 10 tasks, 20M budget, and worst-10 equals the mean so it is not a tail metric
python analysis/collect_csv.py --logdir logs/ > runs.jsonl
python analysis/aggregate.py --jsonl runs.jsonl --groups <group> ... --worst-k 3 5
```

`matched_step.py` and `mechanism.py` / `mechanism_agg.py` (per-task normalizer scales from
checkpoints) work on MT10 and MT50 and with the canonical `--exp-name`.

**From W&B.** By default `metrics.py` and `export_wandb.py` read the published runs at
liyp1220-uci/TOPPO, so the paper's numbers can be recomputed with no setup. Training logs to
project `toppo-repro` instead (see below), so to read your own runs pass
`--entity <you> --project toppo-repro`. In general, a project named with `--project` or
`$WANDB_PROJECT` is read under `--entity`, else `$WANDB_ENTITY`, else your W&B default entity;
`--entity` alone reads `<entity>/TOPPO`, and `$WANDB_ENTITY` alone does not move the source.
`check_runs_index.py --wandb` follows the same rule, and each script prints the W&B source it
reads. The task count is inferred per group/run (50 on MT50, 10 on MT10), and the step check
defaults to that suite's full budget (100M / 20M), because a job that stopped early still reports
`state == "finished"`. Pass `--min-step 0` to disable the check entirely, so that runs without a
logged `_step` are kept too (`export_wandb.py` also accepts `--require-step`):

```bash
# MT50 headline and +LN-c +PopArt rung (TOPPO: 90.9 +/- 1.6, worst-10 56.5 +/- 7.8)
python analysis/metrics.py --groups mt50v2_ppo_apc_cfg1_npaw_lnc_vc1_400x3 \
    mt50v2_ppo_ad_cd_npaw_lnc_vc1_400x3

# MT10 headline
python analysis/metrics.py --groups mt10v2_ppo_apc_cfg1_npaw_lnc_vc1_400x3

# your own re-run of a group, trained with a shipped config (W&B project toppo-repro)
python analysis/metrics.py --entity <you> --project toppo-repro \
    --groups mt10v2_ppo_apc_cfg1_npaw_lnc_vc1_400x3

# per-task scores, then a paired comparison (the paper's +0.43 pp, p = 0.64)
python analysis/export_wandb.py --groups mt50v1_ppo_ad_cd_nrew_lnc_vc0_400x3 \
    mt50v1_ppo_ad_cd_npaw_lnc_vc0_400x3 > refs.jsonl
python analysis/aggregate.py --jsonl refs.jsonl \
    --compare mt50v1_ppo_ad_cd_nrew_lnc_vc0_400x3 mt50v1_ppo_ad_cd_npaw_lnc_vc0_400x3
```

Note that **training** writes to `config.project_name` and does not read `$WANDB_PROJECT`. Every
shipped config, and the code default, sets it to `toppo-repro`, a scratch project under your
default W&B entity (or `$WANDB_ENTITY`), so a reproduction never logs into the published project
by default; pass `--project-name` to `train.py` to change it. See
[`../../docs/env_vars.md`](../../docs/env_vars.md).

Group names are listed in [`../../docs/RUNS.md`](../../docs/RUNS.md).

Conventions: mean +/- **sample std** (never SEM), IQM +/- untrimmed std, paired t-tests over
shared seeds; `aggregate.py --compare` prints the paired mean difference with a t-based 95% CI,
plus the paired t and p. Only runs that reached the full step budget are counted.

### Per-task gradient dumps

The mechanism figures (per-task critic vs actor gradient norms, Gram heatmaps, offline FairGrad
replay) are computed from raw per-task gradient snapshots rather than from logged scalars. Enable
them with:

```bash
python train.py --config configs/ppo_mt10_default.yaml --seed 0 \
    --exp-name mt10_default_grads_s0 \
    --logging.dump-grads-dir auto --logging.dump-grads-every-collects 5
```

A distinct `--exp-name` keeps the dump run from resuming another run's checkpoint. `auto`
resolves to `<exp_dir>/grad_snapshots/`. One `.npz` is written per gated minibatch (first
minibatch of the first repeat, every Nth collect) alongside a single `manifest.json` that records
per-layer names, shapes, and flat offsets. Each snapshot stores the per-task actor-view and
critic-full-view gradients in fp32, the per-task PopArt `(sigma, mu, W, b)` when PopArt is on, the
per-side FairGrad weights and KKT residual when a solver ran, and the realized per-task minibatch
counts. Read them back with `mtrl.utils.grad_snapshots.open_snapshot_dir`, which also exposes Gram
matrices, per-layer slicing, the PopArt `sigma^-2` inversion, and offline FairGrad replay at an
arbitrary alpha through the production solver. The four MT10 cells behind the paper's mechanism
figures are listed in `../../docs/RUNS.md` (Not in the public W&B project).

Off by default: with `dump_grads_dir` unset the instrumentation performs no IO, no extra backward
passes, and draws no random numbers, so runs are unaffected. Budget roughly 27 MB per snapshot at
MT10 and 137 MB at MT50 in fp32 -- keep the cadence at 5 or more for long runs.

## Scale

MT50 is **CPU-bound**: 50 parallel MuJoCo workers, 16-24 cores, ~18 GB RAM, but little GPU memory
(the headline runs peaked at 1.65 GB on an RTX 3090 and 2.16 GB on an H200 NVL). One 100M-step
seed of the headline configuration took 23-24 h on an RTX 3090 (31-32 h for the two seeds that
were preempted and resumed) and about 14 h on an H200 NVL, including its 51 evaluations. With evaluation disabled, the paper's timing study (Appendix E.4,
one RTX 3090 with 16 cores) measured 8.7 h per 100M steps for vanilla PPO and 19.8 h for TOPPO.
MT10 at 20M steps is far cheaper. The `slurm/` templates run 10 seeds as array jobs; they encode a
specific cluster's partitions and paths, so treat them as examples rather than drop-in scripts.
