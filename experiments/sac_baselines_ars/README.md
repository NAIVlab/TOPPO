# SAC-family baselines (ARS, MTSAC and variants)

**These are our own re-implementations/ports, run under our evaluation protocol — not upstream's
code producing upstream's published numbers.** Read that sentence before reading any number here.

## What this actually is

A substantially modified derivative of

> **ARS: Adaptive Reward Scaling for Multi-Task Reinforcement Learning**, Cho et al., ICML 2025
> <https://github.com/ms-cho/Adaptive-Reward-Scaling-ARS-for-MTRL>

pinned at upstream `67f7df3` (which itself derives from
[JAXRL](https://github.com/ikostrikov/jaxrl), MIT © 2021 Ilya Kostrikov).

By line count roughly a third of the shipped code is ours (`.py` / `.sh` / `.sbatch`,
excluding documentation and the patch file):

| | lines |
|---|---:|
| files we wrote from scratch (11) | **1,449** |
| lines we added to upstream files (11 files, 3,037 → 4,215: +1,383 / −205) | **net +1,178** |
| upstream lines still present (22 files: 11 modified, 11 unmodified) | 5,003 |
| upstream vendored code we removed | **251,392** (1,006 files) |

The 11 unmodified upstream files (`policy.py`, `actor.py`, `learner_moore.py`,
`learner_sac_mt_lora.py`, `learner_softmodular.py`, `train_moore.py`,
`train_softmodular.py`, `run_experiments.py`, and three wrappers) ship byte-identical
to `67f7df3`, as does `LICENSE`. Upstream's `README.md` and `requirements.txt` are replaced
outright by ours (this file, and dependency pins for the Meta-World 3.0.0 / JAX 0.4.24 stack).

Both training entry points more than doubled: `train_ars.py` 232 → 532,
`train_ars_lora.py` 239 → 678.

## Why the numbers here are not "reproductions"

Two things changed that make our results a **matched-protocol re-run**, not a reproduction of the
published results:

1. **The environment stack was replaced.** Upstream vendors the mujoco_py-era Meta-World (1,006
   files, ~251k lines, shipped with no license or notice). We deleted it and wrote
   `metaworld3_ars.py` against Meta-World 3.0.0 — the same benchmark version the PPO side uses.
2. **The evaluation protocol is ours.** `build_eval_rounds` runs one episode per benchmark goal,
   50 deterministic episodes per task, partitioned into 5 rounds of 10 parallel copies. This is
   what makes the SAC-side and PPO-side numbers comparable at all.

So our figures should be compared against the published ones only with that caveat attached. The
audit behind the protocol, and the observed gaps against published values, are in
`EVALUATION_PROTOCOL.md`.

## What we added

| File | Lines | Purpose |
|---|---:|---|
| `metaworld3_ars.py` | 316 | Meta-World 3.0.0 backend, V1/V2 reward switch, and the 50-goal evaluation protocol |
| `checkpoint_utils.py` | 350 | preemption-safe checkpointing |
| `fairgrad_host.py` | 132 | host-side α-fairness solver (numpy port of the PPO stack's y-space Newton solver) |
| `popart_jax.py` | 104 | PopArt target normalization for the JAX critic |
| `slurm/`, `scripts/` | 547 | launchers and environment setup |

## Attribution is unaffected by any of this

Having rewritten this much does **not** reduce the obligation: this is a derivative work of
MIT-licensed software. Upstream's `LICENSE` is preserved byte-identical, `NOTICE` supplies the
copyright holder upstream left blank, `PROVENANCE.md` pins the base commit, and `CHANGES.md` plus
`patches/` let anyone mechanically separate theirs from ours. If you use this code, please also
cite the ARS paper (BibTeX from upstream's README; we added the comma missing after `booktitle`):

```bibtex
@inproceedings{cho2025ars,
  title     = {ARS: Adaptive Reward Scaling for Multi-Task Reinforcement Learning},
  author    = {Cho, Myungsik and Park, Jongeui and Kim, Jeonghye and Sung, Youngchul},
  booktitle = {Forty-second International Conference on Machine Learning},
  year      = {2025}
}
```

## Arms

No YAML configs. Each arm is a `case` block of CLI `k=v` pairs, the same in
`slurm/site_a100/mt50_run.sbatch` and `slurm/site_3090_mt50_run.sbatch`, on top of one shared
protocol line. Every arm runs `train_ars` except `ars_lora`, which runs `train_ars_lora`.

| arm | paper label | arm `k=v` | public W&B group (`docs/RUNS.md`) |
|---|---|---|---|
| `ars` | ARS | `use_ars=True hidden_dim=400` | `mt50v2_sac_ad_cd_nars_ln0_vcx_400x4` |
| `ars_ln` | ARS-LN-400 | `use_ars=True hidden_dim=400 critic_layernorm=True critic_init_layernorm=True` | `mt50v2_sac_ad_cd_nars_lnci_vcx_400x4` |
| `ars_ln1024` | ARS-LN-1024 | as `ars_ln`, with `hidden_dim=1024` | `mt50v2_sac_ad_cd_nars_lnci_vcx_1024x4` |
| `ars_lora` | ARS-LoRA | `hidden_dim=400 critic_layernorm=True critic_init_layernorm=True rank=16 threshold=0.65` | `mt50v2_sac_ad_cd_nars_lnci_vcx_400x4_lora16` |
| `mtsac` | SAC-MT (MTSAC) | `use_ars=False hidden_dim=400` | `mt50v2_sac_ad_cd_nnone_ln0_vcx_400x4` |
| `mtsac_cb` | MTSAC + critic balancing | `use_ars=False hidden_dim=400 critic_layernorm=True critic_method=fairgrad use_popart=True popart_method=ema` | `mt50v2_sac_ad_cfg1_npae_lnc_vcx_400x4` |
| `mtsac_fullstack` | MTSAC + full stack | as `mtsac_cb`, plus `actor_method=pcgrad` | `mt50v2_sac_apc_cfg1_npae_lnc_vcx_400x4` |

The shared protocol line is

```
domain=metaworld3 env_name=MT50 activation=tanh n_layer=4 batch_size=100
replay_buffer_size=500000 n_reset=6 eval_interval=20000 eval_distribution=det
rf_version=v2 max_steps=2000000
```

These values are also the `Config` defaults of both trainers, so leaving one out changes
nothing. `max_steps` counts loop iterations, and each iteration steps all 50 task
environments once, so 2M iterations are the paper's 100M env steps (`toppo/env_steps` on W&B).
The trainers reject any `domain` other than `metaworld3`.

The launchers log run `sac_mt50_<arm>_s<seed>` in group `sac_mt50_<arm>`, project
`${WANDB_PROJECT:-toppo-repro}` under your default W&B entity (or `$WANDB_ENTITY`); the trainers'
own default project is also `toppo-repro`, so a re-run never joins the published runs in
liyp1220-uci/TOPPO by default. The public W&B project holds the same runs under the group names
above and keeps the launcher's name in each run's config (`wandb_name`). To log a re-run under the
public group name, append `wandb_group=<group>` (and `project=<project>`) as extra `k=v` pairs;
they come after the launcher's own and take precedence. `../metaworld_ppo/analysis/metrics.py`
reads such runs with `--entity <you> --project toppo-repro` (see
[`../../docs/env_vars.md`](../../docs/env_vars.md)).

To run the 10-seed grid on Slurm, use `slurm/site_a100/launch_mt50_grid.sh` (all 7 arms by
default, dry-run unless `-y`) or `scripts/launch_mt50.sh <arm> <seed>...`, which submits
`slurm/site_3090_mt50_run.sbatch`. Both sbatch files are site templates; their headers list the
environment variables they read.

### MT10

The camera-ready also reports five of these arms on MT10 (Table 2; the four ARS arms also in the
appendix learning-curve figure). Set
`BENCHMARK=MT10` for any of the launchers above, e.g. `BENCHMARK=MT10 scripts/launch_mt50.sh ars 0 1`
or `BENCHMARK=MT10 bash slurm/site_a100/launch_mt50_grid.sh` (its MT10 default is the five arms).
The protocol is unchanged (Meta-World 3.0.0, V2 rewards, H=500, deterministic evaluation on all
50 goals per task); the hyperparameters follow upstream's MT10 recipe instead of its MT50 one, so
the shared protocol line becomes

```
domain=metaworld3 env_name=MT10 activation=tanh n_layer=4 batch_size=100
replay_buffer_size=1000000 n_reset=4 eval_interval=10000 eval_distribution=det
rf_version=v2 max_steps=2000000
```

and `ars_lora` uses `rank=8 threshold=0.8`. 2M iterations × 10 task environments are the paper's
20M env steps; evaluation runs every 100k env steps. Runs log as `sac_mt10_<arm>_s<seed>`.

| arm | paper label | public W&B group (`docs/RUNS.md`) |
|---|---|---|
| `ars` | ARS | `mt10v2_sac_ad_cd_nars_ln0_vcx_400x4` |
| `ars_ln` | ARS-LN-400 | `mt10v2_sac_ad_cd_nars_lnci_vcx_400x4` |
| `ars_ln1024` | ARS-LN-1024 | `mt10v2_sac_ad_cd_nars_lnci_vcx_1024x4` |
| `ars_lora` | ARS-LoRA | `mt10v2_sac_ad_cd_nars_lnci_vcx_400x4_lora8` |
| `mtsac` | SAC-MT (MTSAC) | `mt10v2_sac_ad_cd_nnone_ln0_vcx_400x4` |

`mtsac_cb` and `mtsac_fullstack` were run on MT50 only; the launchers and both sbatch templates
refuse them with `BENCHMARK=MT10`. Unlike the MT50 line, these MT10 values are not the trainers'
`Config` defaults, so a run outside the launchers must pass them explicitly. With `n_reset=4` the ARS reset interval (400k iterations) divides the 2M-iteration
budget; the trainers skip the reset or LoRA attach that would otherwise fall on the final
iteration, after the final evaluation, so the saved final checkpoint is the evaluated policy.

### One seed without Slurm

From this directory, in the Python 3.10 environment below, on a machine with a GPU visible to JAX:

```bash
source scripts/cluster_env.sh
python -u run_experiments.py --main_module_name train_ars --seeds 0 \
  --config_args domain=metaworld3 env_name=MT50 activation=tanh n_layer=4 \
    batch_size=100 replay_buffer_size=500000 n_reset=6 \
    eval_interval=20000 eval_distribution=det rf_version=v2 \
    max_steps=2000000 checkpoint_interval=100000 \
    use_ars=True hidden_dim=1024 critic_layernorm=True critic_init_layernorm=True \
    project="${WANDB_PROJECT:-toppo-repro}" wandb_name=sac_mt50_ars_ln1024_s0 \
    wandb_group=mt50v2_sac_ad_cd_nars_lnci_vcx_1024x4
```

This is the `ars_ln1024` arm, seed 0. For another arm, replace the arm `k=v` pairs (and use
`--main_module_name train_ars_lora` for `ars_lora`). Checkpoints go to
`checkpoints/<group>/<name>/`, and re-running the same command resumes from them.

### Not supported

`train_moore.py` and `train_softmodular.py` (with `learner_moore.py` and
`learner_softmodular.py`) ship byte-identical to upstream. They still default to upstream's
vendored `domain="metaworld"`, which this release does not ship, and they do not implement our
evaluation protocol. No launcher calls them, and no number in the paper comes from them.

## Environment

Requires **exactly Python 3.10** — 3.11+ rejects upstream's mutable dataclass defaults in
`value_net.py`. Install as described in the header of `requirements.txt`, or see
`slurm/site_a100/setup_env_mw3.sh` and the root `docs/environments.md`.

## Credentials

None ship here. Export `WANDB_API_KEY` (or run `wandb login`). When `WANDB_API_KEY` is unset,
the sbatch templates read `~/.wandb_api_key` instead.
