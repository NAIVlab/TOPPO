# Evaluation protocol

The SAC-family numbers in the paper come from **this code, under our protocol** — not from
running upstream's code and reproducing upstream's published table. This document states exactly
what that protocol is and where it differs from upstream's, so the numbers can be read correctly.

## Why a re-run was necessary

Upstream evaluates with a stochastic policy on 10 goals per task over 8 seeds, on a vendored
mujoco_py-era Meta-World. The PPO side of this paper evaluates with a deterministic policy on all
50 benchmark goals per task over 10 seeds, on Meta-World 3.0.0. Comparing those two directly
would not be a comparison of methods — it would be a comparison of protocols. So every SAC-family
arm was re-run under the PPO side's protocol.

## Protocol parity

Every dimension below is matched between the SAC arms and the PPO method:

| Dimension | Setting |
|---|---|
| Environment | Meta-World 3.0.0, V2 reward, gymnasium |
| Observation | state + one-hot(K), sorted task order |
| Horizon | H = 500, truncation-only, bootstrap on truncate |
| Training goals | the 50 benchmark train goals per task, resampled per reset |
| Budget | exactly 100M env steps (2M iterations x 50 envs); evaluation excluded |
| Evaluation policy | deterministic (policy mean, temperature 0) |
| Evaluation episodes | one per benchmark goal: 50 goals x 50 tasks |
| Success rule | success-once: `int(sum(success) > 0)` |
| Reporting | 10 seeds, final checkpoint, mean +/- sample std; per-task rates -> per-seed worst-k -> mean +/- std |
| Baseline hyperparameters | upstream's published MT50 recipes verbatim, no re-tuning |

SAC's 25k-iteration random warmup is counted **inside** the 100M budget, so budget parity is
strict rather than nominal.

## Deviations from upstream's published setup

These are the reasons our figures differ from upstream's published ones. Every deviation is
either direction-neutral or favors the baselines — none favors our method.

**D1 — Environment binding.** Meta-World 3.0.0 / gymnasium instead of upstream's vendored
mujoco_py stack. Direction unsigned, and it affects the baselines and our method identically.
This makes the re-run a *same-environment comparison*, not a reproduction of the published table.

**D2 — Training goal distribution.** We train on the 50 fixed benchmark goals, matching the
benchmark's official protocol and our method's training distribution. Upstream trains on
unlimited random goals. Because training and evaluation goal distributions now coincide, this
**favors the baselines**.

**D3 — Evaluation protocol.** Deterministic policy, 50 goals, 10 seeds, versus upstream's
stochastic policy, 10 goals, 8 seeds. Deterministic evaluation is >= stochastic for a converged
SAC agent and 50 goals reduces variance, so this also **favors the baselines**.

**D4 — LoRA attach decision.** The ARS-LoRA arm makes a single attach-vs-reset decision late in
training by reading its own evaluation success rate against a threshold. Under our protocol that
signal is the deterministic 50-goal evaluation rather than upstream's stochastic 10-goal one,
which biases the decision toward attaching — again **in the baseline's favor**. Per-seed decision
margins should be reported alongside the arm.

**D5 — Arm labeling.** Two arms need precise names. `mtsac` is SAC-MT under upstream's published
backbone hyperparameters. `mtsac_cb` is a **critic-only** port of our stack — critic LayerNorm,
per-task PopArt, FairGrad alpha=1 with the same solver — using EMA PopArt for float32 robustness
and deliberately **omitting** the actor-side PCGrad. It isolates the critic-balancing
contribution; it is not a SAC transplant of the full method.

**D6 — Checkpoint density.** Our evaluation grid is denser than the PPO side's. This is irrelevant
to a final-checkpoint comparison, but a denser grid inflates any "best over training" aggregator,
so subsample to a common grid before reporting best/peak.

## How to read the numbers

Report three columns, and never collapse them: **upstream published** (their seeds, their
protocol, their stack) | **our re-run** (10 seeds, deterministic 50-goal, Meta-World 3.0.0) |
**our method**. The first two are not interchangeable, and the gap between them is protocol, not
reproduction failure.

## Where the protocol lives in the code

`metaworld3_ars.py::build_eval_rounds` partitions the 50 benchmark goals per task into 5 rounds of
10 parallel copies, giving exactly 50 deterministic episodes per task per evaluation. The
success-once rule is in `evaluation.py`.

The protocol is selected by these `train_ars` / `train_ars_lora` config keys. All of them are
`Config` defaults, and the shared `k=v` line of the two sbatch templates in `slurm/` also passes
every one of them explicitly except `eval_all_goals` and `eval_copies_per_task`, which come from
the defaults alone:

| Key | Value | Upstream default |
|---|---|---|
| `domain` | `metaworld3` (the only one accepted) | `metaworld` (vendored, not shipped) |
| `env_name` | `MT50` | `MT10` |
| `rf_version` | `v2` | — |
| `max_steps` | `2000000` iterations x 50 envs = 100M env steps | same |
| `eval_distribution` | `det` | `stc` |
| `eval_all_goals` | `True` (one episode per benchmark goal) | — (first 10 goals) |
| `eval_copies_per_task` | `10` parallel copies per task, so 5 rounds (`train_ars`; fixed at 10 in `train_ars_lora`) | — |
| `eval_interval` | `20000` iterations (every 1M env steps) | `10000` |
| `activation`, `n_layer`, `batch_size` | `tanh`, `4`, `100` | same (`train_ars_lora`: `relu` activation) |
| `replay_buffer_size`, `n_reset` | `500000`, `6` (upstream's MT50 recipe) | `1000000`, `4` |

The MT10 reruns (README, section MT10) keep this protocol but use upstream's MT10 recipe:
`env_name=MT10`, `replay_buffer_size=1000000`, `n_reset=4`, `eval_interval=10000` (every 100k env
steps), `max_steps=2000000` × 10 envs = 20M env steps, and `rank=8 threshold=0.8` for `ars_lora`.
These are not the `Config` defaults; the launchers pass them when `BENCHMARK=MT10`.

The per-arm keys are in the arm table of `README.md`. `train_moore.py` and
`train_softmodular.py` ship unmodified from upstream, do not implement this protocol, and are not
supported; no number in the paper comes from them.
