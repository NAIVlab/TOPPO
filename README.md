# TOPPO: Rethinking PPO for Multi-Task Reinforcement Learning with Critic Balancing

**Yuanpeng Li** (UC Irvine), **Gefei Lin** (George Washington University), **Annie Qu** (UC Santa
Barbara), **Rui Miao** (UT Dallas)

*NeurIPS 2026 (poster).* Paper: [arXiv:2605.11473](https://arxiv.org/abs/2605.11473) (v1 predates
the camera-ready; section, table and appendix numbers below follow the camera-ready) ·
Proceedings: *link to be added* · Training logs:
[wandb.ai/liyp1220-uci/TOPPO](https://wandb.ai/liyp1220-uci/TOPPO)

Code for the paper. TOPPO (Tail-Optimized PPO) revisits multi-task PPO through critic
conditioning. In our diagnostic MT10 runs, per-task critic gradient norms (in raw-loss units)
span about 497× across tasks, against 4.1× for the advantage-normalized actor gradients. We
hypothesize that this imbalance contributes to tail-task failure: tasks with little influence on
the shared critic get poorer value estimates, which distort their policy updates and the data they
collect next. The diagnostics are correlational; they support this hypothesis but do not establish
the full causal chain. Actor-only interventions leave the critic's immediate update geometry
unchanged, so TOPPO conditions the critic — critic LayerNorm (LN-c), per-task PopArt target
normalization, and FairGrad at α = 1 on the critic — and adds PCGrad on the actor.

| Meta-World V2, final checkpoint, 10 seeds (% success, mean ± std) | MT10 mean | MT50 mean | MT50 worst-10 |
|---|---|---|---|
| **TOPPO** (685K / 717K parameters) | 97.2 ± 3.1 | **90.9 ± 1.6** | **56.5 ± 7.8** |
| Vanilla MT-PPO, same backbone | 85.2 ± 6.5 | 78.5 ± 3.3 | 6.6 ± 9.1 |
| ARS-LN-1024, our rerun under the same protocol | 96.9 ± 2.6 | 85.7 ± 3.2 | 43.3 ± 9.8 |

Every number in this table can be recomputed from the public logs with `analysis/metrics.py`
(see [`experiments/metaworld_ppo/README.md`](experiments/metaworld_ppo/README.md#analysis)); the
group names are in [`docs/RUNS.md`](docs/RUNS.md).

## Contents

| Directory | What it is | Provenance |
|---|---|---|
| `experiments/metaworld_ppo/` | The TOPPO method and every Meta-World experiment: the MT10/MT50 ladder, the gradient-combiner ablations, the actor-aggregation sweep, and the critic-normalizer campaign (PopArt vs reward normalization vs return-based scaling vs HL-Gauss). | ours, MIT, except adapted portions listed in `NOTICE` |
| `experiments/sac_baselines_ars/` | SAC-family baselines (ARS, ARS-LN, ARS-LoRA, MTSAC and variants). Heavily modified derivative of Cho et al. 2025 — new Meta-World 3.0.0 backend and **our** evaluation protocol, so its numbers are matched-protocol re-runs, *not* reproductions of the published figures. Roughly a third of the code is ours by line count. | derivative of Cho et al. 2025, MIT. See its `README.md` / `PROVENANCE.md` / `CHANGES.md` |
| `patches/` (inside the ARS dir) | Our modifications as an applyable patch against upstream `67f7df3`. | — |
| `experiments/amp_humanoid/` | The AMP humanoid walk-steering stress test (Section 5.3 and Appendix E.5): a patch against MimicKit `2ed1e6c` plus our modules, the 25 campaign configs, the 250 per-seed evaluation records, and scripts that regenerate Table E.6, Figure E.4 and the 76% result without a GPU. | derivative of MimicKit (Peng, 2025), **Apache-2.0**. See its `README.md` / `PROVENANCE.md` / `CHANGES.md` / `NOTICE` |

The paper's AMP humanoid walk-steering stress test (Section 5.3 and Appendix E.5) is a second task
family, in [`experiments/amp_humanoid/`](experiments/amp_humanoid/). It is built on MimicKit and
the Newton simulator and ships as a patch against MimicKit `2ed1e6c` plus our own modules and
configs (`setup.sh` there assembles the tree). Its 250 runs (25 configurations × 10 seeds) are
public in the W&B project below (report 8), and its 250 per-seed evaluation records ship with the
code, so Table E.6, Figure E.4 and the 76% result can be regenerated without a GPU. Its short H200
timing measurements are not on W&B; they ship as
`experiments/amp_humanoid/results/timing/timing.tsv`. That directory is distributed under
MimicKit's Apache License 2.0, not under the MIT `LICENSE` (see [License](#license)).

[`docs/RUNS.md`](docs/RUNS.md) indexes every W&B run group in the paper: for each Meta-World group
it gives the config and exact command-line overrides that reproduce it, and for each AMP group its
config and the training and evaluation commands.

## Experiment logs

Apart from the four exceptions below, the training runs behind the paper's tables and figures are
public, together with the additional runs indexed in [`docs/RUNS.md`](docs/RUNS.md):
**[https://wandb.ai/liyp1220-uci/TOPPO](https://wandb.ai/liyp1220-uci/TOPPO)** — 1,193 runs in
120 groups (95 Meta-World, 25 AMP humanoid), 10 seeds each except one 3-seed control. Every run
carries the canonical `toppo/*` parameter block. The Meta-World runs also carry per-task metrics
and their training histories, though some histories have gaps or stop early, and 25 of the AMP
runs carry final evaluation summaries only; `docs/RUNS.md` lists these exceptions. Four
measurements are not in the public project:

- the single-shot compute timings (Appendix E.4);
- the four MT10 gradient-dump runs behind the mechanism figures (Appendix D.2);
- the two-epoch real-Gram solver benchmark (Appendix C.5), whose raw output ships as
  `experiments/metaworld_ppo/benchmarks/solver/solverbench_rfv1.csv`;
- the short H200 timing measurements of the AMP stress test (Appendix E.5), which ship with their
  launcher in `experiments/amp_humanoid/` (`results/timing/timing.tsv`, `scripts/time_arm.sh`).

[`docs/RUNS.md`](docs/RUNS.md#not-in-the-public-wb-project) describes all four and gives a
recipe for each. The reports below are written to be read in order; report 0 explains
the naming, the tags and the statistical conventions.

No trained checkpoints or model weights are released. The paper's success rates come from the
evaluation logs in the W&B project above, and checkpoints can be regenerated with the commands in
[`docs/RUNS.md`](docs/RUNS.md). The AMP numbers come from the evaluation records in
`experiments/amp_humanoid/results/eval/`; each names the sha256 of the checkpoint it scored.

| # | Report | What it answers |
|---|---|---|
| 0 | [Start here](https://wandb.ai/liyp1220-uci/TOPPO/reports/--VmlldzoxNzg5MTc5MA==) | Naming, tags, conventions, and a map of the other eight. **Start here.** |
| 1 | [Main results on Meta-World (V2 rewards)](https://wandb.ai/liyp1220-uci/TOPPO/reports/--VmlldzoxNzg5MTc1Mg==) | MT50 and MT10 headline, the four-module ladder, SAC/ARS under one protocol. |
| 2 | [Why the critic](https://wandb.ai/liyp1220-uci/TOPPO/reports/--VmlldzoxNzg5MTc2Ng==) | The diagnostics behind the critic-side hypothesis: per-task gradient spread, solver telemetry, cost. |
| 3 | [Critic-side ablations](https://wandb.ai/liyp1220-uci/TOPPO/reports/--VmlldzoxNzg5MTc2OA==) | Which module does what: the 8-cell cube, per-side combiners, aggregator scope, LN placement, capacity. |
| 4 | [What changes when the reward scales do](https://wandb.ai/liyp1220-uci/TOPPO/reports/--VmlldzoxNzg5MTc3MQ==) | The V1/V2 regime change and the published-baseline head-to-head. |
| 5 | [Is PopArt the right critic normalizer?](https://wandb.ai/liyp1220-uci/TOPPO/reports/--VmlldzoxNzg5MTc4MA==) | PopArt vs reward normalization vs return scaling vs HL-Gauss, and the value-clip confound. |
| 6 | [The SAC side: baselines under one protocol, and a transplant that failed](https://wandb.ai/liyp1220-uci/TOPPO/reports/--VmlldzoxNzg5MTc4MQ==) | Matched-protocol re-runs, and the critic stack transplanted onto MTSAC (it did not help). |
| 7 | [The actor-side sweep that did not change anything](https://wandb.ai/liyp1220-uci/TOPPO/reports/--VmlldzoxNzg5MTc4Mg==) | The EMA-PopArt actor-aggregation sweep of Figure 5 and Appendix E.6 (Table E.7). |
| 8 | [The AMP humanoid family](https://wandb.ai/liyp1220-uci/TOPPO/reports/--VmlldzoxNzg5MjgxOQ==) | The AMP stress test of Section 5.3 and Appendix E.5; its code and per-seed evaluation records are in `experiments/amp_humanoid/`. A second task family: one task, two conflicting rewards, one critic. |

On W&B, `status-paper` marks the groups the camera-ready reports, `status-design-record` the
explorations behind a design choice, and `status-followup` the remaining follow-up runs; report 0
explains every tag. [`docs/RUNS.md`](docs/RUNS.md) is the authoritative index of the runs the paper
uses.

Reports 5, 6 and 7 are the ones least flattering to the paper's own choices; they are included
for the same reason as everything else.

## Quickstart (Meta-World PPO)

```bash
cd experiments/metaworld_ppo
pip install -r requirements.txt          # Python 3.12; Meta-World 3.0.0
export MUJOCO_GL=egl                     # headless rendering
export WANDB_API_KEY=...                 # read from the environment by train.py;
                                         # ~/.wandb_api_key is a fallback in the
                                         # Slurm scripts only, not here

# smoke test: two tiny epochs, no evaluation, no W&B, just to check the stack end to end
# (delete logs/smoke/ before re-running it: training auto-resumes from that directory)
python train.py --config configs/ppo_mt10_default.yaml --seed 0 --device cuda \
    --exp-name smoke --training.epoch 2 --training.step-per-epoch 10000 \
    --training.step-per-collect 5000 --training.batch-size 1000 \
    --training.repeat-per-collect 1 --logging.test-interval 0 --logging.no-use-wandb

# one real MT10 seed (the full 20M-step budget, several hours); it logs to the W&B
# project toppo-repro under your default entity, not to the published project
python train.py --config configs/ppo_mt10_default.yaml --seed 0 --device cuda \
    --exp-name mt10v2_ppo_ad_cd_npaw_lnc_vc1_400x3_s0 \
    --logging.wandb-group mt10v2_ppo_ad_cd_npaw_lnc_vc1_400x3
```

> `--exp-name` is what separates one run from another: each config carries a fixed
> `exp_name` that does not depend on the seed, and training auto-resumes from any
> checkpoint it finds under `logs/<exp_name>/`. [`docs/RUNS.md`](docs/RUNS.md)
> explains this and gives the exact command for every published run group.

MT50 is a **CPU-bound** benchmark: 50 parallel MuJoCo workers, ~16–24 CPU cores and 16–18 GB RAM,
but little GPU memory (the headline runs peaked at 1.65 GB on an RTX 3090 and 2.16 GB on an H200
NVL). Wall-clock time depends on the configuration and the host. One 100M-step seed of the
headline configuration took 23–24 h on an RTX 3090 (31–32 h for the two seeds that were preempted
and resumed) and about 14 h on an H200 NVL, including its 51 evaluations. With evaluation disabled, the paper's timing study (Appendix E.4: one RTX 3090,
16 CPU cores) measured 8.7 h per 100M steps for vanilla PPO and 19.8 h for TOPPO.

## Credentials

**No credentials ship in this repository, and nothing here reads one from inside the tree.**
Supply your own via `WANDB_API_KEY` or `~/.wandb_api_key` (mode 600).

## Configuration layout

`experiments/metaworld_ppo/configs/` holds the paper's Welford-PopArt configs and the normalizer
campaign, including `ppo_mt50_ad_cfg_popart_lnc_rfv{1,2}.yaml` for the Welford and no-normalizer
rows of the actor=default / critic=FairGrad arm. `configs/post_submission/` holds the EMA-PopArt
arms explored after submission; the camera-ready reports their actor-aggregation sweep as a
separate EMA cohort (Figure 5, Appendix E.6). No Welford or no-normalizer paper row routes through
`configs/post_submission/`. The two families are kept apart on purpose: the paper headline is the
Welford (non-EMA) PopArt configuration, and mixing the two has caused a real reporting error
before.

## Portability

Launchers were written for specific Slurm clusters. Institutional paths and account names have
been replaced by environment variables — `${TOPPO_ROOT}`, `${CONDA_ROOT}`, `${WANDB_ENTITY}` — so
they are readable as templates, but they will not run unmodified on another site. The ARS
launchers' variables are listed in the header of
`experiments/sac_baselines_ars/slurm/site_a100/mt50_run.sbatch`. `docs/env_vars.md` documents the
runtime switches that change behavior, including `MTRL_PCGRAD_CPU=1`, a workaround for a CUDA
crash (exit 136) that some GPUs hit in PCGrad-actor configurations such as the headline stack.

## License

Our code is MIT (`LICENSE`), except `experiments/amp_humanoid/`, which modifies and extends
MimicKit and is distributed under MimicKit's Apache License 2.0; that directory carries its own
`LICENSE` (MimicKit's, verbatim) and `NOTICE`. Third-party components and adapted code keep their
own licenses and are recorded in `NOTICE`.

## Citation

```bibtex
@inproceedings{li2026toppo,
  title     = {{TOPPO}: Rethinking {PPO} for Multi-Task Reinforcement Learning with Critic Balancing},
  author    = {Li, Yuanpeng and Lin, Gefei and Qu, Annie and Miao, Rui},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
