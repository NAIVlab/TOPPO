# Changes relative to upstream ARS @ `67f7df3`

Summary: **11 files modified (+1,383 / −205, net +1,178), 2 replaced, 16 added (11 code files,
4 documents, 1 patch), 1 vendored tree removed.**

By line count roughly a third of the shipped code is ours: 1,449 lines in files we wrote
from scratch, plus a net +1,178 lines in upstream files (+1,383 / −205), against 5,003
upstream lines still present across 22 files (11 modified, 11 byte-identical to `67f7df3`).
Both training entry points more than doubled (`train_ars.py` 232 -> 532,
`train_ars_lora.py` 239 -> 678). This matters for interpreting results: the numbers we report
come from this port under our own evaluation protocol, **not** from running upstream's code,
so they are matched-protocol re-runs rather than reproductions of the published figures.

## Modified

Lines are `git apply --numstat` of `patches/0001-toppo-modifications-to-upstream.patch`
against `67f7df3` (added / removed).

| File | Lines | What we changed |
|---|---:|---|
| `train_ars.py` | +336 / −36 | Meta-World 3.0.0 backend selection, V1/V2 reward-version switch, the paper evaluation protocol (50 deterministic goals per task) as the default config, PopArt / α-fairness / actor-PCGrad flags, W&B naming overrides and the default W&B project `toppo-repro`, preemption-safe resume (an unloadable checkpoint is quarantined after falling back), no ARS reset at the final iteration |
| `train_ars_lora.py` | +486 / −47 | the same for the LoRA arm, plus checkpointing of the LoRA-phase state (staged into the atomic checkpoint), environment workers forked before JAX is imported, and no reset or LoRA attach at the final iteration |
| `learner_sac_mt.py` | +163 / −94 | per-side update dispatch (`actor_method`, `critic_method`, `use_popart`): PopArt critic normalization, α-fairness critic aggregation, actor or critic PCGrad |
| `utils.py` | +132 / −17 | backend dispatch (`make_benchmark` / `make_eval_benchmark`) with lazy imports so an unused backend is never imported; total-env-step W&B x-axis and PPO-aligned success keys |
| `critic.py` | +98 / −0 | `sac_update_q_popart`, the critic update in PopArt-normalized space |
| `common.py` | +97 / −2 | `MultiModelFairGrad`, a multi-task model whose gradient step combines per-task gradients with FairGrad (host solver in `fairgrad_host.py`) |
| `dataset_utils.py` | +13 / −4 | unused `d4rl` / old-`gym` imports dropped; float64 observations stored as float32 in the replay buffer |
| `evaluation.py` | +1 / −1 | `gymnasium` import |
| `value_net.py` | +14 / −0 | PopArt per-task affine head on the critic |
| `wrappers/__init__.py` | +10 / −2 | old-gym wrappers imported only when old gym is installed |
| `wrappers/vec_env.py` | +33 / −2 | MuJoCo models of the Meta-World 3.0.0 workers built in bounded batches |

## Replaced (ours outright, not in the patch)

| File | Why |
|---|---|
| `README.md` | describes this port; carries upstream's citation request and BibTeX |
| `requirements.txt` | rewritten for Meta-World 3.0.0 / JAX 0.4.24; upstream's targets the vendored mujoco_py-era stack |

## Added (ours, MIT)

| File | Purpose |
|---|---|
| `metaworld3_ars.py` | Meta-World 3.0.0 env construction; `build_eval_rounds` implements the 50-goals-per-task deterministic protocol |
| `popart_jax.py` | PopArt target normalization for the JAX critic |
| `fairgrad_host.py` | host-side α-fairness solver (numpy port of the PPO stack's y-space Newton solver) |
| `checkpoint_utils.py` | preemption-safe checkpointing: a complete checkpoint survives a kill at any point of a save, and trainer side files are staged before publication |
| `scripts/` (2 files), `slurm/` (5 files) | launchers and environment setup |
| `CHANGES.md`, `EVALUATION_PROTOCOL.md`, `NOTICE`, `PROVENANCE.md` | this record, the evaluation protocol behind the reported numbers, attribution, base commit |
| `patches/0001-toppo-modifications-to-upstream.patch` | the modifications above as an applyable diff |

## Removed

`metaworld/` — 1006 files of vendored mujoco_py-era Meta-World shipped by
upstream without a license or notice. See `PROVENANCE.md`.

## Not included in this release

A second environment backend that was explored during development is out of
scope for this paper and has been removed along with its dispatch branches,
config fields and launcher entries.
