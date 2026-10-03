# Changes relative to MimicKit @ `2ed1e6c`

Summary: **8 upstream files modified (+865 / −33), 11 modules added, 27 configuration files
derived from upstream configurations and 1 added, 1 tool and 4 test files added; no upstream file
deleted.** Upstream's `README.md` and `requirements.txt` are not modified in the assembled tree;
this directory's `README.md` and `requirements.txt` replace them for these experiments.
Everything here is offered under Apache-2.0, the license of MimicKit (see `NOTICE`).

## Modified upstream files

Lines are `git apply --numstat` of `patches/0001-toppo-modifications-to-mimickit-2ed1e6c.patch`
against `2ed1e6c` (added / removed). Each count includes the two-line change notice at the top
of the file (Apache-2.0 4(b)).

| File | Lines | What we changed, and why it is needed |
|---|---:|---|
| `mimickit/learning/ppo_agent.py` | +376 / −8 | The TOPPO path, enabled by a `toppo:` block in the agent config: per-component TD(λ) returns and normalized advantages for K reward components, per-component critic and actor losses, and a combiner step (`default` mean, `pcgrad`, `fairgrad`) on the shared parameters before clipping and the optimizer step; `critic_coef` rescales the combined critic gradient after the combiner. Configs without a `toppo:` block take the original code path. |
| `mimickit/learning/ppo_model.py` | +85 / −4 | K-head critic on a shared trunk, optional biasless LayerNorm after every critic linear layer, per-component PopArt (`toppo.ComponentPopArt`), and the scalar value as the weighted sum of the component values. |
| `mimickit/learning/amp_agent.py` | +179 / −3 | Stores the two weighted reward components (task, style) for the TOPPO path, and adds the detailed evaluation: exact per-episode statistics, a fixed-size reservoir of AMP observations and the style FID (`eval_util.py`). |
| `mimickit/run.py` | +115 / −9 | `--detailed_eval`, `--eval_output`, `--eval_seed`, `--style_samples`, `--style_projection_dim`, `--style_seed`; writes the evaluation record (JSON) with the config paths and sha256 hashes, the checkpoint sha256 and runtime metadata (GPU, CUDA, torch, Python, numpy, simulator package versions). |
| `mimickit/envs/task_steering_env.py` | +68 / −0 | Early termination when the root falls below `min_root_height` or tilts below `min_root_upright` (the walk env sets both to 0.5, so training depends on it), and the steering evaluation metrics (projected speed, speed and velocity error, facing alignment, root height and uprightness). |
| `mimickit/engines/newton_engine.py` | +25 / −8 | Builds the ground contact sensor with whichever argument and output names the installed Newton exposes (`include_total` / `measure_total`, `net_force` / `total_force`); needed for newton 1.3.0. Also adds the missing final newline. |
| `mimickit/learning/agent_builder.py` | +11 / −0 | Registers the `W5AMP`, `SAC` and `AMPSAC` agents. |
| `mimickit/envs/base_env.py` | +6 / −1 | Default `get_eval_metrics()` returning no metrics. |

### Differences from the code that produced the runs

The research versions of these 8 files are recoverable as a diff against `2ed1e6c` (+854 / −33).
The released patch differs from it only by:

* the two-line change notice at the top of each file (16 lines);
* `ppo_agent.py`: one code comment shortened (it named an internal repository);
* `run.py`: the evaluation record no longer stores the host name, the scheduler job id or
  `CUDA_VISIBLE_DEVICES`, and the then-unused `socket` import is gone. No metric, config or seed
  handling changes; the shipped records had the same three fields removed.

## Added modules (`overlay/mimickit/learning/`)

| File | Lines | Purpose |
|---|---:|---|
| `toppo.py` | 392 | PCGrad; FairGrad with the y-space Newton solver and a SciPy least-squares fallback (formulation of OptMN-Lab/FairGrad, MIT); `ComponentPopArt` (Welford or EMA statistics) |
| `eval_util.py` | 297 | Style FID and its feature statistics, exact masked episode statistics, the feature reservoir. Byte-identical to the research file (sha256 `84d2ff42…`). |
| `w5_norm.py` | 303 | The alternative critic normalizers: per-component reward normalization (update order of openai/baselines `VecNormalize`), return-based scaling (Schaul et al., 2021), HL-Gauss value head (Farebrother et al., 2024) |
| `w5_amp_agent.py` | 219 | AMP-PPO with PopArt replaced by one of the `w5_norm` methods (agent name `W5AMP`) |
| `w5_amp_model.py` | 67 | HL-Gauss critic head for `W5AMPAgent` |
| `sac_agent.py` | 730 | Soft actor-critic on the MimicKit agent interface, with the TOPPO critic stack on the twin-Q critic, the `w5_norm` alternatives, and a port of ARS (Cho et al., 2025) to the reward-component axis |
| `sac_model.py` | 306 | Twin-Q critic with optional component heads, LayerNorm, PopArt or HL-Gauss heads |
| `amp_sac_agent.py` | 490 | Off-policy AMP (agent name `AMPSAC`): style reward re-labeled with the current discriminator at update time; discriminator training follows upstream `amp_agent.py` |
| `amp_sac_model.py` | 51 | `SACModel` plus the AMP discriminator (follows upstream `amp_model.py`) |
| `sacamp_fairgrad.py` | 189 | FairGrad for the SAC arms with a fp32 Gram matrix and gradient preconditioning; diagnostics reported in raw gradient space |
| `distribution_tanh_gaussian.py` | 81 | Tanh-squashed Gaussian policy for SAC (builder and log-probability adapted from upstream `distribution_gaussian_diag.py`) |

Five of these modules contain code adapted from upstream MimicKit and carry a header saying so:
`amp_sac_agent.py` and `amp_sac_model.py` (from `amp_agent.py` and `amp_model.py`), `sac_agent.py`
(`load()` extends `base_agent.py`'s, `_decide_action()` follows `ppo_agent.py`), `sac_model.py`
(actor construction and input dictionaries follow `ppo_model.py`) and
`distribution_tanh_gaussian.py` (layer construction, initialization and the Gaussian
log-probability follow `distribution_gaussian_diag.py`). The other six modules (`toppo.py`,
`eval_util.py`, `w5_norm.py`, `w5_amp_agent.py`, `w5_amp_model.py`, `sacamp_fairgrad.py`) are new
code written for this work. Some of them subclass or call MimicKit classes, but they share no code
with MimicKit beyond imports, method signatures and single API calls.

## Configuration and data files (`overlay/data/`)

The 25 agent configs and the 2 env configs are derived from MimicKit configs, so each carries a
change notice at its top (Apache-2.0, section 4(b)) naming the upstream file it comes from. The
notice is a comment: every file still parses (`yaml.safe_load`) to exactly the configuration that
ran (`provenance/config_hashes.tsv`).

* `agents/`: the 25 agent configs of the campaign, one per W&B group (`README.md`, "Runs on W&B").
  * The 9 PPO configs (`amp_task_humanoid_*_agent.yaml`) are modified copies of upstream
    `data/agents/amp_task_humanoid_agent.yaml`. The base PPO config,
    `amp_task_humanoid_long_critic_clip1_agent.yaml`, changes only `iters_per_output` (200 to 10)
    and adds `critic_optimizer.grad_clip: 1.0`. The other eight build on it: a `toppo:` block (the
    two-component critic, LN-c, PopArt, FairGrad and PCGrad settings, with critic gradient clipping
    set to 0.0 in the FairGrad and full-stack rows), `agent_name: W5AMP` plus a `w5_norm:` block
    for the alternative critic normalizers, `disc_reward_weight: 1.105` for `_ppo_reweight`, and a
    critic learning rate of 1e-3 for `_w5_hlgauss_lr1e3`.
  * The 16 SAC configs (`amp_sac_walk_*_agent.yaml`) reuse upstream's network sizes, discriminator
    network and settings, discount and reward weights from the same file, and replace its
    PPO-specific settings with the SAC ones (agent `AMPSAC`, Adam optimizers, replay buffer, update
    schedule, entropy tuning). The variants add `toppo:`, `w5_norm:` or `ars:` blocks or set
    `disc_reward_weight: 1.105`.
  * `tests/test_campaign_configs.py` checks every documented difference between the configs.
  * Apart from the change notice, 10 are byte-identical to the files that ran. 15 also had their
    leading comment block rewritten because it referred to internal tracking notes or log paths:
    `amp_task_humanoid_w5_{rewnorm,retscale,hlgauss}_agent.yaml` and
    `amp_sac_walk_{lnonly,popartonly,k1cd,cb,fullstack,cbreweight,ars,arsreset,utd512,w5_rewnorm,w5_retscale,w5_hlgauss}_agent.yaml`.
    In `amp_sac_walk_utd512_agent.yaml` a stale comment on the update schedule (it still gave the
    base arm's 64 updates per iteration) was also corrected to match the value, 512.
  * File names are unchanged because the evaluation records and the W&B runs refer to them.
* `envs/amp_steering_humanoid_walk_env.yaml` (training) and
  `envs/amp_steering_humanoid_walk_eval_env.yaml` (evaluation): modified copies of upstream
  `data/envs/amp_steering_humanoid_env.yaml` with the DeepMimic walk clip as the motion file, a
  20 s episode (upstream 10 s), target speeds in [0.7, 1.3] m/s (upstream [0.5, 5.0]), target
  changes every 5–10 s (upstream 4–7 s), and fall termination at root height or uprightness 0.5
  (`min_root_height`, `min_root_upright`, which the patch to `task_steering_env.py` adds).
  Upstream's inline unit comments are dropped. The evaluation env differs from the training env
  only in `tar_change_time_min/max` (100 s instead of 5–10 s), so the target does not change
  within an evaluation episode.
* `datasets/dataset_humanoid_deepmimic_walk_smoke.yaml` (added, in MimicKit's dataset format): the
  single-clip walk dataset. Despite the name, it is the dataset of every run.

## Added tools and tests

* `overlay/tools/deepmimic_to_mimickit/deepmimic_to_mimickit.py`: converts a DeepMimic `humanoid3d`
  motion file to MimicKit's motion format (coordinate frame, joint order, quaternion to exponential
  map).
* `overlay/tests/`: `test_toppo.py` (18 tests of the TOPPO components, the K-head critic and the
  evaluation statistics), `test_sac_amp.py` and `test_w5_norm.py` (SAC agents and critic
  normalizers), and `test_campaign_configs.py` (new; the documented differences between the 25
  configs). `test_toppo.py` drops 10 tests of the research tree that need files not shipped here
  (exploratory configs, argument files, other analysis tools, an Isaac Lab port). All 139 tests
  pass on CPU.

## Not included

* Model checkpoints (none of the 250 is released), training logs, episode-level evaluation arrays.
* Exploratory configurations that are not among the 25 groups (other motions, a location task,
  critic-coefficient and Gram-precision variants) and the change to `task_location_env.py` that
  only the location task used.
* An Isaac Lab port that was explored and not used by any published run.
* Cluster launch scripts; `campaign.py` and `scripts/run_arm.sh` replace them.

## Naming crosswalk

| Here (`arm`, analysis scripts) | W&B `toppo/arm` | Paper table row |
|---|---|---|
| `ppo_base` | `ppo_vanilla` | PPO, Base |
| `ppo_popart` | `ppo_criticbal` | PPO, Two-component critic + LN + PopArt |
| `ppo_fairgrad` | `ppo_toppo` | PPO, Above + FG-c, no critic grad clip |
| `ppo_pcgrad` | `ppo_fulltoppo` | PPO, Full stack (also PCGrad-a) |
| `sac_base` | `sac_vanilla` | SAC, Base |
| `sac_pcgrad` | `sac_fullstack` | SAC, Full stack |

The remaining 19 groups are in `campaign.py` and `README.md`. `w5` (file names, `W5AMP`,
`w5_norm`) is an internal development label for the critic-normalizer family, kept because the
evaluation records, the config hashes and the W&B runs refer to these file and agent names.
