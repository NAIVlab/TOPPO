# AMP humanoid walk steering: a two-objective stress test

Code, configurations and per-seed evaluation records for the AMP humanoid experiments of the
TOPPO paper: the appendix section "AMP humanoid: a two-objective control stress test"
(Appendix E.5, with Table E.6 and Figure E.4) and the 76% sentence in Section 5.3.

**This is a two-objective mechanism stress test, not a second multi-task benchmark suite.**
One simulated humanoid learns to walk in a commanded direction at a commanded speed while
staying close to a reference walking clip, with an adversarial motion prior (AMP; Peng et al.,
2021). The reward has two components, task tracking and motion style (the AMP discriminator
reward), weighted 0.5 / 0.5. The question is whether TOPPO's critic-side stack still helps
when the "tasks" are two reward components that share one critic trunk.

The code is a patch plus additions on top of [MimicKit](https://github.com/xbpeng/MimicKit)
(Peng, 2025) at commit `2ed1e6c`, which provides the humanoid character, the steering
environment and the AMP agent. The reference clip is `humanoid3d_walk` from the public
[DeepMimic](https://github.com/xbpeng/DeepMimic) motion data (Peng et al., 2018). Every
training and evaluation run used the [Newton](https://github.com/newton-physics/newton)
physics simulator (1.3.0, MuJoCo-Warp solver). No run used Isaac Gym or Isaac Lab.

## What the paper reports

The campaign has **25 configurations x 10 training seeds** (seeds `260511470` to `260511479`),
all public on W&B (below). The paper's table reports six of them; every configuration in the
table trains for 100M samples. The final checkpoint of each run is evaluated on 4,096 episodes
under one common evaluation environment, with 8,192 motion-style samples; one record per
(group, seed) is canonical.

The style metric is the **squared** Frechet distance (called style FID) between policy and
demonstration AMP observation features, after standardizing by the demonstration statistics
and applying a fixed 64-dimensional random projection. Every arm is scored in the same
feature space; no policy is scored by its own discriminator. Task return is reported next to
it because better motion similarity can cost task tracking. Details are in
[`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md).

| Family | Configuration | W&B group | Task return | Style FID |
|---|---|---|---:|---:|
| PPO | Base | `ampwalk_ppo_ad_cd_nnone_ln0_k1_gc1` | 533.56 ± 20.74 | (5.477 ± 1.327) x 10^4 |
| PPO | Two-component critic + LN + PopArt | `ampwalk_ppo_ad_cd_npaw_lnc_k2_gc1` | 534.99 ± 16.89 | (3.855 ± 0.871) x 10^4 |
| PPO | Above + FG-c, no critic grad clip | `ampwalk_ppo_ad_cfg1_npaw_lnc_k2_gc0` | 550.63 ± 9.01 | (3.635 ± 3.943) x 10^4 |
| PPO | Full stack (also PCGrad-a) | `ampwalk_ppo_apc_cfg1_npaw_lnc_k2_gc0` | 521.11 ± 14.64 | (1.315 ± 0.088) x 10^4 |
| SAC | Base | `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0` | 573.77 ± 4.03 | (5.596 ± 1.522) x 10^6 |
| SAC | Full stack | `ampwalk_sac_apc_cfg1_npaw_lnc_k2_gc0` | 575.33 ± 2.57 | (4.486 ± 1.108) x 10^6 |

Mean ± sample std over the 10 training seeds. Lower FID is better.

* **PPO full stack vs. base:** mean style FID −76.0%, lower in all ten seeds; paired
  d = −41,623 (95% CI [−51,357, −31,889]), i.e. [−93.8%, −58.2%] of the base mean;
  p = 4.7 x 10^−6, significant after Bonferroni correction over the 25 configurations.
  This is evidence for the complete intervention under this protocol. It does not identify
  separate module effects: adding FairGrad on the critic also turns critic gradient clipping
  off (the full stack keeps that change), and the PCGrad step is not significant relative to
  the FairGrad rung (paired d = −2.32 x 10^4, 95% CI [−5.16, 0.52] x 10^4, p = 0.098).
* **The full-stack PPO policy walks more slowly:** paired change in projected speed −0.416
  (95% CI [−0.529, −0.303]); the change in speed error is not significant (95% CI
  [−0.119, 0.023]). The style gain is not a uniform improvement across both objectives.
* **SAC full stack vs. base:** mean style FID −19.8%, but not statistically reliable at n = 10
  (p = 0.078; 95% CI of the relative difference [−42.4%, 2.8%]). The interval shows neither a
  reliable benefit nor equivalence. The SAC FairGrad path forms its Gram matrix in fp32 after
  preconditioning, unlike the PPO path, so cross-family module comparisons also differ in
  implementation.
* **Cost** (one short H200 run each, about 8M samples, not full-budget estimates): the PPO full
  stack adds 4.2% end-to-end time (189.5 to 197.5 s), the SAC full stack 49.5%
  (287.6 to 430.1 s). See [`results/timing/timing.tsv`](results/timing/timing.tsv).
* **Evaluation drift:** evaluations ran on several GPU types. For the four PPO table groups, 45
  repeat evaluations of 33 of their checkpoints (42 of them on other GPU types) moved style FID by
  at most 0.994%; using those records instead of the canonical ones moves the 76.0% reduction by
  at most 0.11 percentage points, and the PCGrad step stays non-significant (p = 0.099 with the
  RTX 3090 records; `analysis/eval_drift.py`). No SAC checkpoint was re-evaluated, and the SAC
  comparison is not device-matched: the SAC base records come from three GPU types (RTX 3090, RTX
  PRO 6000, H200 NVL), the SAC full stack's from H200 NVL only. A drift of about 1% is small next
  to the SAC base seed std (27% of its mean). See
  [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md).

All 25 configurations, including the controls the paper does not tabulate (constant style
reweighting, ARS, alternative critic normalizers, a target-entropy match, a higher
update-to-data ratio), are summarized in [`results/campaign_summary.md`](results/campaign_summary.md),
regenerated by `analysis/campaign_table.py`.

## Contents

| Path | What it is |
|---|---|
| `setup.sh` | Clones MimicKit, checks out `2ed1e6c`, applies the patch, copies `overlay/`, fetches and converts the walk clip, and verifies the result against `MANIFEST.sha256` |
| `patches/0001-toppo-modifications-to-mimickit-2ed1e6c.patch` | Our changes to 8 upstream files (+865 / −33) |
| `overlay/` | Files outside the patch: 11 new modules in `mimickit/learning/`, the 25 agent configs and 2 env configs (derived from MimicKit's configs; each starts with a change notice), 1 dataset config, the motion converter, 4 CPU test files |
| `campaign.py` | The 25-row campaign table and the exact train and eval commands |
| `scripts/run_arm.sh`, `scripts/time_arm.sh` | Train + evaluate one (group, seed) on one GPU; the short timing run |
| `analysis/` | Table, figure, headline, validation, drift and summary scripts (no GPU, no simulator) |
| `results/eval/` | The 250 evaluation records behind every number above (1.1 MB) |
| `results/reevals/` | 45 extra evaluations of 33 checkpoints of the four PPO table groups, mostly on other GPUs (0.2 MB); used only for the drift and record-choice checks |
| `results/train_samples.tsv`, `results/timing/timing.tsv`, `results/campaign_summary.md` | Final training sample counts from W&B, the timing measurements, the 25-group summary |
| `provenance/` | `config_hashes.tsv` (recorded, released and comment-independent config hashes) and `eval_files.tsv` (every shipped record mapped to its original file and hashes) |
| `third_party/deepmimic/` | DeepMimic's MIT license and the hashes of the walk clip; the clip itself is downloaded by `setup.sh` |
| `PROVENANCE.md`, `CHANGES.md`, `EVALUATION_PROTOCOL.md`, `NOTICE`, `LICENSE` | Base commits, every change we made, the metric definition, attribution, MimicKit's Apache-2.0 license |

**No model checkpoints are released.** Each evaluation record carries the `model_sha256` of the
checkpoint it scored, so a checkpoint published later can be matched to its record. The style
FID cannot be recomputed without checkpoints (the records hold aggregate metrics, not
features); every seed-level number in the paper can be recomputed from the records.

## Install

Linux, Python 3.12, one NVIDIA GPU with at least 20 GB of memory (4,096 environments; the
8M-sample timing runs peaked at 13,964–16,794 MiB on an H200 NVL), `git`, `curl` and `tar`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
bash setup.sh                 # creates ./mimickit_toppo (or: bash setup.sh <dir>)
```

`requirements.txt` pins the versions recorded in the evaluation records: torch 2.10.0+cu128
(CUDA 12.8), newton 1.3.0, warp-lang 1.15.0, mujoco 3.10.0 and mujoco-warp 3.10.0.1 (all 250
records), numpy 2.5.1 (225 records; the other 25 show 2.3.5).
Upstream MimicKit's README says it was tested with Newton v1.0.0; the runs used 1.3.0, and the
patch to `newton_engine.py` makes the contact-sensor call work with both sensor APIs. Do not
install `newton[sim]`: that extra pins mujoco 3.8. Other packages were not recorded and are left
unpinned.

`setup.sh` needs the Python environment because it converts the DeepMimic clip with torch. It
stops if any hash differs: the downloaded clip, the converted `.pkl`, or any of the files listed
in `MANIFEST.sha256`.

## Train and evaluate

`campaign.py` (in this directory) prints the commands; run the printed commands from the root of
the assembled tree.

```bash
python campaign.py list
python campaign.py train ampwalk_ppo_apc_cfg1_npaw_lnc_k2_gc0 260511470
python campaign.py eval  ampwalk_ppo_apc_cfg1_npaw_lnc_k2_gc0 260511470
python campaign.py commands > all_500_commands.txt        # 25 groups x 10 seeds x (train, eval)
```

Training (here the PPO full stack, seed 260511470):

```bash
python mimickit/run.py --mode train --num_envs 4096 \
  --engine_config data/engines/newton_engine.yaml \
  --env_config data/envs/amp_steering_humanoid_walk_env.yaml \
  --agent_config data/agents/amp_task_humanoid_fulltoppo_cc100_agent.yaml \
  --out_dir output/ampwalk_ppo_apc_cfg1_npaw_lnc_k2_gc0_s260511470/ \
  --max_samples 100000000 --rand_seed 260511470 \
  --devices cuda:0 --logger txt --visualize false
```

Evaluation of the final checkpoint:

```bash
python mimickit/run.py --mode test --num_envs 4096 \
  --engine_config data/engines/newton_engine.yaml \
  --env_config data/envs/amp_steering_humanoid_walk_eval_env.yaml \
  --agent_config data/agents/amp_task_humanoid_fulltoppo_cc100_agent.yaml \
  --model_file output/ampwalk_ppo_apc_cfg1_npaw_lnc_k2_gc0_s260511470/model.pt \
  --devices cuda:0 --visualize false --test_episodes 4096 --detailed_eval true \
  --style_samples 8192 --style_projection_dim 64 --style_seed 260511701 \
  --eval_seed 260511499 \
  --eval_output results/eval/ampwalk_ppo_apc_cfg1_npaw_lnc_k2_gc0_s260511470.json
```

Every group uses these two templates; only `--agent_config`, the seed and, for
`ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_utd512`, `--max_samples 25000000` change. All 250 training
argument lists of the campaign match the training template (checked against the original
argument files), and every evaluation record states the evaluation settings it used.

`scripts/run_arm.sh <tree> <group> <seed>` runs both steps on one GPU. Keep it to one process
(`--devices cuda:0`): the SAC and ARS code paths assert a single process. As a rough guide,
extrapolating the steady-state throughput in `results/timing/timing.tsv` (H200 NVL, 8 CPU cores),
100M samples take about 30 minutes of training for PPO and 50–80 minutes for SAC, plus
start-up and evaluation.

## Regenerate the table, figure and headline

These scripts read `results/`, `provenance/` and the configs in `overlay/`; they need Python
with scipy, PyYAML, numpy and matplotlib, but no GPU and no simulator. Run them from this
directory.

```bash
python analysis/validate_campaign.py   # coverage, protocol and provenance of the 250 records
python analysis/amp_table.py --check   # the six table rows and all five paired tests; exit 1 on any mismatch
python analysis/amp_table.py --latex   # the table rows as LaTeX
python analysis/amp_figure.py          # amp_transfer.pdf / .png (Figure E.4)
python analysis/eval_drift.py          # the 0.994% drift, the record-choice check, the W&B values below
python analysis/campaign_table.py      # all 25 groups
python analysis/summarize_arms.py --family ppo   # every metric, paired deltas vs. the PPO base
```

`amp_table.py` reproduces the archived full-precision values of the paper's computation
bit for bit, including the 76.0% headline (p = 4.71 x 10^−6) and all confidence intervals.

## Runs on W&B

Logs: [https://wandb.ai/liyp1220-uci/TOPPO](https://wandb.ai/liyp1220-uci/TOPPO), 250 runs in the
25 groups below, 10 seeds each. The group name is the canonical identifier. `arm` is the name
the analysis scripts use; `toppo/arm` is the label stored on the W&B runs.

| W&B group | arm | `toppo/arm` | agent config (`data/agents/`) | samples | paper |
|---|---|---|---|---:|---|
| `ampwalk_ppo_ad_cd_nnone_ln0_k1_gc1` | ppo_base | ppo_vanilla | `amp_task_humanoid_long_critic_clip1_agent.yaml` | 100M | table row 1 |
| `ampwalk_ppo_ad_cd_npaw_lnc_k2_gc1` | ppo_popart | ppo_criticbal | `amp_task_humanoid_component_welford_ln_clip1_agent.yaml` | 100M | table row 2 |
| `ampwalk_ppo_ad_cfg1_npaw_lnc_k2_gc0` | ppo_fairgrad | ppo_toppo | `amp_task_humanoid_toppo_cc100_noclip_agent.yaml` | 100M | table row 3 |
| `ampwalk_ppo_apc_cfg1_npaw_lnc_k2_gc0` | ppo_pcgrad | ppo_fulltoppo | `amp_task_humanoid_fulltoppo_cc100_agent.yaml` | 100M | table row 4; 76% headline |
| `ampwalk_ppo_ad_cd_nnone_ln0_k1_gc1_reweight` | ppo_rw | ppo_reweight | `amp_task_humanoid_ppo_reweight_agent.yaml` | 100M | campaign |
| `ampwalk_ppo_ad_cd_nrew_lnc_k2_gc1` | ppo_rewnorm | ppo_w5rewnorm | `amp_task_humanoid_w5_rewnorm_agent.yaml` | 100M | campaign |
| `ampwalk_ppo_ad_cd_nret_lnc_k2_gc1` | ppo_retscale | ppo_w5retscale | `amp_task_humanoid_w5_retscale_agent.yaml` | 100M | campaign |
| `ampwalk_ppo_ad_cd_nhg_lnc_k2_gc1` | ppo_hlgauss | ppo_w5hlgauss | `amp_task_humanoid_w5_hlgauss_agent.yaml` | 100M | campaign |
| `ampwalk_ppo_ad_cd_nhg_lnc_k2_gc1_lr1e3` | ppo_hlgauss_lr10x | ppo_w5hlgauss_lr1e3 | `amp_task_humanoid_w5_hlgauss_lr1e3_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0` | sac_base | sac_vanilla | `amp_sac_walk_agent.yaml` | 100M | table row 5 |
| `ampwalk_sac_ad_cd_nnone_lnc_k1_gc0` | sac_ln | sac_lnonly | `amp_sac_walk_lnonly_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cd_npaw_ln0_k1_gc0` | sac_popart | sac_popartonly | `amp_sac_walk_popartonly_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cd_npaw_lnc_k1_gc0` | sac_ln_popart | sac_k1cd | `amp_sac_walk_k1cd_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cd_npaw_lnc_k2_gc0` | sac_k2 | sac_cd | `amp_sac_walk_cd_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cfg1_npaw_lnc_k2_gc0` | sac_fairgrad | sac_cb | `amp_sac_walk_cb_agent.yaml` | 100M | campaign |
| `ampwalk_sac_apc_cfg1_npaw_lnc_k2_gc0` | sac_pcgrad | sac_fullstack | `amp_sac_walk_fullstack_agent.yaml` | 100M | table row 6 |
| `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_reweight` | sac_rw | sac_reweight | `amp_sac_walk_reweight_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cfg1_npaw_lnc_k2_gc0_reweight` | sac_fairgrad_rw | sac_cbreweight | `amp_sac_walk_cbreweight_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_ars` | sac_ars | sac_ars | `amp_sac_walk_ars_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_arsreset` | sac_ars_reset | sac_arsreset | `amp_sac_walk_arsreset_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_entmatch` | sac_targetent | sac_entmatch | `amp_sac_walk_entmatch_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_utd512` | sac_utd512 | sac_utd512 | `amp_sac_walk_utd512_agent.yaml` | **25M** | campaign |
| `ampwalk_sac_ad_cfg1_nrew_lnc_k2_gc0` | sac_rewnorm | sac_w5rewnorm | `amp_sac_walk_w5_rewnorm_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cfg1_nret_lnc_k2_gc0` | sac_retscale | sac_w5retscale | `amp_sac_walk_w5_retscale_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cfg1_nhg_lnc_k2_gc0` | sac_hlgauss | sac_w5hlgauss | `amp_sac_walk_w5_hlgauss_agent.yaml` | 100M | campaign |

Group-name grammar:

```
ampwalk_{ppo,sac}_a{d,pc}_c{d,fg1}_n{none,paw,rew,ret,hg}_{ln0,lnc}_k{1,2}_gc{0,1}[_reweight|_ars|_arsreset|_entmatch|_utd512|_lr1e3]
```

| field | values |
|---|---|
| algorithm | `ppo` (AMP-PPO), `sac` (off-policy AMP-SAC) |
| actor `a…` | `ad` default actor update (one loss) · `apc` PCGrad over the two per-component actor losses |
| critic `c…` | `cd` default (one loss; the mean of the component losses when `k2`) · `cfg1` FairGrad, α = 1, over the component value losses on the shared trunk |
| normalizer `n…` | `none` · `paw` PopArt (Welford) · `rew` reward normalization · `ret` return-based scaling · `hg` HL-Gauss |
| LayerNorm | `ln0` none · `lnc` critic LayerNorm (biasless) |
| `k…` | number of critic heads (reward components): 1 or 2 |
| `gc…` | critic gradient clipping: `gc1` clip at 1.0 · `gc0` off |
| suffix | `_reweight` style weight 0.5 -> 1.105 · `_ars` ARS reward scaling · `_arsreset` ARS with learner resets · `_entmatch` SAC target entropy matched to PPO's fixed-std policy · `_utd512` 8x updates per sample (25M samples) · `_lr1e3` critic learning rate x10 |

`w5` is an internal development label for the critic-normalizer family (reward normalization,
return-based scaling, HL-Gauss). It is kept in the file, module, agent (`W5AMP`), config-key
(`w5_norm`) and log-key names because the evaluation records, the config hashes and the W&B
runs refer to these names.

### Caveats on the W&B copies

1. **`ampwalk_ppo_ad_cd_npaw_lnc_k2_gc1`, seeds 260511473–260511475.** W&B shows a second
   evaluation of these three checkpoints, run on an RTX 3090; the paper and `results/eval/` use
   the H200 evaluation of the same checkpoints (same `model_sha256`, same protocol). The group
   therefore reads style FID 38,579.7 ± 8,715.3 and task return 534.994 ± 16.896 on W&B, against
   38,553.5 ± 8,705.7 and 534.988 ± 16.895 in the paper (per-seed drift +0.46%, +0.16%, +0.01%).
   The three RTX 3090 evaluations ship in `results/reevals/`; `analysis/eval_drift.py`
   reproduces both sets of values. No conclusion changes: this group is in none of the headline paired tests.
2. On all 10 runs of that group, `toppo/source_arm` reads `3way_criticbal`. The records actually
   used are listed in `provenance/eval_files.tsv` (`original_file`).
3. **25 runs have an evaluation summary but no training history.** Their training budget cannot
   be confirmed from public data; `analysis/validate_campaign.py` lists them as warnings
   (`results/train_samples.tsv`: `NA`). Three of them are seeds of the SAC base row of the table.
   * seeds 260511473–260511474: `ampwalk_ppo_ad_cd_nnone_ln0_k1_gc1_reweight`
   * seeds 260511473–260511475: `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0`,
     `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_ars`, `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_arsreset`,
     `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_entmatch`, `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_reweight`,
     `ampwalk_sac_ad_cd_npaw_lnc_k1_gc0`, `ampwalk_sac_ad_cd_npaw_lnc_k2_gc0`
   * seed 260511473: `ampwalk_sac_ad_cd_nnone_lnc_k1_gc0`, `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_utd512`
4. The `requirements.txt` attached to the W&B runs describes the environment used to re-log the
   runs, not the training environment. Use `requirements.txt` in this directory.
5. W&B does not record the engine or environment config of these runs; every evaluation record
   here does, and they are identical across all 250.

## Statistics

Every ± is the sample standard deviation over the 10 training seeds (ddof = 1), never the
standard error. Paired comparisons use the seeds shared by the two groups (always all ten):
the paired mean difference d with its 95% Student-t interval, and a two-sided paired t-test.
p > 0.05 means no difference was detected, not that the groups are equivalent. The evaluation
records also contain per-episode `*_std`, `*_sem` and `*_ci95` fields; they describe the 4,096
episodes of one checkpoint and are not used for seed-level reporting.

## Tests

From the assembled tree, CPU only (no simulator needed):

```bash
python -m pytest tests -q
```

`tests/test_campaign_configs.py` checks that every configuration differs from its neighbor in the
ladder exactly as documented; the other three files test the TOPPO components, the SAC agents
and the critic normalizers.

## License and attribution

Everything in this directory is under the Apache License 2.0 (`LICENSE`, MimicKit's license text
verbatim), because it modifies and extends MimicKit. The DeepMimic clip that `setup.sh` downloads
is MIT-licensed. `NOTICE` lists all third-party sources. If you use this code, please also cite
MimicKit, AMP and DeepMimic:

```bibtex
@article{peng2025mimickit,
  title   = {{MimicKit}: A Reinforcement Learning Framework for Motion Imitation and Control},
  author  = {Peng, Xue Bin},
  journal = {arXiv preprint arXiv:2510.13794},
  year    = {2025}
}
@article{peng2021amp,
  title   = {{AMP}: Adversarial Motion Priors for Stylized Physics-Based Character Control},
  author  = {Peng, Xue Bin and Ma, Ze and Abbeel, Pieter and Levine, Sergey and Kanazawa, Angjoo},
  journal = {ACM Transactions on Graphics},
  volume  = {40},
  number  = {4},
  year    = {2021}
}
@article{peng2018deepmimic,
  title   = {{DeepMimic}: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills},
  author  = {Peng, Xue Bin and Abbeel, Pieter and Levine, Sergey and van de Panne, Michiel},
  journal = {ACM Transactions on Graphics},
  volume  = {37},
  number  = {4},
  year    = {2018}
}
```
