# Run index

Every training run behind the paper, by W&B run group. The group name is the
canonical identifier: it appears in the public W&B project and is the key to look
a number up. Each Meta-World row gives the config and the exact command-line overrides
that reproduce it. The paper's AMP humanoid groups are listed separately under
[AMP humanoid stress test](#amp-humanoid-stress-test), with their configs in
`experiments/amp_humanoid/` and their own checks; `analysis/check_runs_index.py` checks only the
Meta-World PPO rows. Four measurements are not in the public W&B project: the compute timings
(Appendix E.4), the MT10 gradient-dump runs (Appendix D.2), the real-Gram solver benchmark
(Appendix C.5) and the AMP H200 timings (Appendix E.5); they are described under
[Not in the public W&B project](#not-in-the-public-wb-project).

**Logs:** [https://wandb.ai/liyp1220-uci/TOPPO](https://wandb.ai/liyp1220-uci/TOPPO) — every run here is public, with a canonical
`toppo/*` parameter block; the Meta-World runs also carry per-task metrics and their training
histories (see [History gaps](#history-gaps) for the exceptions).
To recompute a Meta-World row from these logs, run `python analysis/metrics.py --groups <group>`
from `experiments/metaworld_ppo/`; it reads liyp1220-uci/TOPPO by default. The AMP numbers are
recomputed from the evaluation records shipped in `experiments/amp_humanoid/` (see
[below](#amp-humanoid-stress-test)).
[Report 0](https://wandb.ai/liyp1220-uci/TOPPO/reports/--VmlldzoxNzg5MTc5MA==) is the guide to the project.

Meta-World PPO rows run from `experiments/metaworld_ppo/` (the SAC rows say below where they
run). Every Meta-World row is 10 seeds (`--seed 0` .. `--seed 9`), with two exceptions:
`mt50v1_ppo_ad_cd_nhg101rew_lnc_vc0_400x3` has 3 seeds (`--seed 0` .. `--seed 2`), as the paper
states, and in `mt10v2_ppo_ad_cfg2_nnone_ln0_vc1_400x3` (not in the paper) seeds 3 and 4 stopped
at 14.6M and 1.7M of their 20M steps (W&B tags `qc-partial-seeds`, `qc-truncated`).

```bash
python train.py --config <config> <overrides> \
    --seed <s> \
    --exp-name <group>_s<s> \
    --logging.wandb-group <group>
```

Such a run logs to the W&B project `toppo-repro` under your default entity (or `$WANDB_ENTITY`),
the default of every shipped config, not to the published project. Pass `--project-name` to
change it, and read the runs back with
`python analysis/metrics.py --entity <you> --project toppo-repro --groups <group>`.

On Slurm, `slurm/train.sbatch` runs one (row, seed): set `CONFIG_FILE=<config>`, `SEED=<s>` and
`EXP_NAME=<group>_s<s>`, and put `<overrides> --logging.wandb-group <group>` in `EXTRA_ARGS` (its
header shows how; `SEED` defaults to 0, so set it for every seed). The template adds no method,
normalizer or LayerNorm flag of its own, so every reproducible Meta-World PPO row below (all but
the three `_gjoint` rows, which have no config) trains its own arm through it.

Rows with a PCGrad actor (`apc`) can crash with exit 136 on some GPUs; if they do, prefix the
command with `MTRL_PCGRAD_CPU=1` (see [`env_vars.md`](env_vars.md)).

Runs are reproducible statistically (as seed-aggregate statistics over 10 seeds: mean ± std, or
IQM ± std on V1, as the paper reports them), not bit for bit: `--seed` seeds
Python/NumPy/PyTorch and `metaworld.MT*(seed=)` fixes each task's goal set, but the order in which
training episodes sample goals comes from Meta-World's unseeded env RNG (as in the published runs);
evaluation goals are iterated deterministically.

> **The two identity flags are not optional.** Each shipped config carries a fixed
> `exp_name` and `logging.wandb_group` of its own, and neither depends on `--seed`.
> Without `--exp-name` every seed of a row writes to the same directory
> (`logs/<exp_name>/`), and `train.py` **auto-resumes from whatever `latest.pth` it
> finds there without checking the seed** — so seed 1 silently continues seed 0's run,
> and if seed 0 already finished, the training loop is skipped entirely. It is worse
> across rows: 12 of the directories below are the default target of more than one run
> group (`logs/ppo_mt10_fairgrad` is shared by 8), and 72 of the 80 rows would otherwise
> log under a `wandb_group` that is not the canonical group name in the first column.
> Passing both flags makes each (row, seed) a distinct run, which is what the numbers in
> the reports are.

## Group name grammar

```
mt{10,50}v{1,2}_{ppo,sac}_a<actor>_c<critic>_n<normalizer>_<ln>_<vc>_<arch>[_<variant>]
```

| field | values |
|---|---|
| benchmark | `mt10` / `mt50`, then `v1` / `v2` = Meta-World reward function version |
| actor `a…` | `ad` default · `apc` PCGrad · `afg<α>` FairGrad · `aca<c>` CAGrad |
| critic `c…` | `cd` default · `cpc` PCGrad · `cfg<α>` FairGrad · `cca<c>` CAGrad |
| normalizer `n…` | `none` · `paw` PopArt (Welford) · `pae` PopArt (EMA) · `rew` reward normalization · `ret` return scaling · `rettr` return scaling with truncation · `hg<bins>[rew\|paw]` HL-Gauss · `ars` the SAC/ARS baseline stack |
| LayerNorm | `ln0` none · `lnc` critic · `lnac` actor+critic · `lnci` critic LayerNorm plus `critic_init_layernorm=True`, SAC family (the only flag separating ARS arm `ars_ln` from the LN-only `lnc` of `mtsac_cb`) |
| value clip | `vc1` on · `vc0` off · `vcx` not applicable |
| architecture | `400x3` = `[400,400,400]` both heads; `400x4`, `1024x4`, `400x3c32x3` = width-32 critic |
| variant | `gsplit` = the per-side solve this code implements; `gjoint` = one combiner over the concatenated actor+critic parameter vector; `lora16` = LoRA rank 16 |

> **`paw` (Welford) and `pae` (EMA) are two different PopArt estimators.** The
> headline results use Welford. Absolute numbers are not comparable across the two.

> **The three `_gjoint` groups are not reproducible from this release.** Joint-scope
> aggregation over the concatenated parameter vector was removed before this code was
> cut; `GradientConfig` has no scope option and `mtrl/gradients/scope.py` implements
> only the per-side solve. Their `_gsplit` twins are reproducible and are the
> configuration the paper uses.


## Paper experiments

| group | config | overrides | steps |
|---|---|---|---|
| `mt10v2_ppo_ad_cd_nnone_ln0_vc1_400x3` | `configs/ppo_mt10_vanilla_rfv2.yaml` | — | 20M |
| `mt10v2_ppo_ad_cd_nnone_lnc_vc1_400x3` | `configs/ppo_mt10_default.yaml` | `--ppo.no-use-popart` | 20M |
| `mt10v2_ppo_ad_cd_npaw_lnac_vc1_400x3` | `configs/ppo_mt10_default.yaml` | `--network.ln-actor` | 20M |
| `mt10v2_ppo_ad_cd_npaw_lnc_vc1_400x3` | `configs/ppo_mt10_default.yaml` | — | 20M |
| `mt10v2_ppo_ad_cfg1_nnone_lnc_vc1_400x3` | `configs/ppo_mt10_fairgrad.yaml` | `--ppo.no-use-popart --gradient.actor-method default` | 20M |
| `mt10v2_ppo_ad_cfg1_npaw_lnc_vc1_400x3` | `configs/ppo_mt10_fairgrad.yaml` | `--gradient.actor-method default` | 20M |
| `mt10v2_ppo_apc_cfg1_npaw_lnc_vc1_400x3` | `configs/ppo_mt10_toppo_lnc_rfv2.yaml` | — | 20M |
| `mt50v1_ppo_ad_cd_nhg1000rew_lnc_vc0_400x3` | `configs/ppo_mt50_hlgaussrewnorm_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_ad_cd_nhg101_lnc_vc0_400x3` | `configs/ppo_mt50_hlgauss_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_ad_cd_nhg101rew_lnc_vc0_400x3` | `configs/ppo_mt50_hlgaussrewnorm101_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_ad_cd_nhg2400paw_lnc_vc0_400x3` | `configs/ppo_mt50_hlgausspopart_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_ad_cd_nnone_ln0_vc1_400x3` | `configs/ppo_mt50_vanilla_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_ad_cd_nnone_lnc_vc0_400x3` | `configs/ppo_mt50_novclip_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_ad_cd_nnone_lnc_vc1_400x3` | `configs/ppo_mt50_novclip_lnc_rfv1.yaml` | `--ppo.value-clip` | 100M |
| `mt50v1_ppo_ad_cd_npaw_lnc_vc0_400x3` | `configs/ppo_mt50_popartnovc_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_ad_cd_npaw_lnc_vc1_400x3` | `configs/ppo_mt50_popartnovc_lnc_rfv1.yaml` | `--ppo.value-clip` | 100M |
| `mt50v1_ppo_ad_cd_nret_lnc_vc0_400x3` | `configs/ppo_mt50_retscalenovc_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_ad_cd_nret_lnc_vc1_400x3` | `configs/ppo_mt50_retscale_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_ad_cd_nrettr_lnc_vc1_400x3` | `configs/ppo_mt50_retscaletrunc_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_ad_cd_nrew_lnc_vc0_400x3` | `configs/ppo_mt50_rewnormnovc_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_ad_cd_nrew_lnc_vc1_400x3` | `configs/ppo_mt50_rewnorm_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_ad_cfg1_nnone_lnc_vc1_400x3` | `configs/ppo_mt50_ad_cfg_popart_lnc_rfv1.yaml` | `--ppo.no-use-popart` | 100M |
| `mt50v1_ppo_ad_cfg1_npaw_lnc_vc1_400x3` | `configs/ppo_mt50_ad_cfg_popart_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_apc_cfg1_nhg1000rew_lnc_vc0_400x3` | `configs/ppo_mt50_fullstack_hlgaussrewnorm_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_apc_cfg1_nhg2400paw_lnc_vc0_400x3` | `configs/ppo_mt50_fullstack_hlgausspopart_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_apc_cfg1_npaw_lnc_vc1_400x3` | `configs/ppo_mt50_toppo_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_apc_cfg1_nrew_lnc_vc1_400x3` | `configs/ppo_mt50_fullstack_rewnorm_lnc_rfv1.yaml` | — | 100M |
| `mt50v2_ppo_aca0p5_cca0p5_npaw_lnc_vc1_400x3` | `configs/ppo_mt50_cagrad.yaml` | — | 100M |
| `mt50v2_ppo_ad_cd_nhg101_lnc_vc0_400x3` | `configs/ppo_mt50_hlgauss_lnc_rfv2.yaml` | — | 100M |
| `mt50v2_ppo_ad_cd_nhg600rew_lnc_vc0_400x3` | `configs/ppo_mt50_hlgaussrewnorm_lnc_rfv2.yaml` | — | 100M |
| `mt50v2_ppo_ad_cd_nnone_ln0_vc1_400x3` | `configs/ppo_mt50_vanilla_rfv2.yaml` | — | 100M |
| `mt50v2_ppo_ad_cd_nnone_lnc_vc1_400x3` | `configs/ppo_mt50_default.yaml` | `--ppo.no-use-popart` | 100M |
| `mt50v2_ppo_ad_cd_npaw_lnc_vc1_400x3` | `configs/ppo_mt50_default.yaml` | — | 100M |
| `mt50v2_ppo_ad_cd_nret_lnc_vc1_400x3` | `configs/ppo_mt50_retscale_lnc_rfv2.yaml` | — | 100M |
| `mt50v2_ppo_ad_cd_nrew_lnc_vc1_400x3` | `configs/ppo_mt50_rewnorm_lnc_rfv2.yaml` | — | 100M |
| `mt50v2_ppo_ad_cfg1_nnone_ln0_vc1_400x3` | `configs/ppo_mt50_ad_cfg_popart_lnc_rfv2.yaml` | `--network.no-ln-critic --ppo.no-use-popart` | 100M |
| `mt50v2_ppo_ad_cfg1_nnone_lnc_vc1_400x3` | `configs/ppo_mt50_ad_cfg_popart_lnc_rfv2.yaml` | `--ppo.no-use-popart` | 100M |
| `mt50v2_ppo_ad_cfg1_npaw_lnc_vc1_400x3` | `configs/ppo_mt50_ad_cfg_popart_lnc_rfv2.yaml` | — | 100M |
| `mt50v2_ppo_afg2_cfg1_npaw_lnc_vc1_400x3` | `configs/ppo_mt50_fairgrad.yaml` | — | 100M |
| `mt50v2_ppo_apc_cfg1_npaw_lnc_vc1_400x3` | `configs/ppo_mt50_toppo_lnc_rfv2.yaml` | — | 100M |
| `mt50v2_ppo_apc_cpc_npaw_lnc_vc1_400x3` | `configs/ppo_mt50_pcgrad.yaml` | — | 100M |
| `mt50v2_sac_ad_cd_nars_ln0_vcx_400x4` | — | ARS arm `ars`, see [arm table](../experiments/sac_baselines_ars/README.md#arms) | 100M |
| `mt50v2_sac_ad_cd_nars_lnci_vcx_1024x4` | — | ARS arm `ars_ln1024`, see [arm table](../experiments/sac_baselines_ars/README.md#arms) | 100M |
| `mt50v2_sac_ad_cd_nars_lnci_vcx_400x4` | — | ARS arm `ars_ln`, see [arm table](../experiments/sac_baselines_ars/README.md#arms) | 100M |
| `mt50v2_sac_ad_cd_nars_lnci_vcx_400x4_lora16` | — | ARS arm `ars_lora` (runs `train_ars_lora`), see [arm table](../experiments/sac_baselines_ars/README.md#arms) | 100M |
| `mt50v2_sac_ad_cd_nnone_ln0_vcx_400x4` | — | ARS arm `mtsac`, see [arm table](../experiments/sac_baselines_ars/README.md#arms) | 100M |
| `mt50v2_sac_ad_cfg1_npae_lnc_vcx_400x4` | — | ARS arm `mtsac_cb`, see [arm table](../experiments/sac_baselines_ars/README.md#arms) | 100M |
| `mt50v2_sac_apc_cfg1_npae_lnc_vcx_400x4` | — | ARS arm `mtsac_fullstack`, see [arm table](../experiments/sac_baselines_ars/README.md#arms) | 100M |

The 7 `mt50v2_sac_*` rows run from `experiments/sac_baselines_ars/` instead:
`mkdir -p logs && sbatch slurm/site_a100/mt50_run.sbatch <arm> <seed> wandb_group=<group>` (or
`slurm/site_3090_mt50_run.sbatch` with the same arguments). Both templates write their Slurm log
to `logs/`, which is not in a fresh clone and which Slurm does not create; the launchers
`slurm/site_a100/launch_mt50_grid.sh` and `scripts/launch_mt50.sh` create it themselves. That
directory's README also has a one-seed command without Slurm. Their steps are env steps: 2M loop
iterations × 50 task envs.

`mt50v2_ppo_ad_cd_nnone_ln0_vc1_400x3` mixes two evaluation intervals on W&B: seeds 0–4 and 9
logged `logging.test_interval=1` (evaluated every epoch, 101 checkpoints) and seeds 5–8 logged 2.
No single config can match all ten; `configs/ppo_mt50_vanilla_rfv2.yaml` uses the paper's 2-epoch
grid. Epoch 100 is evaluated under both settings, so final-checkpoint numbers are unaffected, but
the per-checkpoint curves of those six seeds have twice as many points.

### Actor-aggregation sweep (EMA PopArt)

Figure 5 and Appendix E.6 (Table E.7). Every cohort uses EMA PopArt (`pae`), LN-c and FairGrad
(α = 1) on the critic, and varies only the actor aggregator. The paper reports this sweep
separately from the Welford-PopArt headline.

| group | config | overrides | steps |
|---|---|---|---|
| `mt50v1_ppo_aca0p5_cfg1_npae_lnc_vc1_400x3` | `configs/post_submission/ppo_mt50_acag_cfg_popart_ema_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_ad_cfg1_npae_lnc_vc1_400x3` | `configs/post_submission/ppo_mt50_ad_cfg_popart_ema_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_afg0p5_cfg1_npae_lnc_vc1_400x3` | `configs/post_submission/ppo_mt50_afg0.5_cfg_popart_ema_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_afg10_cfg1_npae_lnc_vc1_400x3` | `configs/post_submission/ppo_mt50_afg10_cfg_popart_ema_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_afg1_cfg1_npae_lnc_vc1_400x3` | `configs/post_submission/ppo_mt50_afg1_cfg_popart_ema_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_afg2_cfg1_npae_lnc_vc1_400x3` | `configs/post_submission/ppo_mt50_afg2_cfg_popart_ema_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_afg5_cfg1_npae_lnc_vc1_400x3` | `configs/post_submission/ppo_mt50_afg5_cfg_popart_ema_lnc_rfv1.yaml` | — | 100M |
| `mt50v1_ppo_apc_cfg1_npae_lnc_vc1_400x3` | `configs/post_submission/ppo_mt50_apcg_cfg_popart_ema_lnc_rfv1.yaml` | — | 100M |
| `mt50v2_ppo_aca0p5_cfg1_npae_lnc_vc1_400x3` | `configs/post_submission/ppo_mt50_acag_cfg_popart_ema_lnc_rfv2.yaml` | — | 100M |
| `mt50v2_ppo_ad_cfg1_npae_lnc_vc1_400x3` | `configs/post_submission/ppo_mt50_ad_cfg_popart_ema_lnc_rfv2.yaml` | — | 100M |
| `mt50v2_ppo_afg0p5_cfg1_npae_lnc_vc1_400x3` | `configs/post_submission/ppo_mt50_afg0.5_cfg_popart_ema_lnc_rfv2.yaml` | — | 100M |
| `mt50v2_ppo_afg10_cfg1_npae_lnc_vc1_400x3` | `configs/post_submission/ppo_mt50_afg10_cfg_popart_ema_lnc_rfv2.yaml` | — | 100M |
| `mt50v2_ppo_afg1_cfg1_npae_lnc_vc1_400x3` | `configs/post_submission/ppo_mt50_afg1_cfg_popart_ema_lnc_rfv2.yaml` | — | 100M |
| `mt50v2_ppo_afg2_cfg1_npae_lnc_vc1_400x3` | `configs/post_submission/ppo_mt50_afg2_cfg_popart_ema_lnc_rfv2.yaml` | — | 100M |
| `mt50v2_ppo_afg5_cfg1_npae_lnc_vc1_400x3` | `configs/post_submission/ppo_mt50_afg5_cfg_popart_ema_lnc_rfv2.yaml` | — | 100M |
| `mt50v2_ppo_apc_cfg1_npae_lnc_vc1_400x3` | `configs/post_submission/ppo_mt50_apcg_cfg_popart_ema_lnc_rfv2.yaml` | — | 100M |

## AMP humanoid stress test

Section 5.3 and Appendix E.5 report an AMP humanoid walk-steering stress test built on MimicKit.
Its code, configs and per-seed evaluation records are in
[`experiments/amp_humanoid/`](../experiments/amp_humanoid/) (see its `README.md`); `setup.sh`
there assembles MimicKit `2ed1e6c` plus our patch and files. All 250 runs (25 configurations ×
10 seeds) are public and described in
[report 8](https://wandb.ai/liyp1220-uci/TOPPO/reports/--VmlldzoxNzg5MjgxOQ==).

The AMP rows use MimicKit's own seeds, **`--rand_seed 260511470` .. `260511479`** (not
`--seed 0..9`), and run from the assembled tree with one of two templates, which
`python experiments/amp_humanoid/campaign.py train|eval <group> <seed>` prints:

```bash
python mimickit/run.py --mode train --num_envs 4096 \
  --engine_config data/engines/newton_engine.yaml \
  --env_config data/envs/amp_steering_humanoid_walk_env.yaml \
  --agent_config data/agents/<config> --out_dir output/<group>_s<seed>/ \
  --max_samples <steps> --rand_seed <seed> --devices cuda:0 --logger txt --visualize false

python mimickit/run.py --mode test --num_envs 4096 \
  --engine_config data/engines/newton_engine.yaml \
  --env_config data/envs/amp_steering_humanoid_walk_eval_env.yaml \
  --agent_config data/agents/<config> --model_file output/<group>_s<seed>/model.pt \
  --devices cuda:0 --visualize false --test_episodes 4096 --detailed_eval true \
  --style_samples 8192 --style_projection_dim 64 --style_seed 260511701 \
  --eval_seed 260511499 --eval_output results/eval/<group>_s<seed>.json
```

Neither template writes to W&B: training logs to a text file (`--logger txt`).

| group | config (`data/agents/`) | steps | paper |
|---|---|---|---|
| `ampwalk_ppo_ad_cd_nnone_ln0_k1_gc1` | `amp_task_humanoid_long_critic_clip1_agent.yaml` | 100M | Table E.6: PPO, Base |
| `ampwalk_ppo_ad_cd_npaw_lnc_k2_gc1` | `amp_task_humanoid_component_welford_ln_clip1_agent.yaml` | 100M | Table E.6: PPO, Two-component critic + LN + PopArt |
| `ampwalk_ppo_ad_cfg1_npaw_lnc_k2_gc0` | `amp_task_humanoid_toppo_cc100_noclip_agent.yaml` | 100M | Table E.6: PPO, Above + FG-c, no critic grad clip |
| `ampwalk_ppo_apc_cfg1_npaw_lnc_k2_gc0` | `amp_task_humanoid_fulltoppo_cc100_agent.yaml` | 100M | Table E.6: PPO, Full stack (also PCGrad-a); the 76% result |
| `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0` | `amp_sac_walk_agent.yaml` | 100M | Table E.6: SAC, Base |
| `ampwalk_sac_apc_cfg1_npaw_lnc_k2_gc0` | `amp_sac_walk_fullstack_agent.yaml` | 100M | Table E.6: SAC, Full stack |
| `ampwalk_ppo_ad_cd_nnone_ln0_k1_gc1_reweight` | `amp_task_humanoid_ppo_reweight_agent.yaml` | 100M | campaign |
| `ampwalk_ppo_ad_cd_nrew_lnc_k2_gc1` | `amp_task_humanoid_w5_rewnorm_agent.yaml` | 100M | campaign |
| `ampwalk_ppo_ad_cd_nret_lnc_k2_gc1` | `amp_task_humanoid_w5_retscale_agent.yaml` | 100M | campaign |
| `ampwalk_ppo_ad_cd_nhg_lnc_k2_gc1` | `amp_task_humanoid_w5_hlgauss_agent.yaml` | 100M | campaign |
| `ampwalk_ppo_ad_cd_nhg_lnc_k2_gc1_lr1e3` | `amp_task_humanoid_w5_hlgauss_lr1e3_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cd_nnone_lnc_k1_gc0` | `amp_sac_walk_lnonly_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cd_npaw_ln0_k1_gc0` | `amp_sac_walk_popartonly_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cd_npaw_lnc_k1_gc0` | `amp_sac_walk_k1cd_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cd_npaw_lnc_k2_gc0` | `amp_sac_walk_cd_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cfg1_npaw_lnc_k2_gc0` | `amp_sac_walk_cb_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_reweight` | `amp_sac_walk_reweight_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cfg1_npaw_lnc_k2_gc0_reweight` | `amp_sac_walk_cbreweight_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_ars` | `amp_sac_walk_ars_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_arsreset` | `amp_sac_walk_arsreset_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_entmatch` | `amp_sac_walk_entmatch_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_utd512` | `amp_sac_walk_utd512_agent.yaml` | 25M | campaign |
| `ampwalk_sac_ad_cfg1_nrew_lnc_k2_gc0` | `amp_sac_walk_w5_rewnorm_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cfg1_nret_lnc_k2_gc0` | `amp_sac_walk_w5_retscale_agent.yaml` | 100M | campaign |
| `ampwalk_sac_ad_cfg1_nhg_lnc_k2_gc0` | `amp_sac_walk_w5_hlgauss_agent.yaml` | 100M | campaign |

"campaign" rows belong to the broader 25-configuration campaign the appendix mentions and are not
in its table; `experiments/amp_humanoid/results/campaign_summary.md` summarizes all 25.

Group-name grammar:
`ampwalk_{ppo,sac}_a{d,pc}_c{d,fg1}_n{none,paw,rew,ret,hg}_{ln0,lnc}_k{1,2}_gc{0,1}[_reweight|_ars|_arsreset|_entmatch|_utd512|_lr1e3]`.
`a`/`c`: actor/critic combiner (`d` default, `pc` PCGrad, `fg1` FairGrad α = 1); `n`: critic
normalizer (`paw` Welford PopArt, `rew` reward normalization, `ret` return-based scaling, `hg`
HL-Gauss); `ln0`/`lnc`: critic LayerNorm off/on; `k`: number of critic heads, i.e. reward
components; `gc1`/`gc0`: critic gradient clipping at 1.0 / off. Suffixes: `_reweight` style weight
0.5 -> 1.105, `_ars` / `_arsreset` ARS reward scaling without / with learner resets, `_entmatch`
SAC target entropy matched to PPO's fixed-std policy, `_utd512` 8× updates per sample at 25M
samples, `_lr1e3` critic learning rate ×10. The style FID is `eval/style_projected_fid` in each
run's summary.

W&B caveats (details in `experiments/amp_humanoid/README.md`):

- For seeds 260511473–260511475 of `ampwalk_ppo_ad_cd_npaw_lnc_k2_gc1`, W&B holds a re-evaluation
  of the same checkpoints on an RTX 3090; Table E.6 uses the H200 evaluations shipped in
  `experiments/amp_humanoid/results/eval/`. Per-seed FID drift +0.46%, +0.16%, +0.01%; the group
  reads 38,579.7 ± 8,715.3 on W&B against 38,553.5 ± 8,705.7 in the paper.
  `experiments/amp_humanoid/analysis/eval_drift.py` reproduces both. `toppo/source_arm` on that
  group reads `3way_criticbal` on all 10 runs; the records used are listed in
  `experiments/amp_humanoid/provenance/eval_files.tsv`.
- 25 runs carry final evaluation summaries but no training history, three of them seeds of
  `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0`; `experiments/amp_humanoid/results/train_samples.tsv` marks
  them `NA`.
- The `requirements.txt` attached to the AMP runs on W&B describes the re-logging environment, not
  training; use `experiments/amp_humanoid/requirements.txt`.

The appendix's short H200 timing measurements are not in the W&B project; they ship as
`experiments/amp_humanoid/results/timing/timing.tsv` (see
[Not in the public W&B project](#not-in-the-public-wb-project)).

The AMP rows are checked by `experiments/amp_humanoid/analysis/validate_campaign.py` (coverage,
protocol, config provenance and training budgets of the 250 records) and
`experiments/amp_humanoid/overlay/tests/test_campaign_configs.py` (the documented config
differences), not by `analysis/check_runs_index.py`.

## Not in the public W&B project

**Compute timing (Appendix E.4, Table E.5).** One 10M-step measurement per configuration on an
RTX 3090 with 16 CPU cores and evaluation disabled; PPO logging was off and SAC logging offline.
To time a PPO configuration the same way, take its row above, give it a distinct `--exp-name`
(e.g. `<group>_timing`, so that it does not auto-resume from a finished seed of that row), and add

```bash
--training.epoch 1 --training.step-per-epoch 10000000 --logging.test-interval 0 --logging.no-use-wandb
```

then hours per 100M steps = wall-clock seconds / 360. The table's PPO rows are the MT50 V2 ladder
(vanilla, +LN-c, +PopArt, +FG-c, TOPPO) and three rows that apply FairGrad, CAGrad or PCGrad to
both actor and critic with LN-c and PopArt. The SAC rows ran the matching ARS launcher arm with
`MAX_STEPS=200000` (10M env steps) and `eval_interval=999999999` appended.

**Gradient-snapshot dumps (Appendix D.2).** The mechanism figures use four single-seed MT10 dump
runs from an earlier internal version of this code; they are not in the public W&B project. Three
were configured as 8M-step runs (`--training.epoch 40`); the fourth, the no-LN default-aggregation
cell, was configured for the full 20M and ended at 9.3M. The two default-aggregation cells use
seed 4 and the two FairGrad cells seed 0. The instrumentation is described in
[`experiments/metaworld_ppo/README.md`](../experiments/metaworld_ppo/README.md#per-task-gradient-dumps).
With the released configs, the four cells are

- default aggregation + PopArt, no LN (seed 4): `configs/ppo_mt10_popart.yaml`
- default aggregation + PopArt + LN-c (seed 4): `configs/ppo_mt10_default.yaml`
- FairGrad (α = 2 actor, α = 1 critic) + PopArt, no LN (seed 0): `configs/ppo_mt10_fairgrad.yaml --network.no-ln-critic`
- FairGrad (α = 2 actor, α = 1 critic) + PopArt + LN-c (seed 0): `configs/ppo_mt10_fairgrad.yaml`

each with `--training.epoch 40 --logging.dump-grads-dir auto --logging.dump-grads-every-collects 5`,
the seed shown and a distinct `--exp-name`. `--training.epoch 40` stops at 8M steps, which covers
every snapshot the figures use (0.5M to 8M). A re-run will not reproduce the original snapshots
bit for bit.

**Real-Gram solver benchmark (Appendix C.5).** One seed-0 run of two epochs (2M steps) of the
`mt50v1_ppo_ad_cfg1_npaw_lnc_vc1_400x3` configuration, with evaluation and W&B logging off, that
also solved each of its 1,984 critic Grams (K = 50) with a scipy reference and timed both solvers.
Its raw output ships as
[`experiments/metaworld_ppo/benchmarks/solver/solverbench_rfv1.csv`](../experiments/metaworld_ppo/benchmarks/solver/solverbench_rfv1.csv);
[`realgram_results.md`](../experiments/metaworld_ppo/benchmarks/solver/realgram_results.md) next
to it gives the protocol, the summary and the command that regenerates a capture with the
`MTRL_SOLVER_BENCH` hook.

**AMP timing (Appendix E.5).** The end-to-end overheads quoted for the AMP full stacks (+4.2% for
PPO, +49.5% for SAC) are single short measurements of about 8M samples each on an H200 NVL with
8 CPU cores. They are not on W&B. The eight measurements (four PPO and four SAC configurations,
seed 260511470, training command only) are in
[`experiments/amp_humanoid/results/timing/timing.tsv`](../experiments/amp_humanoid/results/timing/timing.tsv),
and `bash experiments/amp_humanoid/scripts/time_arm.sh <assembled tree> <group>` repeats one of
them; compare configurations only on the same GPU type and CPU count.

## History gaps

- 106 of the 893 Meta-World runs logged fewer history rows than the rest of their group, most by
  one or two rows.
- The largest evaluation gaps are in two V1 reward-normalization seeds:
  `mt50v1_ppo_ad_cd_nrew_lnc_vc1_400x3` seed 6 has 32 of its 51 evaluation points (none between
  48M and 88M), and `mt50v1_ppo_apc_cfg1_nrew_lnc_vc1_400x3` seed 6 has 42 (none between 34M and
  54M). Two headline seeds miss a few points: `mt50v2_ppo_apc_cfg1_npaw_lnc_vc1_400x3` seeds 3
  and 4 have 48 and 49. All four still have their final 100M evaluation.
- Every Meta-World run except the two truncated seeds above reached its full step budget.
- 25 of the 250 AMP runs carry final evaluation summaries only (report 8); they are listed in
  `experiments/amp_humanoid/README.md` and marked `NA` in its `results/train_samples.tsv`.

## Additional experiments (not in the paper)

| group | config | overrides | steps |
|---|---|---|---|
| `mt10v2_ppo_aca0p9_cca0p9_nnone_ln0_vc1_400x3_gjoint` | — | not reproducible from this release (see note above) | 20M |
| `mt10v2_ppo_aca0p9_cca0p9_nnone_ln0_vc1_400x3_gsplit` | `configs/ppo_mt10_cagrad.yaml` | `--network.no-ln-critic --ppo.no-use-popart` | 20M |
| `mt10v2_ppo_aca0p9_cca0p9_npaw_lnc_vc1_400x3` | `configs/ppo_mt10_cagrad.yaml` | — | 20M |
| `mt10v2_ppo_aca0p9_cd_nnone_ln0_vc1_400x3` | `configs/ppo_mt10_cagrad.yaml` | `--network.no-ln-critic --ppo.no-use-popart --gradient.critic-method default` | 20M |
| `mt10v2_ppo_ad_cca0p9_nnone_ln0_vc1_400x3` | `configs/ppo_mt10_cagrad.yaml` | `--network.no-ln-critic --ppo.no-use-popart --gradient.actor-method default` | 20M |
| `mt10v2_ppo_ad_cd_npaw_ln0_vc1_400x3` | `configs/ppo_mt10_popart.yaml` | — | 20M |
| `mt10v2_ppo_ad_cfg1_nnone_ln0_vc1_400x3` | `configs/ppo_mt10_fairgrad.yaml` | `--network.no-ln-critic --ppo.no-use-popart --gradient.actor-method default` | 20M |
| `mt10v2_ppo_ad_cfg1_npaw_ln0_vc1_400x3` | `configs/ppo_mt10_fairgrad.yaml` | `--network.no-ln-critic --gradient.actor-method default` | 20M |
| `mt10v2_ppo_ad_cfg2_nnone_ln0_vc1_400x3` | `configs/ppo_mt10_vanilla_rfv2.yaml` | `--gradient.critic-method fairgrad --gradient.fairgrad-critic-alpha 2.0` | 20M |
| `mt10v2_ppo_ad_cpc_nnone_ln0_vc1_400x3` | `configs/ppo_mt10_vanilla_rfv2.yaml` | `--gradient.critic-method pcgrad` | 20M |
| `mt10v2_ppo_afg2_cd_nnone_ln0_vc1_400x3` | `configs/ppo_mt10_fairgrad.yaml` | `--network.no-ln-critic --ppo.no-use-popart --gradient.critic-method default` | 20M |
| `mt10v2_ppo_afg2_cfg1_nnone_ln0_vc1_400x3` | `configs/ppo_mt10_fairgrad.yaml` | `--network.no-ln-critic --ppo.no-use-popart` | 20M |
| `mt10v2_ppo_afg2_cfg1_npaw_lnc_vc1_400x3` | `configs/ppo_mt10_fairgrad.yaml` | — | 20M |
| `mt10v2_ppo_afg2_cfg2_nnone_ln0_vc1_400x3_gjoint` | — | not reproducible from this release (see note above) | 20M |
| `mt10v2_ppo_afg2_cfg2_nnone_ln0_vc1_400x3_gsplit` | `configs/ppo_mt10_fairgrad.yaml` | `--network.no-ln-critic --ppo.no-use-popart --gradient.fairgrad-critic-alpha 2.0` | 20M |
| `mt10v2_ppo_apc_cd_nnone_ln0_vc1_400x3` | `configs/ppo_mt10_vanilla_rfv2.yaml` | `--gradient.actor-method pcgrad` | 20M |
| `mt10v2_ppo_apc_cpc_nnone_ln0_vc1_400x3_gjoint` | — | not reproducible from this release (see note above) | 20M |
| `mt10v2_ppo_apc_cpc_nnone_ln0_vc1_400x3_gsplit` | `configs/ppo_mt10_pcgrad.yaml` | `--network.no-ln-critic --ppo.no-use-popart` | 20M |
| `mt10v2_ppo_apc_cpc_npaw_lnc_vc1_400x3` | `configs/ppo_mt10_pcgrad.yaml` | — | 20M |
| `mt50v2_ppo_ad_cd_npaw_ln0_vc1_400x3` | `configs/ppo_mt50_popart.yaml` | — | 100M |
| `mt50v2_ppo_ad_cfg1_npae_lnc_vc1_400x3c32x3` | `configs/post_submission/ppo_mt50_ad_cfg_popart_ema_lnc_rfv2.yaml` | `--network.critic 32 32 32` | 100M |
| `mt50v2_ppo_ad_cfg1_npaw_ln0_vc1_400x3` | `configs/ppo_mt50_ad_cfg_popart_lnc_rfv2.yaml` | `--network.no-ln-critic` | 100M |
| `mt50v2_ppo_afg2_cfg1_nnone_ln0_vc1_400x3` | `configs/ppo_mt50_fairgrad.yaml` | `--network.no-ln-critic --ppo.no-use-popart` | 100M |
| `mt50v2_ppo_afg2_cfg1_nnone_ln0_vc1_400x4` | `configs/ppo_mt50_fairgrad.yaml` | `--network.no-ln-critic --ppo.no-use-popart --network.actor 400 400 400 400 --network.critic 400 400 400 400` | 100M |
| `mt50v2_ppo_afg2_cfg1_npaw_lnc_vc1_1024x4` | `configs/ppo_mt50_fairgrad.yaml` | `--network.actor 1024 1024 1024 1024 --network.critic 1024 1024 1024 1024` | 100M |
| `mt50v2_ppo_afg2_cfg1_npaw_lnc_vc1_400x4` | `configs/ppo_mt50_fairgrad.yaml` | `--network.actor 400 400 400 400 --network.critic 400 400 400 400` | 100M |

## Canonical parameters on the runs themselves

The runs span three generations of the config schema (an early one with a single
`gradient.method` plus a `gradient.scope`, a later per-side `actor_method` /
`critic_method`, and the separate SAC/ARS schema), so no single config field is
present on every run. Every run therefore also carries a uniform, schema-independent
`toppo/*` parameter block that can be grouped and filtered on directly:

| key | values |
|---|---|
| `toppo/benchmark` | `MT10`, `MT50` |
| `toppo/reward_version` | `v1`, `v2` |
| `toppo/algorithm` | `ppo`, `sac` |
| `toppo/actor_method`, `toppo/critic_method` | `default`, `pcgrad`, `cagrad`, `fairgrad` |
| `toppo/actor_alpha`, `toppo/critic_alpha` | FairGrad α or CAGrad c; null when the side has no combiner |
| `toppo/normalizer` | `none`, `popart`, `reward_normalization`, `return_scaling`, `return_scaling_truncated`, `hlgauss`, `hlgauss+reward_normalization`, `hlgauss+popart`, `ars` |
| `toppo/popart_method` | `welford`, `ema`, null |
| `toppo/hlgauss_bins` | bin count, null when the critic loss is MSE |
| `toppo/critic_loss` | `mse`, `hlgauss`; null on SAC |
| `toppo/return_scaling` | true for the `return_scaling*` normalizers, false otherwise; null on SAC |
| `toppo/reward_normalization` | `baselines_return` when reward normalization is on (`reward_normalization`, `hlgauss+reward_normalization`), else null |
| `toppo/layernorm` | `none`, `critic`, `actor_critic`, `critic_init` |
| `toppo/value_clip` | true / false; null on SAC, where it does not apply |
| `toppo/actor_arch`, `toppo/critic_arch` | e.g. `400x3`, `1024x4`, `32x3` |
| `toppo/variant` | `gjoint`, `gsplit`, `lora16`, null |
| `toppo/env_steps`, `toppo/seed`, `toppo/group` | budget, seed, canonical group name |

`toppo/critic_loss`, `toppo/return_scaling` and `toppo/reward_normalization` are finer-grained
views of `toppo/normalizer`. The values above are those of the Meta-World runs. The AMP runs carry
the same keys with AMP-specific values (e.g. `toppo/benchmark` = `AMP-humanoid-steering-walk`,
`toppo/reward_version` null, `toppo/actor_arch` = `fc_2layers_1024units`,
`toppo/reward_normalization` = `amp_w5_rewnorm` on its reward-normalization runs, and variants
such as `reweight`, `utd512` or `lr1e3`), plus AMP-only keys such as `toppo/arm` and
`toppo/agent_config`; report 8 documents them. `experiments/amp_humanoid/campaign.py` maps
`toppo/arm` to the arm names its scripts use, and `toppo/agent_config_sha256` equals the recorded
hash in its `provenance/config_hashes.tsv`.

These are derived from the group name and were cross-checked field by field against
whatever each run's own config already stated; every disagreement was resolved before
they were written. They contain no paths, hostnames or account identifiers.

## Checking this table

The checker ships with the code, so this table's correctness is auditable rather than
asserted:

```bash
cd experiments/metaworld_ppo
pip install pyyaml tyro                 # the checker needs nothing else
python analysis/check_runs_index.py     # exits non-zero on any mismatch
```

For each row it loads the listed config with the listed overrides through the real CLI parser
and compares the effective settings against what the group name encodes, field by field:
benchmark, reward version, combiner and alpha on each side, normalizer, PopArt estimator,
`hlgauss_num_bins`, `return_scaling_truncation_as_terminal`, LayerNorm placement, value clip,
and the per-head layer widths. It also checks that the **steps** column matches the benchmark
budget, that the row does not select a linear baseline or per-task heads (no published PPO row
does), and that no group appears twice. Against the paper's hyperparameter table it compares,
field by field, the training schedule (`epoch`, `step_per_epoch`, `step_per_collect`,
`repeat_per_collect`, `batch_size`, `num_tasks` / `num_envs`), the evaluation interval
(`test_interval` 2, i.e. 51 checkpoints) and the shared optimizer and PPO numerics
(`minibatch_sampling`, `learning_rate`, `gamma`, `max_grad_norm`, `gae_lambda`, `clip_range`,
`vf_coef`, `ent_coef`, advantage normalization, recompute advantage).

```bash
pip install wandb
python analysis/check_runs_index.py --wandb   # read-only
```

`--wandb` also reads each row's runs from W&B and compares every field of this release's config
schema that the runs recorded against the effective config, except run identity, output
locations and checkpoint cadence (`seed`, `exp_name`, `project_name`, `config`, `device`,
`logging.log_dir`, `logging.wandb_*`, `logging.save_interval`, `logging.save_best`). Older runs
lack some schema fields (several early groups never recorded `gradient.actor_method` /
`gradient.critic_method` or `network.ln_critic`, so for them the arm rests on the group-name check
above), and legacy keys outside the schema (`gradient.method`, `gradient.scope`, the `toppo/*`
block) are not compared. It reads liyp1220-uci/TOPPO unless `--entity`, `--project` or
`$WANDB_PROJECT` is given, by the same rule as `metrics.py` and `export_wandb.py` (see
[`env_vars.md`](env_vars.md)); `$WANDB_ENTITY` alone does not move it. To check your own re-runs,
pass `--entity <you> --project toppo-repro`.

As of this commit: **80 rows checked, 0 mismatched**; with `--wandb`, **80 rows checked against
793 W&B runs, 0 mismatched** (two NOTEs: the 3-seed `nhg101rew` group and the mixed evaluation
interval of `mt50v2_ppo_ad_cd_nnone_ln0_vc1_400x3`, both described above). The remaining 10 rows
are skipped because they have no config to check: the 7 SAC rows are configured by CLI `k=v`
pairs in the ARS launchers rather than by YAML, and the 3 `_gjoint` rows are not reproducible
from this release (see above).
