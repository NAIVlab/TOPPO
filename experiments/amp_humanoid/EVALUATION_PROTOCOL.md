# Evaluation protocol

Every number reported for the AMP experiments comes from one evaluation record per (group, seed):
the final checkpoint of a 100M-sample training run (25M for the `_utd512` group), evaluated with
`mimickit/run.py --mode test --detailed_eval true`. The 250 records are in `results/eval/`; how the
canonical record was chosen for the 33 checkpoints that were evaluated more than once is under
"Hardware spread and drift" below.

## Fixed settings (identical in all 250 records)

| Setting | Value |
|---|---|
| Evaluation env | `data/envs/amp_steering_humanoid_walk_eval_env.yaml`, sha256 `a2bf83e1…` at run time; the shipped file adds a change-notice comment, so its byte hash differs but it parses to the same configuration (`provenance/config_hashes.tsv`) |
| Engine | `data/engines/newton_engine.yaml` (unmodified upstream), sha256 `85c4897e…`: Newton 1.3.0, MuJoCo-Warp solver, 240 Hz simulation, 30 Hz control |
| Environments / episodes | 4,096 parallel environments, 4,096 episodes (one per environment) |
| Episode | 20 s (600 control steps); a fall ends it early (upstream's early termination on ground contact, plus root height below 0.5 m or uprightness below 0.5, added by the patch) |
| Evaluation seed | `--eval_seed 260511499`, set after the agent and checkpoint are loaded |
| Style samples | 8,192 policy and 8,192 demonstration AMP observations |
| Style projection | 64 dimensions, `--style_seed 260511701` |

The evaluation env differs from the training env in one respect: `tar_change_time_min` and
`tar_change_time_max` are 100 s instead of 5 s and 10 s. The commanded direction and speed (uniform
in [0.7, 1.3] m/s) are therefore drawn once per episode and stay fixed for its 20 s, which makes
the speed metrics well defined. `analysis/validate_campaign.py` checks every setting above against
every record, and `overlay/tests/test_campaign_configs.py` checks the env difference.

## Metrics

**Task return** (`task_return_mean`): the undiscounted sum of the environment's steering reward
over an episode, averaged over the 4,096 episodes. It does not include the style reward.

**Style FID** (`style_projected_fid`), computed by `compute_style_distribution_metrics` in
`overlay/mimickit/learning/eval_util.py`:

1. During evaluation, a reservoir keeps a uniform random sample of 8,192 of the policy's AMP
   discriminator observations (10 stacked frames of character state per sample) from all active
   environment steps. The same number of demonstration observations is drawn from the reference
   clip.
2. Both sets are standardized with the demonstration mean and standard deviation (floor 1e-3).
3. Both are projected to 64 dimensions with one fixed Gaussian random matrix (seed 260511701,
   entries scaled by 1/sqrt(feature dimension)); the matrix is the same for every policy.
4. The metric is the **squared** Frechet distance between Gaussians fitted to the two projected
   sets, ||mu_p − mu_d||^2 + tr(S_p + S_d − 2 (S_p^1/2 S_d S_p^1/2)^1/2), in float64.

Lower is better. The paper's "76%" is a reduction of this squared distance; in distance units
(its square root) the reduction is about 51%. Because the features, the standardization and the
projection are fixed, every policy is scored in the same space; no policy is scored by its own
discriminator. The discriminator-based quantities in the records (`disc_return_*`,
`amp_return_*`, `disc_policy_probability_*`) are each run's own learned score and are diagnostics
only. `style_feature_mean_rms` and `style_feature_std_rms` are the root-mean-square gaps in
standardized feature means and standard deviations.

**Steering metrics** (`target_projected_speed_step_mean`, `target_speed_error_step_mean` and
others) are per-step averages over all active steps: the root velocity projected on the commanded
direction, and its absolute difference from the commanded speed.

`eval_util.py` is byte-identical to the file that produced the records (sha256
`84d2ff428a1cad79c1096c528cb6a741fe122ce92fd0fb779285eb9b95e3f875`).

## Statistics

The unit of analysis is the training seed. Every reported ± is the sample standard deviation over
the 10 seeds (ddof = 1), never the standard error. Comparisons between groups are paired by seed
(the groups share the same 10 seeds): the mean paired difference d with its 95% Student-t interval
(df = 9) and a two-sided paired t-test; where a relative change is given, d and its interval are
divided by the control group's mean. p > 0.05 means that no difference was detected. Bonferroni
corrections, where stated, multiply p by 25, the number of configurations.

The `*_std`, `*_sem` and `*_ci95` fields inside each record are statistics over the 4,096 episodes
of one evaluation. They describe evaluation noise for one checkpoint, are not seed-level
uncertainty, and are not used for any reported number.

## Hardware spread and drift

Evaluations ran on several GPU types (`analysis/validate_campaign.py` prints the counts: RTX 3090,
H200 NVL, RTX PRO 6000 Blackwell, A30, H100 and MIG slices). Simulation and the metric are not
bitwise deterministic across GPU types. `results/reevals/` holds 45 additional evaluations, with
the same protocol, of 33 canonical checkpoints, all from the four PPO table groups: every seed of
the base, FairGrad and full-stack groups, and seeds 260511473–260511475 of the PopArt group.
Across the 57 same-checkpoint pairs, style FID differs by at most 0.994% (median 0.26%) and task
return by at most 0.04% (`analysis/eval_drift.py`). This bound is measured only for those PPO
checkpoints: no SAC checkpoint, and no checkpoint of the other 21 groups, was evaluated more than
once. Because hosts differ, comparisons are made at the seed level only, never by pairing
individual episodes.

Each (group, seed) has exactly one canonical record: the record with `role` = `canonical` in
`provenance/eval_files.tsv`, which also lists the original file behind every shipped record. Only
the 33 checkpoints above have more than one record. For the four PPO table groups, the canonical
records are the H200 NVL evaluation passes that cover all 40 of their checkpoints on one device
type (original file prefixes `ppo10_`, and `ppoctl_` for the seven PopArt seeds that `ppo10_` does
not cover), so the paired PPO comparisons share a device type. The other evaluations of the same
checkpoints (RTX 3090, RTX PRO 6000, and three H200 NVL repeats) are the ones in
`results/reevals/`.

The choice of record does not change the PPO results (`analysis/eval_drift.py`). With the RTX 3090
evaluations in place of the canonical records, the full stack vs. base change in style FID is
−75.94% (p = 4.93 x 10^−6) instead of −76.00% (p = 4.71 x 10^−6), and the PCGrad step has
p = 0.099 instead of 0.098. Over every combination of the available records, the full stack vs.
base change stays between −76.08% and −75.89%.

The SAC table comparison is not device-matched. The ten canonical records of the SAC base group
were evaluated on three GPU types (RTX 3090 x4, RTX PRO 6000 x3, H200 NVL x3) and those of the
SAC full stack on H200 NVL only, and no SAC checkpoint was re-evaluated, so their cross-device
drift was not measured. A drift of the size measured for PPO (about 1% at most) would be small
next to the seed-to-seed spread of the SAC base group (sample std 27% of its mean).
