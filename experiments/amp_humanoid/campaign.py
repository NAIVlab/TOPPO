#!/usr/bin/env python3
"""The AMP humanoid campaign: 25 configurations x 10 training seeds.

This file is the single table behind the README, the analysis scripts and the
launch script. Each row is one public W&B run group (project
liyp1220-uci/TOPPO); the group name is the canonical identifier.

    python campaign.py list                      # the 25 rows
    python campaign.py train <group> <seed>      # training command
    python campaign.py eval  <group> <seed>      # evaluation command
    python campaign.py commands [--group G ...] [--seed S ...]   # both, for many runs

Commands are printed, not executed, and are meant to be run from the root of
the tree that setup.sh assembles. Seeds are the actual random seeds passed to
MimicKit (--rand_seed), 260511470 .. 260511479.
"""

import argparse
import shlex
import sys
from collections import namedtuple

SEEDS = tuple(range(260511470, 260511480))
EVAL_SEED = 260511499
STYLE_SEED = 260511701
STYLE_SAMPLES = 8192
STYLE_PROJECTION_DIM = 64
NUM_ENVS = 4096
TEST_EPISODES = 4096

ENGINE_CONFIG = "data/engines/newton_engine.yaml"
TRAIN_ENV_CONFIG = "data/envs/amp_steering_humanoid_walk_env.yaml"
EVAL_ENV_CONFIG = "data/envs/amp_steering_humanoid_walk_eval_env.yaml"

Row = namedtuple("Row", "group family arm wandb_arm label agent_yaml budget table_row")

# table_row: the row of the paper's Table E.6 (Appendix E.5) this group fills, or None.
CAMPAIGN = [
    # ---- AMP-PPO: the component ladder (table rows 1-4) --------------------------------
    Row("ampwalk_ppo_ad_cd_nnone_ln0_k1_gc1", "ppo", "ppo_base", "ppo_vanilla",
        "AMP-PPO (base)",
        "amp_task_humanoid_long_critic_clip1_agent.yaml", 100_000_000, 1),
    Row("ampwalk_ppo_ad_cd_npaw_lnc_k2_gc1", "ppo", "ppo_popart", "ppo_criticbal",
        "AMP-PPO +K2 +LN +PopArt",
        "amp_task_humanoid_component_welford_ln_clip1_agent.yaml", 100_000_000, 2),
    Row("ampwalk_ppo_ad_cfg1_npaw_lnc_k2_gc0", "ppo", "ppo_fairgrad", "ppo_toppo",
        "AMP-PPO +K2 +LN +PopArt +FairGrad -GradClip",
        "amp_task_humanoid_toppo_cc100_noclip_agent.yaml", 100_000_000, 3),
    Row("ampwalk_ppo_apc_cfg1_npaw_lnc_k2_gc0", "ppo", "ppo_pcgrad", "ppo_fulltoppo",
        "AMP-PPO +K2 +LN +PopArt +FairGrad -GradClip +PCGrad (full stack)",
        "amp_task_humanoid_fulltoppo_cc100_agent.yaml", 100_000_000, 4),
    # ---- AMP-PPO: controls and alternative critic normalizers ---------------------------
    Row("ampwalk_ppo_ad_cd_nnone_ln0_k1_gc1_reweight", "ppo", "ppo_rw", "ppo_reweight",
        "AMP-PPO +RW (style weight 0.5 -> 1.105)",
        "amp_task_humanoid_ppo_reweight_agent.yaml", 100_000_000, None),
    Row("ampwalk_ppo_ad_cd_nrew_lnc_k2_gc1", "ppo", "ppo_rewnorm", "ppo_w5rewnorm",
        "AMP-PPO +K2 +LN +RewNorm",
        "amp_task_humanoid_w5_rewnorm_agent.yaml", 100_000_000, None),
    Row("ampwalk_ppo_ad_cd_nret_lnc_k2_gc1", "ppo", "ppo_retscale", "ppo_w5retscale",
        "AMP-PPO +K2 +LN +RetScale",
        "amp_task_humanoid_w5_retscale_agent.yaml", 100_000_000, None),
    Row("ampwalk_ppo_ad_cd_nhg_lnc_k2_gc1", "ppo", "ppo_hlgauss", "ppo_w5hlgauss",
        "AMP-PPO +K2 +LN +HLGauss",
        "amp_task_humanoid_w5_hlgauss_agent.yaml", 100_000_000, None),
    Row("ampwalk_ppo_ad_cd_nhg_lnc_k2_gc1_lr1e3", "ppo", "ppo_hlgauss_lr10x", "ppo_w5hlgauss_lr1e3",
        "AMP-PPO +K2 +LN +HLGauss (critic lr x10)",
        "amp_task_humanoid_w5_hlgauss_lr1e3_agent.yaml", 100_000_000, None),
    # ---- AMP-SAC: the critic ladder (table rows 5-6) -----------------------------------
    Row("ampwalk_sac_ad_cd_nnone_ln0_k1_gc0", "sac", "sac_base", "sac_vanilla",
        "AMP-SAC (base)",
        "amp_sac_walk_agent.yaml", 100_000_000, 5),
    Row("ampwalk_sac_ad_cd_nnone_lnc_k1_gc0", "sac", "sac_ln", "sac_lnonly",
        "AMP-SAC +LN",
        "amp_sac_walk_lnonly_agent.yaml", 100_000_000, None),
    Row("ampwalk_sac_ad_cd_npaw_ln0_k1_gc0", "sac", "sac_popart", "sac_popartonly",
        "AMP-SAC +PopArt",
        "amp_sac_walk_popartonly_agent.yaml", 100_000_000, None),
    Row("ampwalk_sac_ad_cd_npaw_lnc_k1_gc0", "sac", "sac_ln_popart", "sac_k1cd",
        "AMP-SAC +LN +PopArt",
        "amp_sac_walk_k1cd_agent.yaml", 100_000_000, None),
    Row("ampwalk_sac_ad_cd_npaw_lnc_k2_gc0", "sac", "sac_k2", "sac_cd",
        "AMP-SAC +K2 +LN +PopArt",
        "amp_sac_walk_cd_agent.yaml", 100_000_000, None),
    Row("ampwalk_sac_ad_cfg1_npaw_lnc_k2_gc0", "sac", "sac_fairgrad", "sac_cb",
        "AMP-SAC +K2 +LN +PopArt +FairGrad",
        "amp_sac_walk_cb_agent.yaml", 100_000_000, None),
    Row("ampwalk_sac_apc_cfg1_npaw_lnc_k2_gc0", "sac", "sac_pcgrad", "sac_fullstack",
        "AMP-SAC +K2 +LN +PopArt +FairGrad +PCGrad (full stack)",
        "amp_sac_walk_fullstack_agent.yaml", 100_000_000, 6),
    # ---- AMP-SAC: controls and alternative critic normalizers ---------------------------
    Row("ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_reweight", "sac", "sac_rw", "sac_reweight",
        "AMP-SAC +RW (style weight 0.5 -> 1.105)",
        "amp_sac_walk_reweight_agent.yaml", 100_000_000, None),
    Row("ampwalk_sac_ad_cfg1_npaw_lnc_k2_gc0_reweight", "sac", "sac_fairgrad_rw", "sac_cbreweight",
        "AMP-SAC +K2 +LN +PopArt +FairGrad +RW",
        "amp_sac_walk_cbreweight_agent.yaml", 100_000_000, None),
    Row("ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_ars", "sac", "sac_ars", "sac_ars",
        "AMP-SAC +ARS",
        "amp_sac_walk_ars_agent.yaml", 100_000_000, None),
    Row("ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_arsreset", "sac", "sac_ars_reset", "sac_arsreset",
        "AMP-SAC +ARS with learner resets",
        "amp_sac_walk_arsreset_agent.yaml", 100_000_000, None),
    Row("ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_entmatch", "sac", "sac_targetent", "sac_entmatch",
        "AMP-SAC +H* (target entropy matched to PPO)",
        "amp_sac_walk_entmatch_agent.yaml", 100_000_000, None),
    Row("ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_utd512", "sac", "sac_utd512", "sac_utd512",
        "AMP-SAC +UTD512 (8x updates per sample, 25M samples)",
        "amp_sac_walk_utd512_agent.yaml", 25_000_000, None),
    Row("ampwalk_sac_ad_cfg1_nrew_lnc_k2_gc0", "sac", "sac_rewnorm", "sac_w5rewnorm",
        "AMP-SAC +K2 +LN +FairGrad +RewNorm",
        "amp_sac_walk_w5_rewnorm_agent.yaml", 100_000_000, None),
    Row("ampwalk_sac_ad_cfg1_nret_lnc_k2_gc0", "sac", "sac_retscale", "sac_w5retscale",
        "AMP-SAC +K2 +LN +FairGrad +RetScale",
        "amp_sac_walk_w5_retscale_agent.yaml", 100_000_000, None),
    Row("ampwalk_sac_ad_cfg1_nhg_lnc_k2_gc0", "sac", "sac_hlgauss", "sac_w5hlgauss",
        "AMP-SAC +K2 +LN +FairGrad +HLGauss",
        "amp_sac_walk_w5_hlgauss_agent.yaml", 100_000_000, None),
]

BY_GROUP = {row.group: row for row in CAMPAIGN}
assert len(BY_GROUP) == 25 and len({row.agent_yaml for row in CAMPAIGN}) == 25


def out_dir(group, seed):
    return "output/{}_s{}/".format(group, seed)


def eval_json(group, seed):
    return "results/eval/{}_s{}.json".format(group, seed)


def train_argv(group, seed, device="cuda:0", python="python", max_samples=None, out=None):
    row = BY_GROUP[group]
    return [python, "mimickit/run.py", "--mode", "train",
            "--num_envs", str(NUM_ENVS),
            "--engine_config", ENGINE_CONFIG,
            "--env_config", TRAIN_ENV_CONFIG,
            "--agent_config", "data/agents/" + row.agent_yaml,
            "--out_dir", out or out_dir(group, seed),
            "--max_samples", str(max_samples or row.budget),
            "--rand_seed", str(seed),
            "--devices", device,
            "--logger", "txt",
            "--visualize", "false"]


def eval_argv(group, seed, device="cuda:0", python="python"):
    row = BY_GROUP[group]
    return [python, "mimickit/run.py", "--mode", "test",
            "--num_envs", str(NUM_ENVS),
            "--engine_config", ENGINE_CONFIG,
            "--env_config", EVAL_ENV_CONFIG,
            "--agent_config", "data/agents/" + row.agent_yaml,
            "--model_file", out_dir(group, seed) + "model.pt",
            "--devices", device,
            "--visualize", "false",
            "--test_episodes", str(TEST_EPISODES),
            "--detailed_eval", "true",
            "--style_samples", str(STYLE_SAMPLES),
            "--style_projection_dim", str(STYLE_PROJECTION_DIM),
            "--style_seed", str(STYLE_SEED),
            "--eval_seed", str(EVAL_SEED),
            "--eval_output", eval_json(group, seed)]


def _check(group, seed):
    if group not in BY_GROUP:
        sys.exit("unknown group {!r}; see `python campaign.py list`".format(group))
    if seed not in SEEDS:
        sys.stderr.write("note: {} is not one of the 10 campaign seeds\n".format(seed))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("group")
        p.add_argument("seed", type=int)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--python", default="python")
        if name == "train":
            p.add_argument("--max-samples", type=int, default=None,
                           help="override the budget (e.g. a short timing run)")
            p.add_argument("--out-dir", default=None, help="override output/<group>_s<seed>/")
    p = sub.add_parser("commands")
    p.add_argument("--group", action="append")
    p.add_argument("--seed", type=int, action="append")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--python", default="python")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        print("{:<46} {:<6} {:<18} {:>4} {:>5}  {}".format(
            "group", "family", "arm", "row", "steps", "agent config"))
        for row in CAMPAIGN:
            print("{:<46} {:<6} {:<18} {:>4} {:>4}M  {}".format(
                row.group, row.family, row.arm, row.table_row or "-",
                row.budget // 1_000_000, row.agent_yaml))
    elif args.cmd in ("train", "eval"):
        _check(args.group, args.seed)
        if args.cmd == "train":
            argv_ = train_argv(args.group, args.seed, args.device, args.python,
                               args.max_samples, args.out_dir)
        else:
            argv_ = eval_argv(args.group, args.seed, args.device, args.python)
        print(shlex.join(argv_))
    else:
        groups = args.group or [row.group for row in CAMPAIGN]
        seeds = args.seed or list(SEEDS)
        for group in groups:
            for seed in seeds:
                _check(group, seed)
                print(shlex.join(train_argv(group, seed, args.device, args.python)))
                print(shlex.join(eval_argv(group, seed, args.device, args.python)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
