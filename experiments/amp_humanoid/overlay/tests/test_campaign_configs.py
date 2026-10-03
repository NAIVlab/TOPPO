"""The 25 campaign configurations differ from each other exactly as documented.

The README and CHANGES.md describe every arm as "rung X plus one change". These
tests parse the shipped YAMLs and check each of those statements, so a config
edit that silently adds a second difference fails here. Pure YAML: no simulator,
no torch.

Run from the root of the assembled tree:
    python -m pytest tests/test_campaign_configs.py -q
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "data" / "agents"
ENVS = ROOT / "data" / "envs"
ABSENT = "<absent>"


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flat(d, prefix=""):
    out = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(_flat(value, prefix + key + "."))
        else:
            out[prefix + key] = value
    return out


def _diff(a, b, root=AGENTS):
    fa, fb = _flat(_load(root / a)), _flat(_load(root / b))
    return {k: (fa.get(k, ABSENT), fb.get(k, ABSENT))
            for k in sorted(set(fa) | set(fb)) if fa.get(k, ABSENT) != fb.get(k, ABSENT)}


TOPPO_BLOCK_K2 = {
    "toppo.enabled": True, "toppo.num_tasks": 2,
    "toppo.task_names": ["task_reward", "disc_reward"],
    "toppo.actor_components": False, "toppo.actor_method": "default",
    "toppo.critic_method": "default", "toppo.fairgrad_alpha": 1.0,
    "toppo.fairgrad_rescale": False, "toppo.critic_layer_norm": True,
    "toppo.critic_layer_norm_biasless": True, "toppo.use_popart": True,
    "toppo.popart_method": "welford", "toppo.popart_min_std": 0.01,
}


def test_ppo_ladder_rungs():
    # row 1 -> row 2: add the two-component critic with LN-c and Welford PopArt
    d = _diff("amp_task_humanoid_long_critic_clip1_agent.yaml",
              "amp_task_humanoid_component_welford_ln_clip1_agent.yaml")
    assert d == {k: (ABSENT, v) for k, v in TOPPO_BLOCK_K2.items()}
    # row 2 -> row 3: FairGrad on the critic, and critic gradient clipping off
    d = _diff("amp_task_humanoid_component_welford_ln_clip1_agent.yaml",
              "amp_task_humanoid_toppo_cc100_noclip_agent.yaml")
    assert d == {"critic_optimizer.grad_clip": (1.0, 0.0),
                 "toppo.critic_coef": (ABSENT, 1.0),        # 1.0 is the code default
                 "toppo.critic_method": ("default", "fairgrad")}
    # row 3 -> row 4: PCGrad over the two reward components on the actor
    d = _diff("amp_task_humanoid_toppo_cc100_noclip_agent.yaml",
              "amp_task_humanoid_fulltoppo_cc100_agent.yaml")
    assert d == {"toppo.actor_components": (False, True),
                 "toppo.actor_method": ("default", "pcgrad")}


def test_ppo_controls():
    d = _diff("amp_task_humanoid_long_critic_clip1_agent.yaml",
              "amp_task_humanoid_ppo_reweight_agent.yaml")
    assert d == {"disc_reward_weight": (0.5, 1.105)}
    for name, extra in [
            ("amp_task_humanoid_w5_rewnorm_agent.yaml",
             {"w5_norm.method": "rewnorm", "w5_norm.rewnorm_clip": 10.0}),
            ("amp_task_humanoid_w5_retscale_agent.yaml",
             {"w5_norm.method": "retscale", "w5_norm.retscale_min_sigma": 0.01}),
            ("amp_task_humanoid_w5_hlgauss_agent.yaml",
             {"w5_norm.method": "hlgauss", "w5_norm.hlgauss_num_bins": 141,
              "w5_norm.hlgauss_vmin": -10.0, "w5_norm.hlgauss_vmax": 60.0,
              "w5_norm.hlgauss_sigma_ratio": 0.75})]:
        d = _diff("amp_task_humanoid_component_welford_ln_clip1_agent.yaml", name)
        expected = {"agent_name": ("AMP", "W5AMP"),
                    "toppo.use_popart": (True, False),
                    "toppo.popart_method": ("welford", ABSENT),
                    "toppo.popart_min_std": (0.01, ABSENT)}
        expected.update({k: (ABSENT, v) for k, v in extra.items()})
        assert d == expected, name
    d = _diff("amp_task_humanoid_w5_hlgauss_agent.yaml",
              "amp_task_humanoid_w5_hlgauss_lr1e3_agent.yaml")
    assert list(d) == ["critic_optimizer.learning_rate"]
    assert float(d["critic_optimizer.learning_rate"][0]) == 1e-4
    assert float(d["critic_optimizer.learning_rate"][1]) == 1e-3


def test_sac_ladder_and_controls():
    assert _diff("amp_sac_walk_k1cd_agent.yaml", "amp_sac_walk_lnonly_agent.yaml") \
        == {"toppo.use_popart": (True, False)}
    assert _diff("amp_sac_walk_k1cd_agent.yaml", "amp_sac_walk_popartonly_agent.yaml") \
        == {"toppo.critic_layer_norm": (True, False)}
    assert _diff("amp_sac_walk_k1cd_agent.yaml", "amp_sac_walk_cd_agent.yaml") \
        == {"toppo.num_tasks": (1, 2),
            "toppo.task_names": (["amp_reward"], ["task_reward", "disc_reward"])}
    assert _diff("amp_sac_walk_cd_agent.yaml", "amp_sac_walk_cb_agent.yaml") \
        == {"toppo.critic_method": ("default", "fairgrad"),
            "toppo.fairgrad_gram_dtype": (ABSENT, "float32")}
    # table row 5 -> row 6
    d = _diff("amp_sac_walk_cb_agent.yaml", "amp_sac_walk_fullstack_agent.yaml")
    assert d == {"toppo.actor_components": (False, True),
                 "toppo.actor_method": ("default", "pcgrad")}
    assert set(_diff("amp_sac_walk_agent.yaml", "amp_sac_walk_fullstack_agent.yaml")) \
        == {k for k in _flat(_load(AGENTS / "amp_sac_walk_fullstack_agent.yaml"))
            if k.startswith("toppo.")}
    for name, key, before, after in [
            ("amp_sac_walk_reweight_agent.yaml", "disc_reward_weight", 0.5, 1.105),
            ("amp_sac_walk_entmatch_agent.yaml", "target_entropy", "auto", -44.15),
            ("amp_sac_walk_utd512_agent.yaml", "updates_per_iter", 64, 512)]:
        assert _diff("amp_sac_walk_agent.yaml", name) == {key: (before, after)}, name
    assert _diff("amp_sac_walk_cb_agent.yaml", "amp_sac_walk_cbreweight_agent.yaml") \
        == {"disc_reward_weight": (0.5, 1.105)}
    assert _diff("amp_sac_walk_ars_agent.yaml", "amp_sac_walk_arsreset_agent.yaml") \
        == {"ars.num_resets": (0, 4)}
    for name in ["amp_sac_walk_w5_rewnorm_agent.yaml", "amp_sac_walk_w5_retscale_agent.yaml",
                 "amp_sac_walk_w5_hlgauss_agent.yaml"]:
        d = _diff("amp_sac_walk_cb_agent.yaml", name)
        assert d.pop("toppo.use_popart") == (True, False), name
        assert d.pop("toppo.popart_method") == ("welford", ABSENT), name
        assert d.pop("toppo.popart_min_std") == (0.01, ABSENT), name
        assert all(k.startswith("w5_norm.") and v[0] == ABSENT for k, v in d.items()), name


def test_hlgauss_support_is_shared_by_both_families():
    ppo = _load(AGENTS / "amp_task_humanoid_w5_hlgauss_agent.yaml")["w5_norm"]
    sac = _load(AGENTS / "amp_sac_walk_w5_hlgauss_agent.yaml")["w5_norm"]
    assert ppo == sac


def test_eval_env_differs_from_training_env_only_in_target_schedule():
    d = _diff("amp_steering_humanoid_walk_env.yaml",
              "amp_steering_humanoid_walk_eval_env.yaml", root=ENVS)
    assert d == {"tar_change_time_min": (5.0, 100.0), "tar_change_time_max": (10.0, 100.0)}
    env = _load(ENVS / "amp_steering_humanoid_walk_env.yaml")
    assert env["motion_file"] == "data/datasets/dataset_humanoid_deepmimic_walk_smoke.yaml"
    assert env["min_root_height"] == 0.5 and env["min_root_upright"] == 0.5
