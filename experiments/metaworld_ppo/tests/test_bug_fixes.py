"""Tests for bug fixes in the MTRL framework."""

import glob
import os

import pytest


# ============================================================================
# Bug 4: Explicit use_linear_baseline config
# ============================================================================

def test_explicit_baseline_config():
    """Config should have use_linear_baseline field."""
    from mtrl.utils.config import Config

    config = Config()
    assert hasattr(config, "use_linear_baseline")
    assert config.use_linear_baseline is False

    config2 = Config(use_linear_baseline=True)
    assert config2.use_linear_baseline is True


# --- HL-Gauss + return scaling: forward() must return RAW units ---------------
# The categorical head is trained on target / s_i, so its readout is in scaled
# units.  GAE, advantages and evaluation all expect raw units, so the critic has
# to undo the scaling in forward().  Before this was wired the value function and
# the reward stream were in different units for that combination.
#
# These tests go through the real nn.Module path: a real MultiTaskCritic handed
# to a real MultiTaskPPO, which attaches its ReturnScalingModule to the critic
# itself.  A class-level ``retscale = None`` default on MultiTaskCritic once
# shadowed that submodule (nn.Module keeps it in ``_modules``), so forward()
# silently skipped the rescale; a duck-typed stand-in for the critic cannot see
# that, so none is used here.

_TASKS = ["task-a", "task-b"]
_RAW_OBS_DIM, _ACT_DIM = 4, 2


def _build_hlgauss_algo(use_return_scaling: bool):
    """MultiTaskCritic (HL-Gauss) + MultiTaskPPO, wired as train.py wires them."""
    import gymnasium as gym
    import numpy as np
    import torch
    from tianshou.algorithm.optim import AdamOptimizerFactory
    from torch.distributions import Independent, Normal

    from mtrl.algos.multitask_ppo import MultiTaskPPO
    from mtrl.models.multitask_policy import MultiTaskProbabilisticActorPolicy
    from mtrl.models.networks import (
        MultiTaskActor,
        MultiTaskCritic,
        SharedPreprocessNet,
    )

    def dist_fn(loc_scale):
        loc, scale = loc_scale
        return Independent(Normal(loc=loc, scale=scale), reinterpreted_batch_ndims=1)

    torch.manual_seed(0)
    obs_dim = _RAW_OBS_DIM + len(_TASKS)
    pre = SharedPreprocessNet(input_dim=obs_dim, hidden_sizes=[], activation="relu")
    actor = MultiTaskActor(
        action_shape=_ACT_DIM,
        task_names=_TASKS,
        preprocess_net=pre,
        actor_hidden_sizes=[16],
        actor_head_hidden_sizes=[],
        activation="relu",
        max_action=1.0,
        unbounded=True,
        conditioned_sigma=False,
    )
    # Support [-10, 30]: the near-uniform readout at init sits near 10, far
    # from 0, so a missing rescale cannot hide behind a tiny value.
    critic = MultiTaskCritic(
        task_names=_TASKS,
        preprocess_net=pre,
        critic_hidden_sizes=[16],
        critic_head_hidden_sizes=[],
        activation="relu",
        act_dim=0,
        hlgauss_num_bins=101,
        hlgauss_vmin=-10.0,
        hlgauss_vmax=30.0,
    )
    policy = MultiTaskProbabilisticActorPolicy(
        actor=actor,
        dist_fn=dist_fn,
        task_names=_TASKS,
        deterministic_eval=True,
        action_space=gym.spaces.Box(-1.0, 1.0, (_ACT_DIM,), np.float32),
        observation_space=gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
        action_scaling=True,
        action_bound_method="clip",
    )
    algo = MultiTaskPPO(
        policy=policy,
        critic=critic,
        optim=AdamOptimizerFactory(lr=3e-4),
        task_names=_TASKS,
        value_clip=False,  # required by critic_loss_mode="hlgauss"
        vf_coef=1e-3,
        use_return_scaling=use_return_scaling,
        critic_loss_mode="hlgauss",
    )
    return algo, critic


def _task_obs():
    """Two samples per task: random raw obs + one-hot task tail."""
    import torch

    gen = torch.Generator().manual_seed(1)
    task_ids = torch.tensor([0, 1, 0, 1])
    raw = torch.randn(len(task_ids), _RAW_OBS_DIM, generator=gen)
    one_hot = torch.nn.functional.one_hot(task_ids, len(_TASKS)).float()
    return torch.cat([raw, one_hot], dim=-1), task_ids


def test_hlgauss_forward_undoes_return_scaling():
    import torch

    algo, critic = _build_hlgauss_algo(use_return_scaling=True)
    # The module MultiTaskPPO attached must be the one forward() sees.
    assert critic.retscale is algo.retscale_module

    s = torch.tensor([25.0, 100.0], dtype=torch.float64)
    algo.retscale_module.last_scale.copy_(s)

    obs, task_ids = _task_obs()
    with torch.no_grad():
        v = critic(obs).flatten()
        logits, _ = critic.forward_logits(obs)
        readout = critic.hlgauss.from_logits(logits)
    expected = readout * s.float()[task_ids]
    assert torch.allclose(v, expected, rtol=1e-6), (
        f"forward() = {v.tolist()}, expected readout * s_i = {expected.tolist()}"
    )
    assert torch.all((v - readout).abs() > 1.0), "rescale was not applied"


def test_hlgauss_forward_untouched_without_return_scaling():
    import torch

    _, critic = _build_hlgauss_algo(use_return_scaling=False)
    assert getattr(critic, "retscale", None) is None
    assert "retscale" not in critic._modules

    obs, _ = _task_obs()
    with torch.no_grad():
        v = critic(obs)
        logits, _ = critic.forward_logits(obs)
    # Every configuration without return scaling: plain HL-Gauss readout.
    assert torch.equal(v, critic.hlgauss.from_logits(logits).unsqueeze(-1))


# --- Config loader: reject what it cannot apply ------------------------------
# A misspelled key used to leave its field at the default without a word, and a
# quoted number reached the trainer as a string.  load_config now rejects both,
# and every shipped config must still load.

_PPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("body", [
    "exp_nam: x\n",
    "training:\n  epoch: \"20\"\n",
    "ppo:\n  vf_coeff: 0.5\n",
])
def test_load_config_rejects_bad_keys_and_types(tmp_path, body):
    from mtrl.utils.config import load_config

    path = tmp_path / "bad.yaml"
    path.write_text(body)
    with pytest.raises(ValueError):
        load_config(str(path), cli_args=["--config", str(path)])


def test_every_shipped_config_loads():
    from mtrl.utils.config import Config, load_config

    paths = sorted(glob.glob(os.path.join(_PPO_ROOT, "configs", "**", "*.yaml"),
                             recursive=True))
    assert len(paths) > 50
    # Training logs to a scratch project by default, never to the published one.
    assert Config().project_name == "toppo-repro"
    for path in paths:
        cfg = load_config(path, cli_args=["--config", path])
        assert cfg.project_name == "toppo-repro", path


# --- W&B source: one rule in all three analysis scripts ----------------------

def _load_script(name):
    import importlib.util

    path = os.path.join(_PPO_ROOT, "analysis", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_toppo_analysis_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wandb_source_rule_is_shared(monkeypatch):
    mods = [_load_script(n) for n in ("metrics", "export_wandb", "check_runs_index")]
    cases = []
    for env_entity in (None, "me"):
        for env_project in (None, "mine"):
            for entity in (None, "flag-entity"):
                for project in (None, "TOPPO", "flag-project"):
                    cases.append((env_entity, env_project, entity, project))
    for env_entity, env_project, entity, project in cases:
        for var, val in (("WANDB_ENTITY", env_entity), ("WANDB_PROJECT", env_project)):
            if val is None:
                monkeypatch.delenv(var, raising=False)
            else:
                monkeypatch.setenv(var, val)
        got = {m.__name__: m.resolve_source(entity, project) for m in mods}
        assert len(set(got.values())) == 1, (env_entity, env_project, entity, project, got)
        if entity is None and project is None and env_project is None:
            # $WANDB_ENTITY alone does not move the source.
            assert next(iter(got.values())) == ("liyp1220-uci", "TOPPO")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
