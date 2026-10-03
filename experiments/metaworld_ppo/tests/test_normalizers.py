"""Unit tests for the critic-normalizer alternatives.

Pure torch/numpy — no MetaWorld / tianshou env needed.  Run:
    python -m pytest tests/test_normalizers.py -q
"""

import math

import numpy as np
import pytest
import torch

from mtrl.algos.return_scaling import ReturnScalingModule, _sigma_sq
from mtrl.models.hlgauss import HLGaussHead


# ---------------------------------------------------------------------------
# HL-Gauss
# ---------------------------------------------------------------------------

def test_hlgauss_probs_sum_to_one():
    head = HLGaussHead(num_bins=51, vmin=-10.0, vmax=10.0, sigma_ratio=0.75)
    y = torch.tensor([-10.0, -3.7, 0.0, 5.2, 10.0])
    p = head.to_probs(y)
    assert p.shape == (5, 51)
    assert torch.all(p >= 0)
    assert torch.allclose(p.sum(-1), torch.ones(5), atol=1e-6)


def test_hlgauss_mean_preserving_interior():
    head = HLGaussHead(num_bins=101, vmin=0.0, vmax=100.0, sigma_ratio=0.75)
    y = torch.tensor([13.37, 50.0, 77.7])
    p = head.to_probs(y)
    recon = p @ head.centers
    # Gaussian smoothing keeps the mean to well within one bin width.
    assert torch.allclose(recon, y, atol=head.bin_width / 2)


def test_hlgauss_mass_spread_about_six_bins():
    head = HLGaussHead(num_bins=101, vmin=0.0, vmax=101.0, sigma_ratio=0.75)
    p = head.to_probs(torch.tensor([50.5]))
    significant = (p > 1e-3).sum().item()
    assert 4 <= significant <= 8  # paper: ratio 0.75 → mass on ~6 locations


def test_hlgauss_out_of_range_targets_saturate():
    head = HLGaussHead(num_bins=51, vmin=-10.0, vmax=10.0)
    p = head.to_probs(torch.tensor([1e6, -1e6]))
    recon = p @ head.centers
    assert recon[0] > 10.0 - 2 * head.bin_width
    assert recon[1] < -10.0 + 2 * head.bin_width
    assert torch.isfinite(p).all()


def test_hlgauss_loss_gradient_flows():
    head = HLGaussHead(num_bins=21, vmin=0.0, vmax=10.0)
    logits = torch.zeros(4, 21, requires_grad=True)
    loss = head.loss(logits, torch.tensor([1.0, 2.0, 3.0, 4.0])).mean()
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    # Uniform logits → readout at support midpoint.
    assert torch.allclose(
        head.from_logits(torch.zeros(1, 21)), torch.tensor([5.0]), atol=1e-5
    )


def test_hlgauss_validation():
    with pytest.raises(ValueError):
        HLGaussHead(num_bins=1, vmin=0.0, vmax=1.0)
    with pytest.raises(ValueError):
        HLGaussHead(num_bins=10, vmin=1.0, vmax=1.0)


# ---------------------------------------------------------------------------
# Return-based scaling
# ---------------------------------------------------------------------------

def test_return_scaling_matches_reference():
    torch.manual_seed(0)
    mod = ReturnScalingModule(num_tasks=2, min_sigma=1e-2)

    # Two collects with hand-checkable statistics.
    all_r = {0: [], 1: []}
    all_g = {0: [], 1: []}
    all_G = {0: [], 1: []}
    for _ in range(2):
        n = 64
        task_idx = torch.randint(0, 2, (n,))
        r = torch.randn(n) * torch.where(task_idx == 0, 1.0, 100.0)
        term = (torch.rand(n) < 0.1).double()
        g = 0.99 * (1.0 - term)
        G = torch.randn(n) * torch.where(task_idx == 0, 5.0, 500.0)
        mod.update_and_refresh_scale(task_idx, r, g, G)
        for t in (0, 1):
            m = task_idx == t
            all_r[t] += r[m].tolist()
            all_g[t] += g[m].tolist()
            all_G[t] += G[m].tolist()

    for t in (0, 1):
        var_r = np.var(all_r[t])
        var_g = np.var(all_g[t])
        mean_G2 = np.mean(np.square(all_G[t]))
        sigma_ref = math.sqrt(var_r + var_g * mean_G2)
        sigma_mod = mod.get_diagnostics()["sigma"][t].item()
        assert sigma_mod == pytest.approx(sigma_ref, rel=1e-6)
        # After the last update, scale = max(σ_all, σ_batch, floor) ≥ σ_all.
        assert mod.last_scale[t].item() >= sigma_mod - 1e-9

    # Task with 100× rewards must get a much larger scale.
    scales = mod.get_diagnostics()["scale"]
    assert scales[1].item() > 10 * scales[0].item()


def test_return_scaling_min_sigma_floor():
    mod = ReturnScalingModule(num_tasks=1, min_sigma=1e-2)
    # Zero-reward pre-training regime → floor applies.
    assert mod.last_scale[0].item() == pytest.approx(1e-2)
    mod.update_and_refresh_scale(
        torch.zeros(10, dtype=torch.long),
        torch.zeros(10),
        torch.full((10,), 0.99),
        torch.zeros(10),
    )
    assert mod.last_scale[0].item() == pytest.approx(1e-2)


def test_return_scaling_batch_sigma_guard():
    mod = ReturnScalingModule(num_tasks=1, min_sigma=1e-2)
    # Many silent collects, then a sudden large-reward collect: the batch σ
    # must dominate the (still tiny) all-time σ in that collect's scale.
    for _ in range(20):
        mod.update_and_refresh_scale(
            torch.zeros(100, dtype=torch.long),
            torch.zeros(100),
            torch.full((100,), 0.99),
            torch.zeros(100),
        )
    spike_r = torch.cat([torch.zeros(90), torch.full((10,), 1000.0)])
    mod.update_and_refresh_scale(
        torch.zeros(100, dtype=torch.long),
        spike_r,
        torch.full((100,), 0.99),
        torch.zeros(100),
    )
    sigma_all = mod.get_diagnostics()["sigma"][0].item()
    sigma_batch = math.sqrt(np.var(spike_r.numpy()))
    assert mod.last_scale[0].item() == pytest.approx(sigma_batch, rel=1e-6)
    assert mod.last_scale[0].item() > sigma_all


def test_return_scaling_state_dict_roundtrip():
    mod = ReturnScalingModule(num_tasks=3)
    mod.update_and_refresh_scale(
        torch.tensor([0, 1, 2, 0]),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([0.99, 0.0, 0.99, 0.99]),
        torch.tensor([10.0, 20.0, 30.0, 40.0]),
    )
    fresh = ReturnScalingModule(num_tasks=3)
    fresh.load_state_dict(mod.state_dict())
    assert torch.equal(fresh.n, mod.n)
    assert torch.equal(fresh.last_scale, mod.last_scale)


def test_sigma_sq_empty_task_is_zero():
    z = torch.zeros(2, dtype=torch.float64)
    n = torch.tensor([0.0, 4.0], dtype=torch.float64)
    s_r = torch.tensor([0.0, 8.0], dtype=torch.float64)
    s_r2 = torch.tensor([0.0, 32.0], dtype=torch.float64)
    out = _sigma_sq(n, s_r, s_r2, z.clone(), z.clone(), z.clone())
    assert out[0].item() == 0.0
    assert out[1].item() == pytest.approx(32.0 / 4 - 4.0)  # Var[R] = E[R²]−E[R]²


# ---------------------------------------------------------------------------
# baselines_return reward normalization wrapper
# ---------------------------------------------------------------------------

class _ScriptedEnv(__import__("gymnasium").Env):
    """Minimal gymnasium env emitting a scripted reward sequence."""

    def __init__(self, rewards, done_every):
        import gymnasium as gym

        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)
        self._rewards = list(rewards)
        self._i = 0
        self._done_every = done_every

    def reset(self, **kwargs):
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        r = self._rewards[self._i % len(self._rewards)]
        self._i += 1
        done = (self._i % self._done_every) == 0
        return np.zeros(3, dtype=np.float32), r, done, False, {}


def test_baselines_return_wrapper_matches_reference():
    from mtrl.envs.metaworld_wrapper import RewardNormalizationWrapper

    gamma, clip = 0.99, 10.0
    rng = np.random.RandomState(7)
    rewards = rng.uniform(0, 10, size=200).tolist()

    env = RewardNormalizationWrapper(
        _ScriptedEnv(rewards, done_every=40), method="baselines_return",
        gamma=gamma, clip=clip,
    )
    env.reset()

    # Reference: VecNormalize with a single env (Chan merge, batch=1).
    ret, mean, var, count = 0.0, 0.0, 1.0, 1e-4
    for t, r in enumerate(rewards):
        _, r_norm, terminated, truncated, info = env.step(None)
        ret = gamma * ret + r
        delta = ret - mean
        tot = count + 1.0
        mean += delta / tot
        var = (var * count + delta * (ret - mean)) / tot
        count = tot
        expected = float(np.clip(r / np.sqrt(var + 1e-8), -clip, clip))
        if terminated or truncated:
            ret = 0.0
        assert info["raw_reward"] == pytest.approx(r)
        assert r_norm == pytest.approx(expected, rel=1e-10)


def test_baselines_return_stats_persist_across_reset():
    from mtrl.envs.metaworld_wrapper import RewardNormalizationWrapper

    env = RewardNormalizationWrapper(
        _ScriptedEnv([5.0], done_every=10), method="baselines_return", gamma=0.9,
    )
    env.reset()
    for _ in range(50):
        env.step(None)
    count_before = env._brn_state["count"]
    env.reset()
    assert env._brn_state["ret"] == 0.0
    assert env._brn_state["count"] == count_before  # stats never reset


def test_baselines_return_state_survives_env_rebuild():
    """Checkpoint-resume path: snapshot the unwrapped-env state dict, rebuild
    the env chain, push the state back, and confirm normalization continues
    exactly as an uninterrupted run would."""
    from mtrl.envs.metaworld_wrapper import RewardNormalizationWrapper

    rng = np.random.RandomState(3)
    rewards = rng.uniform(0, 8, size=50).tolist()

    def make():
        e = RewardNormalizationWrapper(
            _ScriptedEnv(rewards, done_every=50), method="baselines_return",
            gamma=0.99,
        )
        e.reset()
        return e

    ref = make()
    for _ in range(150):  # 3 full episodes -> accumulator is 0 at boundary
        ref.step(None)

    # "Preemption": snapshot via the attribute train.py reads with
    # get_env_attr, rebuild, restore with ret zeroed (as train.py does).
    snapshot = dict(ref.env.unwrapped.reward_norm_state)
    resumed = make()
    resumed.env.unwrapped.reward_norm_state = dict(snapshot, ret=0.0)

    for _ in range(75):
        _, r_ref, *_ = ref.step(None)
        _, r_res, *_ = resumed.step(None)
        assert r_res == pytest.approx(r_ref, rel=1e-12)


def _make_full_chain_env(task_id):
    """TimeLimit → RewardNormalization → TaskIDWrapper, like _make_single_env."""
    import gymnasium as gym

    from mtrl.envs.metaworld_wrapper import (
        RewardNormalizationWrapper,
        TaskIDWrapper,
    )

    env = _ScriptedEnv([1.0, 2.0, 3.0, 4.0], done_every=10 ** 9)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=20)
    env = RewardNormalizationWrapper(env, method="baselines_return", gamma=0.99)
    env = TaskIDWrapper(env, task_id=task_id, num_tasks=2, task_name=f"t{task_id}")
    return env


def test_baselines_return_vecenv_attr_roundtrip():
    """The production checkpoint transport: ShmemVectorEnv.get_env_attr must
    see the stats through the outermost TaskIDWrapper (gymnasium 1.x dropped
    wrapper attr delegation) and set_env_attr (which targets env.unwrapped)
    must feed the restored dict back into normalization."""
    from tianshou.env.venvs import ShmemVectorEnv

    venv = ShmemVectorEnv([lambda i=i: _make_full_chain_env(i) for i in range(2)])
    try:
        venv.reset()
        for _ in range(30):
            venv.step(np.zeros((2, 3), dtype=np.float32))

        states = venv.get_env_attr("reward_norm_state")
        assert len(states) == 2
        for st in states:
            assert isinstance(st, dict) and st["count"] > 25

        # Restore a doctored state and verify the next step uses it: with a
        # huge variance the normalized reward collapses toward zero.
        doctored = dict(states[0], var=1e12, ret=0.0)
        venv.set_env_attr("reward_norm_state", doctored, id=0)
        _, rews, *_ = venv.step(np.zeros((2, 3), dtype=np.float32))
        assert abs(rews[0]) < 1e-4
        assert abs(rews[1]) > 1e-2  # untouched env keeps its own scale
    finally:
        venv.close()


# ---------------------------------------------------------------------------
# Mutual exclusions (constructor-level; no env needed)
# ---------------------------------------------------------------------------

def test_critic_accepts_hlgauss_plus_popart_art_only():
    """PopArt x HL-Gauss composes: ART supplies the per-task target
    statistics that let one fixed support serve every task.  The POP half
    has no categorical analog, so only (mu, sigma) are used."""
    import torch
    from mtrl.models.networks import MultiTaskCritic, SharedPreprocessNet

    pre = SharedPreprocessNet(input_dim=12, hidden_sizes=[], activation="relu")
    critic = MultiTaskCritic(
        task_names=["a", "b"],
        preprocess_net=pre,
        critic_hidden_sizes=[32],
        critic_head_hidden_sizes=[],
        popart_enabled=True,
        hlgauss_num_bins=51,
        hlgauss_vmin=-5.0,
        hlgauss_vmax=5.0,
    )
    assert hasattr(critic, "popart"), "ART statistics must be available"
    assert hasattr(critic, "hlgauss")

    obs = torch.zeros(4, 12)
    obs[:2, -2] = 1.0   # task a
    obs[2:, -1] = 1.0   # task b
    logits, task_idx = critic.forward_logits(obs)
    assert logits.shape == (4, 51)

    # forward() must return RAW values: shifting ART's per-task stats must
    # move the readout by exactly (sigma * v_norm + mu).
    v_before = critic(obs).flatten()
    with torch.no_grad():
        critic.popart.mu[0] = 100.0
        critic.popart.sigma[0] = 10.0
    v_after = critic(obs).flatten()
    v_norm = critic.hlgauss.from_logits(logits)
    expected0 = 10.0 * v_norm[:2] + 100.0
    assert torch.allclose(v_after[:2], expected0, atol=1e-4)
    assert torch.allclose(v_after[2:], v_before[2:], atol=1e-6)


def test_critic_rejects_hlgauss_plus_nonempty_head():
    from mtrl.models.networks import MultiTaskCritic, SharedPreprocessNet

    pre = SharedPreprocessNet(input_dim=12, hidden_sizes=[], activation="relu")
    with pytest.raises(ValueError, match="empty critic_head"):
        MultiTaskCritic(
            task_names=["a", "b"],
            preprocess_net=pre,
            critic_hidden_sizes=[32],
            critic_head_hidden_sizes=[16],
            hlgauss_num_bins=51,
            hlgauss_vmin=0.0,
            hlgauss_vmax=10.0,
        )


def test_hlgauss_critic_forward_shapes():
    from mtrl.models.networks import MultiTaskCritic, SharedPreprocessNet

    num_tasks, obs_dim = 3, 8
    pre = SharedPreprocessNet(
        input_dim=obs_dim + num_tasks, hidden_sizes=[], activation="relu"
    )
    critic = MultiTaskCritic(
        task_names=["a", "b", "c"],
        preprocess_net=pre,
        critic_hidden_sizes=[32, 32],
        critic_head_hidden_sizes=[],
        body_use_layer_norm=True,
        hlgauss_num_bins=51,
        hlgauss_vmin=-10.0,
        hlgauss_vmax=10.0,
    )
    obs = torch.randn(5, obs_dim + num_tasks)
    obs[:, obs_dim:] = 0.0
    obs[torch.arange(5), obs_dim + torch.tensor([0, 1, 2, 0, 1])] = 1.0

    logits, task_idx = critic.forward_logits(obs)
    assert logits.shape == (5, 51)
    assert task_idx.tolist() == [0, 1, 2, 0, 1]

    v = critic(obs)
    assert v.shape == (5, 1)
    assert torch.all(v >= -10.0) and torch.all(v <= 10.0)
    with pytest.raises(RuntimeError):
        critic._compute_v_pre(obs)
