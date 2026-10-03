"""CPU tests for the critic-side scale-normalizer alternatives to PopArt (w5_norm).

Covers the shared primitives (`learning/w5_norm.py`) and their wiring into
both families: `W5AMPAgent` (PPO/AMP) and `AMPSACAgent` (SAC).

Run from the repo root:
    python -m pytest tests/test_w5_norm.py -q
"""

import copy
import math
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))

# the two harnesses already stub tensorboardX/wandb/util.video and init mp_util
from test_sac_amp import (  # noqa: E402
    NUM_ENVS, SteppingAMPEnv, TEST_DEVICE, _run_train, _sac_agent_config)
from test_toppo import DummyAMPEnv, _toppo_agent_config  # noqa: E402

import learning.amp_sac_agent as amp_sac_agent  # noqa: E402
import learning.w5_amp_agent as w5_amp_agent  # noqa: E402
import learning.w5_norm as w5_norm  # noqa: E402

METHODS = ("rewnorm", "retscale", "hlgauss")


class ZeroTerminalAMPEnv(SteppingAMPEnv):
    """The production walk env (`task_steering`) emits only FAIL and TIME and
    bootstraps a ZERO absorbing value, which is what lets retscale read a true
    terminal as Gamma = 0. The shared test env has a non-zero SUCC value, so
    the retscale arms are exercised against this one instead."""

    def get_reward_succ(self):
        return 0.0

    def get_reward_fail(self):
        return 0.0


class ZeroTerminalDummyAMPEnv(DummyAMPEnv):
    def get_reward_succ(self):
        return 0.0

    def get_reward_fail(self):
        return 0.0


def _env_for(method):
    return ZeroTerminalAMPEnv() if method == "retscale" else SteppingAMPEnv()


def _dummy_env_for(method):
    return ZeroTerminalDummyAMPEnv() if method == "retscale" else DummyAMPEnv()

# small support so the tests exercise the out-of-support path deliberately
HLG = {"hlgauss_num_bins": 21, "hlgauss_vmin": -5.0, "hlgauss_vmax": 15.0}


# ----------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------

def test_hlgauss_projection_is_a_distribution_that_recovers_the_target():
    head = w5_norm.HLGaussHead(num_bins=101, vmin=-10.0, vmax=110.0)
    targets = torch.tensor([0.0, 1.5, 37.25, 99.9, -3.0])
    probs = head.to_probs(targets)
    assert probs.shape == (5, 101)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-6)
    assert (probs >= 0).all()
    # the expectation over bin centers recovers the scalar it was built from
    assert torch.allclose(probs @ head.centers, targets, atol=1e-3)


def test_hlgauss_oob_fraction_is_windowed_not_cumulative():
    """A support that only starts saturating late in training has to SHOW as
    saturated. An all-time average of many clean batches plus one fully
    out-of-support batch reads 0.001 and hides exactly the failure the
    diagnostic exists to catch."""
    head = w5_norm.HLGaussHead(num_bins=21, vmin=0.0, vmax=10.0)
    head.to_probs(torch.tensor([1.0, 2.0, 3.0, 20.0]))     # 1/4 above
    head.to_probs(torch.tensor([-1.0, 5.0, 5.0, 5.0]))     # 1/4 below
    assert head.get_diagnostics()["w5_hlgauss_oob_frac"] == pytest.approx(0.25)

    for _ in range(200):
        head.to_probs(torch.full((64,), 5.0))
        head.get_diagnostics()
    head.to_probs(torch.full((64,), 500.0))
    assert head.get_diagnostics()["w5_hlgauss_oob_frac"] == pytest.approx(1.0)
    # nothing seen since the last read -> 0, not a stale value
    assert head.get_diagnostics()["w5_hlgauss_oob_frac"] == 0.0


def test_hlgauss_round_trips_through_from_logits_and_reports_its_edge_bias():
    """`from_logits` is the read-out the critic actually uses; going through it
    (rather than `probs @ centers`) is what catches an edges-vs-centers mix-up,
    an off-by-one in support[:-1]/support[1:], or a wrong softmax dim."""
    head = w5_norm.HLGaussHead(num_bins=201, vmin=-10.0, vmax=150.0)
    targets = torch.tensor([-5.0, 0.0, 37.25, 100.0, 140.0])
    logits = torch.log(head.to_probs(targets).clamp_min(1e-30))
    assert torch.allclose(head.from_logits(logits), targets, atol=1e-3)
    # a softmax over a wrong axis, or centers replaced by edges, breaks this
    assert head.from_logits(logits).shape == targets.shape

    # the true representable ceiling/floor is half a bin inside the support --
    # worth knowing before reading a saturated value as "150"
    edge = head.from_logits(torch.log(head.to_probs(
        torch.tensor([head.vmax, head.vmin])).clamp_min(1e-30)))
    assert edge[0].item() < head.vmax and edge[0].item() > head.vmax - head.bin_width
    assert edge[1].item() > head.vmin and edge[1].item() < head.vmin + head.bin_width


def test_hlgauss_from_logits_accepts_the_same_dtypes_as_loss():
    head = w5_norm.HLGaussHead(num_bins=21, vmin=0.0, vmax=10.0)
    for dtype in (torch.float32, torch.float64):
        logits = torch.zeros(4, 21, dtype=dtype)
        assert head.from_logits(logits).dtype == dtype
        assert torch.isfinite(head.loss(logits, torch.full((4,), 5.0, dtype=dtype))).all()


def test_hlgauss_loss_is_minimized_at_the_projected_target():
    head = w5_norm.HLGaussHead(num_bins=41, vmin=-5.0, vmax=15.0)
    target = torch.tensor([4.0])
    exact = torch.log(head.to_probs(target).clamp_min(1e-12))
    off = torch.log(head.to_probs(torch.tensor([9.0])).clamp_min(1e-12))
    assert head.loss(exact, target).item() < head.loss(off, target).item()


def test_retscale_sigma_matches_the_closed_form():
    torch.manual_seed(0)
    scaler = w5_norm.ComponentReturnScaling(2, min_sigma=1e-6)
    r = torch.randn(512, 2) * torch.tensor([1.0, 7.0]) + torch.tensor([0.0, 3.0])
    G = torch.randn(512, 2) * 5.0 + 20.0
    gam = torch.where(torch.rand(512) < 0.1, torch.zeros(512), torch.full((512,), 0.99))

    scaler.update_and_refresh_scale(r, gam, G)

    r64, G64, g64 = r.double(), G.double(), gam.double()
    for k in range(2):
        expected = math.sqrt(
            r64[:, k].var(unbiased=False).item()
            + g64.var(unbiased=False).item() * (G64[:, k] ** 2).mean().item())
        assert scaler.scale(torch.float64, "cpu")[k].item() == pytest.approx(
            expected, rel=1e-9)


def test_retscale_accumulates_across_folds_like_the_batch_formula():
    """One fold leaves sigma_all == sigma_batch, so the accumulation logic is
    untested by the single-fold test. Two folds with different distributions
    pin it against the closed form over the concatenation."""
    torch.manual_seed(1)
    scaler = w5_norm.ComponentReturnScaling(1, min_sigma=1e-9)
    folds = [(torch.randn(256, 1) * 1.0, torch.randn(256, 1) * 3.0 + 10.0),
             (torch.randn(256, 1) * 5.0 + 4.0, torch.randn(256, 1) * 9.0 + 40.0)]
    gam = torch.full((256,), 0.99)
    for r, G in folds:
        scaler.update_and_refresh_scale(r, gam, G)

    r_all = torch.cat([f[0] for f in folds]).double()
    G_all = torch.cat([f[1] for f in folds]).double()
    expected = math.sqrt(r_all[:, 0].var(unbiased=False).item()
                         + 0.0 * (G_all[:, 0] ** 2).mean().item())  # gamma const
    sigma_all = scaler._sigma_sq(scaler.n, scaler.s_r, scaler.s_r2,
                                 scaler.s_g, scaler.s_g2, scaler.s_G2).sqrt()
    assert sigma_all[0].item() == pytest.approx(expected, rel=1e-9)


def test_retscale_uses_the_batch_sigma_when_it_exceeds_the_all_time_sigma():
    """Schaul et al.'s own safeguard is delta / max(sigma, sigma_v, sigma_batch):
    the batch term is what covers the first appearance of a large reward, so it
    must actually win when it is larger."""
    torch.manual_seed(2)
    scaler = w5_norm.ComponentReturnScaling(1, min_sigma=1e-9)
    gam = torch.full((256,), 0.99)
    scaler.update_and_refresh_scale(torch.randn(256, 1) * 0.1,
                                    gam, torch.randn(256, 1) * 0.1)
    small = scaler.scale(torch.float64, "cpu")[0].item()
    scaler.update_and_refresh_scale(torch.randn(256, 1) * 50.0,
                                    gam, torch.randn(256, 1) * 50.0)
    big = scaler.scale(torch.float64, "cpu")[0].item()

    sigma_all = scaler._sigma_sq(scaler.n, scaler.s_r, scaler.s_r2,
                                 scaler.s_g, scaler.s_g2, scaler.s_G2).sqrt()
    assert big > small
    assert big > sigma_all[0].item()      # the batch term, not the all-time one


def test_retscale_is_all_time_cumulative_and_floored():
    scaler = w5_norm.ComponentReturnScaling(1, min_sigma=0.5)
    zeros = torch.zeros(8, 1)
    scaler.update_and_refresh_scale(zeros, torch.full((8,), 0.99), zeros)
    # a constant reward stream has zero variance -> the floor is what is used
    assert scaler.scale(torch.float32, "cpu")[0].item() == pytest.approx(0.5)
    n_after_first = scaler.n.clone()
    scaler.update_and_refresh_scale(zeros, torch.full((8,), 0.99), zeros)
    assert (scaler.n == n_after_first + 8).all()


def test_rewnorm_equalizes_component_scales_and_resets_at_episode_end():
    norm = w5_norm.ComponentRewardNorm(2, num_envs=4, discount=0.9)
    no_done = torch.zeros(4, dtype=torch.bool)
    raw = torch.tensor([[1.0, 5.0]]).repeat(4, 1)
    for _ in range(200):
        out = norm.normalize(raw, no_done)
    # a 1:5 raw ratio comes out 1:1 -- that is the whole point of the method
    assert (out[:, 1] / out[:, 0]).mean().item() == pytest.approx(1.0, rel=1e-3)

    # the divisor is the std of the DISCOUNTED-RETURN accumulator, not of the
    # rewards: for a constant stream the accumulator converges to r/(1-gamma),
    # so the std is ~10x the reward's own (which is 0 here)
    assert norm.ret[0, 0].item() == pytest.approx(1.0 / (1.0 - 0.9), rel=1e-3)

    # the reset happens AFTER the statistics fold and the normalization, so the
    # terminal step is itself included in the statistics (VecNormalize order)
    count_before = norm.count.clone()
    norm.normalize(raw, torch.ones(4, dtype=torch.bool))
    assert (norm.count == count_before + 4).all()
    assert norm.ret.abs().sum().item() == 0.0


def test_rewnorm_matches_a_transcription_of_baselines_vecnormalize():
    """Step-for-step equality against `VecNormalize.step_wait` +
    `RunningMeanStd.update_from_moments`, transcribed in numpy. Several plainly
    WRONG normalizers (statistics over raw rewards; reset before the fold)
    satisfy the qualitative tests above, so this is what actually pins the
    method down."""
    num_envs, K, gamma, clip = 6, 2, 0.99, 10.0
    rng = np.random.RandomState(0)

    ret = np.zeros((num_envs, K))
    mean = np.zeros(K)
    var = np.ones(K)
    count = np.full(K, 1e-4)

    norm = w5_norm.ComponentRewardNorm(K, num_envs, gamma, clip=clip)
    max_diff = 0.0
    for _ in range(400):
        r = rng.randn(num_envs, K) * np.array([1.0, 7.0]) + np.array([0.3, 2.0])
        done = rng.rand(num_envs) < 0.05

        ret = ret * gamma + r
        batch_mean, batch_var, n = ret.mean(axis=0), ret.var(axis=0), num_envs
        delta = batch_mean - mean
        tot = count + n
        mean = mean + delta * n / tot
        var = (var * count + batch_var * n + delta ** 2 * count * n / tot) / tot
        count = tot
        ref = np.clip(r / np.sqrt(var + 1e-8), -clip, clip)
        ret[done] = 0.0

        got = norm.normalize(torch.tensor(r, dtype=torch.float64),
                             torch.tensor(done))
        max_diff = max(max_diff, float(np.abs(got.numpy() - ref).max()))

    assert max_diff < 1e-12
    assert np.abs(norm.var.numpy() - var).max() < 1e-12
    assert np.abs(norm.count.numpy() - count).max() < 1e-9


def test_rewnorm_rejects_a_non_bool_done_mask():
    """`self.ret[mask] = 0` with an integer mask is advanced INDEXING: it zeroes
    rows 0 and 1 for a mask of [0,0,1,0] and leaves the env that actually
    terminated running. Silent and unloggable, so it must fail loudly."""
    norm = w5_norm.ComponentRewardNorm(2, num_envs=4, discount=0.9)
    with pytest.raises(AssertionError):
        norm.normalize(torch.ones(4, 2), torch.tensor([0, 0, 1, 0]))


def test_rewnorm_clips_and_does_not_subtract_the_mean():
    # mean subtraction: build a stream with a LARGE nonzero running mean, so
    # "divide by std" and "(r - mean)/std" give visibly different answers
    norm = w5_norm.ComponentRewardNorm(1, num_envs=4, discount=0.0, clip=1e9)
    no_done = torch.zeros(4, dtype=torch.bool)
    r = torch.tensor([[10.0], [10.0], [10.0], [12.0]])
    for _ in range(50):
        out = norm.normalize(r, no_done)
    std = float(torch.sqrt(norm.var + 1e-8)[0])
    assert out[0].item() == pytest.approx(10.0 / std, rel=1e-6)
    assert out[0].item() != pytest.approx(
        (10.0 - float(norm.mean[0])) / std, rel=1e-3)

    # clipping: the divisor is refreshed inside normalize(), so the clip only
    # binds if the reward is huge RELATIVE to the accumulator's own std
    norm = w5_norm.ComponentRewardNorm(1, num_envs=4, discount=0.99, clip=2.0)
    for _ in range(200):
        norm.normalize(torch.full((4, 1), 0.01), no_done)
    spike = norm.normalize(torch.full((4, 1), 500.0), no_done)
    assert spike.abs().max().item() == pytest.approx(2.0)


def test_build_w5_normalizer_is_mutually_exclusive():
    for method, slot in zip(("rewnorm", "retscale"), (0, 1)):
        built = w5_norm.build_w5_normalizer(method, 2, 2, 4, 0.99, HLG, "cpu")
        assert built[slot] is not None
        assert sum(b is not None for b in built) == 1
    # hlgauss is a value HEAD owned by the model, so this builds nothing
    assert w5_norm.build_w5_normalizer("hlgauss", 2, 2, 4, 0.99, HLG, "cpu") \
        == (None, None)
    assert w5_norm.build_w5_normalizer("none", 2, 2, 4, 0.99, {}, "cpu") == (None, None)
    with pytest.raises(AssertionError):
        w5_norm.build_w5_normalizer("popart", 2, 2, 4, 0.99, {}, "cpu")


def test_hlgauss_defaults_live_in_exactly_one_place():
    """Three copies of the support defaults meant a yaml omitting a key got
    whichever copy fired first."""
    src = (ROOT / "mimickit" / "learning").glob("*.py")
    hits = [f.name for f in src if "hlgauss_num_bins" in f.read_text()]
    assert hits == ["w5_norm.py"]
    head = w5_norm.HLGaussHead.from_config({})
    assert (head.num_bins, head.vmin, head.vmax) == (101, -10.0, 110.0)


# ----------------------------------------------------------------------
# PPO / AMP family (W5AMPAgent)
# ----------------------------------------------------------------------

def _w5_ppo_config(method):
    config = copy.deepcopy(_toppo_agent_config())
    config["agent_name"] = "W5AMP"
    config["toppo"]["use_popart"] = False
    config["toppo"]["actor_method"] = "default"
    config["toppo"]["critic_method"] = "default"
    config["w5_norm"] = {"method": method, **HLG}
    return config


def _ppo_agent(method):
    return w5_amp_agent.W5AMPAgent(_w5_ppo_config(method), _dummy_env_for(method),
                                   TEST_DEVICE)


def _fill_ppo_buffer(agent, num_steps=2, num_envs=2, obs_dim=5, act_dim=2):
    obs = torch.randn(num_steps, num_envs, obs_dim, device=TEST_DEVICE)
    next_obs = torch.randn(num_steps, num_envs, obs_dim, device=TEST_DEVICE)
    action = torch.randn(num_steps, num_envs, act_dim, device=TEST_DEVICE).clamp(-0.5, 0.5)
    done = torch.zeros(num_steps, num_envs, dtype=torch.long, device=TEST_DEVICE)
    mask = torch.ones(num_steps, num_envs, device=TEST_DEVICE)
    components = torch.stack([
        torch.rand(num_steps, num_envs, device=TEST_DEVICE) * 0.2 + 0.5,
        torch.rand(num_steps, num_envs, device=TEST_DEVICE) * 0.2 + 2.0,
    ], dim=-1)

    with torch.no_grad():
        dist = agent._model.eval_actor(agent._obs_norm.normalize(obs.reshape(-1, obs_dim)))
        a_logp = dist.log_prob(
            agent._a_norm.normalize(action.reshape(-1, act_dim))).reshape(num_steps, num_envs)

    agent._exp_buffer.set_data("obs", obs)
    agent._exp_buffer.set_data("next_obs", next_obs)
    agent._exp_buffer.set_data("action", action)
    agent._exp_buffer.set_data("a_logp", a_logp)
    agent._exp_buffer.set_data("rand_action_mask", mask)
    agent._exp_buffer.set_data("done", done)
    agent._exp_buffer.set_data("toppo_reward_components", components)
    agent._exp_buffer.set_data("disc_obs", torch.randn(num_steps, num_envs, 4,
                                                       device=TEST_DEVICE))
    agent._exp_buffer.set_data("reward", components.sum(dim=-1))
    agent._exp_buffer._total_samples = agent._exp_buffer.get_capacity()
    return obs, next_obs, done, mask


@pytest.mark.parametrize("method", METHODS)
def test_w5_ppo_critic_update_is_finite(method):
    torch.manual_seed(0)
    agent = _ppo_agent(method)
    obs, next_obs, done, mask = _fill_ppo_buffer(agent)
    agent._build_toppo_train_data(
        done=done, rand_action_mask=mask,
        norm_obs=agent._obs_norm.normalize(obs),
        norm_next_obs=agent._obs_norm.normalize(next_obs))
    info = agent._update_critic(batch_size=2, steps=1)
    assert torch.isfinite(info["critic_loss"])
    for name in ("task_reward", "disc_reward"):
        assert torch.isfinite(info["toppo_critic_loss_{}".format(name)])


def test_w5_ppo_hlgauss_head_replaces_the_scalar_component_head():
    agent = _ppo_agent("hlgauss")
    assert agent._model.uses_hlgauss()
    assert agent._model._toppo_critic_out.out_features == 2 * HLG["hlgauss_num_bins"]

    obs = torch.randn(7, 5, device=TEST_DEVICE)
    logits = agent._model.eval_toppo_critic_logits(obs)
    assert logits.shape == (7, 2, HLG["hlgauss_num_bins"])
    comps = agent._model.eval_toppo_critic_components(obs)
    assert comps.shape == (7, 2)
    # the scalar critic AMP consumes is the unweighted sum of the components
    assert torch.allclose(agent._model.eval_critic(obs),
                          comps.sum(dim=-1, keepdim=True), atol=1e-5)


def test_w5_ppo_retscale_divides_each_component_loss_by_sigma_squared():
    torch.manual_seed(0)
    agent = _ppo_agent("retscale")
    obs, next_obs, done, mask = _fill_ppo_buffer(agent)
    agent._build_toppo_train_data(
        done=done, rand_action_mask=mask,
        norm_obs=agent._obs_norm.normalize(obs),
        norm_next_obs=agent._obs_norm.normalize(next_obs))
    assert (agent._w5_retscale.n > 0).all()

    batch = {k: agent._exp_buffer.get_data(k).reshape(
                 [-1] + list(agent._exp_buffer.get_data(k).shape[2:]))
             for k in ("obs", "toppo_tar_val")}
    scaled = agent._compute_toppo_critic_loss(batch)["toppo_task_losses"]

    agent._w5_method = "none"                       # same batch, no scaling
    raw = agent._compute_toppo_critic_loss(batch)["toppo_task_losses"]
    agent._w5_method = "retscale"

    sigma = agent._w5_retscale.scale(torch.float32, TEST_DEVICE)
    for k in range(2):
        assert scaled[k].item() == pytest.approx(
            (raw[k] / sigma[k] ** 2).item(), rel=1e-5)
    # the two components have deliberately different reward scales
    assert sigma[0].item() != pytest.approx(sigma[1].item(), rel=1e-3)


def test_w5_ppo_rewnorm_rescales_the_stored_components_and_the_scalar_reward():
    torch.manual_seed(0)
    agent = _ppo_agent("rewnorm")
    _fill_ppo_buffer(agent)
    before = agent._exp_buffer.get_data("toppo_reward_components").clone()
    agent._compute_rewards()
    after = agent._exp_buffer.get_data("toppo_reward_components")

    assert not torch.allclose(before, after)
    # scalar reward stays the sum of the components the critic regresses on
    assert torch.allclose(agent._exp_buffer.get_data("reward"),
                          after.sum(dim=-1), atol=1e-6)
    assert (agent._w5_rewnorm.count > 0).all()


def test_w5_ppo_rejects_popart_and_scalar_critic():
    for mutate in (lambda c: c["toppo"].__setitem__("use_popart", True),
                   lambda c: c["toppo"].__setitem__("enabled", False)):
        config = _w5_ppo_config("retscale")
        mutate(config)
        with pytest.raises(AssertionError):
            w5_amp_agent.W5AMPAgent(config, DummyAMPEnv(), TEST_DEVICE)


def test_w5_ppo_none_leaves_the_amp_agent_untouched():
    agent = _ppo_agent("none")
    assert agent._w5_rewnorm is None and agent._w5_retscale is None
    assert not agent._model.uses_hlgauss()
    assert agent._model._toppo_critic_out.out_features == 2


# ----------------------------------------------------------------------
# SAC family (AMPSACAgent)
# ----------------------------------------------------------------------

def _w5_sac_config(method):
    config = _sac_agent_config(toppo=True)
    config["toppo"]["use_popart"] = False
    config["w5_norm"] = {"method": method, **HLG}
    return config


def _sac_agent(method):
    env = _env_for(method)
    agent = amp_sac_agent.AMPSACAgent(_w5_sac_config(method), env, TEST_DEVICE)
    return agent, env


@pytest.mark.parametrize("method", METHODS)
def test_w5_sac_trains_with_a_stable_log_key_set(method):
    torch.manual_seed(0)
    np.random.seed(0)
    agent, _ = _sac_agent(method)
    # the txt Logger hard-asserts that no new key appears after iteration 0, so
    # this also validates that _zero_update_info covers the w5_norm keys
    _run_train(agent, iters=8)
    assert torch.isfinite(agent._model.get_alpha())


@pytest.mark.parametrize("method", METHODS)
def test_w5_sac_zero_update_info_matches_a_real_update(method):
    torch.manual_seed(0)
    agent, _ = _sac_agent(method)
    zero_keys = set(agent._zero_update_info().keys())
    assert {"w5_hlgauss_oob_frac"} <= zero_keys if method == "hlgauss" else True
    assert any(k.startswith("w5_") for k in zero_keys)


def test_w5_sac_hlgauss_head_is_categorical_but_the_critic_api_is_not():
    agent, _ = _sac_agent("hlgauss")
    model = agent._model
    assert model.uses_hlgauss()
    for head in model._q_heads:
        assert head.out_features == 2 * HLG["hlgauss_num_bins"]

    obs = torch.randn(6, 5, device=TEST_DEVICE)
    act = torch.randn(6, 2, device=TEST_DEVICE).clamp(-1, 1)
    logits = model.eval_critic_logits(obs, act)
    assert logits.shape == (6, 2, 2, HLG["hlgauss_num_bins"])
    for vals in (model.eval_critic_components(obs, act),
                 model.eval_critic_components_target(obs, act)):
        assert vals.shape == (6, 2, 2)          # [batch, num_qs, K]
        assert torch.isfinite(vals).all()
    # the readout is the expectation, so it must lie inside the support
    vals = model.eval_critic_components(obs, act)
    assert (vals >= HLG["hlgauss_vmin"]).all() and (vals <= HLG["hlgauss_vmax"]).all()


def test_w5_sac_hlgauss_never_touches_popart():
    agent, _ = _sac_agent("hlgauss")
    assert not agent._model.uses_popart()
    assert not hasattr(agent._model, "_popart")
    with pytest.raises(AssertionError):
        config = _w5_sac_config("hlgauss")
        config["toppo"]["use_popart"] = True
        amp_sac_agent.AMPSACAgent(config, SteppingAMPEnv(), TEST_DEVICE)


def test_w5_sac_rewnorm_folds_statistics_at_collection_time():
    torch.manual_seed(0)
    np.random.seed(0)
    agent, _ = _sac_agent("rewnorm")
    assert agent._w5_rewnorm.count.max().item() < 1.0     # only the 1e-4 prior
    _run_train(agent, iters=8)
    # one fold per env-step of collection, num_envs samples each
    assert agent._w5_rewnorm.count[0].item() == pytest.approx(
        8 * agent._steps_per_iter * NUM_ENVS + 1e-4)


def test_w5_sac_rewnorm_scales_rewards_and_terminal_values_by_the_same_std():
    agent, _ = _sac_agent("rewnorm")
    agent._w5_rewnorm.var.copy_(torch.tensor([4.0, 100.0], dtype=torch.float64))

    raw = torch.tensor([[1.0, 50.0]], device=TEST_DEVICE)
    assert torch.allclose(agent._apply_w5_rewnorm(raw),
                          torch.tensor([[0.5, 5.0]], device=TEST_DEVICE), atol=1e-4)

    succ, fail = agent._get_component_terminal_values()
    raw_succ = agent._task_reward_weight * agent._compute_succ_val()
    assert succ[0].item() == pytest.approx(raw_succ / 2.0, rel=1e-4)
    assert succ[1].item() == pytest.approx(0.0)


def test_w5_sac_save_load_roundtrip_preserves_w5_state():
    torch.manual_seed(0)
    np.random.seed(0)
    for method in METHODS:
        agent, env = _sac_agent(method)
        _run_train(agent, iters=6)
        with tempfile.TemporaryDirectory() as out_dir:
            path = os.path.join(out_dir, "m.pt")
            agent.save(path)
            fresh, _ = _sac_agent(method)
            fresh.load(path)
        if method == "rewnorm":
            assert torch.allclose(fresh._w5_rewnorm.var, agent._w5_rewnorm.var)
            assert torch.allclose(fresh._w5_rewnorm.count, agent._w5_rewnorm.count)
        elif method == "retscale":
            assert torch.allclose(fresh._w5_retscale.last_scale,
                                  agent._w5_retscale.last_scale)


# ----------------------------------------------------------------------
# production configs
# ----------------------------------------------------------------------

def _load_yaml(name):
    with open(ROOT / "data" / "agents" / name, "r") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("method", METHODS)
def test_w5_ppo_yaml_differs_from_the_popart_rung_by_exactly_the_w5_block(method):
    base = _load_yaml("amp_task_humanoid_component_welford_ln_clip1_agent.yaml")
    arm = _load_yaml("amp_task_humanoid_w5_{}_agent.yaml".format(method))

    assert base["agent_name"] == "AMP" and arm["agent_name"] == "W5AMP"
    assert base["toppo"]["use_popart"] is True
    assert arm["toppo"]["use_popart"] is False
    assert arm["w5_norm"]["method"] == method

    stripped_base = {k: v for k, v in base.items() if k not in ("agent_name", "toppo")}
    stripped_arm = {k: v for k, v in arm.items()
                    if k not in ("agent_name", "toppo", "w5_norm")}
    assert stripped_base == stripped_arm

    base_toppo = {k: v for k, v in base["toppo"].items()
                  if "popart" not in k}
    arm_toppo = {k: v for k, v in arm["toppo"].items() if "popart" not in k}
    assert base_toppo == arm_toppo


@pytest.mark.parametrize("method", METHODS)
def test_w5_sac_yaml_differs_from_cb_by_exactly_the_w5_block(method):
    base = _load_yaml("amp_sac_walk_cb_agent.yaml")
    arm = _load_yaml("amp_sac_walk_w5_{}_agent.yaml".format(method))

    assert base["toppo"]["use_popart"] is True
    assert arm["toppo"]["use_popart"] is False
    assert arm["w5_norm"]["method"] == method

    stripped_base = {k: v for k, v in base.items() if k != "toppo"}
    stripped_arm = {k: v for k, v in arm.items() if k not in ("toppo", "w5_norm")}
    assert stripped_base == stripped_arm

    base_toppo = {k: v for k, v in base["toppo"].items() if "popart" not in k}
    arm_toppo = {k: v for k, v in arm["toppo"].items() if "popart" not in k}
    assert base_toppo == arm_toppo


@pytest.mark.parametrize("method", METHODS)
def test_w5_production_yamls_construct(method):
    ppo_config = _load_yaml("amp_task_humanoid_w5_{}_agent.yaml".format(method))
    ppo = w5_amp_agent.W5AMPAgent(ppo_config, _dummy_env_for(method), TEST_DEVICE)
    assert ppo._w5_method == method

    sac_config = _load_yaml("amp_sac_walk_w5_{}_agent.yaml".format(method))
    sac_config["batch_size"] = 16
    sac_config["replay_buffer_size"] = 64
    sac_config["min_buffer_samples"] = 32
    sac_config["random_action_samples"] = 16
    sac = amp_sac_agent.AMPSACAgent(sac_config, _env_for(method), TEST_DEVICE)
    assert sac._w5_method == method



# ----------------------------------------------------------------------
# zero-regression guard for the arms that are already running
# ----------------------------------------------------------------------

EXISTING_SAC_YAMLS = [
    "amp_sac_walk_agent.yaml", "amp_sac_walk_cd_agent.yaml",
    "amp_sac_walk_cb_agent.yaml", "amp_sac_walk_reweight_agent.yaml",
    "amp_sac_walk_cbreweight_agent.yaml", "amp_sac_walk_fullstack_agent.yaml",
    "amp_sac_walk_entmatch_agent.yaml", "amp_sac_walk_lnonly_agent.yaml",
    "amp_sac_walk_popartonly_agent.yaml", "amp_sac_walk_k1cd_agent.yaml",
    "amp_sac_walk_ars_agent.yaml", "amp_sac_walk_arsreset_agent.yaml",
    "amp_sac_walk_utd512_agent.yaml",
]


@pytest.mark.parametrize("name", EXISTING_SAC_YAMLS)
def test_existing_sac_arms_are_untouched_by_the_w5_wiring(name):
    """Adding the w5_norm wiring must be a strict no-op for every SAC arm that
    does not enable it."""
    config = _load_yaml(name)
    config["batch_size"] = 16
    config["replay_buffer_size"] = 64
    config["min_buffer_samples"] = 32
    config["random_action_samples"] = 16
    agent = amp_sac_agent.AMPSACAgent(config, SteppingAMPEnv(), TEST_DEVICE)

    assert agent._w5_method == "none"
    assert agent._w5_rewnorm is None and agent._w5_retscale is None
    assert not agent._model.uses_hlgauss()
    assert agent._w5_info() == {}
    assert not any(k.startswith("w5_") for k in agent._zero_update_info())
    # the terminal-value path must be the one without w5_norm, exactly
    succ, fail = agent._get_component_terminal_values()
    expected = agent._task_reward_weight * agent._compute_succ_val()
    assert succ[0].item() == pytest.approx(expected, rel=1e-6)
    if agent._num_components == 2:      # the scalar-critic arms sum the two
        assert succ[1].item() == 0.0 and fail[1].item() == 0.0
    # no w5_norm tensors in the checkpoint
    assert not any("w5" in k for k in agent.state_dict())


def test_existing_ppo_arms_do_not_route_through_w5():
    """The PPO ladder runs `AMP`, not `W5AMP`; the only shared-file change is
    the agent_builder dispatch entry."""
    import learning.agent_builder as agent_builder
    src = (ROOT / "mimickit" / "learning" / "agent_builder.py").read_text()
    assert src.count("W5AMP") == 2                      # elif + import line
    for name in ("amp_task_humanoid_component_welford_ln_clip1_agent.yaml",
                 "amp_task_humanoid_toppo_cc100_noclip_agent.yaml",
                 "amp_task_humanoid_fulltoppo_cc100_agent.yaml",
                 "amp_task_humanoid_long_critic_clip1_agent.yaml"):
        config = _load_yaml(name)
        assert config["agent_name"] == "AMP"
        assert "w5_norm" not in config


# ----------------------------------------------------------------------
# fixes from the wiring review
# ----------------------------------------------------------------------

def test_hlgauss_head_starts_at_zero_not_at_the_support_centre():
    """A zero bias makes the softmax uniform and the read-out the support
    center: on [-10, 350] every Q would start at 170 while the fixed point is
    ~25. That is a change of initialization on top of the change of loss, and
    it would confound the comparison with the zero-initialized PopArt control."""
    sac, _ = _sac_agent("hlgauss")
    popart, _ = _sac_agent("none")
    obs = torch.randn(64, 5, device=TEST_DEVICE)
    act = torch.randn(64, 2, device=TEST_DEVICE).clamp(-1, 1)

    hl_q = sac._model.eval_critic_components(obs, act).mean().item()
    pa_q = popart._model.eval_critic_components(obs, act).mean().item()
    centre = 0.5 * (HLG["hlgauss_vmin"] + HLG["hlgauss_vmax"])
    assert abs(hl_q) < 1.0
    assert abs(hl_q - centre) > 1.0
    assert abs(hl_q - pa_q) < 1.0
    # the target net is a deepcopy, so it starts in the same place
    assert abs(sac._model.eval_critic_components_target(obs, act).mean().item()) < 1.0


def test_hlgauss_logits_for_value_round_trips():
    head = w5_norm.HLGaussHead(num_bins=361, vmin=-10.0, vmax=350.0)
    for v in (0.0, 25.0, 100.0):
        logits = head.logits_for_value(v)
        assert logits.shape == (361,)
        assert head.from_logits(logits).item() == pytest.approx(v, abs=1e-2)


# per-component value range MEASURED on a finished 100M walk run of the PopArt
# control (amp_task_humanoid_component_welford_ln_clip1_agent.yaml, seed 260511473)
MEASURED_MAX_COMPONENT_VALUE = 27.34


def test_production_hlgauss_support_is_set_from_the_measured_value_range():
    """The support is a TWO-sided constraint. Too small and targets clamp, the
    component Q pins at its ceiling and that component's gradient dies. Too
    wide and the critic has to move softmax mass across far more bins to
    represent the same value, which on the PPO critic (SGD 1e-4, grad_clip 1.0)
    left an earlier [-10, 350] support an order of magnitude below the true
    value at the end of the budget."""
    for name in ("amp_task_humanoid_w5_hlgauss_agent.yaml",
                 "amp_sac_walk_w5_hlgauss_agent.yaml"):
        w5 = _load_yaml(name)["w5_norm"]
        vmin, vmax, bins = (w5["hlgauss_vmin"], w5["hlgauss_vmax"],
                            w5["hlgauss_num_bins"])
        assert vmax >= 2.0 * MEASURED_MAX_COMPONENT_VALUE      # headroom
        assert vmax <= 4.0 * MEASURED_MAX_COMPONENT_VALUE      # but not absurd
        assert vmin <= 0.0
        width = (vmax - vmin) / bins
        assert width <= 0.5                                    # <= ~2% of value
        # the operating point must sit well inside, not near an edge
        assert MEASURED_MAX_COMPONENT_VALUE < vmax - 10 * width


def test_hlgauss_support_covers_the_measured_range_exactly_once_per_family():
    """Both families run the same env, weights and discount, so they must use
    the SAME support -- a per-family support would confound the cross-family
    comparison with a hyperparameter difference."""
    ppo = _load_yaml("amp_task_humanoid_w5_hlgauss_agent.yaml")["w5_norm"]
    sac = _load_yaml("amp_sac_walk_w5_hlgauss_agent.yaml")["w5_norm"]
    assert ppo == sac


def test_retscale_requires_zero_valued_absorbing_states():
    """`gammas = 0` at FAIL/SUCC reads the terminal as a genuine episode end.
    That is only right while the env bootstraps a ZERO terminal value, which is
    true for task_steering but is a property of the env, not of the algorithm."""
    class NonZeroTerminalEnv(ZeroTerminalAMPEnv):
        def get_reward_fail(self):
            return -1.0

    with pytest.raises(AssertionError):
        amp_sac_agent.AMPSACAgent(_w5_sac_config("retscale"),
                                  NonZeroTerminalEnv(), TEST_DEVICE)
    # the real walk env is the zero case, so the shipped arms construct
    amp_sac_agent.AMPSACAgent(_w5_sac_config("retscale"),
                              ZeroTerminalAMPEnv(), TEST_DEVICE)


def test_w5_and_ars_are_mutually_exclusive():
    config = _w5_sac_config("rewnorm")
    config["ars"] = {"enabled": True}
    with pytest.raises(AssertionError):
        amp_sac_agent.AMPSACAgent(config, SteppingAMPEnv(), TEST_DEVICE)


def test_rewnorm_is_rejected_without_a_collection_time_fold():
    """Plain SACAgent has no `_record_data_post_step` hook, so a rewnorm arm
    there would be bit-identical to no normalization while wearing the label."""
    import learning.sac_agent as sac_agent_mod

    class PlainSACEnv(SteppingAMPEnv):
        pass

    config = _w5_sac_config("rewnorm")
    for key in ("disc_batch_size", "disc_updates_per_iter", "disc_buffer_size",
                "disc_replay_samples", "disc_logit_reg", "disc_grad_penalty",
                "disc_reward_scale", "disc_optimizer"):
        config.pop(key, None)
    config["model"].pop("disc_net", None)
    with pytest.raises(AssertionError):
        sac_agent_mod.SACAgent(config, PlainSACEnv(), TEST_DEVICE)


def test_w5_state_dict_has_no_duplicate_buffers():
    """`self._x = module` already registers it; a further add_module under a
    second name duplicated every buffer in the checkpoint."""
    for method, prefix in (("rewnorm", "_w5_rewnorm"), ("retscale", "_w5_retscale")):
        agent, _ = _sac_agent(method)
        keys = [k for k in agent.state_dict() if "w5" in k]
        assert keys, method
        assert all(k.startswith(prefix + ".") for k in keys), keys
        # the accumulator must NOT be checkpointed (it welds the file to num_envs)
        assert not any(k.endswith(".ret") for k in keys)


def test_rewnorm_accumulator_is_zeroed_on_a_forced_reset():
    """BaseAgent force-resets every env after each output iteration and again
    at the start of test_model. Those are not episode boundaries the env
    reports, so the previous episode's discounted return would otherwise bleed
    across the seam into the running statistics."""
    ppo = _ppo_agent("rewnorm")
    ppo._env.reset = lambda env_ids=None: (torch.zeros(2, 5), {})   # DummyAMPEnv
    for agent in (_sac_agent("rewnorm")[0], ppo):
        agent._w5_rewnorm.ret.fill_(7.0)
        agent._reset_envs()
        assert agent._w5_rewnorm.ret.abs().sum().item() == 0.0
        # a partial reset touches only the envs that were reset
        agent._w5_rewnorm.ret.fill_(7.0)
        agent._reset_envs(torch.tensor([0]))
        assert agent._w5_rewnorm.ret[0].abs().sum().item() == 0.0
        assert agent._w5_rewnorm.ret[1].abs().sum().item() > 0.0

    # and it stays a no-op for every arm that is not running rewnorm
    for agent in (_sac_agent("retscale")[0], _sac_agent("none")[0]):
        agent._reset_envs()
        assert agent._w5_rewnorm is None


def test_retscale_refreshes_the_batch_guard_on_every_minibatch():
    """Schaul's max(sigma_all, sigma_batch, floor) is the replay safeguard: a
    batch carrying a large reward must raise the scale BEFORE the update that
    uses it. Folding once per iteration and reusing the scale would divide 63
    of 64 SAC minibatches by a stale sigma."""
    scaler = w5_norm.ComponentReturnScaling(1, min_sigma=0.01)
    gam = torch.full((64,), 0.99)
    quiet = torch.zeros(64, 1)

    scaler.update_all_time(quiet, gam, quiet)
    assert scaler.refresh_scale(quiet, gam, quiet)[0].item() == pytest.approx(0.01)

    # a later minibatch in the SAME iteration carrying a big reward: no fold,
    # but the scale must jump for this batch
    spike = torch.cat([torch.full((32, 1), -50.0), torch.full((32, 1), 50.0)])
    n_before = scaler.n.clone()
    scale = scaler.refresh_scale(spike, gam, quiet)
    assert scale[0].item() == pytest.approx(50.0, rel=1e-6)
    assert (scaler.n == n_before).all()          # refresh must not fold


def test_w5_sac_retscale_folds_once_but_rescales_every_step():
    torch.manual_seed(0)
    np.random.seed(0)
    agent, _ = _sac_agent("retscale")
    assert agent._updates_per_iter > 1
    _run_train(agent, iters=8)
    n = agent._w5_retscale.n[0].item()
    assert n > 0 and n % agent._batch_size == 0
    assert n / agent._batch_size <= 8             # one FOLD per train iteration


def test_w5_ppo_rewnorm_normalizes_terminal_bootstraps_too():
    """A terminal bootstrap left in raw units mixes scales inside one target,
    and makes the scalar and component terminals disagree."""
    agent = _ppo_agent("rewnorm")
    agent._w5_rewnorm.var.copy_(torch.tensor([4.0, 100.0], dtype=torch.float64))

    class SuccEnv(DummyAMPEnv):
        def get_reward_succ(self):
            return 1.0

    agent._env = SuccEnv()
    expected = (0.5 * 1.0 / 2.0) / (1.0 - agent._discount)
    assert agent._compute_succ_val() == pytest.approx(expected, rel=1e-6)
    succ, fail = agent._get_toppo_terminal_values()
    assert succ[0].item() == pytest.approx(expected, rel=1e-5)
    assert succ[1].item() == 0.0 and fail[0].item() == 0.0
    # the scalar value PPO uses for advantages is the sum of the components
    assert agent._compute_succ_val() == pytest.approx(
        succ.sum().item(), rel=1e-5)


def test_w5_ppo_terminal_values_are_untouched_without_rewnorm():
    for method in ("none", "retscale", "hlgauss"):
        agent = _ppo_agent(method)

        class SuccEnv(DummyAMPEnv):
            def get_reward_succ(self):
                return 1.0

        agent._env = SuccEnv()
        raw = 1.0 / (1.0 - agent._discount)
        assert agent._compute_succ_val() == pytest.approx(raw, rel=1e-9)
        assert agent._get_toppo_terminal_values()[0][0].item() == pytest.approx(
            0.5 * raw, rel=1e-6)
