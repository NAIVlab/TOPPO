import sys
import types
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))

sys.modules.setdefault(
    "tensorboardX",
    types.SimpleNamespace(SummaryWriter=lambda *args, **kwargs: None),
)
sys.modules.setdefault(
    "wandb",
    types.SimpleNamespace(
        run=None,
        init=lambda *args, **kwargs: None,
        log=lambda *args, **kwargs: None,
        Video=lambda *args, **kwargs: None,
    ),
)


class _DummyVideo:
    pass


sys.modules.setdefault("util.video", types.SimpleNamespace(Video=_DummyVideo))

import learning.ppo_model as ppo_model
import learning.toppo as toppo
import learning.amp_agent as amp_agent
import learning.eval_util as eval_util
import envs.base_env as base_env


TEST_DEVICE = os.environ.get("TOPPO_TEST_DEVICE", "cpu")


class DummyEnv:
    def get_obs_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def get_action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def get_num_envs(self):
        return 2

    def get_reward_succ(self):
        return 1.0

    def get_reward_fail(self):
        return 0.0


class DummyAMPEnv(DummyEnv):
    def get_disc_obs_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)


def _toppo_model_config():
    return {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.05,
        "critic_net": "fc_2layers_128units",
        "toppo": {
            "enabled": True,
            "num_tasks": 2,
            "value_weights": [1.0, 1.0],
            "critic_layer_norm": True,
            "use_popart": True,
            "popart_method": "welford",
            "popart_min_std": 1e-2,
        },
    }


def _toppo_agent_config():
    return {
        "agent_name": "AMP",
        "model": {
            "actor_net": "fc_2layers_128units",
            "actor_init_output_scale": 0.01,
            "actor_std_type": "FIXED",
            "action_std": 0.05,
            "critic_net": "fc_2layers_128units",
            "disc_net": "fc_2layers_128units",
        },
        "actor_optimizer": {"type": "SGD", "learning_rate": 1e-4},
        "critic_optimizer": {"type": "SGD", "learning_rate": 1e-4},
        "disc_optimizer": {"type": "SGD", "learning_rate": 1e-4},
        "discount": 0.99,
        "steps_per_iter": 2,
        "iters_per_output": 1,
        "test_episodes": 0,
        "normalizer_samples": 0,
        "use_mixed_precision": False,
        "actor_epochs": 1,
        "actor_batch_size": 1,
        "critic_epochs": 1,
        "critic_batch_size": 1,
        "disc_epochs": 1,
        "disc_batch_size": 1,
        "td_lambda": 0.95,
        "ppo_clip_ratio": 0.2,
        "norm_adv_clip": 4.0,
        "action_bound_weight": 0.0,
        "action_entropy_weight": 0.0,
        "action_reg_weight": 0.0,
        "disc_buffer_size": 8,
        "disc_replay_samples": 2,
        "disc_logit_reg": 0.0,
        "disc_grad_penalty": 1.0,
        "disc_reward_scale": 1.0,
        "task_reward_weight": 0.5,
        "disc_reward_weight": 0.5,
        "toppo": {
            "enabled": True,
            "num_tasks": 2,
            "task_names": ["task_reward", "disc_reward"],
            "actor_method": "pcgrad",
            "critic_method": "fairgrad",
            "fairgrad_alpha": 1.0,
            "fairgrad_rescale": True,
            "critic_layer_norm": True,
            "use_popart": True,
            "popart_method": "welford",
            "popart_min_std": 1e-2,
        },
    }


def test_component_popart_preserves_raw_values():
    torch.manual_seed(1)
    pop = toppo.ComponentPopArt(num_components=2, min_std=1e-4).to(TEST_DEVICE)
    pop.weight.data = torch.tensor([1.2, -0.7], device=TEST_DEVICE)
    pop.bias.data = torch.tensor([0.3, 1.1], device=TEST_DEVICE)
    pop.mu.data = torch.tensor([2.0, -3.0], device=TEST_DEVICE, dtype=torch.float64)
    pop.sigma.data = torch.tensor([0.5, 4.0], device=TEST_DEVICE, dtype=torch.float64)

    values = torch.randn(32, 2, device=TEST_DEVICE)
    raw_before = pop.denormalize(pop.affine(values)).detach().clone()

    targets = torch.stack([
        torch.randn(128, device=TEST_DEVICE) * 10.0 + 50.0,
        torch.randn(128, device=TEST_DEVICE) * 0.2 - 1.0,
    ], dim=-1)
    pop.update_and_renormalize(targets)

    raw_after = pop.denormalize(pop.affine(values)).detach()
    assert torch.allclose(raw_before, raw_after, atol=1e-5)


def test_component_popart_welford_matches_exact_running_moments():
    pop = toppo.ComponentPopArt(num_components=2, min_std=1e-6, method="welford")
    first = torch.tensor([[1.0, 10.0], [3.0, 14.0]])
    second = torch.tensor([[5.0, 18.0], [7.0, 22.0], [9.0, 26.0]])

    pop.update_and_renormalize(first)
    pop.update_and_renormalize(second)

    all_targets = torch.cat([first, second], dim=0).double()
    assert torch.equal(pop.count, torch.tensor([5, 5], dtype=torch.int64))
    assert torch.allclose(pop.mu, all_targets.mean(dim=0))
    assert torch.allclose(pop.sigma, all_targets.var(dim=0, unbiased=False).sqrt())
    assert torch.equal(pop.nu, torch.ones_like(pop.nu))


def test_component_popart_ema_matches_sequential_updates():
    beta = 3e-4
    pop = toppo.ComponentPopArt(
        num_components=2,
        min_std=1e-6,
        method="ema",
        ema_decay=beta,
    )
    targets = torch.tensor(
        [[1.0, 10.0], [3.0, 14.0], [5.0, 18.0], [7.0, 22.0]],
        dtype=torch.float64,
    )

    expected_mu = torch.zeros(2, dtype=torch.float64)
    expected_nu = torch.ones(2, dtype=torch.float64)
    for target in targets:
        expected_mu = (1.0 - beta) * expected_mu + beta * target
        expected_nu = (1.0 - beta) * expected_nu + beta * target.square()

    pop.update_and_renormalize(targets)

    assert torch.allclose(pop.mu, expected_mu, atol=1e-14)
    assert torch.allclose(pop.nu, expected_nu, atol=1e-14)
    assert torch.allclose(
        pop.sigma,
        (expected_nu - expected_mu.square()).clamp_min(0.0).sqrt(),
        atol=1e-14,
    )


def test_pcgrad_and_fairgrad_combine_are_finite():
    grads = [
        torch.tensor([1.0, 0.0, 0.0], device=TEST_DEVICE),
        torch.tensor([-0.5, 1.0, 0.0], device=TEST_DEVICE),
    ]
    pc_grad, pc_diag = toppo.pcgrad_combine(grads)
    fg_grad, fg_diag = toppo.fairgrad_combine(grads, alpha=1.0, rescale=True)

    assert torch.isfinite(pc_grad).all()
    assert torch.isfinite(fg_grad).all()
    assert 0.0 <= pc_diag.conflict_rate <= 1.0
    assert torch.isfinite(fg_diag.weights).all()
    mean_norm = torch.stack(grads).mean(dim=0).norm()
    assert abs(fg_diag.combined_grad_norm_pre_rescale - 2**0.5) < 1e-6
    assert abs(fg_diag.combined_grad_norm_post_rescale - mean_norm.item()) < 1e-6
    assert abs(fg_grad.norm().item() - mean_norm.item()) < 1e-6


def test_fairgrad_raw_alpha_one_has_fixed_norm_and_rescale_preserves_direction():
    grads = [
        torch.tensor([0.04, 0.01, 0.0], device=TEST_DEVICE),
        torch.tensor([0.0, 0.2, 0.03], device=TEST_DEVICE),
    ]
    raw, raw_diag = toppo.fairgrad_combine(grads, alpha=1.0, rescale=False)
    rescaled, rescaled_diag = toppo.fairgrad_combine(grads, alpha=1.0, rescale=True)
    mean_norm = torch.stack(grads).mean(dim=0).norm()

    assert abs(raw.norm().item() - 2**0.5) < 1e-2
    assert abs(rescaled.norm().item() - mean_norm.item()) < 1e-6
    assert torch.nn.functional.cosine_similarity(raw, rescaled, dim=0) > 1.0 - 1e-6
    assert raw_diag.rescale_factor == 1.0
    assert rescaled_diag.rescale_factor < 1.0


def test_fairgrad_satisfies_kkt_and_scipy_fallback():
    grads = [
        torch.tensor([2.0, 0.0], device=TEST_DEVICE),
        torch.tensor([0.0, 0.5], device=TEST_DEVICE),
    ]
    combined, diag = toppo.fairgrad_combine(grads, alpha=1.0)
    assert torch.allclose(combined, torch.ones_like(combined), atol=1e-6)
    assert diag.residual < 1e-8

    gram = torch.tensor([[1.0, 0.2], [0.2, 2.0]], dtype=torch.float64)
    weights, converged, _ = toppo._solve_fairgrad_scipy(
        gram,
        alpha=1.0,
        x_start=torch.ones(2, dtype=torch.float64),
    )
    residual = gram @ weights - weights.reciprocal()
    assert converged
    assert torch.linalg.vector_norm(residual) < 1e-6


def test_toppo_model_component_critic_and_popart_identity():
    torch.manual_seed(2)
    model = ppo_model.PPOModel(_toppo_model_config(), DummyEnv()).to(TEST_DEVICE)
    obs = torch.randn(8, 5, device=TEST_DEVICE)

    components_before = model.eval_toppo_critic_components(obs).detach()
    scalar_before = model.eval_critic(obs).detach()

    assert components_before.shape == (8, 2)
    assert scalar_before.shape == (8, 1)
    assert torch.allclose(scalar_before.squeeze(-1), components_before.sum(dim=-1), atol=1e-6)
    assert any(isinstance(m, torch.nn.LayerNorm) for m in model._critic_layers.modules())
    assert all(
        layer.bias is None
        for layer in model._critic_layers.modules()
        if isinstance(layer, torch.nn.Linear)
    )

    targets = torch.stack([
        torch.randn(64, device=TEST_DEVICE) * 5.0 + 20.0,
        torch.randn(64, device=TEST_DEVICE) * 0.5 - 2.0,
    ], dim=-1)
    model.update_toppo_popart(targets)

    components_after = model.eval_toppo_critic_components(obs).detach()
    scalar_after = model.eval_critic(obs).detach()

    assert torch.allclose(components_before, components_after, atol=1e-4)
    assert torch.allclose(scalar_before, scalar_after, atol=1e-4)


def test_toppo_model_can_load_legacy_layer_norm_bias_architecture():
    config = _toppo_model_config()
    config["toppo"]["critic_layer_norm_biasless"] = False
    model = ppo_model.PPOModel(config, DummyEnv()).to(TEST_DEVICE)

    assert all(
        layer.bias is not None
        for layer in model._critic_layers.modules()
        if isinstance(layer, torch.nn.Linear)
    )


def test_toppo_amp_agent_cpu_update_smoke():
    torch.manual_seed(3)
    import util.mp_util as mp_util
    if mp_util.get_num_procs() == 0:
        mp_util.init(0, 1, TEST_DEVICE, None)

    agent = amp_agent.AMPAgent(_toppo_agent_config(), DummyAMPEnv(), TEST_DEVICE)

    obs = torch.randn(2, 2, 5, device=TEST_DEVICE)
    next_obs = torch.randn(2, 2, 5, device=TEST_DEVICE)
    action = torch.randn(2, 2, 2, device=TEST_DEVICE).clamp(-0.5, 0.5)
    done = torch.zeros(2, 2, dtype=torch.long, device=TEST_DEVICE)
    reward_components = torch.stack([
        torch.randn(2, 2, device=TEST_DEVICE) * 0.1 + 0.5,
        torch.randn(2, 2, device=TEST_DEVICE) * 0.1 + 0.2,
    ], dim=-1)

    flat_obs = obs.reshape(-1, 5)
    flat_action = action.reshape(-1, 2)
    with torch.no_grad():
        dist = agent._model.eval_actor(agent._obs_norm.normalize(flat_obs))
        a_logp = dist.log_prob(agent._a_norm.normalize(flat_action)).reshape(2, 2)

    agent._exp_buffer.set_data("obs", obs)
    agent._exp_buffer.set_data("next_obs", next_obs)
    agent._exp_buffer.set_data("action", action)
    agent._exp_buffer.set_data("a_logp", a_logp)
    agent._exp_buffer.set_data("rand_action_mask", torch.ones(2, 2, device=TEST_DEVICE))
    agent._exp_buffer.set_data("done", done)
    agent._exp_buffer.set_data("toppo_reward_components", reward_components)
    agent._exp_buffer._total_samples = agent._exp_buffer.get_capacity()

    norm_obs = agent._obs_norm.normalize(obs)
    norm_next_obs = agent._obs_norm.normalize(next_obs)
    agent._build_toppo_train_data(
        done=done,
        rand_action_mask=torch.ones(2, 2, device=TEST_DEVICE),
        norm_obs=norm_obs,
        norm_next_obs=norm_next_obs,
    )

    critic_info = agent._update_critic(batch_size=2, steps=1)
    actor_info = agent._update_actor(batch_size=2, num_steps=1)

    assert torch.isfinite(critic_info["critic_loss"])
    assert torch.isfinite(actor_info["actor_loss"])
    assert "toppo_critic_weight_task_reward" in critic_info
    expected_scope_fraction = (
        sum(p.numel() for p in agent._model.get_toppo_critic_shared_params())
        / sum(p.numel() for p in agent._model.get_critic_params())
    )
    assert abs(critic_info["toppo_critic_scope_fraction"].item() - expected_scope_fraction) < 1e-6
    assert "toppo_actor_conflict_rate" in actor_info


def test_toppo_amp_agent_critic_only_actor_uses_scalar_ppo():
    torch.manual_seed(4)
    import util.mp_util as mp_util
    if mp_util.get_num_procs() == 0:
        mp_util.init(0, 1, TEST_DEVICE, None)

    config = _toppo_agent_config()
    config["toppo"]["actor_components"] = False
    agent = amp_agent.AMPAgent(config, DummyAMPEnv(), TEST_DEVICE)

    obs = torch.randn(2, 2, 5, device=TEST_DEVICE)
    next_obs = torch.randn(2, 2, 5, device=TEST_DEVICE)
    action = torch.randn(2, 2, 2, device=TEST_DEVICE).clamp(-0.5, 0.5)
    done = torch.zeros(2, 2, dtype=torch.long, device=TEST_DEVICE)
    reward_components = torch.stack([
        torch.randn(2, 2, device=TEST_DEVICE) * 0.1 + 0.5,
        torch.randn(2, 2, device=TEST_DEVICE) * 0.1 + 0.2,
    ], dim=-1)
    reward = torch.sum(reward_components, dim=-1)

    flat_obs = obs.reshape(-1, 5)
    flat_action = action.reshape(-1, 2)
    with torch.no_grad():
        dist = agent._model.eval_actor(agent._obs_norm.normalize(flat_obs))
        a_logp = dist.log_prob(agent._a_norm.normalize(flat_action)).reshape(2, 2)

    agent._exp_buffer.set_data("obs", obs)
    agent._exp_buffer.set_data("next_obs", next_obs)
    agent._exp_buffer.set_data("action", action)
    agent._exp_buffer.set_data("a_logp", a_logp)
    agent._exp_buffer.set_data("rand_action_mask", torch.ones(2, 2, device=TEST_DEVICE))
    agent._exp_buffer.set_data("reward", reward)
    agent._exp_buffer.set_data("done", done)
    agent._exp_buffer.set_data("toppo_reward_components", reward_components)
    agent._exp_buffer._total_samples = agent._exp_buffer.get_capacity()

    norm_obs = agent._obs_norm.normalize(obs)
    norm_next_obs = agent._obs_norm.normalize(next_obs)
    agent._build_toppo_train_data(
        done=done,
        rand_action_mask=torch.ones(2, 2, device=TEST_DEVICE),
        norm_obs=norm_obs,
        norm_next_obs=norm_next_obs,
    )
    agent._exp_buffer.set_data("tar_val", reward)
    agent._exp_buffer.set_data("adv", torch.randn(2, 2, device=TEST_DEVICE))

    critic_info = agent._update_critic(batch_size=2, steps=1)
    actor_info = agent._update_actor(batch_size=2, num_steps=1)

    assert torch.isfinite(critic_info["critic_loss"])
    assert torch.isfinite(actor_info["actor_loss"])
    assert "toppo_critic_weight_task_reward" in critic_info
    expected_scope_fraction = (
        sum(p.numel() for p in agent._model.get_toppo_critic_shared_params())
        / sum(p.numel() for p in agent._model.get_critic_params())
    )
    assert abs(critic_info["toppo_critic_scope_fraction"].item() - expected_scope_fraction) < 1e-6
    assert "toppo_actor_conflict_rate" not in actor_info


def test_toppo_critic_scope_mean_averages_task_local_heads():
    torch.manual_seed(6)
    import util.mp_util as mp_util
    if mp_util.get_num_procs() == 0:
        mp_util.init(0, 1, TEST_DEVICE, None)

    config = _toppo_agent_config()
    config["toppo"]["actor_components"] = False
    config["toppo"]["critic_layer_norm"] = False
    config["toppo"]["use_popart"] = False
    agent = amp_agent.AMPAgent(config, DummyAMPEnv(), TEST_DEVICE)
    optimizer = agent._critic_optimizer
    params = optimizer._param_list
    shared_params = agent._model.get_toppo_critic_shared_params()
    shared_ids = {id(param) for param in shared_params}
    obs = torch.randn(7, 5, device=TEST_DEVICE)
    targets = torch.tensor([1.5, -0.25], device=TEST_DEVICE)

    def build_losses():
        values = agent._model.eval_toppo_critic_components(obs, normalized=True)
        return [
            torch.mean(torch.square(values[:, i] - targets[i]))
            for i in range(2)
        ]

    task_grads = agent._compute_toppo_task_grads(optimizer, build_losses(), params)
    mean_grad = torch.stack(task_grads).mean(dim=0)
    scoped_task_grads = [
        agent._select_toppo_flat_grad(grad, params, shared_ids)
        for grad in task_grads
    ]
    scoped_grad, _ = toppo.fairgrad_combine(
        scoped_task_grads,
        alpha=agent._toppo_fairgrad_alpha,
        rescale=agent._toppo_fairgrad_rescale,
    )
    expected = agent._replace_toppo_flat_grad(
        mean_grad,
        params,
        shared_ids,
        scoped_grad,
    )

    info = {}
    agent._toppo_optimizer_step(
        optimizer=optimizer,
        task_losses=build_losses(),
        method="fairgrad",
        diag_prefix="toppo_critic",
        info=info,
        scope_params=shared_params,
    )
    actual = toppo.flatten_param_grads(params)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert info["toppo_critic_combined_grad_norm_pre_clip"] > 0
    assert torch.allclose(
        info["toppo_critic_combined_grad_norm_pre_clip"],
        info["toppo_critic_combined_grad_norm_post_clip"],
    )
    assert torch.allclose(
        info["toppo_critic_clip_factor"],
        torch.tensor(1.0, device=TEST_DEVICE),
    )

    offset = 0
    for param in params:
        numel = param.numel()
        if id(param) not in shared_ids:
            assert torch.allclose(
                actual[offset:offset + numel],
                mean_grad[offset:offset + numel],
                atol=1e-6,
            )
        offset += numel


def test_toppo_critic_coef_scales_combined_gradient_post_combine():
    torch.manual_seed(6)
    import util.mp_util as mp_util
    if mp_util.get_num_procs() == 0:
        mp_util.init(0, 1, TEST_DEVICE, None)

    config = _toppo_agent_config()
    config["toppo"]["actor_components"] = False
    config["toppo"]["critic_layer_norm"] = False
    config["toppo"]["use_popart"] = False
    # Disable clipping so the coefficient is not masked by the norm cap.
    config["critic_optimizer"]["grad_clip"] = 0.0
    agent = amp_agent.AMPAgent(config, DummyAMPEnv(), TEST_DEVICE)
    optimizer = agent._critic_optimizer
    params = optimizer._param_list
    shared_params = agent._model.get_toppo_critic_shared_params()
    shared_ids = {id(param) for param in shared_params}
    obs = torch.randn(7, 5, device=TEST_DEVICE)
    targets = torch.tensor([1.5, -0.25], device=TEST_DEVICE)

    def build_losses():
        values = agent._model.eval_toppo_critic_components(obs, normalized=True)
        return [
            torch.mean(torch.square(values[:, i] - targets[i]))
            for i in range(2)
        ]

    # Reference combined gradient at unit scale (no optimizer step taken).
    task_grads = agent._compute_toppo_task_grads(optimizer, build_losses(), params)
    mean_grad = torch.stack(task_grads).mean(dim=0)
    scoped_task_grads = [
        agent._select_toppo_flat_grad(grad, params, shared_ids)
        for grad in task_grads
    ]
    scoped_grad, _ = toppo.fairgrad_combine(
        scoped_task_grads,
        alpha=agent._toppo_fairgrad_alpha,
        rescale=agent._toppo_fairgrad_rescale,
    )
    unit_grad = agent._replace_toppo_flat_grad(mean_grad, params, shared_ids, scoped_grad)

    coef = 0.5
    info = {}
    agent._toppo_optimizer_step(
        optimizer=optimizer,
        task_losses=build_losses(),
        method="fairgrad",
        diag_prefix="toppo_critic",
        info=info,
        scope_params=shared_params,
        grad_scale=coef,
    )
    actual = toppo.flatten_param_grads(params)

    # The applied gradient is exactly coef * the unit-scale combined gradient, and
    # the coefficient is logged for diagnostics.
    assert torch.allclose(actual, coef * unit_grad, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(actual, unit_grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        info["toppo_critic_grad_scale"],
        torch.tensor(coef, device=TEST_DEVICE),
    )


def test_toppo_flat_gradient_step_applies_post_aggregation_clip():
    import util.mp_util as mp_util
    if mp_util.get_num_procs() == 0:
        mp_util.init(0, 1, TEST_DEVICE, None)

    config = _toppo_agent_config()
    config["critic_optimizer"]["grad_clip"] = 0.1
    agent = amp_agent.AMPAgent(config, DummyAMPEnv(), TEST_DEVICE)
    optimizer = agent._critic_optimizer
    flat_grad = torch.ones(
        sum(param.numel() for param in optimizer._param_list),
        device=TEST_DEVICE,
    )

    info = agent._step_optimizer_with_flat_grad(optimizer, flat_grad)

    assert info["combined_grad_norm_pre_clip"] > 0.1
    assert torch.allclose(
        info["combined_grad_norm_post_clip"],
        torch.tensor(0.1, device=TEST_DEVICE),
        atol=1e-5,
    )
    assert info["clip_factor"] < 1.0


def test_toppo_component_gradients_are_aggregated_per_task_for_mp():
    import util.mp_util as mp_util
    if mp_util.get_num_procs() == 0:
        mp_util.init(0, 1, TEST_DEVICE, None)

    agent = amp_agent.AMPAgent(_toppo_agent_config(), DummyAMPEnv(), TEST_DEVICE)
    task_grads = [torch.ones(3), torch.full((3,), 2.0)]
    original_enable_mp = mp_util.enable_mp
    original_reduce = mp_util.reduce_inplace_mean
    calls = []

    def fake_reduce(grad):
        calls.append(grad.clone())
        grad.add_(5.0)

    try:
        mp_util.enable_mp = lambda: True
        mp_util.reduce_inplace_mean = fake_reduce
        agent._aggregate_toppo_task_grads(task_grads)
    finally:
        mp_util.enable_mp = original_enable_mp
        mp_util.reduce_inplace_mean = original_reduce

    assert len(calls) == 2
    assert torch.equal(task_grads[0], torch.full((3,), 6.0))
    assert torch.equal(task_grads[1], torch.full((3,), 7.0))


def test_episode_stats_masks_completed_envs_exactly():
    stats = eval_util.EpisodeStats(
        ["task_return", "disc_return"], num_envs=2, device=TEST_DEVICE
    )
    stats.update(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=TEST_DEVICE),
        torch.tensor(
            [base_env.DoneFlags.NULL.value, base_env.DoneFlags.FAIL.value],
            device=TEST_DEVICE,
        ),
    )
    stats.update(
        torch.tensor([[2.0, 1.0], [100.0, 100.0]], device=TEST_DEVICE),
        torch.tensor(
            [base_env.DoneFlags.TIME.value, base_env.DoneFlags.NULL.value],
            device=TEST_DEVICE,
        ),
        active_mask=torch.tensor([True, False], device=TEST_DEVICE),
    )

    assert stats.get_episodes() == 2
    assert torch.equal(
        stats.get_eps_per_env(), torch.tensor([1, 1], device=TEST_DEVICE)
    )
    assert stats.get_metric_stats("task_return")["mean"] == 3.0
    assert stats.get_metric_stats("disc_return")["mean"] == 3.5
    assert stats.get_done_rate(base_env.DoneFlags.FAIL) == 0.5
    assert stats.get_done_rate(base_env.DoneFlags.TIME) == 0.5
    records = stats.get_episode_records()
    assert records["env_id"].tolist() == [1, 0]
    assert records["episode_index"].tolist() == [0, 0]
    assert records["episode_length"].tolist() == [1, 2]
    assert records["done_flag"].tolist() == [
        base_env.DoneFlags.FAIL.value,
        base_env.DoneFlags.TIME.value,
    ]
    assert records["task_return"].tolist() == [3.0, 3.0]
    assert records["disc_return"].tolist() == [4.0, 3.0]


def test_style_distribution_metrics_detect_shift():
    torch.manual_seed(5)
    demo = torch.randn(256, 12)
    same = eval_util.compute_style_distribution_metrics(demo.clone(), demo, projection_dim=8)
    shifted = eval_util.compute_style_distribution_metrics(demo + 0.5, demo, projection_dim=8)

    assert same["style_feature_mean_rms"] < 1e-10
    assert same["style_feature_std_rms"] < 1e-10
    assert same["style_projected_fid"] < 1e-8
    assert shifted["style_feature_mean_rms"] > 0.1
    assert shifted["style_projected_fid"] > same["style_projected_fid"]


def test_style_distribution_projection_seed_is_reproducible_and_effective():
    torch.manual_seed(15)
    demo = torch.randn(192, 18)
    policy = demo * torch.linspace(0.7, 1.4, 18) + torch.linspace(-0.4, 0.6, 18)

    first = eval_util.compute_style_distribution_metrics(
        policy, demo, projection_dim=7, seed=101
    )
    repeated = eval_util.compute_style_distribution_metrics(
        policy, demo, projection_dim=7, seed=101
    )
    different = eval_util.compute_style_distribution_metrics(
        policy, demo, projection_dim=7, seed=102
    )

    assert first == repeated
    assert abs(first["style_projected_fid"] - different["style_projected_fid"]) > 1e-8


def test_feature_reservoir_is_bounded_and_deterministic():
    first = eval_util.FeatureReservoir(8, seed=19, device=TEST_DEVICE)
    second = eval_util.FeatureReservoir(8, seed=19, device=TEST_DEVICE)
    stream = torch.arange(80, device=TEST_DEVICE, dtype=torch.float32).view(40, 2)
    for chunk in stream.split(7):
        first.add(chunk)
        second.add(chunk)

    assert first.seen == 40
    assert first.values().shape == (8, 2)
    assert torch.equal(first.values(), second.values())


if __name__ == "__main__":
    test_component_popart_preserves_raw_values()
    test_component_popart_welford_matches_exact_running_moments()
    test_component_popart_ema_matches_sequential_updates()
    test_pcgrad_and_fairgrad_combine_are_finite()
    test_fairgrad_raw_alpha_one_has_fixed_norm_and_rescale_preserves_direction()
    test_fairgrad_satisfies_kkt_and_scipy_fallback()
    test_toppo_model_component_critic_and_popart_identity()
    test_toppo_model_can_load_legacy_layer_norm_bias_architecture()
    test_toppo_amp_agent_cpu_update_smoke()
    test_toppo_amp_agent_critic_only_actor_uses_scalar_ppo()
    test_toppo_critic_scope_mean_averages_task_local_heads()
    test_toppo_critic_coef_scales_combined_gradient_post_combine()
    test_toppo_flat_gradient_step_applies_post_aggregation_clip()
    test_toppo_component_gradients_are_aggregated_per_task_for_mp()
    test_episode_stats_masks_completed_envs_exactly()
    test_style_distribution_metrics_detect_shift()
    test_style_distribution_projection_seed_is_reproducible_and_effective()
    test_feature_reservoir_is_bounded_and_deterministic()
