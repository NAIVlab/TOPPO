"""CPU unit tests for the SAC / AMP-SAC agents (new, additive to the repo).

Run from the repo root:
    python -m pytest tests/test_sac_amp.py -q
"""

import copy
import math
import os
import sys
import tempfile
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml


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

import envs.base_env as base_env
import learning.agent_builder as agent_builder
import learning.amp_sac_agent as amp_sac_agent
import learning.base_agent as base_agent
import learning.distribution_tanh_gaussian as distribution_tanh_gaussian
import learning.sac_agent as sac_agent
import learning.sac_model as sac_model
import util.mp_util as mp_util

TEST_DEVICE = os.environ.get("SAC_TEST_DEVICE", "cpu")

# single-process mp_util init (train_model/test_model need get_num_procs() > 0)
if (mp_util.get_num_procs() == 0):
    mp_util.init(0, 1, TEST_DEVICE, None)

NUM_ENVS = 4
OBS_DIM = 5
ACTION_DIM = 2
DISC_OBS_DIM = 6
EP_LEN = 8


class SteppingAMPEnv:
    """Minimal stepping env satisfying the agent-facing surface used by
    BaseAgent/SACAgent/AMPSACAgent (mirrors tests/test_toppo.py's DummyEnv,
    extended with step/reset dynamics and disc obs)."""

    def __init__(self, device=TEST_DEVICE):
        self._device = device
        self._num_envs = NUM_ENVS
        self._step_counts = torch.zeros(self._num_envs, device=device, dtype=torch.long)
        self._obs = torch.zeros([self._num_envs, OBS_DIM], device=device)
        self._info = dict()
        self._steps_taken = 0
        return

    def get_obs_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

    def get_action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def get_disc_obs_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(DISC_OBS_DIM,), dtype=np.float32)

    def get_num_envs(self):
        return self._num_envs

    def get_reward_succ(self):
        return 1.0

    def get_reward_fail(self):
        return 0.0

    def set_mode(self, mode):
        return

    def record_diagnostics(self):
        return dict()

    def fetch_disc_obs_demo(self, n):
        return torch.randn([n, DISC_OBS_DIM], device=self._device) + 1.0

    def reset(self, env_ids=None):
        if (env_ids is None):
            self._step_counts[:] = 0
        else:
            self._step_counts[env_ids] = 0
        self._obs = torch.randn([self._num_envs, OBS_DIM], device=self._device)
        self._info["disc_obs"] = torch.randn([self._num_envs, DISC_OBS_DIM], device=self._device)
        return self._obs, self._info

    def step(self, action):
        assert(action.shape == (self._num_envs, ACTION_DIM))
        self._steps_taken += 1
        self._step_counts += 1

        self._obs = torch.randn([self._num_envs, OBS_DIM], device=self._device)
        self._info["disc_obs"] = torch.randn([self._num_envs, DISC_OBS_DIM], device=self._device)
        reward = torch.rand(self._num_envs, device=self._device)

        done = torch.full([self._num_envs], base_env.DoneFlags.NULL.value,
                          device=self._device, dtype=torch.int)
        timeout = self._step_counts >= EP_LEN
        done[timeout] = base_env.DoneFlags.TIME.value
        # sprinkle occasional FAIL/SUCC terminations to exercise terminal values
        if (self._steps_taken % 5 == 0):
            done[0] = base_env.DoneFlags.FAIL.value
        if (self._steps_taken % 7 == 0):
            done[1] = base_env.DoneFlags.SUCC.value
        return self._obs, reward, done, self._info


def _sac_agent_config(toppo=False, actor_pcgrad=False, ars=None):
    config = {
        "agent_name": "AMPSAC",
        "model": {
            "actor_net": "fc_2layers_128units",
            "actor_init_output_scale": 0.01,
            "actor_init_logstd": -1.0,
            "actor_logstd_min": -5.0,
            "actor_logstd_max": 2.0,
            "critic_net": "fc_2layers_128units",
            "num_critics": 2,
            "disc_net": "fc_2layers_128units",
        },
        "actor_optimizer": {"type": "Adam", "learning_rate": 3e-4},
        "critic_optimizer": {"type": "Adam", "learning_rate": 3e-4, "grad_clip": 0.0},
        "alpha_optimizer": {"type": "Adam", "learning_rate": 3e-4},
        "disc_optimizer": {"type": "SGD", "learning_rate": 2.5e-4, "weight_decay": 1e-4},
        "discount": 0.99,
        "steps_per_iter": 4,
        "iters_per_output": 3,
        "test_episodes": 4,
        "normalizer_samples": 100000000,
        "use_mixed_precision": False,
        "batch_size": 16,
        "updates_per_iter": 2,
        "replay_buffer_size": 64,
        "min_buffer_samples": 32,
        "random_action_samples": 16,
        "tau": 0.005,
        "init_alpha": 1.0,
        "learn_alpha": True,
        "target_entropy": "auto",
        "actor_critic_agg": "min",
        "disc_batch_size": 2,          # multiplier of num_envs -> 8
        "disc_updates_per_iter": 2,
        "disc_buffer_size": 64,
        "disc_replay_samples": 8,
        "disc_logit_reg": 0.01,
        "disc_grad_penalty": 5,
        "disc_reward_scale": 2,
        "task_reward_weight": 0.5,
        "disc_reward_weight": 0.5,
    }
    if (toppo):
        config["toppo"] = {
            "enabled": True,
            "num_tasks": 2,
            "task_names": ["task_reward", "disc_reward"],
            "actor_components": bool(actor_pcgrad),
            "actor_method": "pcgrad" if actor_pcgrad else "default",
            "critic_method": "fairgrad",
            "fairgrad_alpha": 1.0,
            "fairgrad_rescale": False,
            "critic_layer_norm": True,
            "critic_layer_norm_biasless": True,
            "use_popart": True,
            "popart_method": "welford",
            "popart_min_std": 0.01,
            "critic_coef": 1.0,
        }
    if (ars is not None):
        config["ars"] = ars
    return config


def _build_agent(toppo=False, actor_pcgrad=False, ars=None):
    env = SteppingAMPEnv()
    config = _sac_agent_config(toppo=toppo, actor_pcgrad=actor_pcgrad, ars=ars)
    agent = amp_sac_agent.AMPSACAgent(config=config, env=env, device=TEST_DEVICE)
    return agent, env


def test_tanh_gaussian_log_prob_matches_torch_distributions():
    torch.manual_seed(0)
    mean = torch.randn([64, ACTION_DIM])
    logstd = torch.randn([64, ACTION_DIM]).clamp(-2.0, 0.5)
    dist = distribution_tanh_gaussian.DistributionTanhGaussian(mean, logstd)
    a, logp = dist.rsample_with_log_prob()

    assert torch.all(a > -1.0) and torch.all(a < 1.0)

    base = torch.distributions.Independent(
        torch.distributions.Normal(mean, torch.exp(logstd)), 1)
    ref = torch.distributions.TransformedDistribution(
        base, [torch.distributions.transforms.TanhTransform(cache_size=1)])
    ref_logp = ref.log_prob(a)
    assert torch.allclose(logp, ref_logp, atol=1e-4), \
        f"max err {torch.max(torch.abs(logp - ref_logp))}"

    mode = dist.mode
    assert torch.allclose(mode, torch.tanh(mean))


def test_tanh_gaussian_rsample_is_reparameterized():
    mean = torch.zeros([8, ACTION_DIM], requires_grad=True)
    logstd = torch.zeros([8, ACTION_DIM])
    dist = distribution_tanh_gaussian.DistributionTanhGaussian(mean, logstd)
    a, logp = dist.rsample_with_log_prob()
    loss = (a.square().sum() + logp.sum())
    loss.backward()
    assert mean.grad is not None
    assert torch.isfinite(mean.grad).all()


def test_sac_model_shapes_and_popart_output_preservation():
    torch.manual_seed(0)
    env = SteppingAMPEnv()
    config = _sac_agent_config(toppo=True)
    model_config = dict(config["model"])
    model_config["toppo"] = dict(config["toppo"])
    model_config["init_alpha"] = 1.0
    model = sac_model.SACModel(model_config, env)

    obs = torch.randn([16, OBS_DIM])
    act = torch.rand([16, ACTION_DIM]) * 2.0 - 1.0

    q = model.eval_critic_components(obs, act, normalized=False)
    q_tar = model.eval_critic_components_target(obs, act, normalized=False)
    assert q.shape == (16, 2, 2)
    assert q_tar.shape == (16, 2, 2)
    # targets are deep copies of the online nets at init
    assert torch.allclose(q, q_tar, atol=1e-6)

    # PopArt update must preserve raw outputs of BOTH online and target critics
    targets = torch.randn([256, 2]) * torch.tensor([5.0, 0.2]) + torch.tensor([3.0, -1.0])
    q_before = model.eval_critic_components(obs, act, normalized=False)
    tar_before = model.eval_critic_components_target(obs, act, normalized=False)
    model.update_popart(targets)
    q_after = model.eval_critic_components(obs, act, normalized=False)
    tar_after = model.eval_critic_components_target(obs, act, normalized=False)
    assert torch.allclose(q_before, q_after, atol=1e-4), \
        f"online drift {torch.max(torch.abs(q_before - q_after))}"
    assert torch.allclose(tar_before, tar_after, atol=1e-4), \
        f"target drift {torch.max(torch.abs(tar_before - tar_after))}"
    # stats actually moved
    assert not torch.allclose(model._popart.sigma,
                              torch.ones_like(model._popart.sigma))


def test_polyak_update_moves_targets():
    torch.manual_seed(0)
    env = SteppingAMPEnv()
    config = _sac_agent_config(toppo=False)
    model_config = dict(config["model"])
    model_config["init_alpha"] = 1.0
    model = sac_model.SACModel(model_config, env)

    with torch.no_grad():
        for p in model._q_trunks.parameters():
            p.add_(1.0)

    obs = torch.randn([4, OBS_DIM])
    act = torch.rand([4, ACTION_DIM]) * 2.0 - 1.0
    q_tar_before = model.eval_critic_components_target(obs, act)

    tau = 0.25
    online_p = [p.clone() for p in model._q_trunks.parameters()]
    target_p_before = [p.clone() for p in model._tar_q_trunks.parameters()]
    model.polyak_update_targets(tau)
    for p, tp_before, tp in zip(online_p, target_p_before, model._tar_q_trunks.parameters()):
        expected = (1.0 - tau) * tp_before + tau * p
        assert torch.allclose(tp, expected, atol=1e-6)

    q_tar_after = model.eval_critic_components_target(obs, act)
    assert not torch.allclose(q_tar_before, q_tar_after)


def test_layer_norm_only_in_cb_critic():
    agent_cb, _ = _build_agent(toppo=True)
    has_ln = any(isinstance(m, torch.nn.LayerNorm)
                 for m in agent_cb._model._q_trunks.modules())
    assert has_ln
    # biasless: Linear layers in the trunk have no bias
    for m in agent_cb._model._q_trunks.modules():
        if isinstance(m, torch.nn.Linear):
            assert m.bias is None

    agent_v, _ = _build_agent(toppo=False)
    has_ln_v = any(isinstance(m, torch.nn.LayerNorm)
                   for m in agent_v._model._q_trunks.modules())
    assert not has_ln_v
    # actor never gets LN in either arm
    for agent in (agent_cb, agent_v):
        assert not any(isinstance(m, torch.nn.LayerNorm)
                       for m in agent._model._actor_layers.modules())


def test_component_terminal_values():
    agent, env = _build_agent(toppo=True)
    succ, fail = agent._get_component_terminal_values()
    expected_succ = 0.5 * env.get_reward_succ() / (1.0 - 0.99)
    assert succ.shape == (2,)
    assert abs(succ[0].item() - expected_succ) < 1e-4
    assert succ[1].item() == 0.0
    assert fail[0].item() == 0.0 and fail[1].item() == 0.0


def test_reward_component_relabel_matches_disc_reward():
    agent, _ = _build_agent(toppo=True)
    batch = {
        "reward": torch.rand(10),
        "disc_obs": torch.randn([10, DISC_OBS_DIM]),
    }
    comp = agent._compute_reward_components(batch)
    assert comp.shape == (10, 2)

    norm_disc_obs = agent._disc_obs_norm.normalize(batch["disc_obs"])
    disc_r = agent._calc_disc_rewards(norm_disc_obs)
    assert torch.allclose(comp[:, 0], 0.5 * batch["reward"], atol=1e-6)
    assert torch.allclose(comp[:, 1], 0.5 * disc_r, atol=1e-5)
    # disc reward formula sanity: positive, scaled by disc_reward_scale
    assert torch.all(disc_r >= 0.0)


def _run_train(agent, iters=8):
    """Drive train_model for a fixed number of iterations via max_samples."""
    max_samples = iters * agent._steps_per_iter * agent.get_num_envs()
    with tempfile.TemporaryDirectory() as out_dir:
        agent.train_model(max_samples=max_samples, out_dir=out_dir,
                          save_int_models=False, logger_type="txt")
        model_file = os.path.join(out_dir, "model.pt")
        assert os.path.isfile(model_file)
        state = torch.load(model_file, map_location="cpu")
    return state


def test_train_loop_vanilla_runs_and_keys_stable():
    torch.manual_seed(0)
    np.random.seed(0)
    agent, _ = _build_agent(toppo=False)
    # warmup crossing: min_buffer_samples=32 = 2 iters of 4*4; run 8 iters.
    # The txt Logger hard-asserts that no new key appears after iter 0, so
    # completing this run also validates the zero-update-info template.
    state = _run_train(agent, iters=8)
    assert agent._exp_buffer.get_total_samples() == 8 * 4 * NUM_ENVS
    assert torch.isfinite(agent._model.get_alpha())
    assert agent._model.get_alpha().item() > 0.0
    assert len(state) > 0


def test_train_loop_cb_runs_and_keys_stable():
    torch.manual_seed(0)
    np.random.seed(0)
    agent, _ = _build_agent(toppo=True)
    _run_train(agent, iters=8)
    # PopArt stats must have been updated by real critic updates
    assert agent._model._popart.count.sum().item() > 0
    sigma = agent._model._popart.sigma
    assert torch.isfinite(sigma).all()


def test_train_loop_actor_pcgrad_runs():
    torch.manual_seed(0)
    np.random.seed(0)
    agent, _ = _build_agent(toppo=True, actor_pcgrad=True)
    _run_train(agent, iters=8)


def test_zero_update_info_matches_real_update_keys():
    torch.manual_seed(0)
    np.random.seed(0)
    for toppo in (False, True):
        agent, _ = _build_agent(toppo=toppo)
        template_keys = set(agent._zero_update_info().keys())

        # fill the replay buffer past the warmup threshold
        agent._curr_obs, agent._curr_info = agent._reset_envs()
        agent.set_mode(base_agent.AgentMode.TRAIN)
        agent.eval()
        with torch.no_grad():
            agent._rollout_train(12)
        assert agent._exp_buffer.get_sample_count() >= agent._min_buffer_samples

        info = agent._update_model()
        real_keys = set(info.keys())
        assert template_keys == real_keys, (
            f"toppo={toppo} template-only={template_keys - real_keys} "
            f"real-only={real_keys - template_keys}"
        )


def test_save_load_roundtrip():
    torch.manual_seed(0)
    np.random.seed(0)
    agent, _ = _build_agent(toppo=True)
    _run_train(agent, iters=8)

    agent2, _ = _build_agent(toppo=True)
    with tempfile.TemporaryDirectory() as out_dir:
        model_file = os.path.join(out_dir, "model.pt")
        agent.save(model_file)
        agent2.load(model_file)

    for (k1, p1), (k2, p2) in zip(agent.state_dict().items(),
                                  agent2.state_dict().items()):
        assert k1 == k2
        assert torch.allclose(p1, p2, atol=1e-7), f"mismatch in {k1}"

    # deterministic policies must agree after load
    obs = torch.randn([NUM_ENVS, OBS_DIM])
    agent.set_mode(base_agent.AgentMode.TEST)
    agent2.set_mode(base_agent.AgentMode.TEST)
    a1, _ = agent._decide_action(obs, {})
    a2, _ = agent2._decide_action(obs, {})
    assert torch.allclose(a1, a2, atol=1e-6)


def test_test_mode_actions_are_deterministic_and_bounded():
    agent, _ = _build_agent(toppo=False)
    agent.set_mode(base_agent.AgentMode.TEST)
    obs = torch.randn([NUM_ENVS, OBS_DIM])
    a1, _ = agent._decide_action(obs, {})
    a2, _ = agent._decide_action(obs, {})
    assert torch.allclose(a1, a2)
    assert torch.all(a1 >= -1.0) and torch.all(a1 <= 1.0)


def test_replay_buffer_is_not_reset_between_iters():
    torch.manual_seed(0)
    agent, _ = _build_agent(toppo=False)
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    agent.set_mode(base_agent.AgentMode.TRAIN)
    agent.eval()
    with torch.no_grad():
        agent._rollout_train(4)
    count_a = agent._exp_buffer.get_sample_count()
    agent._init_iter()
    with torch.no_grad():
        agent._rollout_train(4)
    count_b = agent._exp_buffer.get_sample_count()
    assert count_b == count_a + 4 * NUM_ENVS


def test_agent_builder_registration(tmp_path):
    config = _sac_agent_config(toppo=True)
    agent_file = tmp_path / "amp_sac_test_agent.yaml"
    with open(agent_file, "w") as f:
        yaml.safe_dump(config, f)

    env = SteppingAMPEnv()
    agent = agent_builder.build_agent(str(agent_file), env, TEST_DEVICE)
    assert isinstance(agent, amp_sac_agent.AMPSACAgent)

    # plain SAC branch
    config_sac = _sac_agent_config(toppo=False)
    config_sac["agent_name"] = "SAC"
    for key in ["disc_net"]:
        config_sac["model"].pop(key)
    for key in ["disc_optimizer", "disc_batch_size", "disc_updates_per_iter",
                "disc_logit_reg", "disc_grad_penalty", "disc_reward_scale",
                "disc_eval_batch_size", "task_reward_weight", "disc_reward_weight"]:
        config_sac.pop(key, None)
    agent_file2 = tmp_path / "sac_test_agent.yaml"
    with open(agent_file2, "w") as f:
        yaml.safe_dump(config_sac, f)
    agent2 = agent_builder.build_agent(str(agent_file2), env, TEST_DEVICE)
    assert isinstance(agent2, sac_agent.SACAgent)
    assert not isinstance(agent2, amp_sac_agent.AMPSACAgent)


def test_production_yamls_construct():
    env = SteppingAMPEnv()
    for name, expect_components, expect_method in [
            ("amp_sac_walk_agent.yaml", False, None),
            ("amp_sac_walk_cd_agent.yaml", True, "default"),
            ("amp_sac_walk_cb_agent.yaml", True, "fairgrad")]:
        path = ROOT / "data" / "agents" / name
        agent = agent_builder.build_agent(str(path), env, TEST_DEVICE)
        assert isinstance(agent, amp_sac_agent.AMPSACAgent)
        if (expect_components):
            assert agent._toppo_enabled
            assert agent._num_components == 2
            assert agent._toppo_critic_method == expect_method
            assert agent._model.uses_popart()
            assert any(isinstance(m, torch.nn.LayerNorm)
                       for m in agent._model._q_trunks.modules())
        else:
            assert not agent._toppo_enabled
            assert agent._num_components == 1
            assert not agent._model.uses_popart()


def test_cd_and_cb_yamls_differ_only_in_critic_method():
    """The architecture-only control must be byte-identical to the CB arm
    except for critic_method, so the comparison isolates FairGrad."""
    cd = yaml.safe_load(open(ROOT / "data" / "agents" / "amp_sac_walk_cd_agent.yaml"))
    cb = yaml.safe_load(open(ROOT / "data" / "agents" / "amp_sac_walk_cb_agent.yaml"))
    assert cd["toppo"].pop("critic_method") == "default"
    assert cb["toppo"].pop("critic_method") == "fairgrad"
    # fairgrad_gram_dtype is a numerics/throughput knob that only exists on the
    # arm that actually runs FairGrad; equivalence to fp64 is asserted by
    # test_fp32_gram_fairgrad_matches_fp64_reference.
    assert cb["toppo"].pop("fairgrad_gram_dtype") == "float32"
    assert cd == cb

    vanilla = yaml.safe_load(open(ROOT / "data" / "agents" / "amp_sac_walk_agent.yaml"))
    # the vanilla arm differs only by the absence of the toppo block
    cb_no_toppo = dict(cb)
    cb_no_toppo.pop("toppo")
    assert vanilla == cb_no_toppo


def test_ars_yamls_differ_from_vanilla_only_by_the_ars_block():
    """The ARS arms must be architecturally identical to the vanilla SAC arm so
    the comparison isolates reward-side scaling."""
    agents = ROOT / "data" / "agents"
    vanilla = yaml.safe_load(open(agents / "amp_sac_walk_agent.yaml"))
    for name, expect_resets in [("amp_sac_walk_ars_agent.yaml", 0),
                                ("amp_sac_walk_arsreset_agent.yaml", 4)]:
        cfg = yaml.safe_load(open(agents / name))
        ars = cfg.pop("ars")
        assert cfg == vanilla, "{} differs from vanilla beyond the ars block".format(name)
        assert ars["enabled"] is True
        assert ars["num_resets"] == expect_resets
        assert "toppo" not in cfg, "ARS arms keep the plain scalar critic"


def test_control_arm_yamls_differ_by_exactly_one_knob():
    """Every control arm must isolate a single factor, or it cannot support the
    causal claim it exists to test."""
    agents = ROOT / "data" / "agents"
    vanilla = yaml.safe_load(open(agents / "amp_sac_walk_agent.yaml"))

    for name, key, value in [("amp_sac_walk_entmatch_agent.yaml", "target_entropy", -44.15),
                             ("amp_sac_walk_utd512_agent.yaml", "updates_per_iter", 512),
                             ("amp_sac_walk_reweight_agent.yaml", "disc_reward_weight", 1.105)]:
        cfg = yaml.safe_load(open(agents / name))
        assert cfg.pop(key) == value
        base = dict(vanilla)
        base.pop(key)
        assert cfg == base, "{} differs from vanilla beyond {}".format(name, key)

    # k1cd vs cd: only the component count
    cd = yaml.safe_load(open(agents / "amp_sac_walk_cd_agent.yaml"))
    k1 = yaml.safe_load(open(agents / "amp_sac_walk_k1cd_agent.yaml"))
    assert cd["toppo"].pop("num_tasks") == 2 and k1["toppo"].pop("num_tasks") == 1
    assert cd["toppo"].pop("task_names") == ["task_reward", "disc_reward"]
    assert k1["toppo"].pop("task_names") == ["amp_reward"]
    assert cd == k1

    # the PPO-matched target entropy must equal the fixed-sigma Gaussian entropy
    import math
    expected = 0.5 * 28 * math.log(2 * math.pi * math.e * 0.05 ** 2)
    assert abs(expected - (-44.15)) < 0.01, expected


def test_critic_architecture_2x2_is_clean():
    """vanilla / lnonly / popartonly / k1cd must form a clean 2x2 on the scalar
    critic, so the pair of ablations isolates LayerNorm from PopArt."""
    agents = ROOT / "data" / "agents"
    k1 = yaml.safe_load(open(agents / "amp_sac_walk_k1cd_agent.yaml"))
    assert k1["toppo"]["num_tasks"] == 1
    assert k1["toppo"]["critic_layer_norm"] is True and k1["toppo"]["use_popart"] is True

    for name, flag in [("amp_sac_walk_lnonly_agent.yaml", "use_popart"),
                       ("amp_sac_walk_popartonly_agent.yaml", "critic_layer_norm")]:
        cfg = yaml.safe_load(open(agents / name))
        assert cfg["toppo"].pop(flag) is False
        ref = {**k1, "toppo": {**k1["toppo"]}}
        ref["toppo"].pop(flag)
        assert cfg == ref, "{} differs from k1cd beyond {}".format(name, flag)

    env = SteppingAMPEnv()
    for name, ln, pa in [("amp_sac_walk_lnonly_agent.yaml", True, False),
                         ("amp_sac_walk_popartonly_agent.yaml", False, True)]:
        agent = agent_builder.build_agent(str(agents / name), env, TEST_DEVICE)
        has_ln = any(isinstance(m, torch.nn.LayerNorm)
                     for m in agent._model._q_trunks.modules())
        assert has_ln == ln, name
        assert agent._model.uses_popart() == pa, name
        assert agent._num_components == 1


def test_new_arm_yamls_construct():
    env = SteppingAMPEnv()
    for name in ["amp_sac_walk_entmatch_agent.yaml", "amp_sac_walk_k1cd_agent.yaml",
                 "amp_sac_walk_utd512_agent.yaml", "amp_sac_walk_reweight_agent.yaml"]:
        agent = agent_builder.build_agent(str(ROOT / "data" / "agents" / name),
                                          env, TEST_DEVICE)
        assert isinstance(agent, amp_sac_agent.AMPSACAgent)
    # entmatch really changes the entropy target
    ent = agent_builder.build_agent(
        str(ROOT / "data" / "agents" / "amp_sac_walk_entmatch_agent.yaml"), env, TEST_DEVICE)
    assert abs(ent._target_entropy - (-44.15)) < 1e-6
    van = agent_builder.build_agent(
        str(ROOT / "data" / "agents" / "amp_sac_walk_agent.yaml"), env, TEST_DEVICE)
    assert van._target_entropy == -float(van.get_action_size())


def test_cb_yaml_uses_fp32_gram():
    cb = yaml.safe_load(open(ROOT / "data" / "agents" / "amp_sac_walk_cb_agent.yaml"))
    assert cb["toppo"]["fairgrad_gram_dtype"] == "float32"


def test_ars_production_yamls_construct():
    env = SteppingAMPEnv()
    for name, expect_resets in [("amp_sac_walk_ars_agent.yaml", 0),
                                ("amp_sac_walk_arsreset_agent.yaml", 4)]:
        path = ROOT / "data" / "agents" / name
        agent = agent_builder.build_agent(str(path), env, TEST_DEVICE)
        assert agent._ars_enabled
        assert agent._ars_num_resets == expect_resets
        # scalar critic, but a 2-component reward decomposition for ARS
        assert agent._num_components == 1
        assert agent._ars_num_components == 2
        assert agent._ars_scale.shape == (2,)
        assert not agent._model.uses_popart()
        assert not any(isinstance(m, torch.nn.LayerNorm)
                       for m in agent._model._q_trunks.modules())


def _correlated_grads(n, norm, corr, seed=0):
    """Two component gradients with a prescribed norm and correlation --
    the real critic components share a trunk, so they are strongly correlated."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, generator=g)
    b = torch.randn(n, generator=g)
    g1 = a / a.norm() * norm
    g2 = corr * a + math.sqrt(1.0 - corr ** 2) * b
    g2 = g2 / g2.norm() * norm
    return [g1, g2]


def test_fp32_gram_matches_fp64_gram():
    """Precision of the KxK Gram is irrelevant; only its cost differs."""
    import learning.sacamp_fairgrad as sacamp_fairgrad

    for norm in [2.28, 0.0035]:
        grads = _correlated_grads(200000, norm, 0.8)
        f32, d32 = sacamp_fairgrad.fairgrad_combine_fp32_gram(
            grads, alpha=1.0, rescale=False, gram_dtype=torch.float32)
        f64, d64 = sacamp_fairgrad.fairgrad_combine_fp32_gram(
            grads, alpha=1.0, rescale=False, gram_dtype=torch.float64)
        rel = ((f32 - f64).norm() / f64.norm()).item()
        assert rel < 1e-5, "norm {}: rel {}".format(norm, rel)
        rel_w = ((d32.weights.double() - d64.weights.double()).norm()
                 / d64.weights.double().norm()).item()
        assert rel_w < 1e-5, rel_w


def test_solver_silently_degrades_at_small_correlated_gradients():
    """The reference solver exits on an ABSOLUTE residual threshold. When the
    per-component gradients are both small and correlated -- the off-policy
    critic case -- the entire right-hand side is below that threshold, so the
    solve returns its diagonal warm start and FairGrad degenerates to
    w ~ 1/||g_i||. The signature is the combined norm leaving sqrt(K)."""
    import learning.sacamp_fairgrad as sacamp_fairgrad
    import learning.toppo as toppo

    grads = _correlated_grads(200000, 0.0035, 0.8)

    ref, ref_diag = toppo.fairgrad_combine(grads, alpha=1.0, rescale=False)
    assert ref_diag.solver_iters == 0, "expected the early exit to trigger"
    assert abs(ref.norm().item() - math.sqrt(2)) > 0.3, ref.norm().item()

    fixed, fixed_diag = sacamp_fairgrad.fairgrad_combine_fp32_gram(
        grads, alpha=1.0, rescale=False)
    assert fixed_diag.solver_iters > 0, "preconditioned solve must iterate"
    assert abs(fixed.norm().item() - math.sqrt(2)) < 1e-3, fixed.norm().item()
    assert not fixed_diag.solver_fallback


def test_preconditioning_is_a_noop_where_the_reference_converges():
    """At on-policy gradient scales the reference already converges; the
    preconditioned path must agree with it in direction and magnitude."""
    import learning.sacamp_fairgrad as sacamp_fairgrad
    import learning.toppo as toppo

    grads = _correlated_grads(200000, 2.28, 0.8)
    ref, ref_diag = toppo.fairgrad_combine(grads, alpha=1.0, rescale=False)
    fixed, _ = sacamp_fairgrad.fairgrad_combine_fp32_gram(grads, alpha=1.0, rescale=False)

    assert ref_diag.solver_iters > 0
    assert abs(ref.norm().item() - math.sqrt(2)) < 1e-3
    cos = torch.dot(ref.double(), fixed.double()) / (ref.double().norm() * fixed.double().norm())
    assert cos.item() > 1 - 1e-9, cos.item()
    rel = ((fixed - ref).norm() / ref.norm()).item()
    assert rel < 1e-4, rel


def test_preconditioning_is_correct_at_alpha_other_than_one():
    """The preconditioning is only magnitude-preserving at alpha=1; at other
    alpha the combined gradient picks up s^((1-a)/(1+a)) and must be undone.
    Tested at PPO gradient scales, where the reference solver converges."""
    import learning.sacamp_fairgrad as sacamp_fairgrad
    import learning.toppo as toppo

    for alpha in [0.5, 1.0, 2.0]:
        grads = _correlated_grads(200000, 2.28, 0.8)
        ref, ref_diag = toppo.fairgrad_combine(grads, alpha=alpha, rescale=False)
        fixed, fixed_diag = sacamp_fairgrad.fairgrad_combine_fp32_gram(
            grads, alpha=alpha, rescale=False)

        cos = torch.dot(ref.double(), fixed.double()) / (
            ref.double().norm() * fixed.double().norm())
        assert cos.item() > 1 - 1e-9, "alpha={}: direction {}".format(alpha, cos.item())
        rel = abs(fixed.norm().item() / ref.norm().item() - 1.0)
        assert rel < 5e-3, "alpha={}: magnitude ratio off by {}".format(alpha, rel)

        # diagnostics must be in RAW gradient space, not the preconditioned one
        rel_mgn = abs(fixed_diag.mean_grad_norm / ref_diag.mean_grad_norm - 1.0)
        assert rel_mgn < 1e-3, "alpha={}: mean_grad_norm {} vs {}".format(
            alpha, fixed_diag.mean_grad_norm, ref_diag.mean_grad_norm)


def test_diagnostics_are_reported_in_raw_gradient_space():
    """The mean/combined gradient norms are logged diagnostics; in the
    preconditioned space mean_grad_norm is off by 1/s (~286x at off-policy
    gradient scales)."""
    import learning.sacamp_fairgrad as sacamp_fairgrad
    import learning.toppo as toppo

    grads = _correlated_grads(200000, 0.0035, 0.8)
    _, ref_diag = toppo.fairgrad_combine(grads, alpha=1.0, rescale=False)
    _, fixed_diag = sacamp_fairgrad.fairgrad_combine_fp32_gram(grads, alpha=1.0, rescale=False)

    raw_mean_norm = torch.stack(grads).double().mean(dim=0).norm().item()
    assert abs(fixed_diag.mean_grad_norm / raw_mean_norm - 1.0) < 1e-4
    assert abs(fixed_diag.mean_grad_norm / ref_diag.mean_grad_norm - 1.0) < 1e-3
    assert abs(fixed_diag.gram_trace / ref_diag.gram_trace - 1.0) < 1e-4


def test_rescale_uses_the_raw_mean_gradient_norm():
    import learning.sacamp_fairgrad as sacamp_fairgrad

    grads = _correlated_grads(200000, 0.0035, 0.8)
    combined, diag = sacamp_fairgrad.fairgrad_combine_fp32_gram(
        grads, alpha=1.0, rescale=True)
    raw_mean_norm = torch.stack(grads).double().mean(dim=0).norm().item()
    assert abs(combined.double().norm().item() / raw_mean_norm - 1.0) < 1e-3


def test_fast_gram_context_manager_restores_toppo():
    import learning.sacamp_fairgrad as sacamp_fairgrad
    import learning.toppo as toppo

    original = toppo.fairgrad_combine
    with sacamp_fairgrad.fast_gram(True):
        assert toppo.fairgrad_combine is not original
    assert toppo.fairgrad_combine is original

    with sacamp_fairgrad.fast_gram(False):
        assert toppo.fairgrad_combine is original
    assert toppo.fairgrad_combine is original

    # even if the body raises
    try:
        with sacamp_fairgrad.fast_gram(True):
            raise ValueError("boom")
    except ValueError:
        pass
    assert toppo.fairgrad_combine is original


def test_fairgrad_gram_dtype_config_is_opt_in():
    agent_default, _ = _build_agent(toppo=True)
    assert not agent_default._fairgrad_fp32_gram, "fp64 Gram must stay the default"
    # the CORRECTNESS fix must be on by default and independent of precision
    assert agent_default._fairgrad_precondition, \
        "preconditioning must not be gated on the precision flag"

    config = _sac_agent_config(toppo=True)
    config["toppo"]["fairgrad_gram_dtype"] = "float32"
    agent_fast = amp_sac_agent.AMPSACAgent(config=config, env=SteppingAMPEnv(),
                                           device=TEST_DEVICE)
    assert agent_fast._fairgrad_fp32_gram


def test_load_tolerates_checkpoint_without_ars_scale():
    """Checkpoints written before the ARS buffer existed must still load."""
    torch.manual_seed(0)
    agent, _ = _build_agent(toppo=False)
    state = agent.state_dict()
    assert "_ars_scale" in state
    legacy = {k: v for k, v in state.items() if k != "_ars_scale"}

    with tempfile.TemporaryDirectory() as out_dir:
        path = os.path.join(out_dir, "legacy.pt")
        torch.save(legacy, path)
        agent2, _ = _build_agent(toppo=False)
        agent2.load(path)
    assert torch.allclose(agent2._ars_scale, torch.ones_like(agent2._ars_scale))


def test_twin_q_aggregation_preserves_scalar_soft_target():
    """With K>1 the ensemble reduction must pick ONE critic per sample:
    sum_k min_i Q[i,k] != min_i sum_k Q[i,k], and the component decomposition
    is only valid if the summed target equals the scalar SAC target."""
    agent, _ = _build_agent(toppo=True)
    torch.manual_seed(0)
    q = torch.randn([32, 2, 2])  # [batch, num_qs, K]

    agg = agent._aggregate_critics(q)
    assert agg.shape == (32, 2)

    # summed aggregate == min over critics of the summed Q
    assert torch.allclose(torch.sum(agg, dim=-1),
                          torch.min(torch.sum(q, dim=-1), dim=1).values, atol=1e-6)

    # each row must be copied wholesale from one critic, not mixed per component
    matches_a = torch.all(torch.isclose(agg, q[:, 0]), dim=-1)
    matches_b = torch.all(torch.isclose(agg, q[:, 1]), dim=-1)
    assert torch.all(matches_a | matches_b)

    # the naive per-component min differs on this input (guards the regression)
    naive = torch.min(q, dim=1).values
    assert not torch.allclose(agg, naive)


def test_twin_q_aggregation_is_plain_min_when_scalar():
    agent, _ = _build_agent(toppo=False)
    q = torch.randn([16, 2, 1])
    assert torch.allclose(agent._aggregate_critics(q), torch.min(q, dim=1).values)


def test_discriminator_trains_during_rl_warmup():
    """AMP-PPO trains its discriminator from the first rollout; gating it on the
    SAC replay warmup would delay the style signal and leave the disc-obs
    normalizer with policy-only statistics."""
    torch.manual_seed(0)
    agent, _ = _build_agent(toppo=False)
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    agent.set_mode(base_agent.AgentMode.TRAIN)
    agent.eval()
    with torch.no_grad():
        agent._rollout_train(agent._steps_per_iter)

    # still inside the RL warmup
    assert agent._exp_buffer.get_sample_count() < agent._min_buffer_samples

    disc_before = [p.clone() for p in agent._model.get_disc_params()]
    agent.train()
    info = agent._update_model()
    disc_after = list(agent._model.get_disc_params())

    assert any(not torch.equal(a, b) for a, b in zip(disc_before, disc_after)), \
        "the discriminator must update during the RL warmup"
    assert info["sac_updates"] == 0.0, "the RL learner must still be in warmup"
    assert float(info["disc_loss"]) != 0.0


def test_disc_negatives_use_window_and_replay_streams():
    torch.manual_seed(0)
    agent, _ = _build_agent(toppo=False)
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    agent.set_mode(base_agent.AgentMode.TRAIN)
    agent.eval()
    with torch.no_grad():
        agent._rollout_train(agent._steps_per_iter)

    # window holds exactly the last steps_per_iter rows
    window = agent._get_disc_window_flat()
    assert window.shape[0] == agent._steps_per_iter * NUM_ENVS

    agent._store_disc_replay_data()
    assert agent._disc_buffer.get_sample_count() > 0

    captured = {}
    original = agent._model.eval_disc

    def spy(disc_obs):
        captured.setdefault("n", disc_obs.shape[0])
        return original(disc_obs)

    agent._model.eval_disc = spy
    demo_pool = agent._record_disc_demo_data()
    agent._compute_disc_loss(demo_pool)
    agent._model.eval_disc = original

    # negatives = window batch ++ replay batch -> twice the disc batch size
    assert captured["n"] == 2 * agent._disc_batch_size


def test_disc_demo_pool_recorded_once_per_iteration():
    """AMP-PPO records one demo sample per collected transition; recording a
    fresh pool inside every disc update would skew the normalizer."""
    torch.manual_seed(0)
    agent, _ = _build_agent(toppo=False)
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    agent.set_mode(base_agent.AgentMode.TRAIN)
    agent.eval()
    with torch.no_grad():
        agent._rollout_train(agent._steps_per_iter)

    calls = []
    original = agent._env.fetch_disc_obs_demo
    agent._env.fetch_disc_obs_demo = lambda n: calls.append(n) or original(n)
    agent.train()
    agent._update_model()
    agent._env.fetch_disc_obs_demo = original

    assert len(calls) == 1, "exactly one demo pool per iteration"
    assert calls[0] == agent._steps_per_iter * NUM_ENVS, \
        "demo pool must be sized to the collected window (1 demo / transition)"


ARS_ON = {"enabled": True, "min_mean": 0.01, "scale_samples": 64,
          "rescale_points": 2, "num_resets": 0, "warm_start_updates": 4}


def test_ars_scale_equalizes_component_means():
    torch.manual_seed(0)
    agent, _ = _build_agent(toppo=True, ars=ARS_ON)

    # fill the replay buffer so the scale can be estimated
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    agent.set_mode(base_agent.AgentMode.TRAIN)
    agent.eval()
    with torch.no_grad():
        agent._rollout_train(12)

    agent._update_ars_scale()
    scale = agent._ars_scale.clone()
    assert scale.shape == (2,)
    assert torch.all(scale >= 1.0 - 1e-6), "scale normalizes UP to the largest component"
    assert torch.isclose(torch.min(scale), torch.tensor(1.0), atol=1e-5), \
        "the largest component must keep scale 1"

    # after scaling, the component means must be equal
    batch = agent._exp_buffer.sample(agent._exp_buffer.get_capacity())
    raw = agent._compute_raw_reward_components(batch)
    scaled = agent._compute_reward_components(batch)
    assert torch.allclose(scaled, raw * scale, atol=1e-5), \
        "_compute_reward_components must apply the ARS scale exactly once"
    means = torch.mean(scaled, dim=0)
    raw_means = torch.mean(raw, dim=0)
    if (torch.min(raw_means) > agent._ars_min_mean):
        assert torch.allclose(means, means[0].expand_as(means), rtol=0.35), \
            "ARS should roughly equalize component means, got {}".format(means)


def test_ars_disabled_is_identity():
    agent, _ = _build_agent(toppo=True)
    comp = torch.rand([8, 2])
    assert torch.equal(agent._apply_ars_scale(comp), comp)
    assert agent._ars_info() == {}


def test_ars_reset_reinitializes_learner_but_not_discriminator():
    torch.manual_seed(0)
    ars = dict(ARS_ON)
    ars["num_resets"] = 2
    agent, _ = _build_agent(toppo=True, ars=ars)
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    agent.set_mode(base_agent.AgentMode.TRAIN)
    agent.eval()
    with torch.no_grad():
        agent._rollout_train(12)

    disc_before = [p.clone() for p in agent._model.get_disc_params()]
    actor_before = [p.clone() for p in agent._model.get_actor_params()]
    obs_norm_before = agent._obs_norm.get_mean().clone()
    buffer_before = agent._exp_buffer.get_sample_count()

    agent.train()
    agent._ars_reset_learner()

    disc_after = list(agent._model.get_disc_params())
    actor_after = list(agent._model.get_actor_params())
    assert len(disc_after) == len(disc_before)
    for p_before, p_after in zip(disc_before, disc_after):
        assert torch.equal(p_before, p_after), "ARS reset must not touch the discriminator"
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, actor_after)), \
        "ARS reset must re-initialize the actor"
    assert torch.equal(agent._obs_norm.get_mean(), obs_norm_before), \
        "ARS reset must retain the observation normalizer"
    assert agent._exp_buffer.get_sample_count() == buffer_before, \
        "ARS reset must retain the replay buffer"

    # the rebuilt optimizers must own the new parameters
    actor_ids = {id(p) for p in agent._model.get_actor_params()}
    assert all(id(p) in actor_ids for p in agent._actor_optimizer._param_list)


def test_ars_logging_keys_stable():
    """The txt Logger freezes its key set after iteration 0, so the warmup
    template must already contain every ARS key."""
    torch.manual_seed(0)
    np.random.seed(0)
    for ars in (ARS_ON, dict(ARS_ON, num_resets=2)):
        agent, _ = _build_agent(toppo=True, ars=ars)
        agent._max_samples = 8 * agent._steps_per_iter * agent.get_num_envs()
        template_keys = set(agent._zero_update_info().keys())

        agent._curr_obs, agent._curr_info = agent._reset_envs()
        agent.set_mode(base_agent.AgentMode.TRAIN)
        agent.eval()
        with torch.no_grad():
            agent._rollout_train(12)
        agent._sample_count = agent._exp_buffer.get_total_samples()

        real_keys = set(agent._update_model().keys())
        assert template_keys == real_keys, (
            "template-only={} real-only={}".format(template_keys - real_keys,
                                                   real_keys - template_keys))


def test_ars_schedule_fires_at_the_reference_points():
    """ARS recomputes the scale at creation and then at evenly spaced points;
    with rescale_points=4 over a 100M budget that is 100k, 20M, 40M, 60M, 80M
    (matching the reference reset_interval = max_steps/(n_reset+1))."""
    agent, _ = _build_agent(toppo=True, ars=dict(ARS_ON, rescale_points=4, num_resets=0))
    agent._max_samples = 100000000
    agent._min_buffer_samples = 100000

    collected = {"n": 0}
    # the schedule reads the LIVE buffer total, not the lagging _sample_count
    agent._collected_samples = lambda: collected["n"]
    fired = []
    agent._update_ars_scale = lambda: fired.append(collected["n"])

    for samples in [0, 50000, 100000, 100001, 19999999, 20000000,
                    39999999, 40000000, 60000000, 80000000, 99000000]:
        collected["n"] = samples
        agent._maybe_apply_ars()

    assert fired == [100000, 20000000, 40000000, 60000000, 80000000], fired
    # exhausted: no further events
    collected["n"] = 100000000
    agent._maybe_apply_ars()
    assert len(fired) == 5


def test_ars_reset_only_at_the_first_num_resets_points():
    agent, _ = _build_agent(toppo=True, ars=dict(ARS_ON, rescale_points=4, num_resets=2))
    agent._max_samples = 100000000
    agent._min_buffer_samples = 100000

    collected = {"n": 0}
    agent._collected_samples = lambda: collected["n"]
    resets = []
    agent._update_ars_scale = lambda: None
    agent._ars_reset_learner = lambda: resets.append(collected["n"])

    for samples in [100000, 20000000, 40000000, 60000000, 80000000]:
        collected["n"] = samples
        agent._maybe_apply_ars()

    # point 0 (creation) never resets; points 1..2 do; points 3..4 do not
    assert resets == [20000000, 40000000], resets


def test_ars_train_loop_runs_with_resets():
    torch.manual_seed(0)
    np.random.seed(0)
    agent, _ = _build_agent(toppo=True, ars=dict(ARS_ON, num_resets=2))
    _run_train(agent, iters=10)
    assert agent._ars_next_point > 0, "at least the creation-point rescale must fire"
    assert torch.all(agent._ars_scale > 0)


def test_ars_scale_applied_to_terminal_values():
    """Terminal component values stand in for bootstrapped reward mass and must
    carry the same ARS scale as the rewards."""
    torch.manual_seed(0)
    agent, _ = _build_agent(toppo=True, ars=ARS_ON)
    with torch.no_grad():
        agent._ars_scale.copy_(torch.tensor([3.0, 1.0]))

    n = 6
    batch = {
        "obs": torch.randn([n, OBS_DIM]),
        "next_obs": torch.randn([n, OBS_DIM]),
        "action": torch.rand([n, ACTION_DIM]) * 2.0 - 1.0,
        "reward": torch.rand(n),
        "disc_obs": torch.randn([n, DISC_OBS_DIM]),
        "done": torch.full([n], base_env.DoneFlags.SUCC.value, dtype=torch.int),
    }
    captured = {}
    original = agent._model.update_popart
    agent._model.update_popart = lambda t: captured.setdefault("tar", t.detach().clone())
    agent._update_critic_step(batch)
    agent._model.update_popart = original

    # both accessors already return ARS-scaled quantities
    succ_vals, _ = agent._get_component_terminal_values()
    scaled_reward = agent._compute_reward_components(batch)
    expected = scaled_reward + 0.99 * succ_vals
    assert torch.allclose(captured["tar"], expected, atol=1e-4)

    # and the scale really is in there: terminal task value scaled by 3, style by 1
    raw_succ = 0.5 * agent._compute_succ_val()
    assert torch.allclose(succ_vals, torch.tensor([3.0 * raw_succ, 0.0]), atol=1e-4)


def test_succ_fail_terminal_bootstrap_in_critic_target():
    """SUCC/FAIL rows must replace the bootstrap with terminal values."""
    torch.manual_seed(0)
    agent, _ = _build_agent(toppo=True)

    n = 8
    batch = {
        "obs": torch.randn([n, OBS_DIM]),
        "next_obs": torch.randn([n, OBS_DIM]),
        "action": torch.rand([n, ACTION_DIM]) * 2.0 - 1.0,
        "reward": torch.rand(n),
        "disc_obs": torch.randn([n, DISC_OBS_DIM]),
        "done": torch.full([n], base_env.DoneFlags.SUCC.value, dtype=torch.int),
    }

    captured = {}
    original = agent._model.update_popart

    def capture_popart(targets):
        captured["tar"] = targets.detach().clone()
        return original(targets)

    agent._model.update_popart = capture_popart
    agent._update_critic_step(batch)

    succ_vals, _ = agent._get_component_terminal_values()
    reward_components = agent._compute_reward_components(batch)
    expected = reward_components + 0.99 * succ_vals
    assert torch.allclose(captured["tar"], expected, atol=1e-4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
