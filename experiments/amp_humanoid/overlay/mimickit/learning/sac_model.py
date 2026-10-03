# Portions of this file are adapted from MimicKit (https://github.com/xbpeng/MimicKit,
# commit 2ed1e6c), Copyright 2025 Xue Bin Peng, licensed under the Apache License 2.0;
# eval_actor, the actor trunk construction and the actor/critic input dictionaries follow
# learning/ppo_model.py. Modified by the TOPPO authors, 2026.
import copy
import numpy as np
import torch

import learning.base_model as base_model
import learning.distribution_tanh_gaussian as distribution_tanh_gaussian
import learning.nets.net_builder as net_builder
import learning.toppo as toppo
import learning.w5_norm as w5_norm
import util.torch_util as torch_util

class SACModel(base_model.BaseModel):
    """Twin-Q SAC model with an optional component critic head.

    The critic mirrors the TOPPO component-critic design from ppo_model.py:
    K reward components share a trunk (optionally biasless Linear + LayerNorm)
    with a Linear(hidden, K) head and a shared ComponentPopArt affine. K = 1
    recovers a standard scalar twin-Q critic. Target networks (trunks, heads,
    and PopArt affine copies) are registered as ordinary submodules so they
    are checkpointed with the agent.
    """

    def __init__(self, config, env):
        super().__init__(config, env)
        self._build_nets(config, env)
        return

    def eval_actor(self, obs):
        h = self._actor_layers(obs)
        a_dist = self._action_dist(h)
        return a_dist

    def eval_critic_components(self, obs, action, normalized=False):
        """Evaluate the online twin critics. Returns [batch, num_qs, num_components]."""
        vals = self._eval_q_nets(self._q_trunks, self._q_heads, obs, action)
        if (self.uses_hlgauss()):
            return self._hlgauss.from_logits(vals)
        if (self._use_popart):
            vals = self._popart.affine(vals)
            if (not normalized):
                vals = self._popart.denormalize(vals)
        return vals

    def eval_critic_components_target(self, obs, action, normalized=False):
        """Evaluate the target twin critics. Returns [batch, num_qs, num_components]."""
        vals = self._eval_q_nets(self._tar_q_trunks, self._tar_q_heads, obs, action)
        if (self.uses_hlgauss()):
            return self._hlgauss.from_logits(vals)
        if (self._use_popart):
            vals = vals * self._tar_popart_weight + self._tar_popart_bias
            if (not normalized):
                vals = self._popart.denormalize(vals)
        return vals

    def uses_hlgauss(self):
        return self._hlgauss is not None

    def get_hlgauss(self):
        return self._hlgauss

    def _eval_q_nets(self, trunks, heads, obs, action):
        q_input = torch.cat([obs, action], dim=-1)
        vals = []
        for trunk, head in zip(trunks, heads):
            h = trunk(q_input)
            out = head(h)
            if (self.uses_hlgauss()):
                out = out.view(*out.shape[:-1], self._num_components,
                               self._hlgauss.num_bins)
            vals.append(out)
        vals = torch.stack(vals, dim=1)
        return vals

    def eval_critic_logits(self, obs, action):
        """[batch, num_qs, K, num_bins] -- HL-Gauss only."""
        assert(self.uses_hlgauss())
        return self._eval_q_nets(self._q_trunks, self._q_heads, obs, action)

    def normalize_value_targets(self, targets):
        if (self._use_popart):
            targets = self._popart.normalize_target(targets)
        return targets

    def update_popart(self, targets):
        if (not self._use_popart):
            return
        old_mu = self._popart.mu.clone()
        old_sigma = self._popart.sigma.clone()
        self._popart.update_and_renormalize(targets)
        new_mu = self._popart.mu.clone()
        new_sigma = self._popart.sigma.clone().clamp_min(self._popart.min_std)

        # Rescale the target-net affine with the same POP transform so raw
        # target values are preserved (denormalize uses the shared mu/sigma).
        with torch.no_grad():
            ratio = (old_sigma / new_sigma).to(dtype=self._tar_popart_weight.dtype)
            self._tar_popart_weight.mul_(ratio)
            new_bias = (old_sigma * self._tar_popart_bias.double() + old_mu - new_mu) / new_sigma
            self._tar_popart_bias.copy_(new_bias.to(dtype=self._tar_popart_bias.dtype))
        return

    def uses_popart(self):
        return self._use_popart

    def get_alpha(self):
        return torch.exp(self._log_alpha)

    def get_actor_params(self):
        params = list(self._actor_layers.parameters()) + list(self._action_dist.parameters())
        return params

    def get_critic_params(self):
        params = []
        for trunk in self._q_trunks:
            params += list(trunk.parameters())
        for head in self._q_heads:
            params += list(head.parameters())
        if (self._use_popart):
            params += list(self._popart.parameters())
        return params

    def get_critic_shared_params(self):
        """FairGrad scope: the shared Q trunks only. Heads and the PopArt
        affine receive plain mean gradients, matching the TOPPO critic scope."""
        params = []
        for trunk in self._q_trunks:
            params += list(trunk.parameters())
        return params

    def get_alpha_params(self):
        return [self._log_alpha]

    def reset_rl_parameters(self):
        """Re-initialize the RL learner (actor, twin critics, targets, alpha,
        PopArt affine) in place, leaving any other submodule (e.g. an AMP
        discriminator in a subclass) untouched. Used by the ARS reset."""
        device = next(self.parameters()).device
        config = dict(self._init_config)
        env = self._init_env

        self._build_actor(config, env)
        self._build_critic(config, env)
        self._build_alpha(config)
        self.to(device)
        return

    @torch.no_grad()
    def polyak_update_targets(self, tau):
        online_params = []
        target_params = []
        for trunk, tar_trunk in zip(self._q_trunks, self._tar_q_trunks):
            online_params += list(trunk.parameters())
            target_params += list(tar_trunk.parameters())
        for head, tar_head in zip(self._q_heads, self._tar_q_heads):
            online_params += list(head.parameters())
            target_params += list(tar_head.parameters())
        if (self._use_popart):
            online_params += [self._popart.weight, self._popart.bias]
            target_params += [self._tar_popart_weight, self._tar_popart_bias]

        for p, tar_p in zip(online_params, target_params):
            tar_p.mul_(1.0 - tau)
            tar_p.add_(tau * p)
        return

    def _build_nets(self, config, env):
        # retained so reset_rl_parameters() can rebuild the RL submodules
        self._init_config = dict(config)
        self._init_env = env

        self._load_component_config(config)
        self._build_actor(config, env)
        self._build_critic(config, env)
        self._build_alpha(config)
        return

    def _load_component_config(self, config):
        toppo_config = config.get("toppo", {})
        self._component_enabled = bool(toppo_config.get("enabled", False))
        if (self._component_enabled):
            self._num_components = int(toppo_config.get("num_tasks", 0))
            assert(self._num_components > 0), "SAC component critic requires num_tasks > 0."
        else:
            self._num_components = 1
        self._use_popart = self._component_enabled and bool(toppo_config.get("use_popart", False))
        # w5_norm [3] (HL-Gauss): a categorical value head REPLACES PopArt. Only the head
        # shape and the read-out change; the trunks, the twin-Q structure and
        # the component axis are untouched, so FairGrad/PCGrad still see one
        # loss per component.
        w5_config = config.get("w5_norm", {})
        self._w5_method = str(w5_config.get("method", "none"))
        self._w5_config = dict(w5_config)
        if (self._w5_method == "hlgauss"):
            assert(not self._use_popart), "HL-Gauss replaces PopArt; enable one."
            assert(self._component_enabled), "HL-Gauss uses the component critic."
        self._hlgauss = None
        self._critic_layer_norm = bool(toppo_config.get("critic_layer_norm", False))
        self._critic_layer_norm_biasless = bool(toppo_config.get("critic_layer_norm_biasless", True))
        self._toppo_config = toppo_config
        return

    def _build_actor(self, config, env):
        net_name = config["actor_net"]
        input_dict = self._build_actor_input_dict(env)
        self._actor_layers, layers_info = net_builder.build_net(net_name, input_dict,
                                                                activation=self._activation)
        in_size = torch_util.calc_layers_out_size(self._actor_layers)
        a_size = self._get_action_size(env)
        self._action_dist = distribution_tanh_gaussian.DistributionTanhGaussianBuilder(
            in_size,
            a_size,
            init_output_scale=config.get("actor_init_output_scale", 0.01),
            init_logstd=config.get("actor_init_logstd", -1.0),
            logstd_min=config.get("actor_logstd_min", -5.0),
            logstd_max=config.get("actor_logstd_max", 2.0),
        )
        return

    def _build_critic(self, config, env):
        num_qs = int(config.get("num_critics", 2))
        assert(num_qs >= 1), "SAC requires at least one critic."
        net_name = config["critic_net"]
        input_dict = self._build_critic_input_dict(env)

        trunks = []
        heads = []
        for i in range(num_qs):
            trunk, layers_info = net_builder.build_net(net_name, input_dict,
                                                       activation=self._activation)
            if (self._critic_layer_norm):
                trunk = self._add_layer_norm(trunk, biasless=self._critic_layer_norm_biasless)
            layers_out_size = torch_util.calc_layers_out_size(trunk)
            head_width = self._num_components
            if (self._w5_method == "hlgauss"):
                if (self._hlgauss is None):
                    self._hlgauss = w5_norm.HLGaussHead.from_config(self._w5_config)
                head_width = self._num_components * self._hlgauss.num_bins
            head = torch.nn.Linear(layers_out_size, head_width)
            torch.nn.init.zeros_(head.bias)
            if (self._w5_method == "hlgauss"):
                with torch.no_grad():
                    head.bias.copy_(self._hlgauss.logits_for_value(0.0)
                                    .repeat(self._num_components))
            trunks.append(trunk)
            heads.append(head)

        self._q_trunks = torch.nn.ModuleList(trunks)
        self._q_heads = torch.nn.ModuleList(heads)

        if (self._use_popart):
            self._popart = toppo.ComponentPopArt(
                self._num_components,
                min_std=float(self._toppo_config.get("popart_min_std", 1e-2)),
                method=self._toppo_config.get("popart_method", "welford"),
                ema_decay=float(self._toppo_config.get("popart_ema_decay", 3e-4)),
            )
            self._tar_popart_weight = torch.nn.Parameter(
                torch.ones(self._num_components, dtype=torch.float32), requires_grad=False)
            self._tar_popart_bias = torch.nn.Parameter(
                torch.zeros(self._num_components, dtype=torch.float32), requires_grad=False)

        self._tar_q_trunks = copy.deepcopy(self._q_trunks)
        self._tar_q_heads = copy.deepcopy(self._q_heads)
        for param in self._tar_q_trunks.parameters():
            param.requires_grad_(False)
        for param in self._tar_q_heads.parameters():
            param.requires_grad_(False)
        return

    def _build_alpha(self, config):
        init_alpha = float(config.get("init_alpha", 1.0))
        assert(init_alpha > 0.0), "init_alpha must be positive."
        init_log_alpha = float(torch.log(torch.tensor(init_alpha)))
        self._log_alpha = torch.nn.Parameter(
            torch.tensor(init_log_alpha, dtype=torch.float32), requires_grad=True)
        return

    def _add_layer_norm(self, layers, biasless=True):
        # Mirrors ppo_model.PPOModel._add_layer_norm (biasless Linear + LayerNorm).
        new_layers = []
        for layer in layers.children():
            if (biasless and isinstance(layer, torch.nn.Linear)):
                layer.bias = None
            new_layers.append(layer)
            if (isinstance(layer, torch.nn.Linear)):
                new_layers.append(torch.nn.LayerNorm(layer.out_features))
        return torch.nn.Sequential(*new_layers)

    def _get_action_size(self, env):
        a_space = env.get_action_space()
        return int(np.prod(a_space.shape))

    def _build_actor_input_dict(self, env):
        obs_space = env.get_obs_space()
        input_dict = {"obs": obs_space}
        return input_dict

    def _build_critic_input_dict(self, env):
        obs_space = env.get_obs_space()
        a_space = env.get_action_space()
        input_dict = {"obs": obs_space, "action": a_space}
        return input_dict
