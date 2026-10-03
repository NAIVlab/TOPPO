"""AMP-PPO with PopArt replaced by one of the scale normalizers in w5_norm.py.

Subclasses AMPAgent, so `learning/{ppo_agent,ppo_model,amp_agent,amp_model}.py`
stay byte-identical -- the PPO ladder these arms are compared against must keep
running exactly the code that produced it.

Exactly one of the three methods is active (`w5_norm.method`), and each is a
drop-in replacement for PopArt at the same rung (component critic + biasless
critic LayerNorm, mean component gradient):

  rewnorm  -- normalize the per-component rewards by the running std of the
              discounted-return accumulator BEFORE the TD(lambda) targets are
              built, so the critic regresses on normalized returns.
  retscale -- leave rewards and targets alone; divide the per-component
              value-loss residual by sigma_k.
  hlgauss  -- replace the per-component squared error with cross-entropy
              against a Gaussian-smoothed categorical projection of the target.
"""

import numpy as np
import torch

import envs.base_env as base_env
import learning.amp_agent as amp_agent
import learning.w5_amp_model as w5_amp_model
import learning.w5_norm as w5_norm

class W5AMPAgent(amp_agent.AMPAgent):
    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        return

    def _load_params(self, config):
        super()._load_params(config)
        w5_config = dict(config.get("w5_norm", {}))
        self._w5_method = str(w5_config.get("method", "none"))
        self._w5_config = w5_config
        valid = ("none", "rewnorm", "retscale", "hlgauss")
        assert(self._w5_method in valid), \
            "w5_norm.method must be one of {}".format(valid)
        if (self._w5_method != "none"):
            assert(self._toppo_enabled), \
                "the w5_norm alternatives replace PopArt on the component critic."
            toppo_config = config.get("toppo", {})
            assert(not bool(toppo_config.get("use_popart", False))), \
                "w5_norm replaces PopArt; do not enable both."
        if (self._w5_method != "none"):
            assert(not bool(config.get("ars", {}).get("enabled", False))), \
                "ARS rescales the raw reward components and w5_norm rescales " \
                "them again downstream; the two compose in an order neither " \
                "method defines. Run them as separate arms."
        if (self._w5_method == "retscale"):
            # Schaul's Gamma_t is the CONTINUATION discount. This benchmark
            # bootstraps terminals as absorbing states worth
            # get_reward_{fail,succ}() / (1 - gamma), so Gamma_t = 0 at a true
            # terminal is only the right reading while those are 0 -- which is
            # the case for task_steering (only FAIL and TIME are emitted, and
            # base_env.get_reward_fail() is 0). Check it rather than assume it.
            assert(float(self._env.get_reward_fail()) == 0.0
                   and float(self._env.get_reward_succ()) == 0.0), \
                "retscale treats FAIL/SUCC as Gamma=0, which assumes a " \
                "zero-valued absorbing state; this env bootstraps a non-zero " \
                "terminal value, so the per-step discount is not 0 there."
        return

    def _build_model(self, config):
        model_config = dict(config["model"])
        if ("toppo" in config):
            model_config["toppo"] = dict(config["toppo"])
            model_config["toppo"]["value_weights"] = \
                [1.0] * int(model_config["toppo"]["num_tasks"])
        model_config["w5_norm"] = dict(config.get("w5_norm", {}))
        self._model = w5_amp_model.W5AMPModel(model_config, self._env)
        return

    def _build_exp_buffer(self, config):
        super()._build_exp_buffer(config)
        # assigning an nn.Module to an attribute already registers it, so the
        # statistics travel with the checkpoint without an explicit add_module
        # (which would duplicate every buffer under a second key)
        self._w5_rewnorm, self._w5_retscale = w5_norm.build_w5_normalizer(
            self._w5_method, self._toppo_num_tasks, self._toppo_num_tasks,
            self.get_num_envs(), self._discount, self._w5_config, self._device)
        return

    def _reset_envs(self, env_ids=None):
        """Zero the rewnorm accumulator on a FORCED reset.

        BaseAgent resets every env after each output iteration and again at the
        start of test_model. Those are not episode boundaries the env reports,
        so without this the discounted return of the previous episode bleeds
        across the seam into the running statistics.
        """
        out = super()._reset_envs(env_ids)
        if (getattr(self, "_w5_rewnorm", None) is not None):
            if (env_ids is None):
                self._w5_rewnorm.ret.zero_()
            else:
                self._w5_rewnorm.ret[env_ids] = 0.0
        return out

    def _compute_rewards(self):
        info = super()._compute_rewards()
        if (self._w5_method != "rewnorm"):
            return info

        # AMPAgent has just written the weighted [0.5*task, 0.5*disc] components.
        # Normalize them in TIME ORDER: the discounted-return accumulator is
        # sequential, and the buffer is [steps_per_iter, num_envs, K].
        components = self._exp_buffer.get_data("toppo_reward_components")
        done = self._exp_buffer.get_data("done")
        normed = torch.empty_like(components)
        for t in range(components.shape[0]):
            done_mask = (done[t] != base_env.DoneFlags.NULL.value)
            normed[t] = self._w5_rewnorm.normalize(components[t], done_mask)
        self._exp_buffer.set_data("toppo_reward_components", normed)

        # keep the scalar reward consistent with the components the critic sees
        self._exp_buffer.set_data("reward", normed.sum(dim=-1))
        info.update(self._w5_rewnorm.get_diagnostics(self._toppo_task_names))
        return info

    def _w5_terminal_value(self, raw_reward):
        """Normalized terminal value of a per-step terminal reward.

        Under rewnorm the rollout rewards are divided by the running std and
        clipped, so a terminal bootstrap left in RAW units would mix scales
        inside one target. The clip belongs at the reward level, so it is
        applied before the geometric sum -- clipping the value would cap it at
        10 instead of 10/(1-gamma). Component 0 is the task stream (the only
        one with a terminal reward); the style component's terminal is 0, as in
        amp_agent.
        """
        value = self._task_reward_weight * float(raw_reward)
        if (self._w5_rewnorm is not None):
            std = float(torch.sqrt(self._w5_rewnorm.var[0] + 1e-8))
            value = float(np.clip(value / std, -self._w5_rewnorm.clip,
                                  self._w5_rewnorm.clip))
        return value / (1.0 - self._discount)

    def _compute_succ_val(self):
        # scalar path (advantages): the scalar value is the unweighted sum of
        # the components, and only the task component has a terminal reward
        if (self._w5_method != "rewnorm"):
            return super()._compute_succ_val()
        return self._w5_terminal_value(self._env.get_reward_succ())

    def _compute_fail_val(self):
        if (self._w5_method != "rewnorm"):
            return super()._compute_fail_val()
        return self._w5_terminal_value(self._env.get_reward_fail())

    def _get_toppo_terminal_values(self):
        if (self._w5_method != "rewnorm"):
            return super()._get_toppo_terminal_values()
        # AMPAgent would multiply _compute_succ_val() by the task weight again;
        # _w5_terminal_value already carries it, so build the vector directly
        succ = torch.zeros(self._toppo_num_tasks, device=self._device,
                           dtype=torch.float32)
        fail = torch.zeros(self._toppo_num_tasks, device=self._device,
                           dtype=torch.float32)
        succ[0] = self._w5_terminal_value(self._env.get_reward_succ())
        fail[0] = self._w5_terminal_value(self._env.get_reward_fail())
        return succ, fail

    def _update_model(self):
        info = super()._update_model()
        if (self._w5_method == "hlgauss"):
            # read AFTER the updates: the targets are projected inside
            # _update_model, so reading in _build_toppo_train_data reported the
            # PREVIOUS iteration's fraction and never the last one's
            info.update(self._model.get_hlgauss().get_diagnostics())
        return info

    def _build_toppo_train_data(self, done, rand_action_mask, norm_obs, norm_next_obs):
        info = super()._build_toppo_train_data(
            done=done, rand_action_mask=rand_action_mask,
            norm_obs=norm_obs, norm_next_obs=norm_next_obs)

        if (self._w5_method == "retscale"):
            # per-step discount: 0 at a true terminal (FAIL/SUCC), gamma
            # otherwise -- TIME truncation still bootstraps, so it is not
            # terminal for the discount.
            terminal = ((done == base_env.DoneFlags.FAIL.value)
                        | (done == base_env.DoneFlags.SUCC.value))
            gammas = torch.where(terminal, torch.zeros_like(terminal, dtype=torch.float32),
                                 torch.full_like(terminal, self._discount, dtype=torch.float32))
            self._w5_retscale.update_and_refresh_scale(
                self._exp_buffer.get_data("toppo_reward_components"),
                gammas,
                self._exp_buffer.get_data("toppo_tar_val"))
            info.update(self._w5_retscale.get_diagnostics(self._toppo_task_names))
        return info

    def _compute_toppo_critic_loss(self, batch):
        if (self._w5_method == "hlgauss"):
            norm_obs = self._obs_norm.normalize(batch["obs"])
            tar_val = batch["toppo_tar_val"]
            logits = self._model.eval_toppo_critic_logits(norm_obs)
            hl = self._model.get_hlgauss()
            task_losses = [torch.mean(hl.loss(logits[:, i], tar_val[:, i]))
                           for i in range(self._toppo_num_tasks)]
            loss = torch.mean(torch.stack(task_losses))
            info = {"critic_loss": loss, "toppo_task_losses": task_losses}
            for i, name in enumerate(self._toppo_task_names):
                info["toppo_critic_loss_{}".format(name)] = task_losses[i].detach()
            return info

        info = super()._compute_toppo_critic_loss(batch)
        if (self._w5_method == "retscale"):
            # Schaul et al. scale the RESIDUAL, i.e. the loss by 1/sigma^2.
            scale = self._w5_retscale.scale(torch.float32, self._device)
            task_losses = [info["toppo_task_losses"][i] / (scale[i] ** 2)
                           for i in range(self._toppo_num_tasks)]
            info["toppo_task_losses"] = task_losses
            info["critic_loss"] = torch.mean(torch.stack(task_losses))
            for i, name in enumerate(self._toppo_task_names):
                info["toppo_critic_loss_{}".format(name)] = task_losses[i].detach()
        return info
