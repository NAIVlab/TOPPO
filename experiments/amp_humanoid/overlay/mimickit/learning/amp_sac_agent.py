# Portions of this file are adapted from MimicKit (https://github.com/xbpeng/MimicKit,
# commit 2ed1e6c), Copyright 2025 Xue Bin Peng, licensed under the Apache License 2.0;
# the discriminator training and reward code mirrors learning/amp_agent.py. Modified by the TOPPO authors, 2026.
import numpy as np
import torch

import envs.base_env as base_env
import learning.amp_agent as amp_agent
import learning.amp_sac_model as amp_sac_model
import learning.base_agent as base_agent
import learning.eval_util as eval_util
import learning.experience_buffer as experience_buffer
import learning.mp_optimizer as mp_optimizer
import learning.normalizer as normalizer
import learning.sac_agent as sac_agent
import util.torch_util as torch_util

class AMPSACAgent(sac_agent.SACAgent):
    """SAC with an AMP-style discriminator (off-policy AMP).

    The env reward stored in the replay buffer is the raw task reward. The
    combined reward is re-labeled at every critic update with the CURRENT
    discriminator (the discriminator is non-stationary, so style rewards
    stored at collection time would go stale under replay). With the
    ``toppo`` block enabled, the twin-Q critic decomposes into the same two
    weighted components as the TOPPO AMP-PPO runs:
    [task_reward_weight * task_r, disc_reward_weight * disc_r].

    Discriminator training matches amp_agent.AMPAgent (BCE demo-vs-agent +
    gradient penalty + logit L2), except agent negatives are drawn from the
    SAC replay buffer instead of an extra on-policy window + disc buffer.
    """

    # Pure helpers reused byte-identically from AMPAgent (no super() calls).
    _calc_disc_rewards = amp_agent.AMPAgent._calc_disc_rewards
    _disc_loss_neg = amp_agent.AMPAgent._disc_loss_neg
    _disc_loss_pos = amp_agent.AMPAgent._disc_loss_pos
    _compute_disc_acc = amp_agent.AMPAgent._compute_disc_acc
    set_detailed_eval = amp_agent.AMPAgent.set_detailed_eval
    get_last_eval_records = amp_agent.AMPAgent.get_last_eval_records

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        self._detailed_eval = False
        self._style_eval_samples = 8192
        self._style_projection_dim = 64
        self._style_eval_seed = 260511701
        self._last_eval_records = None
        return

    def _load_params(self, config):
        super()._load_params(config)

        # disc_batch_size is a MULTIPLIER of num_envs, as in the AMP-PPO yamls
        self._disc_batch_size = int(np.ceil(float(config["disc_batch_size"])
                                            * self.get_num_envs()))
        self._disc_updates_per_iter = int(config["disc_updates_per_iter"])
        self._disc_buffer_size = int(config["disc_buffer_size"])
        self._disc_replay_samples = int(config["disc_replay_samples"])
        self._disc_logit_reg = float(config["disc_logit_reg"])
        self._disc_grad_penalty = float(config["disc_grad_penalty"])
        self._disc_reward_scale = float(config["disc_reward_scale"])
        self._disc_eval_batch_size = int(config.get("disc_eval_batch_size", 0))

        self._task_reward_weight = float(config["task_reward_weight"])
        self._disc_reward_weight = float(config["disc_reward_weight"])
        return

    def _build_exp_buffer(self, config):
        super()._build_exp_buffer(config)

        # Same auxiliary discriminator replay AMP-PPO keeps (amp_agent.py), so
        # negatives are drawn from the same two streams in both families.
        self._disc_buffer = experience_buffer.ExperienceBuffer(
            buffer_length=self._disc_buffer_size, batch_size=1, device=self._device)

        # Ring over the CURRENT on-policy window of disc observations; this is
        # the second negative stream (AMP-PPO uses the rollout window directly).
        disc_obs_space = self._env.get_disc_obs_space()
        disc_obs_dtype = torch_util.numpy_dtype_to_torch(disc_obs_space.dtype)
        self._disc_window = torch.zeros(
            [self._steps_per_iter, self.get_num_envs()] + list(disc_obs_space.shape),
            dtype=disc_obs_dtype, device=self._device)
        self._disc_window_head = 0
        self._disc_window_rows = 0
        return

    def _build_model(self, config):
        model_config = dict(config["model"])
        model_config["init_alpha"] = config.get("init_alpha", 1.0)
        if ("toppo" in config):
            model_config["toppo"] = dict(config["toppo"])
        model_config["w5_norm"] = dict(config.get("w5_norm", {}))
        self._model = amp_sac_model.AMPSACModel(model_config, self._env)
        return

    def _build_optimizer(self, config):
        super()._build_optimizer(config)

        disc_config = config["disc_optimizer"]
        disc_params = [p for p in self._model.get_disc_params() if p.requires_grad]
        self._disc_optimizer = mp_optimizer.MPOptimizer(disc_config, disc_params)
        return

    def _sync_optimizer(self):
        super()._sync_optimizer()
        self._disc_optimizer.sync()
        return

    def _build_normalizers(self):
        super()._build_normalizers()

        disc_obs_space = self._env.get_disc_obs_space()
        disc_obs_dtype = torch_util.numpy_dtype_to_torch(disc_obs_space.dtype)
        self._disc_obs_norm = normalizer.Normalizer(disc_obs_space.shape, clip=10.0,
                                                    device=self._device, dtype=disc_obs_dtype)
        return

    def _record_data_post_step(self, next_obs, r, done, next_info):
        super()._record_data_post_step(next_obs, r, done, next_info)

        disc_obs = next_info["disc_obs"]
        self._exp_buffer.record("disc_obs", disc_obs)

        self._disc_window[self._disc_window_head] = disc_obs
        self._disc_window_head = (self._disc_window_head + 1) % self._steps_per_iter
        self._disc_window_rows = min(self._disc_window_rows + 1, self._steps_per_iter)

        if (self._w5_rewnorm is not None):
            # [1] baselines VecNormalize: the discounted-return accumulator is
            # SEQUENTIAL, so its statistics must be folded at collection time,
            # in time order, with the reward as it stands now. The resulting
            # running std is then applied to the RE-LABELED components at
            # update time -- the same split PopArt uses (statistics from the
            # data stream, scale applied at loss time), and the only option
            # here because the style component is re-labeled off-policy.
            with torch.no_grad():
                norm_disc_obs = self._disc_obs_norm.normalize(disc_obs)
                disc_r = self._calc_disc_rewards(norm_disc_obs)
                components = torch.stack(
                    [self._task_reward_weight * r,
                     self._disc_reward_weight * disc_r], dim=-1)
                self._w5_rewnorm.normalize(
                    components, done != base_env.DoneFlags.NULL.value)

        if (self._need_normalizer_update()):
            self._disc_obs_norm.record(disc_obs)
        return

    def _get_disc_window_flat(self):
        rows = self._disc_window_rows
        window = self._disc_window[:rows]
        return window.reshape([-1] + list(window.shape[2:]))

    def _update_normalizers(self):
        super()._update_normalizers()
        self._disc_obs_norm.update()
        return

    def _get_num_raw_reward_components(self):
        # (task, style) -- independent of the critic head width, so an ARS arm
        # can scale the two reward streams while keeping a scalar critic.
        return 2

    def _get_raw_reward_component_names(self):
        return ["task_reward", "disc_reward"]

    def _compute_raw_reward_components(self, batch):
        task_r = batch["reward"]
        norm_disc_obs = self._disc_obs_norm.normalize(batch["disc_obs"])
        disc_r = self._calc_disc_rewards(norm_disc_obs)

        task_component = self._task_reward_weight * task_r
        disc_component = self._disc_reward_weight * disc_r
        return torch.stack([task_component, disc_component], dim=-1)

    def _apply_w5_rewnorm(self, raw):
        """Divide by the running std of the discounted-return accumulator (no
        mean subtraction) and clip -- VecNormalize semantics. The statistics
        themselves are folded in time order at collection time; only the scale
        is applied here, because the style component is re-labeled off-policy
        and its stored value is stale."""
        if (self._w5_rewnorm is None):
            return raw
        std = torch.sqrt(self._w5_rewnorm.var + 1e-8).to(dtype=raw.dtype)
        return (raw / std).clamp(-self._w5_rewnorm.clip, self._w5_rewnorm.clip)

    def _compute_reward_components(self, batch):
        raw = self._apply_w5_rewnorm(self._compute_raw_reward_components(batch))
        scaled = self._apply_ars_scale(raw)
        if (self._num_components == 2):
            return scaled
        assert(self._num_components == 1), \
            "AMPSAC supports 1 combined component or 2 (task, disc) components."
        return torch.sum(scaled, dim=-1, keepdim=True)

    def _get_component_terminal_values(self):
        # Terminal value of the task component; the style component gets 0,
        # matching amp_agent.AMPAgent._get_toppo_terminal_values. Returned in
        # ARS-scaled units, like the reward components.
        raw_succ = torch.zeros(2, device=self._device, dtype=torch.float32)
        raw_fail = torch.zeros(2, device=self._device, dtype=torch.float32)
        if (self._w5_rewnorm is None):
            raw_succ[0] = self._task_reward_weight * self._compute_succ_val()
            raw_fail[0] = self._task_reward_weight * self._compute_fail_val()
            succ = self._apply_ars_scale(raw_succ.unsqueeze(0)).squeeze(0)
            fail = self._apply_ars_scale(raw_fail.unsqueeze(0)).squeeze(0)
        else:
            # Under reward normalization the terminal value stands in for the
            # NORMALIZED (and clipped) reward stream, so the +-10 clip has to be
            # applied to the per-step reward and the geometric sum taken after
            # -- clipping the value itself would cap it at 10 instead of
            # 10 / (1 - gamma).
            raw_succ[0] = self._task_reward_weight * self._env.get_reward_succ()
            raw_fail[0] = self._task_reward_weight * self._env.get_reward_fail()
            horizon = 1.0 / (1.0 - self._discount)
            succ = self._apply_ars_scale(
                self._apply_w5_rewnorm(raw_succ.unsqueeze(0))).squeeze(0) * horizon
            fail = self._apply_ars_scale(
                self._apply_w5_rewnorm(raw_fail.unsqueeze(0))).squeeze(0) * horizon
        if (self._num_components == 1):
            succ = torch.sum(succ, dim=-1, keepdim=True)
            fail = torch.sum(fail, dim=-1, keepdim=True)
        return succ, fail

    def _update_model(self):
        # The discriminator is deliberately NOT gated on the RL warmup: AMP-PPO
        # trains it from its first rollout, and suppressing it here would both
        # delay the style signal and leave the disc-obs normalizer with
        # policy-only statistics for the first iterations.
        demo_pool = self._record_disc_demo_data()
        self._store_disc_replay_data()

        info = super()._update_model()

        disc_info = dict()
        for i in range(self._disc_updates_per_iter):
            loss_info = self._compute_disc_loss(demo_pool)
            self._disc_optimizer.step(loss_info["disc_loss"])
            torch_util.add_torch_dict(loss_info, disc_info)

        torch_util.scale_torch_dict(1.0 / self._disc_updates_per_iter, disc_info)
        info = {**info, **disc_info}
        return info

    def _record_disc_demo_data(self):
        """One demo pool per iteration, sized to the rollout window, recorded
        into the normalizer exactly once -- matching amp_agent's
        _record_disc_demo_data (1 demo sample per collected transition)."""
        n = max(self._disc_window_rows * self.get_num_envs(), 1)
        disc_obs_demo = self._env.fetch_disc_obs_demo(n)
        if (self._need_normalizer_update()):
            self._disc_obs_norm.record(disc_obs_demo)
        return disc_obs_demo

    def _store_disc_replay_data(self):
        """Mirror of amp_agent._store_disc_replay_data over the current
        window: a bounded random subsample enters the disc replay each iter."""
        disc_obs = self._get_disc_window_flat()
        n = disc_obs.shape[0]
        if (n == 0):
            return
        rand_idx = torch.randperm(n, device=self._device, dtype=torch.long)
        if (self._disc_buffer.is_full()):
            num_samples = min(n, self._disc_replay_samples)
        else:
            num_samples = n
        idx = rand_idx[:num_samples]
        self._disc_buffer.push({"disc_obs": disc_obs[idx].unsqueeze(1)})
        return

    def _sample_rows(self, source, n):
        idx = torch.randint(0, source.shape[0], [n], device=self._device)
        return source[idx]

    def _compute_disc_loss(self, demo_pool):
        """Negatives are the SAME two streams AMP-PPO concatenates: the current
        on-policy window and the discriminator replay buffer."""
        window = self._get_disc_window_flat()
        agent_obs = self._sample_rows(window, self._disc_batch_size)
        if (self._disc_buffer.get_sample_count() > 0):
            replay_obs = self._disc_buffer.sample(self._disc_batch_size)["disc_obs"]
            disc_obs = torch.cat([agent_obs, replay_obs], dim=0)
        else:
            disc_obs = agent_obs
        disc_obs_demo = self._sample_rows(demo_pool, self._disc_batch_size)

        norm_disc_obs_demo = self._disc_obs_norm.normalize(disc_obs_demo)
        norm_disc_obs_demo.requires_grad_(True)

        norm_disc_obs = self._disc_obs_norm.normalize(disc_obs)
        norm_disc_obs.requires_grad_(True)

        disc_agent_logit = self._model.eval_disc(norm_disc_obs)
        disc_demo_logit = self._model.eval_disc(norm_disc_obs_demo)
        disc_agent_logit = disc_agent_logit.squeeze(-1)
        disc_demo_logit = disc_demo_logit.squeeze(-1)

        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # grad penalty (identical structure to amp_agent.AMPAgent._compute_disc_loss)
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, norm_disc_obs_demo,
                                             grad_outputs=torch.ones_like(disc_demo_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)

        disc_agent_grad = torch.autograd.grad(disc_agent_logit, norm_disc_obs,
                                              grad_outputs=torch.ones_like(disc_agent_logit),
                                              create_graph=True, retain_graph=True, only_inputs=True)
        disc_agent_grad = disc_agent_grad[0]
        disc_agent_grad = torch.sum(torch.square(disc_agent_grad), dim=-1)

        disc_grad_penalty = 0.5 * (torch.mean(disc_demo_grad) + torch.mean(disc_agent_grad))
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        with torch.no_grad():
            disc_r = self._calc_disc_rewards(self._disc_obs_norm.normalize(disc_obs))
            disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)

        disc_info = {
            "disc_loss": disc_loss,
            "disc_grad_penalty": disc_grad_penalty.detach(),
            "disc_agent_acc": disc_agent_acc.detach(),
            "disc_demo_acc": disc_demo_acc.detach(),
            "disc_agent_logit": torch.mean(disc_agent_logit).detach(),
            "disc_demo_logit": torch.mean(disc_demo_logit).detach(),
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std,
        }

        if (self._disc_logit_reg != 0):
            logit_weights = self._model.get_disc_logit_weights()
            disc_logit_loss = torch.sum(torch.square(logit_weights))
            disc_loss = disc_loss + self._disc_logit_reg * disc_logit_loss
            disc_info["disc_loss"] = disc_loss
            disc_info["disc_logit_loss"] = disc_logit_loss.detach()

        return disc_info

    def _zero_update_info(self):
        info = super()._zero_update_info()
        zero = torch.zeros(1, device=self._device)
        for key in ["disc_loss", "disc_grad_penalty", "disc_agent_acc", "disc_demo_acc",
                    "disc_agent_logit", "disc_demo_logit", "disc_reward_mean",
                    "disc_reward_std"]:
            info[key] = zero.clone()
        if (self._disc_logit_reg != 0):
            info["disc_logit_loss"] = zero.clone()
        return info

    def _rollout_test(self, num_episodes):
        # Copied from amp_agent.AMPAgent._rollout_test (the detailed AMP eval
        # protocol), with the non-detailed path dispatched to BaseAgent since
        # AMPSACAgent does not inherit from AMPAgent.
        self._last_eval_records = None
        if (not self._detailed_eval):
            return base_agent.BaseAgent._rollout_test(self, num_episodes)

        if (num_episodes == 0):
            return {
                "mean_return": 0.0,
                "mean_ep_len": 0.0,
                "num_eps": 0,
            }

        num_envs = self.get_num_envs()
        episodes_per_env = int(np.ceil(num_episodes / num_envs))
        episode_stats = eval_util.EpisodeStats(
            ["task_return", "disc_return", "amp_return"],
            num_envs=num_envs,
            device=self._device,
        )
        step_moments = eval_util.TensorMoments()
        terminal_moments = eval_util.TensorMoments()
        style_reservoir = eval_util.FeatureReservoir(
            self._style_eval_samples,
            seed=self._style_eval_seed,
            device=self._device,
        )
        demo_style_pool = None
        if (self._style_eval_samples > 0):
            demo_style_pool = self._env.fetch_disc_obs_demo(
                self._style_eval_samples
            ).detach().cpu()
        self._test_return_tracker.reset()

        while True:
            active = episode_stats.get_eps_per_env() < episodes_per_env
            action, action_info = self._decide_action(self._curr_obs, self._curr_info)
            next_obs, task_reward, done, next_info = self._step_env(action)

            disc_obs = next_info["disc_obs"]
            norm_disc_obs = self._disc_obs_norm.normalize(disc_obs)
            disc_inputs = {"disc_obs": norm_disc_obs}
            disc_logits = torch_util.eval_minibatch(
                self._model.eval_disc, disc_inputs, self._disc_eval_batch_size
            ).squeeze(-1)
            disc_probability = torch.sigmoid(disc_logits)
            disc_reward = -torch.log(torch.clamp(1.0 - disc_probability, min=0.0001))
            disc_reward *= self._disc_reward_scale
            amp_reward = (
                self._task_reward_weight * task_reward
                + self._disc_reward_weight * disc_reward
            )

            rewards = torch.stack([task_reward, disc_reward, amp_reward], dim=-1)
            episode_stats.update(rewards, done, active_mask=active)

            tracked_reward = torch.where(active, task_reward, torch.zeros_like(task_reward))
            tracked_done = torch.where(
                active, done, torch.full_like(done, base_env.DoneFlags.NULL.value)
            )
            self._test_return_tracker.update(tracked_reward, tracked_done)

            env_metrics = self._env.get_eval_metrics()
            step_metrics = dict(env_metrics)
            step_metrics["disc_policy_probability"] = disc_probability
            step_metrics["disc_policy_logit"] = disc_logits
            step_moments.update(step_metrics, mask=active)

            terminal_mask = active & (done != base_env.DoneFlags.NULL.value)
            terminal_moments.update(env_metrics, mask=terminal_mask)

            style_reservoir.add(disc_obs, mask=active)

            needs_more_episodes = episode_stats.get_eps_per_env() < episodes_per_env
            reset_done = torch.where(
                needs_more_episodes,
                done,
                torch.full_like(done, base_env.DoneFlags.NULL.value),
            )
            if (torch.any(reset_done != base_env.DoneFlags.NULL.value)):
                self._curr_obs, self._curr_info = self._reset_done_envs(reset_done)
            else:
                self._curr_obs, self._curr_info = next_obs, next_info
            if (torch.all(episode_stats.get_eps_per_env() >= episodes_per_env)):
                break

        test_info = {
            "mean_return": episode_stats.get_metric_stats("task_return")["mean"],
            "mean_ep_len": episode_stats.get_mean_ep_len(),
            "num_eps": episode_stats.get_episodes(),
            "episodes_per_env": episodes_per_env,
            "fall_rate": episode_stats.get_done_rate(base_env.DoneFlags.FAIL),
            "timeout_rate": episode_stats.get_done_rate(base_env.DoneFlags.TIME),
            "native_success_rate": episode_stats.get_done_rate(base_env.DoneFlags.SUCC),
        }
        test_info["survival_rate"] = test_info["timeout_rate"] + test_info["native_success_rate"]

        for metric_name in episode_stats.metric_names:
            metric_stats = episode_stats.get_metric_stats(metric_name)
            for stat_name, value in metric_stats.items():
                test_info[f"{metric_name}_{stat_name}"] = value

        test_info.update(step_moments.summary("step"))
        test_info.update(terminal_moments.summary("terminal"))

        policy_obs = style_reservoir.values()
        if (policy_obs.shape[0] > 0):
            assert(demo_style_pool is not None)
            policy_obs = policy_obs.cpu()
            demo_obs = demo_style_pool[:policy_obs.shape[0]]
            style_info = eval_util.compute_style_distribution_metrics(
                policy_obs,
                demo_obs,
                projection_dim=self._style_projection_dim,
                seed=self._style_eval_seed,
            )
            test_info.update(style_info)

            demo_obs_device = demo_obs.to(self._device)
            norm_demo_obs = self._disc_obs_norm.normalize(demo_obs_device)
            demo_inputs = {"disc_obs": norm_demo_obs}
            demo_logits = torch_util.eval_minibatch(
                self._model.eval_disc, demo_inputs, self._disc_eval_batch_size
            ).squeeze(-1)
            demo_probability = torch.sigmoid(demo_logits)
            demo_reward = -torch.log(torch.clamp(1.0 - demo_probability, min=0.0001))
            demo_reward *= self._disc_reward_scale
            test_info["disc_demo_probability_mean"] = float(demo_probability.mean().item())
            test_info["disc_demo_reward_mean"] = float(demo_reward.mean().item())
            test_info["style_stream_observations"] = style_reservoir.seen

        self._last_eval_records = episode_stats.get_episode_records()
        return test_info
