# Portions of this file are adapted from MimicKit (https://github.com/xbpeng/MimicKit,
# commit 2ed1e6c), Copyright 2025 Xue Bin Peng, licensed under the Apache License 2.0;
# load() extends BaseAgent.load in learning/base_agent.py, and _decide_action() follows
# PPOAgent._decide_action in learning/ppo_agent.py. Modified by the TOPPO authors, 2026.
import numpy as np
import torch

import envs.base_env as base_env
import learning.base_agent as base_agent
import learning.mp_optimizer as mp_optimizer
import learning.ppo_agent as ppo_agent
import learning.sac_model as sac_model
import learning.sacamp_fairgrad as sacamp_fairgrad
import learning.w5_norm as w5_norm
from util.logger import Logger
import util.mp_util as mp_util
import util.torch_util as torch_util

class SACAgent(base_agent.BaseAgent):
    """Off-policy soft actor-critic agent on the MimicKit agent contract.

    The experience buffer is used as a ring replay buffer (it is never reset
    per iteration), and each train iteration performs a fixed number of
    updates, so the update count is identical across processes. The optional
    ``toppo`` config block applies the TOPPO critic-balancing stack to the
    twin-Q critic: a K-component critic head, biasless Linear+LayerNorm
    trunk, Welford ComponentPopArt on Q targets, and FairGrad over the
    per-component TD losses on the shared trunk (heads and the PopArt affine
    receive mean gradients). Gradient-surgery plumbing is reused verbatim
    from PPOAgent so both algorithms share one implementation.
    """

    # Reuse the TOPPO gradient-surgery plumbing byte-identically from PPOAgent.
    # None of these call super(), so unbound reuse is safe.
    _compute_toppo_task_grads = ppo_agent.PPOAgent._compute_toppo_task_grads
    _aggregate_toppo_task_grads = ppo_agent.PPOAgent._aggregate_toppo_task_grads
    _select_toppo_flat_grad = ppo_agent.PPOAgent._select_toppo_flat_grad
    _replace_toppo_flat_grad = ppo_agent.PPOAgent._replace_toppo_flat_grad
    _step_optimizer_with_flat_grad = ppo_agent.PPOAgent._step_optimizer_with_flat_grad
    _add_toppo_diag = ppo_agent.PPOAgent._add_toppo_diag

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        return

    def _load_params(self, config):
        super()._load_params(config)

        self._batch_size = int(config["batch_size"])
        self._updates_per_iter = int(config["updates_per_iter"])
        self._replay_buffer_size = int(config["replay_buffer_size"])
        self._min_buffer_samples = int(config.get("min_buffer_samples", self._batch_size))
        self._random_action_samples = int(config.get("random_action_samples", self._min_buffer_samples))
        self._tau = float(config.get("tau", 0.005))
        self._learn_alpha = bool(config.get("learn_alpha", True))
        self._actor_critic_agg = config.get("actor_critic_agg", "min")
        assert(self._actor_critic_agg in ("min", "first")), \
            "Unsupported actor_critic_agg: {}".format(self._actor_critic_agg)

        assert(self._min_buffer_samples >= self._batch_size), \
            "min_buffer_samples must cover at least one batch."

        toppo_config = config.get("toppo", {})
        self._toppo_enabled = bool(toppo_config.get("enabled", False))
        if (self._toppo_enabled):
            self._num_components = int(toppo_config.get("num_tasks", 0))
            assert(self._num_components > 0), "SAC component critic requires num_tasks > 0."
        else:
            self._num_components = 1
        self._toppo_task_names = list(toppo_config.get(
            "task_names", ["component_{}".format(i) for i in range(self._num_components)]
        ))
        assert(len(self._toppo_task_names) == self._num_components), \
            "task_names length must match num_tasks."
        self._toppo_critic_method = toppo_config.get("critic_method", "default")
        self._toppo_actor_method = toppo_config.get("actor_method", "default")
        self._toppo_actor_components = bool(toppo_config.get("actor_components", False))
        self._toppo_fairgrad_alpha = float(toppo_config.get("fairgrad_alpha", 1.0))
        self._toppo_fairgrad_rescale = bool(toppo_config.get("fairgrad_rescale", False))
        self._toppo_critic_coef = float(toppo_config.get("critic_coef", 1.0))
        # Two INDEPENDENT knobs, see learning/sacamp_fairgrad.py:
        #  - fairgrad_precondition is a CORRECTNESS fix (the solver's absolute
        #    early-exit tolerance is scale-dependent and silently degrades
        #    FairGrad to its diagonal warm start at off-policy gradient scales).
        #    It defaults ON; gating it on the precision flag, as an earlier
        #    version did, meant a config left at the documented fp64 default
        #    silently reverted to the broken solver with no diagnostic.
        #  - fairgrad_gram_dtype is a THROUGHPUT knob (fp64 Gram is ~121x
        #    slower on workstation cards; it shifts the weights by ~1e-7).
        self._fairgrad_precondition = bool(toppo_config.get("fairgrad_precondition", True))
        gram_dtype = str(toppo_config.get("fairgrad_gram_dtype", "float64"))
        assert(gram_dtype in ("float32", "float64")), \
            "fairgrad_gram_dtype must be float32 or float64"
        self._fairgrad_fp32_gram = (gram_dtype == "float32")
        self._toppo_use_popart = self._toppo_enabled and bool(toppo_config.get("use_popart", False))
        valid_methods = {"default", "pcgrad", "fairgrad"}
        assert(self._toppo_critic_method in valid_methods), \
            "Unsupported critic_method: {}".format(self._toppo_critic_method)
        assert(self._toppo_actor_method in valid_methods), \
            "Unsupported actor_method: {}".format(self._toppo_actor_method)

        # PopArt statistics and gradient surgery consume per-process batches,
        # so multi-process training would silently desynchronize them.
        if (mp_util.get_num_procs() > 1):
            assert(not self._toppo_use_popart and self._toppo_critic_method == "default"
                   and self._toppo_actor_method == "default"), \
                "SAC critic balancing supports single-process training only."

        # target entropy defaults to -|A| in squashed action space
        target_entropy = config.get("target_entropy", "auto")
        if (target_entropy == "auto"):
            self._target_entropy = -float(self.get_action_size())
        else:
            self._target_entropy = float(target_entropy)

        self._load_ars_params(config)
        self._load_w5_params(config)
        return

    def _load_w5_params(self, config):
        """w5_norm alternatives to PopArt. Exactly one may be active, and it
        replaces PopArt at the same rung (component critic + LN-c)."""
        w5_config = dict(config.get("w5_norm", {}))
        self._w5_method = str(w5_config.get("method", "none"))
        self._w5_config = w5_config
        valid = ("none", "rewnorm", "retscale", "hlgauss")
        assert(self._w5_method in valid), \
            "w5_norm.method must be one of {}".format(valid)
        if (self._w5_method != "none"):
            assert(self._toppo_enabled), \
                "the w5_norm alternatives replace PopArt on the component critic."
            assert(not self._toppo_use_popart), \
                "w5_norm replaces PopArt; do not enable both."
            assert(mp_util.get_num_procs() == 1), \
                "w5_norm statistics are per-process."
        if (self._w5_method == "rewnorm"):
            # the discounted-return accumulator is folded at collection time,
            # and only AMPSACAgent has that hook -- on plain SACAgent the arm
            # would silently be a no-op wearing the rewnorm label
            assert(type(self)._record_data_post_step
                   is not base_agent.BaseAgent._record_data_post_step), \
                "rewnorm needs the collection-time fold in " \
                "AMPSACAgent._record_data_post_step."
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

    def _load_ars_params(self, config):
        """Adaptive Reward Scaling (Cho et al., ICML 2025), ported to the
        reward-component axis. See _update_ars_scale for the port decisions."""
        ars_config = config.get("ars", {})
        self._ars_enabled = bool(ars_config.get("enabled", False))
        self._ars_min_mean = float(ars_config.get("min_mean", 0.01))
        self._ars_scale_samples = int(ars_config.get("scale_samples", 65536))
        # number of evenly spaced points at which the scale is recomputed
        self._ars_rescale_points = int(ars_config.get("rescale_points", 4))
        # how many of those points also re-initialize the RL learner (faithful
        # ARS resets; 0 = scaling only)
        self._ars_num_resets = int(ars_config.get("num_resets", 0))
        self._ars_warm_start_updates = int(ars_config.get("warm_start_updates", 1000))
        assert(self._ars_num_resets <= self._ars_rescale_points), \
            "ars.num_resets cannot exceed ars.rescale_points."
        if (self._ars_enabled):
            assert(mp_util.get_num_procs() == 1), \
                "ARS reward scaling supports single-process training only."
        self._ars_next_point = 0
        self._max_samples = None
        # ARS operates on the RAW reward decomposition, which is independent of
        # the critic head width: an ARS arm can therefore keep the plain scalar
        # critic and differ from the baseline only in the reward scale.
        self._ars_num_components = self._get_num_raw_reward_components()
        self._ars_component_names = self._get_raw_reward_component_names()
        if (self._ars_enabled):
            assert(self._ars_num_components > 1), \
                "ARS needs a multi-component reward decomposition to be meaningful."
        # registered here (not in _build_model, which subclasses replace) so it
        # is always present and travels with the checkpoint
        self.register_buffer(
            "_ars_scale",
            torch.ones(self._ars_num_components, dtype=torch.float32),
            persistent=True,
        )
        return

    def _toppo_optimizer_step(self, *args, **kwargs):
        """PPOAgent's gradient surgery, optionally with the fp32-Gram FairGrad.

        The surgery itself is unchanged (and is the same code the PPO arms
        run); only the precision of the K x K Gram matrix differs, and only
        for SAC arms that opt in via toppo.fairgrad_gram_dtype.
        """
        with sacamp_fairgrad.fast_gram(self._fairgrad_precondition,
                                       fp32_gram=self._fairgrad_fp32_gram):
            return ppo_agent.PPOAgent._toppo_optimizer_step(self, *args, **kwargs)

    def _collected_samples(self):
        """Live sample count. self._sample_count is refreshed by BaseAgent only
        after the whole train iteration, so warmup/ARS boundaries computed from
        it fire one iteration late."""
        return self._exp_buffer.get_total_samples()

    def _get_num_raw_reward_components(self):
        """Number of components the raw reward decomposes into (>= 1). Plain
        SAC has a scalar reward; AMP-SAC splits it into (task, style)."""
        return 1

    def _get_raw_reward_component_names(self):
        return ["reward"]

    def _build_model(self, config):
        model_config = dict(config["model"])
        model_config["init_alpha"] = config.get("init_alpha", 1.0)
        if ("toppo" in config):
            model_config["toppo"] = dict(config["toppo"])
        model_config["w5_norm"] = dict(config.get("w5_norm", {}))
        self._model = sac_model.SACModel(model_config, self._env)
        return

    def train_model(self, max_samples, out_dir, save_int_models, logger_type):
        # ARS schedules its rescale/reset points against the total budget.
        self._max_samples = max_samples
        return super().train_model(max_samples, out_dir, save_int_models, logger_type)

    def load(self, in_file):
        """Tolerate checkpoints written before the ARS scale buffer existed.

        `_ars_scale` is a 1-D buffer that is the identity when ARS is off, so a
        checkpoint that predates it is fully usable; strict loading would
        otherwise reject it with "Missing key(s) in state_dict".
        """
        state_dict = torch.load(in_file, map_location=self._device)
        if ("_ars_scale" not in state_dict):
            state_dict["_ars_scale"] = torch.ones(
                self._ars_num_components, dtype=torch.float32, device=self._device)
            Logger.print("Checkpoint predates ARS; defaulting _ars_scale to ones.")
        self.load_state_dict(state_dict)
        self._sync_optimizer()
        Logger.print("Loaded model parameters from {:s}".format(in_file))
        return

    def _build_optimizer(self, config):
        # configs are retained so an ARS reset can rebuild the optimizers
        self._actor_optimizer_config = config["actor_optimizer"]
        self._critic_optimizer_config = config["critic_optimizer"]
        self._alpha_optimizer_config = config.get(
            "alpha_optimizer", {"type": "Adam", "learning_rate": 3e-4})
        self._rebuild_rl_optimizers()
        return

    def _sync_optimizer(self):
        self._actor_optimizer.sync()
        self._critic_optimizer.sync()
        self._alpha_optimizer.sync()
        return

    def _build_exp_buffer(self, config):
        super()._build_exp_buffer(config)
        # assigning an nn.Module to an attribute already registers it, so the
        # statistics travel with the checkpoint without an explicit add_module
        # (which would duplicate every buffer under a second key)
        self._w5_rewnorm, self._w5_retscale = w5_norm.build_w5_normalizer(
            self._w5_method, self._get_num_raw_reward_components(),
            self._num_components, self.get_num_envs(), self._discount,
            self._w5_config, self._device)
        self._w5_retscale_folded = False
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

    def _get_exp_buffer_length(self):
        buffer_length = int(np.ceil(self._replay_buffer_size / self.get_num_envs()))
        buffer_length = max(buffer_length, 1)
        capacity = buffer_length * self.get_num_envs()
        assert(self._min_buffer_samples <= capacity), \
            "min_buffer_samples ({}) exceeds replay capacity ({}); " \
            "it can never be reached.".format(self._min_buffer_samples, capacity)
        return buffer_length

    # NOTE: unlike PPO/AWR, _init_iter does NOT reset the experience buffer,
    # so it behaves as a ring replay buffer.

    def _decide_action(self, obs, info):
        norm_obs = self._obs_norm.normalize(obs)
        a_dist = self._model.eval_actor(norm_obs)

        if (self._mode == base_agent.AgentMode.TRAIN):
            # BaseAgent refreshes self._sample_count only AFTER the whole train
            # iteration, so it lags by one iteration; the buffer total is the
            # live count and makes the warmup boundary exact.
            if (self._collected_samples() < self._random_action_samples):
                norm_a = 2.0 * torch.rand([obs.shape[0], self.get_action_size()],
                                          device=self._device) - 1.0
            else:
                norm_a = a_dist.sample()
        elif (self._mode == base_agent.AgentMode.TEST):
            norm_a = a_dist.mode
        else:
            assert(False), "Unsupported agent mode: {}".format(self._mode)

        norm_a = norm_a.detach()
        a = self._a_norm.unnormalize(norm_a)
        a_info = dict()
        return a, a_info

    def _update_model(self):
        self.train()

        if (self._exp_buffer.get_sample_count() < self._min_buffer_samples):
            info = self._zero_update_info()
            return info

        # cleared BEFORE the ARS hook: an ARS learner reset runs
        # warm_start_updates critic steps, which must fold like any other
        self._w5_retscale_folded = False
        ars_info = self._maybe_apply_ars()

        info = dict()
        for i in range(self._updates_per_iter):
            batch = self._exp_buffer.sample(self._batch_size)

            critic_info = self._update_critic_step(batch)
            actor_info = self._update_actor_step(batch)
            self._model.polyak_update_targets(self._tau)

            step_info = {**critic_info, **actor_info}
            torch_util.add_torch_dict(step_info, info)

        torch_util.scale_torch_dict(1.0 / self._updates_per_iter, info)
        info["sac_updates"] = float(self._updates_per_iter)
        info["replay_filled_frac"] = float(self._exp_buffer.get_sample_count()) \
            / float(self._exp_buffer.get_capacity())
        info = {**info, **ars_info, **self._w5_info()}
        return info

    # ------------------------------------------------------------------
    # Adaptive Reward Scaling (ARS)
    # ------------------------------------------------------------------

    def _maybe_apply_ars(self):
        """Recompute the ARS scale (and optionally reset the learner) when a
        scheduled point is reached. Returns the ARS logging dict."""
        if (not self._ars_enabled):
            return dict()

        events = 0.0
        resets = 0.0
        if (self._ars_rescale_points > 0 and self._max_samples is not None):
            # Points are evenly spaced over the sample budget; point 0 is the
            # ARS "agent creation" point of the reference implementation, which
            # fires as soon as the warmup buffer is available.
            collected = self._collected_samples()
            while (self._ars_next_point <= self._ars_rescale_points):
                if (self._ars_next_point == 0):
                    threshold = self._min_buffer_samples
                else:
                    threshold = (self._max_samples * self._ars_next_point
                                 / (self._ars_rescale_points + 1))
                if (collected < threshold):
                    break

                do_reset = (self._ars_next_point > 0
                            and self._ars_next_point <= self._ars_num_resets)
                self._update_ars_scale()
                if (do_reset):
                    self._ars_reset_learner()
                    resets += 1.0
                self._ars_next_point += 1
                events += 1.0

        info = self._ars_info()
        info["ars_events"] = events
        info["ars_resets"] = resets
        return info

    @torch.no_grad()
    def _update_ars_scale(self):
        """ARS scale: equalize the mean magnitude of every reward component to
        the largest one, scale[k] = max_j(mean_j) / max(mean_k, min_mean).

        Port notes vs the Meta-World reference implementation of ARS
        (Cho et al., ICML 2025; ms-cho/Adaptive-Reward-Scaling-ARS-for-MTRL, MIT):
        - There the axis is TASKS with stationary reward functions; here it is
          the AMP reward COMPONENTS, and the style component is non-stationary
          (it is re-labeled from the current discriminator). The means are
          therefore estimated with the CURRENT discriminator over a large
          replay sample, not from stored reward values, and are refreshed at
          `rescale_points` evenly spaced points instead of once.
        - The reference multiplies stored rewards at replay-sample time; we
          apply the same multiplicative scale to the reward components (and to
          the terminal component values, which stand in for bootstrapped
          reward mass) at update time, which is the same thing.
        """
        n = min(self._ars_scale_samples, self._exp_buffer.get_sample_count())
        n = min(n, self._exp_buffer.get_capacity())
        if (n <= 0):
            return
        batch = self._exp_buffer.sample(n)
        components = self._compute_raw_reward_components(batch)
        means = torch.mean(components, dim=0)
        denom = torch.clamp_min(means, self._ars_min_mean)
        scale = torch.max(means) / denom
        self._ars_scale.copy_(scale.to(dtype=self._ars_scale.dtype))
        return

    def _ars_reset_learner(self):
        """Faithful ARS reset: re-initialize the RL learner and warm-start it
        from the retained replay buffer under the new scale.

        The discriminator, its optimizer, the replay buffer and the normalizers
        are retained: in the reference there is no learned reward, so resetting
        the AMP discriminator would reset the reward function itself rather
        than the learner. This deviation is deliberate and disclosed.
        """
        self._model.reset_rl_parameters()
        self._rebuild_rl_optimizers()
        for _ in range(self._ars_warm_start_updates):
            if (self._exp_buffer.get_sample_count() < self._batch_size):
                break
            batch = self._exp_buffer.sample(self._batch_size)
            self._update_critic_step(batch)
            self._update_actor_step(batch)
            self._model.polyak_update_targets(self._tau)
        return

    def _rebuild_rl_optimizers(self):
        actor_params = [p for p in self._model.get_actor_params() if p.requires_grad]
        self._actor_optimizer = mp_optimizer.MPOptimizer(self._actor_optimizer_config, actor_params)
        critic_params = [p for p in self._model.get_critic_params() if p.requires_grad]
        self._critic_optimizer = mp_optimizer.MPOptimizer(self._critic_optimizer_config, critic_params)
        alpha_params = [p for p in self._model.get_alpha_params() if p.requires_grad]
        self._alpha_optimizer = mp_optimizer.MPOptimizer(self._alpha_optimizer_config, alpha_params)
        return

    def _apply_ars_scale(self, raw_components):
        """Scale a [..., num_raw_components] tensor by the ARS scale."""
        if (not self._ars_enabled):
            return raw_components
        assert(raw_components.shape[-1] == self._ars_num_components)
        return raw_components * self._ars_scale.to(dtype=raw_components.dtype)

    def _ars_info(self):
        if (not self._ars_enabled):
            return dict()
        info = {"ars_events": 0.0, "ars_resets": 0.0}
        for k, name in enumerate(self._ars_component_names):
            info["ars_scale_{}".format(name)] = self._ars_scale[k].detach().clone()
        return info

    def _update_critic_step(self, batch):
        norm_obs = self._obs_norm.normalize(batch["obs"])
        norm_next_obs = self._obs_norm.normalize(batch["next_obs"])
        norm_a = self._a_norm.normalize(batch["action"])
        done = batch["done"]

        reward_components = self._compute_reward_components(batch)
        assert(reward_components.shape[-1] == self._num_components), \
            "reward components must decompose the scalar reward into num_tasks " \
            "components; override _compute_reward_components (as AMPSACAgent does) " \
            "for num_tasks > 1."

        with torch.no_grad():
            alpha = self._model.get_alpha().detach()
            next_dist = self._model.eval_actor(norm_next_obs)
            next_a, next_logp = next_dist.rsample_with_log_prob()
            next_q = self._model.eval_critic_components_target(norm_next_obs, next_a,
                                                               normalized=False)
            next_q = self._aggregate_critics(next_q)

            # split the entropy bonus evenly over components so the summed
            # target matches the scalar soft value
            next_vals = next_q - (alpha / self._num_components) * next_logp.unsqueeze(-1)

            # terminal values stand in for bootstrapped reward mass, so they
            # are returned in the same (possibly ARS-scaled) units as the
            # reward components
            succ_vals, fail_vals = self._get_component_terminal_values()
            succ_mask = (done == base_env.DoneFlags.SUCC.value)
            fail_mask = (done == base_env.DoneFlags.FAIL.value)
            next_vals[succ_mask] = succ_vals.to(dtype=next_vals.dtype)
            next_vals[fail_mask] = fail_vals.to(dtype=next_vals.dtype)

            tar_vals = reward_components + self._discount * next_vals

        if (self._w5_method == "hlgauss"):
            # [3] Farebrother et al.: per-component cross-entropy against a
            # Gaussian-smoothed categorical projection of the same scalar
            # target, summed over the twin-Q ensemble exactly as the squared
            # error is. PopArt is off, so the target is in raw value units.
            logits = self._model.eval_critic_logits(norm_obs, norm_a)
            hl = self._model.get_hlgauss()
            ce = hl.loss(logits, tar_vals.unsqueeze(1).expand(-1, logits.shape[1], -1))
            task_losses = [torch.mean(torch.sum(ce[..., k], dim=1))
                           for k in range(self._num_components)]
        else:
            self._model.update_popart(tar_vals)
            norm_tar_vals = self._model.normalize_value_targets(tar_vals)

            pred = self._model.eval_critic_components(
                norm_obs, norm_a, normalized=self._model.uses_popart())
            diff = norm_tar_vals.unsqueeze(1) - pred
            sq_err = torch.square(diff)
            # sum over the twin-Q ensemble, mean over the batch
            task_losses = [torch.mean(torch.sum(sq_err[..., k], dim=1))
                           for k in range(self._num_components)]

            if (self._w5_method == "retscale"):
                # [2] Schaul et al.: divide the value-loss RESIDUAL by sigma_k,
                # i.e. the squared-error loss by sigma_k^2. The statistics are
                # folded with this batch's targets first; gamma is 0 at a true
                # terminal (FAIL/SUCC) and the discount elsewhere, since TIME
                # truncation still bootstraps.
                terminal = (succ_mask | fail_mask)
                gammas = torch.where(
                    terminal, torch.zeros_like(tar_vals[..., 0]),
                    torch.full_like(tar_vals[..., 0], self._discount))
                if (not self._w5_retscale_folded):
                    # the ALL-TIME estimate should see each collect once, not
                    # once per replayed minibatch
                    self._w5_retscale.update_all_time(
                        reward_components, gammas, tar_vals)
                    self._w5_retscale_folded = True
                # ...but max(sigma_all, sigma_batch, floor) is the paper's
                # replay safeguard and has to be recomputed for THIS batch,
                # otherwise a batch carrying a large reward is divided by a
                # stale scale (Schaul et al. Sec 3.2)
                self._w5_retscale.refresh_scale(
                    reward_components, gammas, tar_vals)
                scale = self._w5_retscale.scale(tar_vals.dtype, tar_vals.device)
                task_losses = [task_losses[k] / (scale[k] ** 2)
                               for k in range(self._num_components)]

        info = {
            "critic_loss": torch.mean(torch.stack(task_losses)).detach(),
            "tar_val_mean": torch.mean(tar_vals).detach(),
        }
        for k, name in enumerate(self._toppo_task_names):
            info["critic_loss_{}".format(name)] = task_losses[k].detach()
            info["tar_val_mean_{}".format(name)] = torch.mean(tar_vals[..., k]).detach()

        if (self._toppo_critic_method == "default"):
            loss = self._toppo_critic_coef * torch.mean(torch.stack(task_losses))
            self._critic_optimizer.step(loss)
        else:
            self._toppo_optimizer_step(
                optimizer=self._critic_optimizer,
                task_losses=task_losses,
                method=self._toppo_critic_method,
                diag_prefix="toppo_critic",
                info=info,
                scope_params=self._model.get_critic_shared_params(),
                grad_scale=self._toppo_critic_coef,
            )
        return info

    def _update_actor_step(self, batch):
        norm_obs = self._obs_norm.normalize(batch["obs"])
        alpha = self._model.get_alpha().detach()

        a_dist = self._model.eval_actor(norm_obs)
        a, logp = a_dist.rsample_with_log_prob()
        q = self._model.eval_critic_components(norm_obs, a, normalized=False)
        q = self._aggregate_critics(q)

        use_component_actor = self._toppo_enabled and self._toppo_actor_components
        entropy_term = (alpha / self._num_components) * logp
        task_losses = [torch.mean(entropy_term - q[..., k])
                       for k in range(self._num_components)]
        actor_loss = torch.sum(torch.stack(task_losses))

        info = {
            "actor_loss": actor_loss.detach(),
            "actor_logp": torch.mean(logp).detach(),
            "alpha": alpha.detach(),
        }

        if (use_component_actor and self._toppo_actor_method != "default"):
            self._toppo_optimizer_step(
                optimizer=self._actor_optimizer,
                task_losses=task_losses,
                method=self._toppo_actor_method,
                diag_prefix="toppo_actor",
                info=info,
                scope_params=None,
            )
        else:
            self._actor_optimizer.step(actor_loss)

        alpha_info = self._update_alpha_step(logp.detach())
        info = {**info, **alpha_info}
        return info

    def _update_alpha_step(self, logp):
        if (not self._learn_alpha):
            return {"alpha_loss": torch.zeros(1, device=self._device)}

        log_alpha = self._model._log_alpha
        alpha_loss = -torch.mean(log_alpha * (logp + self._target_entropy))
        self._alpha_optimizer.step(alpha_loss)
        return {"alpha_loss": alpha_loss.detach()}

    def _aggregate_critics(self, q):
        """Reduce [batch, num_qs, K] over the twin-Q ensemble to [batch, K].

        With K > 1 the reduction must pick ONE critic per sample, otherwise
        sum_k min_i Q[i,k] != min_i sum_k Q[i,k] and the summed component
        target no longer equals the scalar SAC soft target. We therefore take
        the critic that minimizes the SUMMED Q and gather its whole component
        vector, which reduces to the standard twin-Q minimum when K == 1.
        """
        if (self._actor_critic_agg == "first"):
            return q[:, 0]

        if (q.shape[-1] == 1):
            return torch.min(q, dim=1).values

        idx = torch.argmin(torch.sum(q, dim=-1), dim=1)
        idx = idx.view(-1, 1, 1).expand(-1, 1, q.shape[-1])
        return torch.gather(q, 1, idx).squeeze(1)

    def _compute_raw_reward_components(self, batch):
        """The UNSCALED reward decomposition ARS collects statistics over."""
        return batch["reward"].unsqueeze(-1)

    def _compute_reward_components(self, batch):
        """What the critic regresses on: the (ARS-scaled) raw components,
        assembled to match the critic head width."""
        return self._apply_ars_scale(self._compute_raw_reward_components(batch))

    def _get_component_terminal_values(self):
        succ = torch.zeros(self._num_components, device=self._device, dtype=torch.float32)
        fail = torch.zeros(self._num_components, device=self._device, dtype=torch.float32)
        succ[0] = self._compute_succ_val()
        fail[0] = self._compute_fail_val()
        return succ, fail

    def _zero_update_info(self):
        """Zero-filled info dict with the same keys as a real update.

        The txt Logger freezes its header set after the first iteration, so
        warmup iterations (before the replay buffer is filled) must emit the
        exact key set that real updates produce.
        """
        zero = torch.zeros(1, device=self._device)
        info = {
            "critic_loss": zero.clone(),
            "tar_val_mean": zero.clone(),
            "actor_loss": zero.clone(),
            "actor_logp": zero.clone(),
            "alpha": self._model.get_alpha().detach(),
            "alpha_loss": zero.clone(),
            "sac_updates": 0.0,
            "replay_filled_frac": float(self._exp_buffer.get_sample_count())
                / float(self._exp_buffer.get_capacity()),
        }
        for name in self._toppo_task_names:
            info["critic_loss_{}".format(name)] = zero.clone()
            info["tar_val_mean_{}".format(name)] = zero.clone()

        if (self._toppo_critic_method != "default"):
            info.update(self._zero_surgery_info("toppo_critic", self._toppo_critic_method))
        if (self._toppo_enabled and self._toppo_actor_components
                and self._toppo_actor_method != "default"):
            info.update(self._zero_surgery_info("toppo_actor", self._toppo_actor_method))
        # ARS keys must be present from the very first logged iteration: the txt
        # Logger asserts that no new key appears after iteration 0.
        info.update(self._ars_info())
        info.update(self._w5_info())
        return info

    def _w5_info(self):
        """w5_norm diagnostics, emitted once per iteration. Like the ARS keys these
        must exist from the first logged iteration (the txt Logger freezes its
        header after iteration 0)."""
        if (self._w5_method == "rewnorm"):
            return self._w5_rewnorm.get_diagnostics(
                self._get_raw_reward_component_names())
        if (self._w5_method == "retscale"):
            return self._w5_retscale.get_diagnostics(self._toppo_task_names)
        if (self._w5_method == "hlgauss"):
            return self._model.get_hlgauss().get_diagnostics()
        return dict()

    def _zero_surgery_info(self, prefix, method):
        zero = torch.zeros(1, device=self._device)
        info = dict()
        if (method == "pcgrad"):
            for key in ["conflict_rate", "pre_grad_norm", "post_grad_norm", "wall_time_ms"]:
                info["{}_{}".format(prefix, key)] = zero.clone()
        elif (method == "fairgrad"):
            for key in ["gram_trace", "weight_entropy", "residual", "solver_converged",
                        "solver_fallback", "used_scipy_fallback", "solver_iters",
                        "rescale_factor", "mean_grad_norm",
                        "combined_grad_norm_pre_rescale", "combined_grad_norm_post_rescale",
                        "wall_time_ms"]:
                info["{}_{}".format(prefix, key)] = zero.clone()
            for name in self._toppo_task_names:
                info["{}_weight_{}".format(prefix, name)] = zero.clone()
        for key in ["grad_scale", "scope_fraction", "combined_grad_norm_pre_clip",
                    "combined_grad_norm_post_clip", "clip_factor"]:
            info["{}_{}".format(prefix, key)] = zero.clone()
        return info
