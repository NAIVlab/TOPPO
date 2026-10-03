"""Critic-side scale-normalization alternatives to PopArt (the `w5_*` arms).

Three cheaper schemes that can replace PopArt on the component critic:

  [1] Reward normalization (openai/baselines#538, `VecNormalize`): divide the
      raw reward by the running std of the DISCOUNTED-RETURN accumulator, no
      mean subtraction, clip to +-10.
  [2] Return-based scaling (Schaul et al. 2021, arXiv:2105.05347): divide the
      value-loss residual by sigma, sigma^2 = Var[R] + Var[gamma] * E[G^2],
      cumulative fp64 statistics, no tunable hyperparameters.
  [3] HL-Gauss (Farebrother et al. 2024, arXiv:2403.03950): replace the value
      MSE with cross-entropy against a Gaussian-smoothed categorical
      projection of the scalar target on a fixed support; read the scalar
      value back as the expectation over bin centers.

Semantics follow the Meta-World implementation of the same three methods in
the TOPPO release (experiments/metaworld_ppo: `mtrl/algos/return_scaling.py`,
`mtrl/models/hlgauss.py`, `mtrl/envs/metaworld_wrapper.py`), so both
experiment families define them identically. The only adaptation is the
axis: there the statistics are PER TASK, here they are PER REWARD COMPONENT
(task, style) -- the same axis PopArt, FairGrad and PCGrad use in this
benchmark.

All state lives in registered buffers so it travels with the checkpoint.

Credits: the reward-normalization update order follows openai/baselines
`VecNormalize` (MIT License, Copyright (c) 2017 OpenAI); return-based scaling
follows Schaul et al. (2021); HL-Gauss follows Farebrother et al. (2024).
"""

import math

import torch
import torch.nn.functional as F


class ComponentRewardNorm(torch.nn.Module):
    """[1] baselines VecNormalize, per reward component.

    Keeps a per-(env, component) discounted-return accumulator and a running
    mean/var over its values, then divides the raw reward by that std. The
    update order is verbatim from `VecNormalize.step_wait`: accumulate with
    the RAW reward, fold into the running statistics, normalize, then zero the
    accumulator wherever the episode ended.
    """

    def __init__(self, num_components, num_envs, discount, clip=10.0, device="cpu"):
        super().__init__()
        self.num_components = int(num_components)
        self.discount = float(discount)
        self.clip = float(clip)

        f64 = dict(dtype=torch.float64, device=device)
        # NOT persistent: baselines does not checkpoint the accumulator, envs
        # are reset on resume anyway, and persisting it would weld the
        # checkpoint to num_envs (a 256-env analysis pass could not load a
        # 4096-env checkpoint). mean/var/count DO travel with the checkpoint.
        self.register_buffer("ret", torch.zeros([num_envs, num_components],
                                                **f64), persistent=False)
        self.register_buffer("mean", torch.zeros(num_components, **f64))
        self.register_buffer("var", torch.ones(num_components, **f64))
        self.register_buffer("count", torch.full((num_components,), 1e-4, **f64))

    @torch.no_grad()
    def normalize(self, rewards, done_mask):
        """rewards: [num_envs, K] raw. done_mask: [num_envs] bool. -> [num_envs, K]."""
        assert(done_mask.dtype == torch.bool), \
            "done_mask must be bool: an integer mask makes self.ret[done_mask] " \
            "advanced indexing, which resets the wrong environments."
        r = rewards.detach().to(dtype=torch.float64)
        self.ret.mul_(self.discount).add_(r)

        n = self.ret.shape[0]
        batch_mean = self.ret.mean(dim=0)
        batch_var = self.ret.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + n
        new_mean = self.mean + delta * (n / total)
        m_a = self.var * self.count
        m_b = batch_var * n
        new_var = (m_a + m_b + delta.pow(2) * self.count * n / total) / total
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total)

        normed = r / torch.sqrt(self.var + 1e-8)
        normed = normed.clamp(-self.clip, self.clip)
        self.ret[done_mask] = 0.0
        return normed.to(dtype=rewards.dtype)

    def get_diagnostics(self, names):
        std = torch.sqrt(self.var + 1e-8)
        return {"w5_rewnorm_std_{}".format(names[k]): float(std[k].item())
                for k in range(self.num_components)}


class ComponentReturnScaling(torch.nn.Module):
    """[2] Schaul et al. return-based scaling, per reward component.

    sigma_k^2 = Var_k[R] + Var_k[gamma] * E_k[G^2] over ALL data seen (fp64),
    and the active scale is max(sigma_all, sigma_batch, min_sigma) -- the
    batch term guards the first appearance of a large reward, the floor guards
    the pre-reward regime. Value-loss residuals for component k are divided by
    that scale. No tunable hyperparameters beyond the floor.
    """

    def __init__(self, num_components, min_sigma=1e-2, device="cpu"):
        super().__init__()
        if not (min_sigma > 0.0):
            raise ValueError("min_sigma must be > 0, got {}".format(min_sigma))
        self.num_components = int(num_components)
        self.min_sigma = float(min_sigma)
        f64 = dict(dtype=torch.float64, device=device)
        for name in ("n", "s_r", "s_r2", "s_g", "s_g2", "s_G2"):
            self.register_buffer(name, torch.zeros(num_components, **f64))
        self.register_buffer("last_scale",
                             torch.full((num_components,), float(min_sigma), **f64))

    @staticmethod
    def _sigma_sq(n, s_r, s_r2, s_g, s_g2, s_G2):
        n_safe = n.clamp(min=1.0)
        var_r = (s_r2 / n_safe - (s_r / n_safe).pow(2)).clamp(min=0.0)
        var_g = (s_g2 / n_safe - (s_g / n_safe).pow(2)).clamp(min=0.0)
        mean_G2 = s_G2 / n_safe
        return torch.where(n > 0, var_r + var_g * mean_G2, torch.zeros_like(var_r))

    def _batch_moments(self, rewards, gammas, returns):
        r = rewards.detach().reshape(-1, self.num_components).to(torch.float64)
        G = returns.detach().reshape(-1, self.num_components).to(torch.float64)
        g = gammas.detach().reshape(-1, 1).to(torch.float64).expand_as(r)
        return {
            "n": torch.full_like(self.n, float(r.shape[0])),
            "s_r": r.sum(dim=0), "s_r2": (r * r).sum(dim=0),
            "s_g": g.sum(dim=0), "s_g2": (g * g).sum(dim=0),
            "s_G2": (G * G).sum(dim=0),
        }

    @torch.no_grad()
    def update_all_time(self, rewards, gammas, returns):
        """Fold one collect into the all-time statistics.

        Kept separate from `refresh_scale` because under replay the two have
        different natural frequencies: the all-time estimate should see each
        collect once, but the batch safeguard has to see every batch.
        """
        for name, val in self._batch_moments(rewards, gammas, returns).items():
            getattr(self, name).add_(val)

    @torch.no_grad()
    def refresh_scale(self, rewards, gammas, returns):
        """Active scale for THIS batch: max(sigma_all, sigma_batch, min_sigma).

        Schaul et al. Sec 3.2 make the batch term imperative in replay
        settings: a large reward has to enter the scale BEFORE the update that
        uses it, otherwise that update divides by a stale, much smaller sigma.
        So this is called per critic minibatch, not once per iteration.
        """
        b = self._batch_moments(rewards, gammas, returns)
        sigma_all = self._sigma_sq(self.n, self.s_r, self.s_r2,
                                   self.s_g, self.s_g2, self.s_G2).sqrt()
        sigma_batch = self._sigma_sq(b["n"], b["s_r"], b["s_r2"],
                                     b["s_g"], b["s_g2"], b["s_G2"]).sqrt()
        self.last_scale.copy_(
            torch.maximum(sigma_all, sigma_batch).clamp(min=self.min_sigma))
        return self.last_scale

    @torch.no_grad()
    def update_and_refresh_scale(self, rewards, gammas, returns):
        """Fold + refresh in one step. Correct where the batch IS the collect
        (the on-policy PPO rollout); SAC drives the two separately."""
        self.update_all_time(rewards, gammas, returns)
        return self.refresh_scale(rewards, gammas, returns)

    def scale(self, dtype, device):
        return self.last_scale.to(dtype=dtype, device=device)

    def get_diagnostics(self, names):
        sigma = self._sigma_sq(self.n, self.s_r, self.s_r2,
                               self.s_g, self.s_g2, self.s_G2).sqrt()
        out = {}
        for k in range(self.num_components):
            out["w5_retscale_sigma_{}".format(names[k])] = float(sigma[k].item())
            out["w5_retscale_scale_{}".format(names[k])] = float(self.last_scale[k].item())
        return out


class HLGaussHead(torch.nn.Module):
    """[3] HL-Gauss categorical value head, shared across reward components.

    `support` holds the num_bins+1 bin EDGES; predictions live on the bin
    CENTERS; sigma is absolute (sigma_ratio * bin_width, 0.75 spreading mass
    over ~6 bins), following the paper's Appendix-A reference.
    """

    def __init__(self, num_bins, vmin, vmax, sigma_ratio=0.75):
        super().__init__()
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2, got {}".format(num_bins))
        if not (vmax > vmin):
            raise ValueError("requires vmin < vmax, got [{}, {}]".format(vmin, vmax))
        if not (sigma_ratio > 0):
            raise ValueError("sigma_ratio must be > 0, got {}".format(sigma_ratio))
        self.num_bins = int(num_bins)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.bin_width = (self.vmax - self.vmin) / self.num_bins
        self.sigma = float(sigma_ratio) * self.bin_width

        support = torch.linspace(self.vmin, self.vmax, self.num_bins + 1)
        self.register_buffer("support", support, persistent=False)
        self.register_buffer("centers", (support[:-1] + support[1:]) / 2.0,
                             persistent=False)
        # Unlike PopArt, HL-Gauss needs the value range as a hyperparameter.
        # Track how often targets fall outside it so a saturated support is
        # visible in the logs instead of silently biasing the value estimate.
        self.register_buffer("oob_count", torch.zeros((), dtype=torch.float64),
                             persistent=False)
        self.register_buffer("seen_count", torch.zeros((), dtype=torch.float64),
                             persistent=False)

    @classmethod
    def from_config(cls, config):
        """The ONE place the HL-Gauss support defaults live; both families'
        models build their head through here."""
        return cls(num_bins=int(config.get("hlgauss_num_bins", 101)),
                   vmin=float(config.get("hlgauss_vmin", -10.0)),
                   vmax=float(config.get("hlgauss_vmax", 110.0)),
                   sigma_ratio=float(config.get("hlgauss_sigma_ratio", 0.75)))

    @torch.no_grad()
    def to_probs(self, target):
        """[...] scalar targets -> [..., num_bins] soft labels."""
        oob = ((target < self.vmin) | (target > self.vmax))
        self.oob_count += oob.sum().double()
        self.seen_count += float(target.numel())
        target = target.clamp(self.vmin, self.vmax).unsqueeze(-1)
        shape = [1] * (target.dim() - 1) + [self.support.numel()]
        cdf = torch.erf((self.support.view(shape) - target)
                        / (math.sqrt(2.0) * self.sigma))
        z = cdf[..., -1] - cdf[..., 0]
        return (cdf[..., 1:] - cdf[..., :-1]) / z.unsqueeze(-1)

    def from_logits(self, logits):
        """[..., num_bins] -> [...] expectation of the softmax over centers."""
        return F.softmax(logits, dim=-1) @ self.centers.to(logits.dtype)

    def loss(self, logits, target):
        """Per-sample cross-entropy [...] against the projected target."""
        probs = self.to_probs(target)
        return -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)

    def logits_for_value(self, value, floor=1e-12):
        """[num_bins] logits whose softmax is the projection of `value`.

        Used to initialize the head BIAS. A zero bias makes the softmax uniform
        and the read-out the support center, so with an asymmetric support
        (here [-10, 350]) every Q would start at 170 instead of ~0 -- a change
        of initialization on top of the change of loss, which would confound
        the comparison against the zero-initialized PopArt control.
        """
        probs = self.to_probs(torch.as_tensor(float(value)))
        return torch.log(probs.clamp_min(floor))

    def get_diagnostics(self):
        """WINDOWED: the counters are zeroed on read, so the reported fraction
        covers only the interval since the previous call (both families call
        this exactly once per train iteration). An all-time average would hide
        a support that starts saturating late -- 1000 clean batches followed by
        one fully out-of-support batch reads 0.001, not 1.0."""
        seen = float(self.seen_count.item())
        oob = float(self.oob_count.item()) / seen if seen > 0 else 0.0
        self.oob_count.zero_()
        self.seen_count.zero_()
        return {"w5_hlgauss_vmin": self.vmin, "w5_hlgauss_vmax": self.vmax,
                "w5_hlgauss_sigma": self.sigma, "w5_hlgauss_oob_frac": oob}


def build_w5_normalizer(method, num_reward_components, num_critic_components,
                        num_envs, discount, config, device):
    """Return the (rewnorm, retscale) pair for `method`, at most one non-None.

    hlgauss is deliberately NOT built here: it is a value HEAD, so it is owned
    by the model (see `HLGaussHead.from_config`, the single source of its
    defaults). Keeping the methods mutually exclusive is what makes each arm a
    single-knob replacement for PopArt.

    rewnorm acts on the RAW reward decomposition (before an optional collapse
    to a scalar critic); retscale acts on the critic head width.
    """
    method = str(method or "none")
    valid = ("none", "rewnorm", "retscale", "hlgauss")
    assert(method in valid), "w5_norm method must be one of {}".format(valid)

    if (method == "rewnorm"):
        return (ComponentRewardNorm(num_reward_components, num_envs, discount,
                                    clip=float(config.get("rewnorm_clip", 10.0)),
                                    device=device), None)
    if (method == "retscale"):
        return (None, ComponentReturnScaling(
            num_critic_components,
            min_sigma=float(config.get("retscale_min_sigma", 1e-2)),
            device=device))
    return (None, None)
