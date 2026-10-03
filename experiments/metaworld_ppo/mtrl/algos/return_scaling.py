"""Per-task return-based scaling (Schaul et al. 2021, arXiv:2105.05347).

Value-loss residuals for task ``i`` are divided by

    s_i = max(σ_i, σ_batch_i, min_sigma),
    σ_i² = Var_i[R] + Var_i[γ] · E_i[G²],

where the statistics run over ALL data ever seen (cumulative, fp64 — the
paper is explicit on both), γ_t is the per-step discount treated as a
random variable (0 at true terminal steps, the constant γ elsewhere), and
G_t is the per-step return.  ``σ_batch_i`` applies the same formula to the
current collect only and guards the first appearance of a large reward
before the all-time statistics catch up; ``min_sigma`` (≈ the value net's
noise level at init, 1e-2) guards the pre-reward regime.

Per-task independent statistics mirror the paper's multi-head experiment
(App. A.4): "value statistics are accumulated separately for each value
head, and determine the scaling factor for that head".

Deviation from the paper as written: G_t uses the raw GAE
λ-returns — the quantity the critic actually regresses on — rather than
pure episodic Monte-Carlo returns, which do not exist under MetaWorld's
time-limit-truncated episodes.  All buffers live in ``state_dict`` so
checkpoint resume keeps the accumulators.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _sigma_sq(
    n: torch.Tensor,
    s_r: torch.Tensor,
    s_r2: torch.Tensor,
    s_g: torch.Tensor,
    s_g2: torch.Tensor,
    s_G2: torch.Tensor,
) -> torch.Tensor:
    """σ² = Var[R] + Var[γ]·E[G²] from raw fp64 moment sums (0 where n=0)."""
    n_safe = n.clamp(min=1.0)
    var_r = (s_r2 / n_safe - (s_r / n_safe).pow(2)).clamp(min=0.0)
    var_g = (s_g2 / n_safe - (s_g / n_safe).pow(2)).clamp(min=0.0)
    mean_G2 = s_G2 / n_safe
    return torch.where(n > 0, var_r + var_g * mean_G2, torch.zeros_like(var_r))


class ReturnScalingModule(nn.Module):
    def __init__(self, num_tasks: int, min_sigma: float = 1e-2) -> None:
        super().__init__()
        if not (min_sigma > 0.0):
            raise ValueError(
                f"return_scaling min_sigma must be > 0 (got {min_sigma!r}); "
                "a non-positive floor divides the value loss by zero before "
                "any reward is seen."
            )
        self.num_tasks = int(num_tasks)
        self.min_sigma = float(min_sigma)
        f64 = dict(dtype=torch.float64)
        self.register_buffer("n",    torch.zeros(num_tasks, **f64))
        self.register_buffer("s_r",  torch.zeros(num_tasks, **f64))
        self.register_buffer("s_r2", torch.zeros(num_tasks, **f64))
        self.register_buffer("s_g",  torch.zeros(num_tasks, **f64))
        self.register_buffer("s_g2", torch.zeros(num_tasks, **f64))
        self.register_buffer("s_G2", torch.zeros(num_tasks, **f64))
        # Frozen once per collect; every minibatch of that collect reads it.
        self.register_buffer("last_scale", torch.full((num_tasks,), min_sigma, **f64))

    @torch.no_grad()
    def update_and_refresh_scale(
        self,
        task_idx: torch.Tensor,
        rewards: torch.Tensor,
        gammas: torch.Tensor,
        returns: torch.Tensor,
    ) -> None:
        """Fold one collect into the all-time statistics and refresh ``last_scale``.

        All inputs are flat ``(N,)`` tensors over the collect: per-sample task
        index, raw env reward, per-step discount (0 at terminal steps), and
        raw value target G.
        """
        device = self.n.device
        task_idx = task_idx.to(device=device, dtype=torch.long)
        r = rewards.detach().to(device=device, dtype=torch.float64)
        g = gammas.detach().to(device=device, dtype=torch.float64)
        G = returns.detach().to(device=device, dtype=torch.float64)

        ones = torch.ones_like(r)
        batch = {name: torch.zeros_like(self.n) for name in
                 ("n", "s_r", "s_r2", "s_g", "s_g2", "s_G2")}
        batch["n"].index_add_(0, task_idx, ones)
        batch["s_r"].index_add_(0, task_idx, r)
        batch["s_r2"].index_add_(0, task_idx, r * r)
        batch["s_g"].index_add_(0, task_idx, g)
        batch["s_g2"].index_add_(0, task_idx, g * g)
        batch["s_G2"].index_add_(0, task_idx, G * G)

        for name, val in batch.items():
            getattr(self, name).add_(val)

        sigma_all = _sigma_sq(
            self.n, self.s_r, self.s_r2, self.s_g, self.s_g2, self.s_G2
        ).sqrt()
        sigma_batch = _sigma_sq(
            batch["n"], batch["s_r"], batch["s_r2"],
            batch["s_g"], batch["s_g2"], batch["s_G2"],
        ).sqrt()
        self.last_scale.copy_(
            torch.maximum(sigma_all, sigma_batch).clamp(min=self.min_sigma)
        )

    def get_diagnostics(self) -> Dict[str, torch.Tensor]:
        """Detached CPU snapshot of per-task all-time σ and the active scale."""
        sigma = _sigma_sq(
            self.n, self.s_r, self.s_r2, self.s_g, self.s_g2, self.s_G2
        ).sqrt()
        return {
            "sigma": sigma.detach().cpu().clone(),
            "scale": self.last_scale.detach().cpu().clone(),
        }
