"""HL-Gauss categorical value head (Farebrother et al. 2024, arXiv:2403.03950).

The critic outputs ``num_bins`` logits over a fixed support
``[vmin, vmax]``.  A scalar target ``y`` is projected to a categorical
distribution by integrating a Gaussian ``N(y, σ²)`` over each bin
(differences of the CDF at the ``num_bins + 1`` bin edges, renormalized
over the truncated range); training minimizes cross-entropy against that
soft target, and the scalar value is read out as the expectation of the
softmax over bin centers.

Conventions follow the paper's Appendix-A reference implementation:
``support`` holds the m+1 bin EDGES (``linspace(vmin, vmax, m+1)``),
predictions live on the m bin CENTERS, and σ is absolute
(``sigma_ratio × bin_width``; 0.75 spreads mass over ~6 bins).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HLGaussHead(nn.Module):
    def __init__(
        self,
        num_bins: int,
        vmin: float,
        vmax: float,
        sigma_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        if num_bins < 2:
            raise ValueError(f"hlgauss num_bins must be >= 2, got {num_bins}")
        if not (vmax > vmin):
            raise ValueError(f"hlgauss requires vmin < vmax, got [{vmin}, {vmax}]")
        if not (sigma_ratio > 0):
            raise ValueError(f"hlgauss sigma_ratio must be > 0, got {sigma_ratio}")
        self.num_bins = int(num_bins)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.bin_width = (self.vmax - self.vmin) / self.num_bins
        self.sigma = float(sigma_ratio) * self.bin_width

        # persistent=False: support/centers are pure functions of
        # (num_bins, vmin, vmax) and must always match the config the run was
        # started with — restoring them from an old checkpoint while the
        # scalar vmin/vmax/sigma came from a changed config would silently
        # desync the clamp range from the projection grid.
        support = torch.linspace(self.vmin, self.vmax, self.num_bins + 1)
        self.register_buffer("support", support, persistent=False)
        self.register_buffer("centers", (support[:-1] + support[1:]) / 2.0, persistent=False)

    @torch.no_grad()
    def to_probs(self, target: torch.Tensor) -> torch.Tensor:
        """Project scalar targets ``(B,)`` to soft labels ``(B, num_bins)``.

        Targets are clipped into the support first; out-of-range mass would
        otherwise pile onto an edge bin anyway, but clipping keeps the
        renormalization well-conditioned.
        """
        target = target.clamp(self.vmin, self.vmax).unsqueeze(-1)
        cdf = torch.erf(
            (self.support.unsqueeze(0) - target) / (math.sqrt(2.0) * self.sigma)
        )
        z = cdf[:, -1] - cdf[:, 0]
        return (cdf[:, 1:] - cdf[:, :-1]) / z.unsqueeze(-1)

    def from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Scalar readout ``(B,)``: expectation of softmax over bin centers."""
        return F.softmax(logits, dim=-1) @ self.centers

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-sample cross-entropy ``(B,)`` against the projected target."""
        probs = self.to_probs(target)
        return -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
