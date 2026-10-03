"""Linear feature baseline (per-task ridge regression).

Features ``φ(s, t) = [obs, obs², t, t², t³, 1]`` (Duan et al., 2016 style),
fit separately per task with ridge regression.  Used as an alternative to
the neural critic for vanilla policy-gradient comparisons; no shipped config
enables it.

Follows garage's ``LinearFeatureBaseline``
(https://github.com/rlworkgroup/garage,
src/garage/np/baselines/linear_feature_baseline.py; MIT, Copyright (c) 2019
Reinforcement Learning Working Group), which descends from rllab.  See NOTICE.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class LinearFeatureBaseline:
    """Per-task linear feature baseline ``V(s) = φ(s, t)^T w``."""

    def __init__(
        self,
        obs_dim: int,
        num_tasks: int,
        reg_coeff: float = 1e-5,
    ) -> None:
        self.obs_dim = obs_dim
        self.num_tasks = num_tasks
        self.reg_coeff = reg_coeff
        self.feature_dim = 2 * obs_dim + 4  # obs + obs² + t + t² + t³ + 1
        self._weights: dict[int, np.ndarray] = {
            i: np.zeros(self.feature_dim) for i in range(num_tasks)
        }

    def _build_features(
        self,
        obs: np.ndarray,
        timesteps: np.ndarray,
    ) -> np.ndarray:
        n = obs.shape[0]
        obs = np.clip(obs, -10, 10)
        t_norm = timesteps.astype(np.float64) / 100.0

        features = np.zeros((n, self.feature_dim), dtype=np.float64)
        features[:, :self.obs_dim] = obs
        features[:, self.obs_dim:2 * self.obs_dim] = obs ** 2
        idx = 2 * self.obs_dim
        features[:, idx] = t_norm
        features[:, idx + 1] = t_norm ** 2
        features[:, idx + 2] = t_norm ** 3
        features[:, idx + 3] = 1.0
        return features

    def fit(
        self,
        obs: np.ndarray,
        returns: np.ndarray,
        timesteps: np.ndarray,
        task_indices: np.ndarray,
    ) -> None:
        """Per-task ridge regression with NaN retry."""
        features = self._build_features(obs, timesteps)

        for task_id in range(self.num_tasks):
            mask = task_indices == task_id
            if mask.sum() < 2:
                continue

            phi = features[mask]
            y = returns[mask].astype(np.float64)

            reg = self.reg_coeff
            for _ in range(5):
                A = phi.T @ phi + reg * np.eye(self.feature_dim, dtype=np.float64)
                b = phi.T @ y
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=-1)
                if not np.any(np.isnan(w)):
                    break
                reg *= 10
            self._weights[task_id] = w

    def predict(
        self,
        obs: np.ndarray,
        timesteps: np.ndarray,
        task_indices: np.ndarray,
    ) -> np.ndarray:
        features = self._build_features(obs, timesteps)
        values = np.zeros(obs.shape[0], dtype=np.float64)

        for task_id in range(self.num_tasks):
            mask = task_indices == task_id
            if mask.sum() == 0:
                continue
            values[mask] = features[mask] @ self._weights[task_id]

        return values


class LinearBaselineCriticWrapper(nn.Module):
    """nn.Module wrapper exposing :class:`LinearFeatureBaseline` as a critic."""

    def __init__(self, obs_dim: int, num_tasks: int, reg_coeff: float = 1e-5) -> None:
        super().__init__()
        self.baseline = LinearFeatureBaseline(
            obs_dim=obs_dim, num_tasks=num_tasks, reg_coeff=reg_coeff,
        )
        self._num_tasks = num_tasks
        # Dummy parameter — Tianshou requires at least one.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self._cached_values: Optional[torch.Tensor] = None

    def set_predictions(self, values: torch.Tensor) -> None:
        self._cached_values = values.detach()

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        act: np.ndarray | torch.Tensor | None = None,
        info: dict | None = None,
    ) -> torch.Tensor:
        if self._cached_values is not None:
            return self._cached_values.unsqueeze(-1)

        # Fallback path used by metrics like explained_variance.
        if isinstance(obs, torch.Tensor):
            obs_np = obs.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs)
        if obs_np.ndim == 1:
            obs_np = obs_np[np.newaxis, :]

        task_indices = obs_np[:, -self._num_tasks:].argmax(axis=-1)
        raw_obs = obs_np[:, :-self._num_tasks]
        timesteps = np.zeros(obs_np.shape[0], dtype=np.float64)
        values = self.baseline.predict(raw_obs, timesteps, task_indices)

        device = self._dummy.device
        return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(-1)
