# Portions of this file are adapted from MimicKit (https://github.com/xbpeng/MimicKit,
# commit 2ed1e6c), Copyright 2025 Xue Bin Peng, licensed under the Apache License 2.0;
# the builder's mean/log-std layers and their initialization and the Gaussian log-probability
# follow learning/distribution_gaussian_diag.py. Modified by the TOPPO authors, 2026.
import numpy as np
import torch

class DistributionTanhGaussianBuilder(torch.nn.Module):
    def __init__(self, in_size, out_size, init_output_scale=0.01, init_logstd=-1.0,
                 logstd_min=-5.0, logstd_max=2.0):
        super().__init__()
        self._logstd_min = float(logstd_min)
        self._logstd_max = float(logstd_max)

        self._mean_net = torch.nn.Linear(in_size, out_size)
        torch.nn.init.uniform_(self._mean_net.weight, -init_output_scale, init_output_scale)
        torch.nn.init.zeros_(self._mean_net.bias)

        self._logstd_net = torch.nn.Linear(in_size, out_size)
        torch.nn.init.uniform_(self._logstd_net.weight, -init_output_scale, init_output_scale)
        torch.nn.init.constant_(self._logstd_net.bias, float(init_logstd))
        return

    def forward(self, input):
        mean = self._mean_net(input)
        logstd = self._logstd_net(input)
        logstd = torch.clamp(logstd, self._logstd_min, self._logstd_max)
        dist = DistributionTanhGaussian(mean=mean, logstd=logstd)
        return dist

class DistributionTanhGaussian():
    """Diagonal Gaussian squashed by tanh, with the log-det correction.

    The squashed sample lies in [-1, 1], matching the framework's normalized
    action convention (BaseAgent._a_norm maps [-1, 1] onto the Box bounds).
    log-probs are computed from the pre-tanh sample, so no atanh is needed.
    """

    def __init__(self, mean, logstd):
        self._mean = mean
        self._logstd = logstd
        self._std = torch.exp(logstd)
        self._dim = mean.shape[-1]
        return

    @property
    def mean(self):
        return self._mean

    @property
    def stddev(self):
        return self._std

    @property
    def mode(self):
        return torch.tanh(self._mean)

    def rsample_with_log_prob(self):
        noise = torch.randn_like(self._mean)
        pre_tanh = self._mean + self._std * noise
        a = torch.tanh(pre_tanh)

        log_prob = self._gauss_log_prob(pre_tanh)
        log_prob = log_prob - self._tanh_log_det(pre_tanh)
        return a, log_prob

    def sample(self):
        with torch.no_grad():
            a, _ = self.rsample_with_log_prob()
        return a

    def _gauss_log_prob(self, x):
        diff = x - self._mean
        logp = -0.5 * torch.sum(torch.square(diff / self._std), dim=-1)
        logp += -0.5 * self._dim * np.log(2.0 * np.pi) - torch.sum(self._logstd, dim=-1)
        return logp

    def _tanh_log_det(self, pre_tanh):
        # log |d tanh(u) / du| = log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u))
        log_det = 2.0 * (np.log(2.0) - pre_tanh - torch.nn.functional.softplus(-2.0 * pre_tanh))
        return torch.sum(log_det, dim=-1)
