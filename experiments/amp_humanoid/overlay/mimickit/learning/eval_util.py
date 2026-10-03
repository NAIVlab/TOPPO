import math

import torch

import envs.base_env as base_env


class FeatureReservoir:
    """Uniform fixed-capacity sample of a streaming feature tensor."""

    def __init__(self, capacity, seed, device):
        self.capacity = max(int(capacity), 0)
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))
        self._keys = None
        self._values = None
        self.seen = 0

    def add(self, values, mask=None):
        if self.capacity == 0:
            return
        values = values.detach()
        if mask is not None:
            values = values[mask]
        values = values.reshape(-1, values.shape[-1])
        count = int(values.shape[0])
        if count == 0:
            return

        keys = torch.rand(
            count,
            generator=self.generator,
            device=self.device,
            dtype=torch.float32,
        )
        if self._values is not None:
            values = torch.cat([self._values, values], dim=0)
            keys = torch.cat([self._keys, keys], dim=0)
        if values.shape[0] > self.capacity:
            keys, indices = torch.topk(keys, self.capacity, sorted=False)
            values = values[indices]
        self._keys = keys
        self._values = values
        self.seen += count

    def values(self):
        if self._values is None:
            return torch.empty(0, 0, device=self.device)
        return self._values


class EpisodeStats:
    """Accumulates exact per-environment episode statistics on device."""

    def __init__(self, metric_names, num_envs, device):
        self.metric_names = list(metric_names)
        self.num_envs = int(num_envs)
        self.device = device
        num_metrics = len(self.metric_names)

        self._return_buf = torch.zeros(
            [self.num_envs, num_metrics], device=device, dtype=torch.float64
        )
        self._length_buf = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        self._eps_per_env = torch.zeros(self.num_envs, device=device, dtype=torch.long)

        self._sum = torch.zeros(num_metrics, device=device, dtype=torch.float64)
        self._sum_sq = torch.zeros(num_metrics, device=device, dtype=torch.float64)
        self._length_sum = torch.zeros(1, device=device, dtype=torch.float64)
        self._length_sum_sq = torch.zeros(1, device=device, dtype=torch.float64)
        self._done_counts = torch.zeros(len(base_env.DoneFlags), device=device, dtype=torch.long)
        self._episodes = 0
        self._record_env_ids = []
        self._record_episode_indices = []
        self._record_values = []
        self._record_lengths = []
        self._record_done_flags = []

    def update(self, values, done, active_mask=None):
        if values.ndim != 2 or values.shape != self._return_buf.shape:
            raise ValueError(
                f"Expected values with shape {tuple(self._return_buf.shape)}, "
                f"got {tuple(values.shape)}."
            )
        if done.shape != self._length_buf.shape:
            raise ValueError(
                f"Expected done with shape {tuple(self._length_buf.shape)}, got {tuple(done.shape)}."
            )

        if active_mask is None:
            active_mask = torch.ones_like(done, dtype=torch.bool)
        else:
            active_mask = active_mask.to(device=self.device, dtype=torch.bool)

        self._return_buf[active_mask] += values[active_mask].double()
        self._length_buf[active_mask] += 1

        completed = active_mask & (done != base_env.DoneFlags.NULL.value)
        completed_ids = completed.nonzero(as_tuple=False).flatten()
        if completed_ids.numel() == 0:
            return

        returns = self._return_buf[completed_ids]
        lengths = self._length_buf[completed_ids].double()
        episode_indices = self._eps_per_env[completed_ids]
        flags = done[completed_ids].long()

        self._record_env_ids.append(completed_ids.detach().cpu())
        self._record_episode_indices.append(episode_indices.detach().cpu())
        self._record_values.append(returns.detach().cpu())
        self._record_lengths.append(lengths.detach().cpu().long())
        self._record_done_flags.append(flags.detach().cpu())

        self._sum += returns.sum(dim=0)
        self._sum_sq += returns.square().sum(dim=0)
        self._length_sum += lengths.sum()
        self._length_sum_sq += lengths.square().sum()

        self._done_counts += torch.bincount(flags, minlength=len(base_env.DoneFlags))
        self._eps_per_env[completed_ids] += 1
        self._episodes += int(completed_ids.numel())

        self._return_buf[completed_ids] = 0.0
        self._length_buf[completed_ids] = 0

    def get_episodes(self):
        return self._episodes

    def get_eps_per_env(self):
        return self._eps_per_env

    def get_mean_ep_len(self):
        if self._episodes == 0:
            return 0.0
        return float((self._length_sum / self._episodes).item())

    def get_done_rate(self, flag):
        if self._episodes == 0:
            return 0.0
        return float((self._done_counts[flag.value].double() / self._episodes).item())

    def get_metric_stats(self, metric_name):
        idx = self.metric_names.index(metric_name)
        return _summary_from_moments(
            self._sum[idx], self._sum_sq[idx], self._episodes
        )

    def get_episode_records(self):
        if self._episodes == 0:
            records = {
                "env_id": torch.zeros(0, dtype=torch.int64).numpy(),
                "episode_index": torch.zeros(0, dtype=torch.int64).numpy(),
                "episode_length": torch.zeros(0, dtype=torch.int64).numpy(),
                "done_flag": torch.zeros(0, dtype=torch.int64).numpy(),
            }
            for name in self.metric_names:
                records[name] = torch.zeros(0, dtype=torch.float64).numpy()
            return records

        values = torch.cat(self._record_values, dim=0)
        records = {
            "env_id": torch.cat(self._record_env_ids, dim=0).numpy(),
            "episode_index": torch.cat(self._record_episode_indices, dim=0).numpy(),
            "episode_length": torch.cat(self._record_lengths, dim=0).numpy(),
            "done_flag": torch.cat(self._record_done_flags, dim=0).numpy(),
        }
        for i, name in enumerate(self.metric_names):
            records[name] = values[:, i].numpy()
        return records


class TensorMoments:
    """Streaming scalar moments for per-step or terminal metrics."""

    def __init__(self):
        self._sum = {}
        self._sum_sq = {}
        self._count = {}

    def update(self, values, mask=None):
        for name, value in values.items():
            flat = value.detach().double().reshape(-1)
            if mask is not None:
                flat = flat[mask.reshape(-1)]
            if flat.numel() == 0:
                continue

            if name not in self._sum:
                self._sum[name] = torch.zeros(1, device=flat.device, dtype=torch.float64)
                self._sum_sq[name] = torch.zeros(1, device=flat.device, dtype=torch.float64)
                self._count[name] = 0

            self._sum[name] += flat.sum()
            self._sum_sq[name] += flat.square().sum()
            self._count[name] += int(flat.numel())

    def summary(self, suffix):
        result = {}
        for name in sorted(self._sum.keys()):
            stats = _summary_from_moments(
                self._sum[name], self._sum_sq[name], self._count[name]
            )
            result[f"{name}_{suffix}_mean"] = stats["mean"]
            result[f"{name}_{suffix}_std"] = stats["std"]
        return result


def compute_style_distribution_metrics(policy_obs, demo_obs, projection_dim=64, seed=260511473):
    """Computes model-independent distribution distances in AMP feature space."""

    if policy_obs.ndim != 2 or demo_obs.ndim != 2:
        raise ValueError("policy_obs and demo_obs must both be rank-2 tensors.")
    if policy_obs.shape[1] != demo_obs.shape[1]:
        raise ValueError("policy_obs and demo_obs must have the same feature dimension.")
    n = min(policy_obs.shape[0], demo_obs.shape[0])
    if n < 2:
        return {
            "style_samples": int(n),
            "style_feature_mean_rms": float("nan"),
            "style_feature_std_rms": float("nan"),
            "style_projected_fid": float("nan"),
        }

    policy = policy_obs[:n].detach().cpu().double()
    demo = demo_obs[:n].detach().cpu().double()
    demo_mean = demo.mean(dim=0)
    demo_std = demo.std(dim=0, unbiased=False).clamp_min(1e-3)
    policy_z = (policy - demo_mean) / demo_std
    demo_z = (demo - demo_mean) / demo_std

    feature_mean_rms = policy_z.mean(dim=0).square().mean().sqrt()
    feature_std_rms = (
        policy_z.std(dim=0, unbiased=False) - demo_z.std(dim=0, unbiased=False)
    ).square().mean().sqrt()

    proj_dim = min(int(projection_dim), policy_z.shape[1], n - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    projection = torch.randn(
        policy_z.shape[1], proj_dim, generator=generator, dtype=torch.float64
    ) / math.sqrt(float(policy_z.shape[1]))
    policy_proj = policy_z @ projection
    demo_proj = demo_z @ projection
    projected_fid = _frechet_gaussian_distance(policy_proj, demo_proj)

    return {
        "style_samples": int(n),
        "style_feature_mean_rms": float(feature_mean_rms.item()),
        "style_feature_std_rms": float(feature_std_rms.item()),
        "style_projected_fid": float(projected_fid),
    }


def _summary_from_moments(total, total_sq, count):
    if count <= 0:
        return {"mean": 0.0, "std": 0.0, "sem": 0.0, "ci95": 0.0}

    mean = total / count
    if count > 1:
        variance = (total_sq - total.square() / count) / (count - 1)
        variance = variance.clamp_min(0.0)
        std = variance.sqrt()
    else:
        std = torch.zeros_like(mean)
    sem = std / math.sqrt(float(count))
    return {
        "mean": float(mean.item()),
        "std": float(std.item()),
        "sem": float(sem.item()),
        "ci95": float((1.96 * sem).item()),
    }


def _frechet_gaussian_distance(x, y):
    mean_x = x.mean(dim=0)
    mean_y = y.mean(dim=0)
    cov_x = _covariance(x)
    cov_y = _covariance(y)
    sqrt_cov_x = _symmetric_matrix_sqrt(cov_x)
    middle = sqrt_cov_x @ cov_y @ sqrt_cov_x
    middle = 0.5 * (middle + middle.T)
    covariance_term = torch.trace(cov_x + cov_y - 2.0 * _symmetric_matrix_sqrt(middle))
    distance = (mean_x - mean_y).square().sum() + covariance_term
    return float(distance.clamp_min(0.0).item())


def _covariance(x):
    centered = x - x.mean(dim=0)
    return centered.T @ centered / max(x.shape[0] - 1, 1)


def _symmetric_matrix_sqrt(matrix):
    matrix = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = eigvals.clamp_min(0.0).sqrt()
    return (eigvecs * eigvals.unsqueeze(0)) @ eigvecs.T
