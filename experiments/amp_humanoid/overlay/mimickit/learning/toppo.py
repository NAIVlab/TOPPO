import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PCGradDiagnostics:
    conflict_rate: float = 0.0
    pre_grad_norm: float = 0.0
    post_grad_norm: float = 0.0
    wall_time_ms: float = 0.0

    def as_dict(self, prefix):
        return {
            f"{prefix}_conflict_rate": self.conflict_rate,
            f"{prefix}_pre_grad_norm": self.pre_grad_norm,
            f"{prefix}_post_grad_norm": self.post_grad_norm,
            f"{prefix}_wall_time_ms": self.wall_time_ms,
        }


@dataclass
class FairGradDiagnostics:
    weights: torch.Tensor
    gram_trace: float = 0.0
    weight_entropy: float = 0.0
    residual: float = 0.0
    solver_converged: bool = False
    solver_fallback: bool = False
    used_scipy_fallback: bool = False
    solver_iters: int = 0
    rescale_factor: float = 1.0
    mean_grad_norm: float = 0.0
    combined_grad_norm_pre_rescale: float = 0.0
    combined_grad_norm_post_rescale: float = 0.0
    wall_time_ms: float = 0.0

    def as_dict(self, prefix, task_names=None):
        data = {
            f"{prefix}_gram_trace": self.gram_trace,
            f"{prefix}_weight_entropy": self.weight_entropy,
            f"{prefix}_residual": self.residual,
            f"{prefix}_solver_converged": float(self.solver_converged),
            f"{prefix}_solver_fallback": float(self.solver_fallback),
            f"{prefix}_used_scipy_fallback": float(self.used_scipy_fallback),
            f"{prefix}_solver_iters": float(self.solver_iters),
            f"{prefix}_rescale_factor": self.rescale_factor,
            f"{prefix}_mean_grad_norm": self.mean_grad_norm,
            f"{prefix}_combined_grad_norm_pre_rescale": self.combined_grad_norm_pre_rescale,
            f"{prefix}_combined_grad_norm_post_rescale": self.combined_grad_norm_post_rescale,
            f"{prefix}_wall_time_ms": self.wall_time_ms,
        }
        if task_names is not None:
            for i, name in enumerate(task_names):
                if i < self.weights.numel():
                    data[f"{prefix}_weight_{name}"] = float(self.weights[i].detach().item())
        return data


class ComponentPopArt(torch.nn.Module):
    """PopArt for a fixed vector of reward components.

    This is adapted from the PopArt module of the TOPPO Meta-World implementation
    (experiments/metaworld_ppo in the TOPPO release), specialized to the AMP
    reward-component axis.
    """

    def __init__(self, num_components, min_std=1e-2, method="welford", ema_decay=3e-4):
        super().__init__()
        if min_std <= 0.0:
            raise ValueError(f"min_std must be positive, got {min_std}.")
        if method not in ("welford", "ema"):
            raise ValueError(f"Unsupported PopArt method: {method}.")
        if method == "ema" and not (0.0 < ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}.")

        self.num_components = int(num_components)
        self.min_std = float(min_std)
        self.method = method
        self.ema_decay = float(ema_decay)

        self.register_buffer("mu", torch.zeros(num_components, dtype=torch.float64))
        self.register_buffer("sigma", torch.ones(num_components, dtype=torch.float64))
        self.register_buffer("m2", torch.zeros(num_components, dtype=torch.float64))
        self.register_buffer("count", torch.zeros(num_components, dtype=torch.int64))
        self.register_buffer("nu", torch.ones(num_components, dtype=torch.float64))

        self.weight = torch.nn.Parameter(torch.ones(num_components, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros(num_components, dtype=torch.float32))

    def affine(self, values):
        weight = self.weight.to(dtype=values.dtype, device=values.device)
        bias = self.bias.to(dtype=values.dtype, device=values.device)
        return values * weight + bias

    def denormalize(self, values):
        sigma = self.sigma.to(dtype=values.dtype, device=values.device)
        mu = self.mu.to(dtype=values.dtype, device=values.device)
        return sigma * values + mu

    def normalize_target(self, targets):
        sigma = self.sigma.to(dtype=targets.dtype, device=targets.device)
        mu = self.mu.to(dtype=targets.dtype, device=targets.device)
        return (targets - mu) / sigma

    @torch.no_grad()
    def update_and_renormalize(self, targets):
        if targets.numel() == 0:
            return
        if targets.shape[-1] != self.num_components:
            raise ValueError(
                f"Expected targets with {self.num_components} components, got {targets.shape[-1]}."
            )

        targets = targets.reshape(-1, self.num_components).detach().double()
        old_mu = self.mu.clone()
        old_sigma = self.sigma.clone()

        if self.method == "welford":
            self._merge_welford(targets)
        else:
            self._merge_ema(targets)

        new_mu = self.mu.clone()
        new_sigma = self.sigma.clone().clamp_min(self.min_std)

        ratio = old_sigma / new_sigma
        self.weight.data.mul_(ratio.to(dtype=self.weight.dtype, device=self.weight.device))
        new_bias = (old_sigma * self.bias.data.double() + old_mu - new_mu) / new_sigma
        self.bias.data.copy_(new_bias.to(dtype=self.bias.dtype, device=self.bias.device))

    @torch.no_grad()
    def _merge_welford(self, targets):
        n = int(targets.shape[0])
        if n == 0:
            return

        batch_mean = targets.mean(dim=0)
        batch_var = targets.var(dim=0, unbiased=False)
        batch_m2 = batch_var * float(n)

        old_count = self.count.double()
        new_count = old_count + float(n)
        delta = batch_mean - self.mu

        self.mu.copy_(self.mu + delta * (float(n) / new_count.clamp_min(1.0)))
        self.m2.copy_(self.m2 + batch_m2 + delta.pow(2) * old_count * float(n) / new_count.clamp_min(1.0))
        self.count.copy_(new_count.to(dtype=torch.int64))
        var = self.m2 / new_count.clamp_min(1.0)
        self.sigma.copy_(var.clamp_min(0.0).sqrt().clamp_min(self.min_std))

    @torch.no_grad()
    def _merge_ema(self, targets):
        n = int(targets.shape[0])
        if n == 0:
            return

        beta = self.ema_decay
        log_decay = math.log1p(-beta)
        decay_n = math.exp(log_decay * n)
        powers = torch.arange(n - 1, -1, -1, device=targets.device, dtype=torch.float64)
        weights = torch.exp(log_decay * powers)

        self.mu.copy_(decay_n * self.mu + beta * (weights.unsqueeze(-1) * targets).sum(dim=0))
        self.nu.copy_(decay_n * self.nu + beta * (weights.unsqueeze(-1) * targets.pow(2)).sum(dim=0))
        var = (self.nu - self.mu.pow(2)).clamp_min(0.0)
        self.sigma.copy_(var.sqrt().clamp_min(self.min_std))


def flatten_param_grads(params):
    flats = []
    for p in params:
        if p.grad is None:
            flats.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        else:
            flats.append(p.grad.detach().reshape(-1))
    if len(flats) == 0:
        return torch.zeros(0)
    return torch.cat(flats)


def set_flat_param_grads(params, flat_grad):
    offset = 0
    for p in params:
        n = p.numel()
        grad = flat_grad[offset:offset + n].view_as(p).to(dtype=p.dtype, device=p.device)
        p.grad = grad.clone()
        offset += n
    if offset != flat_grad.numel():
        raise ValueError("Flat gradient size does not match parameter list.")


def make_zero_loss(params):
    loss = None
    for p in params:
        if p.requires_grad:
            term = p.sum() * 0.0
            loss = term if loss is None else loss + term
    if loss is None:
        return torch.tensor(0.0)
    return loss


def pcgrad_combine(task_grads):
    if len(task_grads) == 0:
        return torch.zeros(0), PCGradDiagnostics()
    if len(task_grads) == 1:
        return task_grads[0].clone(), PCGradDiagnostics()

    t0 = time.perf_counter()
    projected = [g.clone() for g in task_grads]
    total_pairs = 0
    conflicts = 0

    for i in range(len(task_grads)):
        order = list(range(len(task_grads)))
        random.shuffle(order)
        for j in order:
            if i == j:
                continue
            total_pairs += 1
            dot = torch.dot(projected[i], task_grads[j])
            if dot < 0:
                norm_sq = torch.dot(task_grads[j], task_grads[j])
                if norm_sq > 1e-12:
                    projected[i] = projected[i] - (dot / norm_sq) * task_grads[j]
                    conflicts += 1

    pre_grad = torch.stack(task_grads).mean(dim=0)
    combined = torch.stack(projected).mean(dim=0)
    diag = PCGradDiagnostics(
        conflict_rate=float(conflicts / max(total_pairs, 1)),
        pre_grad_norm=float(pre_grad.norm().item()),
        post_grad_norm=float(combined.norm().item()),
        wall_time_ms=(time.perf_counter() - t0) * 1000.0,
    )
    return combined, diag


def _solve_fairgrad_ynewton(gram, alpha, eps=1e-8, max_iter=50, tol=1e-8, eta=1e-2):
    inv_alpha = 1.0 / float(alpha)
    y_max = 50.0
    diag = gram.diag().clamp_min(eps)
    y = -(float(alpha) / (float(alpha) + 1.0)) * diag.log()
    y = y.clamp(-y_max, y_max)

    def residual(curr_y):
        x = curr_y.clamp(max=y_max).exp()
        neg = (-curr_y * inv_alpha).clamp(max=700.0)
        return gram @ x - neg.exp()

    curr_res = residual(y).norm()
    for it in range(max_iter):
        if curr_res <= eta or curr_res <= tol:
            return y.clamp(max=y_max).exp(), True, it

        x = y.clamp(max=y_max).exp()
        jac = gram * x.unsqueeze(0)
        idx = torch.arange(gram.shape[0], device=gram.device)
        jac[idx, idx] += inv_alpha * (-y * inv_alpha).clamp(max=700.0).exp()

        try:
            step = torch.linalg.solve(jac, -residual(y))
        except RuntimeError:
            return y.clamp(max=y_max).exp(), False, it

        step_scale = 1.0
        accepted = False
        for _ in range(30):
            new_y = (y + step_scale * step).clamp(max=y_max)
            new_res = residual(new_y).norm()
            if new_res < curr_res * (1.0 - 1e-4 * step_scale):
                y = new_y
                curr_res = new_res
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            return y.clamp(max=y_max).exp(), False, it

    return y.clamp(max=y_max).exp(), bool(curr_res <= max(eta, tol)), max_iter


def _solve_fairgrad_scipy(gram, alpha, eps=1e-8, x_start=None):
    """Continue a stalled y-Newton solve with a SciPy least-squares fallback.

    The residual G w - w^(-1/alpha) solved with scipy.optimize.least_squares
    follows the FairGrad reference implementation (OptMN-Lab/FairGrad,
    methods/weight_methods.py, MIT License, Copyright (c) 2024 OptMN-Lab)."""
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return x_start, False, 0

    gram_np = gram.detach().double().cpu().numpy()
    num_tasks = gram_np.shape[0]
    if x_start is None or not torch.isfinite(x_start).all():
        x_start_np = np.ones(num_tasks, dtype=np.float64) / num_tasks
    else:
        x_start_np = x_start.detach().double().cpu().numpy()
        x_start_np = np.maximum(x_start_np, eps * (1.0 + 1e-12))

    inv_alpha = 1.0 / float(alpha)

    def objective(weights):
        rhs = np.power(1.0 / np.maximum(weights, eps), inv_alpha)
        return gram_np @ weights - rhs

    try:
        result = least_squares(objective, x_start_np, bounds=(eps, np.inf))
    except Exception:
        return x_start, False, 0

    weights = torch.from_numpy(np.asarray(result.x, dtype=np.float64)).to(
        device=gram.device,
        dtype=torch.float64,
    )
    return weights, bool(result.success), int(result.nfev)


def fairgrad_combine(task_grads, alpha=1.0, rescale=False):
    if alpha <= 0.0:
        raise ValueError(f"FairGrad alpha must be positive, got {alpha}.")
    if len(task_grads) == 0:
        return torch.zeros(0), FairGradDiagnostics(weights=torch.zeros(0))
    if len(task_grads) == 1:
        return task_grads[0].clone(), FairGradDiagnostics(
            weights=torch.ones(1, device=task_grads[0].device, dtype=task_grads[0].dtype),
            solver_converged=True,
        )

    t0 = time.perf_counter()
    grad_mat = torch.stack(task_grads)
    grad_f64 = grad_mat.detach().double()
    gram = grad_f64 @ grad_f64.T

    weights, converged, iters = _solve_fairgrad_ynewton(gram, alpha)
    used_scipy_fallback = False
    solver_fallback = False
    if (not converged) or (not torch.isfinite(weights).all()):
        weights, scipy_ok, scipy_iters = _solve_fairgrad_scipy(
            gram,
            alpha,
            x_start=weights,
        )
        used_scipy_fallback = True
        iters = scipy_iters
        converged = scipy_ok and weights is not None and torch.isfinite(weights).all()
    if (not converged) or weights is None or (not torch.isfinite(weights).all()):
        weights = torch.ones(
            grad_mat.shape[0],
            device=grad_mat.device,
            dtype=torch.float64,
        ) / grad_mat.shape[0]
        converged = False
        solver_fallback = True

    weights = weights.clamp_min(1e-8)

    residual = (gram @ weights - weights.clamp_min(1e-8).pow(-1.0 / float(alpha))).norm()
    combined_f64 = (weights.unsqueeze(1) * grad_f64).sum(dim=0)
    mean_grad_norm = grad_f64.mean(dim=0).norm()
    combined_norm_pre_rescale = combined_f64.norm()
    rescale_factor = 1.0
    if rescale:
        combined_norm = combined_f64.norm().clamp_min(1e-12)
        rescale_factor = float((mean_grad_norm / combined_norm).item())
        combined_f64 = combined_f64 * rescale_factor
    combined_norm_post_rescale = combined_f64.norm()

    normalized_weights = weights / weights.sum().clamp_min(1e-12)
    entropy = -(normalized_weights * normalized_weights.clamp_min(1e-12).log()).sum()
    diag = FairGradDiagnostics(
        weights=weights.detach().to(dtype=grad_mat.dtype),
        gram_trace=float(gram.trace().item()),
        weight_entropy=float(entropy.item()),
        residual=float(residual.item()),
        solver_converged=bool(converged),
        solver_fallback=bool(solver_fallback),
        used_scipy_fallback=bool(used_scipy_fallback),
        solver_iters=int(iters),
        rescale_factor=float(rescale_factor),
        mean_grad_norm=float(mean_grad_norm.item()),
        combined_grad_norm_pre_rescale=float(combined_norm_pre_rescale.item()),
        combined_grad_norm_post_rescale=float(combined_norm_post_rescale.item()),
        wall_time_ms=(time.perf_counter() - t0) * 1000.0,
    )
    return combined_f64.to(dtype=grad_mat.dtype), diag
