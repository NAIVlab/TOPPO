"""PCGrad: Projecting Conflicting Gradients for Multi-Task Learning.

Implements the PCGrad algorithm (Yu et al., 2020) which projects
conflicting task gradients to avoid negative transfer.

Reference: https://arxiv.org/abs/2001.06782
"""

from __future__ import annotations

import random
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class PCGradDiagnostics:
    """Diagnostic information from Euclidean PCGrad projection."""
    pre_grad_norm: float = 0.0
    post_grad_norm: float = 0.0
    grad_shrink_ratio: float = 1.0
    conflict_rate: float = 0.0
    wall_time_ms: float = 0.0

    def as_dict(self, namespace: str = "pcgrad") -> Dict[str, float]:
        return {
            f"{namespace}/pre_grad_norm": self.pre_grad_norm,
            f"{namespace}/post_grad_norm": self.post_grad_norm,
            f"{namespace}/grad_shrink_ratio": self.grad_shrink_ratio,
            f"{namespace}/conflict_rate": self.conflict_rate,
            f"{namespace}/wall_time_ms": self.wall_time_ms,
        }


def project_conflicting_gradients_with_info(
    task_gradients: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], PCGradDiagnostics]:
    """Project conflicting task gradients and return PCGrad diagnostics.

    The inner projection loop uses a freshly shuffled task order per outer
    task, matching the original PCGrad formulation (Yu et al., 2020).
    """
    if not task_gradients:
        return [], PCGradDiagnostics()

    t0 = time.perf_counter()

    # On some compute nodes the CUDA path of this projection dies with
    # SIGFPE (any task count; FairGrad on the same nodes is fine). Setting
    # MTRL_PCGRAD_CPU=1 runs the identical arithmetic on CPU as a workaround;
    # the result is mathematically the same projection.
    _cpu = os.environ.get("MTRL_PCGRAD_CPU", "") == "1"
    if _cpu:
        _dev = task_gradients[0].device
        task_gradients = [g.detach().cpu() for g in task_gradients]

    projected_grads = [g.clone() for g in task_gradients]
    num_tasks = len(task_gradients)
    total_pairs = 0
    total_conflicts = 0

    for i in range(num_tasks):
        order = list(range(num_tasks))
        random.shuffle(order)

        for j in order:
            if i == j:
                continue

            total_pairs += 1
            dot = torch.dot(projected_grads[i], task_gradients[j])
            if dot < 0:
                total_conflicts += 1
                norm_sq = torch.dot(task_gradients[j], task_gradients[j])
                if norm_sq > 1e-12:
                    projected_grads[i] -= (dot / norm_sq) * task_gradients[j]

    pre_mean_grad = torch.stack(task_gradients).mean(dim=0)
    post_mean_grad = torch.stack(projected_grads).mean(dim=0)
    pre_grad_norm = float(pre_mean_grad.norm().item())
    post_grad_norm = float(post_mean_grad.norm().item())
    if pre_grad_norm > 1e-12:
        grad_shrink_ratio = post_grad_norm / pre_grad_norm
    else:
        grad_shrink_ratio = 1.0

    if _cpu:
        projected_grads = [g.to(_dev) for g in projected_grads]

    wall_time_ms = (time.perf_counter() - t0) * 1000.0

    diagnostics = PCGradDiagnostics(
        pre_grad_norm=pre_grad_norm,
        post_grad_norm=post_grad_norm,
        grad_shrink_ratio=float(grad_shrink_ratio),
        conflict_rate=float(total_conflicts / max(total_pairs, 1)),
        wall_time_ms=wall_time_ms,
    )
    return projected_grads, diagnostics


def project_conflicting_gradients(
    task_gradients: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Project conflicting task gradients with the PCGrad rule.

    Returns a new list of projected gradient vectors with the same shapes.
    """
    projected_grads, _ = project_conflicting_gradients_with_info(task_gradients)
    return projected_grads


def pcgrad_combine(
    task_grads: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor], PCGradDiagnostics]:
    """Pure PCGrad combiner matching the :data:`CombineFn` contract.

    Returns ``(mean_of_projected, projected_per_task, diagnostics)``.
    """
    if not task_grads:
        return torch.zeros(0), [], PCGradDiagnostics()
    if len(task_grads) == 1:
        return task_grads[0].clone(), [task_grads[0].clone()], PCGradDiagnostics(wall_time_ms=0.0)
    projected, diag = project_conflicting_gradients_with_info(task_grads)
    combined = torch.stack(projected).mean(dim=0)
    return combined, projected, diag
