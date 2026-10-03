# The solver in ``_cagrad_solve`` is adapted from the official CAGrad code,
# https://github.com/Cranial-XIX/CAGrad (method ``cagrad`` in
# mtrl/mtrl_files/cagrad.py, commit dc3d481).  Our changes: it is a free
# function taking ``alpha`` (upstream ``self.cagrad_c``), the SGD step size
# switches at ``num_tasks >= 50`` instead of ``== 50``, the final
# ``/(1 + alpha**2)`` division (upstream ``/(1 + self.cagrad_c**2)``) is
# selected by ``rescale`` (default 1 reproduces upstream; 0 = no division
# and 2 = ``/(1 + alpha)`` are options upstream's ``cagrad`` does not have,
# and no caller in this repo selects them), and it returns
# ``CAGradDiagnostics``.  The upstream file also carries a
# facebookresearch/mtrl copyright line for its agent scaffolding; none of
# that scaffolding is reproduced here.  The adapted code is used under the
# upstream MIT License:
#
#   MIT License
#
#   Copyright (c) 2022 Bo Liu
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the
#   "Software"), to deal in the Software without restriction, including
#   without limitation the rights to use, copy, modify, merge, publish,
#   distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to
#   the following conditions:
#
#   The above copyright notice and this permission notice shall be included
#   in all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""CAGrad: Conflict-Averse Gradient Descent for multi-task learning.

Liu et al., NeurIPS 2021 (https://arxiv.org/abs/2110.14048).  Solves a
dual optimization for a per-task weight vector that balances all tasks
under a conflict-averseness coefficient α.  The solver is adapted from
https://github.com/Cranial-XIX/CAGrad (MIT); see the header above.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch



@dataclass
class CAGradDiagnostics:
    """Diagnostic information from a single CAGrad solve."""
    gram_trace: float = 0.0
    obj_best: float = 0.0
    wall_time_ms: float = 0.0

    def as_dict(self, namespace: str = "cagrad") -> Dict[str, float]:
        return {
            f"{namespace}/gram_trace": self.gram_trace,
            f"{namespace}/obj_best": self.obj_best,
            f"{namespace}/wall_time_ms": self.wall_time_ms,
        }


def _cagrad_solve(
    grad_vec: torch.Tensor,
    num_tasks: int,
    alpha: float,
    rescale: int,
) -> Tuple[torch.Tensor, CAGradDiagnostics]:
    """Solve  min_w  w^T G̅g + c√(w^T GG w)  on the simplex; return combined g.

    ``rescale`` ∈ {0=none, 1=/(1+α²), 2=/(1+α)}.
    """
    t0 = time.perf_counter()

    grads = grad_vec  # (num_tasks, dim)
    GG = grads.mm(grads.t()).cpu()
    gram_trace = float(torch.diag(GG).sum().item())

    scale = (torch.diag(GG) + 1e-4).sqrt().mean()
    GG = GG / scale.pow(2)
    Gg = GG.mean(1, keepdim=True)
    gg = Gg.mean(0, keepdim=True)

    w = torch.zeros(num_tasks, 1, requires_grad=True)
    if num_tasks >= 50:
        w_opt = torch.optim.SGD([w], lr=50, momentum=0.5)
    else:
        w_opt = torch.optim.SGD([w], lr=25, momentum=0.5)

    c = (gg + 1e-4).sqrt() * alpha

    w_best = None
    obj_best = np.inf
    for i in range(21):
        w_opt.zero_grad()
        ww = torch.softmax(w, 0)
        obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
        if obj.item() < obj_best:
            obj_best = obj.item()
            w_best = w.clone()
        if i < 20:
            obj.backward()
            w_opt.step()

    ww = torch.softmax(w_best, 0)
    gw_norm = (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
    lmbda = c.view(-1) / (gw_norm + 1e-4)
    g = (
        (1.0 / num_tasks + ww * lmbda).view(-1, 1).to(grads.device) * grads
    ).sum(0)

    if rescale == 0:
        pass
    elif rescale == 1:
        g = g / (1 + alpha ** 2)
    else:
        g = g / (1 + alpha)

    diag = CAGradDiagnostics(
        gram_trace=gram_trace,
        obj_best=float(obj_best) if np.isfinite(obj_best) else 0.0,
        wall_time_ms=(time.perf_counter() - t0) * 1000.0,
    )
    return g, diag


def cagrad_combine(
    task_grads: List[torch.Tensor],
    alpha: float = 0.5,
    rescale: int = 1,
) -> Tuple[torch.Tensor, None, Optional[CAGradDiagnostics]]:
    """CAGrad combiner: ``(combined, None, CAGradDiagnostics)``."""
    K = len(task_grads)
    if K == 0:
        return torch.zeros(0), None, None
    if K == 1:
        return task_grads[0].clone(), None, CAGradDiagnostics(wall_time_ms=0.0)
    grad_mat = torch.stack(task_grads)
    combined, diag = _cagrad_solve(grad_mat, K, alpha, rescale)
    return combined, None, diag


