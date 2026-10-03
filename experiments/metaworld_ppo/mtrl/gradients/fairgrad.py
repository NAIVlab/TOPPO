# Source: https://github.com/OptMN-Lab/fairgrad (commit b58bce8), the official
# FairGrad code.  Only the scipy fallback ``_solve_scipy_fallback`` follows it:
# like ``FairGrad.fairgrad`` in methods/weight_methods.py, it solves the KKT
# residual ``G x - x^(-1/alpha)`` with scipy ``least_squares`` from x0 = 1/K
# under a positivity bound.  Our changes: a free function on an fp64 numpy
# Gram, a lower bound ``eps`` instead of 0 with x clamped to it inside the
# residual, an optional warm start from the last y-Newton iterate, and a
# (solution, success, nfev) return that never raises.  The y-space Newton
# solver, its warm start and every other part of this file are ours.  The
# adapted code is used under the upstream MIT License:
#
#   MIT License
#
#   Copyright (c) 2024 OptMN-Lab
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
"""FairGrad: fairness-aware gradient weighting for multi-task learning.

Implements FairGrad (Ban & Ji, ICML 2024, https://arxiv.org/abs/2402.15638).
Task weights solve the α-fairness KKT equation ``G · w = w^{-1/α}`` on the
per-task gradient Gram matrix; α=1 recovers Nash-MTL's bargaining solution.

Solver: log-parameterized (y-space) projected damped Newton on
``F(y) = G · exp(y) − exp(−y/α) = 0`` with ``x = exp(y)``.  Falls back to
scipy ``least_squares`` when Newton fails.  On dead-task rows the Newton
step for ``y_i`` is exactly α, making iteration count ``O(log x_target / α)``.
The scipy fallback follows https://github.com/OptMN-Lab/fairgrad (MIT); see
the header above.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# --- optional solver benchmark ------------------------------------------------
# Set MTRL_SOLVER_BENCH=<path.csv> to time the scipy reference against y-Newton
# on the SAME Gram matrix, once per production solve.  The scipy solution is
# timed and thrown away, so training dynamics are untouched; the only cost is
# the extra solve.  Off by default: with the variable unset nothing below runs
# and no file is opened.
#
# This re-implements the instrument that produced the archived
# benchmarks/solver/solverbench_rfv1.csv (see realgram_results.md there), with
# the same protocol: both timings are solve-only.  ``t_newton_ms`` covers the
# _solve_ynewton call alone (not the fp64 cast, Gram formation, device-to-host
# copy or any fallback), and the reference is scipy ``least_squares`` from
# x0 = 1/K with bounds (_BENCH_SCIPY_LB, inf).  The ``newton_*`` columns are
# Newton's own result; a production scipy fallback, if one ran, is recorded in
# the ``fallback_*`` columns instead of overwriting them.
_BENCH_PATH = os.environ.get("MTRL_SOLVER_BENCH") or None
_BENCH_LOCK = threading.Lock()
_BENCH_SCIPY_LB = 1e-12   # reference lower bound; production fallback uses eps
# The first nine columns match benchmarks/solver/solverbench_rfv1.csv (booleans
# as 0/1) so the committed measurement and a fresh one can be read by the same
# code; the rest are additions.
_BENCH_HEADER = (
    "K", "t_newton_ms", "newton_iters", "newton_converged",
    "t_scipy_ms", "scipy_nfev", "scipy_success", "scipy_resid", "gram_trace",
    "alpha", "newton_resid",
    "fallback_used", "fallback_ok", "fallback_nfev", "t_fallback_ms",
)
_BENCH_STARTED = False


def _bench_record(row: Dict[str, Any]) -> None:
    """Append one solve to the benchmark CSV (no-op unless MTRL_SOLVER_BENCH)."""
    global _BENCH_STARTED
    with _BENCH_LOCK:
        new = not _BENCH_STARTED
        _BENCH_STARTED = True
        d = os.path.dirname(_BENCH_PATH)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(_BENCH_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(_BENCH_HEADER))
            if new and f.tell() == 0:
                w.writeheader()
            w.writerow({k: row.get(k) for k in _BENCH_HEADER})


# Default solver hyperparameters — tuned for K=10/50, α ∈ [1,5].
_YNEWTON_ETA = 1e-2           # residual target: matches scipy's implicit plateau
_YNEWTON_TOL = 1e-8           # tight tolerance (tighter of the two stops)
_YNEWTON_MAX_ITER = 50        # hard iteration cap
_YNEWTON_Y_MAX = 50.0         # soft cap on y = log(x); exp(50) ≈ 5e21 (fp64 safe)
_YNEWTON_ARMIJO_C1 = 1e-4     # Armijo sufficient-decrease constant
_YNEWTON_LINE_SEARCH_MAX = 30 # backtracking cap per iteration


@dataclass
class FairGradDiagnostics:
    """Diagnostic information from FairGrad weight solving."""
    weights: torch.Tensor  # (num_tasks,)
    gram_trace: float = 0.0
    weight_entropy: float = 0.0
    solver_converged: bool = False
    wall_time_ms: float = 0.0
    regularization_eps: float = 0.0   # logged for schema stability; always 0
    residual: float = 0.0             # ||G·w - w^{-1/α}||
    solver_fallback: bool = False     # solver crashed or output non-finite
    # Per-task effective contribution ‖w_i · g_i‖ — distinguishes a
    # huge-weight × near-zero-grad task (harmless, KKT-bounded) from a
    # moderate-weight × real-grad task (driving the direction).
    contribution_norms: Optional[List[float]] = None
    solver_iters: int = 0
    used_scipy_fallback: bool = False

    def as_dict(self, task_names: Optional[List[str]] = None) -> Dict[str, float]:
        d: Dict[str, float] = {
            "fairgrad/gram_trace": self.gram_trace,
            "fairgrad/weight_entropy": self.weight_entropy,
            "fairgrad/wall_time_ms": self.wall_time_ms,
            "fairgrad/regularization_eps": self.regularization_eps,
            "fairgrad/residual": self.residual,
            "fairgrad/solver_fallback": float(self.solver_fallback),
            "fairgrad/solver_iters": float(self.solver_iters),
            "fairgrad/used_scipy_fallback": float(self.used_scipy_fallback),
        }
        if task_names is not None:
            for i, name in enumerate(task_names):
                d[f"fairgrad/weight/{name}"] = float(self.weights[i])
                if (
                    self.contribution_norms is not None
                    and i < len(self.contribution_norms)
                ):
                    d[f"fairgrad/contribution/{name}"] = float(
                        self.contribution_norms[i]
                    )
        return d


def _solve_ynewton(
    G: np.ndarray,
    alpha: float,
    eps: float,
    eta: float = _YNEWTON_ETA,
    tol: float = _YNEWTON_TOL,
    max_iter: int = _YNEWTON_MAX_ITER,
    y_max: float = _YNEWTON_Y_MAX,
) -> Tuple[np.ndarray, bool, int]:
    """Projected damped Newton in y-space where ``x = exp(y)``.

    Solves ``F(y) = G·exp(y) − exp(−y/α) = 0`` with Armijo backtracking.
    Positivity ``x > 0`` is automatic via the exp map; y is soft-capped at
    ``y_max``.  Warm-start ``y_diag = -(α/(α+1))·log(diag(G))`` is exact
    when G is diagonal and typically ~6 orders of magnitude closer to the
    solution than a uniform cold start.

    Returns ``(x_sol, converged, iters)``; ``x_sol`` is always finite.
    """
    inv_a = 1.0 / float(alpha)

    def F(y: np.ndarray) -> np.ndarray:
        x = np.exp(np.minimum(y, y_max))
        neg_y_over_a = np.minimum(-y * inv_a, 700.0)
        return G @ x - np.exp(neg_y_over_a)

    # NOTE: `eps` is documented as the lower bound on x (the weights), but it is
    # also applied here as a floor on diag(G), which lives in squared-gradient-norm
    # units.  This is load-bearing, not cosmetic: for alpha=1 the warm start is
    # exact when G is diagonal, so a task with ||g_i||^2 < eps has its weight
    # pinned at 1/sqrt(eps) = 1e4 instead of its exact 1/||g_i||.  Because the
    # stopping test below is on the ABSOLUTE residual (eta = 1e-2) and a starved
    # row's residual is tiny precisely because its gradient is tiny, the solver
    # then returns at iteration 0 reporting convergence.
    #
    # Measured on the headline MT50 V2 group, this binds on 10.4% of
    # (step, task) pairs -- but almost entirely on tasks that are already SOLVED
    # (door-close, reach, plate-slide, ... all at 99.6-100% success), whose critic
    # gradients are near zero because there is nothing left to learn.  The six
    # hardest tasks hit it on 0.73% of pairs and `assembly` (0% success) never
    # does.  In these runs the floor therefore acts as a cap on solved tasks, not
    # as starvation of hard ones.
    #
    # Do not "fix" this by lowering eps without re-running: `converged` gates the
    # scipy fallback below, so changing either the floor or the stopping test
    # changes the optimization and the published numbers no longer reproduce.
    diag_G = np.maximum(np.diagonal(G), eps)
    y_diag = -(float(alpha) / (float(alpha) + 1.0)) * np.log(diag_G)
    y = np.minimum(np.maximum(y_diag, -y_max), y_max)
    r = float(np.linalg.norm(F(y)))

    for it in range(max_iter):
        if r <= eta or r <= tol:
            return np.exp(np.minimum(y, y_max)), True, it

        x = np.exp(np.minimum(y, y_max))
        # J_y = G · diag(x) + (1/α) · diag(exp(-y/α)); built directly to
        # avoid materializing a K×K diag tensor.
        J = G * x[np.newaxis, :]
        neg_y_over_a = np.minimum(-y * inv_a, 700.0)
        J[np.diag_indices_from(J)] += inv_a * np.exp(neg_y_over_a)

        try:
            p = np.linalg.solve(J, -F(y))
        except np.linalg.LinAlgError:
            return np.exp(np.minimum(y, y_max)), False, it

        t = 1.0
        for _ in range(_YNEWTON_LINE_SEARCH_MAX):
            y_new = np.minimum(y + t * p, y_max)
            r_new = float(np.linalg.norm(F(y_new)))
            if r_new < r * (1.0 - _YNEWTON_ARMIJO_C1 * t):
                break
            t *= 0.5
        else:
            return np.exp(np.minimum(y, y_max)), False, it

        y = y_new
        r = r_new

    return np.exp(np.minimum(y, y_max)), (r <= max(eta, tol)), max_iter


def _solve_scipy_fallback(
    GTG_np: np.ndarray,
    alpha: float,
    eps: float,
    x_start: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool, int]:
    """scipy least_squares fallback for when y-Newton fails to converge.

    Follows ``FairGrad.fairgrad`` in OptMN-Lab/fairgrad
    methods/weight_methods.py (MIT; see the header of this file).

    Callers should pass the last y-Newton iterate as ``x_start``: dead-task
    rows can be ~50 log-units from uniform 1/K, which scipy's trust-region
    plateau barely survives.
    """
    from scipy.optimize import least_squares

    K = GTG_np.shape[0]
    if x_start is None:
        x_start = np.ones(K, dtype=np.float64) / K
    else:
        # Clamp into scipy's (eps, inf) bound box; Newton may have returned
        # a slightly sub-eps coord that least_squares would reject.
        x_start = np.asarray(x_start, dtype=np.float64)
        x_start = np.maximum(x_start, eps * (1.0 + 1e-12))
    inv_alpha = 1.0 / float(alpha)

    def objfn(x: np.ndarray) -> np.ndarray:
        return GTG_np @ x - np.power(1.0 / np.maximum(x, eps), inv_alpha)

    try:
        res = least_squares(objfn, x_start, bounds=(eps, np.inf))
        x_sol = np.asarray(res.x, dtype=np.float64)
        return x_sol, bool(res.success), int(res.nfev)
    except Exception:
        return x_start, False, 0


def solve_fairgrad_weights(
    grad_matrix: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> FairGradDiagnostics:
    """Solve FairGrad task weights via y-space projected Newton with scipy fallback.

    Args:
        grad_matrix: Per-task gradient vectors, shape (num_tasks, grad_dim).
        alpha: Fairness parameter; α=1 recovers Nash bargaining.
        eps: Lower bound for x (to stay in the positive orthant).  It is ALSO
            applied as a floor on diag(G) inside the y-Newton warm start, which
            caps the weight of any task with ||g_i||^2 < eps at 1/sqrt(eps).
            See the note at that line before changing it.

    Returns ``FairGradDiagnostics`` with solved weights x (not simplex-normalized).
    """
    t0 = time.perf_counter()

    num_tasks = grad_matrix.shape[0]
    device = grad_matrix.device

    if num_tasks == 1:
        w = torch.ones(1, device=device, dtype=grad_matrix.dtype)
        return FairGradDiagnostics(
            weights=w,
            solver_converged=True,
            wall_time_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # Gram in fp64: dead-task rows have ‖g_i‖² that underflow to 0 in fp32,
    # destroying information the solver needs.
    grad_f64 = grad_matrix.detach().double()
    G = grad_f64 @ grad_f64.T
    gram_trace = float(G.trace().item())
    GTG_np = G.cpu().numpy()

    # Newton and the fallback are timed separately for the solver benchmark
    # below; the timers do not touch the solve.
    t_newton = time.perf_counter()
    x_sol, converged, iters = _solve_ynewton(GTG_np, alpha, eps)
    newton_ms = (time.perf_counter() - t_newton) * 1000.0
    newton_x, newton_converged, newton_iters = x_sol, converged, iters
    used_scipy_fallback = False
    solver_fallback = False
    fallback_ok, fallback_nfev, fallback_ms = False, 0, 0.0

    if not converged:
        # Hand Newton's last iterate to scipy: on dead-task geometries it's
        # 40+ log-units closer to the true solution than uniform 1/K.
        newton_last = x_sol if np.all(np.isfinite(x_sol)) else None
        t_fallback = time.perf_counter()
        x_sol_scipy, scipy_ok, scipy_nfev = _solve_scipy_fallback(
            GTG_np, alpha, eps, x_start=newton_last,
        )
        fallback_ms = (time.perf_counter() - t_fallback) * 1000.0
        fallback_ok, fallback_nfev = scipy_ok, scipy_nfev
        used_scipy_fallback = True
        iters = scipy_nfev
        if scipy_ok and np.all(np.isfinite(x_sol_scipy)):
            x_sol = x_sol_scipy
            converged = True
        else:
            x_sol = np.ones(num_tasks, dtype=np.float64) / num_tasks
            solver_fallback = True

    if not np.all(np.isfinite(x_sol)):
        x_sol = np.ones(num_tasks, dtype=np.float64) / num_tasks
        solver_fallback = True
        converged = False

    x_sol = np.maximum(x_sol, eps)

    inv_alpha = 1.0 / float(alpha)
    residual_val = float(np.linalg.norm(
        GTG_np @ x_sol - np.power(1.0 / np.maximum(x_sol, eps), inv_alpha)
    ))

    w_f64 = torch.from_numpy(x_sol).to(device=device, dtype=torch.float64)
    w = w_f64.to(dtype=grad_matrix.dtype)

    # Entropy in fp64 — weight sums can hit 1e24+, mixed precision is unsafe.
    w_for_entropy = w_f64 / (w_f64.sum() + eps)
    weight_entropy = float(-(w_for_entropy * (w_for_entropy + eps).log()).sum().item())

    # Read before the benchmark so its extra scipy solve stays out of telemetry.
    wall_time_ms = (time.perf_counter() - t0) * 1000.0

    if _BENCH_PATH is not None:
        # Solve the same Gram with the scipy reference, time it, discard it.
        newton_resid = float(np.linalg.norm(
            GTG_np @ newton_x
            - np.power(1.0 / np.maximum(newton_x, eps), inv_alpha)
        )) if np.all(np.isfinite(newton_x)) else float("nan")
        t_sc = time.perf_counter()
        x_sc, sc_ok, sc_nfev = _solve_scipy_fallback(GTG_np, alpha, _BENCH_SCIPY_LB)
        scipy_ms = (time.perf_counter() - t_sc) * 1000.0
        sc_res = float(np.linalg.norm(
            GTG_np @ x_sc
            - np.power(1.0 / np.maximum(x_sc, _BENCH_SCIPY_LB), inv_alpha)
        )) if np.all(np.isfinite(x_sc)) else float("nan")
        _bench_record({
            "K": num_tasks, "alpha": float(alpha), "gram_trace": gram_trace,
            "t_newton_ms": newton_ms, "newton_iters": int(newton_iters),
            "newton_converged": int(newton_converged),
            "newton_resid": newton_resid,
            "t_scipy_ms": scipy_ms, "scipy_nfev": sc_nfev,
            "scipy_success": int(sc_ok), "scipy_resid": sc_res,
            "fallback_used": int(used_scipy_fallback),
            "fallback_ok": int(fallback_ok), "fallback_nfev": fallback_nfev,
            "t_fallback_ms": fallback_ms,
        })

    return FairGradDiagnostics(
        weights=w.detach(),
        gram_trace=gram_trace,
        weight_entropy=weight_entropy,
        solver_converged=bool(converged),
        wall_time_ms=wall_time_ms,
        regularization_eps=0.0,
        residual=residual_val,
        solver_fallback=solver_fallback,
        solver_iters=int(iters),
        used_scipy_fallback=used_scipy_fallback,
    )


def fairgrad_combine(
    task_grads: List[torch.Tensor],
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, None, Optional[FairGradDiagnostics]]:
    """FairGrad combiner: ``(weighted_sum, None, FairGradDiagnostics)``.

    The combine step runs in fp64 to avoid intermediate overflow — dead-task
    rows push w_i up to ~1e22 and fp32 products can transiently hit ±∞.
    """
    K = len(task_grads)
    if K == 0:
        return torch.zeros(0), None, None
    if K == 1:
        diag = FairGradDiagnostics(
            weights=torch.ones(1, device=task_grads[0].device),
            solver_converged=True,
            wall_time_ms=0.0,
        )
        return task_grads[0].clone(), None, diag
    grad_mat = torch.stack(task_grads)
    orig_dtype = grad_mat.dtype
    grad_mat_f64 = grad_mat.double()
    diag = solve_fairgrad_weights(grad_mat_f64, alpha=alpha)

    w_f64 = diag.weights
    combined_f64 = (w_f64.unsqueeze(1) * grad_mat_f64).sum(dim=0)
    combined = combined_f64.to(dtype=orig_dtype)

    # Per-task effective contribution ‖w_i · g_i‖ — fp64 to match combine.
    with torch.no_grad():
        per_task_mass = (w_f64.unsqueeze(1) * grad_mat_f64).norm(dim=1)
    diag.contribution_norms = [float(v.item()) for v in per_task_mass]
    return combined, None, diag

