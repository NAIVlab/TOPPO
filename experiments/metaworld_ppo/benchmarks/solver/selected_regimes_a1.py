"""Selected-regime FairGrad solver benchmark: the generator of the appendix
solver table (``tab:solver``).

Four solvers on five synthetic Gram regimes, alpha = 1, D = 1024, seed 42.
Each (solver, regime) cell is called once untimed, then timed ``--n-runs``
times (default 7); the median is reported.

Solvers:
  * scipy ``least_squares``: verbatim port of the FairGrad paper's reference
    (Ban & Ji, ICML 2024; github.com/OptMN-Lab/fairgrad,
    methods/weight_methods.py).  x0 = 1/K, ``bounds=(0, inf)``, no
    ``max(x, eps)`` clip.  This is deliberately NOT our hardened
    ``_solve_scipy_fallback``.
  * SGD + momentum: verbatim port of the first-order solver that the FairGrad
    RL release actually runs (same repository, mtrl_files/fairgrad.py):
    lr = 0.1, momentum = 0.5, niter = 20, clip_grad_norm 1.0, clamp_min 1e-8,
    track-best.  It never early-exits.
  * y-Newton from a uniform cold start ``y = -log K``.
  * y-Newton with the diagonal warm start: the production solver,
    ``mtrl.gradients.fairgrad._solve_ynewton``, imported rather than copied.

Regimes (spread = max ||g_i|| / min ||g_i||):
  1. K=50, spread 1x            well conditioned
  2. K=10, spread 1e6x          log-spaced norms, random directions
  3. K=50, spread 1e8x          one small-norm task, random direction
                                (stress test: warm Newton stalls here)
  4. K=50, spread 1e8x          one small-norm task orthogonal to the others
  5. K=50, spread 1e8x          25% small-norm tasks orthogonal to the others

The SGD and y-Newton residuals of a rerun on another host match the paper's
table to its two printed digits, except one cold-start residual (2.6e-9 vs
2.7e-9; see README.md).  scipy's residual and nfev can move with
floating-point details, and wall times, hence the speedup column, depend on
the host; see README.md.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Callable, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mtrl.gradients.fairgrad import _solve_ynewton  # noqa: E402


# -----------------------------------------------------------------------------
# Synthetic Gram regimes
# -----------------------------------------------------------------------------

def _make_small_orthogonal_to_active(
    g: np.ndarray, small_indices: np.ndarray, target_scale: float,
) -> np.ndarray:
    """Project the small-norm rows onto the orthogonal complement of the other
    rows, then rescale them to ``target_scale * sqrt(D)``, so the cross terms
    G[active, small] sit at the fp64 noise floor."""
    K, D = g.shape
    active_mask = np.ones(K, dtype=bool)
    active_mask[small_indices] = False
    if not active_mask.any():
        return g
    _, _, Vt = np.linalg.svd(g[active_mask], full_matrices=False)
    g_out = g.copy()
    for idx in small_indices:
        g_perp = g[idx] - Vt.T @ (Vt @ g[idx])
        n = np.linalg.norm(g_perp)
        if n > 0:
            g_out[idx] = g_perp * (target_scale * np.sqrt(D) / n)
        else:
            g_out[idx] = g[idx] * target_scale
    return g_out


def make_gradient_matrix(K: int, regime: str, *, D: int = 1024, seed: int = 0) -> np.ndarray:
    """(K, D) per-task gradient matrix for a named regime."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((K, D))
    if regime == "well_cond":
        return g
    if regime == "heterogeneous":
        return g * np.logspace(-3.0, 3.0, K)[:, None]
    if regime == "small_rand":
        scales = np.ones(K)
        scales[K // 4] = 1e-8
        return g * scales[:, None]
    if regime == "small_orth":
        return _make_small_orthogonal_to_active(g, np.array([K // 4]), target_scale=1e-8)
    if regime == "small_orth_25pct":
        return _make_small_orthogonal_to_active(g, np.arange(K // 4), target_scale=1e-8)
    raise ValueError(f"unknown regime: {regime!r}")


def gram_from_grads(g: np.ndarray) -> np.ndarray:
    """Gram matrix in fp64, as production forms it."""
    g64 = g.astype(np.float64)
    return g64 @ g64.T


REGIMES = [
    ("well_cond",        50, "K=50, spread 1x"),
    ("heterogeneous",    10, "K=10, rand, spread 1e6x"),
    ("small_rand",       50, "K=50, rand, spread 1e8x"),
    ("small_orth",       50, "K=50, orth, spread 1e8x"),
    ("small_orth_25pct", 50, "K=50, orth 25%, spread 1e8x"),
]


# -----------------------------------------------------------------------------
# Solvers
# -----------------------------------------------------------------------------

def solve_scipy(G: np.ndarray, alpha: float) -> Tuple[np.ndarray, int]:
    """FairGrad paper reference: bounds (0, inf), raw ``(1/x)^(1/alpha)``."""
    from scipy.optimize import least_squares
    K = G.shape[0]

    def objfn(x: np.ndarray) -> np.ndarray:
        return G @ x - np.power(1.0 / x, 1.0 / alpha)

    res = least_squares(objfn, np.ones(K, dtype=np.float64) / K, bounds=(0.0, np.inf))
    return np.asarray(res.x, dtype=np.float64), int(res.nfev)


def solve_fairgrad_sgd(G: np.ndarray, alpha: float) -> Tuple[np.ndarray, int]:
    """FairGrad RL release's solver: 20 SGD+momentum steps on the squared
    residual from w0 = 1/K, returning the best iterate seen."""
    K = G.shape[0]
    inv_a = 1.0 / float(alpha)
    GG = torch.from_numpy(G.astype(np.float64))
    w = (torch.ones(K, dtype=torch.float64) / K).requires_grad_(True)
    optimizer = torch.optim.SGD([w], lr=0.1, momentum=0.5)
    niter = 20
    w_best = w.detach().clone()
    obj_best = float("inf")
    for _ in range(niter):
        optimizer.zero_grad()
        residual = torch.mv(GG, w) - torch.pow(1.0 / (w + 1e-8), inv_a)
        obj = torch.linalg.norm(residual) ** 2
        if obj.item() < obj_best:
            obj_best = obj.item()
            w_best = w.detach().clone()
        obj.backward()
        torch.nn.utils.clip_grad_norm_(w, max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            w.data.clamp_(min=1e-8)
    return w_best.numpy().astype(np.float64), niter


def solve_ynewton_cold(
    G: np.ndarray, alpha: float,
    eta: float = 1e-2, tol: float = 1e-8, max_iter: int = 50, y_max: float = 50.0,
) -> Tuple[np.ndarray, int]:
    """The production y-Newton iteration from a uniform start y = -log K."""
    K = G.shape[0]
    inv_a = 1.0 / float(alpha)

    def F(y: np.ndarray) -> np.ndarray:
        x = np.exp(np.minimum(y, y_max))
        return G @ x - np.exp(np.minimum(-y * inv_a, 700.0))

    y = np.full(K, -np.log(K), dtype=np.float64)
    r = float(np.linalg.norm(F(y)))
    iters = 0
    for it in range(max_iter):
        iters = it
        if r <= eta or r <= tol:
            break
        x = np.exp(np.minimum(y, y_max))
        J = G * x[np.newaxis, :]
        J[np.diag_indices_from(J)] += inv_a * np.exp(np.minimum(-y * inv_a, 700.0))
        try:
            p = np.linalg.solve(J, -F(y))
        except np.linalg.LinAlgError:
            break
        t = 1.0
        for _ in range(30):
            y_new = np.minimum(y + t * p, y_max)
            r_new = float(np.linalg.norm(F(y_new)))
            if r_new < r * (1.0 - 1e-4 * t):
                break
            t *= 0.5
        else:
            break
        y, r = y_new, r_new
    # Reported as iters + 1, as in the run behind the paper's table.
    return np.exp(np.minimum(y, y_max)), iters + 1


def solve_ynewton_warm(G: np.ndarray, alpha: float) -> Tuple[np.ndarray, int]:
    """Production solver (diagonal warm start), production eps = 1e-8."""
    x_sol, _, iters = _solve_ynewton(G, alpha, 1e-8)
    return x_sol, iters


def kkt_residual(G: np.ndarray, x: np.ndarray, alpha: float, eps: float = 1e-12) -> float:
    """||G x - x^(-1/alpha)||_2, the quantity every solver drives down."""
    return float(np.linalg.norm(G @ x - np.power(np.maximum(x, eps), -1.0 / float(alpha))))


def time_solver(fn: Callable, G: np.ndarray, alpha: float, n_runs: int):
    """Median wall time (ms) over ``n_runs`` after one untimed call."""
    fn(G, alpha)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        weights, iters = fn(G, alpha)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), weights, int(iters)


SOLVERS = [
    ("scipy", solve_scipy),
    ("sgd", solve_fairgrad_sgd),
    ("cold", solve_ynewton_cold),
    ("warm", solve_ynewton_warm),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-runs", type=int, default=7)
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default=None, help="also write the rows to this CSV")
    args = ap.parse_args()

    print(f"alpha={args.alpha}  D={args.D}  seed={args.seed}  n_runs={args.n_runs}  "
          f"(median ms; residual ||Gx - x^(-1/alpha)||; target eta = 1e-2)")
    print(f"{'regime':28s} | {'scipy':>8s} {'sgd':>7s} {'cold':>7s} {'warm':>7s} ms | "
          f"{'it cold':>7s} {'it warm':>7s} {'nfev':>5s} | "
          f"{'r scipy':>8s} {'r sgd':>8s} {'r cold':>8s} {'r warm':>8s} | speedup")
    rows = []
    for regime, K, label in REGIMES:
        G = gram_from_grads(make_gradient_matrix(K, regime, D=args.D, seed=args.seed))
        row = {"regime": regime, "K": K}
        for name, fn in SOLVERS:
            t, w, it = time_solver(fn, G, args.alpha, args.n_runs)
            row[f"t_{name}_ms"] = t
            row[f"iters_{name}"] = it
            row[f"r_{name}"] = kkt_residual(G, w, args.alpha)
        row["speedup_warm_vs_scipy"] = row["t_scipy_ms"] / row["t_warm_ms"]
        rows.append(row)
        print(f"{label:28s} | {row['t_scipy_ms']:8.2f} {row['t_sgd_ms']:7.2f} "
              f"{row['t_cold_ms']:7.3f} {row['t_warm_ms']:7.3f}    | "
              f"{row['iters_cold']:7d} {row['iters_warm']:7d} {row['iters_scipy']:5d} | "
              f"{row['r_scipy']:8.1e} {row['r_sgd']:8.1e} {row['r_cold']:8.1e} {row['r_warm']:8.1e} | "
              f"{row['speedup_warm_vs_scipy']:7.0f}x")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
