"""Benchmark alternative solvers for FairGrad's KKT equation G @ x = x^{-1/α}.

Compares:
  - scipy.optimize.least_squares (trf, current baseline, +analytical jac +warm-start)
  - scipy.optimize.least_squares (dogbox)
  - scipy.optimize.root (lm, hybr) on re-parameterized log-domain
  - Custom projected Newton (pure numpy)
  - Custom fixed-point iteration

At K=10 and K=50, across 4 G-matrix regimes:
  - well-conditioned (all tasks active, similar scale)
  - heterogeneous (varying ‖g_i‖, 10× spread)
  - dead-task (one ‖g_i‖ ≈ 1e-10, rest active)
  - extreme-dead (one ‖g_i‖ ≈ 1e-22, mimicking observed v2.12.2 pathology)

Metrics: wall time, residual ‖G @ x − x^{-1/α}‖, produced x_max, convergence flag.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares, root


ALPHA = 2.0
EPS = 1e-8


# ---------------------------------------------------------------------------
# Gram matrix generators
# ---------------------------------------------------------------------------

def gram_well_conditioned(K: int, D: int = 5000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((K, D))
    return g @ g.T


def gram_heterogeneous(K: int, D: int = 5000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((K, D))
    scales = np.geomspace(1e-2, 1e1, K)  # 3 orders of magnitude
    g = g * scales[:, None]
    return g @ g.T


def gram_dead_task(K: int, D: int = 5000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((K, D))
    g[0] = g[0] * 1e-10  # task 0 is dead
    return g @ g.T


def gram_extreme_dead(K: int, D: int = 5000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((K, D))
    g[0] = g[0] * 1e-22  # task 0 below fp32 noise
    return g @ g.T


GRAM_REGIMES = {
    "well_cond": gram_well_conditioned,
    "heterogen": gram_heterogeneous,
    "dead_task": gram_dead_task,
    "extreme_dead": gram_extreme_dead,
}


# ---------------------------------------------------------------------------
# Residual + Jacobian (shared)
# ---------------------------------------------------------------------------

def objfn(G: np.ndarray, alpha: float, eps: float):
    inv_alpha = 1.0 / alpha

    def f(x: np.ndarray) -> np.ndarray:
        return G @ x - np.power(1.0 / np.maximum(x, eps), inv_alpha)
    return f


def jacfn(G: np.ndarray, alpha: float, eps: float):
    inv_alpha = 1.0 / alpha

    def j(x: np.ndarray) -> np.ndarray:
        x_safe = np.maximum(x, eps)
        J = G.copy()
        J[np.diag_indices_from(J)] += inv_alpha * np.power(x_safe, -inv_alpha - 1.0)
        return J
    return j


def residual_norm(G: np.ndarray, x: np.ndarray, alpha: float) -> float:
    x = np.maximum(x, EPS)
    return float(np.linalg.norm(G @ x - np.power(1.0 / x, 1.0 / alpha)))


# ---------------------------------------------------------------------------
# Solvers to benchmark
# ---------------------------------------------------------------------------

def solve_trf_default(G, alpha, eps, x0=None):
    """Current FairGrad implementation: trf, no jac, default tolerances."""
    K = G.shape[0]
    x_start = x0 if x0 is not None else np.ones(K) / K
    f = objfn(G, alpha, eps)
    res = least_squares(f, x_start, bounds=(eps, np.inf))
    return np.asarray(res.x), bool(res.success)


def solve_trf_jac(G, alpha, eps, x0=None):
    """trf with analytical Jacobian."""
    K = G.shape[0]
    x_start = x0 if x0 is not None else np.ones(K) / K
    f = objfn(G, alpha, eps)
    j = jacfn(G, alpha, eps)
    res = least_squares(f, x_start, jac=j, bounds=(eps, np.inf))
    return np.asarray(res.x), bool(res.success)


def solve_trf_jac_loose(G, alpha, eps, x0=None):
    """trf + analytical Jacobian + loose tolerances (1e-4)."""
    K = G.shape[0]
    x_start = x0 if x0 is not None else np.ones(K) / K
    f = objfn(G, alpha, eps)
    j = jacfn(G, alpha, eps)
    res = least_squares(f, x_start, jac=j, bounds=(eps, np.inf),
                        ftol=1e-4, xtol=1e-4, gtol=1e-6)
    return np.asarray(res.x), bool(res.success)


def solve_dogbox(G, alpha, eps, x0=None):
    K = G.shape[0]
    x_start = x0 if x0 is not None else np.ones(K) / K
    f = objfn(G, alpha, eps)
    j = jacfn(G, alpha, eps)
    res = least_squares(f, x_start, jac=j, bounds=(eps, np.inf), method='dogbox')
    return np.asarray(res.x), bool(res.success)


def solve_root_lm_log(G, alpha, eps, x0=None):
    """scipy.optimize.root(method='lm') on log-parameterized x = exp(y).

    Levenberg-Marquardt doesn't support bounds directly, so parameterize
    x = exp(y) to guarantee x > 0 without box constraints.
    """
    K = G.shape[0]
    x_start = x0 if x0 is not None else np.ones(K) / K
    y_start = np.log(np.maximum(x_start, eps))
    inv_alpha = 1.0 / alpha

    def fy(y):
        x = np.exp(y)
        return G @ x - np.power(x, -inv_alpha)

    def jy(y):
        x = np.exp(y)
        # dF/dy = dF/dx · dx/dy = (G + (1/α) diag(x^{-1/α - 1})) · diag(x)
        J = G * x[None, :] + inv_alpha * np.diag(np.power(x, -inv_alpha))
        return J

    try:
        res = root(fy, y_start, jac=jy, method='lm')
        x_sol = np.exp(np.asarray(res.x))
        return x_sol, bool(res.success)
    except Exception:
        return np.asarray(x_start), False


def solve_root_hybr_log(G, alpha, eps, x0=None):
    """MINPACK hybrid on log-parameterized x."""
    K = G.shape[0]
    x_start = x0 if x0 is not None else np.ones(K) / K
    y_start = np.log(np.maximum(x_start, eps))
    inv_alpha = 1.0 / alpha

    def fy(y):
        x = np.exp(y)
        return G @ x - np.power(x, -inv_alpha)

    try:
        res = root(fy, y_start, method='hybr')
        x_sol = np.exp(np.asarray(res.x))
        return x_sol, bool(res.success)
    except Exception:
        return np.asarray(x_start), False


def solve_projected_newton(G, alpha, eps, x0=None, max_iter=50, tol=1e-6):
    """Custom projected damped Newton with backtracking line search.

    J(x) = G + (1/α) diag(x^{-1/α - 1})
    step: p = -J^{-1} F
    projected: x_new = max(x + t·p, eps), t via backtracking.
    """
    K = G.shape[0]
    x = (x0.copy() if x0 is not None else np.ones(K) / K).astype(np.float64)
    x = np.maximum(x, eps)
    inv_alpha = 1.0 / alpha

    def F(x):
        return G @ x - np.power(x, -inv_alpha)

    def norm_F(x):
        return np.linalg.norm(F(x))

    f_prev = norm_F(x)
    for it in range(max_iter):
        if f_prev < tol:
            return x, True
        # Jacobian
        J = G.copy()
        J[np.diag_indices_from(J)] += inv_alpha * np.power(x, -inv_alpha - 1.0)
        try:
            p = np.linalg.solve(J, -F(x))
        except np.linalg.LinAlgError:
            # fall back to pinv
            p = -np.linalg.pinv(J) @ F(x)
        # Backtracking line search (Armijo on ‖F‖²)
        t = 1.0
        for _ in range(20):
            x_new = np.maximum(x + t * p, eps)
            f_new = norm_F(x_new)
            if f_new < f_prev * (1 - 1e-4 * t):
                break
            t *= 0.5
        else:
            # Line search failed; stop here
            return x, False
        x = x_new
        f_prev = f_new
    return x, f_prev < tol * 10


def solve_fixed_point(G, alpha, eps, x0=None, max_iter=100, tol=1e-6):
    """Naive fixed-point: x_{n+1} = (G @ x_n)^{-α} element-wise.

    Comes from solving KKT for each x_i: given current G @ x, x_i^{-1/α} = (G @ x)_i
    → x_i = (G @ x)_i^{-α}.  No convergence guarantee — tests empirically.
    """
    K = G.shape[0]
    x = (x0.copy() if x0 is not None else np.ones(K) / K).astype(np.float64)
    x = np.maximum(x, eps)
    for it in range(max_iter):
        Gx = np.maximum(G @ x, eps)  # clamp to keep power well-defined
        x_new = np.power(Gx, -alpha)
        x_new = np.maximum(x_new, eps)
        if np.linalg.norm(x_new - x) / (np.linalg.norm(x) + eps) < tol:
            return x_new, True
        x = x_new
    return x, False


SOLVERS: Dict[str, Callable] = {
    "trf_default":      solve_trf_default,
    "trf_jac":          solve_trf_jac,
    "trf_jac_loose":    solve_trf_jac_loose,
    "dogbox_jac":       solve_dogbox,
    "root_lm_log":      solve_root_lm_log,
    "root_hybr_log":    solve_root_hybr_log,
    "proj_newton":      solve_projected_newton,
    "fixed_point":      solve_fixed_point,
}


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def time_solver(solver, G, alpha, eps, x0=None, repeat=5):
    """Run solver `repeat` times, return (median_ms, last_x, last_success)."""
    times = []
    x_last, ok_last = None, False
    for _ in range(repeat):
        t0 = time.perf_counter()
        try:
            x, ok = solver(G, alpha, eps, x0=x0)
            x_last, ok_last = x, ok
        except Exception as e:
            return float('nan'), None, False
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), x_last, ok_last


def run_benchmark():
    Ks = [10, 50]
    results = []  # list of dict rows

    for K in Ks:
        for regime, gen in GRAM_REGIMES.items():
            G = gen(K=K, D=5000, seed=42)
            # Two sub-scenarios per regime: cold-start (x0=None) and warm-start
            # (x0 = solution from previous similar G).  Warm-start x0 obtained
            # from a quick trf solve with seed=41 of the same regime.
            G_prev = gen(K=K, D=5000, seed=41)
            x_warm, _ = solve_trf_jac(G_prev, ALPHA, EPS, x0=None)

            for solver_name, solver in SOLVERS.items():
                for start_name, x0 in [("cold", None), ("warm", x_warm)]:
                    t_ms, x, ok = time_solver(solver, G, ALPHA, EPS, x0=x0, repeat=5)
                    if x is None:
                        row = dict(K=K, regime=regime, solver=solver_name,
                                   start=start_name, time_ms=float('nan'),
                                   residual=float('nan'), x_max=float('nan'),
                                   x_min=float('nan'), ok=False)
                    else:
                        row = dict(K=K, regime=regime, solver=solver_name,
                                   start=start_name, time_ms=t_ms,
                                   residual=residual_norm(G, x, ALPHA),
                                   x_max=float(np.max(x)), x_min=float(np.min(x)),
                                   ok=ok)
                    results.append(row)

    return results


def print_results(results):
    # Group by (K, regime), then sort by time within
    by_key = {}
    for r in results:
        key = (r['K'], r['regime'])
        by_key.setdefault(key, []).append(r)

    print("\n" + "=" * 120)
    print(f"FairGrad solver benchmark, α={ALPHA}, eps={EPS}")
    print("time_ms = median over 5 runs; residual = ‖G x − x^{−1/α}‖")
    print("=" * 120)

    for key in sorted(by_key.keys()):
        K, regime = key
        print(f"\n--- K={K}  regime={regime} ---")
        rows = sorted(by_key[key], key=lambda r: (r['solver'], r['start']))
        print(f"{'solver':<20} {'start':<6} {'time_ms':>10} {'residual':>12} "
              f"{'x_max':>14} {'x_min':>12} ok")
        for r in rows:
            tm = f"{r['time_ms']:.3f}" if not np.isnan(r['time_ms']) else "n/a"
            rs = f"{r['residual']:.2e}" if not np.isnan(r['residual']) else "n/a"
            xm = f"{r['x_max']:.2e}" if not np.isnan(r['x_max']) else "n/a"
            xn = f"{r['x_min']:.2e}" if not np.isnan(r['x_min']) else "n/a"
            print(f"{r['solver']:<20} {r['start']:<6} {tm:>10} {rs:>12} "
                  f"{xm:>14} {xn:>12} {int(r['ok'])}")


def print_winners(results):
    """For each (K, regime), print the top-3 fastest solvers that achieve
    residual close to the trf_default baseline (within 100x)."""
    by_key = {}
    for r in results:
        key = (r['K'], r['regime'], r['start'])
        by_key.setdefault(key, []).append(r)

    print("\n" + "=" * 120)
    print("TOP-3 by speed (among those with residual within 100× of trf_default)")
    print("=" * 120)
    for key in sorted(by_key.keys()):
        K, regime, start = key
        rows = by_key[key]
        baseline = next((r for r in rows if r['solver'] == 'trf_default'), None)
        if baseline is None or np.isnan(baseline['residual']):
            continue
        threshold = max(baseline['residual'] * 100.0, 1e-3)
        valid = [r for r in rows
                 if not np.isnan(r['residual']) and r['residual'] < threshold
                 and not np.isnan(r['time_ms'])]
        valid.sort(key=lambda r: r['time_ms'])
        print(f"\nK={K}  regime={regime}  start={start}  "
              f"(trf_default: {baseline['time_ms']:.2f}ms, "
              f"residual={baseline['residual']:.2e})")
        for r in valid[:3]:
            speedup = baseline['time_ms'] / r['time_ms'] if r['time_ms'] > 0 else float('inf')
            print(f"  {r['solver']:<18}  {r['time_ms']:7.2f}ms  "
                  f"({speedup:>5.1f}× faster)  residual={r['residual']:.2e}")


if __name__ == "__main__":
    import argparse

    # No options; --help prints this module's docstring instead of starting the run.
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    print("Running FairGrad solver benchmark...")
    results = run_benchmark()
    print_results(results)
    print_winners(results)
