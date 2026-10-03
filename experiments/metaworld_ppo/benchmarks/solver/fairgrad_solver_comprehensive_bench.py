"""Comprehensive benchmark: scipy vs v2.14.1 (cold Newton) vs v2.14.2 (diag-warmstart Newton).

Sweeps K × magnitude-spread × dead-task-count × inter-task-correlation.
All solvers share the same residual target η=1e-2 and y_max=50 semantics
where applicable (scipy has no direct equivalent).

Metrics per (scenario, solver):
  - median wall time across n_reps runs
  - iteration count (for y-Newton variants) / function evals (scipy)
  - final residual ‖Gw − w^{-1/α}‖
  - x_max reached
  - convergence flag
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares


ALPHA = 2.0
EPS = 1e-8
INV_ALPHA = 1.0 / ALPHA
Y_MAX = 50.0
ETA = 1e-2
MAX_ITER = 50


# ---------------------------------------------------------------------------
# Residual / objective
# ---------------------------------------------------------------------------

def residual(G: np.ndarray, x: np.ndarray) -> float:
    x = np.maximum(x, EPS)
    return float(np.linalg.norm(G @ x - np.power(x, -INV_ALPHA)))


def F_y(G, y):
    x = np.exp(np.minimum(y, Y_MAX))
    return G @ x - np.exp(np.minimum(-y * INV_ALPHA, 700.0))


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def solve_scipy(G):
    """v2.13.x baseline: scipy trf with bounds, no Jacobian."""
    K = G.shape[0]
    x_start = np.ones(K) / K

    def objfn(x):
        return G @ x - np.power(np.maximum(x, EPS), -INV_ALPHA)

    try:
        res = least_squares(objfn, x_start, bounds=(EPS, np.inf))
        x_sol = np.asarray(res.x)
        return x_sol, bool(res.success), res.nfev
    except Exception:
        return x_start, False, 0


def _ynewton(G, y_init, eta=ETA, max_iter=MAX_ITER):
    K = G.shape[0]
    y = y_init.copy()
    y = np.minimum(np.maximum(y, -Y_MAX), Y_MAX)
    r = float(np.linalg.norm(F_y(G, y)))
    for it in range(max_iter):
        if r <= eta:
            return np.exp(np.minimum(y, Y_MAX)), True, it
        x = np.exp(np.minimum(y, Y_MAX))
        J = G * x[np.newaxis, :]
        J[np.diag_indices_from(J)] += INV_ALPHA * np.power(x, -INV_ALPHA)
        try:
            p = np.linalg.solve(J, -F_y(G, y))
        except np.linalg.LinAlgError:
            return np.exp(np.minimum(y, Y_MAX)), False, it
        t = 1.0
        for _ in range(30):
            y_new = np.minimum(y + t * p, Y_MAX)
            r_new = float(np.linalg.norm(F_y(G, y_new)))
            if r_new < r * (1 - 1e-4 * t):
                break
            t *= 0.5
        else:
            return np.exp(np.minimum(y, Y_MAX)), False, it
        y = y_new
        r = r_new
    return np.exp(np.minimum(y, Y_MAX)), False, max_iter


def solve_v2_14_1(G):
    """v2.14.1: y-space Newton with uniform cold-start."""
    K = G.shape[0]
    y_init = np.full(K, -np.log(K), dtype=np.float64)
    return _ynewton(G, y_init)


def solve_v2_14_2(G):
    """v2.14.2: y-space Newton with diagonal-closed-form warm-start."""
    K = G.shape[0]
    diag_G = np.maximum(np.diagonal(G), EPS)
    y_diag = -(ALPHA / (ALPHA + 1.0)) * np.log(diag_G)
    y_cold = np.full(K, -np.log(K), dtype=np.float64)
    # Safety checks from production code
    if np.max(np.abs(y_diag)) > Y_MAX:
        y_init = y_cold
    else:
        r_diag = float(np.linalg.norm(F_y(G, y_diag)))
        r_cold = float(np.linalg.norm(F_y(G, y_cold)))
        y_init = y_diag if r_diag <= r_cold else y_cold
    return _ynewton(G, y_init)


# ---------------------------------------------------------------------------
# G matrix scenario generators
# ---------------------------------------------------------------------------

def gen_G(K, D, scenario, seed=0):
    """Build G according to a named scenario."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((K, D))
    # Normalize to unit
    for i in range(K):
        base[i] = base[i] / np.linalg.norm(base[i])

    if scenario == "uniform":
        scales = np.ones(K)
    elif scenario == "spread_10x":
        scales = np.geomspace(0.5, 5.0, K)
        rng.shuffle(scales)
    elif scenario == "spread_100x":
        scales = np.geomspace(0.1, 10.0, K)
        rng.shuffle(scales)
    elif scenario == "spread_1000x":
        scales = np.geomspace(0.1, 100.0, K)
        rng.shuffle(scales)
    elif scenario == "critic_like":
        # Typical PPO critic: a few tasks with big value_loss, a few almost-learned
        scales = np.concatenate([
            np.array([10.0, 20.0, 50.0]),
            rng.uniform(5.0, 15.0, max(0, K - 6)),
            np.array([0.5, 1.0, 2.0]),
        ])[:K]
        rng.shuffle(scales)
    elif scenario == "actor_like":
        # Typical PPO actor: narrow range around 0.3-1.0
        scales = rng.uniform(0.3, 1.0, K)
    elif scenario == "one_dead":
        scales = rng.uniform(0.5, 5.0, K)
        scales[0] = 1e-5
    elif scenario == "three_dead":
        scales = rng.uniform(0.5, 5.0, K)
        dead_idx = rng.choice(K, 3, replace=False)
        scales[dead_idx] = 1e-5
    elif scenario == "half_dead":
        scales = rng.uniform(0.5, 5.0, K)
        dead_idx = rng.choice(K, K // 2, replace=False)
        scales[dead_idx] = 1e-4
    elif scenario == "correlated":
        # All tasks share direction + small noise
        common = rng.standard_normal(D) * 3.0
        base = common[np.newaxis, :] + rng.standard_normal((K, D)) * 0.3
        for i in range(K):
            base[i] = base[i] / np.linalg.norm(base[i])
        scales = rng.uniform(0.5, 5.0, K)
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    grads = base * scales[:, None]
    return grads @ grads.T


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def time_solver(solver_fn, G, n_reps=5):
    times = []
    result = None
    for _ in range(n_reps):
        t0 = time.perf_counter()
        try:
            result = solver_fn(G)
        except Exception:
            result = None
            times.append(float('nan'))
            continue
        times.append((time.perf_counter() - t0) * 1000)
    if result is None:
        return float('nan'), float('nan'), float('nan'), 0, False
    x, conv, it = result
    r = residual(G, x)
    return float(np.median(times)), r, float(np.max(x)), int(it), bool(conv)


def run_benchmark():
    K_list = [10, 50]
    scenarios = [
        "uniform",
        "actor_like",
        "spread_10x",
        "spread_100x",
        "spread_1000x",
        "critic_like",
        "correlated",
        "one_dead",
        "three_dead",
        "half_dead",
    ]
    D = 10000
    solvers = {
        "scipy": solve_scipy,
        "v2.14.1 cold": solve_v2_14_1,
        "v2.14.2 diag": solve_v2_14_2,
    }

    results = []
    for K in K_list:
        for scenario in scenarios:
            G = gen_G(K, D, scenario, seed=42)
            for solver_name, fn in solvers.items():
                t_ms, r, x_max, iters, conv = time_solver(fn, G, n_reps=5)
                results.append({
                    "K": K, "scenario": scenario, "solver": solver_name,
                    "time_ms": t_ms, "residual": r,
                    "x_max": x_max, "iters": iters, "converged": conv,
                })
    return results


def print_table(results):
    print("\n" + "=" * 130)
    print(f"α={ALPHA}, η={ETA}, y_max={Y_MAX}, D=10000, 5 reps/solver → median wall time")
    print("=" * 130)
    print(f"{'K':>3} {'scenario':<14} {'solver':<14} "
          f"{'time_ms':>9} {'iters':>7} {'residual':>11} {'x_max':>11} {'conv':>5}")
    print("-" * 130)
    current_group = None
    for r in results:
        group = (r['K'], r['scenario'])
        if group != current_group:
            print("-" * 130) if current_group else None
            current_group = group
        print(f"{r['K']:>3} {r['scenario']:<14} {r['solver']:<14} "
              f"{r['time_ms']:>8.2f}  {r['iters']:>6d}  "
              f"{r['residual']:>10.2e}  {r['x_max']:>10.2e}  "
              f"{'✓' if r['converged'] else '✗':>4}")


def print_speedup_summary(results):
    """Pairwise speedup of v2.14.2 vs scipy and v2.14.2 vs v2.14.1."""
    # Group by (K, scenario)
    groups: Dict[Tuple[int, str], Dict[str, dict]] = {}
    for r in results:
        key = (r['K'], r['scenario'])
        groups.setdefault(key, {})[r['solver']] = r

    print("\n" + "=" * 90)
    print("SPEEDUP RATIOS (time_scipy / time_Xsolver)   — bigger is better")
    print("=" * 90)
    print(f"{'K':>3} {'scenario':<14} {'v2.14.1 cold':>15} {'v2.14.2 diag':>15} "
          f"{'diag/cold':>12}")
    print("-" * 90)
    for key in sorted(groups.keys()):
        K, sc = key
        g = groups[key]
        if 'scipy' not in g or 'v2.14.1 cold' not in g or 'v2.14.2 diag' not in g:
            continue
        t_sci = g['scipy']['time_ms']
        t_141 = g['v2.14.1 cold']['time_ms']
        t_142 = g['v2.14.2 diag']['time_ms']
        if t_141 > 0 and t_142 > 0:
            sp_141 = t_sci / t_141
            sp_142 = t_sci / t_142
            sp_diag_vs_cold = t_141 / t_142
            print(f"{K:>3} {sc:<14} {sp_141:>14.1f}× {sp_142:>14.1f}× "
                  f"{sp_diag_vs_cold:>11.2f}×")


if __name__ == "__main__":
    import argparse

    # No options; --help prints this module's docstring instead of starting the run.
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    print("Running comprehensive FairGrad solver benchmark...")
    results = run_benchmark()
    print_table(results)
    print_speedup_summary(results)
