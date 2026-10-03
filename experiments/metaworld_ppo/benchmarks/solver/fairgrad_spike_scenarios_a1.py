"""Comprehensive experiment matrix: how do scipy / x-Newton / y-Newton behave
across the full range of G-matrix pathologies that can cause weight spikes?

Dimensions swept:
  - K ∈ {10, 50}
  - α = SPIKE_ALPHA (default 1.0) everywhere except EXPERIMENT 2, which sweeps
    α ∈ {1, 1.5, 2, 3, 5}  (α=1 is well-posed; α>1 ill-posed on dead tasks)
  - Number of dead tasks: 0, 1, 3, 10% of K, 30% of K
  - Dead-task magnitude: 1e-4, 1e-12, 1e-20
  - G structure:
      (a) iid Gaussian (baseline)
      (b) heterogeneous scale (critic-like, 10^3 spread)
      (c) rank-deficient (active subspace smaller than K)
      (d) highly-correlated active tasks (positive correlation)
      (e) spike-inducing warm start (x_prev contains 1e20)

Metrics per solver:
  - wall time (ms)
  - x_max (maximum weight magnitude — the "spike" indicator)
  - x_min (does the smallest task starve?)
  - residual
  - iterations (if applicable)
  - converged flag
"""
from __future__ import annotations

import os
# ALPHA-parameterized variant: default 1.0 = TOPPO critic-side FairGrad
ALPHA = float(os.environ.get("SPIKE_ALPHA", "1.0"))

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import least_squares


EPS = 1e-8


# -----------------------------------------------------------------------------
# Residual and Jacobians
# -----------------------------------------------------------------------------

def F_x(G, x, alpha):
    return G @ x - np.power(np.maximum(x, EPS), -1.0/alpha)

def residual_x(G, x, alpha):
    return float(np.linalg.norm(F_x(G, x, alpha)))

def F_y(G, y, alpha):
    x = np.exp(np.minimum(y, 700))  # cap y to prevent exp overflow even during search
    return G @ x - np.exp(np.minimum(-y/alpha, 700))

def residual_y(G, y, alpha):
    return float(np.linalg.norm(F_y(G, y, alpha)))


# -----------------------------------------------------------------------------
# Solvers
# -----------------------------------------------------------------------------

def solve_scipy(G, alpha, x0=None):
    K = G.shape[0]
    x_start = x0.copy() if x0 is not None else np.ones(K) / K
    x_start = np.maximum(x_start, EPS)
    try:
        res = least_squares(lambda x: F_x(G, x, alpha), x_start,
                            bounds=(EPS, np.inf))
        return np.asarray(res.x), bool(res.success), res.nfev
    except Exception:
        return x_start, False, 0


def solve_xnewton(G, alpha, x0=None, max_iter=50, eta=1e-2, tol=1e-8):
    K = G.shape[0]
    x = (x0.copy() if x0 is not None else np.ones(K) / K).astype(np.float64)
    x = np.maximum(x, EPS)
    inv_a = 1.0 / alpha
    r = residual_x(G, x, alpha)
    for it in range(max_iter):
        if r <= eta or r <= tol:
            return x, True, it
        J = G.copy()
        J[np.diag_indices_from(J)] += inv_a * np.power(x, -inv_a - 1.0)
        try:
            p = np.linalg.solve(J, -F_x(G, x, alpha))
        except np.linalg.LinAlgError:
            return x, False, it
        t = 1.0
        for _ in range(30):
            x_new = np.maximum(x + t*p, EPS)
            r_new = residual_x(G, x_new, alpha)
            if r_new < r * (1 - 1e-4 * t):
                break
            t *= 0.5
        else:
            return x, False, it
        x = x_new
        r = r_new
    return x, False, max_iter


def solve_ynewton(G, alpha, x0=None, max_iter=50, eta=1e-2, tol=1e-8,
                  y_max=50.0):
    K = G.shape[0]
    if x0 is None:
        y = np.full(K, -np.log(K))
    else:
        y = np.log(np.maximum(x0, EPS))
    y = np.minimum(y, y_max)
    inv_a = 1.0 / alpha
    r = residual_y(G, y, alpha)
    for it in range(max_iter):
        if r <= eta or r <= tol:
            return np.exp(y), True, it
        x = np.exp(y)
        J = G * x[None, :]
        J[np.diag_indices_from(J)] += inv_a * np.power(x, -inv_a)
        try:
            p = np.linalg.solve(J, -F_y(G, y, alpha))
        except np.linalg.LinAlgError:
            return np.exp(y), False, it
        t = 1.0
        for _ in range(30):
            y_new = np.minimum(y + t*p, y_max)
            r_new = residual_y(G, y_new, alpha)
            if r_new < r * (1 - 1e-4 * t):
                break
            t *= 0.5
        else:
            return np.exp(y), False, it
        y = y_new
        r = r_new
    return np.exp(y), False, max_iter


# -----------------------------------------------------------------------------
# G matrix generators
# -----------------------------------------------------------------------------

def make_grads(K, D, num_dead, dead_scale, structure, seed=0):
    """Build the (K, D) grad matrix, then return G = grads @ grads.T.

    structure ∈ {"iid", "hetero", "rank_def", "correlated"}
    """
    rng = np.random.default_rng(seed)

    if structure == "iid":
        g = rng.standard_normal((K, D))
    elif structure == "hetero":
        g = rng.standard_normal((K, D))
        # geometric spread over 3 orders of magnitude
        scales = np.geomspace(1e-2, 1e1, K)
        rng.shuffle(scales)
        g = g * scales[:, None]
    elif structure == "rank_def":
        # Only K//2 independent directions; other tasks' grads are sums of
        # the independent ones.
        indep = rng.standard_normal((K // 2, D))
        weights = rng.standard_normal((K, K // 2))
        g = weights @ indep
    elif structure == "correlated":
        # All tasks share a dominant direction plus small per-task noise
        common = rng.standard_normal((1, D)) * 3.0
        noise = rng.standard_normal((K, D)) * 0.3
        g = common + noise
    else:
        raise ValueError(f"unknown structure {structure!r}")

    # Apply dead-task scaling to the first `num_dead` tasks
    for i in range(num_dead):
        norm = np.linalg.norm(g[i])
        if norm > 0:
            g[i] = g[i] / norm * dead_scale

    return g @ g.T


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    K: int
    alpha: float
    structure: str
    num_dead: int
    dead_scale: float
    scipy_time_ms: float
    scipy_xmax: float
    scipy_xmin: float
    scipy_resid: float
    scipy_nfev: int
    xnewton_time_ms: float
    xnewton_xmax: float
    xnewton_xmin: float
    xnewton_resid: float
    xnewton_iters: int
    xnewton_ok: bool
    ynewton_time_ms: float
    ynewton_xmax: float
    ynewton_xmin: float
    ynewton_resid: float
    ynewton_iters: int
    ynewton_ok: bool


def run_scenario(K, alpha, structure, num_dead, dead_scale, D=5000, seed=42,
                 repeats=3) -> ScenarioResult:
    G = make_grads(K, D, num_dead, dead_scale, structure, seed=seed)

    def t(fn):
        ts = []
        out = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            try:
                out = fn()
            except Exception:
                out = None
                ts.append(float('nan'))
                continue
            ts.append((time.perf_counter() - t0) * 1000)
        return np.median(ts), out

    t_sci, out_sci = t(lambda: solve_scipy(G, alpha))
    t_xn, out_xn = t(lambda: solve_xnewton(G, alpha, eta=1e-2))
    t_yn, out_yn = t(lambda: solve_ynewton(G, alpha, eta=1e-2))

    def safe_stats(out):
        if out is None:
            return float('nan'), float('nan'), float('nan')
        x = out[0]
        xmax = float(np.max(x))
        xmin = float(np.min(x))
        resid = residual_x(G, x, alpha)
        return xmax, xmin, resid

    sci_xmax, sci_xmin, sci_resid = safe_stats(out_sci)
    sci_nfev = out_sci[2] if out_sci is not None else 0
    xn_xmax, xn_xmin, xn_resid = safe_stats(out_xn)
    xn_iters = out_xn[2] if out_xn is not None else 0
    xn_ok = out_xn[1] if out_xn is not None else False
    yn_xmax, yn_xmin, yn_resid = safe_stats(out_yn)
    yn_iters = out_yn[2] if out_yn is not None else 0
    yn_ok = out_yn[1] if out_yn is not None else False

    return ScenarioResult(
        K=K, alpha=alpha, structure=structure, num_dead=num_dead, dead_scale=dead_scale,
        scipy_time_ms=t_sci, scipy_xmax=sci_xmax, scipy_xmin=sci_xmin,
        scipy_resid=sci_resid, scipy_nfev=sci_nfev,
        xnewton_time_ms=t_xn, xnewton_xmax=xn_xmax, xnewton_xmin=xn_xmin,
        xnewton_resid=xn_resid, xnewton_iters=xn_iters, xnewton_ok=xn_ok,
        ynewton_time_ms=t_yn, ynewton_xmax=yn_xmax, ynewton_xmin=yn_xmin,
        ynewton_resid=yn_resid, ynewton_iters=yn_iters, ynewton_ok=yn_ok,
    )


def print_results_table(results: List[ScenarioResult]):
    print(f"\n{'K':>3} {'α':>3} {'struct':>11} {'ndead':>5} {'gscale':>8} | "
          f"{'scipy':>28} | {'xN @ η=1e-2':>30} | {'yN @ η=1e-2':>30}")
    print(f"{'':>3} {'':>3} {'':>11} {'':>5} {'':>8} | "
          f"{'ms   xmax    resid':>28} | "
          f"{'ms  it   xmax    resid':>30} | "
          f"{'ms  it   xmax    resid':>30}")
    print("-" * 150)
    for r in results:
        print(f"{r.K:>3} {r.alpha:>3.1f} {r.structure:>11} {r.num_dead:>5} "
              f"{r.dead_scale:>8.0e} | "
              f"{r.scipy_time_ms:>6.1f} {r.scipy_xmax:>8.1e} {r.scipy_resid:>7.1e} | "
              f"{r.xnewton_time_ms:>5.2f} {r.xnewton_iters:>3} "
              f"{r.xnewton_xmax:>8.1e} {r.xnewton_resid:>7.1e} | "
              f"{r.ynewton_time_ms:>5.2f} {r.ynewton_iters:>3} "
              f"{r.ynewton_xmax:>8.1e} {r.ynewton_resid:>7.1e}")


# -----------------------------------------------------------------------------
# Main experiment matrix
# -----------------------------------------------------------------------------

def experiment_1_structure_sweep():
    """Sweep over structural regimes at K=10 and K=50."""
    print("\n" + "=" * 150)
    print(f"EXPERIMENT 1: Structure × num_dead × dead_scale sweep, α={ALPHA:g}")
    print("=" * 150)
    results = []
    for K in [10, 50]:
        for structure in ["iid", "hetero", "rank_def", "correlated"]:
            for num_dead in [0, 1, max(1, K // 10), max(1, K // 3)]:
                for dead_scale in [1e-4, 1e-12, 1e-20]:
                    if num_dead == 0 and dead_scale != 1e-20:
                        continue  # no dead tasks → scale irrelevant
                    r = run_scenario(K=K, alpha=ALPHA, structure=structure,
                                     num_dead=num_dead, dead_scale=dead_scale)
                    results.append(r)
    print_results_table(results)
    return results


def experiment_2_alpha_sweep():
    """How does α affect solver behavior on dead-task regime?"""
    print("\n" + "=" * 150)
    print("EXPERIMENT 2: α sweep, K=50 with 1 dead task at 1e-20 (iid background)")
    print("=" * 150)
    results = []
    for alpha in [1.0, 1.5, 2.0, 3.0, 5.0]:
        r = run_scenario(K=50, alpha=alpha, structure="iid",
                         num_dead=1, dead_scale=1e-20)
        results.append(r)
    print_results_table(results)
    return results


def experiment_3_warm_start_spike():
    """Warm-starting from a big x after a spike: does y-Newton preserve /
    damp / amplify the spike when G returns to normal?"""
    print("\n" + "=" * 150)
    print(f"EXPERIMENT 3: Warm-start after a spike (K=50, α={ALPHA:g})")
    print("G_prev had a 1-task dead regime; now G is well-conditioned. Does")
    print("the spike-era weight (warm_xmax) get damped quickly?")
    print("=" * 150)
    rng = np.random.default_rng(0)
    g_active = rng.standard_normal((50, 5000))
    G_next = g_active @ g_active.T  # healthy G now

    print(f"{'warm_xmax':>10} | {'scipy':>20} | {'xN':>30} | {'yN':>30}")
    print(f"{'':>10} | {'time xmax resid':>20} | {'time it xmax resid':>30} | {'time it xmax resid':>30}")
    for warm_xmax in [1.0, 1e4, 1e8, 1e12, 1e16, 1e20]:
        x_warm = np.ones(50) * 1e-3
        x_warm[0] = warm_xmax
        # Time scipy with warm-start
        t0 = time.perf_counter()
        x_sci, ok_sci, nfev = solve_scipy(G_next, ALPHA, x0=x_warm)
        t_sci = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        x_xn, _, it_xn = solve_xnewton(G_next, ALPHA, x0=x_warm, eta=1e-2)
        t_xn = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        x_yn, _, it_yn = solve_ynewton(G_next, ALPHA, x0=x_warm, eta=1e-2)
        t_yn = (time.perf_counter() - t0) * 1000

        print(f"{warm_xmax:>10.0e} | "
              f"{t_sci:>5.1f} {np.max(x_sci):>7.1e} {residual_x(G_next, x_sci, ALPHA):>5.1e} | "
              f"{t_xn:>5.1f} {it_xn:>2d} {np.max(x_xn):>7.1e} {residual_x(G_next, x_xn, ALPHA):>5.1e} | "
              f"{t_yn:>5.1f} {it_yn:>2d} {np.max(x_yn):>7.1e} {residual_x(G_next, x_yn, ALPHA):>5.1e}")


def experiment_4_realistic_ppo_gram():
    """Simulate PPO critic Gram: heterogeneous value_loss magnitudes +
    occasional sharp dips when a task is well-fit."""
    print("\n" + "=" * 150)
    print("EXPERIMENT 4: Realistic PPO-critic-like Gram matrices")
    print("  - K=10 (MT10)")
    print("  - 1 task with value_loss ~ fp-noise (peg-insert-style 'dead')")
    print("  - 2 tasks with value_loss ~ 1 (well-learned)")
    print("  - 7 tasks with value_loss ~ 100 (still learning)")
    print("=" * 150)
    rng = np.random.default_rng(42)
    K, D = 10, 5000
    scales = np.concatenate([
        [1e-10],                                 # dead
        [1.0, 1.5],                              # well-learned, small grad
        rng.uniform(10, 20, 7),                  # still learning, moderate grad
    ])
    g = rng.standard_normal((K, D))
    for i in range(K):
        g[i] = g[i] / np.linalg.norm(g[i]) * scales[i]
    G = g @ g.T
    print(f"gram_trace = {np.trace(G):.2e}")
    print(f"Per-task ‖g‖² = [" + ", ".join(f"{G[i,i]:.2e}" for i in range(K)) + "]")

    for eta in [1e-2, 1e-4, 1e-6]:
        print(f"\n--- η = {eta} ---")
        t0 = time.perf_counter()
        x_sci, _, _ = solve_scipy(G, ALPHA)
        t_sci = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        x_xn, _, it_xn = solve_xnewton(G, ALPHA, eta=eta)
        t_xn = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        x_yn, _, it_yn = solve_ynewton(G, ALPHA, eta=eta)
        t_yn = (time.perf_counter() - t0) * 1000

        def fmt(name, x, t, iters=None):
            s = f"  {name:<10s}  {t:>6.2f}ms"
            if iters is not None:
                s += f"  {iters:>2d}iter"
            s += (f"  xmax={np.max(x):>7.1e}  xmin={np.min(x):>7.1e}  "
                  f"resid={residual_x(G, x, ALPHA):>5.1e}")
            return s
        print(fmt("scipy", x_sci, t_sci))
        print(fmt("xNewton", x_xn, t_xn, it_xn))
        print(fmt("yNewton", x_yn, t_yn, it_yn))


def experiment_5_contribution_preserved():
    """Does the effective contribution ‖w·g‖ match between scipy and y-Newton?
    This is the actual optimizer signal; matching this matters more than
    matching x directly."""
    print("\n" + "=" * 150)
    print("EXPERIMENT 5: Per-task contribution ‖w_i g_i‖ comparison")
    print("Identifies whether scipy and y-Newton apply similar effective gradients")
    print("despite x_max differing by orders of magnitude.")
    print("=" * 150)
    rng = np.random.default_rng(42)
    K, D = 10, 5000
    scales = np.concatenate([
        [1e-10],
        rng.uniform(1, 10, K-1),
    ])
    g = rng.standard_normal((K, D))
    for i in range(K):
        g[i] = g[i] / np.linalg.norm(g[i]) * scales[i]
    G = g @ g.T

    for solver_name, solver in [("scipy", lambda: solve_scipy(G, ALPHA)[0]),
                                 ("xNewton @ η=1e-2", lambda: solve_xnewton(G, ALPHA, eta=1e-2)[0]),
                                 ("yNewton @ η=1e-2", lambda: solve_ynewton(G, ALPHA, eta=1e-2)[0])]:
        x = solver()
        # effective contribution per task
        contribs = np.array([np.linalg.norm(x[i] * g[i]) for i in range(K)])
        combined = g.T @ x  # D-dim
        print(f"\n{solver_name}:")
        print(f"  x_max = {np.max(x):.2e}, x_min = {np.min(x):.2e}")
        print(f"  ‖combined‖ = {np.linalg.norm(combined):.3f}")
        print(f"  Per-task ‖w_i g_i‖:  " +
              "  ".join(f"t{i}={c:.2e}" for i, c in enumerate(contribs)))
        print(f"  Dead-task contribution fraction: "
              f"{contribs[0]**2 / (contribs**2).sum() * 100:.1f}%")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    # No options; --help prints this module's docstring instead of starting the run.
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    r1 = experiment_1_structure_sweep()
    r2 = experiment_2_alpha_sweep()
    experiment_3_warm_start_spike()
    experiment_4_realistic_ppo_gram()
    experiment_5_contribution_preserved()
