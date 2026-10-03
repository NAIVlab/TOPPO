"""Synthetic FairGrad scale-invariance controls: the data behind the appendix
figure ``fig:f4``.  Numbers only; the paper's plotting code is not shipped.

Fixed-seed vectors: K = 10 unit directions in R^32 (``default_rng(20260908)``)
with norms log-spaced over [0.1, 10].

  * Single-task rescaling: task 1 is scaled by 49 log-spaced factors in
    [0.01, 100].  Each perturbation gets a fresh tight KKT solve in
    row-norm-conditioned coordinates, verified in the original coordinates
    (relative component residual <= 1e-8).  Recorded: cosine to the
    unperturbed aggregate and the relative vector change, for tight FairGrad
    at alpha = 0.5, 1, 2 and for mean aggregation (panels a, b).
  * Global rescaling: all tasks are scaled together by 65 log-spaced factors
    in [1e-10, 100].  The aggregate norm divided by sqrt(K), the exact alpha = 1
    reference, for the tight solve and for the released solver
    (``mtrl.gradients.fairgrad``: y-Newton at absolute tolerance eta = 1e-2,
    scipy fallback, eps floor) (panel c).

Writes three files to ``--out-dir`` (required, so that the committed copies next
to this script are only replaced on purpose):
raw_scale_invariance_single_task.csv, raw_scale_invariance_global.csv and
raw_scale_invariance_stats.json.  Multithreaded BLAS can move the last digits
(~1e-16) between runs; with OMP_NUM_THREADS=1 the output is bit-reproducible.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import root

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mtrl.gradients.fairgrad import _solve_scipy_fallback, _solve_ynewton  # noqa: E402

SEED, K, D = 20260908, 10, 32


def tight_combine(gradients: np.ndarray, alpha: float):
    """Solve the KKT in row-norm-conditioned coordinates, then verify it.

    With g_i = n_i u_i, z_i = n_i w_i and R = U U^T, the KKT G w = w^(-1/alpha)
    is R z = n^(1/alpha - 1) * z^(-1/alpha): the same equation for any alpha,
    but well conditioned.  This is not the production solver.
    """
    norms = np.linalg.norm(gradients, axis=1)
    unit = gradients / norms[:, None]
    corr = unit @ unit.T
    factor = norms ** (1 / alpha - 1)

    def fun(y):
        z = np.exp(y)
        return corr @ z - factor * z ** (-1 / alpha)

    def jac(y):
        z = np.exp(y)
        return corr * z[None, :] + np.diag(factor / alpha * z ** (-1 / alpha))

    start = np.log(factor) / (1 + 1 / alpha)
    result = root(fun, start, jac=jac, tol=1e-11)
    w = np.exp(result.x) / norms
    G = gradients @ gradients.T
    relative_component_residual = np.max(abs((G @ w) / (w ** (-1 / alpha)) - 1))
    if not np.all(np.isfinite(w)) or relative_component_residual > 1e-8:
        raise RuntimeError(f"tight KKT validation failed: {relative_component_residual}")
    return w @ gradients, float(relative_component_residual)


def production_combine(g: np.ndarray) -> np.ndarray:
    """The released solve path at alpha = 1: Newton, scipy fallback, eps floor."""
    G = g @ g.T
    w, converged, _ = _solve_ynewton(G, 1.0, eps=1e-8)
    if not converged:
        w, okay, _ = _solve_scipy_fallback(G, 1.0, eps=1e-8, x_start=w)
        if not okay:
            w = np.ones(len(g)) / len(g)
    w = np.maximum(w, 1e-8)
    return w @ g


def run():
    rng = np.random.default_rng(SEED)
    g = rng.normal(size=(K, D))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    g *= np.logspace(-1, 1, K)[:, None]
    eigenvalues = np.linalg.eigvalsh(g @ g.T)
    assert eigenvalues.min() > 0

    factors = np.logspace(-2, 2, 49)
    # (alpha, method label); None = mean aggregation.  Labels match the
    # paper's cached CSV.
    specs = [(1.0, r"Tight FG $\alpha=1$"), (2.0, r"Tight FG $\alpha=2$"),
             (0.5, r"Tight FG $\alpha=0.5$"), (None, "Mean aggregation")]
    records, residuals, curves = [], [], {}
    for alpha, label in specs:
        base = g.mean(axis=0) if alpha is None else tight_combine(g, alpha)[0]
        cosines, errors = [], []
        for factor in factors:
            changed = g.copy()
            changed[0] *= factor
            if alpha is None:
                output, residual = changed.mean(axis=0), None
            else:
                output, residual = tight_combine(changed, alpha)
                residuals.append(residual)
            cosine = np.clip(output @ base / (np.linalg.norm(output) * np.linalg.norm(base)), -1, 1)
            error = np.linalg.norm(output - base) / np.linalg.norm(base)
            cosines.append(float(cosine))
            errors.append(float(error))
            records.append({"experiment": "single_task", "factor": float(factor), "method": label,
                            "cosine": float(cosine), "relative_vector_error": float(error),
                            "norm": float(np.linalg.norm(output)),
                            "relative_kkt_residual_max": residual})
        curves[str(alpha)] = {"max_relative_vector_error": max(errors), "min_cosine": min(cosines)}
    order = {r"Tight FG $\alpha=0.5$": 0, r"Tight FG $\alpha=1$": 1,
             r"Tight FG $\alpha=2$": 2, "Mean aggregation": 3}
    records.sort(key=lambda r: order[r["method"]])
    curves = {k: curves[k] for k in ("0.5", "1.0", "2.0", "None")}

    scales = np.logspace(-10, 2, 65)
    actual = np.array([np.linalg.norm(production_combine(s * g)) / math.sqrt(K) for s in scales])
    exact = np.array([np.linalg.norm(tight_combine(s * g, 1)[0]) / math.sqrt(K) for s in scales])
    assert curves["1.0"]["max_relative_vector_error"] < 1e-8

    stats = {"rng_seed": SEED, "K": K, "D": D, "gram_min_eigenvalue": float(eigenvalues.min()),
             "max_relative_component_kkt_residual_tight": max(residuals),
             "single_task_results": curves,
             "production_global_scale_min_norm_ratio": float(actual.min())}
    return records, list(zip(scales, exact, actual)), stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True,
                    help="output directory; the committed raw_scale_invariance_* files "
                         "sit next to this script, so pass that directory only to replace them")
    args = ap.parse_args()
    records, scaling, stats = run()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "raw_scale_invariance_single_task.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    with open(os.path.join(args.out_dir, "raw_scale_invariance_global.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["global_scale", "tight_norm_over_sqrtK", "production_norm_over_sqrtK"])
        w.writerows(scaling)
    with open(os.path.join(args.out_dir, "raw_scale_invariance_stats.json"), "w") as f:
        f.write(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
