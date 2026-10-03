"""Summarize a real-Gram solver benchmark CSV (see realgram_results.md).

Standard library only.  With no argument it summarizes the committed
``solverbench_rfv1.csv``; pass a CSV written by ``MTRL_SOLVER_BENCH`` to compare
a fresh capture with it.  Fresh captures carry an ``alpha`` column, and only
rows at ``--alpha`` (default 1.0, the critic side) are kept.

    python realgram_summary.py [--csv run_logs/solverbench.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics


def percentile(values, q):
    """Linear interpolation between order statistics (numpy's default)."""
    v = sorted(values)
    pos = (len(v) - 1) * q
    lo = math.floor(pos)
    return v[lo] + (pos - lo) * (v[min(lo + 1, len(v) - 1)] - v[lo])


def summarize(path: str, alpha: float) -> dict:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "alpha" in rows[0]:
        rows = [r for r in rows if float(r["alpha"]) == alpha]
    if not rows:
        raise SystemExit(f"{path}: no solve rows (alpha={alpha})")
    newton = [float(r["t_newton_ms"]) for r in rows]
    scipy = [float(r["t_scipy_ms"]) for r in rows]
    speedup = [s / n for n, s in zip(newton, scipy)]
    iters = [int(r["newton_iters"]) for r in rows]

    def dist(v):
        return {"median": statistics.median(v), "mean": statistics.mean(v),
                "p95": percentile(v, 0.95), "max": max(v), "total": sum(v)}

    return {
        "csv": os.path.basename(path),
        "n_solves": len(rows),
        "K": sorted({int(r["K"]) for r in rows}),
        "newton_ms": dist(newton),
        "scipy_ms": dist(scipy),
        "per_solve_speedup": {"median": statistics.median(speedup),
                              "p05": percentile(speedup, 0.05),
                              "p95": percentile(speedup, 0.95), "max": max(speedup)},
        "aggregate_speedup": sum(scipy) / sum(newton),
        "newton_iters": {"median": statistics.median(iters), "max": max(iters)},
        "newton_converged": sum(int(r["newton_converged"]) for r in rows),
        "scipy_success": sum(int(r["scipy_success"]) for r in rows),
        "fallback_used": sum(int(r["fallback_used"]) for r in rows) if "fallback_used" in rows[0] else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "solverbench_rfv1.csv"))
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()
    print(json.dumps(summarize(args.csv, args.alpha), indent=2))


if __name__ == "__main__":
    main()
