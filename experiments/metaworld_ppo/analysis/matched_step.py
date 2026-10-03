r"""Matched-progress comparison across seeds, to expose completion bias.

While a campaign is still running, a group's finished seeds are a
non-random subset: a seed that got preempted more often finishes later, and
if slower seeds are also worse seeds, the partial table flatters the group.
This reads every seed's full eval history from `metric.csv` and scores all
seeds at the largest step ALL of them have reached, so the comparison uses
the same n and the same amount of training for everyone.

Works on MT10 and MT50: the task count is read from each `metric.csv` header
(as `collect_csv.py` does), and a seed counts as finished in the final-only
column once it reaches its suite's full budget (100M for 50 tasks, 20M for
10).  `--n-tasks` and `--budget` override the inferred values.

    python analysis/matched_step.py --logdirs <dir> [<dir> ...] \
        --groups <group> [<group> ...] [--n-tasks 10] [--budget 20000000]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

PREFIX = "test_win_rate/"
SEED_RE = re.compile(r"_s(\d+)$")
# Full training budget (env steps) per suite, keyed by task count.
STEP_BUDGET = {50: 100_000_000, 10: 20_000_000}


def eval_history(
    path: str, n_tasks: int | None = None
) -> tuple[list[tuple[int, float]], int]:
    """(step, cross-task mean win rate) for every row carrying all eval columns.

    Also returns the task count, inferred from the CSV header when ``n_tasks``
    is omitted.  Rows are sorted by step: a resumed run appends to the same
    CSV, so physical row order is not step order.
    """
    out = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if n_tasks is None:
            n_tasks = sum(1 for k in (reader.fieldnames or []) if k.startswith(PREFIX))
        if n_tasks == 0:
            return [], 0
        for row in reader:
            vals = [
                float(v) for k, v in row.items()
                if k.startswith(PREFIX) and v not in (None, "")
            ]
            if len(vals) == n_tasks:
                out.append((int(float(row["step"])), float(np.mean(vals))))
    out.sort(key=lambda r: r[0])
    return out, n_tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdirs", nargs="+", required=True)
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--n-tasks", type=int, default=None,
                    help="tasks expected per eval row; inferred from the CSV header "
                         "when omitted (50 on MT50, 10 on MT10)")
    ap.add_argument("--budget", type=int, default=None,
                    help="step at which a seed counts as finished; defaults to the "
                         "suite's full budget (100000000 for 50 tasks, 20000000 for 10)")
    args = ap.parse_args()

    hist: dict[str, dict[str, list]] = defaultdict(dict)
    budget: dict[str, dict[str, int | None]] = defaultdict(dict)
    for logdir in args.logdirs:
        for path in sorted(glob.glob(os.path.join(logdir, "*", "metric.csv"))):
            exp = path.split(os.sep)[-2]
            g = SEED_RE.sub("", exp)
            if g in args.groups and exp not in hist[g]:
                h, n_tasks = eval_history(path, args.n_tasks)
                if h:
                    hist[g][exp] = h
                    budget[g][exp] = (args.budget if args.budget is not None
                                      else STEP_BUDGET.get(n_tasks))

    print(f"| group | n | matched step | mean@step +/- std | final-only mean (n) |")
    print("|---|---|---|---|---|")
    for g in args.groups:
        seeds = hist.get(g, {})
        if not seeds:
            print(f"| {g} | 0 | — | — | — |")
            continue
        common = min(h[-1][0] for h in seeds.values())
        at = []
        for exp, h in seeds.items():
            # last eval at or before the common step
            v = [s for s in h if s[0] <= common]
            at.append(v[-1][1] * 100 if v else np.nan)
        at_a = np.array([a for a in at if not np.isnan(a)])
        if any(b is None for b in budget[g].values()):
            print(f"[warn] {g}: no step budget known for this task count; "
                  f"pass --budget", file=sys.stderr)
        done = [h[-1][1] * 100 for exp, h in seeds.items()
                if budget[g][exp] is not None and h[-1][0] >= budget[g][exp]]
        fin = (f"{np.mean(done):.1f} (n={len(done)})" if done else "— (n=0)")
        std = at_a.std(ddof=1) if at_a.size > 1 else 0.0
        print(f"| {g} | {len(seeds)} | {common:,} | {at_a.mean():.1f} +/- {std:.1f} | {fin} |")


if __name__ == "__main__":
    main()
