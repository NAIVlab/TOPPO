"""Collect final-checkpoint per-task eval scores from each run's metric.csv.

Why not wandb: a preempted+resumed run re-logs steps W&B rejects as
non-monotonic, so its remote history freezes at the preemption point even
though training ran to the full budget. Measured on our MT50 runs: 5 finished
runs had truncated W&B history (24M-41M) while their local `metric.csv` —
which is appended across resumes — reached 100000000 with all 50
`test_win_rate/*` columns. Reading the final table from W&B would silently
report a mid-training number as the final-checkpoint result for exactly the
runs unlucky enough to be preempted.

Row selection: the HIGHEST-step row that carries all of the run's finite task
scores (50 on MT50, 10 on MT10; the count is read from the CSV header).
Taking merely the last row with *any* eval cell would let a later partial or
NaN row mask a valid final evaluation; and because a resumed run appends to
the same CSV, physical row order is not step order.

Emits one JSON object per run:
    {"exp":..., "step":..., "n_tasks":50, "rows_scanned":..., "scores":{task: rate}}
(`n_tasks` is 10 on MT10.)

Run on each host that owns run directories; the union is aggregated by
`aggregate.py`.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os

PREFIX = "test_win_rate/"


def best_eval_row(
    path: str, n_tasks: int | None = None
) -> tuple[dict[str, float], int, int] | None:
    """(scores, step, rows_scanned) for the highest-step complete eval row.

    ``n_tasks`` is the number of ``test_win_rate/<task>`` columns a row must
    carry to count as complete.  When omitted it is inferred from the CSV
    header, which makes this work on MT10 (10 tasks) as well as MT50 (50).
    """
    best: tuple[dict[str, float], int] | None = None
    n_rows = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if n_tasks is None:
            header = reader.fieldnames or []
            n_tasks = sum(1 for k in header if k.startswith(PREFIX))
            if n_tasks == 0:
                return None
        for row in reader:
            n_rows += 1
            scores: dict[str, float] = {}
            ok = True
            for k, v in row.items():
                if not k.startswith(PREFIX):
                    continue
                if v in (None, ""):
                    ok = False
                    break
                try:
                    x = float(v)
                except ValueError:
                    ok = False
                    break
                if not math.isfinite(x):
                    ok = False
                    break
                scores[k[len(PREFIX):]] = x
            if not ok or len(scores) < n_tasks:
                continue
            try:
                step = int(float(row["step"]))
            except (KeyError, ValueError):
                continue
            if best is None or step > best[1]:
                best = (scores, step)
    if best is None:
        return None
    return best[0], best[1], n_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--n-tasks", type=int, default=None,
                    help="tasks expected per row; inferred from the CSV header "
                         "when omitted (50 on MT50, 10 on MT10)")
    ap.add_argument("--host", default=os.uname().nodename)
    args = ap.parse_args()

    for path in sorted(glob.glob(os.path.join(args.logdir, "*", "metric.csv"))):
        exp = path.split(os.sep)[-2]
        got = best_eval_row(path, args.n_tasks)
        if got is None:
            print(json.dumps({"exp": exp, "host": args.host,
                              "error": f"no row with {args.n_tasks or 'the expected number of'} finite task scores"}))
            continue
        scores, step, n_rows = got
        print(json.dumps({
            "exp": exp,
            "host": args.host,
            "path": path,
            "step": step,
            "n_tasks": len(scores),
            "rows_scanned": n_rows,
            "scores": scores,
        }))


if __name__ == "__main__":
    main()
