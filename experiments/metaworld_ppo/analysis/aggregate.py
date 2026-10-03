r"""Aggregate collected per-run scores into the critic-normalizer table.

Metric conventions are the ones `metrics.py` established by reproducing
all four of the paper's published reference values to the last digit:

- per-seed score  = arithmetic mean of the per-task final eval win rates
                    (50 tasks on MT50, 10 on MT10; taken from the record)
- mean +/- std    = across seeds, sample std (ddof=1)
- IQM             = scipy.stats.trim_mean(per-seed scores, 0.25) — a trimmed
                    mean over the SEED means, not rliable's pooled
                    seed x task IQM (which lands 13-22 pp higher on MT50)
- IQM's +/-       = the same untrimmed ddof=1 std as the mean row (the
                    paper's convention: +/- is always the sample std, never
                    SEM; a bootstrap CI of the mean is printed in its own
                    column, and --compare gives a paired t-test with a 95% CI
                    on the difference, so significance is not eyeballed from
                    overlapping +/- std bars)
- worst-k         = per seed, mean of that seed's k lowest per-task rates,
                    then mean +/- std across seeds

Runs arrive from several hosts, so records are keyed by (group, seed): the
highest step wins, a conflicting record at the SAME step is an error rather
than a silent last-write, and a group is reported with the seeds it is
missing rather than quietly averaging fewer.

    python analysis/aggregate.py --jsonl all.jsonl --groups <g> [<g> ...] \
        [--expect-seeds 0-9] [--require-step 100000000] [--manifest out.tsv] \
        [--compare a b]      # paired across matched seeds, with a t-test

--require-step defaults to the full budget of the suite each record belongs
to, read from its task count: 100M for 50 tasks (MT50), 20M for 10 (MT10).
Pass it explicitly for any other suite; pass 0 to disable the check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import t as t_dist
from scipy.stats import trim_mean, ttest_rel

SEED_RE = re.compile(r"_s(\d+)$")
# Full training budget (env steps) per suite, keyed by task count.
STEP_BUDGET = {50: 100_000_000, 10: 20_000_000}


def split_exp(exp: str) -> tuple[str, int] | None:
    m = SEED_RE.search(exp)
    if not m:
        return None
    return SEED_RE.sub("", exp), int(m.group(1))


def parse_seed_spec(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--require-step", type=int, default=None,
                    help="drop runs below this step. Defaults to the budget implied "
                         "by each record's task count (100000000 for 50 tasks, "
                         "20000000 for 10); pass 0 to disable the check")
    ap.add_argument("--expect-seeds", default=None,
                    help="e.g. 0-9; report which seeds are missing per group")
    ap.add_argument("--worst-k", type=int, nargs="*", default=[5, 10, 20])
    ap.add_argument("--manifest", default=None,
                    help="write an auditable TSV of every run used")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None,
                    help="paired comparison of two groups over matched seeds")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    picked: dict[tuple[str, int], dict] = {}
    conflicts: list[str] = []
    for line in open(args.jsonl):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "error" in d:
            print(f"[skip] {d['exp']}: {d['error']}", file=sys.stderr)
            continue
        key = split_exp(d["exp"])
        if key is None:
            print(f"[skip] {d['exp']}: no _s<seed> suffix, so (group, seed) cannot be "
                  f"recovered. Launch runs with --exp-name <group>_s<seed>; see docs/RUNS.md.",
                  file=sys.stderr)
            continue
        prev = picked.get(key)
        if prev is None or d["step"] > prev["step"]:
            picked[key] = d
        elif d["step"] == prev["step"] and d["scores"] != prev["scores"]:
            conflicts.append(
                f"{d['exp']}: two records at step {d['step']} disagree "
                f"({prev.get('host')} vs {d.get('host')})")
    if conflicts:
        raise SystemExit("CONFLICTING RECORDS — refusing to aggregate:\n  "
                         + "\n  ".join(conflicts))

    by_group: dict[str, dict[int, dict]] = defaultdict(dict)
    incomplete: dict[str, list[str]] = defaultdict(list)
    no_budget: dict[str, list[str]] = defaultdict(list)
    for (g, seed), d in picked.items():
        if args.groups and g not in args.groups:
            continue
        need = args.require_step
        if need is None:
            n_tasks = d.get("n_tasks", len(d["scores"]))
            need = STEP_BUDGET.get(n_tasks)
            if need is None:
                no_budget[g].append(f"s{seed} ({n_tasks} tasks)")
                continue
        if d["step"] < need:
            incomplete[g].append(f"s{seed} (step={d['step']:,})")
            continue
        by_group[g][seed] = d

    expect = parse_seed_spec(args.expect_seeds) if args.expect_seeds else None

    def seed_matrix(g: str) -> tuple[np.ndarray, list[int]]:
        seeds = sorted(by_group[g])
        tasks = sorted(by_group[g][seeds[0]]["scores"])
        m = np.array([[by_group[g][s]["scores"][t] for t in tasks] for s in seeds]) * 100
        return m, seeds

    hdr = ["group", "n", "mean +/- std", "IQM +/- std", "95% CI (boot)"] \
        + [f"worst-{k}" for k in args.worst_k]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join("---" for _ in hdr) + "|")

    rng = np.random.default_rng(0)
    order = args.groups if args.groups else sorted(by_group)
    for g in order:
        if not by_group.get(g):
            print(f"| {g} | 0 | — | — | — |" + " — |" * len(args.worst_k))
            continue
        m, seeds = seed_matrix(g)
        sm = m.mean(axis=1)
        n = len(sm)
        std = sm.std(ddof=1) if n > 1 else 0.0
        iqm = trim_mean(sm, 0.25) if n >= 4 else sm.mean()
        boot = np.array([rng.choice(sm, n, replace=True).mean() for _ in range(args.boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        cells = [g, str(n), f"{sm.mean():.1f} +/- {std:.1f}", f"{iqm:.1f} +/- {std:.1f}",
                 f"[{lo:.1f}, {hi:.1f}]"]
        for k in args.worst_k:
            if k > m.shape[1]:          # e.g. worst-20 on MT10's 10 tasks
                cells.append("n/a")
                continue
            ps = np.sort(m, axis=1)[:, :k].mean(axis=1)
            cells.append(f"{ps.mean():.1f} +/- {(ps.std(ddof=1) if n > 1 else 0.0):.1f}")
        print("| " + " | ".join(cells) + " |")

    if expect:
        print("\n[seed coverage]")
        for g in order:
            have = set(by_group.get(g, {}))
            missing = sorted(expect - have)
            extra = sorted(have - expect)
            note = "complete" if not missing and not extra else \
                   (f"MISSING {missing}" + (f" EXTRA {extra}" if extra else ""))
            print(f"  {g}: n={len(have)} {note}")

    if incomplete:
        print("\n[excluded — not at the final checkpoint yet]")
        for g, items in sorted(incomplete.items()):
            print(f"  {g}: {', '.join(items)}")

    if no_budget:
        print("\n[excluded — no step budget known for this task count; pass --require-step]")
        for g, items in sorted(no_budget.items()):
            print(f"  {g}: {', '.join(items)}")

    if args.compare:
        a, b = args.compare
        sa, sb = by_group.get(a, {}), by_group.get(b, {})
        common = sorted(set(sa) & set(sb))
        if len(common) < 2:
            print(f"\n[compare] {a} vs {b}: only {len(common)} matched seeds — skipped")
        else:
            ma = np.array([np.mean(list(sa[s]["scores"].values())) for s in common]) * 100
            mb = np.array([np.mean(list(sb[s]["scores"].values())) for s in common]) * 100
            diff = ma - mb
            t, p = ttest_rel(ma, mb)
            sd = diff.std(ddof=1)
            half = t_dist.ppf(0.975, len(diff) - 1) * sd / np.sqrt(len(diff))
            print(f"\n[compare] {a} - {b} over {len(common)} matched seeds "
                  f"(seeds {common}):")
            print(f"  mean diff {diff.mean():+.2f} pp "
                  f"(95% CI [{diff.mean() - half:+.2f}, {diff.mean() + half:+.2f}]), "
                  f"sd {sd:.2f}, paired t={t:.2f}, p={p:.4g}")

    if args.manifest:
        with open(args.manifest, "w") as f:
            f.write("group\tseed\tstep\tn_tasks\thost\tpath\tmean_win_rate\n")
            for (g, seed), d in sorted(picked.items()):
                if args.groups and g not in args.groups:
                    continue
                mw = float(np.mean(list(d["scores"].values())))
                f.write(f"{g}\t{seed}\t{d['step']}\t{d['n_tasks']}\t"
                        f"{d.get('host','?')}\t{d.get('path','?')}\t{mw:.6f}\n")
        print(f"\n[manifest] {sum(1 for k in picked if not args.groups or k[0] in args.groups)} "
              f"runs written to {args.manifest}")


if __name__ == "__main__":
    main()
