#!/usr/bin/env python
r"""Metrics: final-checkpoint success-rate aggregation from W&B (MT10 and MT50).

Pulls every run in the requested W&B groups, reads the FINAL (summary) values of
the per-task eval keys ``test_win_rate/<task>-v3`` (50 on MT50, 10 on MT10), and
reports the paper's two reporting conventions:

  V2 convention  -- cross-task mean success rate per seed, then mean +/- std across
                    seeds; plus worst-k (per-seed average of the k lowest per-task
                    rates, then mean +/- std across seeds).
  V1 convention  -- IQM with a 25% trim (rliable / Meta-World+ protocol) +/- std.

VALIDATED IQM DEFINITION (reproduces the paper's published V1 numbers exactly):
  IQM = trim_mean(per-seed cross-task means, proportiontocut=0.25)
        i.e. with 10 seeds: sort the 10 seed means, drop the 2 lowest and the
        2 highest, average the remaining 6.
  std = SAMPLE std (ddof=1) over ALL n seed means -- untrimmed, identical to the
        std in the "mean +/- std" column.

Reproduced from the published runs (each matches the paper):
  V1 IQM    mt50v1_ppo_ad_cd_nnone_lnc_vc1_400x3 -> 64.3 +/- 3.1
            mt50v1_ppo_ad_cd_npaw_lnc_vc1_400x3  -> 76.5 +/- 2.0
  V2 mean   mt50v2_ppo_ad_cd_nnone_lnc_vc1_400x3 -> 87.5 +/- 2.4, worst-10 42.3 +/- 10.1
            mt50v2_ppo_ad_cd_npaw_lnc_vc1_400x3  -> 88.9 +/- 2.3, worst-10 48.9 +/- 9.0

Two other IQM readings are also computed and shown with --iqm-variants:

  iqm_pooled       rliable's canonical definition: scipy.stats.trim_mean with
                   proportiontocut=0.25 over the FLATTENED (n_seeds x n_tasks)
                   score matrix -> a single scalar, no natural std. This does NOT
                   reproduce the paper (on MT50 it lands ~13-22 pp higher).
  iqm_task_trim    per seed, trim 25% of the task scores from each tail and
                   average the middle 50% (13..38 of 50 sorted on MT50) -> one
                   value per seed; mean +/- std across seeds. Also does not
                   reproduce.

Trimming always uses scipy's convention lowercut = int(0.25 * n) (floor), so
n=50 -> drop 12 per tail (26 kept), n=10 -> drop 2 per tail (6 kept),
n=500 -> drop 125 per tail (250 kept).

Suite: the task count is inferred per group from the number of test_win_rate/*
keys its runs carry (or fixed with --n-tasks), and --min-step defaults to that
suite's full budget (100M on MT50, 20M on MT10).

Source: the published runs, liyp1220-uci/TOPPO, unless --entity, --project or
$WANDB_PROJECT is given.  A project named by --project, else $WANDB_PROJECT (any
name, TOPPO included), is read under --entity, else $WANDB_ENTITY, else your W&B
default entity; --entity alone reads <entity>/TOPPO.  $WANDB_ENTITY alone, a
common W&B setup, does not move the source.  metrics.py, export_wandb.py and
check_runs_index.py --wandb share this rule (resolve_source).  Training logs to
project toppo-repro by default, so read your own runs with
--entity <you> --project toppo-repro.

Usage:
    python analysis/metrics.py --groups mt50v2_ppo_apc_cfg1_npaw_lnc_vc1_400x3 \
        mt50v2_ppo_ad_cd_npaw_lnc_vc1_400x3                          # MT50
    python analysis/metrics.py --groups mt10v2_ppo_apc_cfg1_npaw_lnc_vc1_400x3   # MT10
    python analysis/metrics.py --entity <you> --project toppo-repro --groups <group>   # your runs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# The published runs.  --entity / --project (or $WANDB_PROJECT, see
# resolve_source) point the script at your own runs instead.
PUBLIC_ENTITY = "liyp1220-uci"
PUBLIC_PROJECT = "TOPPO"
DEFAULT_PREFIX = "test_win_rate/"
DEFAULT_MEAN_KEY = "test/mean_win_rate"
# Full training budget (env steps) per suite, keyed by task count.
STEP_BUDGET = {50: 100_000_000, 10: 20_000_000}

SEED_RE = re.compile(r"_s(\d+)$")


# --------------------------------------------------------------------------- #
# small stats helpers (no numpy/scipy dependency required)
# --------------------------------------------------------------------------- #
def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def std(xs: Sequence[float], ddof: int = 0) -> float:
    n = len(xs)
    if n - ddof <= 0:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def trim_mean(xs: Sequence[float], proportiontocut: float = 0.25) -> float:
    """scipy.stats.trim_mean semantics: lowercut = int(p*n), uppercut = n - lowercut."""
    s = sorted(xs)
    n = len(s)
    lo = int(proportiontocut * n)
    hi = n - lo
    if lo >= hi:
        return float("nan")
    return mean(s[lo:hi])


def worst_k(xs: Sequence[float], k: int) -> float:
    """Average of the k lowest values."""
    if k > len(xs):
        return float("nan")
    return mean(sorted(xs)[:k])


def fmt(m: float, s: Optional[float] = None, scale: float = 100.0, nd: int = 1) -> str:
    if m != m:  # NaN
        return "n/a"
    if s is None or s != s:
        return f"{m * scale:.{nd}f}"
    return f"{m * scale:.{nd}f} +/- {s * scale:.{nd}f}"


# --------------------------------------------------------------------------- #
# wandb fetch
# --------------------------------------------------------------------------- #
def resolve_source(entity: Optional[str], project: Optional[str]
                   ) -> Tuple[Optional[str], str]:
    """(entity, project) to read; entity None means the caller's W&B default entity.

    Identical in metrics.py, export_wandb.py and check_runs_index.py; keep the
    three in sync.  With no --entity, no --project and no $WANDB_PROJECT, the
    published runs.  Otherwise the project is --project, else $WANDB_PROJECT,
    else TOPPO, and the entity is --entity, else $WANDB_ENTITY, else the W&B
    default entity.  $WANDB_ENTITY alone therefore leaves the published default
    in place instead of pointing it at <your entity>/TOPPO.
    """
    project = project or os.environ.get("WANDB_PROJECT")
    if not project:
        return entity or PUBLIC_ENTITY, PUBLIC_PROJECT
    return entity or os.environ.get("WANDB_ENTITY") or None, project


class SeedRecord:
    def __init__(self, run_name: str, run_id: str, seed: Optional[int], state: str,
                 step: Optional[int], scores: Dict[str, float], logged_mean: Optional[float]):
        self.run_name = run_name
        self.run_id = run_id
        self.seed = seed
        self.state = state
        self.step = step
        self.scores = scores          # task -> final success rate
        self.logged_mean = logged_mean

    @property
    def task_values(self) -> List[float]:
        return [self.scores[t] for t in sorted(self.scores)]

    @property
    def cross_task_mean(self) -> float:
        return mean(self.task_values)


def infer_n_tasks(counts: Sequence[int]) -> Optional[int]:
    """Most common per-task key count among a group's runs (ties -> the larger)."""
    counts = [c for c in counts if c > 0]
    if not counts:
        return None
    return max(set(counts), key=lambda c: (counts.count(c), c))


def fetch_group(api, entity: str, project: str, group: str, prefix: str,
                mean_key: str, n_tasks: Optional[int], min_step: Optional[int],
                require_finished: bool
                ) -> Tuple[List[SeedRecord], List[str], Optional[int], Optional[int]]:
    """Return (accepted seed records, rejection notes, n_tasks, min_step used).

    ``n_tasks=None`` infers the task count from the group's runs.  ``min_step=None``
    uses the full budget of that suite (STEP_BUDGET); ``min_step=0`` disables the
    step check.
    """
    runs = api.runs(f"{entity}/{project}", filters={"group": group}, per_page=200)
    notes: List[str] = []

    parsed = []
    for run in runs:
        try:
            summary = dict(run.summary._json_dict)
        except Exception as exc:  # pragma: no cover - network/API hiccup
            notes.append(f"{run.name}: summary unavailable ({exc})")
            continue

        scores = {}
        for k, v in summary.items():
            if not k.startswith(prefix):
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                scores[k[len(prefix):]] = float(v)
        parsed.append((run, summary, scores))

    if n_tasks is None:
        n_tasks = infer_n_tasks([len(scores) for _, _, scores in parsed])
        if n_tasks is None and parsed:
            notes.append(f"no run in this group carries '{prefix}*' keys; check the "
                         f"group name and --prefix. All runs dropped")
            return [], notes, None, min_step
    if min_step is None:
        # "Final checkpoint at the full budget" is the project's convention, so
        # make it the default rather than something the caller must remember.
        # W&B state == "finished" is not sufficient: a job that stopped early
        # still reports finished and would otherwise be averaged in silently.
        min_step = STEP_BUDGET.get(n_tasks)
        if min_step is None and parsed:
            notes.append(f"no step budget known for {n_tasks} tasks; pass --min-step "
                         f"(or --min-step 0 to disable the check). All runs dropped")
            return [], notes, n_tasks, None
    elif min_step == 0:
        min_step = None

    accepted: List[SeedRecord] = []
    for run, summary, scores in parsed:
        disp = run.name  # wandb python API: .name is the display name
        rid = run.id
        m = SEED_RE.search(disp or "")
        seed = int(m.group(1)) if m else None
        step = summary.get("_step")
        step = int(step) if isinstance(step, (int, float)) else None
        logged_mean = summary.get(mean_key)
        logged_mean = float(logged_mean) if isinstance(logged_mean, (int, float)) else None

        if require_finished and run.state != "finished":
            notes.append(f"{disp} ({rid}): state={run.state}, dropped")
            continue
        if len(scores) != n_tasks:
            notes.append(f"{disp} ({rid}): {len(scores)} '{prefix}*' keys (want {n_tasks}), dropped")
            continue
        if min_step is not None and (step is None or step < min_step):
            notes.append(f"{disp} ({rid}): _step={step} < {min_step}, dropped")
            continue

        accepted.append(SeedRecord(disp, rid, seed, run.state, step, scores, logged_mean))

    # de-duplicate by seed: keep the run with the largest _step, then latest id
    by_seed: Dict[object, SeedRecord] = {}
    for rec in accepted:
        key = rec.seed if rec.seed is not None else rec.run_id
        prev = by_seed.get(key)
        if prev is None:
            by_seed[key] = rec
        else:
            keep, drop = (rec, prev) if (rec.step or 0) > (prev.step or 0) else (prev, rec)
            by_seed[key] = keep
            notes.append(f"duplicate seed {key}: kept {keep.run_id} (_step={keep.step}), "
                         f"dropped {drop.run_id} (_step={drop.step})")

    out = sorted(by_seed.values(), key=lambda r: (r.seed if r.seed is not None else 1 << 30, r.run_id))
    return out, notes, n_tasks, min_step


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def aggregate(records: List[SeedRecord], ks: Sequence[int], ddof: int,
              trim: float = 0.25) -> dict:
    seed_means = [r.cross_task_mean for r in records]
    pooled = [v for r in records for v in r.task_values]

    per_seed_iqm = [trim_mean(r.task_values, trim) for r in records]

    res = {
        "n_seeds": len(records),
        "seeds": [r.seed for r in records],
        "run_ids": [r.run_id for r in records],
        "n_tasks": len(records[0].scores) if records else 0,
        "seed_means": seed_means,
        "mean": mean(seed_means) if seed_means else float("nan"),
        "mean_std": std(seed_means, ddof) if seed_means else float("nan"),
        # IQM variants
        "iqm_task_trim": mean(per_seed_iqm) if per_seed_iqm else float("nan"),
        "iqm_task_trim_std": std(per_seed_iqm, ddof) if per_seed_iqm else float("nan"),
        "iqm_task_trim_per_seed": per_seed_iqm,
        "iqm_seed_trim": trim_mean(seed_means, trim) if seed_means else float("nan"),
        "iqm_seed_trim_std": std(seed_means, ddof) if seed_means else float("nan"),
        "iqm_pooled": trim_mean(pooled, trim) if pooled else float("nan"),
    }
    for k in ks:
        vals = [worst_k(r.task_values, k) for r in records]
        res[f"worst{k}"] = mean(vals) if vals else float("nan")
        res[f"worst{k}_std"] = std(vals, ddof) if vals else float("nan")
        res[f"worst{k}_per_seed"] = vals
    return res


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--groups", nargs="+", required=True,
                   help="W&B group names (docs/RUNS.md lists the published ones)")
    p.add_argument("--entity", default=None,
                   help=f"W&B entity (default: {PUBLIC_ENTITY}, which holds the published "
                        f"runs; when --project or $WANDB_PROJECT names a project: "
                        f"$WANDB_ENTITY, else your W&B default entity)")
    p.add_argument("--project", default=None,
                   help=f"W&B project (default: $WANDB_PROJECT, else {PUBLIC_PROJECT}); "
                        f"naming one, even {PUBLIC_PROJECT}, reads it under --entity, "
                        f"else $WANDB_ENTITY, else your W&B default entity. Runs trained "
                        f"with the shipped configs are in project toppo-repro")
    p.add_argument("--prefix", default=DEFAULT_PREFIX,
                   help="per-task metric key prefix (default: test_win_rate/)")
    p.add_argument("--mean-key", default=DEFAULT_MEAN_KEY,
                   help="logged cross-task mean key, used only as a cross-check")
    p.add_argument("--n-tasks", type=int, default=None,
                   help="per-task keys a run must carry (50 on MT50, 10 on MT10). "
                        "Inferred per group from its runs when omitted")
    p.add_argument("--worst-k", nargs="+", type=int, default=[5, 10, 20])
    p.add_argument("--trim", type=float, default=0.25, help="IQM proportion to cut per tail")
    p.add_argument("--ddof", type=int, default=1,
                   help="std delta-dof: 1 = sample std (default, matches the paper), "
                        "0 = population std")
    p.add_argument("--min-step", type=int, default=None,
                   help="drop runs whose summary _step is below this. Defaults to the "
                        "benchmark budget implied by the task count (100000000 for 50 "
                        "tasks, 20000000 for 10); pass 0 to disable the check")
    p.add_argument("--allow-unfinished", action="store_true",
                   help="keep runs whose state != finished (if they still have all task keys)")
    p.add_argument("--iqm-variants", action="store_true",
                   help="print all three IQM variants instead of just the default one")
    p.add_argument("--per-seed", action="store_true", help="print the per-seed breakdown")
    p.add_argument("--json-out", default=None, help="also dump raw results to this JSON path")
    args = p.parse_args(argv)
    if (args.min_step is None and args.n_tasks is not None
            and args.n_tasks not in STEP_BUDGET):
        p.error("--min-step is required when --n-tasks is neither 50 nor 10")

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb is not importable in this interpreter.", file=sys.stderr)
        return 2

    api = wandb.Api(timeout=60)
    entity, args.project = resolve_source(args.entity, args.project)
    args.entity = entity or api.default_entity
    if not args.entity:
        print(f"ERROR: no W&B entity for project {args.project}; pass --entity.",
              file=sys.stderr)
        return 2
    print(f"W&B source: {args.entity}/{args.project}")

    results = {}
    all_notes = {}
    for g in args.groups:
        try:
            recs, notes, n_tasks, min_step = fetch_group(
                api, args.entity, args.project, g, args.prefix, args.mean_key,
                args.n_tasks, args.min_step, require_finished=not args.allow_unfinished)
        except ValueError as exc:
            if "Could not find project" not in str(exc):
                raise
            print(f"ERROR: W&B project {args.entity}/{args.project} not found; pass "
                  f"--entity/--project (the published runs, {PUBLIC_ENTITY}/"
                  f"{PUBLIC_PROJECT}, are the default when none of --entity, --project and "
                  f"$WANDB_PROJECT is given).", file=sys.stderr)
            return 2
        all_notes[g] = notes
        if not recs:
            results[g] = {"n_seeds": 0}
            continue
        results[g] = aggregate(recs, args.worst_k, args.ddof, args.trim)
        results[g]["min_step"] = min_step
        results[g]["records"] = [
            {"run": r.run_name, "id": r.run_id, "seed": r.seed, "step": r.step,
             "state": r.state, "cross_task_mean": r.cross_task_mean,
             "logged_mean_win_rate": r.logged_mean,
             "per_task": r.scores}
            for r in recs
        ]
        # sanity: our cross-task mean must match the logged test/mean_win_rate
        for r in recs:
            if r.logged_mean is not None and abs(r.logged_mean - r.cross_task_mean) > 1e-6:
                all_notes[g].append(
                    f"{r.run_name}: logged {args.mean_key}={r.logged_mean:.6f} != "
                    f"recomputed cross-task mean {r.cross_task_mean:.6f}")

    # ------------------------------------------------------------------ table
    ks = args.worst_k
    header = ["group", "n_seeds", "n_tasks", "mean +/- std", "IQM +/- std"]
    if args.iqm_variants:
        header += ["IQM_task_trim +/- std", "IQM_pooled"]
    header += [f"worst-{k} +/- std" for k in ks]

    rows = []
    for g in args.groups:
        r = results[g]
        if r.get("n_seeds", 0) == 0:
            rows.append([g, "0"] + ["n/a"] * (len(header) - 2))
            continue
        row = [g, str(r["n_seeds"]), str(r["n_tasks"]), fmt(r["mean"], r["mean_std"]),
               fmt(r["iqm_seed_trim"], r["iqm_seed_trim_std"])]
        if args.iqm_variants:
            row += [fmt(r["iqm_task_trim"], r["iqm_task_trim_std"]),
                    fmt(r["iqm_pooled"])]
        row += [fmt(r[f"worst{k}"], r[f"worst{k}_std"]) for k in ks]
        rows.append(row)

    widths = [max(len(header[i]), *(len(row[i]) for row in rows)) for i in range(len(header))]
    print()
    print("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |")
    print()
    print(f"(all values x100; std = std over ALL n seed values, ddof={args.ddof}; "
          f"IQM = trim_mean(seed means, {args.trim} per tail, scipy floor convention); "
          f"worst-k = per-seed mean of the k lowest per-task rates, then mean/std across seeds; "
          f"metric key '{args.prefix}*' read from the run summary = final logged step)")

    if args.per_seed:
        for g in args.groups:
            r = results[g]
            if r.get("n_seeds", 0) == 0:
                continue
            print(f"\n### {g}")
            for rec in r["records"]:
                print(f"  seed {rec['seed']:>2}  {rec['id']}  _step={rec['step']}  "
                      f"mean={rec['cross_task_mean'] * 100:.2f}")

    for g, notes in all_notes.items():
        if notes:
            print(f"\n[notes] {g}")
            for n in notes:
                print(f"  - {n}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"results": results, "notes": all_notes, "args": vars(args)}, fh, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
