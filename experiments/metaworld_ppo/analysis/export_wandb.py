r"""Export per-task final scores from W&B in collector format.

Each run's final per-task scores are read from its W&B run summary and
emitted in exactly the format `collect_csv.py` produces, so rows read from
W&B and rows collected from local run directories go through the SAME
aggregator and the same metric definitions — the only difference is where
each run's final per-task scores were read from.  This is the route for
recomputing the paper's numbers from the published runs.

Only runs whose W&B history is intact are exported: a preempted+resumed run
re-logs steps W&B rejects, freezing its remote history mid-training, so
`--require-step` rejects anything that did not reach the final checkpoint.
(`collect_csv.py` reads the local `metric.csv`, which is appended across
resumes, and is not affected.)

The suite is inferred per run from its number of task keys, and
`--require-step` defaults to that suite's full budget (100M for 50 tasks,
20M for 10).  `--n-tasks` turns the inference into a strict check.

Source: the published runs, liyp1220-uci/TOPPO, unless --entity, --project or
$WANDB_PROJECT is given.  A project named by --project, else $WANDB_PROJECT (any
name, TOPPO included), is read under --entity, else $WANDB_ENTITY, else your W&B
default entity; --entity alone reads <entity>/TOPPO.  $WANDB_ENTITY alone, a
common W&B setup, does not move the source.  metrics.py, export_wandb.py and
check_runs_index.py --wandb share this rule (resolve_source).  Training logs to
project toppo-repro by default, so read your own runs with
--entity <you> --project toppo-repro.

    python analysis/export_wandb.py --groups <g> [<g> ...] \
        [--n-tasks 50] [--require-step 100000000] > refs.jsonl
"""

from __future__ import annotations

import argparse
import os
import json
import sys

PREFIX = "test_win_rate/"
# The published runs; --entity / --project (or $WANDB_PROJECT, see
# resolve_source) point the script at your own runs instead.
PUBLIC_ENTITY = "liyp1220-uci"
PUBLIC_PROJECT = "TOPPO"
# Full training budget (env steps) per suite, keyed by task count.
STEP_BUDGET = {50: 100_000_000, 10: 20_000_000}


def resolve_source(entity, project):
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", required=True,
                    help="W&B group names (docs/RUNS.md lists the published ones)")
    ap.add_argument("--entity", default=None,
                    help=f"W&B entity (default: {PUBLIC_ENTITY}, which holds the published "
                         f"runs; when --project or $WANDB_PROJECT names a project: "
                         f"$WANDB_ENTITY, else your W&B default entity)")
    ap.add_argument("--project", default=None,
                    help=f"W&B project (default: $WANDB_PROJECT, else {PUBLIC_PROJECT}); "
                         f"naming one, even {PUBLIC_PROJECT}, reads it under --entity, "
                         f"else $WANDB_ENTITY, else your W&B default entity. Runs trained "
                         f"with the shipped configs are in project toppo-repro")
    ap.add_argument("--require-step", "--min-step", dest="require_step", type=int,
                    default=None,
                    help="drop runs whose summary _step is below this. Defaults to the "
                         "full budget of the suite each run belongs to (100000000 for "
                         "50 tasks, 20000000 for 10); pass 0 to disable the check")
    ap.add_argument("--n-tasks", type=int, default=None,
                    help="expected per-task key count (50 on MT50, 10 on MT10); "
                         "inferred per run when omitted")
    args = ap.parse_args()
    if (args.require_step is None and args.n_tasks is not None
            and args.n_tasks not in STEP_BUDGET):
        ap.error("--require-step is required when --n-tasks is neither 50 nor 10")

    import wandb
    api = wandb.Api()
    entity, args.project = resolve_source(args.entity, args.project)
    args.entity = entity or api.default_entity
    if not args.entity:
        sys.exit(f"ERROR: no W&B entity for project {args.project}; pass --entity.")
    print(f"[source] {args.entity}/{args.project}", file=sys.stderr)
    for group in args.groups:
        try:
            runs = list(api.runs(f"{args.entity}/{args.project}",
                                 filters={"group": group}))
        except ValueError as exc:
            if "Could not find project" not in str(exc):
                raise
            sys.exit(f"ERROR: W&B project {args.entity}/{args.project} not found; pass "
                     f"--entity/--project (the published runs, {PUBLIC_ENTITY}/"
                     f"{PUBLIC_PROJECT}, are the default when none of --entity, --project and "
                     f"$WANDB_PROJECT is given).")
        kept = 0
        for r in runs:
            s = dict(r.summary)
            scores = {k[len(PREFIX):]: float(v) for k, v in s.items()
                      if k.startswith(PREFIX) and isinstance(v, (int, float))}
            step = s.get("_step")
            if not scores:
                print(f"[skip] {r.name}: no '{PREFIX}*' keys in its summary",
                      file=sys.stderr)
                continue
            if args.n_tasks is not None and len(scores) != args.n_tasks:
                print(f"[skip] {r.name}: {len(scores)} task keys (want {args.n_tasks})",
                      file=sys.stderr)
                continue
            need = (args.require_step if args.require_step is not None
                    else STEP_BUDGET.get(len(scores)))
            if need is None:
                print(f"[skip] {r.name}: {len(scores)} task keys, so the step budget "
                      f"is unknown; pass --require-step", file=sys.stderr)
                continue
            # 0 disables the step check entirely, as in metrics.py: a run whose
            # summary has no _step is then kept (and exported with step 0).
            if need > 0 and (step is None or int(step) < need):
                print(f"[skip] {r.name}: _step={step} < {need} "
                      f"(state={r.state}; W&B history truncated or unfinished)",
                      file=sys.stderr)
                continue
            print(json.dumps({
                "exp": r.name,
                "host": "wandb",
                "path": f"{args.entity}/{args.project}/{r.id}",
                "step": int(step) if step is not None else 0,
                "n_tasks": len(scores),
                "scores": scores,
            }))
            kept += 1
        print(f"[{group}] exported {kept}/{len(runs)}", file=sys.stderr)


if __name__ == "__main__":
    main()
