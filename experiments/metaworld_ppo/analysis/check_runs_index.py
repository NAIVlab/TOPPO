#!/usr/bin/env python3
"""Verify docs/RUNS.md: every row must reproduce the arm its group name encodes.

For each row, load the listed config with the listed command-line overrides through
the real CLI parser, then compare the resulting effective settings field by field
against the settings decoded from the W&B group name, and the training schedule and
PPO numerics against the paper's hyperparameter table.  Exits non-zero on any
mismatch, so the claim RUNS.md makes about itself is auditable rather than asserted.

With --wandb it also reads the row's runs from W&B (read-only) and compares, one
by one, every field of this release's config schema that the runs recorded against
the effective config; only run identity, output locations and checkpoint cadence
are left out (BOOKKEEPING).  Older runs lack some schema fields (several early
groups never recorded gradient.actor_method / critic_method or network.ln_critic,
so their arm rests on the group-name check above), and legacy keys outside the
schema (gradient.method, gradient.scope, the toppo/* block, ...) are not compared.

W&B source: the published runs, liyp1220-uci/TOPPO, unless --entity, --project or
$WANDB_PROJECT is given; the rule is resolve_source below, shared with metrics.py
and export_wandb.py.  Runs trained with the shipped configs are in project
toppo-repro: check them with --wandb --entity <you> --project toppo-repro.

    python analysis/check_runs_index.py [--runs-md ../../docs/RUNS.md] [--wandb]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _ROOT)


def _load_config_module():
    """Import mtrl/utils/config.py on its own.

    Importing it as `mtrl.utils.config` would execute the package __init__, which
    pulls in the logger, numpy and tianshou. This checker only needs the config
    dataclasses, so it loads the module file directly and stays runnable with
    nothing but pyyaml and tyro installed.
    """
    import importlib.util
    path = os.path.join(_ROOT, "mtrl", "utils", "config.py")
    spec = importlib.util.spec_from_file_location("_mtrl_config_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


load_config = _load_config_module().load_config

GROUP_RE = re.compile(
    r"^mt(?P<bench>10|50)v(?P<rv>1|2)_(?P<algo>ppo|sac)"
    r"_a(?P<actor>[a-z0-9p.]+)_c(?P<critic>[a-z0-9p.]+)_n(?P<norm>[a-z0-9]+)"
    r"_(?P<ln>ln0|lnc|lnac|lnci)_(?P<vc>vc0|vc1|vcx)_(?P<arch>.+)$"
)
ROW_RE = re.compile(r"^\|\s*`(mt\S+?)`\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|")

# normalizer code -> (reward_normalization_method, use_popart, popart_method,
#                     use_return_scaling, critic_loss_mode, hlgauss_num_bins)
NORM = {
    "none":      (None, False, "welford", False, "mse", None),
    "paw":       (None, True, "welford", False, "mse", None),
    "pae":       (None, True, "ema", False, "mse", None),
    "rew":       ("baselines_return", False, "welford", False, "mse", None),
    "ret":       (None, False, "welford", True, "mse", None),
    "rettr":     (None, False, "welford", True, "mse", None),
}
# hg<bins>[rew|paw] is decoded from the code itself
ARCH = {"400x3": [400, 400, 400], "400x4": [400, 400, 400, 400],
        "1024x4": [1024, 1024, 1024, 1024]}

# The paper's hyperparameter table (tab:hyperparameters) and evaluation grid, which
# every published PPO row uses: per-benchmark rollout schedule, evaluation every 2
# epochs (51 checkpoints), and shared optimizer / PPO numerics.
SCHEDULE = {
    "10": dict(num_tasks=10, num_envs=10, epoch=100, step_per_epoch=200_000,
               step_per_collect=100_000, repeat_per_collect=16, batch_size=3_125),
    "50": dict(num_tasks=50, num_envs=50, epoch=100, step_per_epoch=1_000_000,
               step_per_collect=500_000, repeat_per_collect=16, batch_size=15_625),
}
SHARED = dict(test_interval=2, minibatch_sampling="stratified", learning_rate=3e-4,
              gamma=0.99, max_grad_norm=1.0, gae_lambda=0.97, clip_range=0.2,
              vf_coef=1e-3, ent_coef=5e-3, advantage_normalization=True,
              recompute_advantage=True)
PAPER_FIELDS = set(SCHEDULE["10"]) | set(SHARED)

# Recorded on the runs but not part of what is trained or evaluated: run identity,
# output locations and checkpoint cadence.  --wandb compares every other field.
BOOKKEEPING = {"seed", "exp_name", "project_name", "config", "device",
               "logging.log_dir", "logging.wandb_group", "logging.wandb_tags",
               "logging.wandb_notes", "logging.wandb_mode",
               "logging.save_interval", "logging.save_best"}
# The published runs; --entity / --project (or $WANDB_PROJECT, see
# resolve_source) point --wandb at your own runs instead.
PUBLIC_ENTITY = "liyp1220-uci"
PUBLIC_PROJECT = "TOPPO"
SEEDS_PER_ROW = 10


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


def decode(group: str):
    m = GROUP_RE.match(group)
    if not m:
        return None, f"group name does not parse"
    g = m.groupdict()
    if g["algo"] == "sac":
        return None, None                      # configured by the ARS launchers
    arch = g["arch"]
    variant = None
    for suf in ("_gjoint", "_gsplit", "_lora16"):
        if arch.endswith(suf):
            arch, variant = arch[: -len(suf)], suf[1:]
    critic_width = None
    cm = re.match(r"^(.*?)c(\d+)x(\d+)$", arch)
    if cm:                                     # e.g. 400x3c32x3 -> critic [32,32,32]
        arch, critic_width = cm.group(1), [int(cm.group(2))] * int(cm.group(3))
    if arch not in ARCH:
        return None, f"unknown architecture code {arch!r}"
    norm, bins = g["norm"], None
    hg = re.match(r"^hg(\d+)(rew|paw)?$", norm)
    if hg:
        bins = int(hg.group(1))
        tail = hg.group(2)
        rn = "baselines_return" if tail == "rew" else None
        pa = tail == "paw"
        spec = (rn, pa, "welford", False, "hlgauss", bins)
    elif norm in NORM:
        spec = NORM[norm]
    else:
        return None, f"unknown normalizer code {norm!r}"
    rn, pa, pm, rs, cl, bins = spec

    def side(code):
        if code == "d":
            return "default", None
        if code == "pc":
            return "pcgrad", None
        if code.startswith("fg"):
            return "fairgrad", float(code[2:].replace("p", ".")) if code[2:] else None
        if code.startswith("ca"):
            return "cagrad", float(code[2:].replace("p", ".")) if code[2:] else None
        return code, None

    am, aa = side(g["actor"])
    cmeth, ca = side(g["critic"])
    sched = SCHEDULE[g["bench"]]
    return dict(
        env_name="MT" + g["bench"], reward_function_version="v" + g["rv"],
        ln_critic=g["ln"] in ("lnc", "lnac"), ln_actor=g["ln"] == "lnac",
        value_clip=g["vc"] == "vc1", reward_normalization_method=rn,
        use_popart=pa, popart_method=pm, use_return_scaling=rs,
        return_scaling_truncation_as_terminal=(norm == "rettr"),
        critic_loss_mode=cl, hlgauss_num_bins=bins,
        actor_method=am, critic_method=cmeth, actor_alpha=aa, critic_alpha=ca,
        actor_arch=ARCH[arch], critic_arch=critic_width or ARCH[arch],
        variant=variant,
        env_steps=sched["epoch"] * sched["step_per_epoch"],
        use_linear_baseline=False,      # every published PPO row uses a neural critic
        actor_head=[], critic_head=[],  # and no per-task heads
        **sched, **SHARED,
    ), None


def effective(cfg):
    g = cfg.gradient
    aa = (g.fairgrad_actor_alpha if g.actor_method == "fairgrad"
          else g.cagrad_alpha if g.actor_method == "cagrad" else None)
    ca = (g.fairgrad_critic_alpha if g.critic_method == "fairgrad"
          else g.cagrad_alpha if g.critic_method == "cagrad" else None)
    return dict(
        env_name=cfg.env.env_name, reward_function_version=cfg.env.reward_function_version,
        ln_critic=cfg.network.ln_critic, ln_actor=cfg.network.ln_actor,
        value_clip=cfg.ppo.value_clip,
        reward_normalization_method=cfg.env.reward_normalization_method,
        use_popart=cfg.ppo.use_popart, popart_method=cfg.ppo.popart_method,
        use_return_scaling=cfg.ppo.use_return_scaling,
        return_scaling_truncation_as_terminal=cfg.ppo.return_scaling_truncation_as_terminal,
        critic_loss_mode=cfg.ppo.critic_loss_mode,
        hlgauss_num_bins=(cfg.ppo.hlgauss_num_bins
                          if cfg.ppo.critic_loss_mode == "hlgauss" else None),
        actor_method=g.actor_method, critic_method=g.critic_method,
        actor_alpha=aa, critic_alpha=ca,
        actor_arch=list(cfg.network.actor), critic_arch=list(cfg.network.critic),
        use_linear_baseline=bool(cfg.use_linear_baseline),
        actor_head=list(cfg.network.actor_head or []),
        critic_head=list(cfg.network.critic_head or []),
        num_tasks=cfg.env.num_tasks, num_envs=cfg.env.num_envs,
        **{k: getattr(cfg.training, k) for k in (
            "epoch", "step_per_epoch", "step_per_collect", "repeat_per_collect",
            "batch_size", "minibatch_sampling", "learning_rate", "gamma", "max_grad_norm")},
        test_interval=cfg.logging.test_interval,
        **{k: getattr(cfg.ppo, k) for k in (
            "gae_lambda", "clip_range", "vf_coef", "ent_coef",
            "advantage_normalization", "recompute_advantage")},
    )


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[prefix + k] = v
    return out


def _same(a, b) -> bool:
    """Equality that keeps bools apart from ints and lets 2 == 2.0."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))
    return a == b


def compare_with_runs(cfg, runs) -> tuple[list[str], list[str]]:
    """Compare the effective config with every field the runs recorded.

    Returns (divergences, notes).  A field diverges when no run recorded the
    config's value.  Fields absent from a run (older config schemas) are skipped
    for that run; the arm itself is already pinned by the group name.  When the
    runs disagree among themselves and the config matches some of them, that is
    a note, since no single config can match every seed.
    """
    got = {k: v for k, v in _flat(asdict(cfg)).items() if k not in BOOKKEEPING}
    recorded: dict[str, dict[str, list[str]]] = {}
    for run in runs:
        for k, v in _flat(dict(run.config)).items():
            if k in got:
                recorded.setdefault(k, {}).setdefault(json.dumps(v), []).append(run.name)
    divergences, notes = [], []
    for k, by_value in sorted(recorded.items()):
        values = {s: json.loads(s) for s in by_value}
        if all(_same(got[k], v) for v in values.values()):
            continue
        seen = ", ".join(f"{v!r} on {len(by_value[s])}" for s, v in values.items())
        if any(_same(got[k], v) for v in values.values()):
            odd = sorted(n for s, v in values.items() if not _same(got[k], v)
                         for n in by_value[s])
            notes.append(f"{k}: runs disagree ({seen} runs); config has {got[k]!r}; "
                         f"differing: {', '.join(odd)}")
        else:
            divergences.append(f"{k}={got[k]!r}, but the runs recorded {seen} runs")
    return divergences, notes


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-md", default=os.path.join(root, "..", "..", "docs", "RUNS.md"))
    ap.add_argument("--wandb", action="store_true",
                    help="also compare each row with the configs its W&B runs recorded "
                         "(read-only; needs the wandb package)")
    ap.add_argument("--entity", default=None,
                    help=f"W&B entity for --wandb (default: {PUBLIC_ENTITY}, which holds "
                         f"the published runs; when --project or $WANDB_PROJECT names a "
                         f"project: $WANDB_ENTITY, else your W&B default entity)")
    ap.add_argument("--project", default=None,
                    help=f"W&B project for --wandb (default: $WANDB_PROJECT, else "
                         f"{PUBLIC_PROJECT}); naming one, even {PUBLIC_PROJECT}, reads it "
                         f"under --entity, else $WANDB_ENTITY, else your W&B default "
                         f"entity. Runs trained with the shipped configs are in project "
                         f"toppo-repro")
    args = ap.parse_args()
    api = None
    if args.wandb:
        import wandb
        api = wandb.Api(timeout=120)
        entity, args.project = resolve_source(args.entity, args.project)
        args.entity = entity or api.default_entity
        if not args.entity:
            print(f"ERROR: no W&B entity for project {args.project}; pass --entity.")
            return 2
        print(f"W&B source: {args.entity}/{args.project}")

    checked = skipped = bad = n_runs = 0
    seen_rows: set[str] = set()
    for line in open(args.runs_md):
        m = ROW_RE.match(line)
        if not m:
            continue
        group, cfg_cell, ov_cell, steps_cell = m.groups()
        if group in seen_rows:
            print(f"DUPLICATE ROW {group}")
            bad += 1
            continue
        seen_rows.add(group)
        want, err = decode(group)
        if err:
            print(f"UNPARSED {group}: {err}")
            bad += 1
            continue
        cfg_path = cfg_cell.strip("` ")
        if want is None or cfg_path in ("—", "-", ""):
            skipped += 1
            continue
        overrides = [t for t in ov_cell.strip("` ").split() if t not in ("—", "-")]
        variant = want.pop("variant")
        want_steps = want.pop("env_steps")
        steps_txt = steps_cell.strip("` ").upper().replace(" ", "")
        mult = {"M": 1_000_000, "K": 1_000, "B": 1_000_000_000}
        if steps_txt and steps_txt[-1] in mult and steps_txt[:-1].replace(".", "", 1).isdigit():
            stated = int(float(steps_txt[:-1]) * mult[steps_txt[-1]])
            if stated != want_steps:
                print(f"STEPS COLUMN {group}: table says {steps_cell!r}, "
                      f"benchmark budget is {want_steps:,}")
                bad += 1
        elif steps_txt not in ("", "—", "-"):
            print(f"STEPS COLUMN {group}: unparsable {steps_cell!r}")
            bad += 1
        try:
            cfg = load_config(os.path.join(root, cfg_path),
                              cli_args=["--config", os.path.join(root, cfg_path)] + overrides)
        except Exception as e:                                   # noqa: BLE001
            print(f"LOAD FAILED {group}: {type(e).__name__}: {e}")
            bad += 1
            continue
        got = effective(cfg)
        diff = [f"{k}={got[k]!r} "
                f"({'paper table has' if k in PAPER_FIELDS else 'name encodes'} {v!r})"
                for k, v in want.items() if not _same(got[k], v)]
        if variant == "gjoint":
            diff.append("joint-scope aggregation is not implemented in this release")
        if api is not None:
            runs = list(api.runs(f"{args.entity}/{args.project}", filters={"group": group}))
            n_runs += len(runs)
            if not runs:
                diff.append(f"W&B: no runs in {args.entity}/{args.project}")
            elif len(runs) != SEEDS_PER_ROW:
                print(f"NOTE {group}: {len(runs)} runs on W&B, not {SEEDS_PER_ROW}")
            wb_diff, wb_notes = compare_with_runs(cfg, runs)
            diff += [f"W&B: {d}" for d in wb_diff]
            for n in wb_notes:
                print(f"NOTE {group}: {n}")
        checked += 1
        if diff:
            bad += 1
            print(f"MISMATCH {group}\n    config: {cfg_path}"
                  f"\n    overrides: {' '.join(overrides) or '(none)'}")
            for d in diff:
                print(f"      - {d}")
    against = f" against {n_runs} W&B runs" if api is not None else ""
    print(f"\nchecked {checked} rows{against}, {skipped} skipped (SAC / not reproducible), "
          f"{bad} mismatched")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
