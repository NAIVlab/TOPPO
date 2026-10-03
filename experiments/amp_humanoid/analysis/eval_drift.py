#!/usr/bin/env python3
"""Evaluation drift: the same checkpoint evaluated more than once, on different GPUs.

results/reevals/ holds 45 additional evaluations of 33 canonical checkpoints (identical
model_sha256, agent config, environment config, eval seed and episode count) that were
run on other GPUs or in a second pass. All 33 belong to the four PPO groups of the
paper's table; no SAC checkpoint and no other group was evaluated more than once. They
are not used for any reported number. This script pairs every two records of the same
checkpoint and reports the relative difference in style FID and task return, which backs
the paper's statement that repeated evaluations of the same checkpoints show at most
0.994% drift in style FID in the audited cross-device groups.

It then lists the GPU types of the canonical records of the six table groups, and
recomputes the two style-FID tests among the PPO table groups with the other records of
the 33 re-evaluated checkpoints in place of the canonical ones, to show that the choice
of record does not change them.

It also shows where the public W&B values of group ampwalk_ppo_ad_cd_npaw_lnc_k2_gc1
come from: for seeds 260511473-260511475, W&B carries a re-evaluation on an RTX 3090,
while the paper (and results/eval/) use the H200 evaluation of the same checkpoints.

    python analysis/eval_drift.py
"""

import collections
import glob
import itertools
import json
import os
import sys
from statistics import mean, median, stdev

import amp_table
import common


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    records = {}
    for path in sorted(glob.glob(os.path.join(common.EVAL_DIR, "*.json")) +
                       glob.glob(os.path.join(common.REEVAL_DIR, "*.json"))):
        d = load(path)
        records.setdefault(d["metadata"]["model_sha256"], []).append((path, d))

    pairs = []
    for model, recs in records.items():
        if len(recs) < 2:
            continue
        for (pa, a), (pb, b) in itertools.combinations(recs, 2):
            for key in ("agent_config", "env_config_sha256", "engine_config_sha256", "eval_seed",
                        "requested_episodes", "style_samples", "style_projection_dim", "style_seed"):
                assert a["metadata"][key] == b["metadata"][key], (pa, pb, key)
            fa, fb = a["metrics"]["style_projected_fid"], b["metrics"]["style_projected_fid"]
            ta, tb = a["metrics"]["task_return_mean"], b["metrics"]["task_return_mean"]
            pairs.append({
                "a": os.path.relpath(pa, common.PKG), "b": os.path.relpath(pb, common.PKG),
                "gpu_a": a["metadata"]["runtime"]["device_name"],
                "gpu_b": b["metadata"]["runtime"]["device_name"],
                "fid_drift_pct": 100 * abs(fa - fb) / min(fa, fb),
                "task_drift_pct": 100 * abs(ta - tb) / min(abs(ta), abs(tb)),
            })
    n_models = sum(len(r) > 1 for r in records.values())
    print("checkpoints evaluated more than once: {}; same-checkpoint pairs: {}".format(n_models, len(pairs)))
    worst = max(pairs, key=lambda p: p["fid_drift_pct"])
    print("style FID drift: median {:.3f}%, max {:.4f}%  ({} on {} vs {} on {})".format(
        median(p["fid_drift_pct"] for p in pairs), worst["fid_drift_pct"],
        worst["a"], worst["gpu_a"], worst["b"], worst["gpu_b"]))
    same = [p for p in pairs if p["gpu_a"] == p["gpu_b"]]
    if same:
        print("  pairs on the same GPU type: {}, max {:.4f}%".format(
            len(same), max(p["fid_drift_pct"] for p in same)))
    print("task return drift: median {:.4f}%, max {:.4f}%".format(
        median(p["task_drift_pct"] for p in pairs), max(p["task_drift_pct"] for p in pairs)))

    # ---- the public W&B values of the +K2 +LN +PopArt group (audit item: 3 seeds re-evaluated)
    group = "ampwalk_ppo_ad_cd_npaw_lnc_k2_gc1"
    canonical = common.load_group(group)
    wandb_view = dict(canonical)
    for seed in (260511473, 260511474, 260511475):
        rtx = [p for p in glob.glob(os.path.join(common.REEVAL_DIR, "{}_s{}.reeval*.json".format(group, seed)))
               if load(p)["metadata"]["runtime"]["device_name"] == "NVIDIA GeForce RTX 3090"]
        assert len(rtx) == 1, rtx
        wandb_view[seed] = load(rtx[0])
    for label, recs in (("paper / results/eval", canonical), ("public W&B", wandb_view)):
        fid = [r["metrics"]["style_projected_fid"] for r in recs.values()]
        task = [r["metrics"]["task_return_mean"] for r in recs.values()]
        print("{}: {:<22} style FID {:.1f} +/- {:.1f}, task return {:.3f} +/- {:.3f}".format(
            group, label, mean(fid), stdev(fid), mean(task), stdev(task)))

    # ---- GPU types of the canonical records behind the table
    table = {amp_table.KEYS[row.table_row]: row.group for row in common.table_groups()}
    print("\nGPU types of the canonical records of the table groups:")
    for key, grp in table.items():
        devices = collections.Counter(r["metadata"]["runtime"]["device_name"]
                                      for r in common.load_group(grp).values())
        print("  {:<13} {}".format(key, ", ".join("{} x{}".format(d, n) for d, n in sorted(devices.items()))))

    # ---- does the choice of record matter? (only the 33 re-evaluated PPO checkpoints have one)
    alternatives = {}
    for path in sorted(glob.glob(os.path.join(common.REEVAL_DIR, "*.json"))):
        grp, rest = os.path.basename(path).rsplit("_s", 1)
        alternatives.setdefault((grp, int(rest.split(".")[0])), []).append(load(path))
    assert {g for g, _ in alternatives} <= {table[k] for k in ("ppo_base", "ppo_popart",
                                                                "ppo_fairgrad", "ppo_full")}

    def fid_tests(pick):
        arms = {}
        for key in ("ppo_base", "ppo_fairgrad", "ppo_full"):
            recs = common.load_group(table[key])
            for seed in recs:
                chosen = pick(alternatives.get((table[key], seed), []))
                if chosen is not None:
                    recs[seed] = chosen
            arms[key] = {seed: {"style_projected_fid": float(r["metrics"]["style_projected_fid"])}
                         for seed, r in recs.items()}
        return (amp_table.paired(arms, "ppo_full", "ppo_base", "style_projected_fid"),
                amp_table.paired(arms, "ppo_full", "ppo_fairgrad", "style_projected_fid"))

    rtx = lambda recs: next((r for r in recs if r["metadata"]["runtime"]["device_name"]
                             == "NVIDIA GeForce RTX 3090"), None)
    print("\nrecord choice for the {} re-evaluated checkpoints:".format(len(alternatives)))
    for label, pick in (("canonical records (paper)", lambda recs: None),
                        ("RTX 3090 records substituted", rtx)):
        full, step = fid_tests(pick)
        print("  {:<29} full stack vs base {:+.2f}% (p = {:.3g}); "
              "PCGrad step d = {:+.3f} x 10^4 (p = {:.3f})".format(
                  label, full["percent_of_control_mean"], full["p_two_sided"],
                  step["delta_mean"] / 1e4, step["p_two_sided"]))

    # exact range of the relative change over every combination of records
    def fids(key):
        recs = common.load_group(table[key])
        return [[r["metrics"]["style_projected_fid"]
                 for r in [recs[s]] + alternatives.get((table[key], s), [])] for s in sorted(recs)]
    full, base = fids("ppo_full"), fids("ppo_base")
    lo = 100 * (mean(map(min, full)) / mean(map(max, base)) - 1)
    hi = 100 * (mean(map(max, full)) / mean(map(min, base)) - 1)
    print("  any combination of records:   full stack vs base between {:+.2f}% and {:+.2f}%".format(lo, hi))
    return 0


if __name__ == "__main__":
    sys.exit(main())
