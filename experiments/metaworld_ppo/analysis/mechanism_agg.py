"""Aggregate per-task normalizer scales across seeds.

Single-seed spreads are unstable: max/min of the per-task divisor is set by
whichever task happened not to be learned in that seed, and swings by orders
of magnitude between seeds (measured 9.8x, 5.3e3x and 2.0e4x on three seeds
of the same group). What is stable is the per-task scale itself, so we take
the median across seeds for each task and report percentile-based spread
over tasks — figures that can be quoted in the paper without being an
artifact of one run.

Only finished seeds count: a checkpoint is used once its epoch reaches the
`training.epoch` of the config it was trained with, so the same command works
for any suite or budget.

    python analysis/mechanism_agg.py --logdir <dir> --group <group>
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch


def per_task_scale(ckpt_path: str) -> tuple[np.ndarray, list[str], str] | None:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config") if isinstance(ck.get("config"), dict) else {}
    final_epoch = int((cfg.get("training") or {}).get("epoch", 100))
    if int(ck.get("epoch", 0)) < final_epoch:
        return None
    extra = ck.get("algo_extra_state") or {}
    tasks: list[str] = []
    for key in ("_per_task_train_rewards", "_per_task_update_buffer"):
        if isinstance(extra.get(key), dict):
            tasks = list(extra[key].keys())
            break

    sd = ck.get("algorithm_state_dict", {})
    buf = {k.split(".")[-1]: v.double() for k, v in sd.items()
           if k.startswith("retscale_module.")}
    if buf:
        n = buf["n"].clamp(min=1)
        var_r = (buf["s_r2"] / n - (buf["s_r"] / n) ** 2).clamp(min=0)
        var_g = (buf["s_g2"] / n - (buf["s_g"] / n) ** 2).clamp(min=0)
        sigma = (var_r + var_g * (buf["s_G2"] / n)).sqrt().numpy()
        return sigma, tasks, "return-scaling sigma_i"

    states = extra.get("_reward_norm_env_states")
    if states:
        std = np.array([np.sqrt(s.get("var", 1.0) + 1e-8)
                        for s in states if isinstance(s, dict)])
        return std, tasks, "reward-norm return-std divisor"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--group", required=True)
    args = ap.parse_args()

    mats, tasks, label = [], [], ""
    for path in sorted(glob.glob(os.path.join(
            args.logdir, args.group + "_s*", "checkpoints", "latest.pth"))):
        got = per_task_scale(path)
        if got is None:
            continue
        vals, tks, label = got
        mats.append(vals)
        tasks = tks or tasks

    if not mats:
        raise SystemExit(f"no finished checkpoints for {args.group}")

    m = np.vstack(mats)                       # seeds x tasks
    med = np.median(m, axis=0)                # per task, median across seeds
    p5, p50, p95 = np.percentile(med, [5, 50, 95])
    print(f"{args.group}: {label}, {m.shape[0]} finished seeds x {m.shape[1]} tasks")
    print(f"  per-task median-across-seeds: min={med.min():.4g} p5={p5:.4g} "
          f"median={p50:.4g} p95={p95:.4g} max={med.max():.4g}")
    print(f"  robust spread p95/p5 = {p95 / p5:.4g}x   (max/min = {med.max() / med.min():.4g}x, "
          f"unstable — set by whichever task went unlearned)")
    if tasks and len(tasks) == len(med):
        order = np.argsort(-med)
        print("  largest:  " + ", ".join(f"{tasks[i]}={med[i]:.3g}" for i in order[:3]))
        print("  smallest: " + ", ".join(f"{tasks[i]}={med[i]:.3g}" for i in order[-3:]))
    # Per-seed spread, to show how unstable the single-seed figure is.
    ratios = [f"{r.max() / r.min():.3g}x" for r in m[:4]]
    print(f"  single-seed max/min for the first seeds: {', '.join(ratios)}")


if __name__ == "__main__":
    main()
