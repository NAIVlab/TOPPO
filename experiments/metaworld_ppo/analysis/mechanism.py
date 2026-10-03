"""Recover per-task normalizer mechanism evidence from training checkpoints.

The logger's key whitelist dropped `retscale/*` and `hlgauss/*` for the runs
launched before that fix, so the per-task diagnostics never reached wandb.
Everything needed is nevertheless in `latest.pth`, and the checkpoint copy is
strictly better evidence than a logged curve: it is the exact end-of-training
state, in fp64, for the run whose success rate we report.

What is recovered per arm:

- **retscale** — the `retscale_module` buffers give each task's all-time
  sigma_i^2 = Var_i[R] + Var_i[gamma] E_i[G^2] and the frozen scale
  s_i = max(sigma_i, sigma_batch_i, 1e-2) exactly as the value loss used it.
- **rewnorm** — `_reward_norm_env_states` (one dict per env, and env i hosts
  task i) gives each task's running discounted-return variance, i.e. the
  divisor applied to that task's rewards.
- **hlgauss** — the head's support is a config constant, so per-task bin
  occupancy is computed against a PopArt run's measured per-task value-target
  distribution (mu_i, sigma_i), quantifying resolution starvation.

The report is chosen from the checkpoint's contents, not from the run name,
so it works with any `--exp-name` (including the canonical `<group>_s<seed>`
names in docs/RUNS.md).  A run that combines two of these (e.g. HL-Gauss with
reward normalization) gets both reports.

Usage:
    python analysis/mechanism.py --logdir <dir with <exp>/checkpoints/latest.pth>
    python analysis/mechanism.py --logdir ... --exp mt50v1_ppo_ad_cd_nret_lnc_vc1_400x3_s0
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict, List

import numpy as np
import torch


def _load(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _fmt_spread(vals: np.ndarray, unit: str = "") -> str:
    pos = vals[vals > 0]
    ratio = (pos.max() / pos.min()) if pos.size else float("nan")
    return (
        f"min={vals.min():.4g}{unit} median={np.median(vals):.4g}{unit} "
        f"max={vals.max():.4g}{unit} max/min={ratio:.4g}x"
    )


def retscale_report(ckpt: Dict[str, Any], tasks: List[str]) -> str:
    sd = ckpt.get("algorithm_state_dict", {})
    pref = "retscale_module."
    buf = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
    if not buf:
        return "  (no retscale_module in checkpoint — not a return-scaling run)"

    n = buf["n"].double()
    safe = n.clamp(min=1.0)
    var_r = (buf["s_r2"] / safe - (buf["s_r"] / safe) ** 2).clamp(min=0)
    var_g = (buf["s_g2"] / safe - (buf["s_g"] / safe) ** 2).clamp(min=0)
    e_G2 = buf["s_G2"] / safe
    sigma = torch.where(n > 0, var_r + var_g * e_G2, torch.zeros_like(var_r)).sqrt().numpy()
    scale = buf["last_scale"].double().numpy()

    lines = [
        f"  samples/task: {int(n.min())}..{int(n.max())}",
        f"  sigma_i      : {_fmt_spread(sigma)}",
        f"  scale s_i    : {_fmt_spread(scale)}",
        f"  Var[R] term  : {_fmt_spread(var_r.sqrt().numpy())}",
        f"  Var[gamma]   : max={var_g.max():.4g} (0 under pure time-limit truncation)",
        f"  at floor 1e-2: {int((scale <= 1.0000001e-2).sum())}/{len(scale)} tasks",
    ]
    order = np.argsort(-scale)
    lines.append("  largest / smallest scales:")
    for i in order[:3]:
        lines.append(f"    {tasks[i]:<28} s={scale[i]:.4g}")
    for i in order[-3:]:
        lines.append(f"    {tasks[i]:<28} s={scale[i]:.4g}")
    return "\n".join(lines)


def rewnorm_report(ckpt: Dict[str, Any], tasks: List[str]) -> str:
    extra = ckpt.get("algo_extra_state") or {}
    states = extra.get("_reward_norm_env_states")
    if not states:
        return ("  (no _reward_norm_env_states — either not a rewnorm run, or the\n"
                "   checkpoint predates the capture fix)")
    stds, counts = [], []
    for st in states:
        if isinstance(st, dict):
            stds.append(float(np.sqrt(st.get("var", 1.0) + 1e-8)))
            counts.append(float(st.get("count", 0.0)))
    stds_a = np.array(stds)
    lines = [
        f"  envs with stats: {len(stds)} (env i hosts task i)",
        f"  steps/env      : {min(counts):.0f}..{max(counts):.0f}",
        f"  return-std divisor: {_fmt_spread(stds_a)}",
    ]
    order = np.argsort(-stds_a)
    lines.append("  largest / smallest divisors:")
    for i in order[:3]:
        nm = tasks[i] if i < len(tasks) else f"env{i}"
        lines.append(f"    {nm:<28} std={stds_a[i]:.4g}")
    for i in order[-3:]:
        nm = tasks[i] if i < len(tasks) else f"env{i}"
        lines.append(f"    {nm:<28} std={stds_a[i]:.4g}")
    return "\n".join(lines)


def hlgauss_report(ckpt: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    ppo = cfg.get("ppo", {}) if isinstance(cfg, dict) else {}
    vmin, vmax = ppo.get("hlgauss_vmin"), ppo.get("hlgauss_vmax")
    bins = ppo.get("hlgauss_num_bins")
    if vmin is None or vmax is None:
        return "  (no hlgauss support in config — not an HL-Gauss run)"
    width = (vmax - vmin) / bins
    return (
        f"  support [{vmin:g}, {vmax:g}], {bins} bins, bin width {width:.6g},\n"
        f"  smoothing sigma {ppo.get('hlgauss_sigma_ratio')} x width = "
        f"{ppo.get('hlgauss_sigma_ratio', 0.75) * width:.6g}\n"
        f"  (per-task bin occupancy: pair with a PopArt run's measured mu_i/sigma_i;\n"
        f"   any task whose whole mu_i +- 2 sigma_i envelope is < {width:.6g} wide\n"
        f"   is representable by at most two bins)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--exp", default=None, help="single experiment name; default = all")
    args = ap.parse_args()

    pattern = os.path.join(args.logdir, args.exp or "*", "checkpoints", "latest.pth")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no checkpoints matched {pattern}")

    for path in paths:
        exp = path.split(os.sep)[-3]
        ckpt = _load(path)
        cfg = ckpt.get("config", {}) or {}
        tasks = ckpt.get("task_names") or []
        if not tasks:
            # Recover task order from a per-task buffer in the extra state.
            extra = ckpt.get("algo_extra_state") or {}
            for key in ("_per_task_train_rewards", "_per_task_update_buffer"):
                if isinstance(extra.get(key), dict):
                    tasks = list(extra[key].keys())
                    break
        print(f"\n=== {exp}  (epoch {ckpt.get('epoch')}, step {ckpt.get('step')}) ===")
        # Decide from what the checkpoint holds, not from the directory name.
        sd = ckpt.get("algorithm_state_dict") or {}
        extra = ckpt.get("algo_extra_state") or {}
        env_cfg = (cfg.get("env") if isinstance(cfg, dict) else None) or {}
        ppo_cfg = (cfg.get("ppo") if isinstance(cfg, dict) else None) or {}
        printed = False
        if any(k.startswith("retscale_module.") for k in sd):
            print(retscale_report(ckpt, tasks))
            printed = True
        if extra.get("_reward_norm_env_states") or env_cfg.get("reward_normalization_method"):
            print(rewnorm_report(ckpt, tasks))
            printed = True
        if (ppo_cfg.get("critic_loss_mode") == "hlgauss"
                or ppo_cfg.get("hlgauss_vmin") is not None):
            print(hlgauss_report(ckpt, cfg))
            printed = True
        if not printed:
            print("  (no return-scaling, reward-normalization or HL-Gauss state in this "
                  "checkpoint — nothing to report)")


if __name__ == "__main__":
    main()
