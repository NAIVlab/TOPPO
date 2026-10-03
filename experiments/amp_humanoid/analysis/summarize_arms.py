"""Summarize AMP campaign groups: per-metric seed-level mean +/- sample std, and paired deltas.

Each arm contributes one evaluation JSON per training seed. Arms are compared
at the SEED level -- the unit of independence is the training run, not the
episode -- with a paired Student-t over the shared seeds (the arms use the
same seed set, so pairing removes seed-to-seed environment/init variance).
Every +/- in the arm table is the sample standard deviation across training
seeds (ddof=1), never the SEM; the paired table gives the mean difference d
and the half-width of its 95% Student-t interval.

Protocol guards:

* the evaluation protocol (env config hash, eval seed, episode count, style
  sampling) must match across every evaluation, or the run aborts;
* the runtime (host, GPU, package versions) is REPORTED and, when it differs,
  episode-level pairing is disabled and the report says so -- metric-level
  seed comparison stays valid;
* `disc_return`, `amp_return` and `disc_policy_probability` are scored by each
  run's OWN discriminator. They are printed as diagnostics and excluded from
  cross-arm significance tests; cross-arm style claims must use
  `style_projected_fid` and the raw feature moments.

Usage (from experiments/amp_humanoid/):
    python analysis/summarize_arms.py --family ppo          # the 9 PPO groups vs the PPO base
    python analysis/summarize_arms.py --family sac          # the 16 SAC groups vs the SAC base
    python analysis/summarize_arms.py \
        --arm sac_base='results/eval/ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_s*.json' \
        --arm sac_fairgrad='results/eval/ampwalk_sac_ad_cfg1_npaw_lnc_k2_gc0_s*.json' \
        --baseline sac_base [--markdown out.md] [--json out.json]
"""

import argparse
import glob
import json
import math
import os
import re
import sys

# Metrics that are comparable across arms (discriminator-independent).
COMPARABLE_METRICS = [
    "task_return_mean",
    "mean_ep_len",
    "fall_rate",
    "survival_rate",
    "style_projected_fid",
    "style_feature_mean_rms",
    "style_feature_std_rms",
    "root_height_step_mean",
    "root_upright_step_mean",
    "target_speed_error_step_mean",
    "target_velocity_error_step_mean",
    "target_projected_speed_step_mean",
]

# Reported, but NOT comparable across arms: scored by each run's own critic-side
# discriminator rather than a common rater.
OWN_RATER_METRICS = [
    "disc_return_mean",
    "amp_return_mean",
    "disc_policy_probability_step_mean",
]

PROTOCOL_FIELDS = [
    "env_config_sha256",
    "eval_seed",
    "num_envs",
    "requested_episodes",
    "style_projection_dim",
    "style_samples",
    "style_seed",
]

RUNTIME_FIELDS = ["device_name", "torch_version", "cuda_version"]

_SEED_RE = re.compile(r"s(?:eed)?_?(\d{6,})")


def _seed_from_path(path):
    match = _SEED_RE.search(os.path.basename(path))
    if (match is None):
        raise ValueError("cannot infer a seed from {!r}".format(path))
    return int(match.group(1))


def load_arm(spec):
    label, separator, pattern = spec.partition("=")
    if (separator == "" or label == "" or pattern == ""):
        raise ValueError("--arm must be LABEL=GLOB, got {!r}".format(spec))

    paths = sorted(glob.glob(pattern))
    if (len(paths) == 0):
        raise ValueError("no evaluations matched {!r}".format(pattern))

    runs = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        seed = _seed_from_path(path)
        if (seed in runs):
            raise ValueError("duplicate seed {} for arm {}".format(seed, label))
        runs[seed] = {"path": path,
                      "metrics": payload["metrics"],
                      "metadata": payload.get("metadata", {})}
    return label, runs


def check_protocol(arms):
    """Abort on protocol mismatch; return the set of runtimes that were seen."""
    reference = None
    runtimes = set()
    for label, runs in arms.items():
        for seed, run in runs.items():
            metadata = run["metadata"]
            protocol = tuple(str(metadata.get(f)) for f in PROTOCOL_FIELDS)
            if (reference is None):
                reference = (label, seed, protocol)
            elif (protocol != reference[2]):
                raise SystemExit(
                    "protocol mismatch: {}:{} vs {}:{}\n  {}\n  {}".format(
                        reference[0], reference[1], label, seed,
                        dict(zip(PROTOCOL_FIELDS, reference[2])),
                        dict(zip(PROTOCOL_FIELDS, protocol))))
            runtime = metadata.get("runtime", {})
            runtimes.add(tuple(str(runtime.get(f)) for f in RUNTIME_FIELDS))
    return runtimes


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    if (n < 2):
        return mean, 0.0, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    return mean, std, std / math.sqrt(n)


# two-sided Student-t critical values for small samples (alpha = 0.05)
# two-sided Student-t critical values, alpha = 0.05, df 1..30
_T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
           7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
           13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
           19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
           25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def t_critical(df):
    """Exact where tabulated; otherwise CONSERVATIVE (the value for the next
    SMALLER df, which is larger). The previous version rounded to the next
    LARGER df and fell back to 1.96, making the test anti-conservative -- e.g.
    df=11 returned 2.131 instead of 2.201 and df>=21 returned 1.96."""
    if (df <= 0):
        return float("nan")
    try:
        from scipy.stats import t as _student_t
        return float(_student_t.ppf(0.975, df))
    except Exception:
        pass
    if (df in _T_CRIT):
        return _T_CRIT[df]
    smaller = [k for k in _T_CRIT if k < df]
    if (len(smaller) == 0):
        return _T_CRIT[min(_T_CRIT)]
    return _T_CRIT[max(smaller)]


# metrics where a DECREASE is an improvement; everything else improves upward.
# root_height and target_projected_speed are deliberately absent: the first is
# non-directional, the second is "closer to commanded" rather than higher/lower.
LOWER_IS_BETTER = {
    "style_projected_fid",
    "style_feature_mean_rms",
    "style_feature_std_rms",
    "fall_rate",
    "target_speed_error_step_mean",
    "target_velocity_error_step_mean",
}


def paired_delta(baseline_values, candidate_values, metric=None, seeds=None):
    """Paired-by-seed mean difference with a Student-t 95% interval.

    `improved` counts seeds that moved in the BENEFICIAL direction for this
    metric. Counting raw increases (the previous behavior) silently reported
    "1/3 better" for a style-FID reduction, because lower FID is better.
    """
    deltas = [c - b for b, c in zip(baseline_values, candidate_values)]
    mean, std, sem = mean_std(deltas)
    n = len(deltas)
    ci = t_critical(n - 1) * sem if n > 1 else float("nan")
    significant = (n > 1) and (sem > 0) and (abs(mean) > ci)
    lower_better = metric in LOWER_IS_BETTER
    improved = sum(1 for d in deltas if (d < 0) == lower_better and d != 0)
    return {"mean": mean, "std": std, "sem": sem, "ci95": ci,
            "significant": bool(significant), "n": n,
            "improved": improved, "lower_is_better": bool(lower_better),
            "seeds": list(seeds) if seeds is not None else None,
            "deltas": deltas}


def summarize(arms, baseline_label):
    # Diagnostic only. Pairing must be done PER COMPARISON: intersecting the
    # seed sets of every arm silently downgrades every comparison to the seed
    # count of the smallest arm (adding one n=1 arm collapsed the whole table
    # to n=1), which is how a table labeled n=3 came to be computed over 2.
    seed_sets = {label: set(runs) for label, runs in arms.items()}
    shared = sorted(set.intersection(*seed_sets.values())) if seed_sets else []

    report = {"arms": {}, "shared_seeds": shared, "comparisons": {},
              "comparison_seeds": {}}

    for label, runs in arms.items():
        entry = {"seeds": sorted(runs), "metrics": {}}
        for metric in COMPARABLE_METRICS + OWN_RATER_METRICS:
            values = [runs[s]["metrics"][metric] for s in sorted(runs)
                      if metric in runs[s]["metrics"]]
            if (len(values) == 0):
                continue
            mean, std, sem = mean_std(values)
            entry["metrics"][metric] = {
                "mean": mean, "std": std, "sem": sem, "n": len(values),
                "values": values,
                "own_rater": metric in OWN_RATER_METRICS,
            }
        report["arms"][label] = entry

    if (baseline_label is not None):
        base = arms[baseline_label]
        max_n = max((len(set(base) & set(r)) for l, r in arms.items()
                     if l != baseline_label), default=0)
        for label, runs in arms.items():
            if (label == baseline_label):
                continue
            pair_seeds = sorted(set(base) & set(runs))
            report["comparison_seeds"][label] = pair_seeds
            if (len(pair_seeds) == 0):
                report["comparisons"][label] = {}
                continue
            if (len(pair_seeds) < max_n):
                sys.stderr.write(
                    "warning: arm {!r} pairs over only {} seed(s) against the "
                    "baseline while others pair over {} -- its interval is not "
                    "comparable to theirs\n".format(label, len(pair_seeds), max_n))
            deltas = {}
            for metric in COMPARABLE_METRICS:
                if any(metric not in base[s]["metrics"] or metric not in runs[s]["metrics"]
                       for s in pair_seeds):
                    continue
                deltas[metric] = paired_delta(
                    [base[s]["metrics"][metric] for s in pair_seeds],
                    [runs[s]["metrics"][metric] for s in pair_seeds],
                    metric=metric, seeds=pair_seeds)
            report["comparisons"][label] = deltas
    return report


def render_markdown(report, baseline_label, runtimes):
    lines = []
    labels = list(report["arms"])
    lines.append("## Arm summary (mean +/- sample std over training seeds)\n")
    header = "| metric | " + " | ".join(labels) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(labels) + 1))

    for metric in COMPARABLE_METRICS + OWN_RATER_METRICS:
        cells = []
        present = False
        for label in labels:
            entry = report["arms"][label]["metrics"].get(metric)
            if (entry is None):
                cells.append("-")
                continue
            present = True
            cells.append("{:.4g} ± {:.4g} (n={})".format(
                entry["mean"], entry["std"], entry["n"]))
        if (not present):
            continue
        name = metric + (" †" if metric in OWN_RATER_METRICS else "")
        lines.append("| {} | {} |".format(name, " | ".join(cells)))

    lines.append("")
    lines.append("† scored by each run's OWN discriminator -- a training "
                 "diagnostic, not a common-rater score; excluded from the "
                 "significance tests below.")
    lines.append("")

    if (report["comparisons"]):
        lines.append("## Paired-by-seed deltas vs `{}`\n".format(baseline_label))
        lines.append("Each comparison is paired over the seeds THAT ARM shares "
                     "with the baseline, listed per sub-section; intervals from "
                     "different n are not comparable to each other.\n")
        for label, deltas in report["comparisons"].items():
            seeds = report["comparison_seeds"].get(label, [])
            lines.append("### {} (n={}, seeds {})\n".format(
                label, len(seeds), ", ".join(str(s) for s in seeds) or "none"))
            if (not deltas):
                lines.append("_no seeds shared with the baseline_\n")
                continue
            lines.append("| metric | delta | 95% CI | seeds improved | significant |")
            lines.append("|---|---|---|---|---|")
            for metric, d in deltas.items():
                arrow = " (lower better)" if d["lower_is_better"] else ""
                lines.append("| {}{} | {:+.4g} | ±{:.3g} | {}/{} | {} |".format(
                    metric, arrow, d["mean"], d["ci95"], d["improved"], d["n"],
                    "yes" if d["significant"] else "no"))
            lines.append("")

    lines.append("## Runtime\n")
    if (len(runtimes) == 1):
        lines.append("All evaluations share one runtime: {}.".format(
            ", ".join(v for v in sorted(runtimes)[0])))
    else:
        lines.append("**Evaluations span {} runtimes**, so episode-level paired "
                     "tests are not licensed; the seed-level comparison above "
                     "is unaffected:".format(len(runtimes)))
        for runtime in sorted(runtimes):
            lines.append("- " + ", ".join(runtime))
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", default=[],
                        help="LABEL=GLOB, repeatable")
    parser.add_argument("--family", choices=["ppo", "sac"],
                        help="all campaign groups of one family, labeled by arm; "
                             "the family's base group is the default baseline")
    parser.add_argument("--baseline", default=None,
                        help="arm label to compare the others against")
    parser.add_argument("--markdown", default="")
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    if args.family:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import campaign
        eval_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "results", "eval")
        rows = [r for r in campaign.CAMPAIGN if r.family == args.family]
        args.arm = ["{}={}".format(r.arm, os.path.join(eval_dir, r.group + "_s*.json"))
                    for r in rows] + args.arm
        if args.baseline is None:
            args.baseline = rows[0].arm
    if not args.arm:
        parser.error("give --family and/or at least one --arm LABEL=GLOB")

    arms = {}
    for spec in args.arm:
        label, runs = load_arm(spec)
        arms[label] = runs

    if (args.baseline is not None and args.baseline not in arms):
        raise SystemExit("baseline {!r} is not one of {}".format(
            args.baseline, list(arms)))

    runtimes = check_protocol(arms)
    report = summarize(arms, args.baseline)
    markdown = render_markdown(report, args.baseline, runtimes)

    print(markdown)
    if (args.markdown):
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(markdown)
    if (args.json):
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
