#!/usr/bin/env python3
"""Regenerate the paper's AMP table (Table E.6) and the paired tests quoted with it.

Reads the 60 evaluation records of the six table groups from results/eval/ and prints
  * mean +/- sample std (ddof=1) over the 10 training seeds of task return and style FID;
  * five paired two-sided Student-t tests over the shared seeds (df = 9): PPO full stack vs
    base (style FID; the "76%" headline), PPO full stack vs the FairGrad rung (style FID),
    SAC full stack vs base (style FID), and PPO full stack vs base on projected speed and
    on speed error. Each gives the paired mean difference d with its 95% CI, the p-value,
    the Bonferroni(25) p-value, and d and its CI as a percentage of the control mean.

Uncertainty is always the sample std across training seeds, never the SEM, and never the
per-episode statistics stored inside each record (those *_sem / *_ci95 fields describe the
4,096 evaluation episodes of one checkpoint and are not used here).

    python analysis/amp_table.py                 # human-readable
    python analysis/amp_table.py --latex         # the six LaTeX table rows
    python analysis/amp_table.py --check         # exit 1 unless every printed number matches the paper
    python analysis/amp_table.py --json out.json # all unrounded values

Needs scipy (for the t distribution); everything else is the standard library.
"""

import argparse
import json
import math
import sys
from statistics import mean, stdev

from scipy.stats import t, ttest_rel

import common

METRICS = ("task_return_mean", "style_projected_fid",
           "target_projected_speed_step_mean", "target_speed_error_step_mean")
KEYS = {1: "ppo_base", 2: "ppo_popart", 3: "ppo_fairgrad", 4: "ppo_full",
        5: "sac_base", 6: "sac_full"}
LATEX_LABEL = {
    1: ("PPO", "Base"),
    2: ("PPO", "Two-component critic $+$ LN $+$ PopArt"),
    3: ("PPO", "Above $+$ FG-c, no critic grad clip"),
    4: ("PPO", "Full stack (also PCGrad-a)"),
    5: ("SAC", "Base"),
    6: ("SAC", "Full stack"),
}
N_COMPARISONS = 25   # Bonferroni family: the 25-configuration campaign

# What the camera-ready prints (Appendix E.5: Table E.6 and the text below it).
PAPER = {
    "rows": {
        "ppo_base": ("533.56", "20.74", "5.477", "1.327"),
        "ppo_popart": ("534.99", "16.89", "3.855", "0.871"),
        "ppo_fairgrad": ("550.63", "9.01", "3.635", "3.943"),
        "ppo_full": ("521.11", "14.64", "1.315", "0.088"),
        "sac_base": ("573.77", "4.03", "5.596", "1.522"),
        "sac_full": ("575.33", "2.57", "4.486", "1.108"),
    },
    "ppo_full_vs_base_fid": {"pct": "-76.0", "p": "4.7e-06", "ci_pct": ("-93.8", "-58.2"),
                             "seeds_lower": 10, "bonferroni_significant": True},
    "ppo_full_vs_fairgrad_fid": {"p": "0.098", "d_1e4": "-2.32", "ci_1e4": ("-5.16", "0.52")},
    "ppo_full_vs_base_speed": {"d": "-0.416", "ci": ("-0.529", "-0.303")},
    "ppo_full_vs_base_speed_error": {"ci": ("-0.119", "0.023")},
    "sac_full_vs_base_fid": {"pct": "-19.8", "p": "0.078", "ci_pct": ("-42.4", "2.8")},
}


def paired(arms, treatment, control, metric):
    seeds = sorted(arms[control])
    a = [arms[treatment][s][metric] for s in seeds]
    b = [arms[control][s][metric] for s in seeds]
    diffs = [x - y for x, y in zip(a, b)]
    n = len(seeds)
    d = mean(diffs)
    half = float(t.ppf(0.975, n - 1)) * stdev(diffs) / math.sqrt(n)
    test = ttest_rel(a, b)
    base = mean(b)
    return {
        "treatment": treatment, "control": control, "metric": metric, "n": n,
        "delta_mean": d, "delta_sample_std": stdev(diffs), "ci95": [d - half, d + half],
        "t": float(test.statistic), "p_two_sided": float(test.pvalue),
        "bonferroni_25_p": min(1.0, N_COMPARISONS * float(test.pvalue)),
        # percentages scale d and its CI by the observed control mean; they are not a CI
        # for the mean of per-seed ratios
        "percent_of_control_mean": 100 * d / base,
        "ci95_percent_of_control_mean": [100 * (d - half) / base, 100 * (d + half) / base],
        "seeds_treatment_lower": sum(x < y for x, y in zip(a, b)),
    }


def compute():
    arms = {}
    groups = {}
    for row in common.table_groups():
        key = KEYS[row.table_row]
        groups[key] = row.group
        records = common.load_group(row.group)
        arms[key] = {seed: {m: float(rec["metrics"][m]) for m in METRICS}
                     for seed, rec in records.items()}
    summary = {
        "uncertainty": "sample std across training seeds (ddof=1), not SEM; "
                       "paired two-sided Student t over the shared seeds, df=9",
        "groups": groups,
        "arms": {key: {m: {"mean": mean([v[m] for v in arms[key].values()]),
                           "sample_std": stdev([v[m] for v in arms[key].values()])}
                       for m in METRICS}
                 for key in arms},
        "paired_tests": {
            "ppo_full_vs_base_fid": paired(arms, "ppo_full", "ppo_base", "style_projected_fid"),
            "ppo_full_vs_fairgrad_fid": paired(arms, "ppo_full", "ppo_fairgrad", "style_projected_fid"),
            "sac_full_vs_base_fid": paired(arms, "sac_full", "sac_base", "style_projected_fid"),
            "ppo_full_vs_base_speed": paired(arms, "ppo_full", "ppo_base",
                                             "target_projected_speed_step_mean"),
            "ppo_full_vs_base_speed_error": paired(arms, "ppo_full", "ppo_base",
                                                   "target_speed_error_step_mean"),
        },
    }
    return summary


def formatted(summary):
    """The numbers exactly as the paper prints them."""
    out = {"rows": {}}
    for key, arm in summary["arms"].items():
        scale = 1e4 if key.startswith("ppo") else 1e6
        task, fid = arm["task_return_mean"], arm["style_projected_fid"]
        out["rows"][key] = ("%.2f" % task["mean"], "%.2f" % task["sample_std"],
                            "%.3f" % (fid["mean"] / scale), "%.3f" % (fid["sample_std"] / scale))
    tests = summary["paired_tests"]
    r = tests["ppo_full_vs_base_fid"]
    out["ppo_full_vs_base_fid"] = {
        "pct": "%.1f" % r["percent_of_control_mean"], "p": "%.1e" % r["p_two_sided"],
        "ci_pct": tuple("%.1f" % v for v in r["ci95_percent_of_control_mean"]),
        "seeds_lower": r["seeds_treatment_lower"],
        "bonferroni_significant": r["bonferroni_25_p"] < 0.05}
    r = tests["ppo_full_vs_fairgrad_fid"]
    out["ppo_full_vs_fairgrad_fid"] = {"p": "%.3f" % r["p_two_sided"],
                                       "d_1e4": "%.2f" % (r["delta_mean"] / 1e4),
                                       "ci_1e4": tuple("%.2f" % (v / 1e4) for v in r["ci95"])}
    r = tests["ppo_full_vs_base_speed"]
    out["ppo_full_vs_base_speed"] = {"d": "%.3f" % r["delta_mean"],
                                     "ci": tuple("%.3f" % v for v in r["ci95"])}
    r = tests["ppo_full_vs_base_speed_error"]
    out["ppo_full_vs_base_speed_error"] = {"ci": tuple("%.3f" % v for v in r["ci95"])}
    r = tests["sac_full_vs_base_fid"]
    out["sac_full_vs_base_fid"] = {
        "pct": "%.1f" % r["percent_of_control_mean"], "p": "%.3f" % r["p_two_sided"],
        "ci_pct": tuple("%.1f" % v for v in r["ci95_percent_of_control_mean"])}
    return out


def latex_rows(summary):
    lines = []
    for row in common.table_groups():
        key = KEYS[row.table_row]
        family, label = LATEX_LABEL[row.table_row]
        task_m, task_s, fid_m, fid_s = formatted(summary)["rows"][key]
        exp = 4 if family == "PPO" else 6
        lines.append("{} & {} & ${}\\pm{}$ & $({}\\pm{})\\times10^{}$ \\\\".format(
            family, label, task_m, task_s, fid_m, fid_s, exp))
        if row.table_row == 4:
            lines.append("\\midrule")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latex", action="store_true", help="print the six LaTeX rows")
    parser.add_argument("--check", action="store_true",
                        help="compare every printed number with the camera-ready")
    parser.add_argument("--json", help="write all unrounded values to this file")
    args = parser.parse_args(argv)

    summary = compute()
    shown = formatted(summary)
    if args.latex:
        print(latex_rows(summary))
    else:
        print("group | n | task return (mean +/- sample std) | style FID (mean +/- sample std)")
        for key, arm in summary["arms"].items():
            task, fid = arm["task_return_mean"], arm["style_projected_fid"]
            print("{:<40} | 10 | {:.2f} +/- {:.2f} | {:.6g} +/- {:.6g}".format(
                summary["groups"][key], task["mean"], task["sample_std"],
                fid["mean"], fid["sample_std"]))
        print()
        for name, r in summary["paired_tests"].items():
            print("{}: d = {:.6g}, 95% CI [{:.6g}, {:.6g}], p = {:.3g}, Bonferroni(25) p = {:.3g}, "
                  "d = {:.1f}% of control mean (CI [{:.1f}%, {:.1f}%]), treatment lower on {}/{} seeds"
                  .format(name, r["delta_mean"], r["ci95"][0], r["ci95"][1], r["p_two_sided"],
                          r["bonferroni_25_p"], r["percent_of_control_mean"],
                          r["ci95_percent_of_control_mean"][0], r["ci95_percent_of_control_mean"][1],
                          r["seeds_treatment_lower"], r["n"]))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
    if args.check:
        bad = []
        for section, expected in PAPER.items():
            got = shown[section]
            if section == "rows":
                for key, value in expected.items():
                    if tuple(got[key]) != value:
                        bad.append("{} {}: paper {} vs {}".format(section, key, value, got[key]))
            else:
                for field, value in expected.items():
                    if got[field] != value:
                        bad.append("{} {}: paper {} vs {}".format(section, field, value, got[field]))
        if bad:
            print("\nMISMATCH with the camera-ready:\n  " + "\n  ".join(bad))
            return 1
        print("\nall table entries and quoted test statistics match the camera-ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
