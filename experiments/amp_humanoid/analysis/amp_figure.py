#!/usr/bin/env python3
"""Regenerate the paper's AMP figure (Figure E.4) from the shipped evaluation records.

Ten seeds per arm at 100M samples. Pale points are the individual seeds (a fixed
seed-index horizontal offset, no randomness); filled diamonds with whiskers are the
seed mean and the sample std (ddof=1).

  (a) PPO motion style: squared Frechet distance in a fixed feature space
      (style_projected_fid), divided by 1e4.
  (b) SAC motion style, divided by 1e6.
  (c) PPO task return (task_return_mean).
  (d) SAC task return.

    python analysis/amp_figure.py [--out-dir DIR]   # writes amp_transfer.pdf and amp_transfer.png

Needs numpy and matplotlib. The paper's version uses the STIX serif font when it is
installed; matplotlib falls back to DejaVu Serif otherwise, which changes only the type.
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402

WIDTH, HEIGHT = 5.5, 3.6      # inches; the paper includes it at \linewidth
LW_ERR = 0.8
INK, GRID = "#222222", "#E6E6E6"
COLORS = {1: "#7A7A7A", 2: "#3A78C2", 3: "#5B2C83", 4: "#D62728", 5: "#4D4D4D", 6: "#C2677D"}
LABELS = {1: "Base", 2: "LN +\nPopArt", 3: "+FG-c\nclip off", 4: "Full\nstack",
          5: "Base", 6: "Full\nstack"}
JITTER = np.array([-0.09, 0.055, -0.025, 0.11, -0.065, 0.025, 0.085, -0.11, 0.005, -0.04])

STYLE = {
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": None, "savefig.pad_inches": 0.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "STIX", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.0,
    "axes.titlesize": 8.0, "axes.titleweight": "normal", "axes.titlelocation": "left",
    "axes.titlepad": 3.0, "axes.labelsize": 8.0, "axes.labelpad": 2.0, "axes.labelcolor": INK,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.6, "axes.edgecolor": "#333333",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args(argv)

    matplotlib.rcdefaults()
    matplotlib.rcParams.update(STYLE)
    rows = common.table_groups()
    data = {row.table_row: common.load_group(row.group) for row in rows}

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, HEIGHT), layout="constrained")
    panels = [[1, 2, 3, 4], [5, 6]]
    for col, table_rows in enumerate(panels):
        for r, (metric, divisor) in enumerate([("style_projected_fid", 1e4 if col == 0 else 1e6),
                                               ("task_return_mean", 1.0)]):
            ax = axes[r, col]
            for x, table_row in enumerate(table_rows):
                color = COLORS[table_row]
                values = np.array([data[table_row][seed]["metrics"][metric]
                                   for seed in sorted(data[table_row])]) / divisor
                ax.scatter(x + JITTER, values, s=7, marker="o", color=color,
                           linewidths=0, alpha=0.35, zorder=3)
                ax.errorbar(x + 0.2, values.mean(), yerr=values.std(ddof=1), fmt="D",
                            markersize=4.0, markerfacecolor=color, markeredgecolor="white",
                            markeredgewidth=0.5, ecolor=color, elinewidth=LW_ERR,
                            capsize=1.8, capthick=LW_ERR, zorder=4)
            labels = [LABELS[t] for t in table_rows]
            ax.set_xticks(np.arange(len(table_rows)) + 0.1, labels if r else [""] * len(table_rows))
            ax.set_xlim(-0.4, len(table_rows) - 0.4)
            ax.tick_params(axis="x", length=0, pad=3)
            for label in ax.get_xticklabels():
                label.set_linespacing(0.95)
            ax.grid(True, axis="y", color=GRID, lw=0.4)
            ax.set_axisbelow(True)
    for ax, title in zip(axes.flat, ["(a) PPO: motion style", "(b) SAC: motion style",
                                     "(c) PPO: task tracking", "(d) SAC: task tracking"]):
        ax.set_title(title, loc="left", color=INK)
    axes[0, 0].set_ylabel(r"Squared style FID / $10^4$")
    axes[0, 1].set_ylabel(r"Squared style FID / $10^6$")
    axes[1, 0].set_ylabel("Task return")
    axes[1, 1].set_ylabel("Task return")
    axes[0, 0].set_ylim(-0.8, 15.7)
    axes[0, 0].set_yticks([0, 5, 10, 15])
    axes[0, 1].set_ylim(2.4, 9.4)
    axes[0, 1].set_yticks([3, 5, 7, 9])
    axes[1, 0].set_ylim(491, 574)
    axes[1, 0].set_yticks([500, 520, 540, 560])
    axes[1, 1].set_ylim(565, 583)
    axes[1, 1].set_yticks([565, 570, 575, 580])
    fig.get_layout_engine().set(h_pad=3 / 72, hspace=0.06, wspace=0.06)

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(args.out_dir, "amp_transfer." + ext)
        fig.savefig(path, dpi=300)
        print("wrote " + path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
