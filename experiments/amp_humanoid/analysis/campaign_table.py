#!/usr/bin/env python3
"""All 25 campaign groups: task return and style FID, mean +/- sample std over 10 seeds.

The paper's table reports six of these groups (column `row`); this prints all of them,
in the order of campaign.py. For paired comparisons against a family's base group, use
analysis/summarize_arms.py --family ppo|sac.

    python analysis/campaign_table.py            # markdown table on stdout
"""

import sys
from statistics import mean, stdev

import common
from common import campaign


def main():
    print("| W&B group | arm | paper row | steps | task return | style FID |")
    print("|---|---|:---:|---:|---:|---:|")
    for row in campaign.CAMPAIGN:
        recs = common.load_group(row.group)
        task = [r["metrics"]["task_return_mean"] for r in recs.values()]
        fid = [r["metrics"]["style_projected_fid"] for r in recs.values()]
        print("| `{}` | {} | {} | {}M | {:.2f} ± {:.2f} | {:.4g} ± {:.4g} |".format(
            row.group, row.arm, row.table_row or "", row.budget // 1_000_000,
            mean(task), stdev(task), mean(fid), stdev(fid)))
    print("\nMean ± sample std (ddof=1) over the 10 training seeds; style FID is a squared "
          "distance, lower is better. PPO and SAC FIDs differ by two orders of magnitude and "
          "are not compared across families.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
