"""Shared loaders for the analysis scripts. Standard library only."""

import json
import os
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

import campaign  # noqa: E402

EVAL_DIR = os.path.join(PKG, "results", "eval")
REEVAL_DIR = os.path.join(PKG, "results", "reevals")


def eval_path(group, seed, eval_dir=EVAL_DIR):
    return os.path.join(eval_dir, "{}_s{}.json".format(group, seed))


def load_group(group, eval_dir=EVAL_DIR, seeds=campaign.SEEDS):
    """{seed: evaluation record} for one W&B group; every campaign seed must exist."""
    out = {}
    for seed in seeds:
        path = eval_path(group, seed, eval_dir)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            out[seed] = json.load(f)
    return out


def table_groups():
    """The six groups of the paper's AMP table, in table order."""
    rows = sorted((r for r in campaign.CAMPAIGN if r.table_row), key=lambda r: r.table_row)
    assert [r.table_row for r in rows] == [1, 2, 3, 4, 5, 6]
    return rows
