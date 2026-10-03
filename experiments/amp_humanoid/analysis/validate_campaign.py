#!/usr/bin/env python3
"""Integrity check of the 250 shipped AMP evaluation records against the campaign table.

Everything in the AMP table rests on three claims, and each is checked against the
artifacts rather than assumed:

  1. COVERAGE    every one of the 25 groups has exactly the 10 campaign seeds in
                 results/eval/, and nothing else is there.
  2. PROTOCOL    every evaluation used the same env config, engine config, eval seed,
                 episode count and style-metric settings, so seed-level numbers are
                 comparable across groups. (The evaluation GPU differs between records;
                 it is reported, and it only rules out pairing individual episodes.)
  3. PROVENANCE  every record names the agent config of its group; the agent and eval
                 env config hashes it recorded are the recorded (or, for a re-run, the
                 released) hashes in provenance/config_hashes.tsv; every config shipped
                 in overlay/ has the same functional hash (comments ignored) as the one
                 that ran; every evaluated checkpoint is distinct and belongs to its
                 seed; and each training run reached its budget according to
                 results/train_samples.tsv (taken from the public W&B runs).

Every agent and env config ships with a change notice added at its top (Apache-2.0,
section 4(b)), and fifteen agent configs also with a rewritten comment block, so their
byte hashes differ from the recorded ones; their functional hashes do not. Runs without a
training history on W&B (results/train_samples.tsv: NA) are reported as warnings, not
failures: their budget cannot be confirmed from public data.

    python analysis/validate_campaign.py            # exit 1 on any failure
    python analysis/validate_campaign.py --eval-dir <assembled tree>/results/eval

Standard library plus PyYAML.
"""

import argparse
import collections
import csv
import hashlib
import json
import os
import sys

import yaml

import common
from common import PKG, campaign

PROTOCOL_FIELDS = ("env_config", "env_config_sha256", "engine_config", "engine_config_sha256",
                   "eval_seed", "num_envs", "requested_episodes", "detailed_eval",
                   "style_samples", "style_projection_dim", "style_seed")
EXPECTED_PROTOCOL = {
    "env_config": campaign.EVAL_ENV_CONFIG,
    "engine_config": campaign.ENGINE_CONFIG,
    "eval_seed": campaign.EVAL_SEED,
    "num_envs": campaign.NUM_ENVS,
    "requested_episodes": campaign.TEST_EPISODES,
    "detailed_eval": True,
    "style_samples": campaign.STYLE_SAMPLES,
    "style_projection_dim": campaign.STYLE_PROJECTION_DIM,
    "style_seed": campaign.STYLE_SEED,
}


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def functional_sha256(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_tsv(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", default=common.EVAL_DIR)
    parser.add_argument("--configs", default=os.path.join(PKG, "overlay"),
                        help="directory holding data/agents and data/envs (default: overlay/)")
    args = parser.parse_args(argv)

    failures, warnings = [], []
    hashes = {row["file"]: row for row in read_tsv(os.path.join(PKG, "provenance", "config_hashes.tsv"))}
    samples = {(row["wandb_group"], int(row["seed"])): row
               for row in read_tsv(os.path.join(PKG, "results", "train_samples.tsv"))}

    # ---- configs: shipped file vs recorded hash ------------------------------------
    for rel, row in sorted(hashes.items()):
        path = os.path.join(args.configs, rel)
        if not os.path.isfile(path):
            if rel == campaign.ENGINE_CONFIG:
                continue        # unmodified upstream file; it is in the assembled tree, not in overlay/
            failures.append("config missing: " + rel)
            continue
        if sha256_file(path) != row["released_sha256"]:
            failures.append("{}: byte hash differs from released_sha256".format(rel))
        if functional_sha256(path) != row["functional_sha256"]:
            failures.append("{}: functional hash differs from the run-time config".format(rel))
        if (row["recorded_sha256"] == row["released_sha256"]) != row["byte_identical"].startswith("yes"):
            failures.append("{}: byte_identical column is inconsistent".format(rel))

    # ---- records ---------------------------------------------------------------------
    expected = {"{}_s{}.json".format(r.group, s) for r in campaign.CAMPAIGN for s in campaign.SEEDS}
    present = {f for f in os.listdir(args.eval_dir) if f.endswith(".json")}
    for f in sorted(present - expected):
        failures.append("unexpected record in {}: {}".format(args.eval_dir, f))

    protocol = collections.defaultdict(set)
    devices = collections.Counter()
    models = {}
    table = []
    for row in campaign.CAMPAIGN:
        agent_rel = "data/agents/" + row.agent_yaml
        cfg_hashes = set()
        n = 0
        for seed in campaign.SEEDS:
            name = "{}_s{}.json".format(row.group, seed)
            path = os.path.join(args.eval_dir, name)
            if not os.path.isfile(path):
                failures.append("{} s{}: missing {}".format(row.group, seed, name))
                continue
            n += 1
            with open(path, "r", encoding="utf-8") as f:
                m = json.load(f)["metadata"]
            for field in PROTOCOL_FIELDS:
                protocol[field].add(json.dumps(m.get(field)))
            devices[m.get("runtime", {}).get("device_name", "unknown")] += 1
            if m.get("agent_config") != agent_rel:
                failures.append("{} s{}: agent_config is {}, expected {}".format(
                    row.group, seed, m.get("agent_config"), agent_rel))
            cfg_hashes.add(m.get("agent_config_sha256"))
            # the published records carry the run-time hash; a re-run with the shipped
            # (comment-scrubbed) config records the released hash. Both name the same config.
            if m.get("agent_config_sha256") not in (hashes[agent_rel]["recorded_sha256"],
                                                    hashes[agent_rel]["released_sha256"]):
                failures.append("{} s{}: agent_config_sha256 matches neither the recorded nor "
                                "the released hash in config_hashes.tsv".format(row.group, seed))
            model = m.get("model_sha256")
            if not model:
                failures.append("{} s{}: no model_sha256".format(row.group, seed))
            elif model in models:
                failures.append("{} s{}: checkpoint hash collides with {}".format(row.group, seed, models[model]))
            else:
                models[model] = "{} s{}".format(row.group, seed)
            if str(seed) not in (m.get("model_file") or ""):
                failures.append("{} s{}: model_file {!r} does not name the seed".format(
                    row.group, seed, m.get("model_file")))
            rec = samples.get((row.group, seed))
            if rec is None:
                failures.append("{} s{}: no row in results/train_samples.tsv".format(row.group, seed))
            elif rec["final_samples"] == "NA":
                warnings.append("{} s{}: budget not confirmable (no training history on W&B)".format(
                    row.group, seed))
            elif int(rec["budget"]) != row.budget or float(rec["final_samples"]) < 0.999 * row.budget:
                failures.append("{} s{}: {} samples of a {} budget".format(
                    row.group, seed, rec["final_samples"], row.budget))
        if len(cfg_hashes) > 1:
            failures.append("{}: {} distinct agent configs across seeds".format(row.group, len(cfg_hashes)))
        table.append((row.group, row.arm, row.table_row or "-", n))

    for field, values in sorted(protocol.items()):
        if len(values) != 1 or "null" in values:
            failures.append("protocol field {} is absent or differs: {}".format(field, sorted(values)))
        elif field in EXPECTED_PROTOCOL and json.loads(next(iter(values))) != EXPECTED_PROTOCOL[field]:
            failures.append("protocol field {} = {} != campaign.py {}".format(
                field, next(iter(values)), EXPECTED_PROTOCOL[field]))
    env_sha = json.loads(next(iter(protocol["env_config_sha256"]))) if protocol["env_config_sha256"] else None
    # as for the agent configs: the run-time hash, or the shipped file's for a re-run
    env_row = hashes[campaign.EVAL_ENV_CONFIG]
    if env_sha not in (env_row["recorded_sha256"], env_row["released_sha256"]):
        failures.append("recorded eval env hash matches neither the recorded nor the released "
                        "hash of the shipped eval env in config_hashes.tsv")

    print("=" * 78)
    print("COVERAGE / PROTOCOL / PROVENANCE")
    print("=" * 78)
    print("{:<46} {:<18} {:>3} {:>3}".format("group", "arm", "row", "n"))
    for group, arm, table_row, n in table:
        print("{:<46} {:<18} {:>3} {:>3}".format(group, arm, table_row, n))
    print("\nprotocol shared by all {} records:".format(sum(t[3] for t in table)))
    for field in PROTOCOL_FIELDS:
        print("   {:<22} {}".format(field, ", ".join(sorted(protocol[field]))[:70]))
    print("\nevaluation GPUs (differences rule out episode-level pairing only):")
    for name, count in devices.most_common():
        print("   {:4d}  {}".format(count, name))
    print("\ndistinct evaluated checkpoints: {}".format(len(models)))
    if warnings:
        print("\nWARNINGS ({}):".format(len(warnings)))
        for w in warnings:
            print("   ! " + w)
    if failures:
        print("\nFAILURES ({}):".format(len(failures)))
        for f in failures:
            print("   X " + f)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
