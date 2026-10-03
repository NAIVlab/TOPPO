# Provenance: AMP humanoid experiments

## Upstream code

| | |
|---|---|
| Project | MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control |
| Author | Xue Bin Peng (arXiv:2510.13794); BibTeX in `README.md` |
| Repository | https://github.com/xbpeng/MimicKit |
| Pinned commit | `2ed1e6c093bb0829f55d33cb4f7a1731cfe6cb69` ("Update README_AWR.md", 2026-06-23), the tip of upstream `main` when the runs were made and when this directory was cut |
| License | Apache-2.0, "Copyright 2025, Xue Bin Peng"; upstream ships no NOTICE file |

Every one of the 250 published runs was trained and evaluated with this commit plus the changes in
`CHANGES.md`. Upstream files that are not listed there are used unmodified.

## Upstream data

| | |
|---|---|
| Project | DeepMimic (Peng, Abbeel, Levine and van de Panne, 2018) |
| Repository | https://github.com/xbpeng/DeepMimic |
| Commit fetched by `setup.sh` | `1f915c52fcd4b95b5f5f15b759ae91bd81e9a801` (2025-11-27); the clip itself last changed in `3ca04d3` (2018) |
| File | `data/motions/humanoid3d_walk.txt`, 36,185 bytes, sha256 `32d082c3599bc40aac50b1afa196ff1e2cf3842bb40f0541ab088845f6328a41` |
| License | MIT, Copyright (c) 2018 Xue Bin Peng (`third_party/deepmimic/LICENSE`) |
| Converted file | `data/motions/humanoid/deepmimic_humanoid_walk.pkl` (39 frames, 30 fps, wrap), sha256 `5ba7e98c67049673f506f8756ef3b983d6b66aeb5da0abcd482a6eba751ef43a` |

The conversion (`overlay/tools/deepmimic_to_mimickit/deepmimic_to_mimickit.py`, ours) reproduces the
`.pkl` used by the runs byte for byte; `setup.sh` checks both hashes. The DeepMimic paper
(arXiv:1804.02717, Sec. 10) states that its skills are learned from motion-capture data collected
from mocap.cs.cmu.edu and mocap.cs.sfu.ca, manually processed and retargeted to its characters. The
DeepMimic repository distributes the processed clips under its MIT license; for the original
recordings, see the terms of those databases.

## Simulator and runtime

Recorded in the runtime metadata of all 250 evaluation records: Python 3.12.13, torch
2.10.0+cu128 (CUDA 12.8), newton 1.3.0, warp-lang 1.15.0, mujoco 3.10.0, mujoco-warp 3.10.0.1,
numpy 2.5.1 (225 records) or 2.3.5 (25 records, one evaluation host). The engine config
(`data/engines/newton_engine.yaml`, unmodified upstream, sha256 `85c4897e…`) selects Newton's
MuJoCo-Warp solver, at 240 Hz simulation and 30 Hz control. Training-time package versions were
not recorded per run. Isaac Gym and Isaac Lab were not used by any of the 250 runs.

## How to verify our delta

```bash
git clone https://github.com/xbpeng/MimicKit upstream-mimickit
cd upstream-mimickit && git checkout 2ed1e6c
P=/path/to/experiments/amp_humanoid       # this directory
git apply --numstat "$P/patches/0001-toppo-modifications-to-mimickit-2ed1e6c.patch"   # the counts in CHANGES.md
git apply "$P/patches/0001-toppo-modifications-to-mimickit-2ed1e6c.patch"
git diff --stat                            # the 8 modified files, nothing else
```

`setup.sh` does the same and then checks every patched, added and downloaded file against
`MANIFEST.sha256`. The patch covers only the upstream files we **modified** in place; the files we
**added**, including the configs derived from upstream configs under new names, are in `overlay/`
and listed in `CHANGES.md`.

## Relationship to the research code

The runs used a working tree of MimicKit at `2ed1e6c` with uncommitted changes. This directory
was cut from that tree as follows, and the steps were verified when it was built:

1. The research changes to the 8 upstream files, taken as a diff against `2ed1e6c`, apply cleanly
   and reproduce the research files byte for byte (+854 / −33 lines).
2. The released patch differs from that diff only by the change notices (2 lines per file, Apache-2.0
   4(b)), one reworded code comment in `ppo_agent.py`, and the removal of three runtime-metadata
   fields from evaluation output in `run.py` (see `CHANGES.md`). After stripping docstrings and
   assertion messages, the abstract syntax trees of the released and research files are identical
   for 7 of the 8 files; `run.py` differs only by those three fields and the unused `socket` import.
3. The 11 added modules are identical to the research modules up to comments, docstrings and
   assertion messages (same abstract syntax tree after stripping those); `eval_util.py`, which
   defines the style metric, is byte-identical (sha256 `84d2ff42…`, the hash the paper's
   provenance records).
4. The 25 agent configs and the 2 env configs are derived from MimicKit configs, so each now
   starts with a change notice naming its upstream source (Apache-2.0, section 4(b); see
   `CHANGES.md`). Apart from that notice, 10 of the agent configs and both env configs are
   byte-identical to the ones that ran; 15 agent configs also had their leading comment block
   rewritten (it named internal tracking notes or log paths; in
   `amp_sac_walk_utd512_agent.yaml` a stale schedule comment was also corrected). Every one parses
   (`yaml.safe_load`) to exactly the configuration that ran. `provenance/config_hashes.tsv` gives the
   recorded hash, the released hash and a comment-independent functional hash for each; the
   recorded hashes are the `agent_config_sha256` and `env_config_sha256` values inside the
   evaluation records (the agent ones also on the W&B runs). The dataset config is byte-identical.
5. The evaluation records in `results/` are the original records with four fields removed (host
   name, scheduler job id, `CUDA_VISIBLE_DEVICES`, and the path of an episode-level `.npz` that is
   not shipped), renamed to their W&B group; every metric value is unchanged.
   `provenance/eval_files.tsv` maps each one to its original file name and both hashes. The 60
   records behind the paper's table match, by original sha256, the snapshot the paper's own
   recomputation archive pins.
