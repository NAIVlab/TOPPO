# Provenance — SAC-family baselines

## Upstream

| | |
|---|---|
| Project | ARS: Adaptive Reward Scaling for Multi-Task Reinforcement Learning |
| Authors | Myungsik Cho, Jongeui Park, Jeonghye Kim, Youngchul Sung (ICML 2025); BibTeX in `README.md` |
| Repository | https://github.com/ms-cho/Adaptive-Reward-Scaling-ARS-for-MTRL |
| Pinned commit | `67f7df3` ("update readme"), the tip of upstream `main` at the time we forked |
| License | MIT (upstream ships the MIT text with the copyright-holder line left blank) |
| Upstream's own lineage | JAXRL, MIT © 2021 Ilya Kostrikov |

## How to verify our delta

```bash
git clone https://github.com/ms-cho/Adaptive-Reward-Scaling-ARS-for-MTRL upstream-ars
cd upstream-ars && git checkout 67f7df3
P=/path/to/sac_baselines_ars    # this directory
git apply --numstat "$P/patches/0001-toppo-modifications-to-upstream.patch"   # the counts in CHANGES.md
git apply "$P/patches/0001-toppo-modifications-to-upstream.patch"
for f in $(git diff --name-only); do cmp "$f" "$P/$f"; done   # no output: byte-identical
```

The patch covers only the files we **modified**. Files we **added** are listed
in `CHANGES.md` and are present in this directory; they do not appear in the
patch because they have no upstream counterpart. Upstream's `README.md` and
`requirements.txt` are **replaced** by ours rather than patched (also listed in
`CHANGES.md`). `LICENSE` and the remaining 11 upstream `.py` files are
byte-identical to `67f7df3`.

## Deliberate deletion

Upstream vendors a full copy of the mujoco_py-era Meta-World (1006 files,
~251k lines) with no license or notice of its own. We removed it and replaced
it with `metaworld3_ars.py`, which targets Meta-World 3.0.0 as an ordinary
dependency. This both fixes an unattributed-redistribution defect and is what
makes the evaluation protocol match the PPO side.

## Relationship to the private research history

These changes were developed as working-tree edits on top of `67f7df3` and were
never committed upstream-side, which is why this directory — not a commit range —
is the authoritative record.
