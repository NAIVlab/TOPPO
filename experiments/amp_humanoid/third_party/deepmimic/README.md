# DeepMimic walking clip

The AMP experiments use one reference motion: `data/motions/humanoid3d_walk.txt` from
[xbpeng/DeepMimic](https://github.com/xbpeng/DeepMimic) (MIT License, Copyright (c) 2018
Xue Bin Peng; the license text is `LICENSE` next to this file).

| | |
|---|---|
| Repository | https://github.com/xbpeng/DeepMimic |
| Commit fetched by `setup.sh` | `1f915c52fcd4b95b5f5f15b759ae91bd81e9a801` (the file itself last changed in `3ca04d3`, 2018) |
| File | `data/motions/humanoid3d_walk.txt`, 36,185 bytes |
| Converted file | `data/motions/humanoid/deepmimic_humanoid_walk.pkl` (39 frames, 30 fps, wrap loop mode) |

`SHA256SUMS` pins both the downloaded text file and the converted MimicKit motion file.
`setup.sh` downloads the clip into `external/deepmimic/data/motions/` of the assembled tree,
checks its hash, converts it with `tools/deepmimic_to_mimickit/deepmimic_to_mimickit.py`, and
checks the hash of the result. The TOPPO release itself does not redistribute the clip.

The DeepMimic paper (Peng et al., 2018, arXiv:1804.02717, Sec. 10) states that its skills are
learned from motion-capture data collected from mocap.cs.cmu.edu and mocap.cs.sfu.ca, manually
processed and retargeted to its characters. The DeepMimic repository distributes the processed
clips under its MIT license; for the original recordings, see the terms of those databases.
