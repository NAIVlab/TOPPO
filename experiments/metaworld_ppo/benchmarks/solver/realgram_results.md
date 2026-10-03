# In-environment solver benchmark — real MT50 Gram matrices

How much faster is the y-space Newton solver than the scipy reference *inside a
Meta-World training loop*, measured on the Gram matrices actually encountered
during training rather than on synthetic proxies.

## Protocol

- MT50, V1 rewards, FG-c (α=1) + PopArt (Welford, the only PopArt variant in
  the code of the time) + LN-c, actor `default`. PCGrad is deliberately
  excluded so the measurement isolates the solver, and with a default actor
  only critic-side (α=1) solves are logged.
- Seed 0, **2 epochs = 2M env steps**, evaluation disabled. One RTX 3090,
  16 CPU / 64 GB, 22:24 elapsed.
- Instrumentation: an env-gated patch to the FairGrad solver
  (`MTRL_SOLVER_BENCH=<csv>`, default OFF). For each production solve the same
  fp64 Gram is additionally solved with the scipy reference (`least_squares`,
  `x0 = 1/K`, bounds `(1e-12, ∞)`); the scipy solution is timed and discarded,
  so training dynamics are untouched. Both timings are solve-only: the Newton
  time covers the y-Newton call alone, not Gram formation, the device-to-host
  copy or a fallback.
- 1,984 solves logged = 4 collects × 16 repeats × 31 stratified minibatches.
  (`batch_size` 15,625 is 1/32 of a 500k-step collect, but the stratified
  sampler gives each task a 312- or 313-sample quota of its 10,000 samples and
  stops when any task runs out, so each repeat yields 31 full minibatches.)

## Results (n = 1,984 real K=50 Grams)

| | median | mean | p95 | max |
|---|--:|--:|--:|--:|
| y-Newton (diagonal warm start) | 0.40 ms | 0.42 ms | 0.56 ms | 5.5 ms |
| scipy reference | 32.5 ms | 36.5 ms | 55.5 ms | 463 ms |

- y-Newton: median **2 iterations**, max 19, **100% converged**; scipy 100% success.
- **Per-solve speedup: median 83× (p5 56×, p95 138×, max 919×); aggregate 87×.**
- Totals over the 1,984 solves: 0.8 s (Newton) vs 72.3 s (scipy).
- Projected over a full 100M-step seed (99,200 solves): **~0.7 min vs ~1.0 h**.

All of the above recomputes from the committed CSV with
`python realgram_summary.py` (standard library only).

## Caveats

- First 2M steps only. Gram geometry evolves over training, so the capture
  says nothing about late-training small-norm extremes, the regime where scipy
  degrades in the synthetic sweep (README.md). Within the capture, the slow
  scipy solves are an early-training transient: the `max = 463 ms` entry is
  the very first solve, the eight slowest all fall in the first collect, and
  the per-collect maxima are 463, 67, 50 and 82 ms.
- Solve-only timings. The production telemetry `fairgrad_critic/wall_time_ms`
  times fp64 Gram formation plus the solve, so it reads higher, and it depends
  on the group and the host: 5.2 ms per call averaged over the headline group
  `mt50v2_ppo_apc_cfg1_npaw_lnc_vc1_400x3` (0.8–0.9 ms per seed on the H200 NVL
  host of seeds 0–2, 6.9–7.4 ms on the RTX 3090 hosts of seeds 3–9), and
  2.6 ms in the EMA-PopArt V1 group `mt50v1_ppo_apc_cfg1_npae_lnc_vc1_400x3`,
  whose last logged values average 2.46 ms. See "Typical cost in a real run"
  in README.md.
- The CSV stores timings, iteration counts and status flags, not the Gram
  matrices or Newton residuals, so the solves themselves cannot be replayed
  from it.

## Artifacts

- Raw CSV: [`solverbench_rfv1.csv`](solverbench_rfv1.csv), committed here. It
  was captured with the original instrument, a patch that is not part of this
  release.
- The `MTRL_SOLVER_BENCH` hook in `mtrl/gradients/fairgrad.py` re-implements
  that instrument with the protocol above (Newton timer around the y-Newton
  call only; scipy reference from `1/K` with bounds `(1e-12, ∞)`). It has been
  checked on synthetic Grams, not by repeating the capture: with the variable
  set or unset, the solver's weights and every diagnostic except wall time are
  bit-identical to the solver without the hook. To regenerate, from
  `experiments/metaworld_ppo/`:

  ```bash
  rm -f run_logs/solverbench.csv        # the hook appends to an existing file
  rm -rf logs/solverbench               # else train.py resumes from its checkpoints
  MTRL_SOLVER_BENCH=run_logs/solverbench.csv \
  python train.py --config configs/ppo_mt50_ad_cfg_popart_lnc_rfv1.yaml --seed 0 \
      --exp-name solverbench --training.epoch 2 --logging.test-interval 0 \
      --logging.no-use-wandb
  python benchmarks/solver/realgram_summary.py --csv run_logs/solverbench.csv
  ```

  The config is the published W&B group `mt50v1_ppo_ad_cfg1_npaw_lnc_vc1_400x3`:
  V1 rewards, a default actor (no PCGrad, and no actor-side FairGrad adding
  α=2 rows), FG-c, Welford PopArt and LN-c. Checked against that group's seed-0
  run config, only the epoch count and evaluation (which the protocol turns
  off) differ, apart from naming and logging fields. Both `rm` lines matter on
  a rerun: the config checkpoints every 1M steps under `logs/solverbench/`, and
  a second run with the same `--exp-name` would resume from them and log fewer
  rows (or none).
  Expect **exactly 1,984 rows, all `K = 50`, `alpha = 1.0`**; more rows mean
  another FairGrad side was on (the summary script keeps only `alpha = 1.0`).
  ~25 min on one RTX 3090.
- A rerun will not give the archived rows back one for one. Training goal order
  is not seeded by `--seed` and GPU kernels are not deterministic, so the Grams
  differ, and absolute timings depend on the host. Compare distributions.
- Output columns: the first nine match the committed CSV (booleans as 0/1).
  Added: `alpha`; `newton_resid`, the KKT residual of Newton's own iterate; and
  `fallback_used`, `fallback_ok`, `fallback_nfev`, `t_fallback_ms` for the
  production scipy fallback, which runs only when Newton does not converge. The
  `newton_*` columns always describe Newton alone, also on rows where the
  fallback ran.
