# FairGrad solver benchmark

The α-fairness KKT system `G x = x^(-1/alpha)` has to be solved once per minibatch for every side
that uses FairGrad (the critic, in TOPPO), at K = 50 tasks. Whether that is affordable is what makes
FairGrad-on-the-critic a shippable module rather than a thought experiment. This directory holds
the benchmarks behind that claim, **and their raw output**, so the numbers can be checked rather
than taken on trust.

## Benchmarks

Three sets, with very different reliability.

### 1. Real-Gram, in-training (the reliable one)

Gram matrices captured from actual MT50 training, replayed through both solvers.

- `solverbench_rfv1.csv` — 1,984 real K=50 solves
- `realgram_results.md` — the protocol, the analysis and the regeneration recipe
- `realgram_summary.py` — recomputes the summary from the CSV (or from a fresh capture)

**Median 83× faster than scipy** (y-Newton median 0.40 ms vs scipy 32.5 ms), 100% converged,
median 2 Newton iterations. This reproduces exactly from the committed CSV, on any machine.
**Prefer this number.** The CSV was captured with an earlier, unreleased version of the
`MTRL_SOLVER_BENCH` hook; the hook in `mtrl/gradients/fairgrad.py` re-implements it with the
same protocol (see `realgram_results.md`).

### 2. Synthetic spike sweep (host-dependent — read the caveat)

A stress sweep over structure × number of dead tasks × dead-task scale.

- `fairgrad_spike_scenarios_a1.py` — the generator
- `run_alpha1_sweep.sbatch` — the job wrapper
- `raw_alpha1_sweep.out` — **the raw output** (85 rows)
- `warm_vs_cold_a1.py`, `raw_warm_vs_cold_{1,2}.out` — warm- vs cold-start comparison on five
  scenarios of its own (not the regimes of the appendix solver table; see section 3)

Measured over the 85 rows in `raw_alpha1_sweep.out`, and over the 81 of them at α = 1:

| statistic | y-Newton vs scipy, all 85 rows | α = 1 rows only (81) |
|---|---:|---:|
| minimum | **9.2×** | 9.2× |
| 25th percentile | 33.9× | 24.7× |
| **median** | **74.4×** | 74.4× |
| 75th percentile | 196.6× | 231.1× |
| maximum | **6,914×** | 6,914× |

**Do not quote a narrow range from this table as if it were the range.** The speedup spans three
orders of magnitude across rows and is strongly host-dependent: independent re-runs of the same
sweep on different CPUs produced materially different per-row numbers (the ordering is stable, the
magnitudes are not). Any figure taken from here must state the row it came from and the host it
ran on, or be reported as a median with the full spread.

The sweep is also dominated by adversarial rows — near-zero-gradient "dead" tasks at scales down
to 1e-20 — which is exactly where scipy's `least_squares` degrades and where the largest ratios
come from. Those rows are informative about worst-case robustness, not about typical cost.

### 3. Appendix solver table and scale-invariance controls

- `selected_regimes_a1.py` — the generator of the appendix solver table (`tab:solver`): scipy
  (verbatim FairGrad paper reference), the SGD + momentum solver of the FairGrad RL release
  (lr 0.1, momentum 0.5, 20 iterations, track-best), and cold- and warm-start y-Newton, on five
  synthetic regimes at α = 1, D = 1024. The warm-start column calls the production solver.
- `raw_selected_regimes.out` — a rerun of it on a different host from the paper's table, with
  single-threaded BLAS. The paper's table comes from a May 2026 run whose log was not archived.
  The rerun matches the table's SGD and y-Newton residuals to the printed digit, except row 2's
  cold start (2.6e-9 here, 2.7e-9 in the paper). scipy's residuals do not match exactly (rows 1,
  4 and 5 are close; rows 2 and 3 differ by about 70× and 4,000×, all far below the 1e-2 target),
  because its trust-region path is sensitive to floating-point details. Nor do the
  timings: row 3's speedup is 593× here and 95× in the paper. As with the sweep above, quote
  speedups together with the host.
- `scale_invariance_controls.py` — the data behind the synthetic scale-invariance figure
  (`fig:f4`): tight KKT solves under single-task and global rescaling, and the released solver
  under global rescaling. Run with `OMP_NUM_THREADS=1`, its two CSVs,
  `raw_scale_invariance_single_task.csv` and `raw_scale_invariance_global.csv`, are
  byte-identical to the data the paper's figure was drawn from, and
  `raw_scale_invariance_stats.json` holds the same summary values. The plotting code is not
  shipped (matplotlib is not a dependency of this repo).

## Typical cost in a real run

Every FG-c run logs `fairgrad_critic/wall_time_ms`, the time of one critic-side call: fp64 Gram
formation plus the solve (not per-task gradient extraction or recombination). At 496 calls per
collect (16 PPO repeats × 31 stratified minibatches), a 100M-step MT50 seed makes ~99,200 calls.
Gram formation runs on the GPU, so the time per call depends mostly on the GPU. The
denominator below, the 19.8 h TOPPO estimate of Appendix E.4 (evaluation off), was measured on an
RTX 3090. From the public runs (mean over each run's logged history):

| group, seeds | ms per call | per 100M-step seed |
|---|---|---|
| `mt50v2_ppo_apc_cfg1_npaw_lnc_vc1_400x3`, the headline (Welford PopArt, V2), seeds 3–9 on RTX 3090 hosts | 6.9–7.4 per seed | 11–12 min, ~1.0% of 19.8 h (0.96–1.03%) |
| the same group, seeds 0–2 on an H200 NVL host | 0.8–0.9 per seed | ~1.4 min, ~0.17% of the ~14 h such a seed takes there with evaluation on |
| `mt50v1_ppo_apc_cfg1_npae_lnc_vc1_400x3` (EMA PopArt, V1); its runs do not record the GPU | 2.6, mean over the 10 seeds; its last logged values average 2.46 | ~4.4 min (≈0.37% of 19.8 h); at 2.46 ms, ~4.1 min (≈0.34%) |

The headline group's 10-seed mean, 5.2 ms, averages the two kinds of host, so it is not used for
the share. On the GPU of the timing study the cost is about 1% of the 19.8 h evaluation-off
estimate (the figure Appendix E.4 reports); on the H200 NVL it is under 0.2% of a seed's wall
clock with evaluation on.

## Reproducing

```bash
python realgram_summary.py                # the real-Gram summary, from the committed CSV
python fairgrad_spike_scenarios_a1.py     # regenerates the synthetic sweep (host-dependent)
python warm_vs_cold_a1.py                 # warm- vs cold-start
OMP_NUM_THREADS=1 python selected_regimes_a1.py      # appendix solver table (timings host-dependent)
OMP_NUM_THREADS=1 python scale_invariance_controls.py --out-dir /tmp/f4   # fig:f4 data
```

A new real-Gram capture needs a two-epoch MT50 training run; `realgram_results.md` gives the exact
command and the expected row count.

## Other files

- `fairgrad_solver_benchmark.py`, `fairgrad_solver_comprehensive_bench.py` — earlier α = 2
  development benchmarks (including x-space Newton and fixed-point variants). No paper number
  comes from them.
- `fairgrad_solver_formulation.md` — the pre-implementation design memo for the y-Newton solver.
  It is kept for its derivations; `mtrl/gradients/fairgrad.py` is authoritative for the current
  defaults.

## Provenance

`raw_alpha1_sweep.out` and the two warm/cold outputs are the original job outputs, recovered from
the cluster account they were produced on and scrubbed of host and account identifiers. Node names
appear as `${NODE}`. The EXPERIMENT 1 and EXPERIMENT 3 headers of `raw_alpha1_sweep.out` read
"α=2" (a hard-coded label, since fixed in the generator), but the job ran with `SPIKE_ALPHA=1.0`:
all 80 EXPERIMENT 1 rows are α = 1. EXPERIMENT 2 (5 rows) is an explicit α sweep over
{1, 1.5, 2, 3, 5}, so 81 of the 85 rows in the table above are α = 1.
