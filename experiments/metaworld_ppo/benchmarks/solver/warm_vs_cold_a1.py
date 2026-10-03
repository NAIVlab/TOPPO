"""Cold vs diagonal-warm-start y-Newton at ALPHA (default 1.0) on five scenarios:
well-conditioned K=50, K=50 with 1, 5 or 16 near-dead tasks, and a realistic K=10 Gram.
These are NOT the regimes of the appendix solver table (tab:solver); that table's generator
is selected_regimes_a1.py.
Reuses fairgrad_spike_scenarios_a1 (make_grads, solvers). Times: median of 3 reps."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fairgrad_spike_scenarios_a1 as B

A = B.ALPHA
def diag_warm_x0(G, alpha=A, eps=1e-8):
    d = np.maximum(np.diagonal(G), eps)
    return np.power(d, -(alpha / (alpha + 1.0)))

def tmed(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), out

def realistic_gram(seed=42):
    rng = np.random.default_rng(seed)
    K, D = 10, 5000
    scales = np.concatenate([[1e-10], [1.0, 1.5], rng.uniform(10, 20, 7)])
    g = rng.standard_normal((K, D))
    for i in range(K):
        g[i] = g[i] / np.linalg.norm(g[i]) * scales[i]
    return g @ g.T

scen = [
    ("well-cond K=50",   B.make_grads(50, 5000, 0, 1e-20, "iid", seed=42)),
    ("1 near-dead K=50", B.make_grads(50, 5000, 1, 1e-12, "iid", seed=42)),
    ("5 near-dead K=50", B.make_grads(50, 5000, 5, 1e-12, "iid", seed=42)),
    ("16 near-dead K=50",B.make_grads(50, 5000, 16, 1e-12, "iid", seed=42)),
    ("realistic K=10",   realistic_gram()),
]
# BLAS/scipy warmup so the first timed call is not polluted by one-time init
_Gw = B.make_grads(10, 500, 0, 1e-20, "iid", seed=1)
B.solve_scipy(_Gw, A); B.solve_ynewton(_Gw, A, eta=1e-2)
print(f"ALPHA = {A}")
print(f"{'scenario':18s} {'scipy ms':>9s} | {'yN cold ms':>10s} {'it':>3s} | {'yN warm ms':>10s} {'it':>3s} | spd(cold) spd(warm)")
for name, G in scen:
    t_sci, _ = tmed(lambda: B.solve_scipy(G, A), reps=3)
    t_c, (xc, okc, itc) = tmed(lambda: B.solve_ynewton(G, A, eta=1e-2))
    x0 = diag_warm_x0(G)
    t_w, (xw, okw, itw) = tmed(lambda: B.solve_ynewton(G, A, x0=x0, eta=1e-2))
    print(f"{name:18s} {t_sci:9.2f} | {t_c:10.3f} {itc:3d} | {t_w:10.3f} {itw:3d} | {t_sci/t_c:8.0f}x {t_sci/t_w:8.0f}x")
