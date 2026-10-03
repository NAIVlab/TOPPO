"""Regression tests for FairGrad NaN/inf guards (v2.14.0 solver stack).

With v2.14.0, FairGrad uses a two-tier solver:

1. Primary: y-space projected Newton (`_solve_ynewton`) — fast, numerically
   well-behaved (exponents capped at y_max=50 preventing fp overflow).
2. Fallback: scipy.optimize.least_squares — engaged when y-Newton fails
   to converge within max_iter or stalls in line search.
3. Final guard: if BOTH solvers return non-finite output, fall back to
   uniform weights (1/K) and set solver_fallback=True.

These tests verify:
- Clean input: primary y-Newton succeeds, no fallbacks, finite weights.
- y-Newton stall → scipy fallback engaged, flags reflect the path.
- Both solvers return NaN → uniform fallback, weights uniform and finite.
- Combined gradient is always finite regardless of solver path.
"""

from unittest.mock import patch

import numpy as np
import torch

from mtrl.gradients.fairgrad import (
    _solve_scipy_fallback,
    _solve_ynewton,
    fairgrad_combine,
    solve_fairgrad_weights,
)


# -----------------------------------------------------------------------------
# Clean-path tests
# -----------------------------------------------------------------------------

def test_clean_solve_uses_ynewton_no_fallback():
    """Well-conditioned Gram → y-Newton converges, no fallback should fire."""
    torch.manual_seed(0)
    grad = torch.randn(3, 64, dtype=torch.float64)
    diag = solve_fairgrad_weights(grad, alpha=1.0)
    assert diag.solver_fallback is False
    assert diag.used_scipy_fallback is False
    assert diag.solver_converged is True
    assert torch.isfinite(diag.weights).all()
    assert diag.solver_iters > 0 and diag.solver_iters < 50


def test_clean_solve_at_alpha_2():
    """Well-conditioned Gram at α=2 (FairGrad MetaWorld default)."""
    torch.manual_seed(1)
    grad = torch.randn(10, 128, dtype=torch.float64)
    diag = solve_fairgrad_weights(grad, alpha=2.0)
    assert diag.solver_fallback is False
    assert diag.used_scipy_fallback is False
    assert torch.isfinite(diag.weights).all()
    assert (diag.weights > 0).all()


def test_single_task_trivial():
    """K=1 edge case: weights = [1], dtype preserved (v2.14.1)."""
    grad = torch.randn(1, 32, dtype=torch.float64)
    diag = solve_fairgrad_weights(grad, alpha=2.0)
    assert diag.solver_converged is True
    assert diag.weights.dtype == torch.float64
    assert torch.allclose(diag.weights, torch.tensor([1.0], dtype=torch.float64))


def test_single_task_trivial_fp32():
    """K=1 edge case: weights = [1], fp32 input → fp32 weights (v2.14.1)."""
    grad = torch.randn(1, 32, dtype=torch.float32)
    diag = solve_fairgrad_weights(grad, alpha=2.0)
    assert diag.weights.dtype == torch.float32
    assert torch.allclose(diag.weights, torch.tensor([1.0]))


def test_two_tasks_k_equal_2():
    """K=2 edge case: well-conditioned Gram → both solvers converge."""
    torch.manual_seed(5)
    grad = torch.randn(2, 64, dtype=torch.float64)
    diag = solve_fairgrad_weights(grad, alpha=2.0)
    assert diag.solver_converged is True
    assert diag.solver_fallback is False
    assert torch.isfinite(diag.weights).all()
    assert (diag.weights > 0).all()
    # Residual should be well under η=1e-2 default
    assert diag.residual < 0.1


def test_two_tasks_dead_task_k2():
    """K=2 with one dead task: y-Newton handles cleanly."""
    torch.manual_seed(6)
    grad = torch.randn(2, 64, dtype=torch.float64)
    grad[0] = grad[0] / grad[0].norm() * 1e-12
    diag = solve_fairgrad_weights(grad, alpha=2.0)
    assert diag.solver_fallback is False
    assert torch.isfinite(diag.weights).all()
    # Dead task 0 gets higher weight than active task 1 (fairness boosts disadvantaged).
    assert diag.weights[0] > diag.weights[1]


# -----------------------------------------------------------------------------
# y-Newton-failure → scipy-fallback tests
# -----------------------------------------------------------------------------

def _fake_ynewton_stall(*args, **kwargs):
    """Simulate y-Newton stalling (returns non-converged)."""
    K = args[0].shape[0]
    return np.ones(K, dtype=np.float64) / K, False, 50


def test_ynewton_stall_engages_scipy():
    """When y-Newton returns converged=False, scipy fallback should run and
    its result used if scipy succeeds with finite output."""
    torch.manual_seed(2)
    grad = torch.randn(3, 32, dtype=torch.float64)
    with patch(
        "mtrl.gradients.fairgrad._solve_ynewton",
        side_effect=_fake_ynewton_stall,
    ):
        diag = solve_fairgrad_weights(grad, alpha=1.0)
    assert diag.used_scipy_fallback is True
    # scipy should have found a good solution here (Gram is well-conditioned)
    assert diag.solver_fallback is False
    assert torch.isfinite(diag.weights).all()
    assert diag.solver_converged is True


# -----------------------------------------------------------------------------
# Silent-NaN guards
# -----------------------------------------------------------------------------

def _fake_scipy_nan_result(GTG_np, alpha, eps, x_start=None):
    K = GTG_np.shape[0]
    return np.array([float("nan")] + [0.5] * (K - 1), dtype=np.float64), True, 10


def _fake_scipy_inf_result(GTG_np, alpha, eps, x_start=None):
    K = GTG_np.shape[0]
    return np.array([float("inf")] + [0.5] * (K - 1), dtype=np.float64), True, 10


def test_silent_nan_from_both_solvers_triggers_uniform_fallback():
    """Both y-Newton and scipy return non-finite → uniform fallback."""
    grad = torch.randn(3, 8, dtype=torch.float64)
    with patch(
        "mtrl.gradients.fairgrad._solve_ynewton",
        side_effect=_fake_ynewton_stall,
    ), patch(
        "mtrl.gradients.fairgrad._solve_scipy_fallback",
        side_effect=_fake_scipy_nan_result,
    ):
        diag = solve_fairgrad_weights(grad, alpha=1.0)
    assert diag.used_scipy_fallback is True
    assert diag.solver_fallback is True
    assert diag.solver_converged is False
    assert torch.isfinite(diag.weights).all()
    assert torch.allclose(
        diag.weights, torch.full((3,), 1.0 / 3, dtype=torch.float64)
    )


def test_silent_inf_from_scipy_triggers_uniform_fallback():
    grad = torch.randn(3, 8, dtype=torch.float64)
    with patch(
        "mtrl.gradients.fairgrad._solve_ynewton",
        side_effect=_fake_ynewton_stall,
    ), patch(
        "mtrl.gradients.fairgrad._solve_scipy_fallback",
        side_effect=_fake_scipy_inf_result,
    ):
        diag = solve_fairgrad_weights(grad, alpha=1.0)
    assert diag.solver_fallback is True
    assert torch.isfinite(diag.weights).all()


def test_combine_is_finite_after_full_fallback():
    """End-to-end fairgrad_combine never returns NaN, even when both
    solvers fail."""
    grads = [torch.randn(8, dtype=torch.float64) for _ in range(3)]
    with patch(
        "mtrl.gradients.fairgrad._solve_ynewton",
        side_effect=_fake_ynewton_stall,
    ), patch(
        "mtrl.gradients.fairgrad._solve_scipy_fallback",
        side_effect=_fake_scipy_nan_result,
    ):
        combined, _, diag = fairgrad_combine(grads, alpha=1.0)
    assert diag is not None
    assert diag.solver_fallback is True
    assert torch.isfinite(combined).all()


# -----------------------------------------------------------------------------
# Dead-task / huge-weight invariants — v2.14.0 y-Newton must preserve these
# -----------------------------------------------------------------------------

def test_ynewton_handles_dead_task_without_fallback():
    """One task with ‖g‖ ≈ 1e-10: y-Newton should converge within
    max_iter at η=1e-2 (the natural stopping regime), not fall back
    to scipy or uniform."""
    torch.manual_seed(3)
    grad = torch.randn(5, 64, dtype=torch.float64)
    grad[0] = grad[0] / grad[0].norm() * 1e-10  # dead task
    diag = solve_fairgrad_weights(grad, alpha=2.0)
    assert diag.solver_fallback is False
    assert torch.isfinite(diag.weights).all()
    # The dead task gets an elevated weight but finite (y_max=50 → x≤5e21)
    assert diag.weights[0] > diag.weights[1:].mean()


def test_ynewton_huge_weight_does_not_corrupt_combine():
    """Dead task with huge weight: fp64 combine must produce finite output
    even when individual w_i * g_i products are large."""
    torch.manual_seed(4)
    # 10 tasks, one dead at 1e-12, rest O(1)
    grads = [torch.randn(128, dtype=torch.float32) for _ in range(10)]
    grads[0] = grads[0] / grads[0].norm() * 1e-12
    combined, _, diag = fairgrad_combine(grads, alpha=2.0)
    assert torch.isfinite(combined).all()
    assert diag is not None
    # Verify fp64 path did its job: even if diag.weights[0] is huge in fp64,
    # the combined tensor (in fp32) should be finite because w*g products
    # are mathematically bounded by KKT.
    assert diag.weights[0].float().isfinite()


# -----------------------------------------------------------------------------
# v2.14.1 diagnostic fields — must survive through the solver path
# -----------------------------------------------------------------------------

def test_diagnostics_expose_solver_iters_and_scipy_fallback_fields():
    """v2.14.1: FairGradDiagnostics must carry solver_iters and
    used_scipy_fallback for downstream logging."""
    torch.manual_seed(7)
    grad = torch.randn(5, 64, dtype=torch.float64)
    diag = solve_fairgrad_weights(grad, alpha=2.0)
    assert hasattr(diag, "solver_iters")
    assert hasattr(diag, "used_scipy_fallback")
    assert isinstance(diag.solver_iters, int)
    assert isinstance(diag.used_scipy_fallback, bool)
    # Clean Gram → Newton converges without scipy engagement.
    assert diag.solver_iters > 0
    assert diag.used_scipy_fallback is False


def test_diagnostics_as_dict_includes_new_fields():
    """v2.14.1: as_dict must emit solver_iters and used_scipy_fallback
    so they reach wandb once the multitask_ppo buffer whitelist includes them."""
    torch.manual_seed(8)
    grad = torch.randn(4, 32, dtype=torch.float64)
    diag = solve_fairgrad_weights(grad, alpha=2.0)
    d = diag.as_dict()
    assert "fairgrad/solver_iters" in d
    assert "fairgrad/used_scipy_fallback" in d
    assert d["fairgrad/used_scipy_fallback"] == 0.0  # no scipy fallback on clean Gram


# -----------------------------------------------------------------------------
# MTRL_SOLVER_BENCH hook: CSV layout, Newton-only columns, no effect on output
# -----------------------------------------------------------------------------

def test_solver_bench_hook_records_newton_and_fallback_separately(tmp_path, monkeypatch):
    """The newton_* columns stay Newton's own result on a fallback row, booleans
    are written as 0/1, the header extends the archived CSV's nine columns, and
    the hook does not change the solved weights."""
    import csv
    import os

    import mtrl.gradients.fairgrad as fg

    torch.manual_seed(11)
    grad = torch.randn(5, 48, dtype=torch.float64)
    reference = solve_fairgrad_weights(grad, alpha=1.0).weights

    out = tmp_path / "bench.csv"
    monkeypatch.setattr(fg, "_BENCH_PATH", str(out))
    monkeypatch.setattr(fg, "_BENCH_STARTED", False)
    clean = solve_fairgrad_weights(grad, alpha=1.0)
    assert torch.equal(clean.weights, reference)

    real_newton = fg._solve_ynewton

    def stalled_newton(GTG_np, alpha, eps):
        x, _, _ = real_newton(GTG_np, alpha, eps)
        return x, False, 3

    with patch("mtrl.gradients.fairgrad._solve_ynewton", side_effect=stalled_newton):
        forced = solve_fairgrad_weights(grad, alpha=1.0)
    assert forced.used_scipy_fallback is True

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
        f.seek(0)
        header = next(csv.reader(f))
    assert tuple(header) == fg._BENCH_HEADER
    archived = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "benchmarks", "solver", "solverbench_rfv1.csv")
    with open(archived, newline="") as f:
        assert header[:9] == next(csv.reader(f))
    assert len(rows) == 2
    clean_row, fallback_row = rows
    assert clean_row["newton_converged"] == "1" and clean_row["fallback_used"] == "0"
    assert fallback_row["newton_converged"] == "0"
    assert fallback_row["newton_iters"] == "3"
    assert fallback_row["fallback_used"] == "1"
    assert fallback_row["fallback_ok"] in ("0", "1")
    assert int(fallback_row["fallback_nfev"]) == forced.solver_iters
    for row in rows:
        assert row["scipy_success"] in ("0", "1")
        assert float(row["t_newton_ms"]) >= 0.0 and float(row["t_scipy_ms"]) > 0.0
