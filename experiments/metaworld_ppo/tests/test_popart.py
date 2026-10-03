"""Tests for PopArt (Hessel et al., AAAI 2019) in MultiTaskCritic.

Covers the four load-bearing invariants:

1. **popart-off is bit-identical**: a critic with ``popart_enabled=True`` and
   default state (W=1, b=0, μ=0, σ=1) produces the same forward output as
   one with ``popart_enabled=False``.
2. **POP preserves raw V**: after an arbitrary stats update, ``V_raw`` for
   any fixed ``v_pre`` input is unchanged (this is the load-bearing property
   that keeps GAE consistent across stats updates).
3. **Welford merge matches naive concatenation**: incremental per-task
   merges agree with ``numpy.mean`` / ``numpy.std`` on the concatenated samples.
4. **state_dict round-trip**: ``μ, σ, M2, count, W, b`` all persist through
   ``state_dict()`` / ``load_state_dict()`` so SLURM resume is complete
   without touching ``_PERSISTABLE_EXTRA_ATTRS``.
5. **Smoke one-epoch**: ``train.py --ppo.use-popart`` runs to completion
   on the debug config without raising.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

from mtrl.models.networks import (
    MultiTaskCritic,
    PopArtModule,
    SharedPreprocessNet,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEBUG_CONFIG = REPO_ROOT / "configs" / "debug_ppo.yaml"


def _build_critic(
    task_names,
    popart_enabled: bool = False,
    popart_method: str = "welford",
    popart_ema_decay: float = 3e-4,
) -> MultiTaskCritic:
    """Build a MultiTaskCritic matching the production MT10/MT50 arch:
    preprocess=Identity, critic_body=[64,64] → Linear(64,1), value_heads=Identity."""
    # Tiny hidden sizes to keep tests fast; structurally identical to prod.
    obs_dim = 10 + len(task_names)  # 10 raw obs dims + one-hot tail
    pre = SharedPreprocessNet(input_dim=obs_dim, hidden_sizes=[], activation="relu")
    critic = MultiTaskCritic(
        preprocess_net=pre,
        task_names=task_names,
        critic_hidden_sizes=[64, 64],
        critic_head_hidden_sizes=[],
        activation="relu",
        act_dim=0,
        popart_enabled=popart_enabled,
        popart_min_std=1e-2,
        popart_method=popart_method,
        popart_ema_decay=popart_ema_decay,
    )
    return critic


def _random_obs(batch: int, task_names, seed: int = 0) -> torch.Tensor:
    """Build a random obs batch with one-hot task tail, uniform over tasks."""
    rng = torch.Generator().manual_seed(seed)
    T = len(task_names)
    raw = torch.randn(batch, 10, generator=rng)
    task_ids = torch.randint(0, T, (batch,), generator=rng)
    one_hot = torch.zeros(batch, T)
    one_hot.scatter_(1, task_ids.unsqueeze(1), 1.0)
    return torch.cat([raw, one_hot], dim=-1)


# ---------------------------------------------------------------------------
# 1. popart-off is bit-identical
# ---------------------------------------------------------------------------


def test_popart_off_is_bit_identical():
    task_names = [f"task_{i}" for i in range(5)]
    torch.manual_seed(42)
    c_off = _build_critic(task_names, popart_enabled=False)
    torch.manual_seed(42)
    c_on = _build_critic(task_names, popart_enabled=True)
    # The popart-on critic has extra W, b params — load only the shared
    # backbone weights so the two critics agree on v_pre.
    c_on.load_state_dict(c_off.state_dict(), strict=False)

    obs = _random_obs(32, task_names, seed=1)
    v_off = c_off(obs)
    v_on  = c_on(obs)
    assert v_off.shape == v_on.shape == (32, 1)
    # W=1, b=0, μ=0, σ=1 ⟹ denormalize(affine(v_pre)) = v_pre, so outputs
    # must match to within fp32 noise.
    assert torch.allclose(v_off, v_on, atol=1e-6), (
        f"popart-on at default state should be identity: "
        f"max|Δ| = {(v_off - v_on).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# 2. POP preserves V_raw across a stats update
# ---------------------------------------------------------------------------


def test_pop_invariant_preserves_v_raw():
    T = 6
    torch.manual_seed(0)
    pop = PopArtModule(num_tasks=T, min_std=1e-4)
    # Synthetic: non-trivial starting W, b, μ, σ so the POP rescale is non-trivial.
    pop.W.data = torch.tensor([1.0, 0.5, 2.0, -0.3, 1.2, 0.8])
    pop.b.data = torch.tensor([0.0, 1.0, -0.5, 0.2, 0.1, -1.0])
    pop.mu.data    = torch.tensor([0.0, 5.0, -2.0, 10.0, 0.3, -0.8], dtype=torch.float64)
    pop.sigma.data = torch.tensor([1.0, 3.0,  0.5,  2.0, 0.1,  4.0], dtype=torch.float64)
    pop.M2.data    = pop.sigma.data.pow(2) * torch.tensor([10, 20, 30, 15, 25, 5], dtype=torch.float64)
    pop.count.data = torch.tensor([10, 20, 30, 15, 25, 5], dtype=torch.int64)

    # Fixed probe inputs.
    rng = torch.Generator().manual_seed(7)
    v_pre = torch.randn(40, generator=rng)
    task_idx = torch.randint(0, T, (40,), generator=rng)

    v_raw_old = pop.denormalize(pop.affine(v_pre, task_idx), task_idx).detach().clone()

    # Update with a varied set of returns — widely different distributions per task.
    returns_per_task: Dict[int, torch.Tensor] = {
        0: torch.randn(50, generator=rng) * 1.0 + 0.5,
        1: torch.randn(100, generator=rng) * 5.0 + 20.0,   # big shift
        2: torch.randn(30, generator=rng) * 0.2 - 5.0,
        3: torch.randn(200, generator=rng) * 10.0 + 0.0,
        4: torch.randn(20, generator=rng) * 0.05 + 0.3,   # tiny σ → min_std clamp
        5: torch.randn(80, generator=rng) * 2.0 - 3.0,
    }
    pop.update_and_renormalize(returns_per_task)

    v_raw_new = pop.denormalize(pop.affine(v_pre, task_idx), task_idx).detach().clone()

    # Load-bearing assertion: V_raw is preserved bitwise (within fp64→fp32 cast noise).
    max_abs_diff = (v_raw_old - v_raw_new).abs().max().item()
    assert max_abs_diff < 1e-4, (
        f"POP invariant violated: max|ΔV_raw| = {max_abs_diff:.2e}; "
        f"expected near-zero after the per-task rescale."
    )


# ---------------------------------------------------------------------------
# 3. Welford merge agrees with concatenated-sample mean/std
# ---------------------------------------------------------------------------


def test_welford_merge_matches_naive():
    T = 3
    pop = PopArtModule(num_tasks=T, min_std=1e-8)  # tiny floor so it doesn't bite
    rng = np.random.default_rng(123)

    per_task_all: Dict[int, list] = {0: [], 1: [], 2: []}
    # Three sequential merges with different per-task sample sizes.
    for step in range(3):
        batch: Dict[int, torch.Tensor] = {}
        for t in range(T):
            n = rng.integers(20, 80)
            mean = (t + 1) * 10.0
            std = (t + 1) * 2.0
            samples = rng.normal(mean, std, size=int(n))
            batch[t] = torch.from_numpy(samples.astype(np.float64))
            per_task_all[t].extend(samples.tolist())
        pop.update_and_renormalize(batch)

    for t in range(T):
        all_samples = np.array(per_task_all[t])
        naive_mean = float(all_samples.mean())
        naive_std = float(all_samples.std(ddof=0))

        pop_mean = float(pop.mu[t].item())
        pop_std = float(pop.sigma[t].item())

        assert abs(pop_mean - naive_mean) < 1e-6, (
            f"task {t}: Welford μ={pop_mean:.8f} vs naive μ={naive_mean:.8f}"
        )
        # σ can be affected by the min_std clamp (1e-8 here is effectively off).
        rel_err = abs(pop_std - naive_std) / max(abs(naive_std), 1e-12)
        assert rel_err < 1e-6, (
            f"task {t}: Welford σ={pop_std:.8f} vs naive σ={naive_std:.8f} "
            f"(rel_err={rel_err:.2e})"
        )


# ---------------------------------------------------------------------------
# 4. state_dict round-trip
# ---------------------------------------------------------------------------


def test_save_load_state_dict_roundtrip():
    task_names = [f"task_{i}" for i in range(4)]
    critic_src = _build_critic(task_names, popart_enabled=True)
    pop = critic_src.popart

    # Drive state off from defaults via two merges.
    rng = np.random.default_rng(7)
    for _ in range(2):
        batch = {
            i: torch.from_numpy(
                rng.normal(i * 5.0, (i + 1) * 1.5, size=50).astype(np.float64)
            )
            for i in range(4)
        }
        pop.update_and_renormalize(batch)
    # Also nudge W, b off the identity so we catch a W/b-only save bug.
    with torch.no_grad():
        pop.W.data.add_(0.01 * torch.randn_like(pop.W))
        pop.b.data.add_(0.01 * torch.randn_like(pop.b))

    sd = critic_src.state_dict()
    # Confirm all PopArt tensors are present with the expected keys.  ``nu``
    # is registered for the EMA branch but always present so checkpoint
    # shape stays fixed across method choices (and pre-EMA Welford runs
    # carry the default ν=1 forward harmlessly).
    for key in ("popart.mu", "popart.sigma", "popart.M2", "popart.count",
                "popart.nu", "popart.W", "popart.b"):
        assert key in sd, f"Missing key {key!r} in state_dict — state would be lost on resume"

    # Build a fresh critic with identical architecture and load.
    critic_dst = _build_critic(task_names, popart_enabled=True)
    critic_dst.load_state_dict(sd, strict=True)
    pop_dst = critic_dst.popart

    for name in ("mu", "sigma", "M2", "count", "nu", "W", "b"):
        src_val = getattr(pop, name).detach()
        dst_val = getattr(pop_dst, name).detach()
        assert torch.equal(src_val, dst_val), f"{name} round-trip mismatch"


# ---------------------------------------------------------------------------
# 5. min_std > 0 fail-fast — guards the per-sample div-by-zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_min_std", [0.0, -1e-3, -1.0])
def test_popart_min_std_must_be_positive(bad_min_std):
    """``PopArtModule`` must reject min_std <= 0 at construction.

    Rationale: if min_std is 0 (or negative), σ_i can stay at 0 on tasks
    whose initial returns happen to have zero variance, and the per-sample
    normalize step in MultiTaskPPO._update_with_batch (insertion B) would
    divide by zero — silently producing NaN gradients that then propagate
    through the critic.  Catching this at the constructor surfaces the bug
    at train.py setup time rather than several minibatches into a run.
    """
    with pytest.raises(ValueError, match="min_std"):
        PopArtModule(num_tasks=3, min_std=bad_min_std)


# ---------------------------------------------------------------------------
# EMA-variant tests (Hessel 2019 reference: per-sample β, default 3e-4)
# ---------------------------------------------------------------------------


def test_popart_ema_default_decay_is_hessel():
    """Default β must remain the Hessel-2019 multi-task default (3e-4)."""
    pop = PopArtModule(num_tasks=3, method="ema")
    assert pop.method == "ema"
    assert pop.ema_decay == 3e-4, (
        f"Default ema_decay drifted from Hessel 2019 (3e-4); got {pop.ema_decay}"
    )


@pytest.mark.parametrize("bad_decay", [0.0, -1e-3, 1.0, 1.5, float("nan")])
def test_popart_ema_construction_validates_decay(bad_decay):
    """``method='ema'`` must reject β outside (0, 1) and NaN at construction."""
    with pytest.raises(ValueError, match="ema_decay"):
        PopArtModule(num_tasks=3, method="ema", ema_decay=bad_decay)


def test_popart_method_must_be_known():
    """Reject unknown method strings at construction (typos surface immediately)."""
    with pytest.raises(ValueError, match="method"):
        PopArtModule(num_tasks=3, method="welfrod")  # typo on purpose


def test_popart_ema_update_matches_sequential_loop():
    """Vectorized closed-form EMA must agree with an explicit per-sample loop."""
    T = 1
    beta = 1e-2  # large enough for the (1-β)^N decay to bite in N=128
    rng = torch.Generator().manual_seed(2024)
    G = (torch.randn(128, generator=rng, dtype=torch.float64) * 3.0 + 5.0)

    # Closed-form via the module.
    pop = PopArtModule(num_tasks=T, min_std=1e-12, method="ema", ema_decay=beta)
    pop.update_and_renormalize({0: G.clone()})
    mu_vec    = float(pop.mu[0].item())
    nu_vec    = float(pop.nu[0].item())
    sigma_vec = float(pop.sigma[0].item())

    # Reference: sequential per-sample EMA in pure Python on fp64.
    mu_ref = 0.0
    nu_ref = 1.0     # matches the buffer init
    for g in G.tolist():
        mu_ref = (1.0 - beta) * mu_ref + beta * g
        nu_ref = (1.0 - beta) * nu_ref + beta * g * g
    sigma_ref = max(nu_ref - mu_ref * mu_ref, 0.0) ** 0.5

    # fp64 sums in either path — tolerance set to ~10x ulp at these magnitudes.
    assert abs(mu_vec    - mu_ref)    < 1e-10, (mu_vec, mu_ref)
    assert abs(nu_vec    - nu_ref)    < 1e-10, (nu_vec, nu_ref)
    assert abs(sigma_vec - sigma_ref) < 1e-10, (sigma_vec, sigma_ref)


def test_popart_ema_pop_invariant_preserves_v_raw():
    """POP rescale must keep V_raw fixed under the EMA update path too."""
    T = 6
    torch.manual_seed(0)
    pop = PopArtModule(num_tasks=T, min_std=1e-4, method="ema", ema_decay=3e-4)
    # Same non-trivial starting state as the Welford test (minus M2/count;
    # for EMA we set ν consistent with σ via ν = σ² + μ²).
    pop.W.data = torch.tensor([1.0, 0.5, 2.0, -0.3, 1.2, 0.8])
    pop.b.data = torch.tensor([0.0, 1.0, -0.5, 0.2, 0.1, -1.0])
    pop.mu.data    = torch.tensor([0.0, 5.0, -2.0, 10.0, 0.3, -0.8], dtype=torch.float64)
    pop.sigma.data = torch.tensor([1.0, 3.0,  0.5,  2.0, 0.1,  4.0], dtype=torch.float64)
    pop.nu.data    = pop.sigma.data.pow(2) + pop.mu.data.pow(2)

    rng = torch.Generator().manual_seed(7)
    v_pre    = torch.randn(40, generator=rng)
    task_idx = torch.randint(0, T, (40,), generator=rng)

    v_raw_old = pop.denormalize(pop.affine(v_pre, task_idx), task_idx).detach().clone()

    returns_per_task: Dict[int, torch.Tensor] = {
        0: torch.randn(50, generator=rng) * 1.0 + 0.5,
        1: torch.randn(100, generator=rng) * 5.0 + 20.0,
        2: torch.randn(30, generator=rng) * 0.2 - 5.0,
        3: torch.randn(200, generator=rng) * 10.0 + 0.0,
        4: torch.randn(20, generator=rng) * 0.05 + 0.3,
        5: torch.randn(80, generator=rng) * 2.0 - 3.0,
    }
    pop.update_and_renormalize(returns_per_task)

    v_raw_new = pop.denormalize(pop.affine(v_pre, task_idx), task_idx).detach().clone()

    max_abs_diff = (v_raw_old - v_raw_new).abs().max().item()
    assert max_abs_diff < 1e-4, (
        f"POP invariant violated under EMA: max|ΔV_raw| = {max_abs_diff:.2e}"
    )


def test_popart_ema_does_not_touch_welford_buffers():
    """EMA path must leave M2/count untouched (they're the Welford state)."""
    pop = PopArtModule(num_tasks=2, method="ema", ema_decay=1e-2)
    M2_before    = pop.M2.clone()
    count_before = pop.count.clone()
    pop.update_and_renormalize({
        0: torch.randn(64, dtype=torch.float64),
        1: torch.randn(64, dtype=torch.float64) * 5.0 + 10.0,
    })
    assert torch.equal(pop.M2,    M2_before),    "EMA path must not mutate M2"
    assert torch.equal(pop.count, count_before), "EMA path must not mutate count"


def test_popart_welford_does_not_touch_nu_buffer():
    """Welford path must leave nu untouched (it's the EMA state)."""
    pop = PopArtModule(num_tasks=2, method="welford")
    nu_before = pop.nu.clone()
    pop.update_and_renormalize({
        0: torch.randn(64, dtype=torch.float64),
        1: torch.randn(64, dtype=torch.float64) * 5.0 + 10.0,
    })
    assert torch.equal(pop.nu, nu_before), "Welford path must not mutate nu"


def test_popart_ema_state_dict_roundtrip_perturbed_nu():
    """EMA-mode round-trip must preserve a non-default ``nu`` exactly.

    The pre-existing Welford round-trip test leaves ``nu`` at its
    construction default (1.0), so ``torch.equal(ones, ones)`` is trivially
    true.  This test drives ν off via real EMA updates so the equality
    check is load-bearing.
    """
    task_names = [f"task_{i}" for i in range(4)]
    critic_src = _build_critic(
        task_names,
        popart_enabled=True,
        popart_method="ema",
        popart_ema_decay=1e-2,  # large β so ν moves visibly in just two collects
    )
    pop = critic_src.popart

    rng = np.random.default_rng(11)
    for _ in range(2):
        batch = {
            i: torch.from_numpy(
                rng.normal((i + 1) * 3.0, (i + 1) * 1.0, size=80).astype(np.float64)
            )
            for i in range(4)
        }
        pop.update_and_renormalize(batch)
    # Sanity: ν must have actually moved off the default 1.0 — otherwise the
    # round-trip check below would be trivially satisfied like the Welford test.
    assert not torch.allclose(pop.nu, torch.ones_like(pop.nu)), (
        "test setup failed: ν did not move off the default 1.0"
    )

    sd = critic_src.state_dict()
    assert "popart.nu" in sd, "popart.nu must be in state_dict for EMA-mode resumes"

    critic_dst = _build_critic(
        task_names,
        popart_enabled=True,
        popart_method="ema",
        popart_ema_decay=1e-2,
    )
    critic_dst.load_state_dict(sd, strict=True)
    pop_dst = critic_dst.popart

    for name in ("mu", "sigma", "nu", "M2", "count", "W", "b"):
        src_val = getattr(pop, name).detach()
        dst_val = getattr(pop_dst, name).detach()
        assert torch.equal(src_val, dst_val), f"EMA-mode {name} round-trip mismatch"


def test_popart_loads_pre_v2_21_checkpoint_without_nu():
    """v2.18-v2.20 PopArt checkpoints (no ``popart.nu``) must resume cleanly.

    Pre-v2.21 PopArt modules registered six tensors per submodule
    (``mu, sigma, M2, count, W, b``).  The strict default of
    ``load_state_dict`` would reject those checkpoints once we add the
    ``nu`` buffer, breaking resume for every in-flight v2.18+ run with
    ``use_popart=true``.  ``PopArtModule._load_from_state_dict`` synthesizes
    ``nu = σ² + μ²`` to plug the gap; this test pins that contract.
    """
    task_names = [f"task_{i}" for i in range(3)]
    # Build a fresh module then drive Welford state off-defaults so μ/σ are
    # non-trivial — that way the synthesized ν is non-trivial too.
    critic = _build_critic(task_names, popart_enabled=True, popart_method="welford")
    rng = np.random.default_rng(42)
    critic.popart.update_and_renormalize({
        i: torch.from_numpy(rng.normal(i * 2.0, (i + 1) * 1.5, size=64).astype(np.float64))
        for i in range(3)
    })

    # Synthesize a "pre-v2.21" state_dict by stripping popart.nu.  This is
    # exactly the shape v2.18-v2.20 checkpoints have on disk.
    sd = critic.state_dict()
    sd_old = {k: v for k, v in sd.items() if k != "popart.nu"}
    assert "popart.nu" not in sd_old
    # Sanity: every other PopArt key is still there.
    for key in ("popart.mu", "popart.sigma", "popart.M2", "popart.count",
                "popart.W", "popart.b"):
        assert key in sd_old

    # Load into a fresh v2.21 module — must not raise under strict=True.
    critic_resumed = _build_critic(task_names, popart_enabled=True, popart_method="welford")
    critic_resumed.load_state_dict(sd_old, strict=True)
    pop_resumed = critic_resumed.popart

    # μ, σ must round-trip exactly from the source.
    assert torch.equal(pop_resumed.mu,    critic.popart.mu)
    assert torch.equal(pop_resumed.sigma, critic.popart.sigma)
    # ν is synthesized; check it equals σ² + μ² of the loaded checkpoint.
    expected_nu = critic.popart.sigma.pow(2) + critic.popart.mu.pow(2)
    assert torch.allclose(pop_resumed.nu, expected_nu, atol=0.0, rtol=0.0), (
        "synthesized ν must equal σ² + μ² exactly (both stored in fp64)"
    )


# ---------------------------------------------------------------------------
# 6. End-to-end smoke — train.py --ppo.use-popart runs 1 epoch
# ---------------------------------------------------------------------------


def test_smoke_one_epoch_popart(tmp_path):
    """Spawn `train.py --ppo.use-popart` on the debug config for 1 epoch.

    Catches plumbing bugs in:
    - train.py validation + MultiTaskCritic / MultiTaskPPO kwarg passing
    - _update_with_batch insertions (A, B, C + raw_returns cache)
    - logger hookup (get_and_reset_popart_diagnostics)
    - gradient-combiner interaction (this run uses default combiners, so
      the popart-enabled path executes the same critic forward as prod)

    This spawns an env vector + MetaWorld and takes about two minutes on
    CPU; deselect it with `pytest -k "not smoke"`.  The run writes only
    under ``tmp_path``, so every invocation trains from scratch instead of
    auto-resuming from a checkpoint an earlier invocation left behind.
    """
    if not DEBUG_CONFIG.exists():
        pytest.skip(f"debug config not found at {DEBUG_CONFIG}")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "train.py"),
        "--config", str(DEBUG_CONFIG),
        "--ppo.use-popart",
        "--training.epoch", "1",
        "--training.step-per-epoch", "4000",
        "--training.step-per-collect", "1000",
        "--logging.no-use-wandb",
        "--logging.no-use-tensorboard",
        "--logging.no-use-csv",
        "--exp-name", "popart_smoke_test",
        "--logging.log-dir", str(tmp_path / "logs"),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "")
    result = subprocess.run(
        cmd, cwd=str(tmp_path), env=env,
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, (
        f"train.py --ppo.use-popart exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )
    assert "Traceback" not in result.stdout and "Traceback" not in result.stderr
    # A resumed run would skip the training loop and pass vacuously.
    assert "Resuming from checkpoint" not in result.stdout
    assert (tmp_path / "logs" / "popart_smoke_test" / "checkpoints" / "latest.pth").exists()
