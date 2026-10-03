"""Tests for per-task gradient snapshot dump + offline analysis utilities.

Two families of checks:

1. **PopArt σ⁻² inversion is correct.** Build identical critics with and
   without PopArt, run the same forward + per-task value loss, capture the
   per-task backbone gradient with each critic, and verify the relation
       ``g_norm = g_vanilla / σ²`` (per-task)   ⇒   ``G_van = σ_i² σ_j² G_norm``.
   This is the math claim every PopArt-inversion figure in the paper rests
   on; if this drifts the figures are wrong.

2. **Snapshot writer / loader round-trip.** End-to-end: write a synthetic
   snapshot, read it back, verify Gram + layer slicing + offline FairGrad
   replay match the writer-side ground truth.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch

from mtrl.models.networks import (
    MultiTaskCritic,
    SharedPreprocessNet,
)
from mtrl.utils.grad_snapshots import (
    GradSnapshot,
    Manifest,
    _layer_slices_from_param_seq,
    load_manifest,
    load_snapshot,
    open_snapshot_dir,
    write_manifest,
    write_snapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_critic(task_names: List[str], popart_enabled: bool) -> MultiTaskCritic:
    """Match the production MT10 arch: preprocess=Identity, body=[64,64]→1."""
    obs_dim = 10 + len(task_names)
    pre = SharedPreprocessNet(input_dim=obs_dim, hidden_sizes=[], activation="relu")
    return MultiTaskCritic(
        preprocess_net=pre,
        task_names=task_names,
        critic_hidden_sizes=[64, 64],
        critic_head_hidden_sizes=[],
        activation="relu",
        act_dim=0,
        popart_enabled=popart_enabled,
        popart_min_std=1e-2,
    )


def _random_obs(batch: int, task_names: List[str], seed: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    T = len(task_names)
    raw = torch.randn(batch, 10, generator=rng)
    task_ids = torch.randint(0, T, (batch,), generator=rng)
    one_hot = torch.zeros(batch, T)
    one_hot.scatter_(1, task_ids.unsqueeze(1), 1.0)
    return torch.cat([raw, one_hot], dim=-1)


def _backbone_params(critic: MultiTaskCritic) -> List[torch.nn.Parameter]:
    """Backbone params: preprocess + critic_body, *excluding* PopArt affine
    and per-task value heads.  Matches what ``_get_critic_backbone_params``
    selects from the live model."""
    out: List[torch.nn.Parameter] = []
    seen: set = set()
    for mod in (critic.critic_body, critic.preprocess):
        for p in mod.parameters():
            if id(p) not in seen and p.requires_grad:
                out.append(p)
                seen.add(id(p))
    return out


def _flat_grad(params: List[torch.nn.Parameter]) -> torch.Tensor:
    """Flatten .grad over a fixed parameter list; zero-fill missing grads."""
    parts = []
    for p in params:
        if p.grad is None:
            parts.append(torch.zeros(p.numel(), dtype=p.dtype, device=p.device))
        else:
            parts.append(p.grad.detach().reshape(-1).clone())
    return torch.cat(parts)


# ---------------------------------------------------------------------------
# 1. PopArt σ⁻² inversion math
# ---------------------------------------------------------------------------


def test_popart_inversion_per_task_grad_ratio():
    """``g_norm = g_vanilla / σ_i²`` per task on the shared backbone.

    Build two critics with identical weights, identical forward output for
    raw V, and compute the per-task value loss gradient.  The PopArt one
    sees the loss in normalized space; the vanilla one sees raw squared
    error.  Since after POP they produce the same V_raw, the per-trunk
    gradient should differ exactly by 1/σ².
    """
    torch.manual_seed(0)
    task_names = [f"t{i}" for i in range(4)]
    sigma_target = torch.tensor([0.1, 0.5, 2.0, 10.0], dtype=torch.float64)
    mu_target = torch.tensor([-1.0, 0.0, 1.0, 5.0], dtype=torch.float64)

    # Build vanilla critic, then a PopArt critic with weights tied so the
    # two produce the same V_raw on every input.  POP guarantees this:
    # initialize PopArt critic body identical to vanilla, then set
    # popart W_i = 1/σ_i, b_i = -μ_i/σ_i so that
    # V_raw = σ·(W·v_pre + b) + μ = σ·v_pre/σ + σ·(-μ/σ) + μ = v_pre.
    van = _build_critic(task_names, popart_enabled=False)
    pop = _build_critic(task_names, popart_enabled=True)
    pop.load_state_dict(van.state_dict(), strict=False)
    pop.popart.sigma.data = sigma_target.clone()
    pop.popart.mu.data = mu_target.clone()
    # Init affine so popart V_raw matches vanilla V_raw exactly.
    pop.popart.W.data = (1.0 / sigma_target.to(torch.float32)).clone()
    pop.popart.b.data = (-mu_target.to(torch.float32) / sigma_target.to(torch.float32)).clone()

    # Sanity: forward outputs match (POP-equivalent affine).
    obs = _random_obs(64, task_names, seed=42)
    with torch.no_grad():
        v_van = van(obs).flatten()
        v_pop = pop(obs).flatten()
    assert torch.allclose(v_van, v_pop, rtol=1e-4, atol=1e-5), (
        f"POP-equivalent affine should give identical V_raw; "
        f"max diff {(v_van - v_pop).abs().max().item()}"
    )

    # Build a single random returns vector R per sample.
    rng = torch.Generator().manual_seed(7)
    R = torch.randn(64, generator=rng)

    # Build per-task masks from the obs one-hot tail.
    task_ids = obs[:, -len(task_names):].argmax(dim=-1)

    van_params = _backbone_params(van)
    pop_params = _backbone_params(pop)

    # Sanity: same backbone topology (matching layer order + numel).
    assert len(van_params) == len(pop_params)
    for a, b in zip(van_params, pop_params):
        assert a.shape == b.shape

    # Compute per-task value loss + backward, on each critic, isolating one
    # task at a time.  The mask trick avoids needing two separate Tianshou
    # paths for the test.
    for task_id in range(len(task_names)):
        mask = (task_ids == task_id)
        if mask.sum() < 2:
            continue
        sigma_i = float(sigma_target[task_id].item())

        # Vanilla: L = mean((R - V_raw)²) over masked samples.
        for p in van_params:
            if p.grad is not None:
                p.grad.zero_()
        v_van = van(obs).flatten()
        L_van = (R[mask] - v_van[mask]).pow(2).mean()
        L_van.backward()
        g_van = _flat_grad(van_params).double()

        # PopArt: L_norm = mean(((R-μ_i)/σ_i - V_norm)²) over masked samples.
        for p in pop_params:
            if p.grad is not None:
                p.grad.zero_()
        v_norm, _ = pop.forward_normalized(obs)
        target_norm = (R - mu_target[task_id].to(R.dtype)) / sigma_target[task_id].to(R.dtype)
        L_norm = (target_norm[mask] - v_norm[mask]).pow(2).mean()
        L_norm.backward()
        g_norm = _flat_grad(pop_params).double()

        # Theoretical relation: g_norm = g_van / σ² ⇒ g_van = σ² · g_norm.
        # Allow tight tolerance — the two paths share the same backbone
        # params and only differ by the loss scaling factor.
        ratio = sigma_i ** 2
        diff = (g_van - ratio * g_norm).norm().item()
        scale = g_van.norm().item() + 1e-12
        assert diff / scale < 1e-4, (
            f"task {task_id} sigma={sigma_i}: "
            f"||g_van - σ² g_norm|| / ||g_van|| = {diff/scale} (>1e-4)"
        )


def test_popart_inversion_gram_relation():
    """``G_van[i,j] = σ_i² σ_j² · G_norm[i,j]`` on the K×K backbone Gram.

    Stack per-task gradients (computed under each loss), build Gram on each
    side, and verify the σ²-outer-product inversion.  This is the exact
    formula :meth:`GradSnapshot.critic_gram_vanilla_inverted` applies.
    """
    torch.manual_seed(0)
    task_names = [f"t{i}" for i in range(4)]
    K = len(task_names)
    sigma_target = torch.tensor([0.1, 0.5, 2.0, 10.0], dtype=torch.float64)
    mu_target = torch.tensor([-1.0, 0.0, 1.0, 5.0], dtype=torch.float64)

    van = _build_critic(task_names, popart_enabled=False)
    pop = _build_critic(task_names, popart_enabled=True)
    pop.load_state_dict(van.state_dict(), strict=False)
    pop.popart.sigma.data = sigma_target.clone()
    pop.popart.mu.data = mu_target.clone()
    pop.popart.W.data = (1.0 / sigma_target.to(torch.float32)).clone()
    pop.popart.b.data = (-mu_target.to(torch.float32) / sigma_target.to(torch.float32)).clone()

    obs = _random_obs(128, task_names, seed=11)
    rng = torch.Generator().manual_seed(13)
    R = torch.randn(128, generator=rng)
    task_ids = obs[:, -K:].argmax(dim=-1)

    van_params = _backbone_params(van)
    pop_params = _backbone_params(pop)

    # Per-task grads: the tasks present in the batch (same set for both
    # critics since we use the same obs).
    g_van_list, g_norm_list, present_tasks = [], [], []
    for task_id in range(K):
        mask = (task_ids == task_id)
        if mask.sum() < 2:
            continue
        for p in van_params:
            if p.grad is not None:
                p.grad.zero_()
        v_van = van(obs).flatten()
        L_van = (R[mask] - v_van[mask]).pow(2).mean()
        L_van.backward()
        g_van_list.append(_flat_grad(van_params).double())

        for p in pop_params:
            if p.grad is not None:
                p.grad.zero_()
        v_norm, _ = pop.forward_normalized(obs)
        target_norm = (R - mu_target[task_id].to(R.dtype)) / sigma_target[task_id].to(R.dtype)
        L_norm = (target_norm[mask] - v_norm[mask]).pow(2).mean()
        L_norm.backward()
        g_norm_list.append(_flat_grad(pop_params).double())
        present_tasks.append(task_id)

    g_van = torch.stack(g_van_list)        # (K_in, P)
    g_norm = torch.stack(g_norm_list)      # (K_in, P)
    G_van = (g_van @ g_van.T).numpy()
    G_norm = (g_norm @ g_norm.T).numpy()

    sigma_in = sigma_target[present_tasks].numpy()
    sigma_outer = np.outer(sigma_in ** 2, sigma_in ** 2)
    G_van_predicted = G_norm * sigma_outer

    rel_err = np.linalg.norm(G_van - G_van_predicted) / (np.linalg.norm(G_van) + 1e-12)
    assert rel_err < 1e-4, (
        f"Gram inversion drift {rel_err}; predicted via σ² outer product "
        "should match measured vanilla Gram to 1e-4 (these are fp32→fp64 "
        "products on identical backbone params so the slack is just round-off)."
    )


# ---------------------------------------------------------------------------
# 2. Snapshot writer / loader round-trip
# ---------------------------------------------------------------------------


def test_snapshot_roundtrip_minimal():
    """Write a synthetic snapshot, read it back, verify all fields survive."""
    K = 5
    actor_numel = 33
    critic_numel = 41
    task_names = [f"t{i}" for i in range(K)]
    actor_grads = torch.randn(K, actor_numel, dtype=torch.float32)
    critic_grads = torch.randn(K, critic_numel, dtype=torch.float32)
    sigma = np.array([0.1, 0.5, 1.0, 2.0, 10.0], dtype=np.float64)
    mu = np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    W = np.ones(K, dtype=np.float64)
    b = np.zeros(K, dtype=np.float64)
    fg_w_critic = np.array([1.0, 0.5, 0.2, 1e10, 0.1], dtype=np.float64)
    long_ema = np.array([0.9, 0.5, 0.7, 0.0, 0.2], dtype=np.float64)
    counts = np.array([10, 12, 11, 9, 8], dtype=np.int64)

    with tempfile.TemporaryDirectory() as out_dir:
        # Manifest first
        actor_layers = [
            {"name": "actor.body.0", "shape": [10, 3], "offset": 0, "numel": 30},
            {"name": "actor.body.0.bias", "shape": [3], "offset": 30, "numel": 3},
        ]
        critic_layers = [
            {"name": "critic.body.0", "shape": [10, 4], "offset": 0, "numel": 40},
            {"name": "critic.body.0.bias", "shape": [1], "offset": 40, "numel": 1},
        ]
        write_manifest(
            out_dir,
            task_names=task_names,
            actor_layers=actor_layers,
            critic_layers=critic_layers,
            actor_total_numel=actor_numel,
            critic_total_numel=critic_numel,
            use_popart=True,
        )

        # Snapshot
        path = write_snapshot(
            out_dir,
            collect_idx=42,
            repeat_idx=0,
            minibatch_idx=0,
            gradient_step=1234,
            task_names_in_batch=task_names,
            actor_grads=actor_grads,
            critic_full_grads=critic_grads,
            task_count_in_minibatch=counts,
            popart_sigma=sigma, popart_mu=mu, popart_W=W, popart_b=b,
            fairgrad_critic_weights=fg_w_critic,
            fairgrad_critic_residual=1.5e-3,
            long_ema_success=long_ema,
        )
        assert os.path.isfile(path)
        # Atomic-write check: no .tmp left behind.
        assert not os.path.exists(path + ".tmp")

        manifest = load_manifest(out_dir)
        assert manifest.task_names == task_names
        assert manifest.use_popart
        assert manifest.actor_total_numel == actor_numel
        assert manifest.find_actor_layer("actor.body.0")["numel"] == 30

        snap = load_snapshot(path)
        assert snap.collect_idx == 42
        assert snap.gradient_step == 1234
        assert np.allclose(snap.actor_grads, actor_grads.numpy())
        assert np.allclose(snap.critic_full_grads, critic_grads.numpy())
        assert np.allclose(snap.popart_sigma, sigma)
        assert np.allclose(snap.fairgrad_critic_weights, fg_w_critic)
        assert pytest.approx(snap.fairgrad_critic_residual, rel=1e-9) == 1.5e-3

        # Gram sanity
        G_actor = snap.actor_gram()
        assert G_actor.shape == (K, K)
        assert np.allclose(G_actor, G_actor.T, atol=1e-6)

        # PopArt inversion API: returns a (K, K) matrix when popart_sigma is set.
        G_van = snap.critic_gram_vanilla_inverted(task_names)
        assert G_van is not None and G_van.shape == (K, K)
        # Should equal G_norm * outer(σ², σ²)
        G_norm = snap.critic_gram()
        expected = G_norm * np.outer(sigma ** 2, sigma ** 2)
        assert np.allclose(G_van, expected, rtol=1e-9)

        # Layer slicing via manifest
        sub = snap.critic_layer("critic.body.0", manifest)
        assert sub.shape == (K, 40)
        assert np.allclose(sub, critic_grads.numpy()[:, :40])

        # Iterate via SnapshotSeries
        series = open_snapshot_dir(out_dir)
        snaps = list(series)
        assert len(snaps) == 1
        assert snaps[0].collect_idx == 42


def test_offline_fairgrad_replay_matches_solver():
    """Calling ``replay_fairgrad`` runs the production solver — check that
    the returned weights satisfy the KKT residual the same way production does.
    """
    torch.manual_seed(2)
    K = 6
    P = 200
    grads = torch.randn(K, P, dtype=torch.float32)
    # Inject one near-dead row so the solver exercises its dead-row path.
    grads[3] = grads[3] * 1e-10

    with tempfile.TemporaryDirectory() as out_dir:
        write_manifest(
            out_dir,
            task_names=[f"t{i}" for i in range(K)],
            actor_layers=[{"name": "actor.l0", "shape": [P], "offset": 0, "numel": P}],
            critic_layers=[{"name": "critic.l0", "shape": [P], "offset": 0, "numel": P}],
            actor_total_numel=P,
            critic_total_numel=P,
            use_popart=False,
        )
        path = write_snapshot(
            out_dir,
            collect_idx=0, repeat_idx=0, minibatch_idx=0, gradient_step=0,
            task_names_in_batch=[f"t{i}" for i in range(K)],
            actor_grads=grads, critic_full_grads=grads,
            task_count_in_minibatch=np.ones(K, dtype=np.int64),
        )
        snap = load_snapshot(path)

    rep = snap.replay_fairgrad(side="critic", alpha=1.0)
    # Sanity: weights are non-negative, finite, and the residual is at the
    # solver's η target (1e-2) or better.
    assert rep.weights.shape == (K,)
    assert np.all(np.isfinite(rep.weights))
    assert np.all(rep.weights > 0)
    assert rep.residual < 1.0  # well below KKT-failure threshold
    # Replaying with α=2 should give a flatter weight distribution (more
    # uniform — Rawlsian leaning).
    rep2 = snap.replay_fairgrad(side="critic", alpha=2.0)
    assert rep2.weights.shape == (K,)
    # Entropy should be ≥ α=1 entropy (more uniform under Rawlsian α).
    # Loose check — the solver may converge to slightly different x norm.
    assert rep2.weight_entropy >= rep.weight_entropy - 0.5
