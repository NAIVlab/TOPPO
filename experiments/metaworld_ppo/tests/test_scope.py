"""Tests for mtrl.gradients.scope — partition + per-side dispatch.

Covers the invariants that matter for correctness across all four
combinations of the per-side dispatch introduced in v2.13.0:

- actor=None, critic=None        → both sides mean-averaged.
- actor=fn,   critic=None        → legacy ``scope=actor`` semantics.
- actor=None, critic=fn          → legacy ``scope=critic`` semantics
                                   (critic wins shared trunk).
- actor=fn_A, critic=fn_B        → per-side mix.  When fn_A == fn_B this
                                   matches legacy ``scope=split``.

Invariants tested:
- ``PartitionedParams`` builds disjoint actor_only / critic_only / shared sets.
- ``extract_task_slices`` + ``reassemble_full_from_slices`` round-trip.
- The actor combiner sees only ``task_actor_slices``; the critic combiner
  sees only ``task_critic_full_slices``.
- When both sides project, actor wins the shared trunk (v2.7.1 convention).
- When only the critic projects, shared-trunk slots in ``combined_full`` and
  in ``task_actor_slices_post`` both come from the critic projection.
- Task-local slots are mean-averaged in ``combined_full`` and pass through
  raw in ``task_full_grads_post``.
- Adaptive uniform-weight fast path: ``task_full_grads_post`` reflects the
  projection (not the raw inputs).
- Mixed-method smoke: different combiners on each side both run and
  write to their own slots.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import pytest
import torch
import torch.nn as nn

from mtrl.gradients.cagrad import cagrad_combine
from mtrl.gradients.fairgrad import fairgrad_combine
from mtrl.gradients.pcgrad import pcgrad_combine
from mtrl.gradients.per_side import validate_per_side_methods
from mtrl.gradients.scope import (
    PartitionedParams,
    apply_per_side,
    critic_only_slice_from_critic_full,
    extract_task_local_slices,
    extract_task_slices,
    flatten_grads,
    reassemble_full_from_slices,
    unflatten_grads,
)


def _make_module(in_dim: int, out_dim: int) -> nn.Linear:
    torch.manual_seed(0)
    m = nn.Linear(in_dim, out_dim, bias=True)
    return m


def _build_partition_shared_trunk() -> Tuple[PartitionedParams, nn.Module, nn.Module, nn.Module]:
    """Build a partition with a shared trunk (actor ∩ critic)."""
    trunk = _make_module(4, 3)
    actor_head = _make_module(3, 2)
    critic_head = _make_module(3, 1)

    actor_params = list(trunk.parameters()) + list(actor_head.parameters())
    critic_params = list(trunk.parameters()) + list(critic_head.parameters())
    all_params = list({id(p): p for p in actor_params + critic_params}.values())

    partition = PartitionedParams.build(
        all_params=all_params,
        actor_params=actor_params,
        critic_params=critic_params,
    )
    return partition, trunk, actor_head, critic_head


def _build_partition_disjoint() -> Tuple[PartitionedParams, nn.Module, nn.Module]:
    """Build a partition with no shared trunk (default mtrl arch)."""
    actor = _make_module(4, 2)
    critic = _make_module(4, 1)
    all_params = list(actor.parameters()) + list(critic.parameters())
    partition = PartitionedParams.build(
        all_params=all_params,
        actor_params=list(actor.parameters()),
        critic_params=list(critic.parameters()),
    )
    return partition, actor, critic


def _build_partition_with_task_local() -> Tuple[PartitionedParams, nn.Parameter, nn.Module, nn.Module]:
    """Partition where `sigma_param` is task-local (baseline semantics).

    Backbone: actor_body + critic_body.  Task-local: a single
    ``nn.Parameter`` (sigma_param) registered on the actor side that is
    NOT included in the backbone list passed to build().  The optimizer,
    however, sees every param.
    """
    torch.manual_seed(42)
    actor_body = nn.Linear(4, 3)
    critic_body = nn.Linear(4, 1)
    sigma_param = nn.Parameter(torch.zeros(10, 2))  # [num_tasks=10, action_dim=2]

    all_params = list(actor_body.parameters()) + list(critic_body.parameters()) + [sigma_param]
    # Backbone only — mirrors MultiTaskPPO._get_actor_backbone_params.
    actor_backbone = list(actor_body.parameters())
    critic_backbone = list(critic_body.parameters())
    partition = PartitionedParams.build(
        all_params=all_params,
        actor_params=actor_backbone,
        critic_params=critic_backbone,
    )
    return partition, sigma_param, actor_body, critic_body


# --- PartitionedParams ------------------------------------------------------


def test_partition_shared_trunk():
    partition, trunk, actor_head, critic_head = _build_partition_shared_trunk()

    trunk_ids = {id(p) for p in trunk.parameters()}
    assert partition.shared_param_ids == trunk_ids
    assert partition.actor_param_ids == trunk_ids | {id(p) for p in actor_head.parameters()}
    assert partition.critic_param_ids == trunk_ids | {id(p) for p in critic_head.parameters()}

    actor_only = partition.actor_param_ids - partition.shared_param_ids
    critic_only = partition.critic_param_ids - partition.shared_param_ids
    assert len(actor_only) + len(critic_only) + len(partition.shared_param_ids) == len(
        partition.all_params
    )


def test_partition_disjoint_no_shared():
    partition, _, _ = _build_partition_disjoint()
    assert partition.shared_param_ids == set()
    assert not (partition.actor_param_ids & partition.critic_param_ids)
    assert partition.task_local_param_ids == set()


def test_partition_task_local_populated():
    """Baseline semantics: params not in actor/critic backbone land in task_local."""
    partition, sigma, *_ = _build_partition_with_task_local()
    assert id(sigma) in partition.task_local_param_ids
    assert id(sigma) not in partition.actor_param_ids
    assert id(sigma) not in partition.critic_param_ids
    assert partition.task_local_total_numel == sigma.numel()


def test_extract_task_local_slices_empty_when_no_task_local():
    partition, *_ = _build_partition_shared_trunk()
    task_grads = [torch.randn(partition.all_total_numel) for _ in range(3)]
    slices = extract_task_local_slices(task_grads, partition)
    assert all(s.numel() == 0 for s in slices)


def test_extract_task_local_slices_correct():
    partition, sigma, *_ = _build_partition_with_task_local()
    torch.manual_seed(7)
    task_grads = [torch.randn(partition.all_total_numel) for _ in range(3)]
    slices = extract_task_local_slices(task_grads, partition)
    assert len(slices) == 3
    sigma_off, sigma_n = partition.all_offsets[id(sigma)]
    for t, s in enumerate(slices):
        assert s.shape[0] == sigma.numel()
        assert torch.allclose(s, task_grads[t][sigma_off : sigma_off + sigma_n], atol=1e-7)


# --- extract_task_slices / reassemble_full_from_slices ----------------------


def test_extract_task_slices_roundtrip_shared_trunk():
    partition, *_ = _build_partition_shared_trunk()
    torch.manual_seed(1)
    full = torch.randn(partition.all_total_numel)
    actor_slices, critic_full_slices = extract_task_slices([full], partition)
    assert actor_slices[0].shape[0] == partition.actor_total_numel
    assert critic_full_slices[0].shape[0] == partition.critic_full_total_numel

    rebuilt = reassemble_full_from_slices(
        actor_slices[0], critic_full_slices[0], partition, shared_from="actor",
    )
    assert torch.allclose(rebuilt, full, atol=1e-7)


def test_critic_only_slice_extraction():
    partition, *_ = _build_partition_shared_trunk()
    torch.manual_seed(2)
    full = torch.randn(partition.all_total_numel)
    _, critic_full = extract_task_slices([full], partition)
    critic_only = critic_only_slice_from_critic_full(critic_full[0], partition)
    expected_numel = sum(p.numel() for p in partition.critic_only_param_seq)
    assert critic_only.shape[0] == expected_numel


# --- apply_per_side ---------------------------------------------------------


def _tagging_combine_fn(tag: float):
    """A combiner that fills its output with `tag` (projected = per-task tag)."""

    def combine_fn(task_grads_in: List[torch.Tensor]):
        combined = torch.full_like(task_grads_in[0], fill_value=tag)
        projected = [combined.clone() for _ in task_grads_in]
        return combined, projected, None

    return combine_fn


def _counting_combine_fn():
    """A combiner that records the shape of what it saw."""
    seen: List[int] = []

    def combine_fn(task_grads_in: List[torch.Tensor]):
        seen.append(task_grads_in[0].shape[0])
        combined = torch.stack(task_grads_in).mean(dim=0)
        return combined, task_grads_in, None

    return combine_fn, seen


def test_apply_per_side_actor_only_sees_actor_view():
    partition, *_ = _build_partition_shared_trunk()
    torch.manual_seed(3)
    task_grads = [torch.randn(partition.all_total_numel) for _ in range(3)]
    combine_fn, seen = _counting_combine_fn()
    apply_per_side(task_grads, partition, combine_fn, None)
    assert seen == [partition.actor_total_numel]


def test_apply_per_side_critic_only_sees_critic_full_view():
    """v2.7.1 P1: critic combiner must see [shared | critic_only], not critic_only alone."""
    partition, *_ = _build_partition_shared_trunk()
    torch.manual_seed(4)
    task_grads = [torch.randn(partition.all_total_numel) for _ in range(3)]
    combine_fn, seen = _counting_combine_fn()
    apply_per_side(task_grads, partition, None, combine_fn)
    assert seen == [partition.critic_full_total_numel]
    assert partition.shared_param_ids, "fixture must have a shared trunk"
    assert partition.critic_full_total_numel > sum(
        p.numel() for p in partition.critic_only_param_seq
    )


def test_apply_per_side_both_sides_call_respective_combiners():
    partition, *_ = _build_partition_shared_trunk()
    torch.manual_seed(5)
    task_grads = [torch.randn(partition.all_total_numel) for _ in range(3)]
    actor_fn, actor_seen = _counting_combine_fn()
    critic_fn, critic_seen = _counting_combine_fn()
    apply_per_side(task_grads, partition, actor_fn, critic_fn)
    assert actor_seen == [partition.actor_total_numel]
    assert critic_seen == [partition.critic_full_total_numel]


def test_apply_per_side_both_projected_actor_wins_shared():
    """Both sides project: shared trunk slots in combined_full come from actor."""
    partition, trunk, *_ = _build_partition_shared_trunk()
    torch.manual_seed(6)
    task_grads = [torch.randn(partition.all_total_numel) for _ in range(3)]

    result = apply_per_side(
        task_grads,
        partition,
        _tagging_combine_fn(1.0),  # actor
        _tagging_combine_fn(2.0),  # critic
    )
    for p in trunk.parameters():
        p_off, n = partition.all_offsets[id(p)]
        slot = result.combined_full_grad[p_off : p_off + n]
        assert torch.all(slot == 1.0), f"shared slot should come from actor, got {slot}"


def test_apply_per_side_only_critic_projects_critic_wins_shared():
    """Critic alone projects: shared slots in combined_full come from critic."""
    partition, trunk, *_ = _build_partition_shared_trunk()
    torch.manual_seed(7)
    task_grads = [torch.randn(partition.all_total_numel) for _ in range(3)]

    result = apply_per_side(
        task_grads,
        partition,
        None,
        _tagging_combine_fn(7.0),
    )
    for p in trunk.parameters():
        p_off, n = partition.all_offsets[id(p)]
        slot = result.combined_full_grad[p_off : p_off + n]
        assert torch.all(slot == 7.0), f"shared slot should come from critic, got {slot}"


def test_apply_per_side_only_critic_absorbs_shared_into_actor_slice():
    """Critic-projected shared slots must also land in task_actor_slices_post."""
    partition, trunk, *_ = _build_partition_shared_trunk()
    torch.manual_seed(8)
    task_grads = [torch.randn(partition.all_total_numel) for _ in range(3)]

    result = apply_per_side(
        task_grads,
        partition,
        None,
        _tagging_combine_fn(7.0),
        produces_per_task=True,
    )
    for t, actor_post in enumerate(result.task_actor_slices_post):
        for p in partition.all_params:
            if id(p) not in partition.actor_param_ids:
                continue
            a_off, n = partition.actor_offsets[id(p)]
            slot = actor_post[a_off : a_off + n]
            if id(p) in partition.shared_param_ids:
                assert torch.all(slot == 7.0), (
                    f"shared slot should be absorbed from critic projection, got {slot}"
                )
            else:
                raw_p_off, _ = partition.all_offsets[id(p)]
                raw = task_grads[t][raw_p_off : raw_p_off + n]
                assert torch.allclose(slot, raw, atol=1e-7)


def test_apply_per_side_task_local_mean_averaged_in_combined_full():
    """Task-local slots in combined_full equal the plain per-task mean under all dispatch modes."""
    partition, sigma, *_ = _build_partition_with_task_local()
    torch.manual_seed(13)
    task_grads = [torch.randn(partition.all_total_numel) for _ in range(3)]

    def identity_combine(tg):
        projected = [g.clone() for g in tg]
        combined = torch.stack(projected).mean(dim=0)
        return combined, projected, None

    modes = [
        (None, None),                        # both passthrough
        (identity_combine, None),            # actor-only
        (None, identity_combine),            # critic-only
        (identity_combine, identity_combine) # both
    ]
    sigma_p_off, sigma_n = partition.all_offsets[id(sigma)]
    for actor_fn, critic_fn in modes:
        result = apply_per_side(task_grads, partition, actor_fn, critic_fn)
        full_sigma_slot = result.combined_full_grad[sigma_p_off : sigma_p_off + sigma_n]
        expected_mean = torch.stack(
            [g[sigma_p_off : sigma_p_off + sigma_n] for g in task_grads],
        ).mean(dim=0)
        assert torch.allclose(full_sigma_slot, expected_mean, atol=1e-6)


def test_apply_per_side_task_local_passthrough_in_per_task_post():
    """task_full_grads_post carries raw per-task task_local values (no projection)."""
    partition, sigma, *_ = _build_partition_with_task_local()
    torch.manual_seed(14)
    task_grads = [torch.randn(partition.all_total_numel) for _ in range(3)]

    def zeroing_combine(tg):
        projected = [torch.zeros_like(g) for g in tg]
        combined = torch.zeros_like(tg[0])
        return combined, projected, None

    sigma_p_off, sigma_n = partition.all_offsets[id(sigma)]
    for actor_fn, critic_fn in (
        (zeroing_combine, None),
        (None, zeroing_combine),
        (zeroing_combine, zeroing_combine),
    ):
        result = apply_per_side(
            task_grads, partition, actor_fn, critic_fn, produces_per_task=True,
        )
        for t, post in enumerate(result.task_full_grads_post):
            slot = post[sigma_p_off : sigma_p_off + sigma_n]
            raw = task_grads[t][sigma_p_off : sigma_p_off + sigma_n]
            assert torch.allclose(slot, raw, atol=1e-7)


def test_apply_per_side_produces_per_task_post_projection():
    """P2: task_full_grads_post reflects the projection (not the raw inputs)."""
    partition, *_ = _build_partition_shared_trunk()
    torch.manual_seed(9)
    task_grads = [torch.randn(partition.all_total_numel) for _ in range(3)]

    def zeroing_combine(tg):
        projected = [torch.zeros_like(g) for g in tg]
        combined = torch.zeros_like(tg[0])
        return combined, projected, None

    result = apply_per_side(
        task_grads, partition, zeroing_combine, None, produces_per_task=True,
    )
    for post, raw in zip(result.task_full_grads_post, task_grads):
        for p in partition.all_params:
            p_off, n = partition.all_offsets[id(p)]
            if id(p) in partition.actor_param_ids:
                assert torch.all(post[p_off : p_off + n] == 0.0)
            else:
                assert torch.allclose(
                    post[p_off : p_off + n], raw[p_off : p_off + n], atol=1e-7
                )


def test_apply_per_side_diagnostics_namespaced_by_side():
    partition, *_ = _build_partition_shared_trunk()
    torch.manual_seed(15)
    task_grads = [torch.randn(partition.all_total_numel) for _ in range(3)]

    def tagged_combine(tag):
        def fn(tg):
            combined = torch.stack(tg).mean(dim=0)
            return combined, [g.clone() for g in tg], {"tag": tag}
        return fn

    result = apply_per_side(task_grads, partition, tagged_combine("A"), tagged_combine("B"))
    assert result.diagnostics == {"actor": {"tag": "A"}, "critic": {"tag": "B"}}

    result_actor_only = apply_per_side(task_grads, partition, tagged_combine("A"), None)
    assert result_actor_only.diagnostics == {"actor": {"tag": "A"}, "critic": None}


# --- Per-side mix with real combiners --------------------------------------


def test_apply_per_side_mixed_methods_pcgrad_cagrad():
    """Different combiners on each side: outputs land in their own slots
    and don't cross-contaminate (actor slot = pcgrad, critic-only slot = cagrad)."""
    partition, trunk, *_ = _build_partition_shared_trunk()
    torch.manual_seed(16)
    task_grads = [torch.randn(partition.all_total_numel) for _ in range(3)]

    def pcgrad_fn(tg):
        return pcgrad_combine(tg)

    def cagrad_fn(tg):
        return cagrad_combine(tg, alpha=0.5)

    # pcgrad_combine shuffles internally; seed `random` before each invocation
    # so the wrapped apply_per_side call and the direct reference call draw
    # the same orders.
    random.seed(42)
    result = apply_per_side(task_grads, partition, pcgrad_fn, cagrad_fn)

    # Shared trunk should match actor-side PCGrad (actor wins).
    actor_slices, critic_full_slices = extract_task_slices(task_grads, partition)
    random.seed(42)
    pc_combined, _, _ = pcgrad_fn(actor_slices)
    for p in trunk.parameters():
        a_off, _ = partition.actor_offsets[id(p)]
        p_off, n = partition.all_offsets[id(p)]
        assert torch.allclose(
            result.combined_full_grad[p_off : p_off + n],
            pc_combined[a_off : a_off + n],
            atol=1e-6,
        )

    # critic-only params should match cagrad_fn(critic_full_slices) on critic_only slots.
    ca_combined, _, _ = cagrad_fn(critic_full_slices)
    for p in partition.critic_only_param_seq:
        cf_off, n = partition.critic_full_offsets[id(p)]
        p_off, _ = partition.all_offsets[id(p)]
        assert torch.allclose(
            result.combined_full_grad[p_off : p_off + n],
            ca_combined[cf_off : cf_off + n],
            atol=1e-6,
        )


def _check_method_matches_direct_split(partition, task_grads, combine_fn_builder):
    """Helper: running `combine_fn_builder()` on both sides via apply_per_side
    must match running it once on each view directly (actor wins shared).

    ``combine_fn_builder()`` must return a fresh combiner each call in case
    the combiner has stateful solver state.
    """
    actor_slices, critic_full_slices = extract_task_slices(task_grads, partition)
    direct_actor, _, _ = combine_fn_builder()(actor_slices)
    direct_critic, _, _ = combine_fn_builder()(critic_full_slices)

    result = apply_per_side(
        task_grads, partition, combine_fn_builder(), combine_fn_builder(),
    )

    for p in partition.all_params:
        p_off, n = partition.all_offsets[id(p)]
        if id(p) in partition.shared_param_ids or id(p) in (
            partition.actor_param_ids - partition.shared_param_ids
        ):
            a_off, _ = partition.actor_offsets[id(p)]
            expected = direct_actor[a_off : a_off + n]
        else:
            cf_off, _ = partition.critic_full_offsets[id(p)]
            expected = direct_critic[cf_off : cf_off + n]
        assert torch.allclose(
            result.combined_full_grad[p_off : p_off + n], expected, atol=1e-5,
        ), f"mismatch on param id={id(p)} numel={n}"


def test_apply_per_side_cagrad_both_sides_matches_direct_split():
    """CAGrad on both sides reproduces two independent CAGrad solves — the
    legacy ``scope=split`` behaviour under the new API."""
    partition, *_ = _build_partition_shared_trunk()
    torch.manual_seed(20)
    task_grads = [torch.randn(partition.all_total_numel) for _ in range(3)]
    _check_method_matches_direct_split(
        partition,
        task_grads,
        combine_fn_builder=lambda: (lambda tg: cagrad_combine(tg, alpha=0.5)),
    )


def test_apply_per_side_fairgrad_both_sides_matches_direct_split():
    """FairGrad on both sides reproduces two independent FairGrad solves."""
    partition, *_ = _build_partition_shared_trunk()
    torch.manual_seed(21)
    task_grads = [torch.randn(partition.all_total_numel) for _ in range(3)]
    _check_method_matches_direct_split(
        partition,
        task_grads,
        combine_fn_builder=lambda: (lambda tg: fairgrad_combine(tg, alpha=1.0)),
    )


def test_apply_per_side_pcgrad_both_sides_matches_direct_split():
    """Running PCGrad on both sides reproduces the two independent PCGrad
    calls over (actor_view, critic_full_view).  This is the legacy
    ``scope=split`` behaviour under the new API."""
    partition, *_ = _build_partition_shared_trunk()
    torch.manual_seed(10)
    task_grads = [torch.randn(partition.all_total_numel) for _ in range(3)]

    random.seed(42)
    actor_slices, critic_full_slices = extract_task_slices(task_grads, partition)
    direct_actor, _, _ = pcgrad_combine(actor_slices)
    direct_critic, _, _ = pcgrad_combine(critic_full_slices)

    random.seed(42)
    result = apply_per_side(
        task_grads,
        partition,
        lambda tg: pcgrad_combine(tg),
        lambda tg: pcgrad_combine(tg),
    )

    # Actor-only params come from direct_actor; critic-only from direct_critic;
    # shared from direct_actor (actor wins).
    for p in partition.all_params:
        p_off, n = partition.all_offsets[id(p)]
        if id(p) in partition.shared_param_ids or id(p) in (
            partition.actor_param_ids - partition.shared_param_ids
        ):
            a_off, _ = partition.actor_offsets[id(p)]
            expected = direct_actor[a_off : a_off + n]
        else:  # critic_only
            cf_off, _ = partition.critic_full_offsets[id(p)]
            expected = direct_critic[cf_off : cf_off + n]
        assert torch.allclose(result.combined_full_grad[p_off : p_off + n], expected, atol=1e-6)


# --- Per-side validator ----------------------------------------------------


def test_validate_per_side_rejects_shared_plus_mismatched_methods():
    partition, *_ = _build_partition_shared_trunk()
    with pytest.raises(ValueError, match="shared params"):
        validate_per_side_methods(partition, "pcgrad", "cagrad")


def test_validate_per_side_allows_matching_methods_with_shared_trunk():
    partition, *_ = _build_partition_shared_trunk()
    validate_per_side_methods(partition, "pcgrad", "pcgrad")  # no raise
    validate_per_side_methods(partition, "cagrad", "cagrad")  # no raise


def test_validate_per_side_allows_default_on_one_side_with_shared_trunk():
    partition, *_ = _build_partition_shared_trunk()
    validate_per_side_methods(partition, "pcgrad", "default")  # no raise
    validate_per_side_methods(partition, "default", "cagrad")  # no raise


def test_validate_per_side_allows_mismatch_when_no_shared_trunk():
    partition, *_ = _build_partition_disjoint()
    validate_per_side_methods(partition, "pcgrad", "cagrad")  # no raise
    validate_per_side_methods(partition, "fairgrad", "pcgrad")  # no raise


def test_validate_per_side_ignores_zero_numel_shared():
    """The default MT10 arch registers a 0-numel dummy shared param on
    SharedPreprocessNet.  That slot can't conflict (nothing to overwrite),
    so mixed methods must still be allowed."""
    dummy = nn.Parameter(torch.empty(0))
    actor_body = nn.Linear(4, 3)
    critic_body = nn.Linear(4, 1)
    # The dummy is registered on both sides (same instance), mirroring
    # train.py passing one SharedPreprocessNet to actor + critic.
    actor_params = [dummy] + list(actor_body.parameters())
    critic_params = [dummy] + list(critic_body.parameters())
    all_params = list({id(p): p for p in actor_params + critic_params}.values())
    partition = PartitionedParams.build(
        all_params=all_params,
        actor_params=actor_params,
        critic_params=critic_params,
    )
    assert id(dummy) in partition.shared_param_ids
    validate_per_side_methods(partition, "pcgrad", "cagrad")  # no raise
    validate_per_side_methods(partition, "fairgrad", "pcgrad")  # no raise


# --- flatten / unflatten round-trip ----------------------------------------


def test_flatten_unflatten_roundtrip():
    m = _make_module(5, 3)
    params = list(m.parameters())
    for p in params:
        p.grad = torch.randn_like(p)
    flat = flatten_grads(params)
    expected_n = sum(p.numel() for p in params)
    assert flat.shape[0] == expected_n

    unflatten_grads(flat * 2.0, params)
    rebuilt = flatten_grads(params)
    assert torch.allclose(rebuilt, flat * 2.0, atol=1e-7)


# --- FairGrad: the eps floor on diag(G) is load-bearing -----------------------
# `eps` is documented as the lower bound on the weights x, but it is also applied
# as a floor on diag(G).  A task with ||g_i||^2 < eps therefore gets its weight
# pinned at 1/sqrt(eps) rather than its exact 1/||g_i||, and the absolute residual
# test accepts that at iteration 0 while reporting convergence.  This is the
# behaviour the published runs were trained with; pin it so it cannot change
# silently.  See the note in mtrl/gradients/fairgrad.py.

def test_fairgrad_eps_floor_caps_tiny_gradient_weights():
    import torch
    from mtrl.gradients.fairgrad import fairgrad_combine

    # two orthogonal tasks, norms 1 and 1e-6 -> ||g_2||^2 = 1e-12 < eps = 1e-8
    grads = [torch.tensor([1.0, 0.0], dtype=torch.float64),
             torch.tensor([0.0, 1e-6], dtype=torch.float64)]
    _, _, diag = fairgrad_combine(grads)
    w = [float(x) for x in diag.weights]

    assert w[0] == pytest.approx(1.0, rel=1e-9)
    # exact alpha=1 answer would be 1e6; the floor pins it at 1/sqrt(1e-8) = 1e4
    assert w[1] == pytest.approx(1e4, rel=1e-9)
    # and the solver reports convergence without taking a step
    assert diag.solver_converged is True
    assert diag.solver_iters == 0


def test_fairgrad_exact_when_above_the_eps_floor():
    import torch
    from mtrl.gradients.fairgrad import fairgrad_combine

    # norms 1 and 1e-3 -> ||g_2||^2 = 1e-6 > eps, so the warm start is exact
    grads = [torch.tensor([1.0, 0.0], dtype=torch.float64),
             torch.tensor([0.0, 1e-3], dtype=torch.float64)]
    _, _, diag = fairgrad_combine(grads)
    w = [float(x) for x in diag.weights]
    assert w[1] == pytest.approx(1e3, rel=1e-9)
    # equal per-task contribution ||w_i g_i||, which is what alpha=1 targets
    assert w[0] * 1.0 == pytest.approx(w[1] * 1e-3, rel=1e-9)
