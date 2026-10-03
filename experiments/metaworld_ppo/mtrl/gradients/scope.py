"""Actor/critic/shared param partitioning + per-side gradient dispatch.

Applies an independently-chosen multi-task combiner (PCGrad / CAGrad /
FairGrad) to each of the actor and critic views.  Either side may opt out
with ``None`` (mean-average passthrough).  On a shared trunk, the actor
wins when both sides project; otherwise the projecting side owns the slot.

Combiner contract::
    task_grads -> (combined, projected_per_task_or_None, diagnostics)

PCGrad returns per-task projected grads; CAGrad and FairGrad collapse to
a weighted sum and return ``None`` for the per-task slot.

Task-local parameters (in ``all_params`` but neither in ``actor_params``
nor ``critic_params``, e.g. per-task heads, ``sigma_param``) bypass the
projection view.  Their combined gradient is the mean across tasks; running
them through a solver would only inflate self-norms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class PartitionedParams:
    """Actor/critic/shared + task-local param partition with flat offsets.

    ``all_params`` is the authoritative iteration order for
    flatten/unflatten.  Each param falls into one of four categories
    (tracked via ``id(p)`` sets):

    - ``actor_only``  : actor_params \\ critic_params
    - ``critic_only`` : critic_params \\ actor_params
    - ``shared``      : actor_params ∩ critic_params  (shared trunk)
    - ``task_local``  : in all_params but in neither view
    """
    all_params: List[nn.Parameter]
    actor_params: List[nn.Parameter]
    critic_params: List[nn.Parameter]
    actor_param_ids: set
    critic_param_ids: set
    shared_param_ids: set
    task_local_param_ids: set
    actor_param_seq: List[nn.Parameter]
    critic_full_param_seq: List[nn.Parameter]
    critic_only_param_seq: List[nn.Parameter]
    task_local_param_seq: List[nn.Parameter]
    all_offsets: Dict[int, Tuple[int, int]]
    actor_offsets: Dict[int, Tuple[int, int]]
    critic_full_offsets: Dict[int, Tuple[int, int]]
    task_local_offsets: Dict[int, Tuple[int, int]]
    actor_total_numel: int
    critic_full_total_numel: int
    task_local_total_numel: int
    all_total_numel: int

    @classmethod
    def build(
        cls,
        all_params: List[nn.Parameter],
        actor_params: List[nn.Parameter],
        critic_params: List[nn.Parameter],
    ) -> "PartitionedParams":
        actor_param_ids = {id(p) for p in actor_params}
        critic_param_ids = {id(p) for p in critic_params}
        shared_param_ids = actor_param_ids & critic_param_ids
        task_local_param_ids = {
            id(p) for p in all_params
            if id(p) not in actor_param_ids and id(p) not in critic_param_ids
        }

        all_offsets, all_total = _offsets(all_params)
        actor_param_seq = [p for p in all_params if id(p) in actor_param_ids]
        actor_offsets, actor_total = _offsets(actor_param_seq)
        critic_full_param_seq = [p for p in all_params if id(p) in critic_param_ids]
        critic_full_offsets, critic_full_total = _offsets(critic_full_param_seq)
        critic_only_param_seq = [
            p for p in critic_full_param_seq if id(p) not in shared_param_ids
        ]
        task_local_param_seq = [p for p in all_params if id(p) in task_local_param_ids]
        task_local_offsets, task_local_total = _offsets(task_local_param_seq)

        return cls(
            all_params=all_params,
            actor_params=actor_params,
            critic_params=critic_params,
            actor_param_ids=actor_param_ids,
            critic_param_ids=critic_param_ids,
            shared_param_ids=shared_param_ids,
            task_local_param_ids=task_local_param_ids,
            actor_param_seq=actor_param_seq,
            critic_full_param_seq=critic_full_param_seq,
            critic_only_param_seq=critic_only_param_seq,
            task_local_param_seq=task_local_param_seq,
            all_offsets=all_offsets,
            actor_offsets=actor_offsets,
            critic_full_offsets=critic_full_offsets,
            task_local_offsets=task_local_offsets,
            actor_total_numel=actor_total,
            critic_full_total_numel=critic_full_total,
            task_local_total_numel=task_local_total,
            all_total_numel=all_total,
        )


def _offsets(seq: List[nn.Parameter]) -> Tuple[Dict[int, Tuple[int, int]], int]:
    off: Dict[int, Tuple[int, int]] = {}
    cur = 0
    for p in seq:
        off[id(p)] = (cur, p.numel())
        cur += p.numel()
    return off, cur


# --- Flatten / unflatten helpers ---------------------------------------------


def flatten_grads(params: List[nn.Parameter]) -> torch.Tensor:
    """Flatten .grad of each param into a single vector; zero-fill missing grads."""
    if not params:
        return torch.zeros(0)
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.data.reshape(-1))
        else:
            grads.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
    return torch.cat(grads)


def unflatten_grads(flat_grad: torch.Tensor, params: List[nn.Parameter]) -> None:
    """Write slices of `flat_grad` back onto each param's .grad."""
    offset = 0
    for p in params:
        numel = p.numel()
        grad_slice = flat_grad[offset : offset + numel].view_as(p.data)
        if p.grad is not None:
            p.grad.data = grad_slice
        else:
            p.grad = grad_slice.clone()
        offset += numel


def flatten_autograd_grads(
    grads: List[Optional[torch.Tensor]],
    params: List[nn.Parameter],
) -> torch.Tensor:
    """Flatten outputs of ``torch.autograd.grad(..., allow_unused=True)``.

    Zero-fills positions where autograd returned ``None`` (param not in the
    graph).  Used by callers that compute per-task gradients via autograd
    rather than ``loss.backward()``.
    """
    out = []
    for g, p in zip(grads, params):
        if g is None:
            out.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        else:
            out.append(g.reshape(-1))
    return torch.cat(out) if out else torch.zeros(0)


# --- Per-task slice extraction / reassembly ----------------------------------


def extract_task_slices(
    task_full_grads: List[torch.Tensor],
    partition: PartitionedParams,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Build actor-view and critic-full-view slices from per-task full grads."""
    actor_slices: List[torch.Tensor] = []
    critic_full_slices: List[torch.Tensor] = []
    for flat_full in task_full_grads:
        a_parts, c_parts = [], []
        for p in partition.all_params:
            off, n = partition.all_offsets[id(p)]
            part = flat_full[off : off + n]
            if id(p) in partition.actor_param_ids:
                a_parts.append(part)
            if id(p) in partition.critic_param_ids:
                c_parts.append(part)
        actor_slices.append(
            torch.cat(a_parts)
            if a_parts
            else torch.zeros(0, device=flat_full.device, dtype=flat_full.dtype)
        )
        critic_full_slices.append(
            torch.cat(c_parts)
            if c_parts
            else torch.zeros(0, device=flat_full.device, dtype=flat_full.dtype)
        )
    return actor_slices, critic_full_slices


def extract_task_local_slices(
    task_full_grads: List[torch.Tensor],
    partition: PartitionedParams,
) -> List[torch.Tensor]:
    """Per-task slices for task-local params; zero-length when none exist."""
    if not partition.task_local_param_seq:
        return [
            torch.zeros(0, device=g.device, dtype=g.dtype) for g in task_full_grads
        ]
    out: List[torch.Tensor] = []
    for flat_full in task_full_grads:
        parts = []
        for p in partition.task_local_param_seq:
            off, n = partition.all_offsets[id(p)]
            parts.append(flat_full[off : off + n])
        out.append(torch.cat(parts))
    return out


def critic_only_slice_from_critic_full(
    critic_full_slice: torch.Tensor,
    partition: PartitionedParams,
) -> torch.Tensor:
    """Extract the critic_only portion (shared excluded) from a critic-full slice."""
    if not partition.critic_only_param_seq:
        return torch.zeros(
            0, device=critic_full_slice.device, dtype=critic_full_slice.dtype,
        )
    parts = []
    for p in partition.critic_only_param_seq:
        off, n = partition.critic_full_offsets[id(p)]
        parts.append(critic_full_slice[off : off + n])
    return torch.cat(parts)


def actor_critic_grad_norms_from_params(
    partition: PartitionedParams,
) -> Tuple[float, float]:
    """L2 norms of actor-view and critic-only ``param.grad`` slices.

    Slicing: actor_view = actor_only + shared trunk; critic_only = critic
    head only (shared trunk excluded).  Sums on-device and converts with a
    single ``.item()`` per side to keep the GPU pipeline non-blocking.
    """
    actor_sq = None
    for p in partition.actor_param_seq:
        if p.grad is not None:
            sq = p.grad.detach().pow(2).sum()
            actor_sq = sq if actor_sq is None else actor_sq + sq
    critic_sq = None
    for p in partition.critic_only_param_seq:
        if p.grad is not None:
            sq = p.grad.detach().pow(2).sum()
            critic_sq = sq if critic_sq is None else critic_sq + sq
    actor_norm = float(actor_sq.sqrt().item()) if actor_sq is not None else 0.0
    critic_norm = float(critic_sq.sqrt().item()) if critic_sq is not None else 0.0
    return actor_norm, critic_norm


def override_actor_shared_from_critic_full(
    actor_slice: torch.Tensor,
    critic_full_slice: torch.Tensor,
    partition: PartitionedParams,
) -> torch.Tensor:
    """Overwrite shared-trunk slots of ``actor_slice`` with critic-projected values.

    Used when the critic side projects and the actor side passes through,
    so downstream consumers reading the actor view see the same shared
    gradients that ``combined_full`` applies.  No-op without a shared trunk.
    """
    if not partition.shared_param_ids:
        return actor_slice
    out = actor_slice.clone()
    for pid in partition.shared_param_ids:
        a_off, a_n = partition.actor_offsets[pid]
        cf_off, cf_n = partition.critic_full_offsets[pid]
        out[a_off : a_off + a_n] = critic_full_slice[cf_off : cf_off + cf_n]
    return out


def reassemble_full_from_slices(
    actor_slice: torch.Tensor,
    critic_full_slice: torch.Tensor,
    partition: PartitionedParams,
    shared_from: Literal["actor", "critic"] = "actor",
    task_local_slice: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Rebuild a full flat gradient from actor, critic-full, and task-local slices.

    Shared-trunk params take their value from ``shared_from``.  When the
    partition has task-local params, ``task_local_slice`` is required.
    """
    ref = (
        actor_slice if actor_slice.numel()
        else critic_full_slice if critic_full_slice.numel()
        else (task_local_slice if task_local_slice is not None else actor_slice)
    )
    out = torch.empty(
        partition.all_total_numel, device=ref.device, dtype=ref.dtype,
    )
    need_task_local = bool(partition.task_local_param_ids)
    if need_task_local and task_local_slice is None:
        raise ValueError(
            "partition has task_local params; task_local_slice must be provided."
        )
    for p in partition.all_params:
        p_off, n = partition.all_offsets[id(p)]
        pid = id(p)
        if pid in partition.shared_param_ids:
            if shared_from == "actor":
                src_off, _ = partition.actor_offsets[pid]
                out[p_off : p_off + n] = actor_slice[src_off : src_off + n]
            else:
                src_off, _ = partition.critic_full_offsets[pid]
                out[p_off : p_off + n] = critic_full_slice[src_off : src_off + n]
        elif pid in partition.actor_param_ids:
            src_off, _ = partition.actor_offsets[pid]
            out[p_off : p_off + n] = actor_slice[src_off : src_off + n]
        elif pid in partition.critic_param_ids:
            src_off, _ = partition.critic_full_offsets[pid]
            out[p_off : p_off + n] = critic_full_slice[src_off : src_off + n]
        else:  # task_local
            src_off, _ = partition.task_local_offsets[pid]
            out[p_off : p_off + n] = task_local_slice[src_off : src_off + n]  # type: ignore[index]
    return out


# --- Combiner contract + scope dispatcher ------------------------------------


CombineFn = Callable[
    [List[torch.Tensor]],
    Tuple[torch.Tensor, Optional[List[torch.Tensor]], Any],
]
"""Pure combiner: ``task_grads -> (combined, projected_per_task_or_None, diag)``.

PCGrad returns per-task projected grads; CAGrad / FairGrad return ``None``.
"""


@dataclass
class ScopeResult:
    """Outputs of :func:`apply_per_side`.

    - ``combined_full_grad``: flat tensor to apply to the optimizer.
    - ``task_full_grads_post``: per-task full flat grads after projection;
      for passthrough sides these carry the raw input values.
    - ``task_actor_slices_post`` / ``task_critic_full_slices_post``: per-task
      view slices after projection.  When the critic projected alone, the
      projected shared slots are absorbed back into the actor view here.
    - ``diagnostics``: ``{"actor": ..., "critic": ...}`` (slot is ``None``
      when that side passed through).
    """
    combined_full_grad: torch.Tensor
    task_full_grads_post: List[torch.Tensor]
    task_actor_slices_post: List[torch.Tensor]
    task_critic_full_slices_post: List[torch.Tensor]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def apply_per_side(
    task_full_grads: List[torch.Tensor],
    partition: PartitionedParams,
    actor_combine_fn: Optional[CombineFn],
    critic_combine_fn: Optional[CombineFn],
    *,
    produces_per_task: bool = True,
) -> ScopeResult:
    """Dispatch a per-side gradient combine.

    Either combine fn may be ``None`` to mean-average that side.  On a
    shared trunk the actor wins when both project; otherwise the projecting
    side owns the slot.  Set ``produces_per_task=False`` to skip rebuilding
    ``task_full_grads_post``.
    """
    task_actor_slices, task_critic_full_slices = extract_task_slices(
        task_full_grads, partition,
    )
    # Task-local params are mean-averaged for combined_full and passed
    # through per-task in task_full_grads_post.
    task_local_slices = extract_task_local_slices(task_full_grads, partition)
    has_task_local = bool(partition.task_local_param_ids)
    if has_task_local and task_local_slices:
        combined_task_local = torch.stack(task_local_slices).mean(dim=0)
    else:
        ref = task_full_grads[0] if task_full_grads else torch.zeros(0)
        combined_task_local = torch.zeros(
            partition.task_local_total_numel,
            device=ref.device, dtype=ref.dtype,
        )

    if actor_combine_fn is not None:
        combined_actor, projected_actor, da = actor_combine_fn(task_actor_slices)
        actor_projected = True
    else:
        projected_actor = None
        da = None
        actor_projected = False
        if task_actor_slices:
            combined_actor = torch.stack(task_actor_slices).mean(dim=0)
        else:
            ref = task_full_grads[0] if task_full_grads else torch.zeros(0)
            combined_actor = torch.zeros(
                partition.actor_total_numel, device=ref.device, dtype=ref.dtype,
            )

    if critic_combine_fn is not None:
        combined_critic_full, projected_critic_full, dc = critic_combine_fn(
            task_critic_full_slices,
        )
        critic_projected = True
    else:
        projected_critic_full = None
        dc = None
        critic_projected = False
        if task_critic_full_slices:
            combined_critic_full = torch.stack(task_critic_full_slices).mean(dim=0)
        else:
            combined_critic_full = torch.zeros(
                partition.critic_full_total_numel,
                device=combined_actor.device, dtype=combined_actor.dtype,
            )

    # Shared-trunk ownership: critic wins only when critic alone projected.
    shared_from: Literal["actor", "critic"] = (
        "critic" if (critic_projected and not actor_projected) else "actor"
    )
    combined_full = reassemble_full_from_slices(
        combined_actor, combined_critic_full, partition,
        shared_from=shared_from, task_local_slice=combined_task_local,
    )

    task_actor_slices_post = (
        projected_actor if projected_actor is not None else task_actor_slices
    )
    task_critic_full_slices_post = (
        projected_critic_full
        if projected_critic_full is not None
        else task_critic_full_slices
    )
    if shared_from == "critic" and partition.shared_param_ids:
        # Absorb critic-projected shared slots back into the actor view.
        task_actor_slices_post = [
            override_actor_shared_from_critic_full(a, c, partition)
            for a, c in zip(task_actor_slices_post, task_critic_full_slices_post)
        ]

    if produces_per_task:
        task_full_grads_post = [
            reassemble_full_from_slices(
                a, c, partition, shared_from=shared_from, task_local_slice=tl,
            )
            for a, c, tl in zip(
                task_actor_slices_post, task_critic_full_slices_post, task_local_slices,
            )
        ]
    else:
        task_full_grads_post = task_full_grads

    return ScopeResult(
        combined_full_grad=combined_full,
        task_full_grads_post=task_full_grads_post,
        task_actor_slices_post=task_actor_slices_post,
        task_critic_full_slices_post=task_critic_full_slices_post,
        diagnostics={"actor": da, "critic": dc},
    )
