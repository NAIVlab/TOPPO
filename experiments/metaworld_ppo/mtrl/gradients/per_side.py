"""Per-side gradient optimizer wrapper.

Runs one backward pass per task, then applies a (possibly different)
multi-task gradient combiner to each of the actor and critic views via
:func:`mtrl.gradients.scope.apply_per_side`.  Diagnostics are namespaced
``{method}_{actor,critic}``; ``default`` sides emit none.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from mtrl.gradients.cagrad import cagrad_combine
from mtrl.gradients.fairgrad import fairgrad_combine
from mtrl.gradients.pcgrad import pcgrad_combine
from mtrl.gradients.scope import (
    CombineFn,
    PartitionedParams,
    actor_critic_grad_norms_from_params,
    apply_per_side,
    flatten_grads,
    unflatten_grads,
)


VALID_METHODS: frozenset[str] = frozenset(
    {"default", "pcgrad", "cagrad", "fairgrad"}
)


def validate_per_side_methods(
    partition: PartitionedParams,
    actor_method: str,
    critic_method: str,
) -> None:
    """Reject per-side method combinations that conflict on a shared trunk.

    On a shared trunk the actor side owns the slot when both project.  If
    the two sides run DIFFERENT non-default combiners on a trunk with real
    parameters, the critic's output is silently discarded — we raise to
    surface this.  Zero-numel shared params are ignored.
    """
    if actor_method == "default" or critic_method == "default":
        return
    if actor_method == critic_method:
        return
    shared_numel = sum(
        p.numel() for p in partition.all_params if id(p) in partition.shared_param_ids
    )
    if shared_numel == 0:
        return
    raise ValueError(
        f"actor_method={actor_method!r} != critic_method={critic_method!r} "
        f"but the gradient partition has {shared_numel} non-trivial shared "
        f"elements across {len(partition.shared_param_ids)} shared params "
        "(actor ∩ critic). Under per-side projection, only the actor side's "
        "output is kept for shared slots, so the critic combiner's handling "
        "of shared params would be silently discarded. Fix by either "
        "(a) setting actor_method == critic_method, "
        "(b) setting one side to 'default', or "
        "(c) removing the shared layer (e.g. set "
        "``network.preprocess_net: []`` AND ensure the arch does not register "
        "a real shared body) so the partition is disjoint."
    )


def build_combine_fn(
    method: str,
    *,
    cagrad_alpha: float = 0.5,
    cagrad_rescale: int = 1,
    fairgrad_alpha: float = 2.0,
) -> Optional[CombineFn]:
    """Build a pure :data:`CombineFn` for a single side, or ``None`` for default."""
    if method == "default":
        return None
    if method == "pcgrad":
        return lambda tg: pcgrad_combine(tg)
    if method == "cagrad":
        def _ca(tg):
            return cagrad_combine(tg, alpha=cagrad_alpha, rescale=cagrad_rescale)
        return _ca
    if method == "fairgrad":
        def _fg(tg):
            return fairgrad_combine(tg, alpha=fairgrad_alpha)
        return _fg
    raise ValueError(f"Unknown gradient method {method!r}; expected one of {sorted(VALID_METHODS)}.")


class PerSideGradientOptimizer:
    """Optimizer wrapper that runs per-task backward loops + per-side combiners."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        actor_method: str = "default",
        critic_method: str = "default",
        *,
        num_tasks: int = 1,
        max_grad_norm: Optional[float] = None,
        cagrad_alpha: float = 0.5,
        cagrad_rescale: int = 1,
        fairgrad_actor_alpha: float = 2.0,
        fairgrad_critic_alpha: float = 1.0,
    ) -> None:
        for m in (actor_method, critic_method):
            if m not in VALID_METHODS:
                raise ValueError(
                    f"method must be one of {sorted(VALID_METHODS)}, got {m!r}."
                )
        self.optimizer = optimizer
        self.actor_method = actor_method
        self.critic_method = critic_method
        self.max_grad_norm = max_grad_norm
        self._cagrad_alpha = cagrad_alpha
        self._cagrad_rescale = cagrad_rescale
        self._fairgrad_actor_alpha = float(fairgrad_actor_alpha)
        self._fairgrad_critic_alpha = float(fairgrad_critic_alpha)
        self._last_diagnostics_by_ns: Dict[str, Any] = {}
        self._last_task_full_grads: List[torch.Tensor] = []
        self.last_post_clip_diagnostics: Optional[Dict[str, float]] = None
        self._partition_validated = False

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    @property
    def last_diagnostics_by_ns(self) -> Dict[str, Any]:
        return self._last_diagnostics_by_ns

    @property
    def last_task_full_grads(self) -> List[torch.Tensor]:
        return self._last_task_full_grads

    def step(
        self,
        task_losses: List[torch.Tensor],
        partition: PartitionedParams,
    ) -> None:
        """Compute per-task gradients and apply the per-side combiners."""
        if not self._partition_validated:
            validate_per_side_methods(partition, self.actor_method, self.critic_method)
            self._partition_validated = True
        all_params = partition.all_params
        task_full_grads: List[torch.Tensor] = []
        for idx, loss in enumerate(task_losses):
            self.optimizer.zero_grad()
            loss.backward(retain_graph=idx < len(task_losses) - 1)
            task_full_grads.append(flatten_grads(all_params).clone())
        self._last_task_full_grads = task_full_grads

        actor_fn = build_combine_fn(
            self.actor_method,
            cagrad_alpha=self._cagrad_alpha,
            cagrad_rescale=self._cagrad_rescale,
            fairgrad_alpha=self._fairgrad_actor_alpha,
        )
        critic_fn = build_combine_fn(
            self.critic_method,
            cagrad_alpha=self._cagrad_alpha,
            cagrad_rescale=self._cagrad_rescale,
            fairgrad_alpha=self._fairgrad_critic_alpha,
        )

        result = apply_per_side(
            task_full_grads,
            partition,
            actor_fn,
            critic_fn,
            produces_per_task=False,
        )

        # Namespace diagnostics as ``{method}_{side}``.
        ns_map: Dict[str, Any] = {}
        for side, method in (("actor", self.actor_method), ("critic", self.critic_method)):
            if method == "default":
                continue
            diag = result.diagnostics.get(side)
            if diag is None:
                continue
            ns_map[f"{method}_{side}"] = diag
        self._last_diagnostics_by_ns = ns_map

        self.optimizer.zero_grad()
        unflatten_grads(result.combined_full_grad, all_params)
        a_pre, c_pre = actor_critic_grad_norms_from_params(partition)
        if self.max_grad_norm is not None:
            pre_clip_total = float(
                torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
            )
            clip_factor = min(1.0, self.max_grad_norm / max(pre_clip_total, 1e-12))
        else:
            clip_factor = 1.0
        a_norm, c_norm = actor_critic_grad_norms_from_params(partition)
        self.last_post_clip_diagnostics = {
            "actor_norm": a_norm,
            "critic_norm": c_norm,
            "clip_factor": clip_factor,
            "combined_actor_norm_pre_clip": a_pre,
            "combined_critic_norm_pre_clip": c_pre,
        }
        self.optimizer.step()
