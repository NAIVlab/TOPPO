"""Multi-task policy wrappers for Tianshou."""

from __future__ import annotations

from typing import Any, Optional

import torch
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy

import gymnasium as gym


class MultiTaskProbabilisticActorPolicy(ProbabilisticActorPolicy):
    """PPO actor policy that wraps a :class:`MultiTaskActor`.

    Per-task head dispatch is handled by the actor itself based on the
    one-hot task encoding in the observation; this class adds nothing
    beyond plumbing ``task_names`` onto the policy.
    """

    def __init__(
        self,
        *,
        actor: torch.nn.Module,
        dist_fn: Any,
        task_names: list[str],
        deterministic_eval: bool = False,
        action_space: gym.spaces.Space,
        observation_space: Optional[gym.spaces.Space] = None,
        action_scaling: bool = True,
        action_bound_method: str | None = "clip",
    ) -> None:
        super().__init__(
            actor=actor,
            dist_fn=dist_fn,
            deterministic_eval=deterministic_eval,
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
        )
        self._task_names = task_names
