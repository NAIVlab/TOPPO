"""Multi-task PPO algorithm inheriting from Tianshou's PPO.

Adds per-task metrics, per-side gradient combiners (PCGrad / CAGrad /
FairGrad), task-stratified minibatch sampling, PopArt value normalization,
and an optional linear feature baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Literal, TYPE_CHECKING, cast

import numpy as np
import scipy.signal
import torch
from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import to_torch_as
from tianshou.data.batch import Batch
from tianshou.data.buffer.buffer_base import ReplayBuffer
from tianshou.data.stats import SequenceSummaryStats
from tianshou.data.types import BatchWithAdvantagesProtocol, RolloutBatchProtocol

from mtrl.gradients.pcgrad import PCGradDiagnostics
from mtrl.gradients.cagrad import CAGradDiagnostics
from mtrl.gradients.fairgrad import FairGradDiagnostics
from mtrl.gradients.per_side import PerSideGradientOptimizer
from mtrl.gradients.scope import (
    PartitionedParams,
    actor_critic_grad_norms_from_params,
    critic_only_slice_from_critic_full,
    extract_task_slices,
    flatten_grads,
)

if TYPE_CHECKING:
    from mtrl.models.linear_baseline import LinearFeatureBaseline


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of variance in y_true explained by y_pred: 1 − Var[y − ŷ] / Var[y]."""
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


from tianshou.algorithm.modelfree.a2c import A2CTrainingStats


@dataclass
class MultiTaskPPOTrainingStats(A2CTrainingStats):
    approx_kl: SequenceSummaryStats
    clip_fraction: SequenceSummaryStats
    explained_variance: SequenceSummaryStats


# Keys that emit min/max/p05/p95 in addition to mean — clip bites and
# pre-clip norms are heavy-tailed and a single mean hides spikes.
_PERCENTILE_KEYS: frozenset = frozenset({
    "grad/clip_factor",
    "grad/combined_actor_norm_pre_clip",
    "grad/combined_critic_norm_pre_clip",
    "grad/post_clip_actor_norm",
    "grad/post_clip_critic_norm",
})


class MultiTaskPPO(PPO):
    """Multi-task PPO with per-side gradient combiners and PopArt support.

    The actor and critic each select their combiner independently via
    ``actor_method`` / ``critic_method`` ∈ {default, pcgrad, cagrad, fairgrad};
    ``default`` falls back to mean-averaging per-task slices (vanilla PPO).
    """

    def __init__(
        self,
        *,
        policy: ProbabilisticActorPolicy,
        critic: torch.nn.Module,
        optim: OptimizerFactory,
        task_names: List[str],
        actor_method:  Literal["default", "pcgrad", "cagrad", "fairgrad"] = "default",
        critic_method: Literal["default", "pcgrad", "cagrad", "fairgrad"] = "default",
        cagrad_alpha: float = 0.5,
        # Per-side FairGrad α.
        fairgrad_actor_alpha: float = 2.0,
        fairgrad_critic_alpha: float = 1.0,
        eps_clip: float = 0.2,
        dual_clip: float | None = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float | None = None,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        minibatch_sampling: Literal["mixed", "stratified"] = "mixed",
        gamma: float = 0.99,
        use_linear_baseline: bool = False,
        linear_baseline: "LinearFeatureBaseline | None" = None,
        use_popart: bool = False,
        use_return_scaling: bool = False,
        return_scaling_min_sigma: float = 1e-2,
        return_scaling_truncation_as_terminal: bool = False,
        critic_loss_mode: Literal["mse", "hlgauss"] = "mse",
        # Per-task gradient snapshot dump (off by default).  See
        # mtrl/utils/grad_snapshots.py for the on-disk layout and the
        # offline loader / FairGrad-replay utilities.
        dump_grads_dir: Optional[str] = None,
        dump_grads_every_collects: int = 5,
    ) -> None:
        super().__init__(
            policy=policy,
            critic=critic,
            optim=optim,
            eps_clip=eps_clip,
            dual_clip=dual_clip,
            value_clip=value_clip,
            advantage_normalization=advantage_normalization,
            recompute_advantage=recompute_advantage,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            max_batchsize=max_batchsize,
            gamma=gamma,
        )
        valid_methods = {"default", "pcgrad", "cagrad", "fairgrad"}
        for label, val in (("actor_method", actor_method), ("critic_method", critic_method)):
            if val not in valid_methods:
                raise ValueError(
                    f"{label} must be one of {sorted(valid_methods)}, got {val!r}."
                )
        self._max_grad_norm = max_grad_norm
        self._task_names = task_names
        self._num_tasks = len(task_names)
        self._actor_method = actor_method
        self._critic_method = critic_method
        self._cagrad_alpha = cagrad_alpha
        self._fairgrad_actor_alpha = float(fairgrad_actor_alpha)
        self._fairgrad_critic_alpha = float(fairgrad_critic_alpha)
        self._minibatch_sampling = minibatch_sampling
        self._task_name_to_id = {name: i for i, name in enumerate(task_names)}

        self._use_linear_baseline = use_linear_baseline
        self._linear_baseline = linear_baseline

        if use_popart and use_linear_baseline:
            raise ValueError(
                "use_popart=True is incompatible with use_linear_baseline=True: "
                "PopArt requires a renormalizable neural critic head."
            )
        self._use_popart = bool(use_popart)

        # Critic-side normalizer alternatives: each occupies the same
        # slot as PopArt, so they are pairwise mutually exclusive.
        if critic_loss_mode not in ("mse", "hlgauss"):
            raise ValueError(
                f"critic_loss_mode must be 'mse' or 'hlgauss', got {critic_loss_mode!r}."
            )
        self._critic_loss_mode = critic_loss_mode
        self._use_return_scaling = bool(use_return_scaling)
        self._retscale_trunc_as_term = bool(return_scaling_truncation_as_terminal)
        # PopArt is exclusive with return scaling: both rescale the same
        # value-loss residual.  Each of the two composes with HL-Gauss.  Under
        # critic_loss_mode="mse" the per-task scale s_i of return scaling
        # divides the squared residual; the categorical analog is to divide
        # the *regression target* before projecting it onto the fixed support,
        # and to multiply the scalar readout back into raw units (see the
        # hlgauss branch in _update_with_batch, and MultiTaskCritic.forward).
        # That is the point of the composition: it standardizes per-task
        # target variance, which is precisely what a single shared support
        # requires.  With HL-Gauss, PopArt acts as ART only: it supplies the
        # target statistics, and its POP affine has no categorical analog.
        if self._use_popart and self._use_return_scaling:
            raise ValueError(
                "use_popart cannot be combined with use_return_scaling: both "
                "rescale the same value-loss residual."
            )
        if use_linear_baseline and (self._use_return_scaling or critic_loss_mode == "hlgauss"):
            raise ValueError(
                "use_return_scaling / hlgauss require a neural critic "
                "(use_linear_baseline must be False)."
            )
        if critic_loss_mode == "hlgauss":
            if value_clip:
                raise ValueError(
                    "critic_loss_mode='hlgauss' requires value_clip=False: "
                    "PPO value clipping has no cross-entropy analog."
                )
            if not hasattr(critic, "hlgauss"):
                raise ValueError(
                    "critic_loss_mode='hlgauss' requires a MultiTaskCritic "
                    "built with hlgauss_num_bins (missing .hlgauss head)."
                )
        if self._use_return_scaling:
            from mtrl.algos.return_scaling import ReturnScalingModule
            self.retscale_module = ReturnScalingModule(
                num_tasks=len(task_names),
                min_sigma=return_scaling_min_sigma,
            )
            if critic_loss_mode == "hlgauss":
                # The categorical head trains on target / s_i, so the critic's
                # scalar readout is in scaled units.  Hand it the module so
                # MultiTaskCritic.forward() can map back to raw units for GAE,
                # advantages and evaluation.  Without this the value function
                # and the reward stream are in different units.  The assignment
                # also registers the module under the critic, so its buffers
                # appear in state_dict() twice; both keys alias one tensor.
                critic.retscale = self.retscale_module
        # tianshou's ActorCriticOnPolicyAlgorithm defines its own
        # ``return_scaling`` bool that switches on a GLOBAL return
        # normalization inside _add_returns_and_advantages.  Nothing here may
        # shadow or enable it — the per-task module lives on
        # ``retscale_module`` instead.
        if getattr(self, "return_scaling", False):
            raise RuntimeError(
                "tianshou's built-in return_scaling must stay disabled: "
                "per-task scaling is handled by retscale_module."
            )
        # Per-collect HL-Gauss diagnostics (fraction of value targets outside
        # the fixed support), drained by the logger each flush.
        self._hlgauss_diag_buffer: Dict[str, List[float]] = {}

        # Snapshot of the env-side baselines_return reward-normalizer stats
        # (one dict per env), refreshed by train.py right before each rolling
        # checkpoint save and pushed back to the envs on resume.  Lives here
        # so _PERSISTABLE_EXTRA_ATTRS carries it into latest.pth.
        self._reward_norm_env_states: Optional[List[Dict[str, float]]] = None

        self._per_task_train_rewards: Dict[str, List[float]] = {n: [] for n in task_names}
        self._per_task_train_wins: Dict[str, List[float]] = {n: [] for n in task_names}
        self._per_task_train_lengths: Dict[str, List[float]] = {n: [] for n in task_names}

        self._per_task_test_rewards: Dict[str, List[float]] = {n: [] for n in task_names}
        self._per_task_test_wins: Dict[str, List[float]] = {n: [] for n in task_names}
        self._per_task_test_lengths: Dict[str, List[float]] = {n: [] for n in task_names}

        # Keys with "/" become wandb sub-namespaces at flush
        # (e.g. "grad/actor_norm" → "grad/actor_norm/<task>").  grad/* and
        # ploss/* are populated only when a non-default gradient method
        # supplies per-task grads; the flush elides empty buffers.
        self._per_task_update_buffer: Dict[str, Dict[str, List[float]]] = {
            t: {
                "approx_kl": [],
                "clip_fraction": [],
                "explained_variance": [],
                "grad/actor_norm": [],
                "grad/critic_norm": [],
                "ploss/actor": [],
                "ploss/vf": [],
            }
            for t in task_names
        }
        # Cross-task aggregate cached by get_and_reset_per_task_update_stats
        # and drained by get_and_reset_grad_aggregates (logger calls them in
        # that order).
        self._last_grad_aggregate: Dict[str, float] = {}
        self._post_clip_norm_buffer: Dict[str, List[float]] = {
            "grad/post_clip_actor_norm": [],
            "grad/post_clip_critic_norm": [],
            "grad/clip_factor": [],
            "grad/combined_actor_norm_pre_clip": [],
            "grad/combined_critic_norm_pre_clip": [],
        }
        _pcgrad_metric_names = (
            "pre_grad_norm",
            "post_grad_norm",
            "grad_shrink_ratio",
            "conflict_rate",
            "wall_time_ms",
        )
        self._pcgrad_update_buffers: Dict[str, Dict[str, List[float]]] = {
            namespace: {m: [] for m in _pcgrad_metric_names}
            for namespace in ("pcgrad_actor", "pcgrad_critic")
        }
        self._cagrad_update_buffers: Dict[str, Dict[str, List[float]]] = {
            ns: {"gram_trace": [], "obj_best": [], "wall_time_ms": []}
            for ns in ("cagrad_actor", "cagrad_critic")
        }
        _fairgrad_namespaces = (
            "fairgrad_actor",
            "fairgrad_critic",
        )
        self._fairgrad_update_buffers: Dict[str, Dict[str, List[float]]] = {
            ns: {
                "gram_trace": [],
                "weight_entropy": [],
                "wall_time_ms": [],
                "regularization_eps": [],
                "residual": [],
                "solver_fallback": [],
                "solver_iters": [],
                "used_scipy_fallback": [],
            }
            for ns in _fairgrad_namespaces
        }
        self._fairgrad_weight_buffers: Dict[str, Dict[str, List[float]]] = {
            ns: {t: [] for t in task_names}
            for ns in _fairgrad_namespaces
        }
        # Per-task effective contribution ‖w_i · g_i‖.
        self._fairgrad_contribution_buffers: Dict[str, Dict[str, List[float]]] = {
            ns: {t: [] for t in task_names}
            for ns in _fairgrad_namespaces
        }
        self._last_test_stats: Dict[str, Dict[str, float]] = {}
        self._latest_data: Optional[Dict[str, torch.Tensor]] = None

        # Lazy-initialised once the underlying torch optimizer exists.
        # Stays None when both sides are "default".
        self._grad_optimizer: Optional[PerSideGradientOptimizer] = None
        self._scope_partition: Optional[PartitionedParams] = None

        # Per-task gradient snapshot dump state.  ``_dump_collect_counter``
        # is bumped at the top of every ``_update_with_batch`` and gates
        # which collects produce a dump.  ``_dump_manifest_written`` makes
        # manifest.json land exactly once per run (on the first gated dump,
        # when partition + named_parameters are all live).  Every field here
        # is inert while ``dump_grads_dir`` is None: no IO, no extra
        # backward passes, no RNG draws, so baselines stay bit-identical.
        self._dump_grads_dir: Optional[str] = (
            str(dump_grads_dir) if dump_grads_dir else None
        )
        self._dump_grads_every: int = max(1, int(dump_grads_every_collects))
        self._dump_collect_counter: int = 0
        self._dump_manifest_written: bool = False
        # When dumping is on AND both sides are "default" we still need
        # per-task gradients to build Gram matrices.  Build a transient
        # PartitionedParams and run K backward passes per dumped minibatch
        # without touching the optimizer step; only fires on dumped
        # minibatches, so every other step is unchanged.
        self._dump_partition_for_default: Optional[PartitionedParams] = None

    @property
    def task_names(self) -> List[str]:
        return self._task_names

    @property
    def actor_method(self) -> str:
        return self._actor_method

    @property
    def critic_method(self) -> str:
        return self._critic_method

    # Python-dict state that lives outside torch's state_dict; restored
    # on resume so per-task accumulators don't cold-start.
    _PERSISTABLE_EXTRA_ATTRS: tuple = (
        # Linear feature baseline — fitted per-task ridge-regression weights.
        # Plain Python object, not an nn.Module, so state_dict() misses it.
        "_linear_baseline",
        "_per_task_train_rewards",
        "_per_task_train_wins",
        "_per_task_train_lengths",
        "_per_task_test_rewards",
        "_per_task_test_wins",
        "_per_task_test_lengths",
        "_per_task_update_buffer",
        "_last_grad_aggregate",
        "_post_clip_norm_buffer",
        "_pcgrad_update_buffers",
        "_cagrad_update_buffers",
        "_fairgrad_update_buffers",
        "_fairgrad_weight_buffers",
        "_fairgrad_contribution_buffers",
        "_last_test_stats",
        "_reward_norm_env_states",
    )

    def get_training_extra_state(self) -> Dict[str, Any]:
        import copy
        return {
            name: copy.deepcopy(getattr(self, name))
            for name in self._PERSISTABLE_EXTRA_ATTRS
            if hasattr(self, name)
        }

    def set_training_extra_state(self, state: Dict[str, Any]) -> None:
        for name in self._PERSISTABLE_EXTRA_ATTRS:
            if name not in state or not hasattr(self, name):
                continue
            current = getattr(self, name)
            saved = state[name]
            if isinstance(current, dict) and isinstance(saved, dict):
                current.clear()
                current.update(saved)
            elif isinstance(current, set) and isinstance(saved, (set, list, tuple)):
                current.clear()
                current.update(saved)
            else:
                setattr(self, name, saved)

    def _setup_gradient_method(self) -> None:
        """Lazy-init per-side gradient wrapper once the optimizer exists.

        Builds the actor / critic / shared parameter partition consumed by
        the wrapper.  Skipped entirely when both sides are ``default``.
        """
        if self._actor_method == "default" and self._critic_method == "default":
            return
        if self._grad_optimizer is not None:
            return
        if not (hasattr(self, "optim") and self.optim is not None):
            return

        raw_optim = self._get_raw_optimizer()
        self._grad_optimizer = PerSideGradientOptimizer(
            raw_optim,
            actor_method=self._actor_method,
            critic_method=self._critic_method,
            num_tasks=self._num_tasks,
            max_grad_norm=self._max_grad_norm,
            cagrad_alpha=self._cagrad_alpha,
            fairgrad_actor_alpha=self._fairgrad_actor_alpha,
            fairgrad_critic_alpha=self._fairgrad_critic_alpha,
        )

        # The backbone whitelist keeps per-task tensors (sigma_param,
        # mu_heads, value_heads) in task_local so they get mean-averaged
        # instead of inflating the solver's Gram diagonal.
        all_params = self._get_optimizer_params()
        actor_params = self._get_actor_backbone_params()
        critic_params = self._get_critic_backbone_params()
        self._scope_partition = PartitionedParams.build(
            all_params=all_params,
            actor_params=actor_params,
            critic_params=critic_params,
        )

    def _get_raw_optimizer(self) -> torch.optim.Optimizer:
        return cast(
            torch.optim.Optimizer,
            getattr(self.optim, "optim", getattr(self.optim, "_optim", self.optim)),
        )

    def _get_optimizer_params(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        seen: set[int] = set()
        raw_optim = self._get_raw_optimizer()
        for group in raw_optim.param_groups:
            for param in group["params"]:
                if param.requires_grad and id(param) not in seen:
                    params.append(param)
                    seen.add(id(param))
        return params

    def _get_actor_backbone_params(self) -> List[torch.nn.Parameter]:
        """Shared-trunk actor params: preprocess + actor_body + actor_output.

        Per-task tensors (sigma_param, mu_heads) are excluded so their
        task-local self-norms don't inflate the solver's Gram diagonal.
        """
        actor = getattr(self.policy, "actor", None)
        if actor is None:
            return []
        params: List[torch.nn.Parameter] = []
        seen: set[int] = set()

        def _add(module):
            for p in module.parameters():
                if p.requires_grad and id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))

        for name in ("preprocess", "actor_body", "actor_output"):
            sub = getattr(actor, name, None)
            if sub is not None:
                _add(sub)
        return params

    def _get_critic_backbone_params(self) -> List[torch.nn.Parameter]:
        """Shared-trunk critic params: critic_body + (shared) preprocess.

        Per-task value_heads excluded; preprocess is included so the
        partition correctly tags it as a shared trunk with the actor.
        """
        if self._use_linear_baseline:
            return []
        critic = getattr(self, "critic", None)
        if critic is None:
            return []
        params: List[torch.nn.Parameter] = []
        seen: set[int] = set()

        def _add(module):
            for p in module.parameters():
                if p.requires_grad and id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))

        body = getattr(critic, "critic_body", None)
        if body is not None:
            _add(body)
        pre = getattr(critic, "preprocess", None)
        if pre is not None:
            _add(pre)
        return params

    def update_per_task_stats(
        self,
        task_name: str,
        reward: float,
        win: float,
        length: float,
        mode: str = "train",
    ) -> None:
        """Record per-task episode stats; ``mode`` is ``"train"`` or ``"test"``."""
        if mode == "train":
            rewards = self._per_task_train_rewards
            wins = self._per_task_train_wins
            lengths = self._per_task_train_lengths
        else:
            rewards = self._per_task_test_rewards
            wins = self._per_task_test_wins
            lengths = self._per_task_test_lengths

        if task_name in rewards:
            rewards[task_name].append(reward)
            wins[task_name].append(win)
            lengths[task_name].append(length)

    def get_and_reset_per_task_train_stats(self) -> Dict[str, Dict[str, float]]:
        """Get and reset accumulated per-task training statistics."""
        return self._get_and_reset_stats(
            self._per_task_train_rewards,
            self._per_task_train_wins,
            self._per_task_train_lengths,
        )

    def get_and_reset_per_task_test_stats(self) -> Dict[str, Dict[str, float]]:
        """Get and reset accumulated per-task testing statistics."""
        stats = self._get_and_reset_stats(
            self._per_task_test_rewards,
            self._per_task_test_wins,
            self._per_task_test_lengths,
        )
        # Cache so get_per_task_test_stats can return the last known values
        # even after the buffers have been drained.
        if any(m.get("num_episodes", 0) > 0 for m in stats.values()):
            self._last_test_stats = stats
        return stats

    def get_per_task_test_stats(self) -> Dict[str, Dict[str, float]]:
        """Get per-task test stats WITHOUT resetting (for final reporting).

        Returns current buffer contents if available, otherwise returns
        the cached stats from the last get_and_reset call.
        """
        current = self._get_stats_no_reset(
            self._per_task_test_rewards,
            self._per_task_test_wins,
            self._per_task_test_lengths,
        )
        # If current buffers are empty, return cached stats from last reset
        if all(m.get("num_episodes", 0) == 0 for m in current.values()):
            return self._last_test_stats if self._last_test_stats else current
        return current

    @staticmethod
    def _get_stats_no_reset(
        rewards: Dict[str, List[float]],
        wins: Dict[str, List[float]],
        lengths: Dict[str, List[float]],
    ) -> Dict[str, Dict[str, float]]:
        stats = {}
        for name in rewards:
            if rewards[name]:
                stats[name] = {
                    "reward": float(np.mean(rewards[name])),
                    "win_rate": float(np.mean(wins[name])),
                    "episode_length": float(np.mean(lengths[name])),
                    "num_episodes": len(rewards[name]),
                }
            else:
                stats[name] = {
                    "reward": 0.0,
                    "win_rate": 0.0,
                    "episode_length": 0.0,
                    "num_episodes": 0,
                }
        return stats

    def _get_and_reset_stats(
        self,
        rewards: Dict[str, List[float]],
        wins: Dict[str, List[float]],
        lengths: Dict[str, List[float]],
    ) -> Dict[str, Dict[str, float]]:
        stats = self._get_stats_no_reset(rewards, wins, lengths)
        # Reset
        for name in self._task_names:
            rewards[name] = []
            wins[name] = []
            lengths[name] = []
        return stats

    def get_and_reset_per_task_update_stats(self) -> Dict[str, Dict[str, float]]:
        """Drain per-task update buffers; return mean of each metric per task.

        Side-effect: caches ``grad/actor_to_critic_ratio`` for the next
        :meth:`get_and_reset_grad_aggregates` call (logger flush order).
        """
        stats: Dict[str, Dict[str, float]] = {}
        for task in self._task_names:
            task_data = self._per_task_update_buffer[task]
            task_stats: Dict[str, float] = {}
            for metric, values in task_data.items():
                if values:
                    task_stats[metric] = float(np.mean(values))
                task_data[metric] = []
            if task_stats:
                stats[task] = task_stats

        # Cross-task aggregate: grad/actor_to_critic_ratio.
        actor_norms = [
            m["grad/actor_norm"] for m in stats.values() if "grad/actor_norm" in m
        ]
        critic_norms = [
            m["grad/critic_norm"] for m in stats.values() if "grad/critic_norm" in m
        ]
        self._last_grad_aggregate = {}
        if actor_norms and critic_norms:
            mean_critic = float(np.mean(critic_norms))
            if mean_critic > self._eps:
                self._last_grad_aggregate["grad/actor_to_critic_ratio"] = (
                    float(np.mean(actor_norms)) / mean_critic
                )

        return stats

    def get_and_reset_grad_aggregates(self) -> Dict[str, float]:
        """Cross-task aggregates + post-clip drains.

        Must be called AFTER :meth:`get_and_reset_per_task_update_stats` in
        the same logger flush — the former populates the cache that this
        method drains.
        """
        out = dict(self._last_grad_aggregate)
        self._last_grad_aggregate = {}
        for k, vals in self._post_clip_norm_buffer.items():
            if vals:
                out[k] = float(np.mean(vals))
                if k in _PERCENTILE_KEYS:
                    out[k + "_min"] = float(np.min(vals))
                    out[k + "_max"] = float(np.max(vals))
                    out[k + "_p05"] = float(np.percentile(vals, 5))
                    out[k + "_p95"] = float(np.percentile(vals, 95))
            self._post_clip_norm_buffer[k] = []
        a = out.get("grad/post_clip_actor_norm")
        c = out.get("grad/post_clip_critic_norm")
        if a is not None and c is not None and c > self._eps:
            out["grad/post_clip_actor_to_critic_ratio"] = a / c
        return out

    def _record_per_task_grad_ploss(
        self,
        task_names_in_batch: List[str],
        actor_losses: List[torch.Tensor],
        vf_losses: List[torch.Tensor],
        task_full_grads: List[torch.Tensor],
        partition: Optional[PartitionedParams] = None,
    ) -> None:
        """Record per-task pre-projection grad norms + PPO loss decomposition.

        Pure logging on already-computed tensors.  All input lists are
        co-indexed with ``task_names_in_batch``.
        """
        if not task_full_grads:
            return
        if partition is None:
            partition = self._scope_partition
        if partition is None:
            return
        actor_slices, critic_full_slices = extract_task_slices(
            task_full_grads, partition,
        )
        critic_only_slices = [
            critic_only_slice_from_critic_full(c, partition)
            for c in critic_full_slices
        ]
        for idx, task_name in enumerate(task_names_in_batch):
            buf = self._per_task_update_buffer[task_name]
            buf["grad/actor_norm"].append(float(actor_slices[idx].norm().item()))
            buf["grad/critic_norm"].append(float(critic_only_slices[idx].norm().item()))
            buf["ploss/actor"].append(float(actor_losses[idx].detach().item()))
            buf["ploss/vf"].append(float(vf_losses[idx].detach().item()))

    # ------------------------------------------------------------------
    # Per-task gradient snapshot dump (gated on self._dump_grads_dir).
    # ------------------------------------------------------------------

    def _dump_should_fire(
        self,
        repeat_idx: int,
        minibatch_idx_in_repeat: int,
    ) -> bool:
        """Gate: dump on the first minibatch of the first repeat of every Nth collect.

        ``self._dump_collect_counter`` is bumped at the top of
        ``_update_with_batch`` so the modulo cadence is collect-aligned.
        The counter is 1 on the first collect, so N=1 dumps every collect
        and N>1 puts the first dump on collect N.
        """
        if not self._dump_grads_dir:
            return False
        if repeat_idx != 0 or minibatch_idx_in_repeat != 0:
            return False
        return (self._dump_collect_counter % self._dump_grads_every) == 0

    def _build_dump_partition_for_default(self) -> "PartitionedParams":
        """Backbone-only partition for the dump-only per-task backward path.

        Used when both sides are ``default`` (vanilla PPO), where no solver
        has built a partition.  Mirrors what ``_setup_gradient_method``
        builds for a non-default side; cached on the instance.
        """
        all_params = self._get_optimizer_params()
        actor_params = self._get_actor_backbone_params()
        critic_params = self._get_critic_backbone_params()
        return PartitionedParams.build(
            all_params=all_params,
            actor_params=actor_params,
            critic_params=critic_params,
        )

    def _maybe_write_dump_manifest(
        self,
        partition: "PartitionedParams",
    ) -> None:
        """Write manifest.json on the first dump (idempotent).

        Records per-layer (name, shape, offset) for both the actor and the
        critic-full view so the offline loader can slice per-layer grads,
        plus the gradient-method and critic-normalizer regime the dumped
        gradients were produced under.
        """
        if self._dump_manifest_written or not self._dump_grads_dir:
            return
        from mtrl.utils.grad_snapshots import (
            _layer_slices_from_param_seq,
            write_manifest,
        )

        # id(param) -> "<dotted.path>".  Walk the actor and the critic
        # separately so names are anchored at "actor." / "critic." prefixes.
        name_by_id: Dict[int, str] = {}
        actor = getattr(self.policy, "actor", None)
        if actor is not None:
            for n, p in actor.named_parameters():
                if p.requires_grad:
                    name_by_id.setdefault(id(p), f"actor.{n}")
        critic = getattr(self, "critic", None)
        if critic is not None:
            for n, p in critic.named_parameters():
                if p.requires_grad:
                    # setdefault: don't rename shared-trunk params already
                    # claimed under the "actor." prefix.
                    name_by_id.setdefault(id(p), f"critic.{n}")

        actor_layers = _layer_slices_from_param_seq(
            partition.actor_param_seq, name_by_id, partition.actor_offsets,
        )
        critic_layers = _layer_slices_from_param_seq(
            partition.critic_full_param_seq, name_by_id, partition.critic_full_offsets,
        )
        write_manifest(
            self._dump_grads_dir,
            task_names=self._task_names,
            actor_layers=actor_layers,
            critic_layers=critic_layers,
            actor_total_numel=partition.actor_total_numel,
            critic_total_numel=partition.critic_full_total_numel,
            use_popart=self._use_popart,
            extra={
                "actor_method": self._actor_method,
                "critic_method": self._critic_method,
                "fairgrad_actor_alpha": self._fairgrad_actor_alpha,
                "fairgrad_critic_alpha": self._fairgrad_critic_alpha,
                # Critic-side normalizer regime.  The offline PopArt
                # sigma^-2 inversion in GradSnapshot assumes an MSE critic
                # loss in normalized space; under critic_loss_mode="hlgauss"
                # the critic gradient comes from a cross-entropy on a
                # categorical head, so the inversion does not apply and the
                # reader must branch on these fields.
                "critic_loss_mode": self._critic_loss_mode,
                "use_return_scaling": bool(self._use_return_scaling),
            },
        )
        self._dump_manifest_written = True

    def _compute_per_task_full_grads_for_dump(
        self,
        task_losses: List[torch.Tensor],
        partition: "PartitionedParams",
    ) -> List[torch.Tensor]:
        """Run K backward passes to capture per-task flat grads for dumping.

        Used when both sides are ``default`` so vanilla-PPO baselines still
        produce K-task Gram matrices.  Tramples ``param.grad`` — the caller
        must follow with ``self.optim.step(loss)``, which zeroes grads
        before its own backward.

        Every backward uses ``retain_graph=True`` because the caller runs
        another ``loss.backward()`` afterwards; the graph is released when
        ``task_losses`` goes out of scope at the end of the minibatch.
        """
        all_params = partition.all_params
        out: List[torch.Tensor] = []
        for loss in task_losses:
            for p in all_params:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            loss.backward(retain_graph=True)
            out.append(flatten_grads(all_params).clone())
        return out

    def _collect_long_ema_success(self) -> Optional[np.ndarray]:
        """Per-task long-EMA success rate, padded to ``num_tasks``.

        Uses the running mean over recent training episodes (the same signal
        ``logger.py`` drains per flush).  Returns ``None`` only at cold start,
        when no per-task win signal exists at all.
        """
        arr = np.zeros(self._num_tasks, dtype=np.float64)
        any_signal = False
        for i, name in enumerate(self._task_names):
            wins = self._per_task_train_wins.get(name, [])
            if wins:
                arr[i] = float(np.mean(wins[-200:]))  # last ~200 episodes
                any_signal = True
        return arr if any_signal else None

    def _maybe_dump_grad_snapshot(
        self,
        *,
        repeat_idx: int,
        minibatch_idx_in_repeat: int,
        gradient_step: int,
        task_names_in_batch: List[str],
        task_indices: torch.Tensor,
        task_losses: Optional[List[torch.Tensor]],
        cached_task_full_grads: Optional[List[torch.Tensor]],
    ) -> None:
        """Write one snapshot npz if the gate fires.  No-op otherwise.

        ``cached_task_full_grads`` is the per-task flat grad list the
        ``PerSideGradientOptimizer`` already cached when a non-default side
        ran — reusing it costs zero extra backward passes.  When both sides
        are default it is ``None`` and we run K extra backward passes,
        gated to dumped minibatches only.
        """
        if not self._dump_should_fire(repeat_idx, minibatch_idx_in_repeat):
            return
        from mtrl.utils.grad_snapshots import write_snapshot

        # Pick the partition: the solver's for non-default runs, a lazily
        # built backbone-only one for vanilla.
        if self._scope_partition is not None:
            partition = self._scope_partition
        else:
            if self._dump_partition_for_default is None:
                self._dump_partition_for_default = self._build_dump_partition_for_default()
            partition = self._dump_partition_for_default

        if cached_task_full_grads is not None:
            task_full_grads = cached_task_full_grads
        else:
            if not task_losses:
                return  # nothing to dump
            task_full_grads = self._compute_per_task_full_grads_for_dump(
                task_losses, partition,
            )

        if not task_full_grads:
            return

        actor_slices, critic_full_slices = extract_task_slices(
            task_full_grads, partition,
        )
        actor_grads_t = torch.stack(actor_slices, dim=0)         # (K, A)
        critic_grads_t = torch.stack(critic_full_slices, dim=0)  # (K, C)

        # PopArt: the post-collect (mu, sigma) are exactly the values the
        # trunk grads were computed against; W, b are post-renormalization.
        popart_kwargs: Dict[str, np.ndarray] = {}
        if self._use_popart:
            pop = self.critic.popart
            popart_kwargs = {
                "popart_sigma": pop.sigma.detach().cpu().double().numpy(),
                "popart_mu":    pop.mu.detach().cpu().double().numpy(),
                "popart_W":     pop.W.detach().cpu().double().numpy(),
                "popart_b":     pop.b.detach().cpu().double().numpy(),
            }

        # FairGrad diagnostics if either side ran a solver this minibatch.
        # last_diagnostics_by_ns is namespaced "{method}_{side}".
        fg_kwargs: Dict[str, Any] = {}
        if self._grad_optimizer is not None:
            ns_map = self._grad_optimizer.last_diagnostics_by_ns or {}
            for side in ("actor", "critic"):
                d = ns_map.get(f"fairgrad_{side}")
                if d is not None and getattr(d, "weights", None) is not None:
                    fg_kwargs[f"fairgrad_{side}_weights"] = (
                        d.weights.detach().cpu().double().numpy()
                    )
                    fg_kwargs[f"fairgrad_{side}_residual"] = float(
                        getattr(d, "residual", 0.0)
                    )

        # Realized per-task minibatch counts (stratified sampling targets
        # uniform, but recording the realized counts lets the loader spot
        # imbalance).
        task_count = np.bincount(
            task_indices.detach().cpu().numpy().astype(np.int64),
            minlength=self._num_tasks,
        ).astype(np.int64)

        long_ema = self._collect_long_ema_success()
        long_ema_kw: Dict[str, np.ndarray] = (
            {"long_ema_success": long_ema} if long_ema is not None else {}
        )

        # Manifest first (only on the first dump), then the snapshot.
        self._maybe_write_dump_manifest(partition)
        write_snapshot(
            self._dump_grads_dir,
            collect_idx=int(self._dump_collect_counter),
            repeat_idx=int(repeat_idx),
            minibatch_idx=int(minibatch_idx_in_repeat),
            gradient_step=int(gradient_step),
            task_names_in_batch=list(task_names_in_batch),
            actor_grads=actor_grads_t,
            critic_full_grads=critic_grads_t,
            task_count_in_minibatch=task_count,
            **popart_kwargs,
            **fg_kwargs,
            **long_ema_kw,
        )

    def _record_pcgrad_diagnostics(
        self,
        diagnostics: PCGradDiagnostics | None,
        namespace: str = "pcgrad",
    ) -> None:
        """Accumulate PCGrad diagnostics under ``pcgrad_actor`` / ``pcgrad_critic``."""
        if diagnostics is None:
            return
        buffer = self._pcgrad_update_buffers.get(namespace)
        if buffer is None:
            return
        for key, value in diagnostics.as_dict(namespace=namespace).items():
            metric_name = key.split("/", 1)[1]
            if metric_name in buffer:
                buffer[metric_name].append(float(value))

    def get_and_reset_pcgrad_update_stats(self) -> Dict[str, float]:
        """Get and reset accumulated PCGrad diagnostics across all namespaces."""
        stats: Dict[str, float] = {}
        for namespace, buffer in self._pcgrad_update_buffers.items():
            for metric_name, values in buffer.items():
                if values:
                    stats[f"{namespace}/{metric_name}"] = float(np.mean(values))
                    buffer[metric_name] = []
        return stats

    def _record_cagrad_diagnostics(
        self,
        diagnostics: "CAGradDiagnostics | None",
        namespace: str = "cagrad",
    ) -> None:
        """Accumulate CAGrad diagnostics under the given namespace."""
        if diagnostics is None:
            return
        buf = self._cagrad_update_buffers.get(namespace)
        if buf is None:
            return
        buf["gram_trace"].append(float(diagnostics.gram_trace))
        buf["obj_best"].append(float(diagnostics.obj_best))
        buf["wall_time_ms"].append(float(diagnostics.wall_time_ms))

    def get_and_reset_cagrad_update_stats(self) -> Dict[str, float]:
        """Get and reset accumulated CAGrad diagnostics across namespaces."""
        stats: Dict[str, float] = {}
        for namespace, buf in self._cagrad_update_buffers.items():
            for metric_name, values in buf.items():
                if values:
                    stats[f"{namespace}/{metric_name}"] = float(np.mean(values))
                    buf[metric_name] = []
        return stats

    def _record_fairgrad_diagnostics(
        self,
        diagnostics: "FairGradDiagnostics | None",
        namespace: str = "fairgrad",
    ) -> None:
        if diagnostics is None:
            return
        buf = self._fairgrad_update_buffers.get(namespace)
        weight_buf = self._fairgrad_weight_buffers.get(namespace)
        contrib_buf = self._fairgrad_contribution_buffers.get(namespace)
        if buf is None or weight_buf is None or contrib_buf is None:
            return
        buf["gram_trace"].append(float(diagnostics.gram_trace))
        buf["weight_entropy"].append(float(diagnostics.weight_entropy))
        buf["wall_time_ms"].append(float(diagnostics.wall_time_ms))
        buf["regularization_eps"].append(float(diagnostics.regularization_eps))
        buf["residual"].append(float(diagnostics.residual))
        buf["solver_fallback"].append(float(diagnostics.solver_fallback))
        buf["solver_iters"].append(float(diagnostics.solver_iters))
        buf["used_scipy_fallback"].append(float(diagnostics.used_scipy_fallback))
        weights = diagnostics.weights
        contribution_norms = diagnostics.contribution_norms
        for i, name in enumerate(self._task_names):
            if i < len(weights):
                weight_buf[name].append(float(weights[i]))
            if contribution_norms is not None and i < len(contribution_norms):
                contrib_buf[name].append(float(contribution_norms[i]))

    def get_and_reset_fairgrad_update_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        for namespace, buf in self._fairgrad_update_buffers.items():
            for metric_name, values in buf.items():
                if values:
                    stats[f"{namespace}/{metric_name}"] = float(np.mean(values))
                    buf[metric_name] = []
        for namespace, weight_buf in self._fairgrad_weight_buffers.items():
            for name, values in weight_buf.items():
                if values:
                    stats[f"{namespace}/weight/{name}"] = float(np.mean(values))
                    weight_buf[name] = []
        for namespace, contrib_buf in self._fairgrad_contribution_buffers.items():
            for name, values in contrib_buf.items():
                if values:
                    stats[f"{namespace}/contribution/{name}"] = float(np.mean(values))
                    contrib_buf[name] = []
        return stats

    def get_and_reset_popart_diagnostics(self) -> Dict[str, float]:
        """Snapshot per-task PopArt μ, σ, W, b for logging.

        Empty dict when PopArt is off.  Snapshot (not drained) — the
        underlying state is running, so read on every flush.
        """
        if not self._use_popart:
            return {}
        popart = getattr(self.critic, "popart", None)
        if popart is None:
            return {}
        snap = popart.get_diagnostics()
        out: Dict[str, float] = {}
        for i, name in enumerate(self._task_names):
            out[f"popart/mu/{name}"]    = float(snap["mu"][i].item())
            out[f"popart/sigma/{name}"] = float(snap["sigma"][i].item())
            out[f"popart/W/{name}"]     = float(snap["W"][i].item())
            out[f"popart/b/{name}"]     = float(snap["b"][i].item())
        return out

    def get_and_reset_return_scaling_diagnostics(self) -> Dict[str, float]:
        """Snapshot per-task return-scaling σ and active scale s_i.

        Empty dict when return scaling is off.  Snapshot (not drained) —
        the underlying statistics are running, so read on every flush.
        """
        if not self._use_return_scaling:
            return {}
        snap = self.retscale_module.get_diagnostics()
        out: Dict[str, float] = {}
        for i, name in enumerate(self._task_names):
            out[f"retscale/sigma/{name}"] = float(snap["sigma"][i].item())
            out[f"retscale/scale/{name}"] = float(snap["scale"][i].item())
        return out

    def get_and_reset_hlgauss_diagnostics(self) -> Dict[str, float]:
        """Drain per-collect HL-Gauss target-clip fractions (mean per flush)."""
        if not self._hlgauss_diag_buffer:
            return {}
        out = {
            key: float(np.mean(vals))
            for key, vals in self._hlgauss_diag_buffer.items()
            if vals
        }
        self._hlgauss_diag_buffer = {}
        return out

    def _record_grad_diagnostics_by_ns(
        self,
        diagnostics_by_ns: Dict[str, Any],
    ) -> None:
        """Route each (namespace, diagnostic) pair to the right recorder."""
        for ns, diag in diagnostics_by_ns.items():
            if diag is None:
                continue
            if isinstance(diag, PCGradDiagnostics):
                self._record_pcgrad_diagnostics(diag, namespace=ns)
            elif isinstance(diag, CAGradDiagnostics):
                self._record_cagrad_diagnostics(diag, namespace=ns)
            elif isinstance(diag, FairGradDiagnostics):
                self._record_fairgrad_diagnostics(diag, namespace=ns)

    @staticmethod
    def _compute_timesteps_from_episodes(
        batch: Any,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Per-step timesteps within each episode (t=0,1,2,... per episode)."""
        n = len(indices)
        end_flag = np.logical_or(
            np.asarray(batch.terminated, dtype=bool),
            np.asarray(batch.truncated, dtype=bool),
        )
        end_flag[np.isin(indices, buffer.unfinished_index())] = True

        timesteps = np.zeros(n, dtype=np.float64)
        t = 0
        for i in range(n):
            timesteps[i] = t
            if end_flag[i]:
                t = 0
            else:
                t += 1
        return timesteps

    def _add_returns_and_advantages(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithAdvantagesProtocol:
        """Override to use linear baseline when enabled."""
        if not self._use_linear_baseline or self._linear_baseline is None:
            return super()._add_returns_and_advantages(batch, buffer, indices)

        # Linear-baseline path: extract episodes → MC returns → fit baseline
        # → predict values → single GAE pass.  Unbiased MC fit (no value
        # bootstrap) per Duan et al., 2016.

        obs_np = batch.obs
        if isinstance(obs_np, torch.Tensor):
            obs_np = obs_np.cpu().numpy()
        obs_np = np.asarray(obs_np, dtype=np.float64)
        raw_obs = obs_np[:, :-self._num_tasks]
        task_indices = obs_np[:, -self._num_tasks:].argmax(axis=-1)

        timesteps = self._compute_timesteps_from_episodes(batch, buffer, indices)
        end_flag = np.logical_or(
            np.asarray(batch.terminated, dtype=bool),
            np.asarray(batch.truncated, dtype=bool),
        )
        end_flag[np.isin(indices, buffer.unfinished_index())] = True

        rewards = np.asarray(batch.rew, dtype=np.float64)
        mc_returns = np.zeros_like(rewards)
        ep_start = 0
        for i in range(len(rewards)):
            if end_flag[i]:
                ep_end = i + 1
                ep_rewards = rewards[ep_start:ep_end]
                mc_returns[ep_start:ep_end] = scipy.signal.lfilter(
                    [1], [1, float(-self.gamma)], ep_rewards[::-1],
                )[::-1]
                ep_start = ep_end
        if ep_start < len(rewards):
            ep_rewards = rewards[ep_start:]
            mc_returns[ep_start:] = scipy.signal.lfilter(
                [1], [1, float(-self.gamma)], ep_rewards[::-1],
            )[::-1]

        self._linear_baseline.fit(raw_obs, mc_returns, timesteps, task_indices)
        v_s = self._linear_baseline.predict(raw_obs, timesteps, task_indices)

        device = next(self.parameters()).device
        batch.v_s = torch.tensor(v_s, dtype=torch.float32, device=device)

        obs_next_np = batch.obs_next
        if isinstance(obs_next_np, torch.Tensor):
            obs_next_np = obs_next_np.cpu().numpy()
        obs_next_np = np.asarray(obs_next_np, dtype=np.float64)
        raw_obs_next = obs_next_np[:, :-self._num_tasks]
        task_indices_next = obs_next_np[:, -self._num_tasks:].argmax(axis=-1)
        v_s_ = self._linear_baseline.predict(raw_obs_next, timesteps + 1, task_indices_next)

        # Don't bootstrap across true episode boundaries.
        v_s_masked = v_s_ * (~np.asarray(batch.terminated, dtype=bool)).astype(np.float64)

        unnormalized_returns, advantages = self.compute_episodic_return(
            batch, buffer, indices,
            v_s_=v_s_masked, v_s=v_s,
            gamma=self.gamma, gae_lambda=self.gae_lambda,
        )

        batch.returns = to_torch_as(unnormalized_returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
        return cast(BatchWithAdvantagesProtocol, batch)

    def _compute_per_task_actor_terms(
        self,
        minibatch: Batch,
        task_indices: torch.Tensor,
        advantages: torch.Tensor,
        all_log_probs: torch.Tensor,
        all_entropy: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """Compute per-task actor-only PPO terms."""
        task_terms: List[Dict[str, Any]] = []
        for task_id, task_name in enumerate(self._task_names):
            mask = task_indices == task_id
            if mask.sum() == 0:
                continue

            adv_i = advantages[mask]
            logp_old_i = minibatch.logp_old[mask]
            log_prob_i = all_log_probs[mask]
            ratio_i = (log_prob_i - logp_old_i).exp().float()
            ratio_i = ratio_i.reshape(ratio_i.size(0), -1).transpose(0, 1)

            surr1 = ratio_i * adv_i
            surr2 = ratio_i.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_i
            if self.dual_clip:
                clip1 = torch.min(surr1, surr2)
                clip2 = torch.max(clip1, self.dual_clip * adv_i)
                actor_loss = -torch.where(adv_i < 0, clip2, clip1).mean()
            else:
                actor_loss = -torch.min(surr1, surr2).mean()

            entropy = all_entropy[mask].mean()
            clip_fraction = (
                ratio_i.gt(1.0 + self.eps_clip) | ratio_i.lt(1.0 - self.eps_clip)
            ).float().mean()
            approx_kl = (logp_old_i - log_prob_i).mean()

            task_terms.append(
                {
                    "task_id": task_id,
                    "task_name": task_name,
                    "mask": mask,
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                    "actor_objective": actor_loss - self.ent_coef * entropy,
                    "clip_fraction": clip_fraction,
                    "approx_kl": approx_kl,
                }
            )

        return task_terms

    def _iter_minibatches(self, batch: Batch, batch_size: int | None) -> Iterator[Batch]:
        """Yield minibatches using mixed or task-stratified sampling."""
        split_batch_size = batch_size or -1
        if self._minibatch_sampling == "mixed":
            yield from batch.split(split_batch_size, merge_last=True)
            return

        if batch_size is None or batch_size <= 0:
            yield batch
            return

        if batch_size < self._num_tasks:
            raise ValueError(
                "Stratified minibatch sampling requires batch_size >= num_tasks, "
                f"got batch_size={batch_size}, num_tasks={self._num_tasks}.",
            )

        obs = batch.obs
        if isinstance(obs, torch.Tensor):
            task_indices = obs[:, -self._num_tasks:].argmax(dim=-1).detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs)
            task_indices = obs_np[:, -self._num_tasks:].argmax(axis=-1)

        per_task_indices = [
            np.random.permutation(np.flatnonzero(task_indices == task_id))
            for task_id in range(self._num_tasks)
        ]
        task_counts = np.array([len(idx) for idx in per_task_indices], dtype=np.float64)
        total_samples = task_counts.sum()

        # Proportional quota: each task's share of the minibatch matches its
        # share of the buffer.  Every present task gets at least 1 sample.
        present_mask = task_counts > 0
        num_present = int(present_mask.sum())
        if num_present == 0:
            return

        raw_quotas = task_counts / max(total_samples, 1.0) * batch_size
        task_batch_sizes = np.zeros(self._num_tasks, dtype=np.int64)
        task_batch_sizes[present_mask] = np.maximum(np.floor(raw_quotas[present_mask]).astype(np.int64), 1)

        # Distribute remaining slots to largest-remainder tasks
        remainder = batch_size - task_batch_sizes.sum()
        if remainder > 0:
            fractional = raw_quotas - task_batch_sizes
            fractional[~present_mask] = -1.0
            top_ids = np.argsort(-fractional)[:remainder]
            task_batch_sizes[top_ids] += 1

        cursor = np.zeros(self._num_tasks, dtype=np.int64)
        while True:
            minibatch_parts = []
            for task_id in range(self._num_tasks):
                quota = task_batch_sizes[task_id]
                if quota == 0:
                    continue

                start = cursor[task_id]
                end = start + quota
                # Stop as soon as any task can't fill its quota — every
                # yielded minibatch keeps the same stratified composition.
                # With uniform-per-task collection, all tasks exhaust on the
                # same iteration so no samples are wasted.
                if end > len(per_task_indices[task_id]):
                    return

                minibatch_parts.append(per_task_indices[task_id][start:end])
                cursor[task_id] = end

            if not minibatch_parts:
                return

            minibatch_indices = np.concatenate(minibatch_parts)
            minibatch_indices = np.random.permutation(minibatch_indices)
            yield cast(Batch, batch[minibatch_indices])

    def _compute_value_loss(
        self,
        minibatch: Batch,
        value: torch.Tensor,
        residual_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MSE value loss; ``residual_scale`` (per-sample, no grad) divides the
        residuals before squaring — return-based scaling's δ/σ (Schaul et al.
        2021, Eq. 1)."""
        if self.value_clip:
            v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(
                -self.eps_clip,
                self.eps_clip,
            )
            vf1 = minibatch.returns - value
            vf2 = minibatch.returns - v_clip
            if residual_scale is not None:
                vf1 = vf1 / residual_scale
                vf2 = vf2 / residual_scale
            return torch.max(vf1.pow(2), vf2.pow(2)).mean()
        residual = minibatch.returns - value
        if residual_scale is not None:
            residual = residual / residual_scale
        return residual.pow(2).mean()

    def _compute_per_task_losses(
        self,
        minibatch: Batch,
        task_indices: torch.Tensor,
        advantages: torch.Tensor,
        all_log_probs: torch.Tensor,
        all_values: Optional[torch.Tensor],
        all_entropy: torch.Tensor,
        per_sample_vf_loss: Optional[torch.Tensor] = None,
        residual_scale: Optional[torch.Tensor] = None,
    ) -> tuple[List[torch.Tensor], List[str], List[torch.Tensor], List[torch.Tensor]]:
        """Compute per-task PPO losses for the per-side gradient wrapper.

        Returns ``(task_losses, task_names_in_batch, task_actor_losses,
        task_vf_losses)`` — co-indexed lists, one entry per task whose
        minibatch mask is non-empty.  Per-task actor/vf losses are returned
        alongside the combined loss so the caller can log
        ``ploss/{actor,vf}/<task>`` without recomputation.

        ``per_sample_vf_loss`` (HL-Gauss CE) replaces the MSE computation
        when given; ``residual_scale`` (return-based scaling) divides the
        MSE residuals per sample.
        """
        task_losses: List[torch.Tensor] = []
        task_names_in_batch: List[str] = []
        task_actor_losses: List[torch.Tensor] = []
        task_vf_losses: List[torch.Tensor] = []
        actor_terms = self._compute_per_task_actor_terms(
            minibatch=minibatch,
            task_indices=task_indices,
            advantages=advantages,
            all_log_probs=all_log_probs,
            all_entropy=all_entropy,
        )
        for term in actor_terms:
            vf_loss_i = torch.tensor(0.0, device=all_log_probs.device)
            if per_sample_vf_loss is not None:
                vf_loss_i = per_sample_vf_loss[term["mask"]].mean()
            elif not self._use_linear_baseline and all_values is not None:
                value_i = all_values[term["mask"]]
                returns_i = minibatch.returns[term["mask"]]
                scale_i = (
                    residual_scale[term["mask"]] if residual_scale is not None else None
                )
                if self.value_clip:
                    v_old_i = minibatch.v_s[term["mask"]]
                    v_clip_i = v_old_i + (value_i - v_old_i).clamp(
                        -self.eps_clip,
                        self.eps_clip,
                    )
                    vf1 = returns_i - value_i
                    vf2 = returns_i - v_clip_i
                    if scale_i is not None:
                        vf1 = vf1 / scale_i
                        vf2 = vf2 / scale_i
                    vf_loss_i = torch.max(vf1.pow(2), vf2.pow(2)).mean()
                else:
                    residual_i = returns_i - value_i
                    if scale_i is not None:
                        residual_i = residual_i / scale_i
                    vf_loss_i = residual_i.pow(2).mean()

            # vf_coef × FairGrad interaction: scaling per-task critic grads by
            # c (i.e. vf_coef) feeds G = c²·H into the α-fairness equation
            # G·w = w^{−1/α}, so weights absorb c^{−2α/(α+1)} and the combined
            # output scales as c^{(1−α)/(α+1)} — exactly zero at α=1. That
            # invariance holds for the EXACT KKT solution only.  The shipped
            # solver floors diag(G) at eps=1e-8 and stops on an ABSOLUTE
            # residual (eta=1e-2), and neither scales with c: shrinking c makes
            # the floor bind on more tasks and Newton stop earlier (often at
            # iteration 0, from the diagonal warm start).  With the default
            # critic α=1 (Nash bargaining), changing vf_coef around the
            # shipped 1e-3 therefore still changes both the direction and the
            # magnitude of the FairGrad-projected critic step; it is not a
            # no-op.  See the note above the diag(G) floor in
            # mtrl/gradients/fairgrad.py.
            task_losses.append(
                term["actor_loss"] + self.vf_coef * vf_loss_i - self.ent_coef * term["entropy"]
            )
            task_names_in_batch.append(term["task_name"])
            task_actor_losses.append(term["actor_loss"])
            task_vf_losses.append(vf_loss_i)

        return task_losses, task_names_in_batch, task_actor_losses, task_vf_losses

    def _update_with_batch(
        self,
        batch: Batch,
        batch_size: int | None,
        repeat: int,
    ) -> MultiTaskPPOTrainingStats:
        self._setup_gradient_method()
        # Bump the dump-cadence counter once per collect.  It advances
        # whether or not a dump fires; ``_dump_should_fire`` decides which
        # collects actually produce a snapshot.
        self._dump_collect_counter += 1

        # PopArt: update per-task (μ, σ) and POP-renormalize the per-task
        # affine layer.  Runs once per collect BEFORE the minibatch loop so
        # all minibatches see the updated stats.  POP guarantees raw V is
        # invariant across the update, so the just-consumed GAE result
        # remains consistent under recompute_advantage.
        if self._use_popart or self._use_return_scaling or self._critic_loss_mode == "hlgauss":
            obs_full = batch.obs
            if isinstance(obs_full, np.ndarray):
                obs_full = torch.as_tensor(
                    obs_full, device=next(self.parameters()).device,
                )
            task_idx_full = obs_full[:, -self._num_tasks:].argmax(dim=-1)
            returns_full = batch.returns
            if not isinstance(returns_full, torch.Tensor):
                returns_full = torch.as_tensor(
                    returns_full, device=task_idx_full.device,
                )

        if self._use_popart:
            per_task_returns: Dict[int, torch.Tensor] = {}
            for i in range(self._num_tasks):
                mask_i = (task_idx_full == i)
                if mask_i.sum() > 0:
                    per_task_returns[i] = returns_full[mask_i]
            self.critic.popart.update_and_renormalize(per_task_returns)

        # Return-based scaling: fold this collect into the all-time per-task
        # statistics and freeze s_i = max(σ_i, σ_batch_i, min_sigma) for all
        # of this collect's minibatches.  Same once-per-collect cadence as
        # PopArt's stats update.  γ_t is 0 at true terminal steps only —
        # time-limit truncation bootstraps, so it keeps the constant γ.
        if self._use_return_scaling:
            rew_full = torch.as_tensor(
                np.asarray(batch.rew), device=task_idx_full.device, dtype=torch.float64,
            )
            ends = np.asarray(batch.terminated, dtype=bool)
            if self._retscale_trunc_as_term:
                ends = np.logical_or(ends, np.asarray(batch.truncated, dtype=bool))
            terminated_full = torch.as_tensor(
                ends.astype(np.float64), device=task_idx_full.device,
            )
            gamma_full = self.gamma * (1.0 - terminated_full)
            self.retscale_module.update_and_refresh_scale(
                task_idx_full, rew_full, gamma_full, returns_full,
            )

        # HL-Gauss: track how often value targets fall outside the fixed
        # support — the quantity that exposes support-misfit per task.
        if self._critic_loss_mode == "hlgauss":
            hl = self.critic.hlgauss
            with torch.no_grad():
                # Diagnose the quantity that is ACTUALLY projected onto the
                # support: under return scaling that is the scaled target
                # R_i / s_i, not the raw return.
                diag_targets = returns_full
                if self._use_popart:
                    # ART normalizes the target before projection, so the
                    # support sees (R - mu_i) / sigma_i, not R.
                    diag_targets = self.critic.popart.normalize_target(
                        diag_targets, task_idx_full,
                    )
                if self._use_return_scaling:
                    diag_targets = diag_targets / self.retscale_module.last_scale.to(
                        device=returns_full.device, dtype=returns_full.dtype,
                    )[task_idx_full]
                oob = ((diag_targets < hl.vmin) | (diag_targets > hl.vmax)).double()
                self._hlgauss_diag_buffer.setdefault(
                    "hlgauss/target_clip_frac", [],
                ).append(float(oob.mean().item()))
                for i, task_name in enumerate(self._task_names):
                    mask_i = (task_idx_full == i)
                    n_i = int(mask_i.sum())
                    if n_i > 0:
                        self._hlgauss_diag_buffer.setdefault(
                            f"hlgauss/target_clip_frac/{task_name}", [],
                        ).append(float(oob[mask_i].mean().item()))
                    # Per-task target spread measured in BIN units -- the
                    # RESOLUTION diagnostic.  target_clip_frac only catches a
                    # too-NARROW support; a too-COARSE one is completely
                    # silent (clip_frac reads 0.0 while a task's entire value
                    # range sits inside one bin and its critic collapses to a
                    # constant).  That is the failure mode the un-normalized
                    # HL-Gauss arm hit on 5/50 tasks, and nothing logged it.
                    if n_i > 1:
                        sd_i = float(diag_targets[mask_i].std().item())
                        self._hlgauss_diag_buffer.setdefault(
                            f"hlgauss/target_std_bins/{task_name}", [],
                        ).append(sd_i / hl.bin_width)

        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        approx_kls, clip_fractions, explained_vars = [], [], []

        gradient_steps = 0
        for step in range(repeat):
            if self.recompute_adv and step > 0:
                batch = cast(
                    Batch,
                    self._add_returns_and_advantages(batch, self._buffer, self._indices),
                )
            for mb_idx_in_repeat, minibatch in enumerate(
                self._iter_minibatches(batch, batch_size)
            ):
                gradient_steps += 1

                dist = self.policy(minibatch).dist

                obs_t = minibatch.obs
                if isinstance(obs_t, np.ndarray):
                    obs_t = torch.as_tensor(obs_t, device=next(self.parameters()).device)
                if obs_t.dim() == 1:
                    obs_t = obs_t.unsqueeze(0)
                task_indices = obs_t[:, -self._num_tasks:].argmax(dim=-1)

                # PopArt: normalize returns + v_s into the PopArt target
                # space up-front so downstream losses operate in normalized
                # space.  Cache raw returns for the EV metric (which
                # compares raw V to raw returns).  Reassigning ``minibatch``
                # attributes does NOT propagate back to ``batch`` — Tianshou
                # Batch indexing yields fresh attribute references.
                raw_returns_for_metric: Optional[torch.Tensor] = None
                if self._use_popart:
                    raw_returns_for_metric = minibatch.returns.clone()
                    pop = self.critic.popart
                    sigma_i = pop.sigma[task_indices].to(minibatch.returns.dtype)
                    mu_i    = pop.mu[task_indices].to(minibatch.returns.dtype)
                    minibatch.returns = (minibatch.returns - mu_i) / sigma_i
                    minibatch.v_s     = (minibatch.v_s     - mu_i) / sigma_i

                advantages = minibatch.adv.clone()
                if self.advantage_normalization:
                    for i in range(self._num_tasks):
                        mask = (task_indices == i)
                        if mask.sum() > 1:
                            t_adv = advantages[mask]
                            advantages[mask] = (t_adv - t_adv.mean()) / (t_adv.std() + self._eps)
                        # mask.sum() == 1: keep raw advantage; zeroing it
                        # would kill the gradient.

                all_log_probs = dist.log_prob(minibatch.act)
                all_entropy = dist.entropy()

                # Clipped surrogate and dual clip (and the value clip in
                # _compute_value_loss) follow tianshou 2.0.0
                # PPO._update_with_batch (MIT; see NOTICE).
                ratios = (all_log_probs - minibatch.logp_old).exp().float()
                ratios = ratios.reshape(ratios.size(0), -1).transpose(0, 1)
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
                
                if self.dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self.dual_clip * advantages)
                    clip_loss = -torch.where(advantages < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                
                per_sample_vf: Optional[torch.Tensor] = None
                residual_scale: Optional[torch.Tensor] = None
                if self._use_linear_baseline:
                    value = minibatch.v_s
                    vf_loss = torch.tensor(0.0, device=minibatch.adv.device)
                elif self._critic_loss_mode == "hlgauss":
                    # Checked BEFORE _use_popart: with both on, the critic is
                    # categorical and ART only supplies the target statistics
                    # (minibatch.returns was already normalized in place above).
                    logits, _ = self.critic.forward_logits(minibatch.obs)
                    hl_target = minibatch.returns
                    if self._use_return_scaling:
                        # Categorical analog of return-based scaling: divide
                        # the regression TARGET by the frozen per-task scale
                        # s_i so every task's targets reach the shared support
                        # with unit variance.  Dividing the cross-entropy
                        # itself would be meaningless -- CE has no residual.
                        residual_scale = self.retscale_module.last_scale.to(
                            device=task_indices.device, dtype=hl_target.dtype
                        )[task_indices]
                        hl_target = hl_target / residual_scale
                    per_sample_vf = self.critic.hlgauss.loss(logits, hl_target)
                    vf_loss = per_sample_vf.mean()
                    # Scalar readout for logging only — the CE loss above is
                    # the sole gradient path into the critic.  Map back to RAW
                    # units so EV / advantage diagnostics stay comparable with
                    # the unscaled arms.
                    value = self.critic.hlgauss.from_logits(logits.detach())
                    if self._use_return_scaling:
                        value = value * residual_scale
                    if self._use_popart:
                        value = self.critic.popart.denormalize(value, task_indices)
                elif self._use_popart:
                    # Pre-(σ,μ) prediction; loss in normalized space.
                    value, _ = self.critic.forward_normalized(minibatch.obs)
                    vf_loss = self._compute_value_loss(minibatch, value)
                else:
                    value = self.critic(minibatch.obs).flatten()
                    if self._use_return_scaling:
                        # Stats buffers live on CPU (fp64 accumulation);
                        # move the frozen scale to the batch device to index.
                        residual_scale = self.retscale_module.last_scale.to(
                            device=task_indices.device, dtype=value.dtype
                        )[task_indices]
                    vf_loss = self._compute_value_loss(minibatch, value, residual_scale)

                ent_loss = all_entropy.mean()
                loss = clip_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

                if self._grad_optimizer is not None:
                    all_values = (
                        value
                        if not (self._use_linear_baseline or per_sample_vf is not None)
                        else None
                    )
                    (
                        task_losses,
                        task_names_in_batch,
                        task_actor_losses,
                        task_vf_losses,
                    ) = self._compute_per_task_losses(
                        minibatch, task_indices, advantages, all_log_probs, all_values, all_entropy,
                        per_sample_vf_loss=per_sample_vf,
                        residual_scale=residual_scale,
                    )
                    if task_losses:
                        assert self._scope_partition is not None
                        self._grad_optimizer.step(
                            task_losses,
                            partition=self._scope_partition,
                        )
                        self._record_grad_diagnostics_by_ns(
                            self._grad_optimizer.last_diagnostics_by_ns,
                        )
                        # Pure logging: pre-projection per-task grads cached
                        # by the solver during its backward loop.
                        last_grads = self._grad_optimizer.last_task_full_grads
                        if last_grads:
                            self._record_per_task_grad_ploss(
                                task_names_in_batch=task_names_in_batch,
                                actor_losses=task_actor_losses,
                                vf_losses=task_vf_losses,
                                task_full_grads=last_grads,
                            )
                        post_clip = self._grad_optimizer.last_post_clip_diagnostics
                        if post_clip is not None:
                            self._post_clip_norm_buffer[
                                "grad/post_clip_actor_norm"
                            ].append(post_clip["actor_norm"])
                            self._post_clip_norm_buffer[
                                "grad/post_clip_critic_norm"
                            ].append(post_clip["critic_norm"])
                            self._post_clip_norm_buffer[
                                "grad/clip_factor"
                            ].append(post_clip["clip_factor"])
                            pre_a = post_clip.get("combined_actor_norm_pre_clip")
                            pre_c = post_clip.get("combined_critic_norm_pre_clip")
                            if pre_a is not None:
                                self._post_clip_norm_buffer[
                                    "grad/combined_actor_norm_pre_clip"
                                ].append(pre_a)
                            if pre_c is not None:
                                self._post_clip_norm_buffer[
                                    "grad/combined_critic_norm_pre_clip"
                                ].append(pre_c)
                        # Snapshot dump, gated by collect cadence + first
                        # repeat / first minibatch.  Reuses the per-task flat
                        # grads the solver already cached, so this costs zero
                        # extra backward passes.  No-op when dumping is off
                        # or this minibatch is not gated.
                        self._maybe_dump_grad_snapshot(
                            repeat_idx=step,
                            minibatch_idx_in_repeat=mb_idx_in_repeat,
                            gradient_step=gradient_steps,
                            task_names_in_batch=task_names_in_batch,
                            task_indices=task_indices,
                            task_losses=None,
                            cached_task_full_grads=last_grads,
                        )
                    else:
                        self.optim.step(loss)
                else:
                    # Vanilla PPO path.  When the dump gate fires on this
                    # minibatch, run K extra backward passes BEFORE the
                    # global step so the per-task grads are measured at the
                    # same model state as the vanilla update.  Costs K
                    # backwards on dumped minibatches only; when dumping is
                    # off, ``_dump_should_fire`` short-circuits to False and
                    # nothing below it runs.
                    if self._dump_should_fire(step, mb_idx_in_repeat):
                        all_values_for_dump = (
                            value
                            if not (self._use_linear_baseline or per_sample_vf is not None)
                            else None
                        )
                        (
                            dump_task_losses,
                            dump_task_names,
                            _,
                            _,
                        ) = self._compute_per_task_losses(
                            minibatch, task_indices, advantages,
                            all_log_probs, all_values_for_dump, all_entropy,
                            per_sample_vf_loss=per_sample_vf,
                            residual_scale=residual_scale,
                        )
                        self._maybe_dump_grad_snapshot(
                            repeat_idx=step,
                            minibatch_idx_in_repeat=mb_idx_in_repeat,
                            gradient_step=gradient_steps,
                            task_names_in_batch=dump_task_names,
                            task_indices=task_indices,
                            task_losses=dump_task_losses,
                            cached_task_full_grads=None,
                        )
                    self.optim.step(loss)

                with torch.no_grad():
                    if self._use_linear_baseline:
                        value_for_metrics = minibatch.v_s
                    else:
                        # Raw V so EV is reported in natural reward scale
                        # even under PopArt (POP keeps raw V stable).
                        value_for_metrics = self.critic(minibatch.obs).flatten()
                    # Under PopArt minibatch.returns is in normalized space;
                    # use cached raw returns so EV is comparable across runs.
                    returns_for_metric = (
                        raw_returns_for_metric
                        if (self._use_popart and raw_returns_for_metric is not None)
                        else minibatch.returns
                    )
                    approx_kl = (minibatch.logp_old - all_log_probs).mean()
                    clip_fraction = (
                        ratios.gt(1 + self.eps_clip) | ratios.lt(1 - self.eps_clip)
                    ).float().mean()
                    ev = explained_variance(
                        value_for_metrics.cpu().numpy(),
                        returns_for_metric.cpu().numpy(),
                    )

                    for i, task_name in enumerate(self._task_names):
                        mask = (task_indices == i)
                        if mask.sum() > 0:
                            kl = (minibatch.logp_old[mask] - all_log_probs[mask]).mean()
                            self._per_task_update_buffer[task_name]["approx_kl"].append(kl.item())

                            task_ratios = ratios[:, mask]
                            cf = (task_ratios.gt(1 + self.eps_clip) | task_ratios.lt(1 - self.eps_clip)).float().mean()
                            self._per_task_update_buffer[task_name]["clip_fraction"].append(cf.item())

                            task_val = value_for_metrics[mask]
                            task_ret = returns_for_metric[mask]
                            ev_task = explained_variance(task_val.cpu().numpy(), task_ret.cpu().numpy())
                            self._per_task_update_buffer[task_name]["explained_variance"].append(ev_task)

                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())
                approx_kls.append(approx_kl.item())
                clip_fractions.append(clip_fraction.item())
                explained_vars.append(ev)

        self._latest_data = {
            "values": value.detach().cpu(),
            "returns": minibatch.returns.detach().cpu(),
            "advantages": minibatch.adv.detach().cpu(),
        }

        return MultiTaskPPOTrainingStats(
            loss=SequenceSummaryStats.from_sequence(losses),
            actor_loss=SequenceSummaryStats.from_sequence(clip_losses),
            vf_loss=SequenceSummaryStats.from_sequence(vf_losses),
            ent_loss=SequenceSummaryStats.from_sequence(ent_losses),
            approx_kl=SequenceSummaryStats.from_sequence(approx_kls),
            clip_fraction=SequenceSummaryStats.from_sequence(clip_fractions),
            explained_variance=SequenceSummaryStats.from_sequence(explained_vars),
            gradient_steps=gradient_steps,
        )
