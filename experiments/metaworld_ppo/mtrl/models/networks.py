"""Multi-task neural network architectures (Tianshou-compatible).

Building blocks:
- ``SharedPreprocessNet`` — optional shared MLP at the obs frontier.
- ``MultiTaskActor`` — shared preprocess + actor body + per-task heads.
- ``MultiTaskCritic`` — shared preprocess + critic body + per-task heads,
  with optional PopArt renormalization on the per-task value heads.

All components are configurable via lists of hidden sizes; ``None`` or
``[]`` means identity / flatten.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tianshou.utils.net.common import (
    MLP,
    ModuleWithVectorOutput,
    AbstractContinuousActorProbabilistic,
)
from tianshou.utils.net.continuous import AbstractContinuousCritic


ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def _get_activation(name: str) -> type:
    if name.lower() not in ACTIVATION_MAP:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from: {list(ACTIVATION_MAP.keys())}"
        )
    return ACTIVATION_MAP[name.lower()]


def build_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    activation: str = "relu",
    output_dim: int = 0,
    use_layer_norm: bool = False,
) -> nn.Module:
    """Build an MLP from a list of hidden sizes.

    With ``use_layer_norm=True`` we insert pre-activation ``LayerNorm``
    between every hidden ``Linear`` and its activation, and drop the bias
    on those Linears — LN's mean-subtraction makes the bias gradient
    identically zero, and LN's own β already provides a post-projection
    bias.  The final output projection (no activation, no trailing LN)
    keeps ``bias=True``.
    """
    if not hidden_sizes and output_dim == 0:
        return nn.Identity()

    act_cls = _get_activation(activation)
    layers: List[nn.Module] = []
    prev_dim = input_dim

    for h in hidden_sizes:
        layers.append(nn.Linear(prev_dim, h, bias=not use_layer_norm))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act_cls())
        prev_dim = h

    if output_dim > 0:
        layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*layers)


class SharedPreprocessNet(ModuleWithVectorOutput):
    """Shared preprocessing network for multi-task learning.

    Processes raw observations through an MLP. This is shared across all tasks.

    Args:
        input_dim: Observation dimension (including one-hot task ID if used).
        hidden_sizes: List of hidden layer sizes. Empty = flatten/identity.
        activation: Activation function name.
        use_layer_norm: If True, add LayerNorm after every hidden Linear.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Optional[List[int]] = None,
        activation: str = "relu",
        use_layer_norm: bool = False,
    ) -> None:
        output_dim = hidden_sizes[-1] if hidden_sizes and len(hidden_sizes) > 0 else input_dim
        super().__init__(output_dim=output_dim)
        self._input_dim = input_dim

        if hidden_sizes and len(hidden_sizes) > 0:
            self._net = build_mlp(input_dim, hidden_sizes, activation, use_layer_norm=use_layer_norm)
            self._output_dim = hidden_sizes[-1]
        else:
            self._net = nn.Flatten()
            self._output_dim = input_dim
            # Add dummy parameter to track device
            self._dummy_param = nn.Parameter(torch.empty(0))

    def forward(
        self,
        obs: torch.Tensor | np.ndarray,
        state: Any = None,
        info: dict | None = None,
    ) -> Tuple[torch.Tensor, Any]:
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        # Move to model device if parameters exist
        try:
            device = next(self._net.parameters()).device
        except StopIteration:
            # Use dummy parameter if available
            if hasattr(self, "_dummy_param"):
                device = self._dummy_param.device
            else:
                # Should not happen given init logic, but safe fallback
                device = obs.device

        obs = obs.to(device)
            
        return self._net(obs), state

    def get_output_dim(self) -> int:
        return self._output_dim


class MultiTaskActor(AbstractContinuousActorProbabilistic):
    """Multi-task actor with shared body and per-task heads.

    Architecture::
        obs → preprocess_net → actor_body → per_task_head → (mu, sigma)

    ``conditioned_sigma=False`` uses a per-task learned parameter for σ;
    ``True`` adds per-task σ-heads.  LayerNorm flags target the body /
    head hidden Linears only — final output projections never get LN.
    """

    def __init__(
        self,
        *,
        preprocess_net: ModuleWithVectorOutput,
        action_shape: Sequence[int] | int | np.int64,
        task_names: List[str],
        actor_hidden_sizes: Optional[List[int]] = None,
        actor_head_hidden_sizes: Optional[List[int]] = None,
        activation: str = "relu",
        max_action: float = 1.0,
        unbounded: bool = False,
        conditioned_sigma: bool = False,
        body_use_layer_norm: bool = False,
        head_use_layer_norm: bool = False,
    ) -> None:
        # Call the grandparent (nn.Module) directly since we implement the
        # abstract interface ourselves.
        nn.Module.__init__(self)

        self.preprocess = preprocess_net
        self._action_dim = int(np.prod(action_shape))
        self._max_action = max_action
        self._unbounded = unbounded
        self._conditioned_sigma = conditioned_sigma
        self._task_names = task_names
        self._num_tasks = len(task_names)
        self._actor_head_hidden_sizes = actor_head_hidden_sizes
        self._last_shared_latent: torch.Tensor | None = None

        preprocess_dim = preprocess_net.get_output_dim()

        # Shared actor body
        if self._actor_head_hidden_sizes and len(self._actor_head_hidden_sizes) > 0:
            if actor_hidden_sizes and len(actor_hidden_sizes) > 0:
                self.actor_body = build_mlp(
                    preprocess_dim, actor_hidden_sizes, activation,
                    use_layer_norm=body_use_layer_norm,
                )
                body_out_dim = actor_hidden_sizes[-1]
            else:
                self.actor_body = nn.Identity()
                body_out_dim = preprocess_dim
            self.actor_output = nn.Identity()
        else:
            # Keep a learned shared latent even when there are no task-specific heads.
            output_dim = self._action_dim
            if conditioned_sigma:
                output_dim *= 2

            if actor_hidden_sizes and len(actor_hidden_sizes) > 0:
                self.actor_body = build_mlp(
                    preprocess_dim, actor_hidden_sizes, activation,
                    use_layer_norm=body_use_layer_norm,
                )
                body_out_dim = actor_hidden_sizes[-1]
            else:
                self.actor_body = nn.Identity()
                body_out_dim = preprocess_dim
            self.actor_output = nn.Linear(body_out_dim, output_dim)

        # Per-task heads for mu
        self.mu_heads = nn.ModuleDict()
        for name in task_names:
            if self._actor_head_hidden_sizes and len(self._actor_head_hidden_sizes) > 0:
                head = build_mlp(
                    body_out_dim, self._actor_head_hidden_sizes, activation, self._action_dim,
                    use_layer_norm=head_use_layer_norm,
                )
            else:
                head = nn.Identity()
            # Replace special chars in task names for ModuleDict keys
            self.mu_heads[name.replace("-", "_")] = head

        # Sigma (per-task or shared)
        if conditioned_sigma:
            self.sigma_heads = nn.ModuleDict()
            for name in task_names:
                if self._actor_head_hidden_sizes and len(self._actor_head_hidden_sizes) > 0:
                    head = build_mlp(
                        body_out_dim, self._actor_head_hidden_sizes, activation, self._action_dim,
                        use_layer_norm=head_use_layer_norm,
                    )
                else:
                    head = nn.Identity()
                self.sigma_heads[name.replace("-", "_")] = head
        else:
            self.sigma_param = nn.Parameter(torch.zeros(self._num_tasks, self._action_dim))

        # Build key mapping for task name -> module dict key
        self._key_map = {name: name.replace("-", "_") for name in task_names}
        self._task_name_to_id = {name: i for i, name in enumerate(task_names)}

        # Store output dim for compatibility
        self._output_dim = self._action_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def get_preprocess_net(self) -> ModuleWithVectorOutput:
        return self.preprocess

    @property
    def last_shared_latent(self) -> torch.Tensor | None:
        """Most recent shared actor latent from forward()."""
        return self._last_shared_latent

    def _extract_task_info(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """Decode (obs, per-sample task names, task ids) from a one-hot batch."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        task_one_hot = obs[:, -self._num_tasks:]
        task_ids = task_one_hot.argmax(dim=-1)
        task_ids_np = task_ids.detach().cpu().numpy()
        task_names_per_sample = [self._task_names[tid] for tid in task_ids_np]
        return obs, task_names_per_sample, task_ids

    def forward(
        self,
        obs: torch.Tensor | np.ndarray,
        state: Any = None,
        info: dict | None = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Forward pass through multi-task actor.

        Args:
            obs: Observations (with one-hot task ID appended).
            state: Recurrent state (unused for MLP).
            info: Additional info dict.

        Returns:
            Tuple of ((mu, sigma), state).
        """
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        obs = obs.to(next(self.parameters()).device)

        _, task_names_per_sample, task_ids = self._extract_task_info(obs)

        # Forward through shared networks
        features, state = self.preprocess(obs, state, info)
        features = self.actor_body(features)
        self._last_shared_latent = features

        if self._actor_head_hidden_sizes is None or len(self._actor_head_hidden_sizes) == 0:
            outputs = self.actor_output(features)
            if self._conditioned_sigma:
                mu, sigma = outputs.chunk(2, dim=-1)
                sigma = torch.clamp(sigma, min=-20, max=2).exp()
            else:
                mu = outputs
                sigma = torch.clamp(
                    self.sigma_param[task_ids], min=-20, max=2
                ).exp()

            if not self._unbounded:
                mu = self._max_action * torch.tanh(mu)

            return (mu, sigma), state

        # Compute mu and sigma per task using batch indexing
        batch_size = features.shape[0]
        mu = torch.zeros(batch_size, self._action_dim, device=features.device)
        sigma = torch.zeros(batch_size, self._action_dim, device=features.device)

        # Group samples by task for efficient computation
        task_groups: Dict[str, List[int]] = {}
        for i, tname in enumerate(task_names_per_sample):
            key = self._key_map[tname]
            if key not in task_groups:
                task_groups[key] = []
            task_groups[key].append(i)

        for key, indices in task_groups.items():
            idx = torch.tensor(indices, device=features.device)
            task_features = features[idx]
            mu[idx] = self.mu_heads[key](task_features)

            if self._conditioned_sigma:
                sigma[idx] = torch.clamp(
                    self.sigma_heads[key](task_features), min=-20, max=2
                ).exp()
            else:
                task_id = self._task_name_to_id[
                    self._task_names[
                        next(
                            i
                            for i, n in enumerate(self._task_names)
                            if self._key_map[n] == key
                        )
                    ]
                ]
                sigma[idx] = torch.clamp(
                    self.sigma_param[task_id], min=-20, max=2
                ).exp().unsqueeze(0).expand_as(task_features[:, :self._action_dim])

        if not self._unbounded:
            mu = self._max_action * torch.tanh(mu)

        return (mu, sigma), state


class MultiTaskCritic(AbstractContinuousCritic):
    """Multi-task critic with shared body, per-task heads, optional PopArt.

    Architecture::
        obs → preprocess_net → critic_body → per_task_head → V

    LayerNorm flags target hidden Linears only.  ``popart_enabled=True``
    attaches a per-task affine that runs in normalized target space; raw V
    output is preserved across stats updates by POP renormalization.
    """

    def __init__(
        self,
        *,
        preprocess_net: ModuleWithVectorOutput,
        task_names: List[str],
        critic_hidden_sizes: Optional[List[int]] = None,
        critic_head_hidden_sizes: Optional[List[int]] = None,
        activation: str = "relu",
        act_dim: int = 0,
        apply_preprocess_net_to_obs_only: bool = False,
        flatten_input: bool = True,
        popart_enabled: bool = False,
        popart_min_std: float = 1e-2,
        popart_method: Literal["welford", "ema"] = "welford",
        popart_ema_decay: float = 3e-4,
        body_use_layer_norm: bool = False,
        head_use_layer_norm: bool = False,
        hlgauss_num_bins: Optional[int] = None,
        hlgauss_vmin: Optional[float] = None,
        hlgauss_vmax: Optional[float] = None,
        hlgauss_sigma_ratio: float = 0.75,
    ) -> None:
        nn.Module.__init__(self)
        self.output_dim = 1  # Set output_dim for ModuleWithVectorOutput

        self.preprocess = preprocess_net
        self._apply_preprocess_net_to_obs_only = apply_preprocess_net_to_obs_only
        self._flatten_input = flatten_input
        self._task_names = task_names
        self._num_tasks = len(task_names)
        self._critic_head_hidden_sizes = critic_head_hidden_sizes
        self._use_popart = bool(popart_enabled)
        self._use_hlgauss = hlgauss_num_bins is not None

        if self._use_hlgauss:
            # PopArt + HL-Gauss is allowed and uses ART only: the per-task
            # (mu_i, sigma_i) statistics normalize the regression TARGET before
            # projection, which is what lets one fixed support serve 50 tasks.
            # The POP half (W, b renormalization preserving raw V across a
            # stats update) has no categorical analog -- the logits define a
            # distribution over a FIXED support, so a (mu, sigma) update
            # silently re-scales what those logits decode to.  That is a real
            # limitation of the composition and must be disclosed, not a bug.
            if critic_head_hidden_sizes and len(critic_head_hidden_sizes) > 0:
                raise ValueError(
                    "HL-Gauss critic requires empty critic_head: the shared "
                    "body's output layer already emits the bin logits."
                )
            if not (critic_hidden_sizes and len(critic_hidden_sizes) > 0):
                raise ValueError("HL-Gauss critic requires a non-empty critic body.")
            from mtrl.models.hlgauss import HLGaussHead
            self.hlgauss = HLGaussHead(
                num_bins=hlgauss_num_bins,
                vmin=hlgauss_vmin,
                vmax=hlgauss_vmax,
                sigma_ratio=hlgauss_sigma_ratio,
            )

        preprocess_dim = preprocess_net.get_output_dim()

        # When preprocess applies only to obs, actions are concatenated
        # after preprocessing, so the body sees preprocess_dim + act_dim.
        body_input_dim = preprocess_dim + act_dim if apply_preprocess_net_to_obs_only else preprocess_dim

        if self._use_hlgauss:
            # Shared body ends in Linear(hidden, num_bins); no per-task heads.
            self.critic_body = build_mlp(
                body_input_dim, critic_hidden_sizes, activation, hlgauss_num_bins,
                use_layer_norm=body_use_layer_norm,
            )
            self.value_heads = nn.ModuleDict(
                {name.replace("-", "_"): nn.Identity() for name in task_names}
            )
            self._key_map = {name: name.replace("-", "_") for name in task_names}
            self._output_dim = 1
            if self._use_popart:
                self.popart = PopArtModule(
                    self._num_tasks,
                    min_std=popart_min_std,
                    method=popart_method,
                    ema_decay=popart_ema_decay,
                )
            return

        # Shared critic body
        if self._critic_head_hidden_sizes and len(self._critic_head_hidden_sizes) > 0:
            if critic_hidden_sizes and len(critic_hidden_sizes) > 0:
                self.critic_body = build_mlp(
                    body_input_dim, critic_hidden_sizes, activation,
                    use_layer_norm=body_use_layer_norm,
                )
                body_out_dim = critic_hidden_sizes[-1]
            else:
                self.critic_body = nn.Identity()
                body_out_dim = body_input_dim
        else:
            if critic_hidden_sizes and len(critic_hidden_sizes) > 0:
                self.critic_body = build_mlp(
                    body_input_dim, critic_hidden_sizes, activation, 1,
                    use_layer_norm=body_use_layer_norm,
                )
                body_out_dim = 1
            else:
                self.critic_body = nn.Linear(body_input_dim, 1)
                body_out_dim = 1

        # Per-task value heads
        self.value_heads = nn.ModuleDict()
        for name in task_names:
            if self._critic_head_hidden_sizes and len(self._critic_head_hidden_sizes) > 0:
                head = build_mlp(
                    body_out_dim, self._critic_head_hidden_sizes, activation, 1,
                    use_layer_norm=head_use_layer_norm,
                )
            else:
                head = nn.Identity()
            self.value_heads[name.replace("-", "_")] = head

        self._key_map = {name: name.replace("-", "_") for name in task_names}
        self._output_dim = 1

        # When disabled, popart attribute is omitted so state_dict() carries
        # no PopArt keys.
        if self._use_popart:
            self.popart = PopArtModule(
                self._num_tasks,
                min_std=popart_min_std,
                method=popart_method,
                ema_decay=popart_ema_decay,
            )

    def get_output_dim(self) -> int:
        return self._output_dim

    def _extract_task_ids(self, obs: torch.Tensor) -> List[str]:
        """Decode per-sample task names from a one-hot batch."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        task_ids = obs[:, -self._num_tasks:].argmax(dim=-1).detach().cpu().numpy()
        return [self._task_names[tid] for tid in task_ids]

    def _body_features(
        self,
        obs: np.ndarray | torch.Tensor,
        act: np.ndarray | torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """preprocess → critic_body output, per-sample task ids, prepped obs."""
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if act is not None and isinstance(act, np.ndarray):
            act = torch.as_tensor(act, dtype=torch.float32)
        device = next(self.parameters()).device
        obs = obs.to(device)
        if act is not None:
            act = act.to(device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Forward through shared preprocess
        if self._apply_preprocess_net_to_obs_only:
            features, _ = self.preprocess(obs)
            if act is not None:
                features = torch.cat([features, act], dim=-1)
        else:
            if act is not None:
                input_tensor = torch.cat([obs, act], dim=-1)
            else:
                input_tensor = obs
            features, _ = self.preprocess(input_tensor)

        features = self.critic_body(features)
        task_idx = obs[:, -self._num_tasks:].argmax(dim=-1)
        return features, task_idx, obs

    def forward_logits(
        self,
        obs: np.ndarray | torch.Tensor,
        act: np.ndarray | torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """HL-Gauss bin logits ``(B, num_bins)`` + per-sample task ids."""
        if not self._use_hlgauss:
            raise RuntimeError("forward_logits requires the HL-Gauss critic.")
        logits, task_idx, _ = self._body_features(obs, act)
        return logits, task_idx

    def _compute_v_pre(
        self,
        obs: np.ndarray | torch.Tensor,
        act: np.ndarray | torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pre-PopArt scalar value prediction + per-sample task ids."""
        if self._use_hlgauss:
            raise RuntimeError(
                "_compute_v_pre is scalar-only; the HL-Gauss critic goes "
                "through forward()/forward_logits()."
            )
        features, task_idx, obs_prepped = self._body_features(obs, act)

        if self._critic_head_hidden_sizes is None or len(self._critic_head_hidden_sizes) == 0:
            # No per-task head: critic_body's final Linear → scalar already.
            return features.flatten(), task_idx

        # Per-task value computation via per-task heads.
        batch_size = features.shape[0]
        values = torch.zeros(batch_size, 1, device=features.device)
        task_names_per_sample = self._extract_task_ids(obs_prepped)
        task_groups: Dict[str, List[int]] = {}
        for i, tname in enumerate(task_names_per_sample):
            key = self._key_map[tname]
            if key not in task_groups:
                task_groups[key] = []
            task_groups[key].append(i)
        for key, indices in task_groups.items():
            idx = torch.tensor(indices, device=features.device)
            task_features = features[idx]
            values[idx] = self.value_heads[key](task_features)
        return values.flatten(), task_idx

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        act: np.ndarray | torch.Tensor | None = None,
        info: dict | None = None,
    ) -> torch.Tensor:
        """Raw value prediction (PopArt-denormalized when enabled).

        Returning raw V keeps GAE / advantage / value-clip math in the
        natural reward scale; POP guarantees raw V is invariant across
        PopArt stats updates.
        """
        if self._use_hlgauss:
            logits, task_idx = self.forward_logits(obs, act)
            v = self.hlgauss.from_logits(logits)
            # MultiTaskPPO attaches its ReturnScalingModule as ``self.retscale``
            # when critic_loss_mode="hlgauss" is combined with
            # ppo.use_return_scaling; it is absent in every other case.  Read
            # it with getattr and keep no class-level ``retscale = None``
            # default: nn.Module stores an assigned submodule in ``_modules``,
            # which attribute lookup consults only after the class, so such a
            # default would shadow the attached module.
            retscale = getattr(self, "retscale", None)
            if retscale is not None:
                # Under return-based scaling the categorical head is trained on
                # target / s_i, so its readout is in scaled units.  GAE,
                # advantages and evaluation all expect raw units, so undo it
                # here — the loss path does the same for its own logging
                # readout (multitask_ppo.py, hlgauss branch).
                scale = retscale.last_scale.to(
                    device=v.device, dtype=v.dtype,
                )[task_idx]
                v = v * scale
            if self._use_popart:
                # Readout lives in ART-normalized space; map back to raw so
                # GAE / advantages / eval stay in the natural reward scale.
                v = self.popart.denormalize(v, task_idx)
            return v.unsqueeze(-1)
        v_pre, task_idx = self._compute_v_pre(obs, act)
        if self._use_popart:
            v_norm = self.popart.affine(v_pre, task_idx)
            v_raw = self.popart.denormalize(v_norm, task_idx)
            return v_raw.unsqueeze(-1)
        return v_pre.unsqueeze(-1)

    def forward_normalized(
        self,
        obs: np.ndarray | torch.Tensor,
        act: np.ndarray | torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-task affine prediction ``W_i·v_pre + b_i`` (no σ/μ).

        Used by the PopArt critic loss path.  When PopArt is off this is
        the plain pre-PopArt scalar so callers share one API.
        """
        v_pre, task_idx = self._compute_v_pre(obs, act)
        if self._use_popart:
            return self.popart.affine(v_pre, task_idx), task_idx
        return v_pre, task_idx


class PopArtModule(nn.Module):
    """Per-task PopArt rescaler (van Hasselt 2016 / Hessel et al., AAAI 2019).

    Two halves of "Multi-task Deep RL with PopArt":

    - **ART**: per-task running mean ``μ_i`` and std ``σ_i`` of returns;
      critic loss runs against ``(R - μ_i) / σ_i``.  Two estimators:

      * ``method="welford"`` (default) — unbounded online mean/variance via
        a Welford merge of every per-task batch ever seen.  Tracks the
        *cumulative* return distribution; this repo's original behavior.
      * ``method="ema"`` — the original-paper exponential moving average,
        ``μ ← (1-β)μ + βY``, ``ν ← (1-β)ν + βY²``,
        ``σ = √max(ν - μ², 0)``.  Recency-weighted; matches the canonical
        multi-task PopArt reference.

    - **POP**: when ``(μ_i, σ_i)`` change, the per-task affine layer
      ``(W_i, b_i)`` is renormalized so the raw output
      ``V_raw = σ_i · (W_i·v_pre + b_i) + μ_i`` is unchanged for every
      input ``v_pre``.  Identical for both estimators.

    State (shape ``(num_tasks,)``, all captured by ``state_dict()``):

    - ``mu, sigma``  — fp64 buffers, used by both estimators.
    - ``M2, count``  — Welford state; mutated only when ``method="welford"``,
      always registered so the state_dict shape does not depend on the
      method choice.
    - ``nu``         — raw second moment ``E[G²]``; mutated only when
      ``method="ema"``, always registered for the same reason.  Init 1 so
      ``σ = √(ν - μ²) = 1`` at construction, matching the Welford init.
    - ``W, b``       — parameters, init ``W=1, b=0`` so the identity start
      matches PopArt-off.

    Args:
        num_tasks: Number of tasks (length of the one-hot tail).
        min_std: Floor on per-task σ to avoid div-by-zero at training start.
        method: ``"welford"`` (this repo's original) or ``"ema"`` (reference).
        ema_decay: Per-sample EMA β, consumed only when ``method="ema"``;
            default 3e-4 matches Hessel et al. 2019 multi-task PopArt.
    """

    def __init__(
        self,
        num_tasks: int,
        min_std: float = 1e-2,
        method: Literal["welford", "ema"] = "welford",
        ema_decay: float = 3e-4,
    ) -> None:
        super().__init__()
        if not (min_std > 0.0):
            raise ValueError(
                f"PopArt min_std must be > 0 (got {min_std!r}); a non-positive "
                "floor leaves σ at 0 on tasks whose initial returns have zero "
                "variance, and the per-sample normalize_target / minibatch "
                "rescale would divide by zero."
            )
        if method not in ("welford", "ema"):
            raise ValueError(
                f"PopArt method must be 'welford' or 'ema' (got {method!r})."
            )
        if method == "ema":
            # Strict open interval: β=0 freezes stats forever, β=1 erases
            # history every sample (σ collapses to 0 after a single point).
            # NaN fails this comparison, so it is caught by the same branch.
            if not (0.0 < ema_decay < 1.0):
                raise ValueError(
                    f"PopArt ema_decay must be in (0, 1) (got {ema_decay!r}); "
                    "0 freezes stats, 1 collapses σ to 0 on every update."
                )
        self.num_tasks = int(num_tasks)
        self.min_std = float(min_std)
        self.method = method
        self.ema_decay = float(ema_decay)
        # fp64 buffers for stable Welford merges and for the EMA's ν-μ²
        # subtraction; fp32 params share dtype with the rest of the critic
        # and the autograd graph.
        self.register_buffer("mu",    torch.zeros(num_tasks, dtype=torch.float64))
        self.register_buffer("sigma", torch.ones (num_tasks, dtype=torch.float64))
        # Welford state — mutated only by _merge_welford.  Always registered
        # so the state_dict shape is fixed across method choices.
        self.register_buffer("M2",    torch.zeros(num_tasks, dtype=torch.float64))
        self.register_buffer("count", torch.zeros(num_tasks, dtype=torch.int64))
        # EMA second-moment state — mutated only by _merge_ema.  Init to 1.0
        # so σ = √max(ν - μ², 0) = 1 at construction (matches Welford σ=1).
        self.register_buffer("nu",    torch.ones (num_tasks, dtype=torch.float64))
        self.W = nn.Parameter(torch.ones(num_tasks, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32))

    def affine(self, v_pre: torch.Tensor, task_idx: torch.Tensor) -> torch.Tensor:
        """Per-task affine ``W_i · v_pre + b_i`` (the *normalized* prediction)."""
        return self.W[task_idx] * v_pre + self.b[task_idx]

    def denormalize(self, v_norm: torch.Tensor, task_idx: torch.Tensor) -> torch.Tensor:
        """Map normalized prediction back to raw scale: ``σ_i · v_norm + μ_i``."""
        sigma = self.sigma[task_idx].to(v_norm.dtype)
        mu    = self.mu[task_idx].to(v_norm.dtype)
        return sigma * v_norm + mu

    def normalize_target(
        self, returns: torch.Tensor, task_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize per-task returns into the PopArt target space."""
        sigma = self.sigma[task_idx].to(returns.dtype)
        mu    = self.mu[task_idx].to(returns.dtype)
        return (returns - mu) / sigma

    @torch.no_grad()
    def update_and_renormalize(
        self, returns_per_task: Dict[int, torch.Tensor],
    ) -> None:
        """Merge new per-task returns into the running stats, then POP-rescale.

        Dispatches to :meth:`_merge_welford` or :meth:`_merge_ema` per task
        according to ``self.method``.  Tasks absent from ``returns_per_task``
        (or with empty tensors) are left untouched.  POP guarantees raw V is
        invariant across the (μ, σ) update under either estimator.
        """
        for i, R in returns_per_task.items():
            if R is None:
                continue
            if int(R.numel()) == 0:
                continue
            mu_old    = self.mu[i].clone()
            sigma_old = self.sigma[i].clone()

            if self.method == "welford":
                self._merge_welford(i, R)
            else:  # "ema"
                self._merge_ema(i, R)

            # POP renormalization preserves raw V across the stats update.
            # Identical for both estimators.
            ratio = (sigma_old / self.sigma[i]).to(self.W.dtype)
            self.W.data[i] = ratio * self.W.data[i]
            new_b = (
                sigma_old * self.b.data[i].to(torch.float64) + mu_old - self.mu[i]
            ) / self.sigma[i]
            self.b.data[i] = new_b.to(self.b.dtype)

    def _merge_welford(self, i: int, R: torch.Tensor) -> None:
        """Welford incremental merge of one task's batch into (μ, σ, M2, count)."""
        R64 = R.detach().to(dtype=torch.float64, device=self.mu.device)
        n_batch = int(R64.numel())
        batch_mean = R64.mean()
        batch_M2 = ((R64 - batch_mean) ** 2).sum()

        n_old = int(self.count[i].item())
        n_new = n_old + n_batch

        delta = batch_mean - self.mu[i]
        self.mu[i]    = self.mu[i] + delta * n_batch / n_new
        self.M2[i]    = (
            self.M2[i] + batch_M2 + delta.pow(2) * n_old * n_batch / n_new
        )
        self.sigma[i] = (self.M2[i] / n_new).sqrt().clamp(min=self.min_std)
        self.count[i] = n_new

    def _merge_ema(self, i: int, R: torch.Tensor) -> None:
        """Per-sample EMA update, vectorized closed form.

        For a task batch ``G_0..G_{N-1}`` (in the rollout order preserved by
        boolean-mask indexing) this is the exact equivalent of N sequential
        per-sample EMA updates::

            μ_N = (1-β)^N μ_0 + β Σ_j (1-β)^{N-1-j} G_j
            ν_N = (1-β)^N ν_0 + β Σ_j (1-β)^{N-1-j} G_j²

        ``log1p(-β)`` avoids catastrophic cancellation at small β; the
        ``clamp_min(0.0)`` on ``ν - μ²`` guards the sub-ulp negative that EMA
        can produce when a task's return distribution collapses to a
        near-point mass.
        """
        G = R.detach().to(dtype=torch.float64, device=self.mu.device)
        N = int(G.numel())
        beta = self.ema_decay
        log1mb = math.log1p(-beta)
        decay  = math.exp(log1mb * N)              # (1-β)^N
        exps = torch.arange(N - 1, -1, -1, dtype=torch.float64, device=G.device)
        w    = torch.exp(log1mb * exps)            # (1-β)^{N-1-j}
        s1 = (w * G).sum()
        s2 = (w * G.pow(2)).sum()
        self.mu[i] = decay * self.mu[i] + beta * s1
        self.nu[i] = decay * self.nu[i] + beta * s2
        var_i = (self.nu[i] - self.mu[i].pow(2)).clamp_min(0.0)
        self.sigma[i] = var_i.sqrt().clamp(min=self.min_std)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Backward-compat shim for PopArt checkpoints written before ``nu``.

        Those checkpoints carry six PopArt tensors (``mu, sigma, M2, count,
        W, b``), so loading them into this module under the strict default
        would fail with ``Missing key(s) in state_dict: "...popart.nu"``.
        We synthesize ``nu = σ² + μ²``, which is exact for a Welford run
        whose σ was not raised by the ``min_std`` clamp (σ² = M2/count =
        E[G²] - μ²), and gives a Welford→EMA switch the empirical second
        moment as its starting ν.  Checkpoints that do contain ``nu`` skip
        this branch and load it verbatim.
        """
        nu_key    = prefix + "nu"
        sigma_key = prefix + "sigma"
        mu_key    = prefix + "mu"
        if nu_key not in state_dict and sigma_key in state_dict and mu_key in state_dict:
            sigma = state_dict[sigma_key].to(torch.float64)
            mu    = state_dict[mu_key].to(torch.float64)
            state_dict[nu_key] = sigma.pow(2) + mu.pow(2)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def get_diagnostics(self) -> Dict[str, torch.Tensor]:
        """Detached CPU snapshot of per-task μ, σ, W, b."""
        return {
            "mu":    self.mu.detach().cpu().clone(),
            "sigma": self.sigma.detach().cpu().clone(),
            "W":     self.W.detach().cpu().clone(),
            "b":     self.b.detach().cpu().clone(),
        }
