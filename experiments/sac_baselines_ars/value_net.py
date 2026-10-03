from typing import Callable, Sequence, Optional, Dict

import jax.numpy as jnp
from flax import linen as nn

from common import (
    MLP,
    Dtype,
    LoRAMulti_MLP,
    ModularGatedNet,
    OrthogonalLayer1D,
    default_init,
)


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_rate: Optional[float] = -1
    layer_norm: bool = False
    init_layer_norm: bool = False
    dtype: Optional[Dtype] = jnp.float32

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP(
            (*self.hidden_dims, 1),
            activations=self.activations,
            dropout_rate=self.dropout_rate,
            layer_norm=self.layer_norm,
            init_layer_norm=self.init_layer_norm,
            dtype=self.dtype,
        )(inputs)

        return jnp.squeeze(critic, -1)


class EnsembleCritic(nn.Module):
    hidden_dim: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    n_ensemble: int = 10
    dropout_rate: Optional[float] = -1
    layer_norm: bool = False
    init_layer_norm: bool = False
    dtype: Optional[Dtype] = jnp.float32
    name: str = "critic"
    # PopArt per-task affine head (critic balancing): when on, the module
    # output is the NORMALIZED q̂ — raw Q = sigma_i * q̂ + mu_i with (mu, sigma)
    # tracked in the Learner.  init W=1, b=0 => identity, bit-identical to
    # popart=False at construction.  The affine is shared across the ensemble
    # (POP rescaling in the learner adjusts these two params in-place).
    popart: bool = False
    n_task: int = 1

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_ensemble,
        )

        q_values = ensemble(
            self.hidden_dim,
            self.activations,
            self.dropout_rate,
            self.layer_norm,
            self.init_layer_norm,
            self.dtype,
        )(observations, actions)
        if self.popart:
            w = self.param("popart_w", nn.initializers.ones, (self.n_task,))
            b = self.param("popart_b", nn.initializers.zeros, (self.n_task,))
            if q_values.ndim == 3:  # (n_ensemble, n_task, batch)
                q_values = q_values * w[None, :, None] + b[None, :, None]
            # 2-D init trace (dummy (1, obs_dim) inputs): identity at init
            # (W=1, b=0) — params are registered above either way.
        return q_values


class LoRACritic(nn.Module):
    hidden_dims: Sequence[int]
    rank: int
    alpha: float = 1.0
    n_task: int = 1
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    init_layer_norm: bool = False
    layer_norm: bool = False
    dropout_rate: Optional[float] = -1

    @nn.compact
    def __call__(
        self,
        base_params: Optional[Dict],
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = LoRAMulti_MLP(
            (*self.hidden_dims, 1),
            rank=self.rank,
            alpha=self.alpha,
            n_task=self.n_task,
            activations=self.activations,
            base_params=base_params,
            dropout_rate=self.dropout_rate,
            layer_norm=self.layer_norm,
            init_layer_norm=self.init_layer_norm,
        )(inputs)

        return jnp.squeeze(critic, -1)


class EnsembleLoRACritic(nn.Module):
    hidden_dims: Sequence[int]
    rank: int
    scale: jnp.array
    alpha: float = 1.0
    n_task: int = 1
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    n_ensemble: int = 10
    base_params: Optional[Dict] = None
    init_layer_norm: bool = False
    layer_norm: bool = False
    dropout_rate: Optional[float] = -1
    name: str = "critic"

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        ensemble = nn.vmap(
            target=LoRACritic,
            in_axes=(0, None, None),
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_ensemble,
        )

        q_values = ensemble(
            hidden_dims=self.hidden_dims,
            rank=self.rank,
            alpha=self.alpha,
            n_task=self.n_task,
            activations=self.activations,
            init_layer_norm=self.init_layer_norm,
            layer_norm=self.layer_norm,
            dropout_rate=self.dropout_rate,
        )(self.base_params, observations, actions)
        q_values = q_values / jnp.reshape(self.scale, (1, self.n_task, 1))
        return q_values


class ModularGatedCritic(nn.Module):
    base_hidden_dims: Sequence[int]
    em_hidden_dims: Sequence[int]
    num_layers: int
    num_modules: int
    module_hidden: int
    gating_hidden: int
    num_gating_layers: int
    n_task: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations[..., : -self.n_task], actions], -1)
        critic = ModularGatedNet(
            1,
            self.base_hidden_dims,
            self.em_hidden_dims,
            self.num_layers,
            self.num_modules,
            self.module_hidden,
            self.gating_hidden,
            self.num_gating_layers,
            activations=self.activations,
        )(inputs, observations[..., -self.n_task :])

        return jnp.squeeze(critic, -1)


class EnsembleModularGatedCritic(nn.Module):
    base_hidden_dims: Sequence[int]
    em_hidden_dims: Sequence[int]
    num_layers: int
    num_modules: int
    module_hidden: int
    gating_hidden: int
    num_gating_layers: int
    n_task: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    n_ensemble: int = 2
    name: str = "critic"

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        ensemble = nn.vmap(
            target=ModularGatedCritic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_ensemble,
        )

        q_values = ensemble(
            self.base_hidden_dims,
            self.em_hidden_dims,
            self.num_layers,
            self.num_modules,
            self.module_hidden,
            self.gating_hidden,
            self.num_gating_layers,
            self.n_task,
            self.activations,
        )(observations, actions)
        return q_values


class CriticMixtureMH(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    n_task: int = 1
    n_expert: int = 4
    dropout_rate: Optional[float] = -1
    layer_norm: bool = False
    init_layer_norm: bool = False
    dtype: Optional[Dtype] = jnp.float32

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        n_task, batch_size, obs_dim = observations.shape
        inputs = jnp.concatenate([observations, actions], -1)

        ensemble = nn.vmap(
            target=MLP,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_expert,
        )
        hs = ensemble(  # shape => [n_task, batch_size, n_expert, hidden_dims[-1]]
            (*self.hidden_dims,),
            self.activations,
            False,
            self.dropout_rate,
            self.layer_norm,
            self.init_layer_norm,
            self.dtype,
        )(inputs)
        features = OrthogonalLayer1D()(hs)
        # features = OrthogonalLayerQR()(hs)

        task_context = jnp.repeat(
            jnp.expand_dims(jnp.eye(self.n_task)[jnp.arange(n_task)], axis=-2),
            batch_size,
            axis=1,
        )
        w = nn.Dense(  # shape => [n_task, batch_size, n_expert]
            self.n_expert, dtype=self.dtype, kernel_init=default_init(), use_bias=False
        )(task_context)

        features = jnp.squeeze(jnp.expand_dims(w, axis=-2) @ features, axis=-2)
        features = nn.tanh(features)  # shape => [n_task, batch_size, hidden_dims[-1]]

        q_values = nn.Dense(self.n_task, dtype=self.dtype, kernel_init=default_init())(
            features
        )  # shape => [n_task, batch_size, n_task]
        task_indices = jnp.arange(observations.shape[0])
        q = q_values[task_indices, :, task_indices]

        return q


class EnsembleCriticMixtureMH(nn.Module):
    hidden_dim: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    n_ensemble: int = 10
    n_task: int = 1
    n_expert: int = 4
    dropout_rate: Optional[float] = -1
    layer_norm: bool = False
    init_layer_norm: bool = False
    dtype: Optional[Dtype] = jnp.float32
    name: str = "critic"

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        ensemble = nn.vmap(
            target=CriticMixtureMH,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_ensemble,
        )

        q_values = ensemble(
            self.hidden_dim,
            self.activations,
            self.n_task,
            self.n_expert,
            self.dropout_rate,
            self.layer_norm,
            self.init_layer_norm,
            self.dtype,
        )(observations, actions)
        return q_values


class Alpha(nn.Module):
    init_value: jnp.ndarray = jnp.ones((1, 1))
    dtype: Optional[Dtype] = jnp.float32
    name: str = "alpha"

    @nn.compact
    def __call__(self):
        log_alpha = self.param("log_alpha", lambda key: jnp.log(self.init_value))
        return jnp.asarray(jnp.exp(log_alpha), dtype=self.dtype)
