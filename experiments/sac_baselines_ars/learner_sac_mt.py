"""Unified multi-task (multi-head) SAC learner.

Upstream (ms-cho/ARS) learner extended with per-side gradient-combiner
selection and the TOPPO critic-balancing stack:

  - ``actor_method``  : "default" | "pcgrad"
  - ``critic_method`` : "default" | "pcgrad" | "fairgrad"  (FG-c, alpha=1)
  - ``use_popart``    : PopArt per-task value-target normalization (with
                        LN-c via ``critic_layernorm`` this completes the
                        SAC port of the paper's critic stack)

The legacy ``use_pcgrad=True`` maps to pcgrad on BOTH sides (bit-equivalent
to upstream's ``_update_jit_pcgrad`` flow); all-default flags reproduce the
upstream ``_update_jit`` flow exactly.
"""

from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

import policy
import popart_jax
import value_net
from actor import sac_update_actor, sac_update_actor_pcgrad
from common import (
    Batch,
    InfoDict,
    Model,
    MultiModel,
    MultiModelFairGrad,
    MultiModelPCGrad,
    PRNGKey,
    activation_func,
)
from critic import (
    sac_update_alpha,
    sac_update_q,
    sac_update_q_pcgrad,
    sac_update_q_popart,
)


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params
    )
    return target_critic.replace(params=new_target_params)


def make_update_fn(
    actor_method: str,
    critic_method: str,
    use_popart: bool,
    popart_method: str,
    popart_ema_decay: float,
    popart_min_std: float,
):
    """Build the jitted update; method flags are static via closure."""
    per_task_critic = critic_method in ("pcgrad", "fairgrad")

    @jax.jit
    def _update(
        rng: PRNGKey,
        actor,
        critic,
        target_critic,
        alpha,
        popart_state,
        batch: Batch,
        discount: float,
        tau: float,
        target_entropy: float,
    ):
        key, rng = jax.random.split(rng)
        if use_popart:
            new_critic, target_critic, new_popart, critic_info = sac_update_q_popart(
                key,
                actor,
                critic,
                target_critic,
                alpha,
                batch,
                discount,
                popart_state,
                popart_method,
                popart_ema_decay,
                popart_min_std,
                per_task_critic,
            )
        elif per_task_critic:
            new_critic, critic_info = sac_update_q_pcgrad(
                key, actor, critic, target_critic, alpha, batch, discount
            )
            new_popart = popart_state
        else:
            new_critic, critic_info = sac_update_q(
                key, actor, critic, target_critic, alpha, batch, discount
            )
            new_popart = popart_state

        key, rng = jax.random.split(rng)
        if actor_method == "pcgrad":
            new_actor, actor_info = sac_update_actor_pcgrad(
                key, actor, new_critic, alpha, batch
            )
        else:
            new_actor, actor_info = sac_update_actor(
                key, actor, new_critic, alpha, batch
            )
        new_alpha, alpha_info = sac_update_alpha(
            alpha, actor_info["log_pi"], target_entropy
        )

        new_target_critic = target_update(new_critic, target_critic, tau)

        return (
            rng,
            new_actor,
            new_critic,
            new_target_critic,
            new_alpha,
            new_popart,
            {**critic_info, **actor_info, **alpha_info},
        )

    return _update


class Learner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (400, 400, 400, 400),
        discount: float = 0.99,
        tau: float = 0.005,
        dropout_rate: Optional[float] = None,
        n_task: int = 1,
        n_critic: int = 2,
        softplus: bool = True,
        critic_layernorm: bool = False,
        critic_init_layernorm: bool = False,
        activation: str = "tanh",
        multi_head: bool = False,
        use_pcgrad: bool = False,
        actor_method: str = "default",
        critic_method: str = "default",
        fairgrad_critic_alpha: float = 1.0,
        use_popart: bool = False,
        popart_method: str = "ema",
        popart_ema_decay: float = 3e-4,
        popart_min_std: float = 1e-2,
    ):
        """Multi-task SAC learner.

        ``use_pcgrad=True`` (legacy) forces pcgrad on both sides.  The
        critic-balancing arm is ``critic_method="fairgrad"`` +
        ``use_popart=True`` + ``critic_layernorm=True``.
        """

        if use_pcgrad:
            actor_method = "pcgrad"
            critic_method = "pcgrad"
        if actor_method not in ("default", "pcgrad"):
            raise ValueError(f"actor_method {actor_method!r}")
        if critic_method not in ("default", "pcgrad", "fairgrad"):
            raise ValueError(f"critic_method {critic_method!r}")

        self.tau = tau
        self.discount = discount
        self.n_task = n_task
        self.use_pcgrad = use_pcgrad
        self.actor_method = actor_method
        self.critic_method = critic_method
        self.use_popart = use_popart

        critic_base_opt = optax.adam
        actor_base_opt = optax.adam
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, alpha_key = jax.random.split(rng, 4)
        actor_p_key = critic_p_key = None
        if actor_method == "pcgrad" or critic_method == "pcgrad":
            # split exactly as upstream so use_pcgrad runs stay seed-comparable
            rng, actor_p_key, critic_p_key = jax.random.split(rng, 3)

        action_dim = actions.shape[-1]
        self.target_entropy = -float(action_dim)

        actor_cls = (
            policy.NormalTanhMulitHeadPolicy if multi_head else policy.NormalTanhPolicy
        )
        actor_kwargs = {
            "hidden_dims": hidden_dims,
            "action_dim": action_dim,
            "log_std_min": -5.0,
            "dropout_rate": dropout_rate,
            "state_dependent_std": True,
            "tanh_squash_distribution": True,
            "activations": nn.relu,
            "name": "actor",
            "softplus": softplus,
        }
        init_observations = observations
        init_actions = actions
        if multi_head:
            actor_kwargs["n_task"] = n_task
            init_observations = observations[np.newaxis]
            init_actions = actions[np.newaxis]
        actor_def = actor_cls(**actor_kwargs)

        def _create_model(model_def, inputs, method, tx=None, rng_key=None):
            if method == "pcgrad":
                return MultiModelPCGrad.create(
                    model_def, inputs=inputs, tx=tx, n_task=n_task, rng=rng_key
                )
            if method == "fairgrad":
                return MultiModelFairGrad.create(
                    model_def,
                    inputs=inputs,
                    tx=tx,
                    n_task=n_task,
                    fairgrad_alpha=fairgrad_critic_alpha,
                )
            return MultiModel.create(model_def, inputs=inputs, tx=tx, n_task=n_task)

        actor_opt = actor_base_opt(learning_rate=actor_lr)
        actor = _create_model(
            actor_def,
            inputs=[actor_key, init_observations],
            method=actor_method,
            tx=actor_opt,
            rng_key=actor_p_key,
        )

        alpha_def = value_net.Alpha(name="alpha", init_value=jnp.ones((n_task, 1)))
        alpha = MultiModel.create(
            alpha_def,
            inputs=[alpha_key],
            tx=optax.adam(learning_rate=alpha_lr),
            n_task=n_task,
        )

        critic_def = value_net.EnsembleCritic(
            hidden_dims,
            activations=activation_func(activation),
            n_ensemble=n_critic,
            name="critic",
            layer_norm=critic_layernorm,
            init_layer_norm=critic_init_layernorm,
            popart=use_popart,
            n_task=n_task,
        )
        critic = _create_model(
            critic_def,
            inputs=[critic_key, init_observations, init_actions],
            method=critic_method,
            tx=critic_base_opt(learning_rate=critic_lr),
            rng_key=critic_p_key,
        )
        target_critic = _create_model(
            critic_def,
            inputs=[critic_key, init_observations, init_actions],
            method=critic_method,
            rng_key=critic_p_key,
        )
        self.actor = actor
        self.alpha = alpha
        self.critic = critic
        self.models = {"alpha": self.alpha, "actor": self.actor, "critic": self.critic}
        self.target_critic = target_critic
        self.popart_state = popart_jax.init_state(n_task)

        self._update_fn = make_update_fn(
            actor_method,
            critic_method,
            use_popart,
            popart_method,
            popart_ema_decay,
            popart_min_std,
        )
        self.rng = rng

    def sample_actions(
        self,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = "log_prob",
        params=None,
    ) -> jnp.ndarray:
        actor_params = params or self.actor.params

        rng, actions = policy.sample_actions(
            self.rng,
            self.actor.apply_fn,
            actor_params,
            observations,
            temperature,
            distribution,
        )

        actions = np.squeeze(actions)
        actions = np.asarray(actions)

        self.rng = rng
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        (
            new_rng,
            new_actor,
            new_critic,
            new_target_critic,
            new_alpha,
            new_popart,
            info,
        ) = self._update_fn(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.alpha,
            self.popart_state,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
        )

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.alpha = new_alpha
        self.popart_state = new_popart
        self.models = {"alpha": self.alpha, "actor": self.actor, "critic": self.critic}

        return info
