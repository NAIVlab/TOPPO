"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

import policy
import value_net
from actor import moore_update_actor
from common import Batch, InfoDict, Model, PRNGKey, MultiModel, activation_func
from critic import sac_update_alpha, sac_update_q
import flax.linen as nn


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params
    )

    return target_critic.replace(params=new_target_params)


@jax.jit
def _update_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    alpha: Model,
    batch: Batch,
    discount: float,
    tau: float,
    target_entropy: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    key, rng = jax.random.split(rng)
    new_actor, actor_info = moore_update_actor(key, actor, critic, alpha, batch)
    new_alpha, alpha_info = sac_update_alpha(
        alpha, actor_info["log_pi"], target_entropy
    )

    key, rng = jax.random.split(rng)
    new_critic, critic_info = sac_update_q(
        key, new_actor, critic, target_critic, new_alpha, batch, discount
    )

    new_target_critic = target_update(new_critic, target_critic, tau)

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic,
        new_alpha,
        {**critic_info, **actor_info, **alpha_info},
    )


@jax.jit
def _update_critic_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    alpha: Model,
    batch: Batch,
    discount: float,
    tau: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    key, rng = jax.random.split(rng)
    new_critic, critic_info = sac_update_q(
        key, actor, critic, target_critic, alpha, batch, discount
    )

    new_target_critic = target_update(new_critic, target_critic, tau)

    return (
        rng,
        new_critic,
        new_target_critic,
        {**critic_info},
    )


class Learner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 1e-4,
        hidden_dims: Sequence[int] = (400, 400, 400),
        discount: float = 0.99,
        tau: float = 0.005,
        dropout_rate: Optional[float] = None,
        n_task: int = 1,
        n_expert: int = 4,
        n_critic: int = 2,
        softplus: bool = False,
        activation: str = "relu",
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        self.tau = tau
        self.discount = discount
        self.n_task = n_task

        critic_base_opt = optax.adam
        actor_base_opt = optax.adam
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, alpha_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        self.target_entropy = -float(action_dim)

        actor_def = policy.NormalTanhMixtureMHPolicy(
            hidden_dims,
            action_dim,
            n_task=n_task,
            n_expert=n_expert,
            log_std_min=-10.0,
            log_std_max=2.0,
            dropout_rate=dropout_rate,
            state_dependent_std=True,
            tanh_squash_distribution=True,
            activations=nn.relu,
            name="actor",
            softplus=softplus,
        )

        actor_opt = actor_base_opt(learning_rate=actor_lr)

        actor = MultiModel.create(
            actor_def, inputs=[actor_key, observations], tx=actor_opt, n_task=n_task
        )

        alpha_def = value_net.Alpha(name="alpha", init_value=jnp.ones((n_task, 1)))
        alpha = MultiModel.create(
            alpha_def,
            inputs=[alpha_key],
            tx=optax.adam(learning_rate=alpha_lr),
            n_task=n_task,
        )

        critic_def = value_net.EnsembleCriticMixtureMH(
            hidden_dims,
            activations=activation_func(activation),
            n_ensemble=n_critic,
            n_task=n_task,
            n_expert=n_expert,
            name="critic",
        )
        critic = MultiModel.create(
            critic_def,
            inputs=[critic_key, observations, actions],
            tx=critic_base_opt(learning_rate=critic_lr),
            n_task=n_task,
        )

        target_critic = MultiModel.create(
            critic_def, inputs=[critic_key, observations, actions], n_task=n_task
        )
        self.actor = actor
        self.alpha = alpha
        self.critic = critic
        self.models = {"alpha": self.alpha, "actor": self.actor, "critic": self.critic}
        self.target_critic = target_critic

        self.rng = rng

    def sample_actions(
        self,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = "log_prob",
        params=None,
    ) -> jnp.ndarray:
        actor_params = params or self.actor.params
        if len(observations.shape) == 2:
            observations = np.expand_dims(observations, axis=-2)

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

    def update(self, batch: Batch, critic_only: bool = False) -> InfoDict:
        if critic_only:
            new_rng, new_critic, new_target_critic, info = _update_critic_jit(
                self.rng,
                self.actor,
                self.critic,
                self.target_critic,
                self.alpha,
                batch,
                self.discount,
                self.tau,
            )

            self.rng = new_rng
            self.critic = new_critic
            self.target_critic = new_target_critic
        else:
            new_rng, new_actor, new_critic, new_target_critic, new_alpha, info = (
                _update_jit(
                    self.rng,
                    self.actor,
                    self.critic,
                    self.target_critic,
                    self.alpha,
                    batch,
                    self.discount,
                    self.tau,
                    self.target_entropy,
                )
            )

            self.rng = new_rng
            self.actor = new_actor
            self.critic = new_critic
            self.target_critic = new_target_critic
            self.alpha = new_alpha

        self.models = {"alpha": self.alpha, "actor": self.actor, "critic": self.critic}

        return info
