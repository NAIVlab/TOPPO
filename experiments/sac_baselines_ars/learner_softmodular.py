"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

import policy
import value_net
from actor import sac_update_actor
from common import Batch, InfoDict, Model, PRNGKey, MultiModel, activation_func
from critic import sac_update_alpha, softmodular_update_q
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
    new_critic, critic_info = softmodular_update_q(
        key, actor, critic, target_critic, alpha, batch, discount
    )

    key, rng = jax.random.split(rng)
    new_actor, actor_info = sac_update_actor(key, actor, new_critic, alpha, batch)
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
        {**critic_info, **actor_info, **alpha_info},
    )


class Learner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        base_hidden_dims: Sequence[int] = (400, 400),
        em_hidden_dims: Sequence[int] = (400,),
        num_layers: int = 2,
        num_modules: int = 2,
        module_hidden: int = 256,
        gating_hidden: int = 256,
        num_gating_layers: int = 2,
        discount: float = 0.99,
        tau: float = 0.005,
        dropout_rate: Optional[float] = None,
        n_task: int = 1,
        n_critic: int = 2,
        softplus: bool = True,
        activation: str = "relu",
    ):
        self.tau = tau
        self.discount = discount
        self.n_task = n_task

        critic_base_opt = optax.adam
        actor_base_opt = optax.adam
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, alpha_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        self.target_entropy = -float(action_dim)

        actor_def = policy.NormalTanhModularGatedPolicy(
            base_hidden_dims,
            em_hidden_dims,
            num_layers,
            num_modules,
            module_hidden,
            gating_hidden,
            num_gating_layers,
            action_dim,
            n_task,
            log_std_min=-5.0,
            dropout_rate=dropout_rate,
            state_dependent_std=True,
            tanh_squash_distribution=True,
            activations=nn.relu,
            name="actor",
            softplus=softplus,
        )

        optimiser = actor_base_opt(learning_rate=actor_lr)
        actor = MultiModel.create(
            actor_def, inputs=[actor_key, observations], tx=optimiser, n_task=n_task
        )

        alpha_def = value_net.Alpha(name="alpha", init_value=jnp.ones((n_task, 1)))
        alpha = MultiModel.create(
            alpha_def,
            inputs=[alpha_key],
            tx=optax.adam(learning_rate=alpha_lr),
            n_task=n_task,
        )

        critic_def = value_net.EnsembleModularGatedCritic(
            base_hidden_dims,
            em_hidden_dims,
            num_layers,
            num_modules,
            module_hidden,
            gating_hidden,
            num_gating_layers,
            n_task,
            activations=activation_func(activation),
            n_ensemble=n_critic,
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
