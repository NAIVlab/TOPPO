"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

import policy
import value_net
from actor import sac_update_actor
from common import (
    Batch,
    InfoDict,
    Model,
    PRNGKey,
    MultiModel,
    activation_func,
    extract_dense_kernels,
    extract_ln_params,
)
from critic import sac_update_alpha, sac_update_q
import flax.linen as nn
from flax.core import unfreeze


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
    new_critic, critic_info = sac_update_q(
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
        hidden_dims: Sequence[int] = (400, 400, 400, 400),
        discount: float = 0.99,
        tau: float = 0.005,
        dropout_rate: Optional[float] = None,
        n_task: int = 1,
        n_critic: int = 2,
        softplus: bool = True,
        critic_layernorm: bool = False,
        critic_init_layernorm: bool = False,
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

        actor_def = policy.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            log_std_min=-5.0,
            dropout_rate=dropout_rate,
            state_dependent_std=True,
            tanh_squash_distribution=True,
            activations=nn.relu,
            name="actor",
            softplus=softplus,
        )

        actor_optimiser = actor_base_opt(learning_rate=actor_lr)
        actor = MultiModel.create(
            actor_def,
            inputs=[actor_key, observations],
            tx=actor_optimiser,
            n_task=n_task,
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

        self.use_lora = False
        self.rng = rng

    def sample_actions(
        self,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = "log_prob",
        params=None,
    ) -> jnp.ndarray:
        actor_params = self.lora_actor.params if self.use_lora else self.actor.params
        actor = self.lora_actor if self.use_lora else self.actor

        actor_params = params or actor_params

        rng, actions = policy.sample_actions2(
            self.rng,
            actor,
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
        if self.use_lora:
            info = self.update_lora(batch)
        else:
            info = self.update_base(batch)

        return info

    def update_base(self, batch: Batch) -> InfoDict:
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
        self.models.update(
            {"alpha": self.alpha, "actor": self.actor, "critic": self.critic}
        )

        return info

    def update_lora(self, batch: Batch) -> InfoDict:
        if "lora_critic" in self.models:
            new_rng, new_actor, new_critic, new_target_critic, new_alpha, info = (
                _update_jit(
                    self.rng,
                    self.lora_actor,
                    self.lora_critic,
                    self.lora_target_critic,
                    self.alpha,
                    batch,
                    self.discount,
                    self.tau,
                    self.target_entropy,
                )
            )

            self.rng = new_rng
            self.lora_actor = new_actor
            self.lora_critic = new_critic
            self.lora_target_critic = new_target_critic
            self.alpha = new_alpha
            self.models.update(
                {
                    "alpha": self.alpha,
                    "lora_actor": self.lora_actor,
                    "lora_critic": self.lora_critic,
                }
            )
        else:
            new_rng, new_actor, new_critic, new_target_critic, new_alpha, info = (
                _update_jit(
                    self.rng,
                    self.lora_actor,
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
            self.lora_actor = new_actor
            self.critic = new_critic
            self.target_critic = new_target_critic
            self.alpha = new_alpha
            self.models.update(
                {
                    "alpha": self.alpha,
                    "lora_actor": self.lora_actor,
                    "critic": self.critic,
                }
            )

        return info

    def init_LoRA_modules(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        hidden_dims: Sequence[int],
        rank: int,
        alpha: float,
        scale: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        n_critic: int = 2,
        activation: str = "relu",
        critic_layernorm: bool = False,
        critic_init_layernorm: bool = False,
        softplus: bool = True,
    ):
        action_dim = actions.shape[-1]
        critic_base_opt = optax.adamw
        actor_base_opt = optax.adamw
        rng, actor_key, critic_key = jax.random.split(self.rng, 3)

        actor_base_params = {}
        actor_base_params.update(
            {
                "MLP": extract_dense_kernels(self.actor.params["MLP_0"]),
                "means": self.actor.params["means"],
                "log_stds": self.actor.params["log_stds"],
            }
        )

        lora_actor_def = policy.LoRANormalTanhPolicy(
            hidden_dims,
            action_dim,
            rank,
            state_dependent_std=True,
            log_std_min=-5.0,
            tanh_squash_distribution=True,
            softplus=softplus,
            activations=nn.relu,
            alpha=alpha,
            n_task=self.n_task,
            base_params=actor_base_params,
            name="lora_actor",
        )
        self.lora_actor = MultiModel.create(
            lora_actor_def,
            inputs=[actor_key, observations],
            tx=actor_base_opt(learning_rate=actor_lr),
            n_task=self.n_task,
        )
        critic_base_params = {}
        critic_base_params.update(
            extract_dense_kernels(self.critic.params["VmapCritic_0"]["MLP_0"]),
        )
        lora_critic_def = value_net.EnsembleLoRACritic(
            hidden_dims=hidden_dims,
            rank=rank,
            scale=scale,
            alpha=alpha,
            n_task=self.n_task,
            activations=activation_func(activation),
            base_params=critic_base_params,
            init_layer_norm=critic_init_layernorm,
            layer_norm=critic_layernorm,
            n_ensemble=n_critic,
        )

        self.lora_critic = MultiModel.create(
            lora_critic_def,
            inputs=[critic_key, observations, actions],
            tx=critic_base_opt(learning_rate=critic_lr),
            n_task=self.n_task,
        )
        self.lora_target_critic = MultiModel.create(
            lora_critic_def,
            inputs=[critic_key, observations, actions],
            n_task=self.n_task,
        )

        if critic_layernorm or critic_init_layernorm:
            pretrained_ln_params = extract_ln_params(
                self.critic.params["VmapCritic_0"]["MLP_0"]
            )

            lora_critic_params = unfreeze(self.lora_critic.params)
            lora_target_critic_params = unfreeze(self.lora_critic.params)
            for ln_key in pretrained_ln_params.keys():
                # Guard if the LN key also exists in the new LoRA net
                if ln_key in lora_critic_params["VmapLoRACritic_0"]["LoRAMulti_MLP_0"]:
                    lora_critic_params["VmapLoRACritic_0"]["LoRAMulti_MLP_0"][
                        ln_key
                    ] = pretrained_ln_params[ln_key]
                    lora_target_critic_params["VmapLoRACritic_0"]["LoRAMulti_MLP_0"][
                        ln_key
                    ] = pretrained_ln_params[ln_key]

            self.lora_critic = self.lora_critic.replace(params=(lora_critic_params))
            self.lora_target_critic = self.lora_target_critic.replace(
                params=(lora_target_critic_params)
            )

        self.models.update(
            {
                "alpha": self.alpha,
                "lora_actor": self.lora_actor,
                "lora_critic": self.lora_critic,
            }
        )

        self.rng = rng
        return
