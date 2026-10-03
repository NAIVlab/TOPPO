from typing import Tuple
import jax.numpy as jnp
import jax
from common import Batch, InfoDict, Model, Params, PRNGKey, MultiModelPCGrad


def sac_update_actor(
    key: PRNGKey, actor: Model, critic: Model, alpha: Model, batch: Batch
) -> Tuple[Model, InfoDict]:
    alpha_value = alpha()

    def actor_loss_fn(
        actor_params: Params, task_id=None
    ) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({"params": actor_params}, batch.observations)
        actions = dist.sample(seed=key)

        qs = critic(batch.observations, actions)
        q = qs[0]
        log_probs = dist.log_prob(actions)
        actor_loss = alpha_value * log_probs - q

        return actor_loss[task_id].mean(), {
            "actor_loss": actor_loss.mean(axis=-1),
            "q_actor": q.mean(axis=-1),
            "log_pi": log_probs.mean(axis=-1),
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def sac_update_actor_pcgrad(
    key: PRNGKey,
    actor: MultiModelPCGrad,
    critic: MultiModelPCGrad,
    alpha: Model,
    batch: Batch,
) -> Tuple[Model, InfoDict]:
    alpha_value = alpha()

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({"params": actor_params}, batch.observations)
        actions = dist.sample(seed=key)

        qs = critic(batch.observations, actions)
        q = qs[0]
        log_probs = dist.log_prob(actions)
        actor_loss = alpha_value * log_probs - q

        return actor_loss.mean(axis=-1), {
            "actor_loss": actor_loss.mean(axis=-1),
            "q_actor": q.mean(axis=-1),
            "log_pi": log_probs.mean(axis=-1),
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def softmodular_update_actor(
    key: PRNGKey, actor: Model, critic: Model, alpha: Model, batch: Batch
) -> Tuple[Model, InfoDict]:
    alpha_value = alpha()
    reweight_term = jax.nn.softmax(-jnp.log(alpha_value), axis=0)

    def actor_loss_fn(
        actor_params: Params, task_id=None
    ) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({"params": actor_params}, batch.observations)
        actions = dist.sample(seed=key)

        qs = critic(batch.observations, actions)
        q = qs[0]
        log_probs = dist.log_prob(actions)
        actor_loss = (alpha_value * log_probs - q) * reweight_term

        return actor_loss[task_id].mean(), {
            "actor_loss": actor_loss.mean(axis=-1),
            "q_actor": q.mean(axis=-1),
            "log_pi": log_probs.mean(axis=-1),
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def moore_update_actor(
    key: PRNGKey, actor: Model, critic: Model, alpha: Model, batch: Batch
) -> Tuple[Model, InfoDict]:
    alpha_value = alpha()

    def actor_loss_fn(
        actor_params: Params, task_id=None
    ) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({"params": actor_params}, batch.observations)
        actions = dist.sample(seed=key)

        qs = critic(batch.observations, actions)
        q = jnp.min(qs, axis=0)
        log_probs = dist.log_prob(actions)
        actor_loss = alpha_value * log_probs - q

        return actor_loss[task_id].mean(), {
            "actor_loss": actor_loss.mean(axis=-1),
            "q_actor": q.mean(axis=-1),
            "log_pi": log_probs.mean(axis=-1),
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info
