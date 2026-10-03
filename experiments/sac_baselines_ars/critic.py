from typing import Tuple
import jax.numpy as jnp
import jax
from common import Batch, InfoDict, Model, Params, PRNGKey
import popart_jax


def sac_update_q(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    alpha: Model,
    batch: Batch,
    discount: float,
) -> Tuple[Model, InfoDict]:
    alpha_value = alpha()
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)

    next_qs = target_critic(batch.next_observations, next_actions)
    next_q = jnp.min(next_qs, axis=0)

    target_q = batch.rewards + discount * batch.masks * (
        next_q - alpha_value * next_log_probs
    )

    def critic_loss_fn(
        critic_params: Params, task_id=None
    ) -> Tuple[jnp.ndarray, InfoDict]:
        qs = critic.apply({"params": critic_params}, batch.observations, batch.actions)
        critic_loss = jnp.sum((qs - target_q) ** 2, axis=0)
        return critic_loss[task_id].mean(), {
            "critic_loss": critic_loss.mean(axis=-1),
            "q": qs[0].mean(axis=-1),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


def sac_update_q_pcgrad(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    alpha: Model,
    batch: Batch,
    discount: float,
) -> Tuple[Model, InfoDict]:
    alpha_value = alpha()
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)

    next_qs = target_critic(batch.next_observations, next_actions)
    next_q = jnp.min(next_qs, axis=0)

    target_q = batch.rewards + discount * batch.masks * (
        next_q - alpha_value * next_log_probs
    )

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        qs = critic.apply({"params": critic_params}, batch.observations, batch.actions)
        critic_loss = jnp.sum((qs - target_q) ** 2, axis=0)
        return critic_loss.mean(axis=-1), {
            "critic_loss": critic_loss.mean(axis=-1),
            "q": qs[0].mean(axis=-1),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


def sac_update_q_popart(
    key: PRNGKey,
    actor: Model,
    critic,
    target_critic,
    alpha: Model,
    batch: Batch,
    discount: float,
    popart_state,
    popart_method: str,
    popart_ema_decay: float,
    popart_min_std: float,
    per_task_loss: bool,
):
    """Critic update with PopArt target normalization (critic balancing).

    The critic outputs NORMALIZED q̂ (per-task affine head inside the
    module); raw targets are formed by denormalizing the target critic's
    bootstrap, stats are updated (ART), both critics' heads are POP-rescaled
    to the new stats, and the TD loss is computed in normalized space.

    Returns (new_critic, surgered_target_critic, new_popart_state, info).
    ``per_task_loss=True`` yields the vector loss the PCGrad/FairGrad
    apply_gradient expects; False the legacy scalar mean.
    """
    alpha_value = alpha()
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)

    next_q_norm = jnp.min(
        target_critic(batch.next_observations, next_actions), axis=0
    )
    # sigma > 0 => min over the ensemble commutes with the affine denorm
    next_q_raw = popart_jax.denormalize(next_q_norm, popart_state)

    # Temperature-scale consistency: the actor maximizes q_norm - alpha*logpi,
    # i.e. alpha lives in NORMALIZED units.  The raw-space soft target must
    # therefore use sigma_i * alpha_i so that, after normalization, the TD
    # target matches the actor's objective exactly (otherwise the effective
    # raw temperature drifts with each task's value scale).
    alpha_raw = alpha_value * popart_state["sigma"][:, None]
    target_q_raw = batch.rewards + discount * batch.masks * (
        next_q_raw - alpha_raw * next_log_probs
    )

    new_state = popart_jax.update_stats(
        popart_state, target_q_raw, popart_method, popart_ema_decay, popart_min_std
    )
    critic = critic.replace(
        params=popart_jax.pop_rescale_params(critic.params, popart_state, new_state)
    )
    target_critic = target_critic.replace(
        params=popart_jax.pop_rescale_params(
            target_critic.params, popart_state, new_state
        )
    )
    target_q = popart_jax.normalize_targets(target_q_raw, new_state)

    if per_task_loss:

        def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
            qs = critic.apply(
                {"params": critic_params}, batch.observations, batch.actions
            )
            critic_loss = jnp.sum((qs - target_q) ** 2, axis=0)
            return critic_loss.mean(axis=-1), {
                "critic_loss": critic_loss.mean(axis=-1),
                # "q" stays comparable with the non-popart arms: RAW scale
                "q": popart_jax.denormalize(qs[0], new_state).mean(axis=-1),
                "q_norm": qs[0].mean(axis=-1),
                "popart_mu": new_state["mu"],
                "popart_sigma": new_state["sigma"],
            }

    else:

        def critic_loss_fn(
            critic_params: Params, task_id=None
        ) -> Tuple[jnp.ndarray, InfoDict]:
            qs = critic.apply(
                {"params": critic_params}, batch.observations, batch.actions
            )
            critic_loss = jnp.sum((qs - target_q) ** 2, axis=0)
            return critic_loss[task_id].mean(), {
                "critic_loss": critic_loss.mean(axis=-1),
                "q": popart_jax.denormalize(qs[0], new_state).mean(axis=-1),
                "q_norm": qs[0].mean(axis=-1),
                "popart_mu": new_state["mu"],
                "popart_sigma": new_state["sigma"],
            }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, target_critic, new_state, info


def sac_update_alpha(
    alpha: Model, log_pi: jnp.array, target_entropy: float
) -> Tuple[Model, InfoDict]:

    def actor_loss_fn(
        alpha_params: Params, task_id=None
    ) -> Tuple[jnp.ndarray, InfoDict]:
        alpha_value = alpha.apply({"params": alpha_params})
        alpha_loss = alpha_value[:, 0] * (-log_pi - target_entropy)
        return alpha_loss[task_id].mean(), {
            "alpha_loss": alpha_loss.mean(axis=-1),
            "alpha_value": alpha_value.mean(axis=-1),
        }

    new_alpha, info = alpha.apply_gradient(actor_loss_fn)
    return new_alpha, info


def softmodular_update_q(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    alpha: Model,
    batch: Batch,
    discount: float,
) -> Tuple[Model, InfoDict]:
    alpha_value = alpha()
    reweight_term = jax.nn.softmax(-jnp.log(alpha_value), axis=0)
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)

    next_qs = target_critic(batch.next_observations, next_actions)
    next_q = jnp.min(next_qs, axis=0)

    target_q = batch.rewards + discount * batch.masks * (
        next_q - alpha_value * next_log_probs
    )

    def critic_loss_fn(
        critic_params: Params, task_id=None
    ) -> Tuple[jnp.ndarray, InfoDict]:
        qs = critic.apply({"params": critic_params}, batch.observations, batch.actions)
        critic_loss = jnp.sum((qs - target_q) ** 2, axis=0) * reweight_term
        return critic_loss[task_id].mean(), {
            "critic_loss": critic_loss.mean(axis=-1),
            "q": qs[0].mean(axis=-1),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
