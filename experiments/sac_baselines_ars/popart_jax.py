"""PopArt (van Hasselt 2016 / Hessel 2019) for the JAX SAC critic.

Per-task value-target normalization — the D1 (scale-spread) lever of the
TOPPO critic stack, ported to SAC:

  - the critic module outputs NORMALIZED q̂ through a per-task affine head
    ``(popart_w, popart_b)`` (see value_net.EnsembleCritic, popart=True);
  - raw Q = sigma_i * q̂ + mu_i with per-task running stats (mu, sigma) of the
    RAW Bellman targets, tracked here (outside the network);
  - ART: stats update per critic update — "ema" (Hessel per-sample beta,
    applied as the closed-form batch power) or "welford" (Chan/Golub/LeVeque
    merge of every batch, the TOPPO paper's estimator);
  - POP: on every stats change, ``(popart_w, popart_b)`` of BOTH critic and
    target critic are rescaled so raw outputs are preserved exactly:
        w' = w * sigma_old / sigma_new
        b' = (sigma_old * b + mu_old - mu_new) / sigma_new

State is a plain dict of (K,) float32 arrays — a jit-friendly pytree that
also serializes through the flax checkpoint blob.
"""

from __future__ import annotations

import jax.numpy as jnp


def init_state(n_task: int):
    return {
        "mu": jnp.zeros((n_task,), jnp.float32),
        "nu": jnp.ones((n_task,), jnp.float32),   # E[y^2] (ema)
        "m2": jnp.zeros((n_task,), jnp.float32),  # sum sq dev (welford)
        "count": jnp.zeros((n_task,), jnp.float32),
        "sigma": jnp.ones((n_task,), jnp.float32),
    }


def update_stats(state, y_raw, method: str, ema_decay: float, min_std: float):
    """y_raw: (n_task, batch) raw per-task Bellman targets."""
    y_raw = y_raw.astype(jnp.float32)
    batch = y_raw.shape[-1]
    m = jnp.mean(y_raw, axis=-1)
    if method == "ema":
        # per-sample beta applied as one closed-form batch step
        beta_b = 1.0 - (1.0 - ema_decay) ** batch
        v2 = jnp.mean(y_raw**2, axis=-1)
        mu = (1.0 - beta_b) * state["mu"] + beta_b * m
        nu = (1.0 - beta_b) * state["nu"] + beta_b * v2
        var = nu - mu**2
        new = {**state, "mu": mu, "nu": nu}
    elif method == "welford":
        # CAVEAT: stats are float32 (x64 is off in this codebase); the
        # full-history count reaches 2e8 on a 2M-update run, past float32's
        # exact-integer range (1.6e7) — use "ema" for long runs; welford is
        # only trustworthy for short/diagnostic runs here.
        v = jnp.mean((y_raw - m[:, None]) ** 2, axis=-1)  # batch variance
        cnt_new = state["count"] + batch
        delta = m - state["mu"]
        mu = state["mu"] + delta * batch / cnt_new
        m2 = (
            state["m2"]
            + v * batch
            + delta**2 * state["count"] * batch / cnt_new
        )
        var = m2 / cnt_new
        new = {**state, "mu": mu, "m2": m2, "count": cnt_new}
    else:
        raise ValueError(f"popart method must be ema|welford, got {method!r}")

    sigma = jnp.sqrt(jnp.clip(var, min_std**2, 1e12))
    new["sigma"] = sigma
    return new


def denormalize(q_norm, state):
    """(..., n_task, batch) normalized q̂ -> raw Q using per-task stats."""
    return state["sigma"][:, None] * q_norm + state["mu"][:, None]


def normalize_targets(y_raw, state):
    return (y_raw - state["mu"][:, None]) / state["sigma"][:, None]


def pop_rescale_params(params, old_state, new_state):
    """POP: rescale the per-task affine head so raw outputs are preserved.

    ``params`` is the (frozen or plain) param dict of an EnsembleCritic with
    popart=True — ``popart_w``/``popart_b`` live at the top level.
    """
    import flax.core

    ratio = old_state["sigma"] / new_state["sigma"]
    w = params["popart_w"] * ratio
    b = (
        old_state["sigma"] * params["popart_b"]
        + old_state["mu"]
        - new_state["mu"]
    ) / new_state["sigma"]

    if isinstance(params, flax.core.FrozenDict):
        return params.copy({"popart_w": w, "popart_b": b})
    p = dict(params)
    p["popart_w"] = w
    p["popart_b"] = b
    return p
