"""FairGrad task-weight solver for the JAX SAC learner (critic balancing).

Ports the TOPPO repo's solver (mtrl/gradients/fairgrad.py — Ban & Ji, ICML
2024) to numpy and exposes it to jitted JAX code through jax.pure_callback:
the Gram matrix is computed on-device in fp32 (grads pre-normalized by the
max task-grad norm for conditioning — the alpha=1 combined gradient is
invariant to that rescaling, Theorem 1), the (K, K) Gram ships to the host
(~10 KB), and the weights come back fp32.

Solver: y-space projected damped Newton on ``F(y) = G.exp(y) - exp(-y/a)``
with the diagonal closed-form warm start ``y = -(a/(a+1)).log(diag G)``,
Armijo backtracking, fp64 throughout; scipy least_squares fallback (it follows
https://github.com/OptMN-Lab/fairgrad, commit b58bce8, MIT; see the root NOTICE),
uniform weights as the last resort — the same three-tier chain as the TOPPO repo.
"""

from __future__ import annotations

import numpy as np

_ETA = 1e-2
_TOL = 1e-8
_MAX_ITER = 50
_Y_MAX = 50.0
_ARMIJO_C1 = 1e-4
_LS_MAX = 30


def _solve_ynewton(G: np.ndarray, alpha: float, eps: float):
    K = G.shape[0]
    inv_a = 1.0 / float(alpha)

    def F(y):
        x = np.exp(np.minimum(y, _Y_MAX))
        neg_y_over_a = np.minimum(-y * inv_a, 700.0)
        return G @ x - np.exp(neg_y_over_a)

    diag_G = np.maximum(np.diagonal(G), eps)
    y = np.clip(-(alpha / (alpha + 1.0)) * np.log(diag_G), -_Y_MAX, _Y_MAX)
    r = float(np.linalg.norm(F(y)))

    for it in range(_MAX_ITER):
        if r <= _ETA or r <= _TOL:
            return np.exp(np.minimum(y, _Y_MAX)), True
        x = np.exp(np.minimum(y, _Y_MAX))
        J = G * x[np.newaxis, :]
        neg_y_over_a = np.minimum(-y * inv_a, 700.0)
        J[np.diag_indices_from(J)] += inv_a * np.exp(neg_y_over_a)
        try:
            p = np.linalg.solve(J, -F(y))
        except np.linalg.LinAlgError:
            return np.exp(np.minimum(y, _Y_MAX)), False
        t = 1.0
        for _ in range(_LS_MAX):
            y_new = np.minimum(y + t * p, _Y_MAX)
            r_new = float(np.linalg.norm(F(y_new)))
            if r_new < r * (1.0 - _ARMIJO_C1 * t):
                break
            t *= 0.5
        else:
            return np.exp(np.minimum(y, _Y_MAX)), False
        y, r = y_new, r_new

    return np.exp(np.minimum(y, _Y_MAX)), (r <= max(_ETA, _TOL))


def solve_weights_host(gram: np.ndarray, alpha: float = 1.0, eps: float = 1e-8):
    """Host callback: (K, K) fp32 Gram -> (K,) fp32 FairGrad weights."""
    G = np.asarray(gram, dtype=np.float64)
    K = G.shape[0]
    if K == 1:
        return np.ones(1, dtype=np.float32)

    x, converged = _solve_ynewton(G, alpha, eps)

    if not (converged and np.all(np.isfinite(x))):
        try:
            from scipy.optimize import least_squares

            inv_a = 1.0 / float(alpha)
            x0 = (
                np.maximum(x, eps)
                if np.all(np.isfinite(x))
                else np.full(K, 1.0 / K)
            )
            res = least_squares(
                lambda v: G @ v - np.power(1.0 / np.maximum(v, eps), inv_a),
                x0=x0,
                bounds=(eps, np.inf),
                method="trf",
                max_nfev=200,
            )
            if np.all(np.isfinite(res.x)):
                x = res.x
            else:
                x = np.full(K, 1.0 / K)
        except Exception:
            x = np.full(K, 1.0 / K)

    if not np.all(np.isfinite(x)):
        x = np.full(K, 1.0 / K)
    return np.maximum(x, eps).astype(np.float32)


def fairgrad_combine_flat(grads_flat, alpha: float = 1.0):
    """Inside-jit combine: (K, dim) fp32 per-task flat grads -> (dim,) fp32.

    Returns ``(combined, weights_normalized_scale)``; combined = sum_i w_i g_i
    solved on the norm-normalized Gram (scale-invariant at alpha=1: weights
    from g/c solve to c.w, so sum w'_i g'_i == sum w_i g_i).
    """
    import jax
    import jax.numpy as jnp

    K = grads_flat.shape[0]
    if float(alpha) == 1.0:
        # conditioning normalization — exactly invariant ONLY at alpha=1
        # (weights from g/c solve to c*w, so w' @ gn == w @ g)
        norms = jnp.linalg.norm(grads_flat, axis=1)
        gmax = jnp.maximum(jnp.max(norms), 1e-12)
        gn = grads_flat / gmax
    else:
        gn = grads_flat  # solve the raw Gram; host fp64 handles conditioning
    gram = gn @ gn.T

    w = jax.pure_callback(
        lambda g: solve_weights_host(g, alpha),
        jax.ShapeDtypeStruct((K,), jnp.float32),
        gram.astype(jnp.float32),
    )
    combined = w @ gn  # == sum_i w_i g_i on the raw grads (invariance)
    return combined, w
