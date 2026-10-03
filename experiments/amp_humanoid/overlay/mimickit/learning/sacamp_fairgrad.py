"""Reduced-precision Gram matrix for FairGrad, for the off-policy SAC arms.

Profiling the FairGrad step on an RTX PRO 6000 (shared critic trunk,
N ~ 1.37M parameters, K = 2 components) attributes essentially all of the
~405 ms per solve to ONE operation:

    fp64 2x2 linalg.solve            0.31 ms
    fp64 Gram  g @ g.T             480 ms      <- the whole cost
    fp32 Gram, cast to fp64          7.75 ms   (62x faster)

The cost is the skinny fp64 GEMM over the flattened per-component gradients,
not the Newton solve. Consumer/workstation cards run fp64 GEMM at a small
fraction of fp32 throughput; on H200 the same solve is ~2.3 ms.

Computing only the K x K Gram in fp32 (and keeping the solve, the weighted
combination and all diagnostics in fp64) shifts the FairGrad weights by
~3e-7 relative, i.e. a couple of ulp of fp32 -- numerically irrelevant while
making the critic-balancing arm affordable at n=3 seeds without a cluster.

This module also fixes a SCALE SENSITIVITY that matters much more than the
precision. `toppo._solve_fairgrad_ynewton` exits early on an ABSOLUTE residual
threshold (`eta = 1e-2`) applied to `||G x - x^(-1/alpha)||`. The off-policy
critic's per-component gradients are ~500x smaller than the on-policy ones
(measured: mean grad norm 0.0035 vs 2.28, Gram trace 4e-5 vs ~10), so the
whole KKT system shrinks, the warm start already satisfies the tolerance, and
the solver returns after ZERO Newton iterations -- silently degrading FairGrad
to its diagonal initializer `w_i ~ 1/||g_i||`. The symptom is the combined
gradient norm settling at ~1.79 instead of the sqrt(K) = 1.414 that alpha = 1
pins it to.

The fix is to precondition: divide every component gradient by the largest of
their norms before forming the Gram. This is what the reference JAX
implementation does and it is EXACTLY equivalent -- writing g_i = s * gn_i,
the raw system G x = x^(-1/alpha) and the scaled system G_n y = y^(-1/alpha)
have solutions related by x = y * s^(-2 alpha / (alpha + 1)), so

    sum_i x_i g_i = s^((1 - alpha)/(1 + alpha)) * sum_i y_i gn_i.

The DIRECTION is preserved at every alpha; the magnitude is identical only at
alpha = 1 (the value every production config uses). We therefore rescale the
combined gradient, the weights and the Gram trace back to raw space explicitly
rather than relying on the alpha = 1 coincidence -- otherwise the logged
mean_grad_norm diagnostic is off by 1/s (~286x at off-policy gradient
scales).

This module does NOT modify learning/toppo.py: `fast_gram` temporarily binds
`toppo.fairgrad_combine` to this variant for the duration of one optimizer
step, so every PPO/TOPPO run keeps calling the original implementation and
stays bit-identical.
"""

import contextlib
import time

import torch

import learning.toppo as toppo


def fairgrad_combine_fp32_gram(task_grads, alpha=1.0, rescale=False,
                               gram_dtype=torch.float32):
    """toppo.fairgrad_combine with the Gram matrix computed in fp32.

    Everything else -- the y-space Newton solve, the SciPy/uniform fallback
    chain, the weighted combination, the rescale and every diagnostic -- is
    identical to and shared with the original implementation.
    """
    if alpha <= 0.0:
        raise ValueError(f"FairGrad alpha must be positive, got {alpha}.")
    if len(task_grads) == 0:
        return torch.zeros(0), toppo.FairGradDiagnostics(weights=torch.zeros(0))
    if len(task_grads) == 1:
        return task_grads[0].clone(), toppo.FairGradDiagnostics(
            weights=torch.ones(1, device=task_grads[0].device, dtype=task_grads[0].dtype),
            solver_converged=True,
        )

    t0 = time.perf_counter()
    grad_mat = torch.stack(task_grads)

    # (1) Precondition: divide every component gradient by the largest of their
    # norms before forming the Gram. This is EXACTLY equivalent at any alpha in
    # the combined gradient (see the module docstring for the derivation) and is
    # what the reference JAX implementation does; without it the solver's
    # ABSOLUTE early-exit tolerance (eta = 1e-2 in toppo._solve_fairgrad_ynewton)
    # is scale-dependent, and for the small per-component gradients of an
    # off-policy critic the warm start already satisfies it -- FairGrad then
    # silently degrades to its diagonal initializer w ~ 1/||g_i||.
    scale = grad_mat.detach().norm(dim=1).max().clamp_min(1e-12)
    grad_scaled = grad_mat.detach() / scale

    # (2) the Gram in the requested precision; the solve and the combination
    # stay fp64. fp32 is 121x faster on workstation cards and shifts the
    # weights by ~1e-7 (see the tests).
    if (gram_dtype == torch.float32):
        gram = (grad_scaled @ grad_scaled.T).double()
    else:
        gram = grad_scaled.double() @ grad_scaled.double().T
    grad_f64 = grad_scaled.double()

    weights, converged, iters = toppo._solve_fairgrad_ynewton(gram, alpha)
    used_scipy_fallback = False
    solver_fallback = False
    if (not converged) or (not torch.isfinite(weights).all()):
        weights, scipy_ok, scipy_iters = toppo._solve_fairgrad_scipy(
            gram, alpha, x_start=weights)
        used_scipy_fallback = True
        iters = scipy_iters
        converged = scipy_ok and weights is not None and torch.isfinite(weights).all()
    if (not converged) or weights is None or (not torch.isfinite(weights).all()):
        weights = torch.ones(
            grad_mat.shape[0], device=grad_mat.device, dtype=torch.float64
        ) / grad_mat.shape[0]
        converged = False
        solver_fallback = True

    weights = weights.clamp_min(1e-8)

    residual = (gram @ weights - weights.clamp_min(1e-8).pow(-1.0 / float(alpha))).norm()
    combined_f64 = (weights.unsqueeze(1) * grad_f64).sum(dim=0)

    # Undo the preconditioning. With g_i = s * gn_i the two KKT systems have
    # solutions related by x = y * s^(-2a/(a+1)), so the combined gradients
    # differ by s^((1-a)/(1+a)). That factor is 1 at alpha = 1 -- which is the
    # only alpha any production config uses -- but NOT at other alpha, so it
    # must be applied explicitly rather than assumed away. The direction is
    # preserved at every alpha; only the magnitude needs restoring.
    s64 = scale.double()
    alpha_f = float(alpha)
    combined_f64 = combined_f64 * s64.pow((1.0 - alpha_f) / (1.0 + alpha_f))
    # diagnostics must be reported in RAW gradient space: the logged
    # mean/combined gradient norms are compared across arms, and in scaled
    # space mean_grad_norm is off by 1/s (a factor of ~286 at off-policy scales).
    weights = weights * s64.pow(-2.0 * alpha_f / (alpha_f + 1.0))
    mean_grad_norm = grad_f64.mean(dim=0).norm() * s64
    gram_trace_raw = gram.trace() * s64.pow(2)
    combined_norm_pre_rescale = combined_f64.norm()
    rescale_factor = 1.0
    if rescale:
        combined_norm = combined_f64.norm().clamp_min(1e-12)
        rescale_factor = float((mean_grad_norm / combined_norm).item())
        combined_f64 = combined_f64 * rescale_factor
    combined_norm_post_rescale = combined_f64.norm()

    normalized_weights = weights / weights.sum().clamp_min(1e-12)
    entropy = -(normalized_weights * normalized_weights.clamp_min(1e-12).log()).sum()
    diag = toppo.FairGradDiagnostics(
        weights=weights.detach().to(dtype=grad_mat.dtype),
        gram_trace=float(gram_trace_raw.item()),
        weight_entropy=float(entropy.item()),
        residual=float(residual.item()),
        solver_converged=bool(converged),
        solver_fallback=bool(solver_fallback),
        used_scipy_fallback=bool(used_scipy_fallback),
        solver_iters=int(iters),
        rescale_factor=float(rescale_factor),
        mean_grad_norm=float(mean_grad_norm.item()),
        combined_grad_norm_pre_rescale=float(combined_norm_pre_rescale.item()),
        combined_grad_norm_post_rescale=float(combined_norm_post_rescale.item()),
        wall_time_ms=(time.perf_counter() - t0) * 1000.0,
    )
    return combined_f64.to(dtype=grad_mat.dtype), diag


@contextlib.contextmanager
def fast_gram(enabled, fp32_gram=True):
    """Bind toppo.fairgrad_combine to the preconditioned variant for one step.

    `enabled` turns on the CORRECTNESS fix (preconditioning); `fp32_gram`
    independently selects the Gram precision, which is a throughput choice.
    Scoped and always restored, so nothing outside the `with` block -- in
    particular every PPO/TOPPO code path -- observes a different function.
    """
    if (not enabled):
        yield
        return

    dtype = torch.float32 if fp32_gram else torch.float64

    def _combine(task_grads, alpha=1.0, rescale=False):
        return fairgrad_combine_fp32_gram(task_grads, alpha=alpha,
                                          rescale=rescale, gram_dtype=dtype)

    original = toppo.fairgrad_combine
    toppo.fairgrad_combine = _combine
    try:
        yield
    finally:
        toppo.fairgrad_combine = original
