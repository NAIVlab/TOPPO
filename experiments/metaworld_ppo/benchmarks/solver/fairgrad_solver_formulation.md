# FairGrad solver: formulation, ill-posedness, and the implicit-regularizer view

> **Historical design memo.** Written before the y-space Newton solver was implemented, and kept
> for its derivations. Its measurements are α = 2 development benchmarks, not numbers from the
> paper, and some of its expectations did not survive (for instance, the "fewer than 15
> iterations" of §5 fails on adversarial Grams, where the shipped warm-started solver can hit
> its 50-iteration cap; see row 3 of `raw_selected_regimes.out`). For the
> shipped solver and its defaults see `mtrl/gradients/fairgrad.py`; for measured speedups see
> [README.md](README.md). Section 7 summarizes what shipped.

Source script: [fairgrad_solver_benchmark.py](fairgrad_solver_benchmark.py).

## 1. Problem statement

**FairGrad weight-solving problem.** Given $K$ per-task gradient vectors $\{g_i\}_{i=1}^K$, $g_i \in \mathbb{R}^D$, let $G \in \mathbb{R}^{K \times K}$ be the Gram matrix $G_{ij} = \langle g_i, g_j \rangle$. For a fairness parameter $\alpha > 0$, the FairGrad weights $x \in \mathbb{R}^K_{>0}$ satisfy the KKT stationarity condition

$$
G x = x^{-1/\alpha} \qquad (x_i > 0 \ \forall i), \tag{1}
$$

where $x^{-1/\alpha}$ is taken element-wise. The combined gradient applied to the optimizer is

$$
\tilde g = \sum_{i=1}^K x_i g_i = X^\top g, \qquad X = [g_1 \mid \cdots \mid g_K] \in \mathbb{R}^{D \times K}.
$$

Dotting (1) with $x$ gives the **KKT norm identity**

$$
\|\tilde g\|^2 = x^\top G x = \sum_{i=1}^K x_i \cdot x_i^{-1/\alpha} = \sum_{i=1}^K x_i^{1 - 1/\alpha}. \tag{2}
$$

For $\alpha = 1$: $\|\tilde g\|^2 = K$ (constant, safe). For $\alpha = 2$: $\|\tilde g\|^2 = \sum_i \sqrt{x_i}$ (grows with the square root of weights).

## 2. Ill-posedness in the dead-task regime

Let task $i$ be **dead**: $\|g_i\| \to 0$. Then row $i$ of $G$ is zero, so (1) restricted to row $i$ reads

$$
0 \;=\; (Gx)_i \;\neq\; x_i^{-1/\alpha},
$$

which is **inconsistent for any finite $x_i$**. In the limit, $x_i \to \infty$ formally satisfies the equation, but this limit is not attainable numerically: for any finite $x_i$ the residual $|(Gx)_i - x_i^{-1/\alpha}| = x_i^{-1/\alpha} > 0$.

Empirically, in multi-task PPO the "dead-task" row arises whenever a task's gradient falls below fp precision — either because (a) its value/policy loss has been near-perfectly fit, or (b) its samples are outside the minibatch. The result is that (1) **has no solution in $\mathbb{R}^K_{>0}$**, and any solver returns a "solution" only in the least-squares sense

$$
\hat x \;=\; \arg\min_{x \in [\epsilon, \infty)^K} \; \|Gx - x^{-1/\alpha}\|^2. \tag{3}
$$

The minimum is *not* zero; it equals $\sum_i \big(x_i^{-1/\alpha}|_{x_i \to \text{upper}}\big)^2$, which approaches zero only as $x_i \to \infty$. **The least-squares problem (3) is therefore unbounded below in $x_i$ along every dead-task coordinate.**

## 3. What a solver actually does: two regimes

Solvers cope with (3) by implicitly truncating the dead-task ray. Empirically, we observe two qualitative regimes on synthetic $G$ with one task of gradient magnitude $\|g_{\text{dead}}\|$:

**Regime A — scipy trust-region (trf)**. Trust-region reflective with default tolerances $(\text{ftol}, \text{xtol}, \text{gtol}) = 10^{-8}$ and finite-difference Jacobian stops when the predicted step is smaller than $\text{xtol}$ relative to $x_i$. Because the Jacobian along the dead ray flattens as $x_i$ grows (the derivative of $-x_i^{-1/\alpha}$ w.r.t. $x_i$ is $(1/\alpha) x_i^{-1/\alpha - 1} \to 0$), trust-region terminates **after a fixed number of iterations** (~20 in the tested regime) at a residual plateau. This plateau is dictated by the step-bound mechanics of trf, not by the problem:

$$
\text{trf plateau: } \quad \hat x_i^{\text{trf}} \;\approx\; C_\text{trf}, \quad \|\text{residual}\|_{\text{trf}} \;\approx\; \eta_\text{trf}, \quad \text{independent of } \|g_{\text{dead}}\|.
$$

For $\alpha = 2$, K=10, $D = 5000$ our benchmark gives $C_\text{trf} \approx 7.3 \cdot 10^3$ and $\eta_\text{trf} \approx 10^{-2}$, **invariant across $\|g_{\text{dead}}\| \in [10^{-16}, 10^{-24}]$**.

**Regime B — projected Newton with strong convergence.** A damped Newton iteration with projection onto $[\epsilon, \infty)$ and Armijo backtracking (§5), run to tolerance $\tau = 10^{-6}$ (the benchmark script's setting; the shipped solver uses $\tau = 10^{-8}$), exploits the analytical Jacobian $J = G + (1/\alpha)\,\mathrm{diag}(x^{-1/\alpha - 1})$ and pushes $x$ along the dead ray until line search stalls or tolerance is met. Empirically:

$$
\text{proj-Newton plateau: } \quad \hat x_i^{\text{PN}} \;\sim\; \|g_{\text{dead}}\|^{-\alpha/(\alpha+1)}, \quad \|\text{residual}\|_{\text{PN}} \;\sim\; 10^{-8}.
$$

For $\alpha = 2$ and $\|g_{\text{dead}}\| = 10^{-22}$, this gives $\hat x \sim 10^{29}$ in principle, though fp64 machine limits truncate near $10^{16}$.

**The gap is 12 orders of magnitude**. Both solvers are "correct" for the unbounded problem (3); they differ in how aggressively they chase the unattainable infimum.

## 4. The implicit-regularizer view

The scipy-trf plateau $C_\text{trf}$ is not a mathematical property of (1) — it is a **side effect of trust-region step mechanics**. But operationally, it plays the role of a regularizer: it caps the weight assigned to dead tasks at some machine-dependent constant, preventing the weighted sum from being dominated by a nearly-zero gradient amplified to arbitrary magnitude.

We make this explicit. Define the **residual-target regularized FairGrad problem**

$$
\mathcal{P}(\eta): \quad \text{find } x \in [\epsilon, \infty)^K \text{ such that } \|Gx - x^{-1/\alpha}\|_2 \leq \eta. \tag{4}
$$

For $\eta = 0$, $\mathcal{P}$ recovers the original KKT system (1), which is ill-posed when any task is dead. For $\eta > 0$, $\mathcal{P}$ is **well-posed**: the level set is non-empty (starts containing $x_0 = \mathbf{1}/K$ once $\eta$ exceeds the initial residual) and bounded on the active subspace. The dead-task coordinate is constrained only through the residual slack: any $x_i \in [\epsilon, x_i^{-1/\alpha}(\eta)]$ is admissible, and the solver's choice within this interval is determined by the iteration dynamics.

The scipy default behavior is recovered by $\eta \approx \eta_\text{trf} \approx 10^{-2}$ (empirically observed plateau).

## 5. The proposed solver

Two variants of damped Newton, differing in the coordinate system:

### 5a. x-space projected Newton

**Input:** $G \in \mathbb{R}^{K \times K}$, $\alpha > 0$, $\epsilon > 0$ (lower bound), $x_0 \in \mathbb{R}^K_{>0}$ (initial point), $\eta \geq 0$ (residual target), $\tau > 0$ (tight tolerance), $N$ (max iterations).

**Output:** $\hat x \in \mathbb{R}^K_{>0}$ with $\|G\hat x - \hat x^{-1/\alpha}\| \leq \max(\eta, \tau)$.

**Algorithm.**

1. $x \leftarrow \max(x_0, \epsilon)$; $r \leftarrow \|F(x)\|$ where $F(x) = Gx - x^{-1/\alpha}$.
2. For $n = 1, \ldots, N$:
    a. If $r \leq \eta$ or $r \leq \tau$: **return** $x$ (early stop).
    b. Compute Jacobian $J = G + (1/\alpha)\,\mathrm{diag}(x^{-1/\alpha - 1})$.
    c. Solve $Jp = -F(x)$ for the Newton step $p$.
    d. Backtracking line search: start with $t = 1$; while $\|F(\max(x + tp, \epsilon))\| \geq r \cdot (1 - c_1 t)$ for $c_1 = 10^{-4}$, set $t \leftarrow t/2$. Cap at 30 backtracks.
    e. $x \leftarrow \max(x + tp, \epsilon)$; $r \leftarrow \|F(x)\|$.
3. Return $x$ (non-converged case).

### 5b. y-space Newton (log-reparameterized)

Re-parameterize $x = e^y$, $y \in \mathbb{R}^K$.  The positivity constraint becomes automatic and "weight diverging to infinity" is a linear walk in $y$.  The equation and Jacobian become

$$
F(y) = G\,e^y - e^{-y/\alpha}, \qquad J_y[i,j] = G_{ij}\,e^{y_j} + \frac{1}{\alpha}\,e^{-y_i/\alpha}\,\delta_{ij}.
$$

For a dead-task row $i$ (where $G_{i,:} \approx 0$), the diagonal element dominates:

$$
F_i(y) \approx -e^{-y_i/\alpha}, \qquad J_{ii} \approx \frac{1}{\alpha}\,e^{-y_i/\alpha},
$$

so the Newton step on that coordinate is $\Delta y_i = -F_i/J_{ii} = \alpha$, **constant**.  This means y-space Newton walks $y_i \mapsto y_i + \alpha$ on dead coordinates each iteration — a predictable linear walk toward $+\infty$ rather than an unbounded Newton ray that line-search must repeatedly shrink.

**Algorithm.**

1. $y \leftarrow \min(y_0, y_{\max})$ (soft cap to prevent $\exp$ overflow, e.g. $y_{\max} = 50$); $r \leftarrow \|F(y)\|$.
2. For $n = 1, \ldots, N$:
    a. If $r \leq \eta$ or $r \leq \tau$: **return** $x = e^y$ (early stop).
    b. Compute $x = e^y$; assemble $J_y = G \cdot \mathrm{diag}(x) + (1/\alpha)\,\mathrm{diag}(x^{-1/\alpha})$.
    c. Solve $J_y p = -F(y)$.
    d. Backtracking: $t \leftarrow 1$; while $\|F(\min(y + tp, y_{\max}))\| \geq r(1 - c_1 t)$, $t \leftarrow t/2$.
    e. $y \leftarrow \min(y + tp, y_{\max})$; $r \leftarrow \|F(y)\|$.
3. Return $x = e^y$ (non-converged case).

### 5c. Choice between x- and y-space

**Empirical comparison** ($\alpha = 2$, $K = 50$, extreme dead task $\|g\| = 10^{-20}$):

| residual target $\eta$ | x-Newton | y-Newton | winner |
|---|---|---|---|
| $10^{-2}$ (scipy-match) | 0.94 ms / 12 iter | **0.59 ms / 7 iter** | **y-space 1.6× faster** |
| $10^{-4}$               | 1.64 ms / 21 iter | **0.99 ms / 12 iter** | **y-space 1.7× faster** |
| $10^{-6}$               | 2.23 ms / 29 iter | **1.31 ms / 16 iter** | **y-space 1.7× faster** |
| $10^{-8}$               | **2.53 ms** (resid $10^{-9}$) | 4.39 ms (stalls at $10^{-7}$) | **x-space**: can push deeper |

**y-space is strictly preferable for $\eta \geq 10^{-6}$** (the practical range for FairGrad, where we want to match scipy's implicit $\eta_\text{trf} \approx 10^{-2}$).  **x-space is only better when pursuing true KKT convergence** ($\eta < 10^{-7}$), a regime we don't need.

The mechanism: y-space iteration count is smaller because the step size for dead coordinates is self-calibrating ($\Delta y_i = \alpha$, matched to the problem's natural log-scale), while x-space Newton tries large steps that require aggressive backtracking.

**Parameters.** Default $\eta = 10^{-2}$, $\tau = 10^{-8}$ (never binding: the $\eta$ stop fires first), $N = 50$, $\epsilon = 10^{-8}$, $y_{\max} = 50$ (corresponds to $x \lesssim 5 \cdot 10^{21}$, below fp64 exponent limit).

**Cost.** Per iteration: one $K \times K$ Jacobian assembly, one $K \times K$ linear solve ($O(K^3)$ via LAPACK), up to 30 $F$ evaluations for line search ($O(K^2)$ each). At $K = 50, N \approx 7$, total sub-millisecond on CPU.

**Guarantees.** On well-conditioned $G$ (no dead tasks): local quadratic convergence, $N \sim \log\log(1/\tau)$.  On dead-task $G$ at $\eta \geq 10^{-2}$: deterministic termination in fewer than 15 iterations in y-space, 30 in x-space.

## 6. Empirical validation (K=10, K=50)

Across 4 regimes (well-conditioned, heterogeneous, dead-task, extreme-dead) at $D = 5000$, $\alpha = 2$:

| regime | K | scipy-trf time | proj-Newton time | speedup | residual match |
|---|---|---|---|---|---|
| well-conditioned | 10 | 4.0 ms | 0.24 ms | **17×** | equal ($\sim 10^{-13}$) |
| heterogeneous    | 10 | 8.6 ms | 0.28 ms | **31×** | equal |
| dead-task        | 10 | 96 ms  | 3.1 ms  | **30×** | equal |
| extreme-dead     | 10 | 7.9 ms | 1.0 ms  | 7.9×   | proj-Newton **5 orders better** |
| well-conditioned | 50 | 16 ms  | 0.40 ms | **40×** | equal |
| heterogeneous    | 50 | 39 ms  | 0.79 ms | **49×** | equal |
| dead-task        | 50 | 548 ms | 4.1 ms  | **132×** | equal |
| extreme-dead     | 50 | 44 ms  | 1.6 ms  | 28×    | proj-Newton **4 orders better** |

The MT50 dead-task regime (the one that most closely matched the production Grams observed at the time) sees the largest speedup.

## 7. What shipped

`mtrl/gradients/fairgrad.py` implements the y-space Newton of §5b with residual target
$\eta = 10^{-2}$ (the residual plateau of scipy's trust region, §3–4), $\tau = 10^{-8}$,
$N = 50$, $\epsilon = 10^{-8}$ and $y_{\max} = 50$. Two details differ from this memo:

- **Diagonal warm start.** $y_i^0 = -\frac{\alpha}{\alpha+1}\log \max(G_{ii}, \epsilon)$, exact when
  $G$ is diagonal, replaces the uniform start.
- **Fallback.** scipy `least_squares` (bounds $(\epsilon, \infty)$) runs only when Newton does
  not converge, and starts from Newton's last iterate rather than from $\mathbf{1}/K$; if it fails
  too, the weights fall back to uniform. A non-finite-output guard sits on top.

The measured cost of this solver is in [README.md](README.md) and
[realgram_results.md](realgram_results.md).
