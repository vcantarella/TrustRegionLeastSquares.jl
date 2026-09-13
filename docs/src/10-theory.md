```@meta
CurrentModule = TrustRegionLeastSquares
```

# Theory

>Throughout, the residual is $r: \mathbb{R}^n \to \mathbb{R}^m$ and its Jacobian $J$ is $m \times n$: **$m$ residuals, $n$ parameters**. A problem is *overdetermined* when $m > n$ and *underdetermined* when $m < n$.

## Nonlinear Least-squares

```math
f(x) = \frac{1}{2} \sum_{j=1}^{m} r_j^2(x) = \frac{1}{2} \|r(x)\|^2
```

Least-squares problems have **a nice structure**: the Jacobian ($J$) of the residual function ($r$)
gives you the gradient of the loss function,

```math
\nabla f(x) = J(x)^T r(x),
```

and most of the **Hessian**,

```math
\nabla^2 f(x) = J^T J + \sum_{j=1}^{m} r_j \nabla^2 r_j \approx J^T J ,
```

where the approximation is good when the residuals are small, or when the model is close to linear
near the solution. So the Jacobian alone buys you the gradient exactly and the Hessian nearly for
free.

> If you use autodiff to get a gradient of $f$ and hand it to a general-purpose optimizer, e.g.
> BFGS, you are throwing this structure away — see [Benchmarks](@ref benchmarks) for the performance
> penalty!

**Typical use case:** fitting a model ($\hat{y}$) to observations/data ($y$):
$r_j(x) = y_j - \hat{y}_j(x)$.

## Trust-Region Methods

Most *robust* and successful nonlinear least-squares solvers use some form of the *trust-region*
method to constrain the solution of the subproblem:

```math
\min_p \frac{1}{2} \|J p + r\|^2 \quad \text{subject to} \quad \|p\| \leq \Delta .
```

The idea behind it is that subproblem is trustworth is only around the current point (a ball of radius $\Delta$).the subproblem was built from derivatives atthe current iterate and therefore it is not reliable far from it.

This arrives at the following optimality conditions [NW06; Theorem 4.1](@cite): $p$
solves the subproblem if and only if there is a $\lambda \geq 0$ with

```math
(J^T J + \lambda I) p = -J^T r , \qquad \lambda (\Delta - \|p\|) = 0 ,
```

and $J^TJ + \lambda I$ positive semidefinite.

The second condition states that either
$\lambda = 0$, the unconstrained step already fits inside the region, or the step sits exactly on
the boundary, $\|p\| = \Delta$.

### Exact Trust Region Algorithm
The main algorithm consists of:

1. Compute $r(x_k)$ and $J(x_k)$ at the current iterate.
2. Solve the subproblem: if the Gauss–Newton step ($\lambda = 0$) satisfies $\|p\| \leq \Delta_k$,
   take it; otherwise find $\lambda > 0$ such that $\|p(\lambda)\| = \Delta_k$.
   > Solvers need a way to pin down $\lambda$: *Levenberg–Marquardt (LM)* implementations define and
   > update $\lambda$ directly, whereas *Trust-Region (TR)* methods update $\Delta$. The popular
   > current Julia TR implementations only approximate the matching $\lambda$, e.g. with the dogleg
   > method.
3. Evaluate the step, update $\lambda$ or $\Delta$, and continue until convergence.

Finding the $\lambda$ that matches the radius is a one-dimensional root-finding problem, and this
package solves it by safeguarded Newton iteration on

```math
\varphi_2(\lambda) = \frac{1}{\Delta} - \frac{1}{\|p(\lambda)\|} ,
```

rather than on $\|p(\lambda)\| - \Delta$ directly. The reason is that $\varphi_2$ is nearly linear in
$\lambda$ near its root, so Newton's method converges in a handful of steps
[NW06; §4.3](@cite), while $\|p(\lambda)\|$ is highly nonlinear there.

Each of those steps needs the damped system solved again at a new $\lambda$ — which is what makes
the choice of factorization worth thinking about.

## Factorization strategies

### Standard QR and QRChol

For the undamped Gauss–Newton step ($\lambda = 0$), the condition above reduces to the **normal
equations**:

```math
(J^T J) p = -J^T r .
```

These are the first-order conditions of the linear least-squares problem $\min_p \|Jp + r\|$, so we
do not have to form $J^TJ$ at all: we can solve

```math
\min_p \|J p + r\|
```

directly from a factorization of $J$ itself, which in Julia is the left division

```julia
p = J \ -r
```

In this package that is a column-pivoted QR of $J$.

Forming $J^TJ$ requires $J$ to have full column rank to be invertible, and the pivoted QR does not:
when $J$ is rank-deficient it still returns a *basic* solution (one with at most $\text{rank}(J)$
nonzeros), which is usually good enough for a nonlinear least-squares step.

More importantly, the QR route is numerically better behaved than a Cholesky factorization of
$J^TJ$, because forming $J^TJ$ **squares the condition number**:

When the Gauss–Newton step falls outside the trust region, we need the damped system

```math
(J^T J + \lambda I) p = -J^T r
```

for the $\lambda$ that puts $\|p\| = \Delta$. Rather than factorizing that whole left-hand side, note
that it is itself a normal-equations system for a taller least-squares problem:

```math
\begin{bmatrix} J \\ \sqrt{\lambda} I \end{bmatrix}^T
\begin{bmatrix} J \\ \sqrt{\lambda} I \end{bmatrix} p =
\begin{bmatrix} J \\ \sqrt{\lambda} I \end{bmatrix}^T
\begin{bmatrix} -r \\ 0 \end{bmatrix} ,
```

since the left factor multiplies out to $J^TJ + \lambda I$ and the right to $-J^Tr$. So we just do a
QR of the augmented matrix $[J;\ \sqrt{\lambda} I]$ and never square anything. That is
[`QRStrategy`](@ref).

[`QRCholStrategy`](@ref) is similar, but factorizes $(J^TJ + \lambda I)$ by Cholesky instead,
assuming the $\lambda$ term improves the conditioning enough to make that safe. `cholesky` is much
faster than QR — it is the default for that reason. Ill-conditioned matrices can still occur, which are numerically
indefinite in practice, and the factorization fails. In that case it is better to rely on `QRStrategy`.

### LQ and LQChol

When the system is **underdetermined**: $J$ has fewer rows than columns ($m < n$), i.e. there
are more parameters than residuals. Then $Jp = -r$ has infinitely many solutions. Any two of them
differ by a vector in the null space of $J$, which has dimension $n - m$.

In that case the **minimum-norm solution**, the one that moves the least:

```math
p^\star = \arg\min \|p\| \quad \text{subject to} \quad J p = -r .
```

That is the right default for two reasons. The step stays inside the region where the linear model
was built, which is the whole premise of a trust region; and it does not wander along directions the
residuals say nothing about, which is what keeps the iterates from drifting on the solution manifold.

#### The undamped step

The solution to that constrained problem is

```math
p^\star = J^T (J J^T)^{-1} (-r),
```

Note that the matrix to invert is $J J^T$,
which is $m \times m$, rather than $J^TJ$, which is $n \times n$. When $n \gg m$ this means solving a much smaller problem.

In the LQ strategy, we do not form $J J^T$ either. Take a column-pivoted QR of the **transpose**,

```math
J^T P = Q R \quad \Longleftrightarrow \quad J = P R^T Q^T ,
```

which is an LQ factorization of $J$, since $R^T$ is lower trapezoidal. Substituting into $Jp = -r$:

```math
P R^T Q^T p = -r \quad \Longrightarrow \quad R^T (Q^T p) = -P^T r .
```

Now write the two sides in blocks. $R^T$ is $m \times n$ and splits as $R^T = [\,L \ \ 0\,]$ with $L$
lower triangular and $m \times m$; write $Q^Tp = [\,y;\ w\,]$ the same way, $y$ the first $m$
components. The equation then reads $Ly = -P^Tr$: the tail $w$ has dropped out entirely, because the
zero block multiplies it. That $w$ is the null-space component, free to be anything. Since $Q$ is
orthogonal it does not change lengths, so

```math
\|p\|^2 = \|Q^Tp\|^2 = \|y\|^2 + \|w\|^2 ,
```

and the minimum-norm choice is simply $w = 0$:

```math
L y = -P^T r \quad \text{(lower-triangular solve)}, \qquad p = Q \begin{bmatrix} y \\ 0 \end{bmatrix} .
```

If $L$ turns out to be rank-deficient, a second QR of the leading rows of $R$ — a *complete
orthogonal decomposition* — recovers the minimum-norm solution in that case too.

#### The damped step

For the damped system the same trick works.

First thing is to replace in the damped newton equation $p$ for $J^T z$. Then:

```math
(J^T J + \lambda I) J^T z = J^T (J J^T + \lambda I) z ,
```

so it is enough to solve

```math
(J J^T + \lambda I) z = -r , \qquad p = J^T z .
```

Three things fall out of this at once:

- The system is $m \times m$, the small dimension, and we never build an $n \times n$ matrix.
- It is positive definite for any $\lambda > 0$, **whatever the rank of $J$**. The damping does the
  regularizing that the underdetermined problem needs anyway.
- $p = J^Tz$ lies in the range of $J^T$, which is the orthogonal complement of the null space of $J$.
  So the step is automatically the *regularized minimum-norm step*.


The two strategies differ only in how that system is solved:

- [`LQCholStrategy`](@ref) forms $J J^T + \lambda I$ and factorizes it by Cholesky. It is
  $m \times m$ and fast, and it squares the condition number just as `QRChol` does.
- [`LQStrategy`](@ref) avoids that by a QR of the augmented matrix
  $[J^T;\ \sqrt{\lambda} I\,]$, using the same identity as before:

```math
\begin{bmatrix} J^T \\ \sqrt{\lambda} I \end{bmatrix}^T
\begin{bmatrix} J^T \\ \sqrt{\lambda} I \end{bmatrix}
= J J^T + \lambda I ,
```

  so its $R$ satisfies $R^TR = J J^T + \lambda I$ without the product ever being formed.

The idea of solving the underdetermined LM step through $J J^T$ with an LQ factorization is from
[CJK26; Appendix B](@cite), pointed out in [Joh26](@cite).

> `LQChol` is the faster of the two; switch to `LQ` if the Jacobian is badly conditioned. 
