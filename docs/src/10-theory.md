# Theory

## Nonlinear Least-squares

$$ f(x) = 1 / 2 \sum_{j=1}^k r_j ^2 (x) $$

Least-squares problems have **a nice structure**: the residual function's ($r$) Jacobian ($J$) provides the gradient of the loss function ($\nabla f = J(x)^T r(x)$) and most of the **Hessian** ($\nabla^2 f = J^T J + \sum_j r_j \nabla^2 r_j$ approx $J^T J$ for small residuals).

>If you use autodiff to get a gradient of $f$ and hand it to a general-purpose optimizer,e.g., BFGS, you are throwing this structure away, see Banchmarks for the performance penalty!

**Typical use case:** Fitting a model ($m$) to observations/data ($y$): $r_j(x) = y - m(x)$

## Trust-Region Methods

Most *robust* and successful nonlinear least-squares solvers use some form of the *trust-region* method to constrain the solution to the subproblem:

$$ \min_p \frac{1}{2} ||J p  + r||^2 $$

,subject to $||p|| \leq \Delta$.

This arrives at the following optimality conditions:
$$ (J^T J + \lambda I)p = -J^T r $$

$$ \lambda(\Delta - ||p||) = 0 $$

The main algorithm consists of:
+ Compute $r(x_k)$, $J(x_k)$ at the current iterate.
+ Solve the subproblem: if the Gauss–Newton step ($\lambda = 0$) satisfies $||p|| <= \Delta_k$, take it; otherwise find $\lambda > 0$ such that $||p(\lambda)|| = \Delta_k$.
    > Solvers need a way to pin down $\lambda$: *Levenberg–Marquardt (LM)* implementations define and update $\lambda$ directly, whereas *Trust-Region (TR)* methods update $\Delta$. Current Julia's TR popular implementations only approximate the matching $\lambda$, e.g., with the dogleg method.

3. Evaluate the solution, update $\lambda$ or $\Delta$ and continue until convergence.

## Factorization strategies

### Standard QR and QRChol

For the standard Newton step, without trust region penalty the Gauss-Newton step is:

$$ (J^TJ)p = -J^Tr $$

We may simplify the equation by left-multiplying both sides by $J^{-T}$:

$$ Jp = -r $$

and we solve for the step $p$ with the left division operator:

$$ p = J \ -r $$

Contrary to the standard formulation, where $J^TJ$ is invertible, the Jacobian $J$ is not guaranteed to be invertible. Nevertheless, the QR factorization will work and produce a solution (basic solution), which is often sufficient for nonlinear least-squares problems.

This is often more robust than standard cholesky factorization of $J^TJ$, because the multiplication $J^TJ$ squares the condition number, which may make the inversion numerically unstable.

When the solution is not within the trust-region radius, we need to solve the damped newton formulation within a newton update scheme to find the $\lambda$ that matches the radius:

$$ (J^TJ + \lambda I)p = -J^T r$$

,subject to $||p|| \leq \Delta$.

Instead of QR the whole left side, we can factorize both sides as :

$$
[J; \sqrt \lambda I]^T [J; \sqrt \lambda I] p = [J; \sqrt \lambda I]^T [-r; 0]
$$

and we just do QR on $[J; \sqrt \lambda I]$.

The `QRChol()` strategy is similar, but instead uses the cholesky factorization of $(J^TJ + \lambda I)$ because it assumes that the $\lambda$ addition improves the conditioning well enough to makes this operation possible. `cholesky` is much faster than QR.

### LQ and LQChol

When the system is undetermined (the Jacobian has more rows `m` than columns `n`, i.e., there are more parameters than residuals). The solution to the inversion (left division) is not unique. Therefore it is convenient to derive the minimum norm solution, the solution that deviates the least from the initial condition.


> Deriving the minimum-norm solution for the LQ and the damped newton equation.