```@meta
CurrentModule = TrustRegionLeastSquares
```

# TrustRegionLeastSquares

A Levenberg–Marquardt trust-region solver for nonlinear least squares, with support for box
constraints, underdetermined problems and badly scaled variables. Source and benchmarks:
[TrustRegionLeastSquares.jl](https://github.com/vcantarella/TrustRegionLeastSquares.jl).

## Quick start

The residual and Jacobian are in-place: `res!(f, x)` fills the residual vector and `jac!(J, x)` the
Jacobian. The fourth argument is the number of residuals.

```julia
using TrustRegionLeastSquares

# Rosenbrock as a least-squares problem: f = [10(x₂ - x₁²), 1 - x₁]
rosenbrock!(f, x) = (f[1] = 10 * (x[2] - x[1]^2); f[2] = 1 - x[1]; f)
rosenbrock_jac!(J, x) = (J[1, 1] = -20 * x[1]; J[1, 2] = 10; J[2, 1] = -1; J[2, 2] = 0; J)

x, f, g, iter = lm_trust_region!(rosenbrock!, rosenbrock_jac!, [-1.2, 1.0], 2)
# x ≈ [1.0, 1.0]
```

## Choosing a strategy and a scaling

The fifth and sixth positional arguments pick how the trust-region subproblem is factorized and how
the variables are scaled:

```julia
x, f, g, iter = lm_trust_region!(
    rosenbrock!, rosenbrock_jac!, [-1.2, 1.0], 2,
    QRStrategy(),        # or QRCholStrategy() (default), LQStrategy(), LQCholStrategy()
    JacobianScaling(),   # or NoScaling() (default)
)
```

- [`QRCholStrategy`](@ref) is the default and the cheapest per iteration, but it forms `JᵀJ` and so
  squares the condition number.
- [`QRStrategy`](@ref) factorizes `[J; √λ D]` directly and stays accurate at any conditioning.
- [`LQStrategy`](@ref) and [`LQCholStrategy`](@ref) are for underdetermined problems (rows ≤ cols)
  and return the minimum-norm step.
- [`JacobianScaling`](@ref) applies Moré's non-decreasing column-norm scaling, which is what makes
  badly scaled problems tractable. But may introduce ill-condition,

## Box constraints

Pass `lb` and `ub`. Iterates stay feasible and may rest on a bound; convergence is measured by the
projected gradient. Here the upper bound on `x₁` moves the solution to `(0.5, 0.25)`:

```julia
x, f, g, iter = lm_trust_region!(
    rosenbrock!, rosenbrock_jac!, [-1.2, 1.0], 2;
    lb = [-2.0, -2.0], ub = [0.5, 2.0],
)
# x ≈ [0.5, 0.25], with x[1] resting on its bound
```

See [`lm_trust_region!`](@ref) for the trust-region and tolerance keywords, and the
[Reference](@ref reference) page for everything else.
