# TrustRegionLeastSquares.jl

[![Build Status](https://github.com/vcantarella/TrustRegionLeastSquares.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/vcantarella/TrustRegionLeastSquares.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://vcantarella.github.io/TrustRegionLeastSquares.jl/dev)
[![Coverage](https://codecov.io/gh/vcantarella/TrustRegionLeastSquares.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/vcantarella/TrustRegionLeastSquares.jl)

A Levenberg–Marquardt (LM) trust-region (TR) solver for nonlinear least squares in Julia, based on solving the trust-region subproblem exactly.

Features:
* box constraints
* underdetermined problems
* badly scaled variables
* Pure Julia, one dependency (`LinearAlgebra`).

Solving the trust-region subproblem exactly is more involved per iteration than the alternatives, such as updating the LM damping parameter by a heuristic (e.g. `NonlinearSolve.LevenbergMarquardt`) or solving the subproblem approximately by the dogleg method (`NonlinearSolve.TrustRegionDogleg`). `NonlinearSolve.TrustRegion` also solves the same subproblem nearly exactly, so the differentiator here is not exactness itself but the factorization strategies — including minimum-norm steps for underdetermined problems — and a step policy that keeps the number of Jacobian evaluations down. That is especially useful in real-world problems such as fitting an ODE or PDE model, or a simulation, where the residual and Jacobian evaluations dominate the run time. It is also often more robust than the alternatives (see the [benchmarks](https://vcantarella.github.io/TrustRegionLeastSquares.jl/dev/15-benchmarks/)).

## Installation

```julia
using Pkg
Pkg.add("TrustRegionLeastSquares")
```

## Usage

Define the residual (`res!(f, x)`) and Jacobian (`jac!(J, x)`) functions **in-place**, then pass
them with the initial guess `x0` and the number of residuals:

```julia
using TrustRegionLeastSquares

# Rosenbrock as a least-squares problem: f = [10(x₂ - x₁²), 1 - x₁]
rosenbrock!(f, x) = (f[1] = 10 * (x[2] - x[1]^2); f[2] = 1 - x[1]; f)
rosenbrock_jac!(J, x) = (J[1, 1] = -20 * x[1]; J[1, 2] = 10; J[2, 1] = -1; J[2, 2] = 0; J)

x, f, g, iter = lm_trust_region!(rosenbrock!, rosenbrock_jac!, [-1.2, 1.0], 2)
# x ≈ [1.0, 1.0]
```

The factorization strategy and variable scaling are optional positional arguments, and box
constraints are keywords:

```julia
x, f, g, iter = lm_trust_region!(
    rosenbrock!, rosenbrock_jac!, [-1.2, 1.0], 2,
    QRStrategy(),        # or QRCholStrategy() (default), LQStrategy(), LQCholStrategy()
    JacobianScaling();   # or NoScaling() (default)
    lb = [-2.0, -2.0], ub = [0.5, 2.0],
)
# x ≈ [0.5, 0.25]
```

`QRCholStrategy` is the fast default; `QRStrategy` is for very ill-conditioned problems;
`LQCholStrategy` and `LQStrategy` return minimum-norm steps for underdetermined problems (more
parameters than residuals).

## Documentation

- [Quick start](https://vcantarella.github.io/TrustRegionLeastSquares.jl/dev/): strategies, scaling and bounds
- [Theory](https://vcantarella.github.io/TrustRegionLeastSquares.jl/dev/10-theory/): the method and its factorizations
- [Benchmarks](https://vcantarella.github.io/TrustRegionLeastSquares.jl/dev/15-benchmarks/): comparison against eleven other Julia and Python solvers
- [Bibliography](https://vcantarella.github.io/TrustRegionLeastSquares.jl/dev/20-bibliography/): the published methods the solver implements
- [API reference](https://vcantarella.github.io/TrustRegionLeastSquares.jl/dev/95-reference/)

If you use this package in work you publish, see `CITATION.cff`.
