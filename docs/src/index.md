# nonlinearlstr.jl

Documentation for [nonlinearlstr.jl](https://github.com/vcantarella/nonlinearlstr).

## Overview

nonlinearlstr.jl is a Julia package for nonlinear least squares optimization using trust region methods.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/vcantarella/nonlinearlstr")
```

## Quick Start

The residual and Jacobian are in-place; the fourth argument is the number of residuals.

```julia
using nonlinearlstr

residual!(f, x) = (f[1] = x[1]^2 + x[2]^2 - 1; f[2] = x[1] - x[2]; f)
jacobian!(J, x) = (J[1, 1] = 2x[1]; J[1, 2] = 2x[2]; J[2, 1] = 1; J[2, 2] = -1; J)

x, f, g, iter = lm_trust_region!(residual!, jacobian!, [0.5, 0.5], 2)
```

Box constraints are keyword arguments:

```julia
x, f, g, iter = lm_trust_region!(residual!, jacobian!, [0.5, 0.5], 2; lb = [0.0, 0.0], ub = [0.6, 1.0])
```

## Features

- Levenberg–Marquardt trust-region method with exact subproblem solves
- Four factorization strategies, including minimum-norm steps for underdetermined problems
- Moré's diagonal scaling for badly scaled variables
- Bound constraints via Coleman–Li scaling with a projected, Cauchy-safeguarded step
