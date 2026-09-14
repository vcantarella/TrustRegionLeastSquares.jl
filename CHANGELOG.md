# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [Unreleased]

## [0.1.0] - 2026-09-11

First release. The package was developed as `nonlinearlstr`, which could not be registered: the
General registry requires a name beginning with a capital letter.

### Added

- `lm_trust_region!`, a Levenberg–Marquardt trust-region solver for `min ½‖f(x)‖²`, with in-place
  residual and Jacobian callbacks. The trust-region subproblem is solved exactly, with the damping
  parameter found by safeguarded Newton iteration on `‖Dp‖ = Δ` (Moré 1978, MINPACK `lmpar`).
- Box constraints through the same entry point, via `lb` and `ub`. The step is computed in the
  Coleman–Li affine-scaled norm, projected onto the box, and required to achieve a fraction of the
  decrease of a generalized Cauchy step (Macconi, Morini & Porcelli 2009). Iterates stay feasible
  and may rest on a bound; convergence is measured by the projected gradient.
- Four factorization strategies: `QRCholStrategy` (default), `QRStrategy`, and `LQStrategy` /
  `LQCholStrategy` for underdetermined problems, which return the minimum-norm step through a
  complete orthogonal decomposition and so also handle rank-deficient Jacobians.
- Two variable scalings: `NoScaling` and `JacobianScaling`, the latter being Moré's non-decreasing
  column-norm scaling from MINPACK `lmder`.
- A cross-package benchmark suite under `benchmark/`, comparing the solver against NonlinearSolve,
  JSOSolvers, LeastSquaresOptim, LsqFit, NLLSsolver, Optim, PRIMA and SciPy on NLSProblems and
  CUTEst, with the fairness decisions recorded in `benchmark/AUDIT.md`.

### Known limitations

- The Cholesky strategies form `JᵀJ`, squaring the condition number. Accuracy degrades from about
  `cond(J) = 1e8`, and beyond roughly `1e12` the factorization can fail with a `PosDefException`.
  Use `QRStrategy` on ill-conditioned problems; this is documented on the strategy itself.
- Julia 1.10 is supported for use, but the repository's single-manifest workspace needs 1.12, so on
  1.10 run the tests through `Pkg.test()` rather than `julia --project=test`.

<!-- Links -->

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

<!-- Versions -->

[unreleased]: https://github.com/vcantarella/TrustRegionLeastSquares.jl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/vcantarella/TrustRegionLeastSquares.jl/releases/tag/v0.1.0
