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


Solving the trust-region subproblem exactly is computationally more costly than the alternatives, such as updating the LM damping parameter by a heuristic (e.g. `NonlinearSolve.LevenbergMarquardt`) or solving the subproblem approximately (`NonlinearSolve.TrustRegion`, which uses the dogleg method). But the exact solve goes through careful factorizations, which keeps the number of Jacobian evaluations down. That is especially useful in real-world problems such as fitting an ODE or PDE model, or a simulation, where the residual and Jacobian evaluations dominate the run time. It is also often more robust than the alternatives (see Benchmarks below).

The current solution scheme has the following characteristics:

- **Unconstrained and box-constrained** least squares through one entry point.
- **Four factorization strategies**, including minimum-norm steps for underdetermined problems.
- **Moré's diagonal scaling** for variables that differ by orders of magnitude.
- **A reproducible benchmark suite** comparing it against nine other solvers across the Julia and Python ecosystems.


## Installation

```julia
using Pkg
Pkg.add("TrustRegionLeastSquares")
```

## The solver

- **Unconstrained NLLS**: Levenberg–Marquardt trust-region method, `min ‖J p + f‖` subject to `‖D p‖ ≤ Δ`, with the damping parameter found by safeguarded Newton iteration (Moré 1978).
- **Bounded NLLS**: the trust-region step in the Coleman–Li affine-scaled norm, projected onto the box and safeguarded by a generalized Cauchy step (Macconi, Morini & Porcelli 2009). Iterates stay feasible and may rest on a bound; no active-set management.
- **Four factorization strategies**: `QRCholStrategy` (pivoted QR for the Gauss–Newton step, Cholesky of `JᵀJ + λD²` for damped steps; the default), `QRStrategy` (QR of the augmented `[J; √λ D]`, stable at any conditioning), and `LQStrategy` / `LQCholStrategy` for underdetermined problems, which return the minimum-norm step via a complete orthogonal decomposition.
- **Buffer reuse in the hot loop**: one factorization of `J` per iteration, reused across candidate `λ` values.

### Usage

Define the residual (`res!(f, x)`) and Jacobian (`jac!(J, x)`) functions **in-place**:

```julia
using TrustRegionLeastSquares

# Rosenbrock as a least-squares problem: f = [10(x₂ - x₁²), 1 - x₁]
rosenbrock!(f, x) = (f[1] = 10 * (x[2] - x[1]^2); f[2] = 1 - x[1]; f)
rosenbrock_jac!(J, x) = (J[1, 1] = -20 * x[1]; J[1, 2] = 10; J[2, 1] = -1; J[2, 2] = 0; J)
```
Call the entry function to solve the nonlinear least-squares problem, giving as positional
arguments the initial guess `x0` and the number of residuals `n`:

```julia
x, f, g, iter = lm_trust_region!(rosenbrock!, rosenbrock_jac!, [-1.2, 1.0], 2)
# x ≈ [1.0, 1.0]
```

A different factorization strategy and variable scaling are also positional arguments:

```julia
x, f, g, iter = lm_trust_region!(
    rosenbrock!, rosenbrock_jac!, [-1.2, 1.0], 2,
    TrustRegionLeastSquares.QRStrategy(),        # subproblem strategy (default: QRCholStrategy())
    TrustRegionLeastSquares.JacobianScaling(),   # variable scaling   (default: NoScaling())
)
```

The existing factorization strategies are:

- `QRCholStrategy()`: solves the Newton step via QR factorization and the damped Newton step via Cholesky factorization. It is generally robust and fast, and it is the default choice.
- `QRStrategy()`: solves both the Newton and the damped-Newton step via QR. More robust than `QRChol`, but a bit slower. Recommended for very ill-conditioned problems.
- `LQCholStrategy()`: the recommended method for underdetermined systems (more parameters than residuals). Instead of a basic solution, the Newton and damped-Newton steps return the minimum-norm solution. It solves the Newton step via the LQ factorization of $J$ (a QR of $J^T$), and the damped step via a Cholesky factorization of $J J^T + \lambda I$, substituting $p = J^T z$ so that $(J J^T + \lambda I) z = -f$ is the only system that has to be solved.
- `LQStrategy()`: the same as `LQChol`, but solves both the Newton and the damped-Newton step with the LQ factorization. A bit slower than `LQChol`.

The default scaling is `NoScaling()`, which seems to be the better choice in general. You can scale the problem by the Jacobian (`JacobianScaling()`) to improve the solution, but that can also make the problem ill-conditioned. For a poorly conditioned problem I would rather rescale it in the residual and Jacobian definitions themselves, for example by applying a `log` or `exp` transformation to the variable `x`.

For box constraints, pass `lb` and `ub`. Here the upper bound on `x₁` moves the solution to `(0.5, 0.25)`, where `x₁` rests on its bound:

```julia
x, f, g, iter = lm_trust_region!(
    rosenbrock!, rosenbrock_jac!, [-1.2, 1.0], 2;
    lb = [-2.0, -2.0], ub = [0.5, 2.0],
)
# x ≈ [0.5, 0.25]
```

`lm_trust_region!` returns the solution, the residuals and gradient there, and the iteration count. See its docstring for the trust-region and tolerance keywords.

### Methodology

This package implements the nonlinear least-squares trust-region method described in Chapters 4 and 10 of Nocedal & Wright [[NW06]](#references). The extra sources behind each part are named below and listed under [References](#references). The overall scheme is close to the TRF method of SciPy's `least_squares()`, but we use carefully chosen factorizations (QR, Cholesky and the minimum-norm variants) rather than SciPy's SVD, which makes this package significantly faster at no cost in robustness.

**Trust region framework** [[NW06]](#references) Algorithm 4.1. At each iteration the algorithm minimizes a model of the objective `F(x) = 0.5 * ||f(x)||^2` around the current point:
```math
\min_p \| J_k p + f_k \|^2 \quad \text{subject to} \quad \| D_k p \| \le \Delta_k
```
where `J_k` is the Jacobian, `f_k` the residuals, `D_k` a scaling matrix, and `Δ_k` the trust region radius.

**Subproblem solution.** The constrained subproblem is equivalent to the regularized system
```math
(J_k^T J_k + \lambda D_k^T D_k) p = -J_k^T f_k
```
for a Lagrange multiplier `λ ≥ 0` ([[NW06]](#references) Theorem 4.1), found by safeguarded Newton iteration on `φ₂(λ) = 1/Δ - 1/‖Dp(λ)‖` — a reformulation chosen because it is nearly linear in `λ` near the root, where `‖Dp(λ)‖ = Δ` is not ([[NW06]](#references) §4.3), with the bracket and safeguard of [[Mor78]](#references). The factorization of `J` is computed once per iteration and reused across candidate `λ` values — this is why the solver can afford exact subproblem solves while keeping Jacobian evaluations to a minimum.

**Underdetermined problems** [[CJK26]](#references) Appendix B, [[SGJ26]](#references). With fewer residuals than parameters, `JᵀJ` is large and rank-deficient. The `LQStrategy` and `LQCholStrategy` variants instead solve the damped system through the small, full-rank `J Jᵀ`, which returns a regularized minimum-norm step, and use an LQ factorization so that the condition number is not squared.

**Bounds** [[MMP09]](#references), with the affine scaling of [[CL96]](#references). The trust region is measured in the affine-scaled norm `‖D_k |v(x)|^{-1/2} p‖ ≤ Δ_k`, where `|v_i|` is the distance from `x_i` to the bound its negative gradient points at, so a variable near an active bound can barely move towards it. The resulting step is projected onto the box, and accepted only if it achieves a fixed fraction of the decrease of a generalized Cauchy step along the scaled steepest descent; otherwise it is moved towards that Cauchy step until it does. That is the generalized Cauchy step of [[MMP09]](#references) eq. (8) and its fraction-of-decrease condition (11), and it is what makes the method globally convergent to a point satisfying the bound-constrained first-order conditions, measured by the projected gradient `‖x - P(x - g)‖`.

## Benchmarks

### Setup

```julia
using TrustRegionLeastSquares                           # v0.1 — TRLS
import NonlinearSolve: TrustRegion, LevenbergMarquardt  # v4.30
import JSOSolvers: tron                                 # v0.14
import LeastSquaresOptim: LevenbergMarquardt, QR        # v0.8
import LsqFit: curve_fit                                # v0.16
import NLLSsolver: levenbergmarquardt                   # v4.1
import Optim: BFGS, LBFGS                               # v2.3
import PRIMA: newuoa, bobyqa                            # v0.2 — deriv-free
scipy = pyimport("scipy")                               # SciPy 1.18 — TRF
using NLSProblems                                       # v0.5 — test set
using CairoMakie                                        # plotting
```
88 test problems from NLSProblems.jl; `gtol`, `ftol` = 1e-8 (see docs).

**Caveat: packages differ in termination criteria and interfaces, so timings are not perfectly apples-to-apples — per-solver settings in the repo docs.**


![NLLS solver performance](docs/src/assets/benchmarks/nlls_solver_performance.png)
*Figure 1: Performance profile and summary on the full unconstrained NLSProblems set (88 problems). TRLS and SciPy are the only solvers that reach a 100% success rate, and TRLS gets there in 0.283 s of cumulative solve time against SciPy's 1.335 s. Solvers with steeper early curves (NLLSsolver, LeastSquaresOptim) are faster per problem but plateau below 100%.*

On these small, microsecond-scale problems, per-iteration overhead dominates and the lightest wrappers win the left side of the profile:
- NLLSsolver solves its 97.7% faster than anyone else. The package also enforces the use of StaticArrays, so perhaps not a fair comparison.
- Most Julia packages have very fast solvers, about an order of magnitude faster than SciPy's baseline method.
- Only TRLS and SciPy solve 100% of the problems to the minimum cost reported. Of the two, TRLS is about 5× faster.



#### Real-world benchmark: expensive evaluations

To simulate a realistic model — where each Jacobian evaluation means re-solving an ODE/PDE or running a simulation — the second benchmark injects a 200 ms delay into every Jacobian *and* gradient evaluation (the gradient `Jᵀf` requires the Jacobian, so gradient-based solvers pay it too).

![NLLS solver performance with expensive Jacobians](docs/src/assets/benchmarks/nlls_solver_performance_delay.png)
*Figure 2: Same 88 problems with 200 ms per Jacobian and gradient evaluation; colors and markers as in Figure 1. Evaluation count now dominates wall-clock, and the few-iteration LM methods cluster at the front.*


#### Bound-constrained problems

![Bounded solver performance](docs/src/assets/benchmarks/bounded_solver_performance.png)
*Figure 3: Performance profile on 71 bound-constrained problems: CUTEst NLS problems (whose
residuals are encoded as constraints), the bound-constrained NLSProblems set, and 15 hand-written
`ADNLSModel` problems with a mix of active and inactive bounds. TRLS leads on success rate and is an
order of magnitude ahead of SciPy on cumulative time.*

## References

The solver implements published methods. Source comments cite these by key with a section,
algorithm or equation number; `docs/src/20-bibliography.md` says which part of the code came from
which result.

<a id="references"></a>

- **[NW06]** J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer (2006).
  [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5) — the trust-region
  iteration (Algorithm 4.1), the characterization of the subproblem solution (Theorem 4.1), the
  Cauchy point (Algorithm 4.2), the λ-iteration on `φ₂(λ) = 1/Δ - 1/‖p(λ)‖` (§4.3), and
  Gauss–Newton and Levenberg–Marquardt with the augmented-matrix implementation (§10.3). Most of
  this package comes from these two chapters.
- **[Mor78]** J. J. Moré, "The Levenberg–Marquardt algorithm: implementation and theory", in
  *Numerical Analysis*, G. A. Watson (ed.), Lecture Notes in Mathematics 630, Springer (1978),
  pp. 105–116. [doi:10.1007/BFb0067700](https://doi.org/10.1007/BFb0067700) — the λ bracket and
  safeguard, the non-decreasing diagonal scaling, and the radius update, as in MINPACK's `lmder`
  and `lmpar`.
- **[CJK26]** M. Chen, S. G. Johnson and A. Karalis, "Inverse design of multiresonance filters via
  quasi-normal mode theory", *Optics Express* **34**(4), 5729–5752 (2026).
  [doi:10.1364/OE.579219](https://doi.org/10.1364/OE.579219) ·
  [arXiv:2504.10219](https://arxiv.org/abs/2504.10219) — Appendix B, "Underdetermined
  Levenberg–Marquardt algorithm", which the `LQStrategy` and `LQCholStrategy` variants implement.
- **[SGJ26]** S. G. Johnson, reply in ["Should NonlinearLeastSquaresProblem be used for deep
  learning?"](https://discourse.julialang.org/t/should-nonlinearleastsquaresproblem-be-used-for-deep-learning/135793/4),
  Julia Discourse, 23 February 2026 — the post that prompted the LQ strategies and points at
  [CJK26].
- **[MMP09]** M. Macconi, B. Morini and M. Porcelli, "A Gauss–Newton method for solving
  bound-constrained underdetermined nonlinear systems", *Optimization Methods and Software*
  **24**(2), 219–235 (2009).
  [doi:10.1080/10556780902753031](https://doi.org/10.1080/10556780902753031) — the generalized
  Cauchy step (eq. 8) and the fraction-of-decrease condition (11) used for box constraints.
- **[CL96]** T. F. Coleman and Y. Li, "An interior trust region approach for nonlinear minimization
  subject to bounds", *SIAM Journal on Optimization* **6**(2), 418–445 (1996).
  [doi:10.1137/0806023](https://doi.org/10.1137/0806023) — the affine scaling `v(x)`. Only the
  scaling is taken from here: iterates in this solver may rest on a bound.
- **[MGH81]** J. J. Moré, B. S. Garbow and K. E. Hillstrom, "Testing unconstrained optimization
  software", *ACM Transactions on Mathematical Software* **7**(1), 17–41 (1981).
  [doi:10.1145/355934.355936](https://doi.org/10.1145/355934.355936) — the test problems and
  reference objective values the unit tests assert against.
- **[DM02]** E. D. Dolan and J. J. Moré, "Benchmarking optimization software with performance
  profiles", *Mathematical Programming* **91**(2), 201–213 (2002).
  [doi:10.1007/s101070100263](https://doi.org/10.1007/s101070100263) — the performance profiles in
  the figures above.
- **[Luk96]** L. Lukšan, "Hybrid methods for large sparse nonlinear least squares", *Journal of
  Optimization Theory and Applications* **89**(3), 575–595 (1996).
  [doi:10.1007/BF02275350](https://doi.org/10.1007/BF02275350) — the six hard exponential-fit
  problems in `hard_luksan.jl`.

## Status

In development, but the solver API is stable and tested: 562 unit tests across every strategy,
scaling and bound configuration, on Julia LTS and latest. The benchmark suite is intended as a
reproducible reference for comparing NLLS implementations.

If you use this package in work you publish, see `CITATION.cff`.
