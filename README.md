# TrustRegionLeastSquares.jl

[![Build Status](https://github.com/vcantarella/TrustRegionLeastSquares.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/vcantarella/TrustRegionLeastSquares.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://vcantarella.github.io/TrustRegionLeastSquares.jl/dev)
[![Coverage](https://codecov.io/gh/vcantarella/TrustRegionLeastSquares.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/vcantarella/TrustRegionLeastSquares.jl)

A Levenberg–Marquardt trust-region solver for nonlinear least squares in Julia, with support for
box constraints, underdetermined problems and badly scaled variables. Pure Julia, one dependency
(`LinearAlgebra`).

It is built for the case where **evaluating the model is the expensive part** — the residual is an
ODE solve, a PDE solve, a simulation — so the solver spends effort per iteration (exact trust-region
subproblem solves, careful factorizations) to keep the number of Jacobian evaluations down. On the
88-problem NLSProblems set it solves every problem, at a median of 12.5 iterations, and it is the
only solver besides SciPy to do so. When each Jacobian evaluation is made to cost 200 ms, standing in
for a model that is an ODE solve or a simulation, it has both the lowest median and the lowest
cumulative solve time of any solver at 100% success. On microsecond-scale toy problems, lighter
wrappers are faster per problem; that trade is deliberate and is documented in the benchmarks below.

- **Unconstrained and box-constrained** least squares through one entry point.
- **Four factorization strategies**, including minimum-norm steps for underdetermined problems.
- **Moré's diagonal scaling** for variables that differ by orders of magnitude.
- **A reproducible benchmark suite** comparing it against nine other solvers across the Julia and
  Python ecosystems, with the fairness work written down rather than assumed.

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

The residual and Jacobian are **in-place**: `res!(f, x)` fills the residual vector, `jac!(J, x)` the Jacobian. The fourth argument is the number of residuals.

```julia
using TrustRegionLeastSquares

# Rosenbrock as a least-squares problem: f = [10(x₂ - x₁²), 1 - x₁]
rosenbrock!(f, x) = (f[1] = 10 * (x[2] - x[1]^2); f[2] = 1 - x[1]; f)
rosenbrock_jac!(J, x) = (J[1, 1] = -20 * x[1]; J[1, 2] = 10; J[2, 1] = -1; J[2, 2] = 0; J)

x, f, g, iter = lm_trust_region!(rosenbrock!, rosenbrock_jac!, [-1.2, 1.0], 2)
# x ≈ [1.0, 1.0]
```

A different factorization strategy and variable scaling are positional arguments:

```julia
x, f, g, iter = lm_trust_region!(
    rosenbrock!, rosenbrock_jac!, [-1.2, 1.0], 2,
    TrustRegionLeastSquares.QRStrategy(),        # subproblem strategy (default: QRCholStrategy())
    TrustRegionLeastSquares.JacobianScaling(),   # variable scaling   (default: NoScaling())
)
```

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

This package is an implementation of published methods, not a new one. Chapters 4 and 10 of
Nocedal & Wright [[NW06]](#references) supply the framework and most of the details; the sources
for each part are named below and listed under [References](#references), and the source comments
cite them by the same keys with section, algorithm or equation numbers.

**Trust region framework** [[NW06]](#references) Algorithm 4.1. At each iteration the algorithm minimizes a model of the objective `F(x) = 0.5 * ||f(x)||^2` around the current point:
```math
min_p  || J_k p + f_k ||^2  \quad \text{subject to} \quad || D_k p || \le \Delta_k
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

### What is compared

| Solver (label in figures) | Package | Method |
|---|---|---|
| TRLS | `TrustRegionLeastSquares.jl` v0.1 | LM trust region, `QRStrategy` |
| NonlinearSolve-TR / -LM | NonlinearSolve.jl v4.30 | TrustRegion, LevenbergMarquardt |
| NonlinearSolve-GNBK / -GNLF | NonlinearSolve.jl v4.30 | GaussNewton with backtracking and with Li–Fukushima line search |
| JSO-TRON | JSOSolvers.jl v0.14 | TRON |
| LSO-Levenberg-QR | LeastSquaresOptim.jl v0.8 | Levenberg–Marquardt (QR) |
| LsqFit-LM | LsqFit.jl v0.16 | Levenberg–Marquardt |
| NLLSsolver-LM | NLLSsolver.jl v4.1 | Levenberg–Marquardt |
| Optim-BFGS / -L-BFGS | Optim.jl v2.3 | quasi-Newton on 0.5‖r‖² |
| PRIMA-NEWUOA | PRIMA.jl v0.2 | NEWUOA (derivative-free; bounded and hard-problem runs only) |
| Scipy-LeastSquares | SciPy 1.18 (Python 3.13, via PythonCall) | `least_squares` (TRF) — the **baseline** |

Test problems come from **NLSProblems.jl** v0.5 (via NLPModels.jl); CUTEst problems are supported by the same harness. Measured on Julia 1.12, macOS aarch64; exact package versions are pinned in `benchmark/Manifest.toml`.

The `TRLS` row runs the `QRStrategy` variant, which is what the figures show. The package default is
`QRCholStrategy`, which is faster per iteration but squares the condition number; `internal_variants.jl`
compares all four strategies against each other.

### Fairness

- Every solver receives the **same analytic residual and Jacobian**, wrapped as in-place closures with buffers preallocated once, so no solver pays extra allocation or wrapper cost per evaluation.
- Timing is the **minimum over repeated solves** (Chairmarks.jl), with problem construction excluded from the timed region.
- A run counts as a **success** only if it reaches the best cost found by *any* solver on that problem (within 1e-4, absolute or relative) — converging to a worse local minimum does not count. The success metric is therefore **cost-based** and independent of each package's self-reported convergence flag.

### Termination criteria: one value, each solver's own criteria

Solver packages implement different stopping *criteria*, so they cannot be forced onto
one identical rule. Instead, **every tolerance a solver exposes is set to 1e-8**, and
each solver terminates through whatever criteria it natively implements at that value:

| Criterion class | Who tests it (all at 1e-8) |
|---|---|
| First-order (gradient) | TRLS (projected gradient `‖x − P(x − g)‖₂`, which is `‖Jᵀr‖₂` when unbounded), SciPy `gtol` (∞-norm, scaled), Optim `g_tol` (∞-norm), LsqFit `g_tol`, TRON `atol` (projected gradient, `rtol=0`), LSO `g_tol` (∞-norm) |
| Step size | **Disabled** where it is a single-step give-up test (SciPy `xtol = None`, LSO/LsqFit `x_tol = 0`): one small step is the weakest evidence of optimality — it fires during slow crawls — and "this work" has no such test, so removing it keeps the criterion sets symmetric. Kept where structural: NonlinearSolve's stall detector (32 *consecutive* steps ≤ `abstol` — its only exit at nonzero residual) and PRIMA's `rhoend` (a DFO method's resolution parameter) |
| Cost stagnation | TRLS `ftol` (relative: reduction against the cost itself), SciPy `ftol`, LSO `f_tol`, NLLSsolver `reldcost` |
| Residual norm | NonlinearSolve `abstol` (on `‖r‖₂`) |
| Trust-region resolution | PRIMA `rhoend` (the x-resolution of a derivative-free method) |

A caveat stated openly: tolerances on different quantities are *not* equally strict even
at the same number — near a minimum, cost error scales as the square of parameter error,
and gradient scales carry the Jacobian's magnitude. A single value does not remove that
incommensurability; it removes the *choice* of per-class pairing as a degree of freedom.
What makes the comparison robust is the success metric: a run counts as a success based
on its achieved cost (within 1e-4 of the best found), which is orders of magnitude
looser than any 1e-8 criterion, so reported success rates are insensitive to the exact
tolerance value — the tolerances only decide when a solver stops polishing.

Iteration budget: **`max_iter = 400`** for every iterative solver (passed explicitly,
including to LSO whose own default is 1000). Hidden wall-clock caps are removed
(TRON and NLLSsolver default to a 30 s `max_time`; the suite lifts both). One cap is
added deliberately, and only in the delay benchmark: Optim-BFGS and Optim-L-BFGS get
`time_limit = 900` (15 minutes), because the `iterations` budget does not bound the
gradient evaluations their line search makes, and at 200 ms per gradient those two rows
were half of that benchmark's entire runtime. It is three orders of magnitude above the
scale at which they normally finish, a capped run is scored on the cost it reached, and
`benchmark/AUDIT.md` sets out why this is not the same as the hidden defaults that were
removed.

### Known limitations (read before citing numbers)

- **NonlinearSolve has no gradient-based stopping for NLLS.** Its API terminates on the
  residual norm (`abstol`) plus a stalled-step detector (32 consecutive steps below
  `abstol` → `StalledSuccess`, its documented "found a local minimum" exit). On the
  **30/88 problems with nonzero residual at the solution**, no residual tolerance can
  fire, so its exit there is the stall path — the same criterion *class* as this work's
  trust-radius-collapse exit (`radius < 1e-8`), but it pays a ~32-iteration confirmation
  tail that our immediate exit does not. This mismatch is inherent to its API; treat
  small timing gaps against NonlinearSolve accordingly.
- **NLLSsolver stops on cost decrease only** (`reldcost = 1e-8`); it exposes no gradient
  criterion. Different class from everyone else, kept because it is what the package offers.
- **PRIMA-NEWUOA is derivative-free** and budgeted as `maxfun = 400 × n_vars`
  (one "iteration" ≈ one model rebuild). It stops at trust-region radius
  `rhoend = 1e-8`. Shown for reference, not as a like-for-like competitor.
- **SciPy is called from Julia via PythonCall**: every residual/Jacobian evaluation
  crosses the Julia↔Python boundary, and SciPy's timings include that overhead. Its
  budget is `max_nfev = 1000` *evaluations* (not 400 iterations — TRF uses ~1–3
  evaluations per iteration, so the budgets are comparable but not identical units).
- **Norm conventions differ**: this work tests `‖g‖₂ ≤ 1e-8` while SciPy/Optim/LSO test
  ∞-norm variants. Since `‖g‖∞ ≤ ‖g‖₂`, competitors stop at or before our criterion —
  the mismatch, where it matters, favors the competitors' timings.
- **Iteration counts are not comparable across solvers** (LM iterations vs. SciPy
  Jacobian evaluations vs. PRIMA function evaluations; LsqFit does not expose a count and
  is recorded as 0). They appear in the CSVs for context and are never plotted.
- **The delay benchmark (Figure 2) delays only Jacobian and gradient evaluations**
  (200 ms); residual and objective evaluations stay cheap. This *flatters* line-search
  methods (BFGS/L-BFGS), whose objective-only line-search evaluations would also be
  expensive in a real simulator — their Figure 2 results are favorable upper bounds.
- **Remaining wrapper asymmetries**: TRON consumes the NLPModels model directly rather
  than the extracted closures (thin call-path difference); NLLSsolver's static-size
  residual API requires a per-evaluation `collect`/`SVector` wrapper (overhead the others
  don't pay — it is nonetheless the fastest solver on small problems); Optim's
  quasi-Newton baselines run the scalar `0.5‖r‖²` formulation with out-of-place closures.

### Metrics

Figures report **Dolan–Moré performance profiles** [[DM02]](#references): for each solver and problem, the performance ratio τ = time / best solver's time on that problem; the curve shows the fraction of all problems solved within τ. The height at τ = 1 reads as "how often is this solver the fastest", the right-hand asymptote as robustness. The summary table gives the success rate and the cumulative-time speed-up relative to the SciPy baseline.

### Running it

```bash
# 1. Run the benchmarks (writes raw results to benchmark/results/*.csv)
julia --project=benchmark benchmark/scripts/compare_unconstrained.jl
julia --project=benchmark benchmark/scripts/compare_with_delay.jl

# 2. Build the figures from the saved results (seconds; no solver runs)
julia --project=benchmark benchmark/scripts/plot_results.jl
```

Both benchmark scripts accept `MAX_VARS` and `PROBLEM_LIMIT` environment variables for quick capped runs (e.g. `PROBLEM_LIMIT=5 julia --project=benchmark ...`). Plotting is deliberately decoupled from benchmarking: figure styling can be iterated without re-running solvers. Additional scripts cover bound-constrained problems (`compare_bounded.jl`) and internal subproblem-strategy comparisons (`internal_variants.jl`).

### Results

#### Standard benchmark: cheap evaluations

![NLLS solver performance](docs/src/assets/benchmarks/nlls_solver_performance.png)
*Figure 1: Performance profile and summary on the full unconstrained NLSProblems set (88 problems). TRLS and SciPy are the only solvers that reach a 100% success rate, and TRLS gets there in 0.283 s of cumulative solve time against SciPy's 1.335 s. Solvers with steeper early curves (NLLSsolver, LeastSquaresOptim) are faster per problem but plateau below 100%.*

| Solver | Success % | Median iterations | Cumulative time |
|---|---|---|---|
| **TRLS** | **100.0** | 12.5 | 0.283 s |
| Scipy-LeastSquares | **100.0** | 13.0 | 1.335 s |
| NLLSsolver-LM | 97.7 | 13.5 | 0.034 s |
| NonlinearSolve-TR | 94.3 | 33.0 | 0.110 s |
| NonlinearSolve-LM | 94.3 | 39.0 | 0.171 s |
| JSO-TRON | 93.2 | 16.0 | 0.117 s |
| NonlinearSolve-GNBK | 90.9 | 16.0 | 0.312 s |
| Optim-L-BFGS | 90.9 | 33.0 | 0.219 s |
| Optim-BFGS | 89.8 | 32.0 | 0.133 s |
| NonlinearSolve-GNLF | 87.5 | 15.0 | 0.189 s |
| LSO-Levenberg-QR | 87.5 | 13.0 | 0.127 s |
| LsqFit-LM | 84.1 | not exposed | 1.470 s |

On these small, microsecond-scale problems, per-iteration overhead dominates and the lightest wrappers win the left side of the profile: NLLSsolver solves its 97.7% roughly eight times faster in total than TRLS solves its 100%. What the extra per-iteration cost (exact subproblem solves, careful factorizations) buys is the last few problems, at a median of 12.5 iterations.

#### Real-world benchmark: expensive evaluations

To simulate a realistic model — where each Jacobian evaluation means re-solving an ODE/PDE or running a simulation — the second benchmark injects a 200 ms delay into every Jacobian *and* gradient evaluation (the gradient J′r requires the Jacobian, so gradient-based solvers must pay it too).

![NLLS solver performance with expensive Jacobians](docs/src/assets/benchmarks/nlls_solver_performance_delay.png)
*Figure 2: Same 88 problems with 200 ms per Jacobian and gradient evaluation; colors and markers as in Figure 1. Evaluation count now dominates wall-clock, and the few-iteration LM methods cluster at the front.*

| Solver | Success % | Median iterations | Median time | Cumulative time | vs SciPy |
|---|---|---|---|---|---|
| **TRLS** | **100.0** | 12.5 | 2.34 s | 425.8 s | **1.16×** |
| Scipy-LeastSquares | **100.0** | 13.0 | 2.66 s | 495.7 s | 1.00× |
| NLLSsolver-LM | 97.7 | 13.5 | 2.75 s | 444.3 s | 1.12× |
| NonlinearSolve-TR | 94.3 | 33.0 | 3.05 s | 566.0 s | 0.88× |
| NonlinearSolve-LM | 94.3 | 39.0 | 5.08 s | 780.2 s | 0.64× |
| Optim-L-BFGS | 90.9 | 33.0 | 8.43 s | 2567.6 s | 0.19× |
| Optim-BFGS | 89.8 | 32.0 | 9.15 s | 2330.1 s | 0.21× |
| LSO-Levenberg-QR | 87.5 | 13.0 | 2.64 s | 419.5 s | 1.18× † |
| LsqFit-LM | 84.1 | not exposed | 2.95 s | 518.2 s | 0.96× † |

† computed over that solver's own smaller successful set, so not comparable with the 100% rows.

This is the regime the solver is designed for: **fewer steps beat cheaper steps** once the model is
expensive. The ordering inverts relative to Figure 1 — the lightest wrappers no longer win, because
wrapper overhead is now invisible next to a 200 ms evaluation, and what is left is how many
evaluations each method needs. TRLS has both the lowest median time and the lowest cumulative time of
any solver at 100% success.

The line-search quasi-Newton methods pay the most, at roughly a fifth of the baseline's throughput:
their line searches evaluate the gradient several times per iteration, and every one of those costs
200 ms. That is the honest shape of the trade, and it is why those two rows carry a 15-minute
wall-clock cap here (see Benchmark limitations). **The cap bound on two cells of 792**, both `tp297`,
where BFGS and L-BFGS were still iterating at a cost of about −1e-14 — machine zero — and were
recorded as not converged at 910.8 s and 903.5 s. Both still count as successes because the cost
matches the best found, and **the Optim success rates are identical to the uncapped August run**
(89.8% and 90.9%), so the cap changed no outcome.

One dependency change worth recording: `tp297` with Optim-L-BFGS took 55.7 s in August under Optim
2.2.1 and exceeded 900 s here under 2.3.1, at 267 iterations against 189. The same cell for BFGS went
from 577 s to over 900 s. Nothing in this package touches that path, so it is an upstream change; it
is also why the cap earns its place now when it would not have been needed before.

#### Underdetermined problems

Every problem in the unconstrained set, with residual rows cropped so there are fewer equations than
unknowns (`crop_nls_functions`). Each cropped problem has a zero-residual solution, generically a
whole manifold of them, so the interesting question is not only whether a solver lands on the
manifold but how far it travels to get there: a minimum-norm method stays near `x0`.

| Solver | Success % | Median iterations | Cumulative time | Median ‖x*-x0‖ |
|---|---|---|---|---|
| LM-QR-scaled | **100.0** | 4.0 | 0.022 s | 2.200 |
| **TRLS** | **100.0** | 4.0 | 0.052 s | 2.193 |
| NonlinearSolve-TR | **100.0** | 6.0 | 0.032 s | 2.190 |
| Scipy-LeastSquares | **100.0** | 16.0 | 0.833 s | 2.200 |
| NonlinearSolve-GNBK | 98.9 | 5.0 | 0.048 s | 2.200 |
| Optim-BFGS | 98.9 | 13.0 | 0.076 s | 2.200 |
| Optim-L-BFGS | 97.7 | 15.5 | 0.032 s | 2.200 |
| NonlinearSolve-GNLF | 96.6 | 5.0 | 0.021 s | 2.200 |
| NLLSsolver-LM | 96.6 | 6.0 | 0.019 s | 2.190 |
| NonlinearSolve-LM | 95.5 | 9.0 | 0.022 s | 2.035 |
| LSO-Levenberg-QR | 86.4 | 6.5 | 0.038 s | 2.200 |
| LsqFit-LM | 83.0 | not exposed | 0.027 s | 2.200 |

**A note on the threshold, because it changes the ranking completely.** This suite used to score
success at `1e-12` absolute on the cost, which is four orders of magnitude tighter than the `1e-8`
every solver is configured with. At `1e-12` TRLS scored 59%, SciPy 91% and Optim-BFGS 86% — all of
them stopping where they were told to while others happened to keep polishing. A success threshold
tighter than the configured stopping tolerance measures the threshold, not the solver, so it is now
`1e-8`, matching the rest of the suite. Both sets of numbers are in `benchmark/AUDIT.md`.

The strategy that is actually designed for this case is the LQ family, which returns the
minimum-norm step through a complete orthogonal decomposition. `internal_variants.jl` scores it
separately: `LM-LQ` reaches 100% with the highest min-norm rate (0.97 of problems within 1e-3 of the
smallest `‖x*-x0‖` any variant found), and `JacobianScaling` measurably hurts there, dropping the
rate to 0.80 — scaling pulls the step away from the minimum-norm direction.

#### Hard exponential fits, where this solver does not win

Six deliberately nasty exponential-fitting problems from Lukšan [[Luk96]](#references), run both in
their original parameterization and with `x = exp(y)`. Parameters span orders of magnitude: A.3
starts at `[0.02, 4000, 250]` and A.5 at `[1e5, 1e5, 1.08, 1.31]`.

| Variant | TRLS (no scaling) | TRLS + `JacobianScaling` | Best in field |
|---|---|---|---|
| Original | 2 / 6 | **4 / 6** | 4 / 6 (also NonlinearSolve-LM, NonlinearSolve-PolyAlg, LSO-DogLeg-QR) |
| `x = exp(y)` | 4 / 6 | 4 / 6 | **6 / 6** (NonlinearSolve-PolyAlg) |

Two things this set is kept for, neither of them flattering:

- **Scaling is not optional here.** In the original parameterization, turning on `JacobianScaling`
  takes this solver from 2 of 6 to 4 of 6. Column-norm scaling is the whole difference between
  failing most of the set and matching the best anyone manages, which is the clearest argument for
  that option existing. On A.4 the two scaled variants are the only solvers in a field of 22 that
  reach the best cost at all.
- **A polyalgorithm beats a single method.** Under the log reparameterization
  NonlinearSolve's `FastShortcutNLLSPolyalg` solves all six while this solver solves four. Switching
  strategies when one stalls is something this package does not do, and on problems like these it
  wins.

The regression guard in `hard_luksan.jl` is therefore on the scaled variant, not the default one:
asserting that the unscaled configuration keeps working on problems that need scaling would be
asserting the wrong thing.

#### Bound-constrained problems

![Bounded solver performance](docs/src/assets/benchmarks/bounded_solver_performance.png)
*Figure 3: Performance profile on 71 bound-constrained problems: CUTEst NLS problems (whose
residuals are encoded as constraints), the bound-constrained NLSProblems set, and 15 hand-written
`ADNLSModel` problems with a mix of active and inactive bounds. TRLS leads on success rate and is an
order of magnitude ahead of SciPy on cumulative time.*

| Solver | Success % | Median iterations | Cumulative time |
|---|---|---|---|
| **TRLS** | **84.5** | 18.0 | 0.081 s |
| LM-QR-scaled | 83.1 | 16.0 | 0.067 s |
| LM-QRChol | 83.1 | 18.0 | 0.048 s |
| Scipy-LeastSquares | 81.7 | 18.0 | 0.853 s |
| Scipy-LSMR | 69.0 | 17.0 | 3.267 s |
| LsqFit-LM | 64.8 | not exposed | 0.318 s |
| LSO-Levenberg-QR | 63.4 | 35.0 | 0.041 s |
| PRIMA-BOBYQA | 62.0 | 204 (nf) | 2.770 s |
| JSO-TRON | 29.6 | 21.0 | 0.477 s |
| NonlinearSolve-PolyAlg | 16.9 | 1449.5 | 0.743 s |
| NonlinearSolve-TrustRegion | 8.5 | 33.5 | 0.0004 s |
| NonlinearSolve-GaussNewton | 7.0 | 450.0 | 0.006 s |
| NonlinearSolve-LevenbergMarquardt | 1.4 | 450.0 | 0.002 s |

Three things to read carefully before citing the low rows:

- **No solver violated its bounds** on any problem, so none of these rates is a feasibility penalty.
- **JSO-TRON's 29.6% measures harness compatibility, not the algorithm.** TRON consumes the
  `NLPModels` model directly rather than the extracted residual/Jacobian closures the others get, and
  it refuses any model carrying general constraints: it declines 47 of the 71 problems outright with
  *"tron should only be called for unconstrained or bound-constrained problems"*, because the CUTEst
  problems encode their residuals as constraints. On the 24 it accepts it is competitive. Fixing this
  means building a genuinely bound-constrained NLS model out of the CUTEst encoding, which the
  harness does not yet do.
- **NonlinearSolve's low rates are not crashes.** It ran every problem and returned; its median
  iteration counts sit at or near the 450-iteration budget, so it is failing to converge within the
  budget on these problems rather than erroring. Its bound-constrained path is much newer than its
  unconstrained one.

The two internal variants in the table (`LM-QR-scaled`, `LM-QRChol`) are there to show that the
choice of factorization strategy barely matters on this set: all three sit within 1.4 points of each
other. `internal_variants.jl` compares all eight strategy-and-scaling combinations properly.

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
