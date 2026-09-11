# Benchmark fairness audit — working report

*2026-08-08, pre-JuliaCon. Companion to the README's "Termination criteria" and
"Known limitations" sections; this is the working record of what was checked, what
was changed, and why.*

## Scope

Cross-solver fairness of the two poster benchmarks (`compare_unconstrained.jl`,
`compare_with_delay.jl`): iteration budgets, termination tolerances, hidden caps,
timing methodology, and success determination. All competitor defaults verified
against the **installed package sources** (versions from `benchmark/Manifest.toml`),
not against documentation.

## Problem set

- **88 unconstrained NLS problems** (NLSProblems.jl, ≤ 999 variables) in both figures.
- Main figure: **11 solvers** → 968 runs. Delay figure: **9 solvers** → 792 runs
  (NLLSsolver-LM added 2026-08-08; PRIMA, TRON, NLLSsolver were previously absent —
  TRON stays out because it consumes the NLPModels model directly and the Jacobian
  delay cannot be injected).
- 58/88 problems are zero-residual at the solution (`best cost ≤ 5e-17`); **30/88 have
  genuinely nonzero residual** — these are the problems where residual-norm stopping
  criteria (NonlinearSolve) can never fire and stall/step exits matter.

## Findings and resolutions

| # | Finding | Resolution |
|---|---|---|
| 1 | NonlinearSolve ran with default `abstol` = 3e-13 on `‖r‖₂` — unreachable at nonzero residual; only exits were a 32-step stall window or `maxiters`. Median 72 iterations vs 14.5 for comparable LM solvers; ~6.4 s/problem pure penalty in the delay figure. | Explicit `abstol = 1e-8`, keeping its NLLS-default `AbsNormSafeBestTerminationMode` (the stall exit → `StalledSuccess` is its "found a local minimum" answer and its only exit on the 30 nonzero-residual problems). Median dropped to 39. |
| 2 | LeastSquaresOptim received no kwargs → private defaults: **1000 iterations** (everyone else 400) and its own tolerances. | `iterations = max_iter, g_tol = 1e-8, x_tol = 0` passed explicitly. |
| 3 | Hidden 30 s wall-clock caps: TRON `max_time = 30.0`, NLLSsolver `maxtime = 30.0` — no other solver had one. TRON also stopped at `atol + rtol·‖g₀‖` with `atol = rtol = √eps` (per-problem-varying). | TRON: `atol = 1e-8, rtol = 0, max_time = Inf`. NLLSsolver: `maxtime = 1e6` (stored as `UInt64` ns; `Inf` throws). TRON success rose 88.6% → 93.2%. |
| 4 | `lm_trust_region!` printed to stdout every iteration **inside the timed region**. | `verbose::Bool = false` kwarg guards all prints (`src/algorithms.jl`). |
| 5 | Tolerances were mixed per class (gradient 1e-6, others 1e-8). | **Uniform 1e-8 on every exposed tolerance** (decision below). |
| 6 | Single-step `x_tol` criteria fire on slow crawl without optimality evidence; "TRLS" has no such test. | Disabled where single-step: SciPy `xtol = None`, LSO/LsqFit `x_tol = 0`. Kept where structural: NonlinearSolve stall (32 *consecutive* steps; only exit at nonzero residual), PRIMA `rhoend` (DFO resolution parameter). |

### Resolved with the bounded solver (2026-09-11)

- `dispatch.jl` Scipy-LSMR: the `@be` timing call omitted `tr_solver = "lsmr"`, timing the
  default TRF configuration against LSMR's solution. Added.
- `dispatch.jl` NonlinearSolve bounds heuristic `any(lb .> -1e-30) || any(ub .< 1e30)`
  silently dropped bounds for e.g. `lb = -5, ub = Inf`; now `any(isfinite, lb) || any(isfinite, ub)`.
  The compounding issue — an unconstrained interloper setting an unreachable `min_solution` for the
  whole problem — is closed in `evaluate.jl`: `compare_with_best` now maps the cost of any solve
  that violates its bounds to `Inf` before taking the per-problem minimum.
- `dispatch.jl` strategy dispatch tested `contains(solver_name, "LQ")` before `"LQChol"`, so every
  `LM-LQChol` row was in fact produced by `LQStrategy`. The chain now tests `LQChol` first, which
  means LQChol columns in results produced before this date are mislabelled LQ columns.
- `run.jl` stored `nvars`/`nresiduals` swapped. Correct in `run.jl`; `hard_luksan.jl` carried the
  same swap and is fixed too.

## The termination-criteria decision

Final scheme: **every tolerance a solver exposes is set to 1e-8**; each solver stops
via its native criteria. Single-step `x_tol` tests disabled (see #6). Rationale and the
full criterion-class table are in the README ("Termination criteria: one value, each
solver's own criteria"). Key facts established along the way, all verified in installed
sources:

- **NonlinearSolve norms the residual, not the step.** The termination functor's
  parameter is *named* `du`, but every NonlinearSolve solver binds `cache.fu` to it
  (`NonlinearSolveFirstOrder/src/solve.jl:388` → `check_and_update!(cache, fu, u, uprev)`
  in `NonlinearSolveBase/src/termination_conditions.jl:325`), and `fu` is `f(u)` = the
  residual (`evaluate_f`, `utils.jl:185`; buffer sized from `resid_prototype`, which has
  residual dimension ≠ parameter dimension on overdetermined problems). The docs'
  "Δu denotes the increment" phrasing is wrong for this path → upstream docs issue.
- Consequently `NormTerminationMode` ("both tolerances") is residual-only in both
  branches and has **no stall exit** — it would strand the 30 nonzero-residual problems
  at `maxiters`. SafeBest + `abstol = 1e-8` is the correct NLLS configuration.
- **Empirically confirmed** by a since-removed scratch script (self-contained; deps
  NLSProblems + NLPModels + NonlinearSolve): 62 LM/TR solves on
  31 nonzero-residual problems → **0× `Success`**, 45× `StalledSuccess`, 17× `MaxIters`,
  with 48/62 at stationary points (`‖J'F‖ < 1e-4`); zero-residual problems get
  `Success` 105/114 times. Fully converged runs (e.g. mgh23-LM, `‖J'F‖ = 5.8e-8`)
  still burn all 400 iterations. The script also proves `reltol` inert: a generous
  `reltol = 1e-3` added to every solve changed 0/175 reproducible solves (one
  FP-bistable solve, mgh34/TR, flips between the two non-`Success` codes and is
  flagged by a base-repeat determinism control).
- Tolerances on different quantities are not equally strict at equal numbers (cost error
  ~ square of parameter error); uniform 1e-8 removes the pairing *choice*, and the
  cost-based success metric (1e-4 to best-found) is orders of magnitude looser than any
  criterion, so success rates are insensitive to the exact value.

## Final per-solver configuration (both scripts, `max_iter = 400`)

| Solver | Explicit settings | Native criteria left active (defaults) |
|---|---|---|
| TRLS (`lm_trust_region!`) | `gtol = 1e-8, ftol = 1e-8, verbose = false` | radius collapse `< 1e-8` |
| NonlinearSolve TR/LM | `abstol = 1e-8, maxiters = 400` | SafeBest stall exit (32 steps ≤ abstol) |
| JSO-TRON | `atol = 1e-8, rtol = 0, max_time = Inf` | — |
| LSO-Levenberg-QR | `iterations = 400, g_tol = 1e-8, x_tol = 0` | `f_tol = 1e-8` default |
| SciPy `least_squares` (TRF) | `gtol = 1e-8, xtol = None, max_nfev = 1000` | `ftol = 1e-8` default |
| NLLSsolver-LM | `reldcost = 1e-8, maxiters = 400, maxtime = 1e6` | `absdcost/dstep = 1e-15` (off) |
| Optim BFGS/L-BFGS | `g_tol = 1e-8, iterations = 400` | x/f tolerances off (Optim default) |
| LsqFit-LM | `g_tol = 1e-8, x_tol = 0, maxIter = 400` | — |
| PRIMA-NEWUOA | `rhoend = 1e-8, maxfun = 400·n_vars` | — (derivative-free reference) |

## Results — uniform-1e-8 configuration

### 2026-09-11 re-run

Every figure and number below was re-measured on 2026-09-11 against the finished solver. The table
from the 2026-08-08 run is kept underneath for comparison, but it describes code with four defects
that have since been fixed, so treat it as historical rather than as a baseline:

- The `QRStrategy` λ-update had the wrong sign, so the step overshot the trust-region boundary (9%
  past it on the case that exposed it, with a normal-equation residual of 0.25). `QRStrategy` is the
  variant the `TRLS` row runs, so this affected the headline figures directly.
- `ftol` compared the cost reduction against `max(cost, 1)`, making it an absolute floor once the
  cost fell below 1. On Powell badly scaled the solver stopped at a gradient of 1.18; it is now a
  relative test and reaches 1.2e-6.
- The radius update now follows MINPACK `lmder`, scaling off the step actually taken.
- The λ-iteration cap went from 6 to 10, which is what an ill-conditioned Jacobian needs to place
  the boundary.

Also: labels changed from "This work" to `TRLS`, and `LM-LQChol` rows produced before this date were
actually produced by `LQStrategy` (the dispatch chain tested `"LQ"` first), so pre-2026-09-11 LQChol
columns in any CSV are mislabelled LQ columns.

Main figure (88 unconstrained NLSProblems, Julia 1.12, macOS aarch64):

| Solver | Success % | Median iters (succ) | Total time (succ) |
|---|---|---|---|
| TRLS | **100.0** | 12.5 | 0.283 s |
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
| LsqFit-LM | 84.1 | n/a (not exposed) | 1.470 s |

Against the 2026-08-08 run: success rates and median iterations are unchanged (100.0 / 12.5), and
cumulative solve time fell from 0.495 s to 0.283 s, so the margin over SciPy went from 2.7x to 4.7x.
The factorization-buffer reuse in the λ-iteration is the likely cause; competitors moved by less than
a percentage point, as expected for untouched code.

Bounded figure (71 bound-constrained problems, first run of the finished bounded solver):

| Solver | Success % | Median iters (succ) | Total time (succ) |
|---|---|---|---|
| TRLS | **84.5** | 18.0 | 0.081 s |
| LM-QR-scaled | 83.1 | 16.0 | 0.067 s |
| LM-QRChol | 83.1 | 18.0 | 0.048 s |
| Scipy-LeastSquares | 81.7 | 18.0 | 0.853 s |
| Scipy-LSMR | 69.0 | 17.0 | 3.267 s |
| LsqFit-LM | 64.8 | n/a | 0.318 s |
| LSO-Levenberg-QR | 63.4 | 35.0 | 0.041 s |
| PRIMA-BOBYQA | 62.0 | 204 (nf) | 2.770 s |
| JSO-TRON | 29.6 | 21.0 | 0.477 s |
| NonlinearSolve-PolyAlg | 16.9 | 1449.5 | 0.743 s |
| NonlinearSolve-TrustRegion | 8.5 | 33.5 | 0.0004 s |
| NonlinearSolve-GaussNewton | 7.0 | 450.0 | 0.006 s |
| NonlinearSolve-LevenbergMarquardt | 1.4 | 450.0 | 0.002 s |

Two caveats, both recorded in the README caption:

- **No solver violated its bounds** on any of the 71 problems, so the `bounds_satisfied` scoring fix
  added on 2026-09-11 changed no result here. It remains the right scoring rule; it simply did not bind.
- **JSO-TRON's rate is harness-limited, not algorithmic.** It consumes the NLPModel directly and
  refuses any model with general constraints, declining 47 of 71 with *"tron should only be called for
  unconstrained or bound-constrained problems"* — the CUTEst problems encode residuals as constraints.
  **New deferred item:** build a genuinely bound-constrained NLS model from the CUTEst encoding so TRON
  is measured on the algorithm. Until then do not quote TRON's bounded rate as a capability.
- NonlinearSolve's low rates are non-convergence within the 450-iteration budget, not errors: zero
  failed runs, medians at or near the cap.

Underdetermined figure (88 problems, residual rows cropped). **Finding #7, new on 2026-09-11:** this
suite scored success at `atol = 1e-12` on the cost while `dispatch.jl` configures every exposed
tolerance to `1e-8`. A threshold four orders of magnitude tighter than the configured stopping
tolerance measures the threshold, not the solver — and it penalised several solvers for stopping
exactly where told. `COST_ATOL` is now `1e-8`; both scorings are below so nothing is hidden.

| Solver | Success @ 1e-12 (old) | Success @ 1e-8 (now) | Median iters | Total time | Median ‖x*-x0‖ |
|---|---|---|---|---|---|
| TRLS | 59.1 | **100.0** | 4.0 | 0.052 s | 2.193 |
| LM-QR-scaled | 56.8 | **100.0** | 4.0 | 0.022 s | 2.200 |
| NonlinearSolve-TR | 98.9 | **100.0** | 6.0 | 0.032 s | 2.190 |
| Scipy-LeastSquares | 90.9 | **100.0** | 16.0 | 0.833 s | 2.200 |
| NonlinearSolve-GNBK | 97.7 | 98.9 | 5.0 | 0.048 s | 2.200 |
| Optim-BFGS | 86.4 | 98.9 | 13.0 | 0.076 s | 2.200 |
| Optim-L-BFGS | 89.8 | 97.7 | 15.5 | 0.032 s | 2.200 |
| NonlinearSolve-GNLF | 96.6 | 96.6 | 5.0 | 0.021 s | 2.200 |
| NLLSsolver-LM | 96.6 | 96.6 | 6.0 | 0.019 s | 2.190 |
| NonlinearSolve-LM | 95.5 | 95.5 | 9.0 | 0.022 s | 2.035 |
| LSO-Levenberg-QR | 86.4 | 86.4 | 6.5 | 0.038 s | 2.200 |
| LsqFit-LM | 81.8 | 83.0 | n/a | 0.027 s | 2.200 |

The solvers that moved are exactly the ones with a cost-based stopping test; those that stop on the
residual norm or the gradient (NLLSsolver, NonlinearSolve-GNLF, LSO) are unchanged, which is the
signature of a threshold artifact rather than a capability difference. Cross-check: at `1e-4`, the
tolerance the overdetermined suite uses, the ranking is identical to `1e-8`.

Note that `TRLS` and `LM-QR-scaled` here are QR strategies, not the LQ family that actually computes
minimum-norm steps; `internal_variants.jl` scores those, where `LM-LQ` leads on min-norm rate at 0.97
against 0.80 for the scaled variants.

### Historical: 2026-08-08 run

| Solver | Success % | Median iters (succ) | Total time (succ) |
|---|---|---|---|
| TRLS | **100.0** | 12.5 | 0.495 s |
| Scipy-LeastSquares | **100.0** | 13.0 | 1.329 s |
| NLLSsolver-LM | 97.7 | 13.5 | 0.035 s |
| NonlinearSolve-LM | 94.3 | 39.0 | 0.256 s |
| JSO-TRON | 93.2 | 16.0 | 0.113 s |
| NonlinearSolve-TR | 90.9 | 32.5 | 0.100 s |
| Optim-L-BFGS | 90.9 | 33.0 | 0.117 s |
| Optim-BFGS | 89.8 | 33.0 | 0.083 s |
| PRIMA-NEWUOA | 87.5 | 277 (nf) | 90.2 s |
| LSO-Levenberg-QR | 87.5 | 13.0 | 0.120 s |
| LsqFit-LM | 84.1 | n/a (not exposed) | 0.071 s |

Headline: **TRLS 100% at 2.7× SciPy**; enabling `ftol = 1e-8` cut its median
iterations 14.5 → 12.5. Every fairness fix moved a *competitor* up (TRON +4.6 pts,
NonlinearSolve-LM median 72 → 39, PRIMA +1.1, LsqFit +2.3) and the headline survived —
cite this when questioned. Delay-figure rerun at these settings in progress; expect the
few-iteration LM cluster (TRLS / SciPy / NLLSsolver ≈ 1×) ahead of
NonlinearSolve-LM (stall tail) and BFGS/L-BFGS (line-search gradient evaluations).

## Reproduce

```bash
julia --project=benchmark benchmark/scripts/compare_unconstrained.jl
julia --project=benchmark benchmark/scripts/compare_with_delay.jl
julia --project=benchmark benchmark/scripts/plot_results.jl
```

Package versions: NonlinearSolve 4.20.1 / NonlinearSolveBase 2.31.3, JSOSolvers 0.14.8,
LeastSquaresOptim 0.8.10, LsqFit 0.16.1, NLLSsolver 4.0.8, Optim 2.2.1, PRIMA 0.2.4,
SciPy 1.18.0 (conda-forge, via PythonCall), Julia 1.12, Chairmarks 1.3.1.