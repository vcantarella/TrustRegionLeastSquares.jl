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
| 6 | Single-step `x_tol` criteria fire on slow crawl without optimality evidence; "This work" has no such test. | Disabled where single-step: SciPy `xtol = None`, LSO/LsqFit `x_tol = 0`. Kept where structural: NonlinearSolve stall (32 *consecutive* steps; only exit at nonzero residual), PRIMA `rhoend` (DFO resolution parameter). |

### Deferred (bounded script only — fix before `compare_bounded.jl` ships)

- `dispatch.jl` Scipy-LSMR: the `@be` timing call omits `tr_solver = "lsmr"` — it times
  the default TRF configuration against LSMR's solution.
- `dispatch.jl` NonlinearSolve bounds heuristic `any(lb .> -1e-30) || any(ub .< 1e30)`
  silently drops bounds for e.g. `lb = -5, ub = Inf`; should test `isfinite`. Compounded
  by `is_success` not checking `bounds_satisfied`, an unconstrained interloper can set an
  unreachable `min_solution` for the whole problem.
- `run.jl` stores `nvars = prob_data.n` (residual count) and `nresiduals = prob_data.m`
  (variable count) — swapped in the CSVs (not plotted anywhere).

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
- **Empirically confirmed** by `benchmark/scripts/mwe_nonlinearsolve_stall.jl`
  (self-contained; deps NLSProblems + NLPModels + NonlinearSolve): 62 LM/TR solves on
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
| This work (`lm_trust_region!`) | `gtol = 1e-8, ftol = 1e-8, verbose = false` | radius collapse `< 1e-8` |
| NonlinearSolve TR/LM | `abstol = 1e-8, maxiters = 400` | SafeBest stall exit (32 steps ≤ abstol) |
| JSO-TRON | `atol = 1e-8, rtol = 0, max_time = Inf` | — |
| LSO-Levenberg-QR | `iterations = 400, g_tol = 1e-8, x_tol = 0` | `f_tol = 1e-8` default |
| SciPy `least_squares` (TRF) | `gtol = 1e-8, xtol = None, max_nfev = 1000` | `ftol = 1e-8` default |
| NLLSsolver-LM | `reldcost = 1e-8, maxiters = 400, maxtime = 1e6` | `absdcost/dstep = 1e-15` (off) |
| Optim BFGS/L-BFGS | `g_tol = 1e-8, iterations = 400` | x/f tolerances off (Optim default) |
| LsqFit-LM | `g_tol = 1e-8, x_tol = 0, maxIter = 400` | — |
| PRIMA-NEWUOA | `rhoend = 1e-8, maxfun = 400·n_vars` | — (derivative-free reference) |

## Results — uniform-1e-8 configuration

Main figure (fresh, 2026-08-08):

| Solver | Success % | Median iters (succ) | Total time (succ) |
|---|---|---|---|
| This work | **100.0** | 12.5 | 0.495 s |
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

Headline: **This work 100% at 2.7× SciPy**; enabling `ftol = 1e-8` cut its median
iterations 14.5 → 12.5. Every fairness fix moved a *competitor* up (TRON +4.6 pts,
NonlinearSolve-LM median 72 → 39, PRIMA +1.1, LsqFit +2.3) and the headline survived —
cite this when questioned. Delay-figure rerun at these settings in progress; expect the
few-iteration LM cluster (This work / SciPy / NLLSsolver ≈ 1×) ahead of
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