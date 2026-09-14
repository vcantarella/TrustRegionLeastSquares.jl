# Cross-package comparison on UNDERDETERMINED problems: the unconstrained NLSProblems.jl
# suite with residual rows cropped so every problem has fewer equations than unknowns
# (see crop_nls_functions in problems/wrappers.jl). Run with the benchmark environment:
#   julia --project=benchmark benchmark/scripts/compare_underdetermined.jl
# Knobs (env vars):
#   CROP_RATIO    - keep ceil(ratio * nvar) residuals, capped at nvar - 1 (default 0.5)
#   MAX_VARS      - only include problems with at most this many variables (default 999)
#   PROBLEM_LIMIT - cap the number of problems for a quick run (default: no cap)
# Every cropped problem has a known optimum of 0 (generically a whole manifold of
# zero-residual points), so success is a tight ABSOLUTE cost tolerance rather than the
# overdetermined suite's 1e-4, which would accept stalls at ‖r‖ ≈ 0.01. JSO-TRON is
# omitted: it consumes the NLSModel directly, not the cropped closures.
include(joinpath(@__DIR__, "..", "harness.jl"))
using NonlinearSolve, DataFrames, LsqFit
import TrustRegionLeastSquares as TRLS
const CROP_RATIO = parse(Float64, get(ENV, "CROP_RATIO", "0.5"))
const MAX_VARS = parse(Int, get(ENV, "MAX_VARS", "999"))
const PROBLEM_LIMIT = parse(Int, get(ENV, "PROBLEM_LIMIT", "0"))  # 0 = no cap
# Success threshold on the cost. It must not be tighter than the tolerance every solver is
# configured with (dispatch.jl sets every exposed tolerance to 1e-8), or the metric measures the
# threshold rather than the solver: a solver that stops at cost 1e-9 because it was told to stop at
# 1e-8 is obeying instructions. At the previous 1e-12 this suite scored TRLS at 59%, SciPy at 91%
# and Optim-BFGS at 86%, all of them stopping where configured while the others happened to keep
# polishing; at 1e-8 TRLS, NonlinearSolve-TR and SciPy all reach 100% and the comparison falls to
# iterations, time and the min-norm distance, which is what this suite is for. 1e-8 on the cost is
# ‖r‖ ≲ 1.4e-4 on a problem whose optimum is exactly 0.
const COST_ATOL = 1e-8
let probs = find_nlls_problems(MAX_VARS)
    global nls_problems = PROBLEM_LIMIT > 0 ? probs[1:min(PROBLEM_LIMIT, end)] : probs
end

# pd.m is the variable count, pd.n the residual count.
crop(pd) = crop_nls_functions(pd, clamp(ceil(Int, CROP_RATIO * pd.m), 1, pd.m - 1))

solvers = [
    ("TRLS", TRLS.lm_trust_region!),
    ("LM-QR-scaled", TRLS.lm_trust_region!),
    ("NonlinearSolve-TR", NonlinearSolve.TrustRegion),
    ("NonlinearSolve-LM", NonlinearSolve.LevenbergMarquardt),
    ("NonlinearSolve-GNBK", () -> NonlinearSolve.GaussNewton(linesearch = BackTracking())),
    (
        "NonlinearSolve-GNLF",
        () -> NonlinearSolve.GaussNewton(linesearch = LiFukushimaLineSearch()),
    ),
    ("LSO-Levenberg-QR", LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.QR())),
    ("Scipy-LeastSquares", nothing),
    ("NLLSsolver-LM", NLLSsolver.levenbergmarquardt),
    ("LsqFit-LM", nothing),
    ("Optim-BFGS", nothing),
    ("Optim-L-BFGS", nothing),
]

nls_results = nlls_benchmark(nls_problems, solvers; max_iter = 400, transform = crop)

df_nls = DataFrame(nls_results)
@assert all(df_nls.nresiduals .< df_nls.nvars) "every problem must be underdetermined"
results_dir = normpath(joinpath(@__DIR__, "..", "results"))
mkpath(results_dir)
CSV.write(
    joinpath(results_dir, "nlls_results_underdetermined.csv"),
    select(df_nls, Not(:x_opt)),
)

include(joinpath(@__DIR__, "..", "evaluate.jl"))
df_nls_proc = compare_with_best(df_nls; atol = COST_ATOL)
display(evaluate_solvers(df_nls_proc))

# Figures are built separately from the saved CSV:
#   julia --project=benchmark benchmark/scripts/plot_results.jl
