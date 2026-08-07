# Cross-package comparison on unconstrained NLSProblems.jl problems.
# Run with the benchmark environment:
#   julia --project=benchmark benchmark/scripts/compare_unconstrained.jl
# Knobs (env vars):
#   MAX_VARS      - only include problems with at most this many variables (default 999)
#   PROBLEM_LIMIT - cap the number of problems for a quick run (default: no cap)
include(joinpath(@__DIR__, "..", "harness.jl"))
using NLPModels
using JSOSolvers
using PRIMA
using NonlinearSolve
using Revise
using DataFrames
using nonlinearlstr
using LsqFit
const MAX_VARS = parse(Int, get(ENV, "MAX_VARS", "999"))
const PROBLEM_LIMIT = parse(Int, get(ENV, "PROBLEM_LIMIT", "0"))  # 0 = no cap
let probs = find_nlls_problems(MAX_VARS)
    global nls_problems = PROBLEM_LIMIT > 0 ? probs[1:min(PROBLEM_LIMIT, end)] : probs
end

solvers = [
    # nonlinearlstr (LM-QR, the method this poster presents)
    ("This work", nonlinearlstr.lm_trust_region!),

    # PRIMA (Best: NEWUOA for unconstrained)
    ("PRIMA-NEWUOA", nothing),

    # NonlinearSolve.jl (short labels for the poster legend; dispatch matches the
    # "NonlinearSolve-" prefix)
    ("NonlinearSolve-TR", NonlinearSolve.TrustRegion),
    ("NonlinearSolve-LM", NonlinearSolve.LevenbergMarquardt),

    # JSOSolvers (Best: TRON)
    ("JSO-TRON", tron),

    # LeastSquaresOptim (Best: Levenberg-QR)
    ("LSO-Levenberg-QR", LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.QR())),

    # Scipy (Best: LeastSquares)
    ("Scipy-LeastSquares", nothing),

    # NLLSsolver (Best: LevenbergMarquardt)
    ("NLLSsolver-LM", NLLSsolver.levenbergmarquardt),

    #LsqFit: lets see how it goes
    ("LsqFit-LM", nothing),

    # Quasi-Newton baselines (Optim.jl)
    ("Optim-BFGS", nothing),
    ("Optim-L-BFGS", nothing),
]

# Run benchmark
nls_results = nlls_benchmark(nls_problems, solvers, max_iter = 400)

# Convert to DataFrame and persist raw results so figures can be rebuilt without
# rerunning the benchmark (see scripts/plot_results.jl). x_opt (a vector per row)
# is dropped — CSV would stringify it and no downstream step needs it.
df_nls = DataFrame(nls_results)
results_dir = normpath(joinpath(@__DIR__, "..", "results"))
mkpath(results_dir)
CSV.write(joinpath(results_dir, "nlls_results.csv"), select(df_nls, Not(:x_opt)))

include(joinpath(@__DIR__, "..", "evaluate.jl"))

df_nls_proc = compare_with_best(df_nls)
summary_nls = evaluate_solvers(df_nls_proc)
display(summary_nls)

using Test
@testset "Solver Performance Tests" begin
    # Check that our solvers perform reasonably well (success rate > 90% relative to best)
    # Note: These thresholds might need adjustment based on the specific problem set difficulty
    if !isempty(summary_nls)
        row = summary_nls[summary_nls.solver .== "This work", :]
        if !isempty(row)
            @test row[1, :percentage_success] > 0.9
        end
    end
end

# Figures are built separately from the saved CSV:
#   julia --project=benchmark benchmark/scripts/plot_results.jl
