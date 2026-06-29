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
    # nonlinearlstr solvers (keep all)
    ("LM-QR", nonlinearlstr.lm_trust_region!),
    ("LM-SVD", nonlinearlstr.lm_trust_region!),
    ("LM-QR-Recursive", nonlinearlstr.lm_trust_region!),

    # PRIMA (Best: NEWUOA for unconstrained)
    ("PRIMA-NEWUOA", nothing),

    # NonlinearSolve.jl (keep all)
    ("NonlinearSolve-TrustRegion", NonlinearSolve.TrustRegion),
    ("NonlinearSolve-LevenbergMarquardt", NonlinearSolve.LevenbergMarquardt),
    ("NonlinearSolve-GaussNewton", NonlinearSolve.GaussNewton),
    ("NonlinearSolve-PolyAlg", NonlinearSolve.FastShortcutNLLSPolyalg),

    # JSOSolvers (Best: TRON)
    ("JSO-TRON", tron),

    # LeastSquaresOptim (Best: Levenberg-QR)
    ("LSO-Levenberg-QR", LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.QR())),

    # Scipy (Best: LeastSquares)
    ("Scipy-LeastSquares", nothing),

    # NLLSsolver (Best: LevenbergMarquardt)
    ("NLLSsolver-levenbergmarquardt", NLLSsolver.levenbergmarquardt),

    #LsqFit: lets see how it goes
    ("LsqFit-LM", nothing),
]

# Run benchmark
nls_results = nlls_benchmark(nls_problems, solvers, max_iter = 400)

# Convert to DataFrame
df_nls = DataFrame(nls_results)

include(joinpath(@__DIR__, "..", "evaluate.jl"))

df_nls_proc = compare_with_best(df_nls)
summary_nls = evaluate_solvers(df_nls_proc)
display(summary_nls)

using Test
@testset "Solver Performance Tests" begin
    # Check that our solvers perform reasonably well (success rate > 90% relative to best)
    # Note: These thresholds might need adjustment based on the specific problem set difficulty
    if !isempty(summary_nls)
        qr_row = summary_nls[summary_nls.solver .== "LM-QR", :]
        svd_row = summary_nls[summary_nls.solver .== "LM-SVD", :]

        if !isempty(qr_row)
            @test qr_row[1, :percentage_success] > 0.9
        end
        if !isempty(svd_row)
            @test svd_row[1, :percentage_success] > 0.9
        end
    end
end

fig_nls = build_performance_plots(df_nls_proc)
save(joinpath(plots_dir(), "nlls_solver_performance.png"), fig_nls)
