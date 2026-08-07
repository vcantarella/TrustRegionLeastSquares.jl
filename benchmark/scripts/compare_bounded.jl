include(joinpath(@__DIR__, "..", "harness.jl"))
include(joinpath(@__DIR__, "..", "problems", "custom_bounded.jl"))
using NLPModels
using ADNLPModels
using JSOSolvers
using PRIMA
using NonlinearSolve
using Revise
using DataFrames
using nonlinearlstr
using Test

# 1. Find and prepare problems from three sources:
#    - NLSProblems.jl bound-constrained NLS (Hock-Schittkowski / Schittkowski hs*, tp*)
#    - CUTEst bound-constrained NLS (objtype="none" + finite variable bounds); residuals are
#      the constraints. Guarded so the script still runs if CUTEst isn't installed.
#    - curated ADNLSModel set (MGH analytic + NIST StRD, mix of active/realistic bounds)
nls_problems = find_bounded_problems(100)
try
    append!(nls_problems, find_cutest_bounded_nlls_problems(50))
catch e
    @warn "Skipping CUTEst bounded problems (CUTEst unavailable?)" exception =
        (e, catch_backtrace())
end
custom_problems = nameof.(get_custom_problems())   # all 15 ADNLSModel constructors
append!(nls_problems, custom_problems)
# 2. Define solvers
solvers = [
    ("TRF", nonlinearlstr.lm_trust_region_reflective),
    ("JSO-TRON", tron),
    ("Scipy-LeastSquares", nothing),  # Special handling in test_solver_on_problem
    ("Scipy-LSMR", nothing),  # Special handling in test_solver_on_problem
    ("PRIMA-BOBYQA", nothing),  # Special handling in test_solver_on_problem
    ("LSO-Levenberg-QR", LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.QR())),
    ("NonlinearSolve-TrustRegion", NonlinearSolve.TrustRegion),
    ("NonlinearSolve-LevenbergMarquardt", NonlinearSolve.LevenbergMarquardt),
    ("NonlinearSolve-GaussNewton", NonlinearSolve.GaussNewton),
    ("NonlinearSolve-PolyAlg", NonlinearSolve.FastShortcutNLLSPolyalg),
]

# 3. Run benchmark
println("Running benchmark on $(length(nls_problems)) problems...")
nls_results = nlls_benchmark(nls_problems, solvers, max_iter = 450)

# 4. Analyze results
df_nls = DataFrame(nls_results)

include(joinpath(@__DIR__, "..", "evaluate.jl"))

df_nls_proc = compare_with_best(df_nls)
summary_nls = evaluate_solvers(df_nls_proc)

println("\n\n" * "="^60)
println("SOLVER PERFORMANCE SUMMARY")
println("="^60)
display(summary_nls)

# 5. Tests
@testset "Bounded Solver Performance" begin
    # Check that TRF has a reasonable success rate (e.g., > 80% relative to best)
    # Note: Success definition in compare_with_best is being close to the best found solution

    trf_success = summary_nls[summary_nls.solver .== "TRF", :percentage_success]
    if !isempty(trf_success)
        @test trf_success[1] > 0.7
    end
end

# 6. Detailed Failure Analysis (Optional, printed to console)
println("\n" * "="^60)
println("DETAILED FAILURE ANALYSIS")
println("="^60)

for row in eachrow(summary_nls)
    if row.percentage_success < 1.0
        println(
            "\nSolver '$(row.solver)' success rate: $(round(row.percentage_success * 100, digits=1))%",
        )

        failed = df_nls_proc[
            (df_nls_proc.solver .== row.solver) .& (.!df_nls_proc.is_success),
            :,
        ]
        if nrow(failed) > 0
            sort!(failed, :gap_to_best, rev = true)
            for r in eachrow(first(failed, 5)) # Show top 5 failures
                println("  Problem: $(r.problem)")
                println("    Obj: $(r.final_cost) vs Best: $(r.min_solution)")
                println("    Gap: $(r.gap_to_best)")
            end
        end
    end
end

# 7. Plots
fig_nls = build_performance_plots(df_nls_proc)
bounded_plot_path = joinpath(plots_dir(), "bounded_solver_performance.png")
save(bounded_plot_path, fig_nls)
println("\nPerformance plot saved to '$bounded_plot_path'")
