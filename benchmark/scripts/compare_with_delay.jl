include(joinpath(@__DIR__, "..", "harness.jl"))
using NLPModels
using JSOSolvers
using PRIMA
using NonlinearSolve
using Revise
using DataFrames
using nonlinearlstr
using LsqFit
nls_problems = find_nlls_problems(999)[1:min(20, end)]

solvers = [
    # nonlinearlstr solvers (keep all)
    ("LM-QR", nonlinearlstr.lm_trust_region!),
    ("LM-SVD", nonlinearlstr.lm_trust_region!),
    ("LM-QR-Recursive", nonlinearlstr.lm_trust_region!),

    # NonlinearSolve.jl (keep all)
    ("NonlinearSolve-TrustRegion", NonlinearSolve.TrustRegion),
    ("NonlinearSolve-LevenbergMarquardt", NonlinearSolve.LevenbergMarquardt),
    ("NonlinearSolve-GaussNewton", NonlinearSolve.GaussNewton),
    ("NonlinearSolve-PolyAlg", NonlinearSolve.FastShortcutNLLSPolyalg),

    # LeastSquaresOptim (Best: Levenberg-QR)
    ("LSO-Levenberg-QR", LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.QR())),

    # Scipy (Best: LeastSquares)
    ("Scipy-LeastSquares", nothing),

    #LsqFit: lets see how it goes
    ("LsqFit-LM", nothing),
]

# We need a custom benchmark loop to inject the delay into the jacobian functions
function nlls_benchmark_with_delay(problems, solvers; max_iter = 100)
    results = []
    max_problems = length(problems)
    for (i, prob_name) in enumerate(problems)
        println("\n" * "="^60)
        println("Problem $i/$max_problems: $prob_name")

        # Create problem instance
        local nlp
        local prob_data
        if isa(prob_name, String)
            nlp = CUTEstModel(prob_name)
            prob_data = create_cutest_functions(nlp)
        else
            nlp = eval(prob_name)()
            prob_data = create_nls_functions(nlp)
        end

        # Inject 1-second delay into Jacobian functions
        orig_jac = prob_data.jacobian_func
        prob_data = merge(prob_data, (jacobian_func = x -> (sleep(1.0); orig_jac(x)),))

        if hasproperty(prob_data, :jacobian_func!)
            orig_jac! = prob_data.jacobian_func!
            prob_data = merge(
                prob_data,
                (jacobian_func! = (J, x) -> (sleep(1.0); orig_jac!(J, x)),),
            )
        end

        println("  Variables: $(prob_data.n)")
        println("  Residuals: $(prob_data.m)")
        initial_obj = prob_data.obj_func(prob_data.x0)

        # Test each solver
        problem_results = []
        for (solver_name, solver_func) in solvers
            print("    Testing $solver_name... ")
            result =
                test_solver_on_problem(solver_name, solver_func, prob_data, nlp, max_iter)
            if result.success && result.converged
                println(
                    "✓ obj=$(round(result.final_cost, digits=8)), iters=$(result.iterations)",
                )
            else
                status = result.success ? "no convergence" : "failed"
                println("✗ $status")
            end
            result_with_problem = merge(
                result,
                (
                    problem = String(prob_name),
                    nvars = prob_data.n,
                    nresiduals = prob_data.m,
                    initial_objective = initial_obj,
                    improvement = initial_obj - result.final_cost,
                ),
            )
            push!(problem_results, result_with_problem)
        end
        append!(results, problem_results)
        finalize(nlp)
    end
    return results
end

# Run benchmark
nls_results = nlls_benchmark_with_delay(nls_problems, solvers, max_iter = 400)

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
