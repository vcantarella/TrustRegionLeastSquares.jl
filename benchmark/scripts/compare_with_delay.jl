include(joinpath(@__DIR__, "..", "harness.jl"))
using NLPModels
using JSOSolvers
using PRIMA
using NonlinearSolve
using Revise
using DataFrames
import TrustRegionLeastSquares as TRLS
using LsqFit
# Full problem suite by default; same knobs as compare_unconstrained.jl for quick runs:
#   MAX_VARS      - only include problems with at most this many variables (default 999)
#   PROBLEM_LIMIT - cap the number of problems (default: no cap)
const MAX_VARS = parse(Int, get(ENV, "MAX_VARS", "999"))
const PROBLEM_LIMIT = parse(Int, get(ENV, "PROBLEM_LIMIT", "0"))  # 0 = no cap
let probs = find_nlls_problems(MAX_VARS)
    global nls_problems = PROBLEM_LIMIT > 0 ? probs[1:min(PROBLEM_LIMIT, end)] : probs
end

solvers = [
    # TrustRegionLeastSquares (LM-QR, the method this poster presents)
    ("This work", TRLS.lm_trust_region!),

    # NonlinearSolve.jl (short labels for the poster legend; dispatch matches the
    # "NonlinearSolve-" prefix)
    ("NonlinearSolve-TR", NonlinearSolve.TrustRegion),
    ("NonlinearSolve-LM", NonlinearSolve.LevenbergMarquardt),

    # LeastSquaresOptim (Best: Levenberg-QR)
    ("LSO-Levenberg-QR", LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.QR())),

    # Scipy (Best: LeastSquares)
    ("Scipy-LeastSquares", nothing),

    # NLLSsolver (Best: LevenbergMarquardt). The delay reaches it through the
    # ProbDataResidual wrapper, whose computeresjacstatic calls the (delayed)
    # prob_data.jacobian_func; its LM inner retries only re-evaluate the cheap
    # residual, consistent with the other LM solvers.
    ("NLLSsolver-LM", NLLSsolver.levenbergmarquardt),

    #LsqFit: lets see how it goes
    ("LsqFit-LM", nothing),

    # Quasi-Newton baselines (Optim.jl)
    ("Optim-BFGS", nothing),
    ("Optim-L-BFGS", nothing),
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

        # Inject a 200 ms delay into Jacobian functions (simulates an expensive model)
        orig_jac = prob_data.jacobian_func
        prob_data = merge(prob_data, (jacobian_func = x -> (sleep(0.2); orig_jac(x)),))

        if hasproperty(prob_data, :jacobian_func!)
            orig_jac! = prob_data.jacobian_func!
            prob_data = merge(
                prob_data,
                (jacobian_func! = (J, x) -> (sleep(0.2); orig_jac!(J, x)),),
            )
        end

        # The gradient of 0.5‖r‖² is J'r — computing it requires the Jacobian, so the
        # gradient-based solvers (BFGS/L-BFGS) must pay the same delay or the
        # expensive-jacobian comparison would be unfair in their favor.
        orig_grad = prob_data.grad_func
        prob_data = merge(prob_data, (grad_func = x -> (sleep(0.2); orig_grad(x)),))

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

# Convert to DataFrame and persist raw results so figures can be rebuilt without
# rerunning the benchmark (see scripts/plot_results.jl).
df_nls = DataFrame(nls_results)
results_dir = normpath(joinpath(@__DIR__, "..", "results"))
mkpath(results_dir)
CSV.write(joinpath(results_dir, "nlls_results_delay.csv"), select(df_nls, Not(:x_opt)))

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
