function nlls_benchmark(problems, solvers; max_iter = 100)
    """Run comprehensive elapsed on NLLS problems"""
    # Define solvers to test
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
        # Create Julia functions
        # Note: prob_data.n is the residual count, prob_data.m the variable count.
        println("  Variables: $(prob_data.m)")
        println("  Residuals: $(prob_data.n)")
        initial_obj = prob_data.obj_func(prob_data.x0)
        println("  Initial objective: $initial_obj")
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
        # Clean up
        finalize(nlp)
    end
    return results
end
