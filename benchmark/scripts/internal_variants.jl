include(joinpath(@__DIR__, "..", "harness.jl"))
using DataFrames
using nonlinearlstr
using Test
using CairoMakie
using Chairmarks

# 1. Find problems
nls_problems = find_nlls_problems(1000) # Benchmark on all available NLS problems

# 2. Define solvers
solvers = [
    ("LM-SVD", nonlinearlstr.lm_trust_region!),
    ("LM-QR", nonlinearlstr.lm_trust_region!),
    ("LM-QR-Recursive", nonlinearlstr.lm_trust_region!),
    ("LM-VCChol", nonlinearlstr.lm_trust_region_v2!),
]

# 3. Run benchmark
println("Running benchmark on $(length(nls_problems)) problems...")

prob = create_nls_functions(eval(nls_problems[17])())
prob_data = prob
res! = prob.residual_func!
jac! = prob.jacobian_func!
x0 = prob.x0
nonlinearlstr.lm_trust_region!(res!, jac!, x0, prob.n, nonlinearlstr.QRSolve())
nonlinearlstr.lm_trust_region_v2!(
    res!,
    jac!,
    x0,
    prob.n,
    nonlinearlstr.QRCholStrategy(),
    nonlinearlstr.NoScaling();
    max_iter = 400,
    gtol = 1e-6,
)

println(
    @be nonlinearlstr.lm_trust_region!(
        $res!,
        $jac!,
        $x0,
        $prob.n,
        $nonlinearlstr.QRSolve(),
    )
)
println(@be nonlinearlstr.lm_trust_region_v2!($res!, $jac!, $x0, $prob.n))


nls_results = nlls_benchmark(nls_problems, solvers, max_iter = 400)

# 4. Analyze results
df_nls = DataFrame(nls_results)

include(joinpath(@__DIR__, "..", "evaluate.jl"))

df_nls_proc = compare_with_best(df_nls)
summary_nls = evaluate_solvers(df_nls_proc)

println("\n\n" * "="^60)
println("NONLINEARLSTR SOLVER PERFORMANCE SUMMARY")
println("="^60)
display(summary_nls)

# 5. Plots
fig_nls = build_performance_plots(df_nls_proc)
nonlinearlstr_plot_path = joinpath(plots_dir(), "nonlinearlstr_solver_performance.png")
save(nonlinearlstr_plot_path, fig_nls)
println("\nPerformance plot saved to '$nonlinearlstr_plot_path'")
