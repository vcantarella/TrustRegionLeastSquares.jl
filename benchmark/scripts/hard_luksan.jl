# Hard exponential-fit problems A.1-A.6 from [Luk96]: L. Lukšan, "Hybrid methods for large sparse
# nonlinear least squares", Journal of Optimization Theory and Applications 89(3), 575-595 (1996),
# doi:10.1007/BF02275350. Solved in their original and in a
# log-parameterized form (x = exp(y), started from y0 = log(x0)).
#   julia --project=benchmark benchmark/scripts/hard_luksan.jl
include(joinpath(@__DIR__, "..", "harness.jl"))
include(joinpath(@__DIR__, "..", "evaluate.jl"))
using DataFrames, CSV, ForwardDiff, LinearAlgebra, Test
using PRIMA, NonlinearSolve, LeastSquaresOptim
import TrustRegionLeastSquares as TRLS

# The six problems of [Luk96], Appendix A
fa1(x, t) = x[1] + x[2]*exp(x[3]*t)
ti1 = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50]
yi1 = [16.7, 26.8, 16.9, 17.1, 17.2, 17.4, 17.6, 17.9, 18.1, 18.7]
resa1(x) = [fa1(x, t) - y for (t, y) in zip(ti1, yi1)]
x01 = [20, 2, 0.5]
resa1(x01)
fa2(x, t) = exp(x[1]*t)+exp(x[2]*t)
ti2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
yi2 = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
resa2(x) = [fa2(x, t) - y for (t, y) in zip(ti2, yi2)]
x02 = [0.3, 0.2]
resa2(x02)
fa3(x, t) = x[1]*exp(x[2]/(x[3]+t))
ti3 = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125]
yi3 = [
    34780,
    28610,
    23650,
    19630,
    16370,
    13720,
    11540,
    9744,
    8261,
    7030,
    6005,
    5147,
    4427,
    3820,
    3307,
    2872,
]
resa3(x) = [fa3(x, t) - y for (t, y) in zip(ti3, yi3)]
x03 = [0.02, 4000, 250]
resa3(x03)
fa4(x, t) = x[1]*exp(-x[3]*t)+x[2]*exp(-x[4]*t)
ti4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
yi4 = [99.6, 67.1, 45.9, 31.9, 22.5, 16.1, 11.7, 8.6, 6.38, 4.78]
resa4(x) = [fa4(x, t) - y for (t, y) in zip(ti4, yi4)]
x04 = [1, 1, 1, 1]
fa5(x, t) = x[1]*exp(-x[3]*t)+x[2]*exp(-x[4]*t)
ti5 = [
    7.448,
    7.448,
    7.552,
    7.607,
    7.847,
    7.877,
    7.969,
    8.176,
    8.176,
    8.523,
    8.552,
    8.903,
    9.114,
    9.284,
    9.439,
]
yi5 = [
    57.554,
    53.546,
    45.29,
    51.286,
    31.623,
    27.952,
    19.498,
    16.444,
    21.777,
    13.996,
    11.803,
    7.727,
    4.764,
    4.305,
    3.006,
]
resa5(x) = [fa5(x, t) - y for (t, y) in zip(ti5, yi5)]
x05 = [1e5, 1e5, 1.079, 1.31]
resa5(x05)
fa6(x, t) = x[1]*t^x[3] + x[2]*t^x[4]
ti6 = 12:23
yi6 = [7.31, 7.55, 7.8, 8.05, 8.31, 8.57, 8.84, 9.12, 9.4, 9.69, 9.99, 10.3]
resa6(x) = [fa6(x, t) - y for (t, y) in zip(ti6, yi6)]
x06 = [1e3, 0.01, 2, 100]
resa6(x06)


solvers = [
    ("TRLS", TRLS.lm_trust_region!),
    ("LM-QR-scaled", TRLS.lm_trust_region!),
    ("LM-QRChol", TRLS.lm_trust_region!),
    ("LM-QRChol-scaled", TRLS.lm_trust_region!),
    ("PRIMA-NEWUOA", nothing),  # Special handling
    ("PRIMA-BOBYQA", nothing),  # Special handling
    ("NonlinearSolve-TrustRegion", NonlinearSolve.TrustRegion),
    ("NonlinearSolve-LevenbergMarquardt", NonlinearSolve.LevenbergMarquardt),
    ("NonlinearSolve-GaussNewton", NonlinearSolve.GaussNewton),
    ("NonlinearSolve-PolyAlg", NonlinearSolve.FastShortcutNLLSPolyalg),
    ("JSO-TRON", tron),
    ("JSO-TRUNK", trunk),
    ("LSO-DogLeg-QR", LeastSquaresOptim.Dogleg(LeastSquaresOptim.QR())),
    ("LSO-Levenberg-QR", LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.QR())),
    ("LSO-DogLeg-chol", LeastSquaresOptim.Dogleg(LeastSquaresOptim.Cholesky())),
    (
        "LSO-Levenberg-chol",
        LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.Cholesky()),
    ),
    ("LSO-DogLeg-LSMR", LeastSquaresOptim.Dogleg(LeastSquaresOptim.LSMR())),
    ("LSO-Levenberg-LSMR", LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.LSMR())),
    ("Scipy-LeastSquares", nothing),
    ("Scipy-LSMR", nothing),
    ("NLLSsolver-levenbergmarquardt", NLLSsolver.levenbergmarquardt),
    ("NLLSsolver-dogleg", NLLSsolver.dogleg),
]

# Adapter: (name, residual, x0) -> the prob_data shape test_solver_on_problem expects.
function make_prob_data_from_res(name, resf, x0)
    x0 = float.(x0)
    jac(x) = ForwardDiff.jacobian(resf, x)
    n, m = size(jac(x0))
    return (
        n = n,
        m = m,
        x0 = x0,
        bl = fill(-Inf, m),
        bu = fill(Inf, m),
        residual_func = resf,
        jacobian_func = jac,
        residual_func! = (r, x) -> (r .= resf(x); r),
        jacobian_func! = (J, x) -> ForwardDiff.jacobian!(J, resf, x),
        obj_func = x -> 0.5 * dot(resf(x), resf(x)),
        grad_func = x -> jac(x)' * resf(x),
        hess_func = x -> jac(x)' * jac(x),
        problem = name,
    )
end

function run_luksan(problems, tag)
    results = []
    for prob_data in problems
        println("Problem $(prob_data.problem) ($tag), x0 = $(prob_data.x0)")
        flush(stdout)   # stdout is block-buffered to a file: without this, slow looks like hung
        for (solver_name, solver_func) in solvers
            print("  Testing $solver_name... ")
            flush(stdout)
            result =
                test_solver_on_problem(solver_name, solver_func, prob_data, nothing, 400)
            if result.success && result.converged
                println(
                    "✓ obj=$(round(result.final_cost, digits = 8)), iters=$(result.iterations)",
                )
            else
                println(result.success ? "✗ no convergence" : "✗ failed")
            end
            flush(stdout)
            push!(
                results,
                merge(
                    result,
                    (
                        problem = prob_data.problem,
                        nvars = prob_data.m,
                        nresiduals = prob_data.n,
                        initial_objective = prob_data.obj_func(prob_data.x0),
                    ),
                ),
            )
        end
    end
    df = DataFrame(results)
    CSV.write(
        joinpath(@__DIR__, "..", "results", "nlls_results_luksan_$tag.csv"),
        select(df, Not(:x_opt)),
    )
    df_proc = compare_with_best(df)
    summary_df = evaluate_solvers(df_proc)
    display(summary_df)
    # The guard is on LM-QR-scaled, not on the unscaled TRLS row. These are exponential fits whose
    # parameters span orders of magnitude (A.3 starts at [0.02, 4000, 250], A.5 at [1e5, 1e5, 1.08,
    # 1.31]), which is exactly what JacobianScaling exists for: it takes 4 of the 6 where the
    # unscaled configuration takes 2. Asserting on the unscaled row would be asserting that the
    # wrong tool for the job keeps working.
    rate(solver) = summary_df[summary_df.solver .== solver, :percentage_success][1]
    @test rate("LM-QR-scaled") > 0.49
    println(
        "TRLS variants on the $tag set: unscaled $(rate("TRLS")), scaled $(rate("LM-QR-scaled"))",
    )
    figpath = joinpath(plots_dir(), "hardluksan_nls_solver_performance_$tag.png")
    save(figpath, build_performance_plots(df_proc))
    println("plot: $figpath")
    return summary_df
end

residuals = [resa1, resa2, resa3, resa4, resa5, resa6]
starts = [x01, x02, x03, x04, x05, x06]
names = ["A.1", "A.2", "A.3", "A.4", "A.5", "A.6"]
run_luksan(
    [make_prob_data_from_res(nm, r, x0) for (nm, r, x0) in zip(names, residuals, starts)],
    "original",
)
run_luksan(
    [
        make_prob_data_from_res(nm, x -> r(exp.(x)), log.(x0)) for
        (nm, r, x0) in zip(names, residuals, starts)
    ],
    "log",
)
