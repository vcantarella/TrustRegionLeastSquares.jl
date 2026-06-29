include(joinpath(@__DIR__, "..", "harness.jl"))
using NLPModels
using JSOSolvers
using PRIMA
using NonlinearSolve
using Revise
using DataFrames
using nonlinearlstr

nls_problems = find_nlls_problems(999)


solvers = [
    ("LM-QR", nonlinearlstr.lm_trust_region),
    ("LM-QR-scaled", nonlinearlstr.lm_trust_region),
    ("LM-SVD", nonlinearlstr.lm_trust_region),
    ("PRIMA-NEWUOA", nothing),  # Special handling
    ("PRIMA-BOBYQA", nothing),  # Special handling
    ("NonlinearSolve-TrustRegion", NonlinearSolve.TrustRegion),  # Special handling
    ("NonlinearSolve-LevenbergMarquardt", NonlinearSolve.LevenbergMarquardt),  # Special handling
    ("NonlinearSolve-GaussNewton", NonlinearSolve.GaussNewton),  # Special handling
    ("NonlinearSolve-PolyAlg", NonlinearSolve.FastShortcutNLLSPolyalg),  # Special handling
    # not working properly
    # ("MINPACK-lm", NonlinearSolve.CMINPACK(;method=:lm)),  # Special handling
    # ("MINPACK-hybr", NonlinearSolve.CMINPACK(;method=:hybr)),  # Special handling
    ("JSO-TRON", tron),  # Special handling
    ("JSO-TRUNK", trunk),  # Special handling
    ("LSO-DogLeg-QR", LeastSquaresOptim.Dogleg(LeastSquaresOptim.QR())),
    ("LSO-Levenberg-QR", LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.QR())),
    ("LSO-DogLeg-chol", LeastSquaresOptim.Dogleg(LeastSquaresOptim.Cholesky())),
    (
        "LSO-Levenberg-chol",
        LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.Cholesky()),
    ),
    ("LSO-DogLeg-LSMR", LeastSquaresOptim.Dogleg(LeastSquaresOptim.LSMR())),
    ("LSO-Levenberg-LSMR", LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.LSMR())),
    ("Scipy-LeastSquares", nothing),  # Special handling
    ("Scipy-LSMR", nothing),  # Special handling
    ("NLLSsolver-levenbergmarquardt", NLLSsolver.levenbergmarquardt),
    ("NLLSsolver-dogleg", NLLSsolver.dogleg),
]
# nls_problems = nls_problems[1:3]  # limit for testing
nls_results = nlls_benchmark(nls_problems, solvers, max_iter = 400)
#cutest_results = nlls_benchmark(cutest_problems, solvers, max_iter=100)

# Convert to DataFrame
df_nls = DataFrame(nls_results)

include(joinpath(@__DIR__, "..", "evaluate.jl"))

df_nls_proc = compare_with_best(df_nls)
summary_nls = evaluate_solvers(df_nls_proc)
display(summary_nls)
using Test
@test summary_nls[summary_nls[!, :solver] .== "LM-QR", :percentage_success][1] > 0.9
@test summary_nls[summary_nls[!, :solver] .== "LM-SVD", :percentage_success][1] > 0.9
fig_nls = build_performance_plots(df_nls_proc)
save(joinpath(plots_dir(), "nlls_solver_performance.png"), fig_nls)

# # testing minpack for debugging
# prob_name = nls_problems[1]
# nlp = eval(prob_name)()
# prob_data = create_nls_functions(nlp)
# res = test_solver_on_problem(
#     "NLLSsolver-dogleg",
#     NLLSsolver.dogleg,
#     prob_data,
#     nlp
# )
