# The TrustRegionLeastSquares variants (QRChol / QR / LQ / LQChol subproblem strategies, with and
# without JacobianScaling) scored against each other.
# Run with:
#   julia --project=benchmark benchmark/scripts/internal_variants.jl
# Knobs (env vars, as in the compare_* scripts): MAX_VARS, PROBLEM_LIMIT, CROP_RATIO.
include(joinpath(@__DIR__, "..", "harness.jl"))
include(joinpath(@__DIR__, "..", "evaluate.jl"))
using DataFrames, CSV
import TrustRegionLeastSquares as TRLS

const MAX_VARS = parse(Int, get(ENV, "MAX_VARS", "999"))
const PROBLEM_LIMIT = parse(Int, get(ENV, "PROBLEM_LIMIT", "0"))  # 0 = no cap
const CROP_RATIO = parse(Float64, get(ENV, "CROP_RATIO", "0.5"))
let probs = find_nlls_problems(MAX_VARS)
    global nls_problems = PROBLEM_LIMIT > 0 ? probs[1:min(PROBLEM_LIMIT, end)] : probs
end
const results_dir = normpath(joinpath(@__DIR__, "..", "results"))
crop(pd) = crop_nls_functions(pd, clamp(ceil(Int, CROP_RATIO * pd.m), 1, pd.m - 1))

# The label selects strategy and scaling in dispatch.jl.
lm = TRLS.lm_trust_region!
variants =
    [("LM-QRChol", lm), ("LM-QR", lm), ("LM-QRChol-scaled", lm), ("LM-QR-scaled", lm)]
lq_variants =
    [("LM-LQ", lm), ("LM-LQChol", lm), ("LM-LQ-scaled", lm), ("LM-LQChol-scaled", lm)]   # LQ needs residuals <= variables

# Run the variants through the harness and score by cost atol (shared by both suites).
function run_and_score(suite, solvers, atol; transform = identity)
    println("\n" * "="^60 * "\n$suite suite\n" * "="^60)
    df_v2 = select(
        DataFrame(nlls_benchmark(nls_problems, solvers; max_iter = 400, transform)),
        Not(:x_opt),
    )
    CSV.write(joinpath(results_dir, "nlls_results_internal_$suite.csv"), df_v2)
    return compare_with_best(df_v2; atol)
end

function finish_suite(suite, df_proc, tbl)
    display(tbl)
    figpath = joinpath(plots_dir(), "trls_internal_$suite.png")
    save(figpath, build_performance_plots(df_proc; background = :white))
    println("plot: $figpath")
end

function run_suite(suite, solvers, atol; transform = identity)
    df_proc = run_and_score(suite, solvers, atol; transform)
    tbl = evaluate_solvers(df_proc)
    finish_suite(suite, df_proc, tbl)
    return tbl
end

# Underdetermined variant of run_suite. Same cost-atol scoring, plus a min-norm
# comparison: every cropped problem has a manifold of zero-residual solutions, and a
# min-norm method lands on the one nearest x0. So among the solvers that PASS the cost
# atol, the smaller ‖x-x0‖ is the better step. We report, over passing solves only,
# the median ‖x-x0‖ and a "min-norm rate" — the fraction of ALL problems where the
# solver both passed AND its ‖x-x0‖ is within `rtol` of the smallest any solver reached
# on that problem (ties count for everyone within rtol).
function run_suite_underdetermined(solvers, atol; transform = crop, rtol = 1e-3)
    df_proc = run_and_score("underdetermined", solvers, atol; transform)
    @assert all(df_proc.nresiduals .< df_proc.nvars) "every problem must be underdetermined"
    tbl = evaluate_solvers(df_proc)

    passers = df_proc[df_proc.is_success, :]
    best = combine(groupby(passers, :problem), :dist_x0 => minimum => :best_dist)
    passers = leftjoin(passers, best, on = :problem)
    passers.is_minnorm = passers.dist_x0 .<= passers.best_dist .* (1 + rtol) .+ 1e-12
    nprob = length(unique(df_proc.problem))
    minnorm = combine(
        groupby(passers, :solver),
        :dist_x0 => median => :median_dist_x0_passing,
        :is_minnorm => (x -> sum(x) / nprob) => :minnorm_rate,
    )
    # rev by rate, but missing (solvers with no passers) sort to the bottom, not the top.
    tbl = sort!(
        leftjoin(tbl, minnorm, on = :solver),
        :minnorm_rate;
        rev = true,
        by = x -> coalesce(x, -Inf),
    )
    finish_suite("underdetermined", df_proc, tbl)
    return tbl
end

run_suite("overdetermined", variants, 1e-4)
run_suite_underdetermined([variants; lq_variants], 1e-4)
