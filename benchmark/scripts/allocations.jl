# Time, bytes and allocation counts per strategy × scaling, for one subproblem solve and for a
# full 20-iteration lm_trust_region! run, on a synthetic exponential fit r_i = exp(a_iᵀx) − b_i.
#   julia --project=benchmark benchmark/scripts/allocations.jl
using Chairmarks, DataFrames, CSV, LinearAlgebra, Random, nonlinearlstr
const NL = nonlinearlstr

function expfit(n, m; seed = 1)
    Random.seed!(seed)
    A = randn(n, m) / sqrt(m)
    b = exp.(A * randn(m))
    res!(r, x) = (mul!(r, A, x); r .= exp.(r) .- b; r)
    jac!(J, x) = (J .= exp.(A * x) .* A; J)
    return res!, jac!, zeros(m)
end

sizes = [(20, 10), (200, 50), (2000, 200), (10, 20), (50, 200), (200, 2000)]
strategies = (NL.QRCholStrategy(), NL.QRStrategy(), NL.LQStrategy(), NL.LQCholStrategy())
scalings = (NL.NoScaling(), NL.JacobianScaling())
rows = []
for (n, m) in sizes, strategy in strategies, scaling in scalings
    strategy isa Union{NL.LQStrategy,NL.LQCholStrategy} && n > m && continue
    res!, jac!, x0 = expfit(n, m)
    # gtol = ftol = min_trust_radius = 0: no early exit, exactly 20 iterations
    full = minimum(
        @be NL.lm_trust_region!(
            res!,
            jac!,
            x0,
            n,
            strategy,
            scaling;
            max_iter = 20,
            gtol = 0.0,
            ftol = 0.0,
            min_trust_radius = 0.0,
        )
    )
    J, f = zeros(n, m), zeros(n)
    jac!(J, x0)
    res!(f, x0)
    cache = NL.subproblem_cache_init(strategy, scaling, J)
    Δ = 0.5 * norm(cache.scaling_matrix * (pinv(J) * f))     # boundary solution: the λ-iteration runs
    sub = minimum(@be NL.solve_subproblem(J, f, Δ, cache, 0.0))
    push!(
        rows,
        (
            rows = n,
            cols = m,
            strategy = string(nameof(typeof(strategy))),
            scaling = string(nameof(typeof(scaling))),
            solve_time = full.time,
            solve_bytes = full.bytes,
            solve_allocs = full.allocs,
            bytes_per_iteration = full.bytes / 20,
            subproblem_time = sub.time,
            subproblem_bytes = sub.bytes,
            subproblem_allocs = sub.allocs,
        ),
    )
    println(rows[end])
end
df = DataFrame(rows)
show(df; allrows = true, allcols = true)
println()
CSV.write(joinpath(@__DIR__, "..", "results", "allocations.csv"), df)
