using Test, Chairmarks, LinearAlgebra, Random
isdefined(Main, :MGH) || include(joinpath(@__DIR__, "..", "problems.jl"))

# Gross-regression tripwire, not a zero-allocation claim: the λ-iteration deliberately builds
# small temporaries for readability, and every LAPACK Q product allocates its workspace. The
# ceilings are ~3× the numbers measured on Julia 1.12 / macOS; they only fire on something like a
# dense per-iteration matrix copy sneaking into the hot loop.
# Roughly 3x the values measured on Julia 1.12 / macOS aarch64 (worst case on these shapes: 145 KB
# for a subproblem solve, 52 KB per solver iteration). The λ-iteration assembles its damped system
# into cache buffers and factorizes in place, so nothing there scales with the problem; what is left
# is LAPACK's own workspace, ~50 KB per Q product or pivoted-QR solve, which buffer reuse cannot
# remove. benchmark/scripts/allocations.jl measures the same thing across larger shapes.
const SUBPROBLEM_CEIL = 500_000     # bytes, one solve_subproblem call
const ITERATION_CEIL = 160_000      # bytes per iteration of lm_trust_region!

@testset "allocations: solve_subproblem" begin
    Random.seed!(1)
    for (n, m) in ((20, 10), (10, 20)), strategy in STRATEGIES
        wide_only(strategy) && n > m && continue
        J, f = randn(n, m), randn(n)
        cache = TRLS.subproblem_cache_init(strategy, TRLS.NoScaling(), J)
        p_gn_norm = norm(pinv(J) * f)
        for (path, Δ) in (("Gauss-Newton", 2 * p_gn_norm), ("boundary", 0.5 * p_gn_norm))
            bytes = minimum(@be TRLS.solve_subproblem($J, $f, $Δ, $cache, 0.0)).bytes
            println("  $(label(strategy)) $(n)×$(m) $path: $bytes bytes")
            @test bytes <= SUBPROBLEM_CEIL
        end
    end
end

@testset "allocations per iteration of lm_trust_region!" begin
    rosen!(f, x) = (f[1] = 10(x[2] - x[1]^2); f[2] = 1 - x[1]; f)
    rosen_jac!(J, x) = (J[1, 1] = -20x[1]; J[1, 2] = 10; J[2, 1] = -1; J[2, 2] = 0; J)
    # gtol = ftol = min_trust_radius = 0 disables every early exit, so the solver runs exactly max_iter iterations.
    run(strategy, k) = TRLS.lm_trust_region!(
        rosen!,
        rosen_jac!,
        [-1.2, 1.0],
        2,
        strategy;
        max_iter = k,
        gtol = 0.0,
        ftol = 0.0,
        min_trust_radius = 0.0,
    )
    for strategy in STRATEGIES, bounded in (false, true)
        kw = bounded ? (lb = [-2.0, -2.0], ub = [0.5, 2.0]) : (;)
        bytes(k) = minimum(
            @be TRLS.lm_trust_region!(
                rosen!,
                rosen_jac!,
                [-1.2, 1.0],
                2,
                $strategy;
                max_iter = $k,
                gtol = 0.0,
                ftol = 0.0,
                min_trust_radius = 0.0,
                $kw...,
            )
        ).bytes
        per_iteration = (bytes(12) - bytes(2)) / 10
        println(
            "  $(label(strategy))$(bounded ? " bounded" : ""): $(round(Int, per_iteration)) bytes/iteration",
        )
        @test per_iteration <= ITERATION_CEIL
    end
end
