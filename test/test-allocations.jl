# Gross-regression tripwires, not zero-allocation claims. The ceilings are roughly 3x the values
# measured on Julia 1.12 / macOS aarch64 (worst case on these shapes: 145 KB for a subproblem solve,
# 52 KB per solver iteration). The λ-iteration assembles its damped system into cache buffers and
# factorizes in place, so nothing there scales with the problem; what is left is LAPACK's own
# workspace, about 50 KB per Q product or pivoted-QR solve, which buffer reuse cannot remove.
# benchmark/scripts/allocations.jl measures the same thing across larger shapes.
@testsnippet AllocationCeilings begin
    const SUBPROBLEM_CEIL = 500_000     # bytes, one solve_subproblem call
    const ITERATION_CEIL = 160_000      # bytes per iteration of lm_trust_region!
end

@testitem "allocations: solve_subproblem" tags = [:unit, :slow] setup =
    [Problems, Solver, AllocationCeilings] begin
    using Chairmarks

    Random.seed!(1)
    for (n, m) in ((20, 10), (10, 20)), strategy in Problems.STRATEGIES
        Problems.wide_only(strategy) && n > m && continue
        J, f = randn(n, m), randn(n)
        cache = TRLS.subproblem_cache_init(strategy, TRLS.NoScaling(), J)
        p_gn_norm = norm(pinv(J) * f)
        for (path, Δ) in (("Gauss-Newton", 2 * p_gn_norm), ("boundary", 0.5 * p_gn_norm))
            bytes = minimum(@be TRLS.solve_subproblem($J, $f, $Δ, $cache, 0.0)).bytes
            println("  $(Problems.label(strategy)) $(n)×$(m) $path: $bytes bytes")
            @test bytes <= SUBPROBLEM_CEIL
        end
    end
end

@testitem "allocations per iteration of lm_trust_region!" tags = [:unit, :slow] setup =
    [Problems, Solver, AllocationCeilings] begin
    using Chairmarks

    rosen!(f, x) = (f[1] = 10(x[2] - x[1]^2); f[2] = 1 - x[1]; f)
    rosen_jac!(J, x) = (J[1, 1] = -20x[1]; J[1, 2] = 10; J[2, 1] = -1; J[2, 2] = 0; J)
    for strategy in Problems.STRATEGIES, bounded in (false, true)
        # gtol = ftol = min_trust_radius = 0 disables every early exit, so the solver runs exactly
        # max_iter iterations and the difference between two budgets is the per-iteration cost.
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
            "  $(Problems.label(strategy))$(bounded ? " bounded" : ""): $(round(Int, per_iteration)) bytes/iteration",
        )
        @test per_iteration <= ITERATION_CEIL
    end
end
