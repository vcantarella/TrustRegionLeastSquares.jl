using Test
using Chairmarks
using LinearAlgebra
using nonlinearlstr

# Allocation regression guard for the steady-state subproblem path. We measure with
# Chairmarks (already a test dep) rather than AllocCheck to keep the unit environment
# light and portable. This is a GROSS regression tripwire, not a zero-alloc claim: the
# subproblem solve inherently allocates a factorization (observed baselines on a 20x10
# problem: solve_subproblem ~23-102 KB, find_λ_scaled ~19-52 KB, strategy-dependent).
# The ceiling is set well above those baselines so it only fires on a large regression
# (e.g. a per-iteration dense temporary reintroduced in the hot loop), not on
# BLAS/version noise.
const ALLOC_CEIL = 250_000  # bytes

function _alloc_problem()
    n, m = 10, 20
    f = zeros(m)
    f[1:n] .= 0.5
    J = zeros(m, n)
    J[1:n, 1:n] .= I(n)
    return f, J, 1.0
end

const ALLOC_STRATEGIES = [
    (nonlinearlstr.QRSolve(), nonlinearlstr.NoScaling(), "LM-QR"),
    (nonlinearlstr.SVDSolve(), nonlinearlstr.NoScaling(), "LM-SVD"),
    (nonlinearlstr.QRrecursiveSolve(), nonlinearlstr.NoScaling(), "LM-QR-Recursive"),
]

@testset "solve_subproblem allocations" begin
    f, J, radius = _alloc_problem()
    for (strat, scaling, name) in ALLOC_STRATEGIES
        cache = nonlinearlstr.SubproblemCache(strat, scaling, J)
        allocs =
            minimum(@be nonlinearlstr.solve_subproblem($strat, $J, $f, $radius, $cache)).bytes
        println("  $name solve_subproblem: $allocs bytes")
        @test allocs <= ALLOC_CEIL
    end
end

@testset "find_λ_scaled allocations" begin
    f, J, radius = _alloc_problem()
    for (strat, scaling, name) in ALLOC_STRATEGIES
        cache = nonlinearlstr.SubproblemCache(strat, scaling, J)
        allocs = minimum(
            @be nonlinearlstr.find_λ_scaled(
                $strat,
                $cache,
                $radius,
                $J,
                $cache.scaling_matrix,
                $f,
                200,
                1e-6,
            )
        ).bytes
        println("  $name find_λ_scaled: $allocs bytes")
        @test allocs <= ALLOC_CEIL
    end
end
