using Test
using LinearAlgebra
using nonlinearlstr

# Correctness tests for the bound-constrained reflective solver. Previously the
# reflective method was only exercised by a benchmark success-rate check; these are
# self-contained unit tests on problems with known optima.
#
# Problem family: linear residual r(x) = x - target, so f(x) = 0.5‖x - target‖² with
# Jacobian J = I. The unconstrained optimum is `target`; with active bounds the optimum
# is the projection of `target` onto the box.
#
# Note: x0 must be nonzero. With norm_overrides_initial_radius=true (the default) and
# x0 = 0, the initial trust radius is norm(D*x0) = 0, so no step is possible and the
# solver bails immediately. The benchmark problems always start from a nonzero x0.

@testset "Reflective bounded solver" begin
    n = 3
    res(target) = x -> x .- target
    jac(x) = Matrix{Float64}(I, length(x), length(x))
    x0 = fill(0.1, n)

    @testset "infinite bounds recover unconstrained optimum" begin
        target = [2.0, -1.5, 0.7]
        x, f, g, iter = nonlinearlstr.lm_trust_region_reflective(
            res(target),
            jac,
            copy(x0);
            lb = fill(-Inf, n),
            ub = fill(Inf, n),
            max_iter = 200,
        )
        @test x ≈ target atol = 1e-4
        @test norm(g) < 1e-3
    end

    @testset "active upper bound projects optimum" begin
        target = [2.0, 2.0, 2.0]
        ub = [1.0, 1.0, 1.0]
        lb = fill(-Inf, n)
        x, f, g, iter = nonlinearlstr.lm_trust_region_reflective(
            res(target),
            jac,
            copy(x0);
            lb = lb,
            ub = ub,
            max_iter = 200,
        )
        @test all(x .<= ub .+ 1e-8)            # feasible
        @test x ≈ ub atol = 1e-4               # optimum sits on the active bound
    end

    @testset "active lower bound projects optimum" begin
        target = [-2.0, -2.0, -2.0]
        lb = [-1.0, -1.0, -1.0]
        ub = fill(Inf, n)
        x, f, g, iter = nonlinearlstr.lm_trust_region_reflective(
            res(target),
            jac,
            copy(x0);
            lb = lb,
            ub = ub,
            max_iter = 200,
        )
        @test all(x .>= lb .- 1e-8)            # feasible
        @test x ≈ lb atol = 1e-4
    end

    @testset "mixed: one bound active, one interior" begin
        target = [5.0, 0.25, -5.0]
        lb = [-1.0, -1.0, -1.0]
        ub = [1.0, 1.0, 1.0]
        x, f, g, iter = nonlinearlstr.lm_trust_region_reflective(
            res(target),
            jac,
            copy(x0);
            lb = lb,
            ub = ub,
            max_iter = 200,
        )
        @test all(lb .- 1e-8 .<= x .<= ub .+ 1e-8)
        @test x ≈ clamp.(target, lb, ub) atol = 1e-4   # projection of target onto box
    end
end
