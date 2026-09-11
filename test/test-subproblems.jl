@testsnippet SubproblemChecks begin
    using LinearAlgebra, Random
    import TrustRegionLeastSquares as TRLS

    # Backward-error measure for the normal equations (JᵀJ + λD²) p = -Jᵀf of the returned (λ, p).
    normal_equation_error(J, f, D, λ, p) =
        norm((J'J + λ * D^2) * p + J'f) / (opnorm(J)^2 * norm(p) + norm(J'f))
    on_boundary(Dp_norm, Δ; θ = 1e-4) = (1 - θ) * Δ <= Dp_norm <= (1 + θ) * Δ

    # Conditioning fixture: singular values 1 … 10^-k, so cond(J) = 10^k exactly.
    function illconditioned_problem(k; n = 40, m = 6)
        Random.seed!(k)
        U = Matrix(qr(randn(n, n)).Q)[:, 1:m]
        V = Matrix(qr(randn(m, m)).Q)
        J = U * Diagonal(exp10.(range(0, -k, length = m))) * V
        return J, randn(n)
    end
end

@testitem "subproblem algebra per strategy" tags = [:unit, :validation, :fast] setup =
    [Problems, SubproblemChecks] begin
    @testset "$(Problems.label(strategy)) / $(Problems.label(scaling))" for strategy in
                                                                            Problems.STRATEGIES,
        scaling in Problems.SCALINGS

        Random.seed!(7)
        for (n, m) in ((12, 5), (5, 12))
            Problems.wide_only(strategy) && n > m && continue
            J = randn(n, m)
            J[:, 1] .*= 1e3                               # badly scaled column: D ≠ I under JacobianScaling
            f = randn(n)
            cache = TRLS.subproblem_cache_init(strategy, scaling, J)
            D = cache.scaling_matrix
            if scaling isa TRLS.JacobianScaling
                @test D.diag ≈ [norm(J[:, i]) for i = 1:m]
            else
                @test D == I
            end
            p_gn = -pinv(J) * f                           # least-squares (tall) or minimum-norm (wide) solution
            # Gauss–Newton step inside the region: λ = 0 and p is the pseudo-inverse solution
            λ = TRLS.solve_subproblem(J, f, 2 * norm(D * p_gn), cache, 0.0)
            @test λ == 0
            @test cache.p ≈ p_gn rtol = 1e-10
            # Step on the boundary: ‖Dp‖ = Δ within θ and the normal equations hold for the returned λ
            Δ = 0.5 * norm(D * p_gn)
            λ = TRLS.solve_subproblem(J, f, Δ, cache, 0.0)
            @test λ > 0
            @test on_boundary(norm(D * cache.p), Δ)
            @test normal_equation_error(J, f, D, λ, cache.p) < 1e-12
            # Warm start from the converged λ reproduces the step
            p_boundary = copy(cache.p)
            @test TRLS.solve_subproblem(J, f, Δ, cache, λ) ≈ λ
            @test cache.p ≈ p_boundary
        end
    end
end

@testitem "JacobianScaling: Moré's non-decreasing column norms" tags =
    [:unit, :validation, :fast] setup = [SubproblemChecks] begin
    Random.seed!(2)
    J = randn(8, 3)
    D = TRLS.scaling!(Diagonal(zeros(3)), TRLS.JacobianScaling(), J)
    column_norms = copy(D.diag)
    TRLS.scaling!(D, TRLS.JacobianScaling(), 0.1 * J)
    @test D.diag == column_norms                          # smaller norms leave D unchanged
    TRLS.scaling!(D, TRLS.JacobianScaling(), 10 * J)
    @test D.diag ≈ 10 * column_norms
    J[:, 2] .= 0
    @test TRLS.scaling!(Diagonal(zeros(3)), TRLS.JacobianScaling(), J).diag[2] == 1   # zero column → 1
    @test TRLS.scaling!(D, TRLS.NoScaling(), J) == I
end

@testitem "rank-deficient wide J: minimum-norm Gauss–Newton step (COD)" tags =
    [:unit, :validation, :fast] setup = [SubproblemChecks] begin
    Random.seed!(3)
    for (n, m, r) in ((4, 9, 2), (5, 12, 4), (3, 7, 1))
        J = randn(n, r) * randn(r, m)                     # exact rank r
        f = randn(n)
        for strategy in (TRLS.LQStrategy(), TRLS.LQCholStrategy())
            cache = TRLS.subproblem_cache_init(strategy, TRLS.NoScaling(), J)
            @test TRLS.numerical_rank(cache.factorization) == r
            @test TRLS.solve_subproblem(J, f, 1e6, cache, 0.0) == 0
            @test cache.p ≈ pinv(J) * (-f) rtol = 1e-8
            @test norm(J' * (J * cache.p + f)) < 1e-8 * norm(J) * norm(f)
        end
    end
end

# Conditioning. The QR strategies factorize J itself and stay accurate throughout; the Cholesky
# strategies form JᵀJ (or J D⁻² Jᵀ), squaring cond(J), so they degrade and eventually fail — by
# design, see the QRCholStrategy docstring. cond(J) = 1e8 already means cond(JᵀJ) = 1e16 ≈ 1/eps.
@testitem "ill-conditioned J: QR strategies stay on the boundary" tags =
    [:unit, :validation, :fast] setup = [Problems, SubproblemChecks] begin
    @testset "cond(J) = 1e$k" for k in (4, 8, 12)
        J, f = illconditioned_problem(k)
        D = Diagonal(ones(size(J, 2)))
        p_gn = -J \ f
        for strategy in (TRLS.QRStrategy(), TRLS.LQStrategy()), fraction in (1e-3, 0.5)
            size(J, 1) > size(J, 2) && Problems.wide_only(strategy) && continue
            Δ = fraction * norm(p_gn)
            cache = TRLS.subproblem_cache_init(strategy, TRLS.NoScaling(), J)
            λ = TRLS.solve_subproblem(J, f, Δ, cache, 0.0)
            @test on_boundary(norm(cache.p), Δ)
            @test normal_equation_error(J, f, D, λ, cache.p) < 1e-10
        end
    end
end

@testitem "ill-conditioned J: the Cholesky strategies degrade as cond(JᵀJ) grows" tags =
    [:unit, :validation, :fast] setup = [Problems, SubproblemChecks] begin
    D = Diagonal(ones(6))
    # cond(JᵀJ) = 1e8: still fine, and the model value agrees with the stable QR strategy.
    J, f = illconditioned_problem(4)
    p_gn = -J \ f
    for fraction in (1e-3, 0.5)
        Δ = fraction * norm(p_gn)
        model_value = map((TRLS.QRCholStrategy(), TRLS.QRStrategy())) do strategy
            cache = TRLS.subproblem_cache_init(strategy, TRLS.NoScaling(), J)
            λ = TRLS.solve_subproblem(J, f, Δ, cache, 0.0)
            @test on_boundary(norm(cache.p), Δ)
            @test normal_equation_error(J, f, D, λ, cache.p) < 1e-10
            return sum(abs2, J * cache.p + f) / 2
        end
        @test model_value[1] ≈ model_value[2] rtol = 1e-8
    end
    # cond(JᵀJ) = 1e16: the returned (λ, p) still satisfies the normal equations, but the
    # λ-iteration can no longer place ‖Dp‖ on the boundary at any iteration count.
    J, f = illconditioned_problem(8)
    Δ = 0.5 * norm(-J \ f)
    let cache = TRLS.subproblem_cache_init(TRLS.QRCholStrategy(), TRLS.NoScaling(), J)
        λ = TRLS.solve_subproblem(J, f, Δ, cache, 0.0)
        @test normal_equation_error(J, f, D, λ, cache.p) < 1e-10
    end
    # cond(JᵀJ) = 1e24: the λ the bracket asks for is below eps·σmax², so JᵀJ + λD² — positive
    # definite in exact arithmetic — may be numerically indefinite and the factorization fails.
    # Whether it does depends on the LAPACK build, so accept either outcome, but never a step that
    # silently violates the normal equations.
    J, f = illconditioned_problem(12)
    Δ = 0.5 * norm(-J \ f)
    for strategy in (TRLS.QRCholStrategy(), TRLS.LQCholStrategy())
        size(J, 1) > size(J, 2) && Problems.wide_only(strategy) && continue
        cache = TRLS.subproblem_cache_init(strategy, TRLS.NoScaling(), J)
        try
            λ = TRLS.solve_subproblem(J, f, Δ, cache, 0.0)
            @test normal_equation_error(J, f, D, λ, cache.p) < 1e-10
        catch err
            @test err isa PosDefException
        end
    end
end
