# A Jacobian with an exactly-zero column, on a residual that has no root. Both halves are
# degenerate at once: J is rank 1 of 2, so the second variable has no influence on the residual at
# all, and the least-squares solution sits at a nonzero residual. Grown out of a scratch file that
# compared the behaviour against NonlinearSolve, which stalled on it.
#
#   f(x) = [x₁ - 1, x₁² - 4]        no root: x₁ = 1 versus x₁ = ±2
#   J(x) = [1 0; 2x₁ 0]             column 2 is exactly zero
#
# Minimizing ½‖f‖² over x₁ gives d/dx₁ = (x₁ - 1) + 2x₁(x₁² - 4) = 0, i.e. 2x₁³ - 7x₁ - 1 = 0,
# whose relevant root is x₁ ≈ 1.9389. That polynomial is the test oracle, so no constant is baked in.
@testitem "degenerate Jacobian with a zero column" tags = [:unit, :validation, :fast] setup =
    [Problems, Solver] begin
    degenerate(x) = [x[1] - 1, x[1]^2 - 4]
    x0 = [0.5, 0.5]

    @testset "$(Problems.label(strategy)) / $(Problems.label(scaling))" for strategy in
                                                                            Problems.STRATEGIES,
        scaling in Problems.SCALINGS

        s = Problems.solve(degenerate, x0, strategy, scaling; gtol = 1e-10, max_iter = 200)
        @test all(isfinite, s.x) && all(isfinite, s.f) && all(isfinite, s.g)
        @test abs(2 * s.x[1]^3 - 7 * s.x[1] - 1) < 1e-8      # a genuine stationary point of the 1-D problem
        @test s.x[2] == x0[2]                                # the zero column cannot move x₂
        @test s.cost > 0.4                                   # the residual is nonzero at the solution
    end

    @testset "the rank deficiency is seen, and the zero column is scaled to one" begin
        J = zeros(2, 2)
        Problems.inplace(degenerate)[2](J, x0)
        @test J[:, 2] == [0.0, 0.0]
        # JacobianScaling's zero-column rule: a zero column would otherwise give D_ii = 0 and an
        # unbounded step in that direction.
        D = TRLS.scaling!(Diagonal(zeros(2)), TRLS.JacobianScaling(), J)
        @test D.diag[2] == 1
        # The LQ strategies factorize Jᵀ and must report rank 1, which routes the Gauss–Newton step
        # through the complete-orthogonal-decomposition branch.
        cache = TRLS.subproblem_cache_init(TRLS.LQStrategy(), TRLS.NoScaling(), J)
        @test TRLS.numerical_rank(cache.factorization) == 1
    end
end
