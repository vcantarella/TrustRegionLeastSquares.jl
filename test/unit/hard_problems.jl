using Test, LinearAlgebra, Random, ForwardDiff
isdefined(Main, :MGH) || include(joinpath(@__DIR__, "..", "problems.jl"))

# Moré–Garbow–Hillstrom problems from their standard starting points, every strategy × scaling.
# Fstar is Σr² at the reference minimizer. Stationarity is checked relative to ‖J‖‖f‖, because the
# problems differ in scale by many orders of magnitude (Meyer's residuals are O(1e4)) — except at a
# zero residual, where f → 0 forces g = Jᵀf → 0 and the cost itself is the stronger statement.
@testset "MGH $(problem.name): $(label(strategy)) / $(label(scaling))" for problem in MGH,
    strategy in STRATEGIES,
    scaling in SCALINGS

    n, m = length(problem.r(problem.x0)), length(problem.x0)
    wide_only(strategy) && n > m && continue
    s = solve(problem.r, problem.x0, strategy, scaling; gtol = 1e-10, max_iter = 500)
    Fstar = problem.Fstar
    @test any(
        F -> isapprox(2 * s.cost, F; rtol = 1e-5, atol = 1e-12),
        Fstar isa Tuple ? Fstar : (Fstar,),
    )
    J = ForwardDiff.jacobian(problem.r, s.x)
    @test s.cost < 1e-12 || norm(s.g) <= 1e-5 * opnorm(J) * norm(s.f)
end

@testset "underdetermined crops of MGH problems: $(name)" for (name, r, x0, rows) in (
    ("Bard", bard, [1.0, 1.0, 1.0], 1:2),
    ("Box 3D", box3d, [0.0, 10.0, 20.0], 1:2),
    ("Powell singular", powell_singular, [3.0, -1.0, 0.0, 1.0], 1:3),
    ("Osborne 1", osborne1, [0.5, 1.5, -1.0, 0.01, 0.02], 1:4),
    ("Watson 6", watson, zeros(6), 1:5),
)
    cropped = crop(r, rows)
    cost0 = sum(abs2, cropped(x0)) / 2
    for strategy in STRATEGIES, scaling in SCALINGS
        # Watson is ill-conditioned and Moré scaling pushes cond(JᵀJ) past 1/eps, so the Cholesky
        # strategies may fail outright here — documented in the QRCholStrategy docstring. Whether
        # they do depends on the LAPACK build; anything that does return must still find the
        # zero-residual solution.
        s = solve_or_indefinite(
            cropped,
            x0,
            strategy,
            scaling;
            gtol = 1e-12,
            max_iter = 500,
        )
        @test s === nothing ? name == "Watson 6" : s.cost < 1e-12 * max(1, cost0)
    end
end

@testset "linear underdetermined system: LQ family returns the minimum-norm solution" begin
    Random.seed!(11)
    A = randn(4, 10)
    b = randn(4)
    for strategy in (NL.LQStrategy(), NL.LQCholStrategy())
        s = solve(
            x -> A * x - b,
            zeros(10),
            strategy,
            NL.NoScaling();
            initial_radius = 1e3,
            norm_overrides_initial_radius = false,
        )
        @test s.x ≈ pinv(A) * b rtol = 1e-10
        @test s.iter == 1
    end
end
