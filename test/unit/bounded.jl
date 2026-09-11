using Test, LinearAlgebra, Random
isdefined(Main, :MGH) || include(joinpath(@__DIR__, "..", "problems.jl"))

feasible(x, lb, ub) = all(lb .<= x .<= ub)
# KKT for bounds: each variable is either interior with g_i ≈ 0, or on a bound with the gradient pushing outwards.
kkt(x, g, lb, ub; tol = 1e-6) = all(
    i ->
        (x[i] == lb[i] && g[i] >= -tol) ||
        (x[i] == ub[i] && g[i] <= tol) ||
        abs(g[i]) < tol,
    eachindex(x),
)

@testset "linear residual x − target: optimum is the projection of target onto the box" begin
    x0 = fill(0.1, 3)
    for (target, lb, ub) in (
            ([2.0, -1.5, 0.7], fill(-Inf, 3), fill(Inf, 3)),      # infinite bounds: unconstrained optimum
            ([2.0, 2.0, 2.0], fill(-Inf, 3), fill(1.0, 3)),        # active upper bound
            ([-2.0, -2.0, -2.0], fill(-1.0, 3), fill(Inf, 3)),     # active lower bound
            ([5.0, 0.25, -5.0], fill(-1.0, 3), fill(1.0, 3)),      # mixed: two active, one interior
        ),
        strategy in STRATEGIES                                   # J = I is square: every strategy applies

        res!, jac! = inplace(x -> x .- target)
        x, f, g, iter =
            TRLS.lm_trust_region!(res!, jac!, x0, 3, strategy; lb, ub, gtol = 1e-10)
        @test x ≈ clamp.(target, lb, ub) atol = 1e-8
        @test feasible(x, lb, ub)
        @test kkt(x, g, lb, ub)
    end
end

@testset "Rosenbrock with x1 ≤ 0.5: $(label(strategy)) / $(label(scaling))" for strategy in
                                                                                STRATEGIES,
    scaling in SCALINGS
    # f1 = 0 forces x2 = x1², then 0.5(1 − x1)² is minimized by pushing x1 to its bound: x* = (0.5, 0.25).
    res!, jac! = inplace(rosenbrock)
    lb, ub = [-2.0, -2.0], [0.5, 2.0]
    x, f, g, iter = TRLS.lm_trust_region!(
        res!,
        jac!,
        [-1.2, 1.0],
        2,
        strategy,
        scaling;
        lb,
        ub,
        gtol = 1e-8,
    )
    @test x ≈ [0.5, 0.25] atol = 1e-6
    @test x[1] == ub[1]                                        # the bound genuinely binds
    @test feasible(x, lb, ub) && kkt(x, g, lb, ub)
    @test iter < 50
end

@testset "bound handling: fixed variable, infeasible start, one-sided, interior" begin
    res!, jac! = inplace(rosenbrock)
    # fixed variable lb == ub: x2 stays put, x1 minimizes (1−x1)² + 100(0.3−x1²)²
    x, f, g, iter = TRLS.lm_trust_region!(
        res!,
        jac!,
        [-1.2, 1.0],
        2;
        lb = [-2.0, 0.3],
        ub = [2.0, 0.3],
        gtol = 1e-8,
    )
    @test x[2] == 0.3 && abs(g[1]) < 1e-6
    # infeasible x0 is clamped into the box before the first step
    x, f, g, iter =
        TRLS.lm_trust_region!(res!, jac!, [5.0, 5.0], 2; ub = [0.5, 1.0], gtol = 1e-8)
    @test x ≈ [0.5, 0.25] atol = 1e-6
    @test x[1] == 0.5
    # interior optimum: same minimizer as the unconstrained run
    x, f, g, iter = TRLS.lm_trust_region!(
        res!,
        jac!,
        [-1.2, 1.0],
        2;
        lb = [-10.0, -10.0],
        ub = [10.0, 10.0],
        gtol = 1e-10,
    )
    @test x ≈ [1.0, 1.0] atol = 1e-8
    # one-sided bounds cutting Beale's optimum (3, 0.5)
    res!, jac! = inplace(beale)
    x, f, g, iter =
        TRLS.lm_trust_region!(res!, jac!, [1.0, 1.0], 3; ub = [2.0, Inf], gtol = 1e-8)
    @test x[1] == 2.0 && kkt(x, g, [-Inf, -Inf], [2.0, Inf])
    x, f, g, iter =
        TRLS.lm_trust_region!(res!, jac!, [1.0, 1.0], 3; lb = [-Inf, 0.6], gtol = 1e-8)
    @test x[2] == 0.6 && kkt(x, g, [-Inf, 0.6], [Inf, Inf])
    @test_throws ArgumentError TRLS.lm_trust_region!(
        res!,
        jac!,
        [1.0, 1.0],
        3;
        lb = [1.0, 0.0],
        ub = [0.0, 1.0],
    )
end

@testset "Coleman–Li distances" begin
    x, g = fill(0.5, 4), [-1.0, 1.0, -1.0, 1.0]
    lb, ub = [0.0, 0.2, 0.0, -Inf], [1.0, 1.0, Inf, Inf]
    @test TRLS.coleman_li_distances!(zeros(4), x, g, lb, ub) == [0.5, 0.3, 1.0, 1.0]   # ub − x, x − lb, ∞ bound, ∞ bound
    @test TRLS.coleman_li_distances!(zeros(4), [1.0, 0.2, 0.5, 0.5], g, lb, ub)[1:2] ==
          [0.0, 0.0]   # on the bound, pushing outwards
end

@testset "projected gradient norm" begin
    Random.seed!(5)
    g, x = randn(6), 1e8 * randn(6)
    @test TRLS.projected_gradient_norm(g, x, fill(-Inf, 6), fill(Inf, 6)) ≈ norm(g) rtol =
        1e-15   # |x| ≫ |g|: no cancellation
    @test TRLS.projected_gradient_norm([-3.0, 2.0], [1.0, 0.0], [-Inf, 0.0], [1.0, Inf]) == 0        # on bounds, gradient outwards
    @test TRLS.projected_gradient_norm([-3.0, 2.0], [0.9, 0.5], [-Inf, 0.0], [1.0, Inf]) ≈
          sqrt(0.1^2 + 0.5^2)
end

@testset "Cauchy step: ω is limited by the model, the trust region, or the box" begin
    Random.seed!(8)
    n, m = 6, 4
    J = randn(n, m)
    lb, ub = [0.0, 0.0, -Inf, -Inf], [1.0, 1.0, Inf, Inf]
    x = [0.9, 0.5, 0.5, 0.5]
    cache = TRLS.BoundedCache(TRLS.subproblem_cache_init(TRLS.QRStrategy(), TRLS.NoScaling(), J))
    model_decrease(p, g) = -dot(g, p) - sum(abs2, J * p) / 2

    # The three candidates of MMP eq. 8, written out independently of the implementation.
    function candidate_ω(x, g, Δ, lb, ub)
        D = TRLS.affine_scaling!(cache, x, g, lb, ub)
        d = -TRLS.coleman_li_distances!(zeros(m), x, g, lb, ub) .* g
        box = minimum(
            d[i] > 0 ? (ub[i] - x[i]) / d[i] : d[i] < 0 ? (lb[i] - x[i]) / d[i] : Inf
            for i = 1:m
        )
        return (model = -dot(g, d) / sum(abs2, J * d), region = Δ / norm(D * d), box = box),
        d,
        D
    end

    @testset "pC = ω d with ω the smallest of the three candidates: $name" for (
        name,
        f,
        Δ,
    ) in (
        ("model-limited", randn(n), 1e6),            # large Δ, moderate gradient
        ("region-limited", randn(n), 1e-3),          # small Δ
        ("box-limited", 1e6 * randn(n), 1e6),        # the ray–box limit is 1/|gⱼ|, so a large gradient binds it
    )
        g = J' * f
        ω, d, D = candidate_ω(x, g, Δ, lb, ub)
        decrease = TRLS.cauchy_step!(cache, J, g, x, Δ, lb, ub)
        ω_min = min(ω.model, ω.region, ω.box)
        @test ω_min == getproperty(ω, Symbol(split(name, "-")[1]))      # the intended candidate is the binding one
        @test cache.pC ≈ ω_min * d rtol = 1e-12
        @test decrease ≈ ω_min * (-dot(g, d)) - ω_min^2 / 2 * sum(abs2, J * d) rtol = 1e-12
        @test decrease ≈ model_decrease(cache.pC, g) rtol = 1e-10
        @test decrease > 0
        @test norm(D * cache.pC) <= Δ * (1 + 1e-12)
        @test feasible(x + cache.pC, lb, ub)
    end

    @testset "a KKT point gives a zero step" begin
        # x on its bounds with the gradient pushing outwards: |v| = 0, so d = 0 and there is no decrease.
        g, x_kkt = [-1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.5, 0.5]
        TRLS.affine_scaling!(cache, x_kkt, g, lb, ub)
        @test TRLS.cauchy_step!(cache, J, g, x_kkt, 1.0, lb, ub) == 0
    end
end

@testset "Cauchy-fraction safeguard" begin
    Random.seed!(8)
    n, m = 6, 4
    J, f = randn(n, m), randn(n)
    x, lb, ub = [0.9, 0.5, 0.5, 0.5], [0.0, 0.0, -Inf, -Inf], [1.0, 1.0, 1.0, Inf]
    g = J' * f
    cache = TRLS.BoundedCache(TRLS.subproblem_cache_init(TRLS.QRStrategy(), TRLS.NoScaling(), J))
    TRLS.affine_scaling!(cache, x, g, lb, ub)
    model_decrease(p) = -dot(g, p) - sum(abs2, J * p) / 2
    decrease_cauchy = TRLS.cauchy_step!(cache, J, g, x, 1e3, lb, ub)

    # An ascent step must be pulled towards pC until the fraction β₁ holds with equality.
    cache.inner.p .= 0.1 * g
    predicted = TRLS.projected_step!(cache, J, g, x, lb, ub, decrease_cauchy; β₁ = 0.1)
    @test predicted ≈ 0.1 * decrease_cauchy rtol = 1e-10
    @test model_decrease(cache.p) ≈ predicted rtol = 1e-10
    @test feasible(x + cache.p, lb, ub)
    # A step that already beats the fraction is only projected onto the box.
    cache.inner.p .= 0.5 * cache.pC
    predicted = TRLS.projected_step!(cache, J, g, x, lb, ub, decrease_cauchy; β₁ = 0.1)
    @test cache.p ≈ clamp.(x .+ 0.5 * cache.pC, lb, ub) .- x
    @test predicted ≈ model_decrease(cache.p) rtol = 1e-12
    @test predicted >= 0.1 * decrease_cauchy
end
