# Shared setup for the test items. `Problems` is a `@testmodule`, so it is evaluated once per test
# process and the solver specializations its `solve` triggers are shared by every item that uses it
# — a `@testsnippet` would splice these definitions into each item separately and recompile them.
#
# `MGH` is the Moré–Garbow–Hillstrom (1981) collection with the reference values from that paper:
# `Fstar` is Σ rᵢ² (twice the cost ½‖r‖² this package minimizes).

@testmodule Problems begin
    using ForwardDiff, LinearAlgebra
    import TrustRegionLeastSquares as TRLS


    "Turn an out-of-place residual `r(x)` into the in-place `(res!, jac!)` pair the solver takes."
    inplace(r) = ((f, x) -> (f .= r(x); f), (J, x) -> (ForwardDiff.jacobian!(J, r, x); J))

    "Keep only `rows` of the residual (underdetermined variants)."
    crop(r, rows) = x -> r(x)[rows]

    """
    Run a solve that may hit the Cholesky strategies' conditioning limit. Returns the result, or
    `nothing` when the normal matrix was numerically indefinite and `cholesky!` threw. Which of the two
    happens on a borderline problem depends on the LAPACK build, so tests must accept either.
    """
    function solve_or_indefinite(args...; kwargs...)
        try
            return solve(args...; kwargs...)
        catch err
            err isa PosDefException || rethrow()
            return nothing
        end
    end

    function solve(r, x0, strategy, scaling; kwargs...)
        res!, jac! = inplace(r)
        x, f, g, iter = TRLS.lm_trust_region!(
            res!,
            jac!,
            copy(x0),
            length(r(x0)),
            strategy,
            scaling;
            kwargs...,
        )
        return (; x, f, g, iter, cost = dot(f, f) / 2)
    end

    const BARD_Y = [
        0.14,
        0.18,
        0.22,
        0.25,
        0.29,
        0.32,
        0.35,
        0.39,
        0.37,
        0.58,
        0.73,
        0.96,
        1.34,
        2.10,
        4.39,
    ]
    const MEYER_Y = [
        34780,
        28610,
        23650,
        19630,
        16370,
        13720,
        11540,
        9744,
        8261,
        7030,
        6005,
        5147,
        4427,
        3820,
        3307,
        2872,
    ]
    const KO_Y = [
        0.1957,
        0.1947,
        0.1735,
        0.1600,
        0.0844,
        0.0627,
        0.0456,
        0.0342,
        0.0323,
        0.0235,
        0.0246,
    ]
    const KO_U = [4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625]
    const OSBORNE1_Y = [
        0.844,
        0.908,
        0.932,
        0.936,
        0.925,
        0.908,
        0.881,
        0.850,
        0.818,
        0.784,
        0.751,
        0.718,
        0.685,
        0.658,
        0.628,
        0.603,
        0.580,
        0.558,
        0.538,
        0.522,
        0.506,
        0.490,
        0.478,
        0.467,
        0.457,
        0.448,
        0.438,
        0.431,
        0.424,
        0.420,
        0.414,
        0.411,
        0.406,
    ]

    rosenbrock(x) = [10(x[2] - x[1]^2), 1 - x[1]]
    freudenstein_roth(x) = [
        -13 + x[1] + ((5 - x[2]) * x[2] - 2) * x[2],
        -29 + x[1] + ((x[2] + 1) * x[2] - 14) * x[2],
    ]
    powell_badly_scaled(x) = [1e4 * x[1] * x[2] - 1, exp(-x[1]) + exp(-x[2]) - 1.0001]
    brown_badly_scaled(x) = [x[1] - 1e6, x[2] - 2e-6, x[1] * x[2] - 2]
    beale(x) = [y - x[1] * (1 - x[2]^i) for (i, y) in enumerate((1.5, 2.25, 2.625))]
    jennrich_sampson(x) = [2 + 2i - (exp(i * x[1]) + exp(i * x[2])) for i = 1:10]
    function helical_valley(x)
        θ = atan(x[2] / x[1]) / (2π) + (x[1] < 0 ? 0.5 : 0.0)   # MGH's literal definition
        return [10(x[3] - 10θ), 10(sqrt(x[1]^2 + x[2]^2) - 1), x[3]]
    end
    bard(x) = [
        y - (x[1] + i / ((16 - i) * x[2] + min(i, 16 - i) * x[3])) for
        (i, y) in enumerate(BARD_Y)
    ]
    meyer(x) = [x[1] * exp(x[2] / (45 + 5i + x[3])) - y for (i, y) in enumerate(MEYER_Y)]
    box3d(x) =
        [exp(-0.1i * x[1]) - exp(-0.1i * x[2]) - x[3] * (exp(-0.1i) - exp(-i)) for i = 1:10]
    powell_singular(x) = [
        x[1] + 10x[2],
        sqrt(5) * (x[3] - x[4]),
        (x[2] - 2x[3])^2,
        sqrt(10) * (x[1] - x[4])^2,
    ]
    kowalik_osborne(x) = [
        y - x[1] * (u^2 + u * x[2]) / (u^2 + u * x[3] + x[4]) for (u, y) in zip(KO_U, KO_Y)
    ]
    osborne1(x) = [
        y - (x[1] + x[2] * exp(-10(i - 1) * x[4]) + x[3] * exp(-10(i - 1) * x[5])) for
        (i, y) in enumerate(OSBORNE1_Y)
    ]
    function watson(x)
        n = length(x)
        r = [
            sum((j - 1) * x[j] * t^(j - 2) for j = 2:n) -
            sum(x[j] * t^(j - 1) for j = 1:n)^2 - 1 for t in (1:29) ./ 29
        ]
        return [r; x[1]; x[2] - x[1]^2 - 1]
    end

    # (name, residual, x0, Fstar = Σr² at the reference minimizer). `nothing` Fstar: assert convergence only.
    const MGH = [
        (name = "Rosenbrock", r = rosenbrock, x0 = [-1.2, 1.0], Fstar = 0.0),
        (
            name = "Freudenstein-Roth",
            r = freudenstein_roth,
            x0 = [0.5, -2.0],
            Fstar = (0.0, 48.98425),
        ),
        (
            name = "Powell badly scaled",
            r = powell_badly_scaled,
            x0 = [0.0, 1.0],
            Fstar = 0.0,
        ),
        (name = "Brown badly scaled", r = brown_badly_scaled, x0 = [1.0, 1.0], Fstar = 0.0),
        (name = "Beale", r = beale, x0 = [1.0, 1.0], Fstar = 0.0),
        (name = "Jennrich-Sampson", r = jennrich_sampson, x0 = [0.3, 0.4], Fstar = 124.362),
        (name = "Helical valley", r = helical_valley, x0 = [-1.0, 0.0, 0.0], Fstar = 0.0),
        (name = "Bard", r = bard, x0 = [1.0, 1.0, 1.0], Fstar = 8.21487e-3),
        (name = "Meyer", r = meyer, x0 = [0.02, 4000.0, 250.0], Fstar = 87.9458),
        (name = "Box 3D", r = box3d, x0 = [0.0, 10.0, 20.0], Fstar = 0.0),
        (
            name = "Powell singular",
            r = powell_singular,
            x0 = [3.0, -1.0, 0.0, 1.0],
            Fstar = 0.0,
        ),
        (
            name = "Kowalik-Osborne",
            r = kowalik_osborne,
            x0 = [0.25, 0.39, 0.415, 0.39],
            Fstar = 3.07505e-4,
        ),
        (
            name = "Osborne 1",
            r = osborne1,
            x0 = [0.5, 1.5, -1.0, 0.01, 0.02],
            Fstar = 5.46489e-5,
        ),
        (name = "Watson 6", r = watson, x0 = zeros(6), Fstar = 2.28767e-3),
    ]

    const STRATEGIES =
        (TRLS.QRCholStrategy(), TRLS.QRStrategy(), TRLS.LQStrategy(), TRLS.LQCholStrategy())
    const SCALINGS = (TRLS.NoScaling(), TRLS.JacobianScaling())

    "`true` for the strategies that require rows <= cols."
    wide_only(strategy) = strategy isa Union{TRLS.LQStrategy,TRLS.LQCholStrategy}

    "Short display name of a strategy or scaling, for test-set titles."
    label(x) = string(nameof(typeof(x)))
end

# Spliced into each test item's scope: the things a body uses directly rather than through
# `Problems.`. Cheap to repeat, since none of it compiles anything.
@testsnippet Solver begin
    using LinearAlgebra, Random
    import TrustRegionLeastSquares as TRLS
end
