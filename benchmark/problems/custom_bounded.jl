using NLPModels
using LinearAlgebra
using ForwardDiff

"""
    KowalikOsborne()
    
Classic enzyme kinetics problem.
Dimensions: 4 variables, 11 residuals.
Bounds: 0 ≤ x ≤ Inf
"""
function KowalikOsborne()
    # Data
    t = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
    y = [
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

    # Residual Function F(x)
    function kowalik_residual(x)
        res = zeros(eltype(x), 11)
        for i = 1:11
            # Model: y = (x1 * (t^2 + x2 * t)) / (t^2 + x3 * t + x4)
            denom = t[i]^2 + x[3] * t[i] + x[4]
            pred = (x[1] * (t[i]^2 + x[2] * t[i])) / denom
            res[i] = pred - y[i]
        end
        return res
    end

    # Setup
    x0 = [0.25, 0.39, 0.41, 0.28]
    lvar = zeros(4)
    uvar = fill(Inf, 4)

    return ADNLSModel(kowalik_residual, x0, 11, lvar, uvar, name = "KowalikOsborne")
end

"""
    Meyer()
    
Notoriously stiff problem with a deep valley.
Dimensions: 3 variables, 16 residuals.
Bounds: 0 ≤ x ≤ Inf
"""
function Meyer()
    # Data
    t = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
    ]
    y = [
        34780.0,
        28610.0,
        23650.0,
        19630.0,
        16370.0,
        13720.0,
        11540.0,
        9744.0,
        8261.0,
        7030.0,
        6005.0,
        5147.0,
        4427.0,
        3820.0,
        3307.0,
        2872.0,
    ]

    # Residual Function
    function meyer_residual(x)
        res = zeros(eltype(x), 16)
        for i = 1:16
            # Model: y = x1 * exp( x2 / (t + x3) )
            pred = x[1] * exp(x[2] / (t[i] + x[3]))
            res[i] = pred - y[i]
        end
        return res
    end

    # Setup
    x0 = [0.02, 4000.0, 250.0]
    lvar = zeros(3)
    uvar = fill(Inf, 3)

    return ADNLSModel(meyer_residual, x0, 16, lvar, uvar, name = "Meyer")
end

"""
    Osborne1()
    
Exponential fitting problem.
Dimensions: 5 variables, 33 residuals.
Bounds: 0 ≤ x ≤ Inf (Note: x3 is often negative in unconstrained fit, but we bound it here)
"""
function Osborne1()
    # Data
    t = [
        0.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
        110.0,
        120.0,
        130.0,
        140.0,
        150.0,
        160.0,
        170.0,
        180.0,
        190.0,
        200.0,
        210.0,
        220.0,
        230.0,
        240.0,
        250.0,
        260.0,
        270.0,
        280.0,
        290.0,
        300.0,
        310.0,
        320.0,
    ]
    y = [
        0.844,
        0.908,
        0.866,
        0.931,
        0.934,
        0.972,
        1.037,
        1.045,
        1.021,
        0.848,
        0.861,
        1.130,
        1.094,
        1.116,
        1.124,
        1.101,
        1.081,
        1.048,
        1.009,
        0.949,
        0.881,
        0.876,
        0.918,
        0.961,
        1.003,
        1.033,
        1.050,
        1.041,
        1.019,
        0.973,
        0.962,
        0.950,
        0.927,
    ]

    # Residual Function
    function osborne1_residual(x)
        res = zeros(eltype(x), 33)
        for i = 1:33
            # Model: y = x1 + x2*exp(-x4*t) + x3*exp(-x5*t)
            pred = x[1] + x[2] * exp(-x[4] * t[i]) + x[3] * exp(-x[5] * t[i])
            res[i] = pred - y[i]
        end
        return res
    end

    # Setup
    x0 = [0.5, 1.5, -1.0, 0.01, 0.02]
    # Note: x3 is historically negative. If you want STRICTLY positive parameters 
    # (common in bounded benchmarks), set lvar[3]=0. If you want to allow the "real" fit:
    lvar = [0.0, 0.0, -Inf, 0.0, 0.0]
    uvar = fill(Inf, 5)

    return ADNLSModel(osborne1_residual, x0, 33, lvar, uvar, name = "Osborne1")
end

"""
    BoxBOD()
    
Stiff flat valley problem.
Dimensions: 2 variables, 6 residuals.
Bounds: 0 ≤ x ≤ Inf
"""
function BoxBOD()
    t = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    y = [109.0, 149.0, 149.0, 191.0, 213.0, 224.0]

    function boxbod_residual(x)
        res = zeros(eltype(x), 6)
        for i = 1:6
            # Model: y = exp(-x1 * t) - exp(-x2 * t)
            # Use 'pred' to avoid type instability
            term1 = exp(-x[1] * t[i])
            term2 = exp(-x[2] * t[i])
            res[i] = term1 - term2 - (y[i]/1.0) # Scale if needed, here pure residual
        end
        return res
    end

    x0 = [1.0, 1.0]
    lvar = [0.0, 0.0]
    uvar = [Inf, Inf]

    return ADNLSModel(boxbod_residual, x0, 6, lvar, uvar, name = "BoxBOD")
end

"""
    AlphaPinene()
    
Chemical kinetics using Matrix Exponential.
Dimensions: 5 variables, 40 residuals.
Bounds: 0 ≤ x ≤ Inf
"""
function AlphaPinene()
    # ... (Keep existing data definitions) ...
    times = [1230.0, 3060.0, 4920.0, 7800.0, 10680.0, 15030.0, 22620.0, 36420.0]
    y_obs = [
        88.35 7.3 2.3 0.4 1.75;
        76.4 15.6 4.5 0.7 2.8;
        65.1 23.1 5.3 1.1 5.8;
        50.4 32.9 6.0 1.5 9.3;
        37.5 42.7 6.0 1.9 12.0;
        25.9 49.1 5.9 2.2 17.0;
        14.0 57.4 5.1 2.6 21.0;
        4.5 63.1 3.8 2.9 25.7
    ]
    y0 = [100.0, 0.0, 0.0, 0.0, 0.0]

    function pinene_residual(theta)
        p1, p2, p3, p4, p5 = theta
        z = zero(p1)
        # Construct A with correct types (Duals)
        A = [
            -(p1+p2) z z z z;
            p1 z z z z;
            p2 z -(p3+p4) z p5;
            z z p3 z z;
            z z p4 z -p5
        ]

        res = Vector{eltype(theta)}()
        for i = 1:length(times)
            # CHANGE HERE: Use generic_mat_exp instead of exp
            y_pred = generic_mat_exp(A * times[i]) * y0

            append!(res, y_pred - y_obs[i, :])
        end
        return res
    end

    x0 = [5.84e-5, 2.65e-5, 1.63e-5, 2.77e-4, 4.61e-5]
    lvar = zeros(5)
    uvar = fill(Inf, 5)

    return ADNLSModel(pinene_residual, x0, 40, lvar, uvar, name = "AlphaPinene")
end

# ---------------------------------------------------------------------------------------
# Moré-Garbow-Hillstrom (MGH, 1981) analytic NLS problems with designed box constraints.
#
# These have closed-form residuals and KNOWN unconstrained minimizers, so the bounds can be
# chosen to be genuinely ACTIVE (the box clips the unconstrained optimum on >=1 coordinate)
# or REALISTIC (physically motivated, usually inactive). Each problem's docstring states the
# unconstrained minimizer x* and the bound regime. No data tables -> no transcription risk.
# ---------------------------------------------------------------------------------------

"""
    RosenbrockBounded()

MGH #1 (Rosenbrock). F = [10(x2 - x1^2), 1 - x1]; unconstrained min x* = (1, 1), f = 0.
Bounds: ACTIVE — uvar = [0.5, Inf] clips x1 (wants 1) to 0.5.
2 variables, 2 residuals.
"""
function RosenbrockBounded()
    function res(x)
        return [10.0 * (x[2] - x[1]^2), 1.0 - x[1]]
    end
    x0 = [-1.2, 1.0]
    lvar = [-2.0, -2.0]
    uvar = [0.5, 2.0]
    return ADNLSModel(res, x0, 2, lvar, uvar, name = "RosenbrockBounded")
end

"""
    BealeBounded()

MGH #5 (Beale). f_i = y_i - x1 (1 - x2^i), y = [1.5, 2.25, 2.625], i = 1,2,3.
Unconstrained min x* = (3, 0.5), f = 0. Bounds: ACTIVE — uvar = [2.0, 0.9] clips x1 (wants 3).
2 variables, 3 residuals.
"""
function BealeBounded()
    y = [1.5, 2.25, 2.625]
    function res(x)
        r = zeros(eltype(x), 3)
        for i = 1:3
            r[i] = y[i] - x[1] * (1.0 - x[2]^i)
        end
        return r
    end
    x0 = [1.0, 1.0]
    lvar = [0.0, 0.0]
    uvar = [2.0, 0.9]
    return ADNLSModel(res, x0, 3, lvar, uvar, name = "BealeBounded")
end

"""
    PowellBadlyScaled()

MGH #3 (Powell badly scaled). f1 = 1e4 x1 x2 - 1; f2 = exp(-x1) + exp(-x2) - 1.0001.
Unconstrained min x* ≈ (1.098e-5, 9.106), f = 0. Bounds: REALISTIC — non-negative (inactive).
2 variables, 2 residuals.
"""
function PowellBadlyScaled()
    function res(x)
        return [1.0e4 * x[1] * x[2] - 1.0, exp(-x[1]) + exp(-x[2]) - 1.0001]
    end
    x0 = [0.0, 1.0]
    lvar = [0.0, 0.0]
    uvar = [Inf, Inf]
    return ADNLSModel(res, x0, 2, lvar, uvar, name = "PowellBadlyScaled")
end

"""
    BrownBadlyScaled()

MGH #4 (Brown badly scaled). f1 = x1 - 1e6; f2 = x2 - 2e-6; f3 = x1 x2 - 2.
Unconstrained min x* = (1e6, 2e-6), f = 0. Bounds: ACTIVE — uvar = [1e5, Inf] clips x1.
2 variables, 3 residuals.
"""
function BrownBadlyScaled()
    function res(x)
        return [x[1] - 1.0e6, x[2] - 2.0e-6, x[1] * x[2] - 2.0]
    end
    x0 = [1.0, 1.0]
    lvar = [0.0, 0.0]
    uvar = [1.0e5, Inf]
    return ADNLSModel(res, x0, 3, lvar, uvar, name = "BrownBadlyScaled")
end

"""
    FreudensteinRoth()

MGH #2 (Freudenstein and Roth). f1 = -13 + x1 + ((5 - x2) x2 - 2) x2;
f2 = -29 + x1 + ((x2 + 1) x2 - 14) x2. Global min x* = (5, 4), f = 0 (interior).
Bounds: REALISTIC — non-negative box, optimum interior (inactive).
2 variables, 2 residuals.
"""
function FreudensteinRoth()
    function res(x)
        f1 = -13.0 + x[1] + ((5.0 - x[2]) * x[2] - 2.0) * x[2]
        f2 = -29.0 + x[1] + ((x[2] + 1.0) * x[2] - 14.0) * x[2]
        return [f1, f2]
    end
    x0 = [0.5, -2.0]
    lvar = [0.0, 0.0]
    uvar = [Inf, Inf]
    return ADNLSModel(res, x0, 2, lvar, uvar, name = "FreudensteinRoth")
end

"""
    HelicalValleyBounded()

MGH #7 (Helical valley). 3D helix; unconstrained min x* = (1, 0, 0), f = 0.
Bounds: ACTIVE — lvar = [2.0, -10, -10] forces x1 >= 2 (wants 1).
3 variables, 3 residuals.
"""
function HelicalValleyBounded()
    function res(x)
        # Two-argument atan gives the proper quadrant angle without a branch (the original
        # MGH x1>0/x1<0 split is exactly atan2), keeping it safe for AD/sparsity tracers.
        θ = atan(x[2], x[1]) / (2π)
        f1 = 10.0 * (x[3] - 10.0 * θ)
        f2 = 10.0 * (sqrt(x[1]^2 + x[2]^2) - 1.0)
        f3 = x[3]
        return [f1, f2, f3]
    end
    x0 = [-1.0, 0.0, 0.0]
    lvar = [2.0, -10.0, -10.0]
    uvar = [10.0, 10.0, 10.0]
    return ADNLSModel(res, x0, 3, lvar, uvar, name = "HelicalValleyBounded")
end

"""
    JennrichSampson()

MGH #6 (Jennrich and Sampson). f_i = 2 + 2 i - (exp(i x1) + exp(i x2)), i = 1..10.
Min f ≈ 124.362 at x1 = x2 ≈ 0.2578 (interior). Bounds: REALISTIC — non-negative (inactive).
2 variables, 10 residuals.
"""
function JennrichSampson()
    function res(x)
        r = zeros(eltype(x), 10)
        for i = 1:10
            r[i] = 2.0 + 2.0 * i - (exp(i * x[1]) + exp(i * x[2]))
        end
        return r
    end
    x0 = [0.3, 0.4]
    lvar = [0.0, 0.0]
    uvar = [Inf, Inf]
    return ADNLSModel(res, x0, 10, lvar, uvar, name = "JennrichSampson")
end

"""
    Box3DBounded()

MGH #12 (Box 3-D). f_i = exp(-t_i x1) - exp(-t_i x2) - x3 (exp(-t_i) - exp(-10 t_i)),
t_i = 0.1 i, i = 1..10. Unconstrained min x* = (1, 10, 1), f = 0.
Bounds: ACTIVE — uvar = [Inf, 5.0, Inf] clips x2 (wants 10).
3 variables, 10 residuals.
"""
function Box3DBounded()
    t = [0.1 * i for i = 1:10]
    function res(x)
        r = zeros(eltype(x), 10)
        for i = 1:10
            r[i] =
                exp(-t[i] * x[1]) - exp(-t[i] * x[2]) -
                x[3] * (exp(-t[i]) - exp(-10.0 * t[i]))
        end
        return r
    end
    x0 = [0.0, 10.0, 20.0]
    lvar = [0.0, 0.0, -10.0]
    uvar = [Inf, 5.0, Inf]
    return ADNLSModel(res, x0, 10, lvar, uvar, name = "Box3DBounded")
end

# ---------------------------------------------------------------------------------------
# NIST StRD nonlinear-regression problems (certified data + solutions, fetched verbatim
# from itl.nist.gov/div898/strd/nls). Real data-fitting problems with box constraints.
# ---------------------------------------------------------------------------------------

"""
    Misra1aBounded()

NIST StRD Misra1a. Model y = b1 (1 - exp(-b2 x)). Certified (b1, b2) = (238.942, 5.5016e-4),
residual sum of squares 0.12455 (so ‖F‖ ≈ 0.3529). 14 observations.
Bounds: ACTIVE — uvar = [200, Inf] clips b1 (certified 238.9 > 200).
2 variables, 14 residuals.
"""
function Misra1aBounded()
    x = [
        77.6,
        114.9,
        141.1,
        190.8,
        239.9,
        289.0,
        332.8,
        378.4,
        434.8,
        477.3,
        536.8,
        593.1,
        689.1,
        760.0,
    ]
    y = [
        10.07,
        14.73,
        17.94,
        23.93,
        29.61,
        35.18,
        40.02,
        44.82,
        50.76,
        55.05,
        61.01,
        66.40,
        75.47,
        81.78,
    ]
    function res(b)
        r = zeros(eltype(b), 14)
        for i = 1:14
            r[i] = b[1] * (1.0 - exp(-b[2] * x[i])) - y[i]
        end
        return r
    end
    x0 = [100.0, 2.0e-4]
    lvar = [0.0, 0.0]
    uvar = [200.0, Inf]
    return ADNLSModel(res, x0, 14, lvar, uvar, name = "Misra1aBounded")
end

"""
    Rat42Bounded()

NIST StRD Rat42 (sigmoidal growth). Model y = b1 / (1 + exp(b2 - b3 x)).
Certified (b1, b2, b3) = (72.462, 2.6181, 0.067359), RSS 8.0565 (so ‖F‖ ≈ 2.838). 9 obs.
Bounds: REALISTIC — b1, b3 >= 0 (both certified positive, so inactive).
3 variables, 9 residuals.
"""
function Rat42Bounded()
    x = [9.0, 14.0, 21.0, 28.0, 42.0, 57.0, 63.0, 70.0, 79.0]
    y = [8.93, 10.80, 18.59, 22.33, 39.35, 56.11, 61.73, 64.62, 67.08]
    function res(b)
        r = zeros(eltype(b), 9)
        for i = 1:9
            r[i] = b[1] / (1.0 + exp(b[2] - b[3] * x[i])) - y[i]
        end
        return r
    end
    x0 = [100.0, 1.0, 0.1]
    lvar = [0.0, -Inf, 0.0]
    uvar = [Inf, Inf, Inf]
    return ADNLSModel(res, x0, 9, lvar, uvar, name = "Rat42Bounded")
end

# Problems whose box is designed to be ACTIVE at the optimum (stress the projection logic).
function get_active_bound_problems()
    return [
        RosenbrockBounded,
        BealeBounded,
        BrownBadlyScaled,
        HelicalValleyBounded,
        Box3DBounded,
        Misra1aBounded,
    ]
end

# Problems with realistic (usually inactive) bounds.
function get_realistic_bound_problems()
    return [PowellBadlyScaled, FreudensteinRoth, JennrichSampson, Rat42Bounded]
end

# Helper to return all custom bounded problems as a list of constructor functions.
function get_custom_problems()
    return vcat(
        [KowalikOsborne, Meyer, Osborne1, BoxBOD, AlphaPinene],
        get_active_bound_problems(),
        get_realistic_bound_problems(),
    )
end

# Helper: Generic matrix exponential compatible with ForwardDiff and Tracers
function generic_mat_exp(A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    # 1. Scaling
    val_norm = maximum(abs, A)

    # Robust q calculation:
    # If T is a Tracer (symbolic), numerical functions like log2/ceil might fail 
    # or return non-integers. We catch this and default q=0.
    q = 0
    try
        computed_q = max(0, ceil(Int, log2(val_norm)))
        # Double check we got an actual integer to avoid loop errors later
        if computed_q isa Integer
            q = computed_q
        end
    catch
        # Fallback for symbolic types: q=0 is sufficient for sparsity detection
        # and numerically appropriate for AlphaPinene's small parameters.
        q = 0
    end

    A_scaled = A / (2^q)

    # 2. Taylor Series (Order 12)
    # Note: using I(n) requires LinearAlgebra to be loaded
    res = Matrix{T}(I, n, n)
    term = Matrix{T}(I, n, n)
    for k = 1:12
        term = term * A_scaled / k
        res += term
    end

    # 3. Squaring
    for _ = 1:q
        res = res * res
    end
    return res
end
