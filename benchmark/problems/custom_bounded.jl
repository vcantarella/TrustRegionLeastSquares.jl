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

# Helper to return all new problems as a list of functions
function get_custom_problems()
    return [KowalikOsborne, Meyer, Osborne1, BoxBOD, AlphaPinene]
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
