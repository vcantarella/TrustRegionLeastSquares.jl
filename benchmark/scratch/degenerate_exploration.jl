using NonlinearSolve
import TrustRegionLeastSquares as TRLS


function f!(F, x, p)
    F[1] = x[1] - 1.0
    F[2] = x[1]^2 - 4.0
end

function j!(J, x, p)
    J[1, 1] = 1.0
    J[2, 1] = 2.0 * x[1]
    J[1, 2] = 1e-200
    J[2, 2] = 1e-200
end

x0 = [0.5, 0.5]
prob = NonlinearProblem(NonlinearFunction(f!; jac = j!), x0, nothing)

sol_tr = solve(prob, TrustRegion(), show_trace = Val(true), trace_level = TraceAll(10))       # Stalled, no NaN
sol_lm = solve(
    prob,
    NonlinearSolve.LevenbergMarquardt(),
    show_trace = Val(true),
    trace_level = TraceAll(10),
) # MaxIters, no NaN


function f!(F, x)
    F[1] = x[1] - 1.0
    F[2] = x[1]^2 - 4.0
end

function j!(J, x)
    J[1, 1] = 1.0
    J[2, 1] = 2.0 * x[1]
    J[1, 2] = 0.0
    J[2, 2] = 0.0
end
sol_nlstr = TRLS.lm_trust_region!(
    f!,
    j!,
    x0,
    2,
    TRLS.LQStrategy();
    verbose = true,
)

J = zeros(2, 2)
using SparseArrays
j!(J, x0)

Jsp = sparse(J)
