using Profile
using nonlinearlstr
using LinearAlgebra
using StaticArrays

# Setup problem
n = 50
m = 100
x0 = ones(n)

function res(x)
    r = zeros(m)
    r[1:n] .= x .- 1.0
    r[(n+1):end] .= 0.1
    return r
end

function jac(x)
    J = zeros(m, n)
    J[1:n, 1:n] .= I(n)
    return J
end

x = copy(x0)
f = res(x)
J = jac(x)
radius = 0.1

println("Profiling LM-QR...")
strategy = nonlinearlstr.QRSolve()
scaling = nonlinearlstr.NoScaling()
cache = nonlinearlstr.SubproblemCache(strategy, scaling, J)

# Warmup
nonlinearlstr.solve_subproblem(strategy, J, f, radius, cache)

Profile.clear()
@profile for i = 1:1000
    nonlinearlstr.solve_subproblem(strategy, J, f, radius, cache)
end

Profile.print(format = :flat, sortedby = :count, mincount = 50)
