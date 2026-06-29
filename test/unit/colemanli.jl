using LinearAlgebra
using Test
using nonlinearlstr

# Test the zero matrix for infinite bounds:
n = 10
x = randn(n)
J = randn(12, n)
lb = fill(-Inf, n)
ub = fill(Inf, n)
g = randn(n)
@testset "Bounded Scaling" begin
    @testset "Coleman & Li Scaling" begin
        strat = nonlinearlstr.ColemanandLiScaling()
        D, Jv, v = nonlinearlstr.scaling(strat, J; x, lb, ub, g, τ = 1e-16)
        @test D == I(n)
        @test Jv == zeros(n, n)
    end
end
