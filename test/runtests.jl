using nonlinearlstr
using Test

# Unit / correctness suite only. This is intentionally lightweight and portable
# (pure Julia + LinearAlgebra), so it runs on every OS in CI. The cross-package
# benchmarks live under ../benchmark with their own (heavy) environment and are
# run manually, not as part of `Pkg.test()`.
@testset "nonlinearlstr.jl" begin
    include("unit/jet.jl")
    include("unit/subproblems.jl")
    include("unit/colemanli.jl")
    include("unit/bounded.jl")
    include("unit/allocations.jl")
end
