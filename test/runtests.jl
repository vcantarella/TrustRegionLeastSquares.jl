using nonlinearlstr
using Test

# Unit / correctness suite: pure Julia + LinearAlgebra + ForwardDiff, so it runs on every OS in CI.
# The cross-package benchmarks live under ../benchmark with their own (heavy) environment.
#
# Each file opens its own @testset and is included at top level on purpose: wrapping these includes
# in one outer `@testset begin` puts every statement into a single inference unit and inflates
# compile time from seconds to tens of minutes.
include("problems.jl")
include("unit/jet.jl")
include("unit/subproblems.jl")
include("unit/hard_problems.jl")
include("unit/bounded.jl")
include("unit/allocations.jl")
