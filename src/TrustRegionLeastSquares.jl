module TrustRegionLeastSquares
using LinearAlgebra
include("scaling.jl")
include("caches.jl")
include("subproblems.jl")
include("bounded.jl")
include("algorithms.jl")
export lm_trust_region!
end
