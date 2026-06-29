abstract type SolverCache end
abstract type Strategy end
struct QRCholStrategy <: Strategy end

mutable struct QRCholCache{F,Fa,D,T} <: SolverCache
    factorization::F
    scaling_matrix::D
    factorization_aug::Fa

    p::Vector{T}             # Step direction 
    q::Vector{T}             # intermediate array for newton update
    b_aug::Vector{T}         # complicated solver righthandside
    D²p::Vector{T}
    D²::D
    JᵀJ::AbstractMatrix{T}

    function QRCholCache(
        strategy::S,
        scaling_strat::Sc,
        J::AbstractMatrix{T};
        kwargs...,
    ) where {S<:Strategy,Sc<:ScalingStrategy,T}

        n, m = size(J)
        J_buffer = copy(J)
        J_aug_buffer = Matrix(J'J + I)
        F = qr!(J_buffer, ColumnNorm())
        Fa = cholesky!(J_aug_buffer)
        D = one(T) .* I(m)
        scaling!(D, scaling_strat; kwargs...)

        p = zeros(T, m)
        q = zeros(T, m)
        b_aug = zeros(T, m)
        D²p = zeros(T, m)
        D² = D*D
        JᵀJ = J'J

        new{typeof(F),typeof(Fa),typeof(D),T}(F, D, Fa, p, q, b_aug, D²p, D², JᵀJ)
    end
end

function subproblem_cache_init(
    strat::S,
    scaling_strat::Sc,
    J::AbstractMatrix;
    kwargs...,
) where {S<:Strategy,Sc<:ScalingStrategy}
    if strat isa QRCholStrategy
        return QRCholCache(strat, scaling_strat, J; kwargs...)
    end
end

function update_cache!(cache::QRCholCache, J)
    copyto!(cache.factorization.factors, J)
    cache.factorization = qr!(cache.factorization.factors, ColumnNorm())
end
