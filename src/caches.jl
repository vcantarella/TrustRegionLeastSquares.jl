abstract type Strategy end

"""
    QRCholStrategy()

Gauss–Newton step from a column-pivoted QR of `J`; damped steps `(JᵀJ + λD²) p = -Jᵀf` from a
Cholesky factorization of the normal matrix. Cheapest per λ-iteration, but the normal matrix squares
`cond(J)`, so accuracy degrades from about `cond(J) = 1e8` and beyond roughly `1e12` the matrix —
positive definite in exact arithmetic — is numerically indefinite and `cholesky!` throws a
`PosDefException`. Prefer [`QRStrategy`](@ref) for ill-conditioned problems. The default.
"""
struct QRCholStrategy <: Strategy end

"""
    QRStrategy()

Gauss–Newton step from a column-pivoted QR of `J`; damped steps from a QR of the augmented
matrix `[J; √λ D]` (Moré 1978). Numerically stable for any `cond(J)`. This is the variant
labelled "This work" in the benchmarks.
"""
struct QRStrategy <: Strategy end

"""
    LQStrategy()

For underdetermined problems (rows ≤ cols). The Gauss–Newton step is the minimum-norm solution,
from a column-pivoted QR of `Jᵀ` (complete orthogonal decomposition when rank-deficient);
damped steps from a QR of `[D⁻¹Jᵀ; √λ I]`.
"""
struct LQStrategy <: Strategy end

"""
    LQCholStrategy()

As [`LQStrategy`](@ref), with damped steps from a Cholesky factorization of the rows × rows
matrix `J D⁻² Jᵀ + λI`.
"""
struct LQCholStrategy <: Strategy end

abstract type SolverCache end

"""
    QRCache(strategy, scaling, J)

Workspace for `QRStrategy` / `QRCholStrategy`: the pivoted QR of `J` (refreshed in place on every
accepted step), the scaling matrix `D`, and the step `p` with its products.
"""
mutable struct QRCache{S<:Strategy,F,T} <: SolverCache
    factorization::F                         # qr(J, ColumnNorm())
    scaling_matrix::Diagonal{T,Vector{T}}    # D
    p::Vector{T}                             # step (cols)
    Dp::Vector{T}
    q::Vector{T}                             # workspace for dϕ/dλ (cols)
    Jp::Vector{T}                            # (rows)
end
function QRCache(::S, scaling::ScalingStrategy, J::AbstractMatrix{T}) where {S<:Strategy,T}
    n, m = size(J)
    D = scaling!(Diagonal(zeros(T, m)), scaling, J)
    F = qr!(Matrix(J), ColumnNorm())
    return QRCache{S,typeof(F),T}(F, D, zeros(T, m), zeros(T, m), zeros(T, m), zeros(T, n))
end

"""
    LQCache(strategy, scaling, J)

Workspace for `LQStrategy` / `LQCholStrategy` (rows ≤ cols): the pivoted QR of `Jᵀ`, so that
`J = P Rᵀ Qᵀ`, plus the row-length vectors `z`, `q` of the damped system.
"""
mutable struct LQCache{S<:Strategy,F,T} <: SolverCache
    factorization::F                         # qr(Jᵀ, ColumnNorm())
    scaling_matrix::Diagonal{T,Vector{T}}    # D
    p::Vector{T}                             # step (cols)
    Dp::Vector{T}
    z::Vector{T}                             # (J D⁻² Jᵀ + λI) z = -f  (rows)
    q::Vector{T}                             # workspace for dϕ/dλ (rows)
    Jp::Vector{T}                            # (rows)
end
function LQCache(::S, scaling::ScalingStrategy, J::AbstractMatrix{T}) where {S<:Strategy,T}
    n, m = size(J)
    n <= m || throw(ArgumentError("$S needs rows ≤ cols, got $(size(J))"))
    D = scaling!(Diagonal(zeros(T, m)), scaling, J)
    F = qr!(Matrix(J'), ColumnNorm())
    return LQCache{S,typeof(F),T}(
        F,
        D,
        zeros(T, m),
        zeros(T, m),
        zeros(T, n),
        zeros(T, n),
        zeros(T, n),
    )
end

subproblem_cache_init(s::Union{QRStrategy,QRCholStrategy}, scaling, J) =
    QRCache(s, scaling, J)
subproblem_cache_init(s::Union{LQStrategy,LQCholStrategy}, scaling, J) =
    LQCache(s, scaling, J)

"Refactorize the new Jacobian into the cached buffer and update the scaling matrix."
function update_cache!(cache::QRCache, J, scaling)
    copyto!(cache.factorization.factors, J)
    cache.factorization = qr!(cache.factorization.factors, ColumnNorm())
    scaling!(cache.scaling_matrix, scaling, J)
end
function update_cache!(cache::LQCache, J, scaling)
    copyto!(cache.factorization.factors, J')
    cache.factorization = qr!(cache.factorization.factors, ColumnNorm())
    scaling!(cache.scaling_matrix, scaling, J)
end
