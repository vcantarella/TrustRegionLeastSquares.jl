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
matrix `[J; √λ D]` (Moré 1978). Numerically stable for any `cond(J)`. This is the variant the
benchmarks report as `TRLS`.
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

# Every cache owns the factorization of `J` (or `Jᵀ`), refreshed in place on each accepted step, the
# scaling matrix `D`, the step `p` with its products, and the buffer the damped system of the
# λ-iteration is assembled and factorized in. Nothing in the λ-iteration allocates a matrix: the
# damped system is written into that buffer and factorized in place, which is why each strategy has
# its own cache rather than one shared type with unused fields.

"""
    QRCholCache(scaling, J)

Workspace for [`QRCholStrategy`](@ref). `JᵀJ` depends only on `J`, so it is formed once per accepted
step; `damped` receives `JᵀJ + λD²` and then its Cholesky factor on every λ-iteration.
"""
mutable struct QRCholCache{F,T} <: SolverCache
    factorization::F                         # qr(J, ColumnNorm())
    scaling_matrix::Diagonal{T,Vector{T}}    # D
    JᵀJ::Matrix{T}
    damped::Matrix{T}                        # JᵀJ + λD², overwritten by its Cholesky factor
    p::Vector{T}                             # step (cols)
    Dp::Vector{T}
    D²p::Vector{T}
    q::Vector{T}                             # workspace for dϕ/dλ (cols)
    Jᵀf::Vector{T}
    Jp::Vector{T}                            # (rows)
end
function QRCholCache(scaling::ScalingStrategy, J::AbstractMatrix{T}) where {T}
    n, m = size(J)
    D = scaling!(Diagonal(zeros(T, m)), scaling, J)
    F = qr!(Matrix(J), ColumnNorm())
    return QRCholCache(
        F,
        D,
        Matrix(J' * J),
        zeros(T, (m, m)),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, n),
    )
end

"""
    QRCache(scaling, J)

Workspace for [`QRStrategy`](@ref). `augmented` holds `[J; √λ D]` and `rhs` the matching
`[-f; 0]`; `qr!` consumes `augmented`, so its blocks are rewritten on every λ-iteration.
"""
mutable struct QRCache{F,T} <: SolverCache
    factorization::F                         # qr(J, ColumnNorm())
    scaling_matrix::Diagonal{T,Vector{T}}    # D
    augmented::Matrix{T}                     # [J; √λ D], (rows + cols) × cols
    rhs::Vector{T}                           # [-f; 0]
    p::Vector{T}                             # step (cols)
    Dp::Vector{T}
    D²p::Vector{T}
    q::Vector{T}                             # workspace for dϕ/dλ (cols)
    Jᵀf::Vector{T}
    Jp::Vector{T}                            # (rows)
end
function QRCache(scaling::ScalingStrategy, J::AbstractMatrix{T}) where {T}
    n, m = size(J)
    D = scaling!(Diagonal(zeros(T, m)), scaling, J)
    F = qr!(Matrix(J), ColumnNorm())
    return QRCache(
        F,
        D,
        zeros(T, (n + m, m)),
        zeros(T, n + m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, n),
    )
end

"""
    LQCache(scaling, J)

Workspace for [`LQStrategy`](@ref) (rows ≤ cols): the pivoted QR of `Jᵀ`, so that `J = P Rᵀ Qᵀ`,
`D⁻¹Jᵀ` (which depends on `D` and so is reformed on every solve), and `augmented` holding
`[D⁻¹Jᵀ; √λ I]`. `z` and `q` are the row-length vectors of the damped system.
"""
mutable struct LQCache{F,T} <: SolverCache
    factorization::F                         # qr(Jᵀ, ColumnNorm())
    scaling_matrix::Diagonal{T,Vector{T}}    # D
    D⁻¹Jᵀ::Matrix{T}                         # cols × rows
    augmented::Matrix{T}                     # [D⁻¹Jᵀ; √λ I], (cols + rows) × rows
    p::Vector{T}                             # step (cols)
    Dp::Vector{T}
    z::Vector{T}                             # (J D⁻² Jᵀ + λI) z = -f  (rows)
    q::Vector{T}                             # workspace for dϕ/dλ (rows)
    Jᵀf::Vector{T}                           # (cols)
    Jp::Vector{T}                            # (rows)
end
function LQCache(scaling::ScalingStrategy, J::AbstractMatrix{T}) where {T}
    n, m = size(J)
    n <= m || throw(ArgumentError("LQStrategy needs rows ≤ cols, got $(size(J))"))
    D = scaling!(Diagonal(zeros(T, m)), scaling, J)
    F = qr!(Matrix(J'), ColumnNorm())
    return LQCache(
        F,
        D,
        zeros(T, (m, n)),
        zeros(T, (m + n, n)),
        zeros(T, m),
        zeros(T, m),
        zeros(T, n),
        zeros(T, n),
        zeros(T, m),
        zeros(T, n),
    )
end

"""
    LQCholCache(scaling, J)

Workspace for [`LQCholStrategy`](@ref) (rows ≤ cols). `gram` holds `J D⁻² Jᵀ`, reformed on every
solve because it depends on `D`; `damped` receives `gram + λI` and then its Cholesky factor.
"""
mutable struct LQCholCache{F,T} <: SolverCache
    factorization::F                         # qr(Jᵀ, ColumnNorm())
    scaling_matrix::Diagonal{T,Vector{T}}    # D
    D⁻²Jᵀ::Matrix{T}                         # cols × rows
    gram::Matrix{T}                          # J D⁻² Jᵀ, rows × rows
    damped::Matrix{T}                        # gram + λI, overwritten by its Cholesky factor
    p::Vector{T}                             # step (cols)
    Dp::Vector{T}
    z::Vector{T}                             # (J D⁻² Jᵀ + λI) z = -f  (rows)
    q::Vector{T}                             # workspace for dϕ/dλ (rows)
    Jᵀf::Vector{T}                           # (cols)
    Jp::Vector{T}                            # (rows)
end
function LQCholCache(scaling::ScalingStrategy, J::AbstractMatrix{T}) where {T}
    n, m = size(J)
    n <= m || throw(ArgumentError("LQCholStrategy needs rows ≤ cols, got $(size(J))"))
    D = scaling!(Diagonal(zeros(T, m)), scaling, J)
    F = qr!(Matrix(J'), ColumnNorm())
    return LQCholCache(
        F,
        D,
        zeros(T, (m, n)),
        zeros(T, (n, n)),
        zeros(T, (n, n)),
        zeros(T, m),
        zeros(T, m),
        zeros(T, n),
        zeros(T, n),
        zeros(T, m),
        zeros(T, n),
    )
end

subproblem_cache_init(::QRCholStrategy, scaling, J) = QRCholCache(scaling, J)
subproblem_cache_init(::QRStrategy, scaling, J) = QRCache(scaling, J)
subproblem_cache_init(::LQStrategy, scaling, J) = LQCache(scaling, J)
subproblem_cache_init(::LQCholStrategy, scaling, J) = LQCholCache(scaling, J)

"Refactorize the new Jacobian into the cached buffer and update the scaling matrix."
function update_cache!(cache::Union{QRCholCache,QRCache}, J, scaling)
    copyto!(cache.factorization.factors, J)
    cache.factorization = qr!(cache.factorization.factors, ColumnNorm())
    cache isa QRCholCache && mul!(cache.JᵀJ, J', J)   # depends on J alone, not on D
    return scaling!(cache.scaling_matrix, scaling, J)
end
function update_cache!(cache::Union{LQCache,LQCholCache}, J, scaling)
    copyto!(cache.factorization.factors, J')
    cache.factorization = qr!(cache.factorization.factors, ColumnNorm())
    return scaling!(cache.scaling_matrix, scaling, J)
end
