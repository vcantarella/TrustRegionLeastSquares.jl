using LinearAlgebra

"""
    solve_gauss_newton(J, r)

Solve the Gauss-Newton step: J^T J δ = -J^T r using QR factorization.
This is more stable than forming J^T J explicitly.
If J is rank-deficient, we use the SVD to solve the system.
"""
function solve_gauss_newton(J, r::AbstractVector)
    hard_case = false
    if rank(J) < size(J, 2)
        # Use SVD for rank-deficient case
        δ = svd(J)\-r
        hard_case = true
    else
        δ = J \ -r  # Solve J δ = -r directly for full rank (implemented as QR by default)
    end
    return δ, hard_case
end












"""
    solve_gauss_newton_v2(J, r)

Solves the Gauss-Newton step. If J is rank-deficient, it computes the
minimal-norm solution and returns a vector from the null space of J
to handle the "hard case". This version determines rank from the SVD itself
for efficiency and robustness.
"""
function solve_gauss_newton_v2(J, r::AbstractVector)
    # 1. Compute the SVD factorization once. This is our single source of truth.
    F = svd(J)

    # 2. Compute the minimal-norm Gauss-Newton step using the SVD object.
    δ = F \ -r

    # 3. Determine the effective rank from the singular values.
    # A robust tolerance for identifying "zero" singular values.
    # F.S[1] is the largest singular value. If the matrix is all zeros,
    # F.S[1] will be zero, and eps(0.0) is a very small number, which is fine.
    tol = maximum(size(J)) * eps(F.S[1])

    # Count how many singular values are greater than the tolerance.
    # This is the effective rank of the matrix J.
    effective_rank = count(s -> s > tol, F.S)

    # 4. Check for rank deficiency (the hard case condition).
    # Does a non-trivial null space exist?
    if effective_rank < size(J, 2)
        # Yes, the matrix is rank-deficient. A null space exists.
        # The first column in V after the effective rank is a null-space vector.
        # This index is now guaranteed to be valid and in-bounds because
        # effective_rank is strictly less than the number of columns.
        null_space_idx = effective_rank + 1
        z = F.V[:, null_space_idx]

        return δ, z, true

    else
        # No, the matrix is full rank. No null space exists.
        return δ, nothing, false
    end
end

function solve_gauss_newton_v3(J, r::AbstractVector)
    # This function now correctly handles both tall (m>=n) and wide (m<n) matrices.
    F = svd(J)
    δ = F \ -r

    # Determine rank deficiency for the m >= n case.
    # For m < n, the matrix is always "rank-deficient" in the sense of having a null space.
    is_rank_deficient = false
    if size(J, 1) >= size(J, 2)
        tol = maximum(size(J)) * eps(F.S[1])
        effective_rank = count(s -> s > tol, F.S)
        if effective_rank < size(J, 2)
            is_rank_deficient = true
        end
    end

    # Return the SVD factorization as well, to avoid re-computing it.
    return δ, is_rank_deficient, F
end

"""
    compute_gradient(J, r::AbstractVector)

Compute gradient J^T r of the loss function L(x) = 1/2 ||r||^2.
"""
function compute_gradient(J, r::AbstractVector)
    return J' * r
end

"""
    compute_hessian_approximation(qrls::QRPivoted)

Compute Gauss-Newton Hessian approximation H ≈ J^T J using QR factorization.
TODO: Check if we need to correct for pivoting in the QR factorization.
"""
function compute_hessian_approximation(qrls::QRPivoted)
    return qrls.R' * qrls.R
end

using LinearAlgebra

"""
    lsq_box(A::AbstractMatrix, b::AbstractVector, lo::AbstractVector, hi::AbstractVector)

Return the least-square solution `x̂` minimizing `‖b - Ax‖₂` subject to the
box constraints `lo ≤ x ≤ hi` (elementwise).  (An upper/lower bound may
be `±Inf`, respectively, to remove a constraint.)
Source: stevengj (https://discourse.julialang.org/t/suggestions-needed-for-bound-constrained-least-squares-solver/35611/13)
"""
function lsq_box(
    A::AbstractMatrix{<:Number},
    b::AbstractVector{<:Number},
    lo::AbstractVector{<:Number},
    hi::AbstractVector{<:Number};
    maxiter::Integer = 100,
    rtol::Real = eps(float(eltype(A)))*sqrt(size(A, 2)),
    atol::Real = 0,
)
    Base.require_one_based_indexing(A, b, lo, hi)
    x = A \ b
    @. x = clamp(x, lo, hi)
    #AᵀA = A'*A
    Aᵀb = A'b
    g = A'*(A*x)
    g .-= Aᵀb # gradient ∇ₓ of ½‖b - Ax‖²
    inactive = Bool[
        lo[i] < hi[i] && (x[i] != lo[i] || g[i] ≤ 0) && (x[i] != hi[i] || g[i] ≥ 0) for
        i in eachindex(x)
    ]
    all(inactive) && return x
    active = map(!, inactive)
    xprev = copy(x)
    for iter = 1:maxiter
        xa = A[:, inactive] \ (b - A[:, active]*x[active])
        x[inactive] = xa
        @. x = clamp(x, lo, hi)
        g .= mul!(g, A', A*x) .- Aᵀb
        for i in eachindex(x)
            inactive[i] =
                lo[i] < hi[i] && (x[i] != lo[i] || g[i] ≤ 0) && (x[i] != hi[i] || g[i] ≥ 0)
        end
        all(i -> inactive[i] == !active[i], eachindex(active)) && return x # convergence: active set unchanged 
        norm(x - xprev) ≤ max(rtol*norm(x), atol) && return x # convergence: x not changing much
        xprev .= x
        @. active = !inactive
    end
    #error("convergence failure: $maxiter iterations reached")
    println("Warning!: convergence failure: $maxiter iterations reached")
    return x
end

# """
#     qr_trust_region_step(qrls::QRLeastSquares, residuals::AbstractVector, 
#                          radius::Real, lb::AbstractVector, ub::AbstractVector,
#                          x::AbstractVector; λ::Real = 0.0)

# Solve trust region subproblem using QR factorization:
# min_δ  0.5 * ||r + J*δ||² + 0.5*λ*||δ||²
# s.t.   ||δ|| ≤ radius
#        lb ≤ x + δ ≤ ub
# """
# function qr_trust_region_step(J, qrls::QRPivoted, residuals::AbstractVector, 
#                               radius::Real, lb::AbstractVector, ub::AbstractVector,
#                               x::AbstractVector; λ::Real = 0.0)
#     # Augmented system for regularization:
#     # [J]     [δ]   = [-r]
#     # [√λ*I]         [0]

#     if λ > 0
#         # Create augmented QR system
#         m, n = size(J)
#         J_aug = [qrls.Q * qrls.R; sqrt(λ) * I(n)]
#         r_aug = [residuals; zeros(n)]
#         # F_aug = qr(J_aug, ColumnNorm())
#         # Q_aug = F_aug.Q
#         # R_aug = UpperTriangular(F_aug.R)

#         # # Solve augmented system
#         # rhs = - (Q_aug' * r_aug)
#         δ_gn = J_aug \ (-r_aug)
#         #δ_gn = δ_gn[qrls.p]  # Apply permutation
#     else
#         δ_gn = solve_gauss_newton(J, qrls, residuals)
#     end

#     # Check bounds and trust region constraints
#     δ_bounded = clamp.(δ_gn, lb - x, ub - x)
#     # if norm(δ_bounded) <= radius
#     #     return δ_bounded
#     # else
#     #     # Need to solve constrained trust region problem
#     #     # This is a simplified version - you might want to use your tcg solver here
#     #     δ_tr = radius * δ_gn / norm(δ_gn)
#     #     return clamp.(δ_tr, lb - x, ub - x)
#     # end
#     return δ_bounded
# end

# """
#     levenberg_marquardt_step(qrls::QRLeastSquares, residuals::AbstractVector, λ::Real)

# Compute Levenberg-Marquardt step: (J^T J + λI) δ = -J^T r using QR factorization.
# """
# function levenberg_marquardt_step(qrls::QRLeastSquares{T}, residuals::AbstractVector, λ::Real) where T
#     # Solve (J^T J + λI) δ = -J^T r
#     # Using augmented system approach:
#     # [J]     [δ]   = [-r]
#     # [√λ*I]         [0]

#     m, n = qrls.m, qrls.n
#     J_aug = [qrls.Q * qrls.R; sqrt(λ) * I(n)]
#     r_aug = [residuals; zeros(T, n)]

#     # QR factorization of augmented system
#     F_aug = qr(J_aug)
#     Q_aug = Matrix(F_aug.Q)
#     R_aug = Matrix(F_aug.R)

#     # Solve
#     rhs = -Q_aug' * r_aug
#     δ = R_aug \ rhs

#     return δ
# end

"""
    condition_number(qrls::QRLeastSquares)

Estimate condition number of J^T J from QR factorization.
"""
function condition_number(qrls::QRPivoted, τ::Real = 1e-12)
    r_diag = abs.(diag(qrls.R))
    return maximum(r_diag) / minimum(r_diag[r_diag .> τ])
end
