
function solve_subproblem(
    J::AbstractMatrix{T},
    f::AbstractVector{T},
    Δ::Real,
    cache::QRCache;
    maxiters = 6,
    θ = 1e-4,
) where {T<:Real}
    F = cache.factorization
    D = cache.scaling_matrix

    p = cache.p
    ldiv!(p, F, -f)
    if norm(p) <= radius
        λ = zero(T)
        return λ
    else
        l₀ = 0.0
        u₀ = norm((J'f) ./ diag(D))/Δ
        λ₀ = max(1e-3*u₀, √(l₀*u₀))
        λ = λ₀
        uₖ = u₀
        lₖ = l₀
        b_aug = cache.b_aug
        Dp = cache.Dp
        q = cache.q
        b_aug .= [-f; zeros(size(J, 2))]
        for i = 1:maxiters
            factorize_aug!(cache, [J; √λ*D])
            Fa = cache.factorization_aug
            ldiv!(p, Fa, b_aug)
            mul!(Dp, D, p)
            norm_Dp = norm(Dp)
            if (1-θ)*Δ < norm_Dp < (1+θ)*Δ
                break
            end
            ϕ = norm_Dp-Δ
            if ϕ < 0
                uₖ = λ
            else
                lₖ = λ
            end
            q .= diag(D) .^ 2 .* p
            ldiv!(LowerTriangular(F.R'), q)
            denom = norm(q)^2
            λ = λ - ϕ/Δ*norm_Dp^2/denom
            if !(lₖ <= λ <= uₖ)
                λ = max(lₖ+0.01*(uₖ-lₖ), √(lₖ*uₖ))
            end
        end
        return λ
    end
end


function solve_subproblem(
    J::AbstractMatrix{T},
    f::AbstractVector{T},
    Δ::Real,
    cache::LQCache;
    maxiters = 6,
    θ = 1e-4,
) where {T<:Real}
    F = cache.factorization
    D = cache.scaling_matrix
    n, m = size(J)

    p = cache.p
    P = F.P
    z = cache.z
    mul!(z, P', -f)
    ldiv!(LowerTriangular(F.R'), z)
    mul!(p, Matrix(F.Q), z)
    if norm(p) <= Δ
        λ = zero(T)
        return λ
    else
        l₀ = 0.0
        u₀ = norm((J'f) ./ diag(D))/Δ
        λ₀ = max(1e-3*u₀, √(l₀*u₀))
        λ = λ₀
        uₖ = u₀
        lₖ = l₀

        Dp = cache.Dp
        DJ = cache.DJ
        DJ .= J' ./ diag(D)
        q = cache.q
        for i = 1:maxiters
            factorize_aug!(cache, [DJ; √λ*I(n)])
            Fa = cache.factorization_aug
            z .= -f
            Ru = UpperTriangular(Fa.R)
            Lu = LowerTriangular(Fa.R')
            ldiv!(Lu, z)
            ldiv!(Ru, z)
            mul!(p, J', z)
            p .= p ./ (diag(D) .^ 2) # D⁻²
            mul!(Dp, D, p)
            norm_Dp = norm(Dp)
            if (1-θ)*Δ < norm_Dp < (1+θ)*Δ
                break
            end
            ϕ = norm_Dp-Δ
            if ϕ < 0
                uₖ = λ
            else
                lₖ = λ
            end
            q .= z
            ldiv!(Lu, q)
            denom = λ*norm(q)^2 - norm(z)^2
            λ = λ - ϕ/Δ*norm_Dp^2/denom
            if !(lₖ <= λ <= uₖ)
                λ = max(lₖ+0.01*(uₖ-lₖ), √(lₖ*uₖ))
            end
        end
        return λ
    end
end



# function solve_subproblem(
#     strategy::QRSolve,
#     J::AbstractMatrix{T},
#     f::AbstractVector{T},
#     Δ::Real,
#     cache::QRCache,
#     λ_old::Real;
#     maxiters = 6,
#     θ = 1e-4,
# ) where {T<:Real}
#     F = cache.factorization
#     D = cache.scaling_matrix

#     p = cache.p
#     ldiv!(p, F, -f)
#     if norm(p) <= radius
#         λ = zero(T)
#         return λ
#     else
#         l₀ = 0.0
#         u₀ = norm((J'f) ./ diag(D))/Δ
#         λ₀ = max(1e-3*u₀, √(l₀*u₀))
#         if λ_old == 0.0
#             λ = λ₀
#         else
#             λ = λ_old
#         end
#         uₖ = u₀
#         lₖ = l₀
#         b_aug = cache.b_aug
#         Dp = cache.Dp
#         q = cache.q
#         b_aug .= [-f; zeros(size(J, 2))]
#         for i = 1:maxiters
#             factorize_aug!(cache, [J; √λ*D])
#             Fa = cache.factorization_aug
#             ldiv!(p, Fa, b_aug)
#             mul!(Dp, D, p)
#             norm_Dp = norm(Dp)
#             if (1-θ)*Δ < norm_Dp < (1+θ)*Δ
#                 break
#             end
#             ϕ = norm_Dp-Δ
#             if ϕ < 0
#                 uₖ = λ
#             else
#                 lₖ = λ
#             end
#             q .= diag(D) .^ 2 .* p
#             ldiv!(LowerTriangular(F.R'), q)
#             denom = norm(q)^2
#             λ = λ - ϕ/Δ*norm_Dp^2/denom
#             if !(lₖ <= λ <= uₖ)
#                 λ = max(lₖ+0.01*(uₖ-lₖ), √(lₖ*uₖ))
#             end
#         end
#         return λ
#     end
# end


# """
#     find_λ_scaled(strategy::SVDSolve, F, Δ, J, D, f, maxiters, θ=1e-4)

# Find the Lagrange multiplier λ for the scaled trust region subproblem using SVD.

# This function solves for λ such that ‖D*p‖ = Δ where p is computed using
# the SVD factorization to solve the regularized system.

# # Arguments
# - `strategy::SVDSolve`: SVD factorization strategy
# - `F`: SVD factorization object
# - `Δ`: Trust region radius
# - `J`: Jacobian matrix
# - `D`: Diagonal scaling matrix
# - `f`: Residual vector
# - `maxiters`: Maximum number of iterations for λ search
# - `θ`: Tolerance for trust region constraint satisfaction (default: 1e-4)

# # Returns
# - `λ`: Lagrange multiplier
# - `p`: Step direction satisfying ‖D*p‖ ≈ Δ

# # Algorithm
# Uses Newton's method with safeguarding to find λ. The step computation
# uses the SVD factorization for numerical stability with ill-conditioned systems.
# """
# function find_λ_scaled(strategy::SVDSolve, cache, Δ, J, D, f, λ_old, maxiters, θ = 1e-4)
#     F = cache.factorization
#     l₀ = 0.0
#     u₀ = norm(D*(J'f))/Δ
#     λ₀ = max(1e-3*u₀, √(l₀*u₀))
#     λ = λ₀
#     if λ_old == 0.0
#             λ = λ₀
#         else
#             λ = λ_old
#         end
#     uₖ = u₀
#     lₖ = l₀
#     p = cache.p
#     dpdλ = cache.dpdλ
#     for i = 1:maxiters
#         p = solve_augmented(strategy::SVDSolve, F, J, D, -f, λ)
#         if (1-θ)*Δ < norm(D*p) < (1+θ)*Δ
#             break
#         end
#         ϕ = norm(D*p)-Δ
#         if ϕ < 0
#             uₖ = λ
#         else
#             lₖ = λ
#         end
#         dpdλ = solve_for_dp_dlambda_scaled(strategy::SVDSolve, F, D, λ, -f)
#         λ = λ - (norm(D*p)-Δ)/Δ*((D*p)'*(D*p)/(p'*D'*(D*dpdλ)))
#         if !(lₖ ≤ λ ≤ uₖ)
#             λ = max(lₖ+0.01*(uₖ-lₖ), √(lₖ*uₖ))
#         end
#     end
#     return λ, p
# end


# """
#     solve_augmented(::SVDSolve, svdls::LinearAlgebra.SVD, J::AbstractMatrix, D::Diagonal, b::AbstractVector, λ::Real)

# Solve the regularized linear system using SVD factorization.

# Computes the solution to:
#     (JᵀJ + λDᵀD)δ = b

# using the SVD factorization J = UΣVᵀ. The solution is computed as:
#     δ = Σᵢ (σᵢ/(σᵢ² + λdᵢ²)) * (uᵢᵀb) * vᵢ

# where σᵢ are singular values, dᵢ are diagonal elements of D, and uᵢ, vᵢ are
# left and right singular vectors.

# # Arguments
# - `::SVDSolve`: SVD strategy indicator
# - `svdls`: SVD factorization object containing U, Σ, V
# - `J`: Jacobian matrix (not directly used, provided for interface consistency)
# - `D`: Diagonal scaling matrix
# - `b`: Right-hand side vector
# - `λ`: Regularization parameter

# # Returns
# - `δ`: Solution vector to the regularized system
# """
# function solve_augmented(
#     ::SVDSolve,
#     svdls::LinearAlgebra.SVD,
#     J::AbstractMatrix,
#     D::Diagonal,
#     b::AbstractVector,
#     λ::Real,
# )
#     U = svdls.U
#     V = svdls.V
#     σs = svdls.S
#     n = length(σs)
#     δ = zeros(size(J, 2))
#     for i = 1:n
#         δ += (σs[i]/(σs[i]^2 + λ*D[i, i]^2))*(U[:, i]'*b)*V[:, i]
#     end
#     return δ
# end

# """
#     solve_for_dp_dlambda_scaled(::SVDSolve, svdls::LinearAlgebra.SVD, D::AbstractMatrix, λ, b)

# Compute the derivative dp/dλ for the SVD-based trust region subproblem.

# This function computes how the step direction p changes with respect to the
# Lagrange multiplier λ using the SVD factorization. The derivative is computed
# analytically using the SVD representation.

# For p = Σᵢ (σᵢ/(σᵢ² + λdᵢ²)) * (uᵢᵀb) * vᵢ, the derivative is:
#     dp/dλ = Σᵢ (-σᵢdᵢ²/(σᵢ² + λdᵢ²)²) * (uᵢᵀb) * vᵢ

# # Arguments
# - `::SVDSolve`: SVD strategy indicator
# - `svdls`: SVD factorization object containing U, Σ, V
# - `D`: Diagonal scaling matrix
# - `λ`: Current Lagrange multiplier
# - `b`: Right-hand side vector

# # Returns
# - `dpdλ`: Derivative of step direction with respect to λ
# """
# function solve_for_dp_dlambda_scaled(
#     ::SVDSolve,
#     svdls::LinearAlgebra.SVD,
#     D::AbstractMatrix,
#     λ,
#     b,
# )
#     # Perform the SVDSolve factorization to get the factors explicitly
#     U = svdls.U
#     V = svdls.V
#     σs = svdls.S
#     # Get diagonal elements of D
#     d = diag(D)
#     n = length(σs)
#     # Calculate dδ/dλ using the derived formula
#     dpdλ = zeros(size(V, 1))
#     for i = 1:n
#         coeff = -σs[i] * d[i]^2 / (σs[i]^2 + λ * d[i]^2)^2
#         dpdλ += coeff * dot(U[:, i], b) * V[:, i]
#     end
#     return dpdλ
# end
