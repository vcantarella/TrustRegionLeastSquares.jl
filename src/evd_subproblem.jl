

function factorize(::EVDSolve, J)
    m, n = size(J)
    if m ≥ n
        return eigen(J'*J)
    else # m < n
        return eigen(J*J')
    end
end



"""
    factorize!(cache, strategy, J)

Updates the factorization in `cache`. Tries to use in-place operations to reduce allocations.
"""
function factorize!(cache::EVDSubproblemCache, strategy::EVDSolve, J::AbstractMatrix)
    # 1. Fast Path: If we have a buffer and J is compatible, use in-place
    if cache.J_buffer !== nothing
        # Copy J into the buffer. This avoids allocating a new matrix for the input.
        m, n = size(J)
        if m ≥ n
            copyto!(cache.J_buffer, J'*J)
        else # m < n
            copyto!(cache.J_buffer, J*J')
        end

        cache.factorization = eigen!(cache.J_buffer)

        return cache.factorization
    end

    # 2. Fallback: Standard allocating version
    #    Hit if J is sparse, StaticArray, or J_buffer was not allocated.
    cache.factorization = factorize(strategy, J)
    return cache.factorization
end



"""
    solve_subproblem(strategy::SubProblemStrategy, J::AbstractMatrix{T}, f::AbstractVector{T}, radius::Real, cache) where {T<:Real}

Solve the trust region subproblem to find the optimal step direction and Lagrange multiplier.

The subproblem being solved is:
    min_{δ} ½‖f + J*δ‖² subject to ‖D*δ‖ ≤ radius

where D is the scaling matrix from the cache.

# Arguments
- `strategy`: Subproblem solving strategy (QRSolve or SVDSolve)
- `J`: Jacobian matrix of the residual function
- `f`: Current residual vector
- `radius`: Trust region radius constraint
- `cache`: SubproblemCache containing factorization and scaling information

# Returns
- `λ`: Lagrange multiplier for the trust region constraint
- `δ`: Optimal step direction satisfying the trust region constraint

# Algorithm
If the Gauss-Newton step ‖δ_gn‖ ≤ radius, then λ = 0 and δ = δ_gn.
Otherwise, finds λ > 0 such that ‖D*δ‖ = radius using iterative methods.
"""
function solve_subproblem(
    strategy::EVDSolve,
    J::AbstractMatrix{T},
    f::AbstractVector{T},
    radius::Real,
    cache::EVDSubproblemCache,
) where {T<:Real}
    F = cache.factorization
    Q = F.vectors
    Λ = F.values
    Dk = cache.scaling_matrix
    z = cache.z

    m, n = size(J)
    p = cache.p # alias

    if m ≥ n
        # OVERDETERMINED CASE (F is the EVD of J' * J, size n x n)
        # Solve: (J' * J) * p = -J' * f

        # # 1. Calculate RHS: p = -J' * f 
        # mul!(p, J', -f)
        p .= Q * (Q' * J' * -f) ./ Λ

    else # m < n
        # UNDERDETERMINED CASE (F is the EVD of J * J', size m x m)
        # Solve dual: (J * J') * z = -f, then map back: p = J' * z

        # 1. Calculate RHS: z = -f
        # z .= -f

        z .= Q * (Q' * -f) ./ Λ
        p .= J'*z

    end

    if norm(p) <= radius
        λ = zero(T)
        δ = p
    else
        λ, δ = find_λ_scaled(strategy, cache, radius, J, Dk, f, 200, 1e-6)
    end
    return λ, δ
end



"""
    find_λ_scaled(strategy::EVDSolve, F, Δ, J, D, f, maxiters, θ=1e-4)

Find the Lagrange multiplier λ for the scaled trust region subproblem using SVD.

This function solves for λ such that ‖D*p‖ = Δ where p is computed using
the SVD factorization to solve the regularized system.

# Arguments
- `strategy::EVDSOlve`: SVD factorization strategy
- `F`: SVD factorization object
- `Δ`: Trust region radius
- `J`: Jacobian matrix
- `D`: Diagonal scaling matrix
- `f`: Residual vector
- `maxiters`: Maximum number of iterations for λ search
- `θ`: Tolerance for trust region constraint satisfaction (default: 1e-4)

# Returns
- `λ`: Lagrange multiplier
- `p`: Step direction satisfying ‖D*p‖ ≈ Δ

# Algorithm
Uses Newton's method with safeguarding to find λ. The step computation
uses the SVD factorization for numerical stability with ill-conditioned systems.
"""
function find_λ_scaled(strategy::EVDSolve, cache, Δ, J, D, f, maxiters, θ = 1e-4)
    F = cache.factorization
    l₀ = 0.0
    u₀ = norm(D*(J'f))/Δ
    λ₀ = max(1e-3*u₀, √(l₀*u₀))
    λ = λ₀
    uₖ = u₀
    lₖ = l₀
    p = zeros(size(J, 2))
    for i = 1:maxiters
        p, dpdλ = solve_augmented_with_derivative(strategy, F, J, -f, λ)
        if (1-θ)*Δ < norm(p) < (1+θ)*Δ
            break
        end
        ϕ = norm(p)-Δ
        if ϕ < 0
            uₖ = λ
        else
            lₖ = λ
        end
        λ = λ - (norm(p)-Δ)/Δ*((p)'*(p)/(p'*dpdλ))
        if !(uₖ < λ <= lₖ)
            λ = max(lₖ+0.01*(uₖ-lₖ), √(lₖ*uₖ))
        end
    end
    return λ, p
end

function solve_augmented_with_derivative(
    ::EVDSolve,
    F::Eigen,
    J::AbstractMatrix,
    b::AbstractVector,
    λ::Real,
)
    Q = F.vectors
    Λ = F.values
    m, n = size(J)


    if m ≥ n
        # Reconstruct in the primal space
        # 1. Project b onto the eigenvectors (Do this ONCE)
        q_b = Q' * J' * b

        # 2. Calculate the scaling factors for p and dp/dλ
        # Notice the element-wise division and the square (.^2) for the derivative
        scale_p = q_b ./ (Λ .+ λ)
        scale_dp = .-q_b ./ ((Λ .+ λ) .^ 2) # The negative sign is crucial here!
        p = Q * scale_p
        dpdλ = Q * scale_dp

        return p, dpdλ

    else # m < n
        # 1. Project b onto the eigenvectors (Do this ONCE)
        q_b = Q' * b

        # 2. Calculate the scaling factors for p and dp/dλ
        # Notice the element-wise division and the square (.^2) for the derivative
        scale_p = q_b ./ (Λ .+ λ)
        scale_dp = .-q_b ./ ((Λ .+ λ) .^ 2) # The negative sign is crucial here!
        # Reconstruct in the dual space
        z = Q * scale_p
        dz_dλ = Q * scale_dp

        # Map both back to the primal space
        p = J' * z
        dpdλ = J' * dz_dλ

        return p, dpdλ
    end
end
