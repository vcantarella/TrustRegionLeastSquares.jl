"""
    factorize(::QRSolve, J)

Compute QR factorization with column pivoting for the Jacobian matrix J.

# Arguments
- `::QRSolve`: Strategy indicator for QR factorization
- `J`: Jacobian matrix to factorize

# Returns
- QR factorization object with column pivoting
"""
function factorize(::QRSolve, J)
    n, m = size(J)
    if n ≥ m
        return qr(J, ColumnNorm())
    else
        return qr(J', ColumnNorm())
    end
end

function factorize(::QRrecursiveSolve, J)

    return qr(J, ColumnNorm())

end

"""
    factorize(::SVDSolve, J)

Compute Singular Value Decomposition (SVD) for the Jacobian matrix J.

# Arguments
- `::SVDSolve`: Strategy indicator for SVD factorization
- `J`: Jacobian matrix to factorize

# Returns
- SVD factorization object
"""
factorize(::SVDSolve, J) = svd(J)


# --- Robust Factorize! Implementation ---

"""
    factorize!(cache, strategy, J)

Updates the factorization in `cache`. Tries to use in-place operations to reduce allocations.
"""
function factorize!(cache::QRSubproblemCache, strategy, J::AbstractMatrix)
    # 1. Fast Path: If we have a buffer and J is compatible, use in-place
    if cache.J_buffer !== nothing
        # Copy J into the buffer. This avoids allocating a new matrix for the input.


        if strategy isa QRSolve
            n, m = size(J)
            if n ≥ m
                copyto!(cache.J_buffer, J)
                cache.factorization = qr!(cache.J_buffer, ColumnNorm())
            else
                copyto!(cache.J_buffer, J')
                cache.factorization = qr!(cache.J_buffer, ColumnNorm())
            end
            return cache.factorization
        elseif strategy isa QRrecursiveSolve
            copyto!(cache.J_buffer, J)
            cache.factorization = qr!(cache.J_buffer, ColumnNorm())
            return cache.factorization

        elseif strategy isa SVDSolve
            copyto!(cache.J_buffer, J)
            # SVD Path: In-place input
            # svd! destroys the input (J_buffer) to save the copy allocation.
            # It still allocates U, S, Vt result arrays.
            cache.factorization = svd!(cache.J_buffer)
            return cache.factorization
        end
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
    strategy::St,
    J::AbstractMatrix{T},
    f::AbstractVector{T},
    radius::Real,
    cache::QRSubproblemCache,
) where {St<:QRSolve,T<:Real}
    F = cache.factorization
    Dk = cache.scaling_matrix

    n, m = size(J)
    # Note: F \ -f might allocate if not careful, but usually acceptable for the check.
    # To be strictly zero-alloc, we would need to ldiv! into cache.p here.
    δgn = cache.p
    if n ≥ m
        ldiv!(δgn, F, -f)
    else
        # UNDERDETERMINED: F = qr(J', ColumnNorm())
        # We must manually calculate the minimum-norm solution.
        P = F.P
        z = cache.z
        mul!(z, P', -f)
        ldiv!(LowerTriangular(F.R'), z)
        mul!(δgn, Matrix(F.Q), z)
    end

    if norm(δgn) <= radius
        λ = zero(T)
        δ = δgn
    else
        if n ≥ m
            λ, δ = find_λ_scaled(strategy, cache, radius, J, Dk, f, 200, 1e-6)
        else
            λ, δ = find_λ_scaled_undetermined(strategy, cache, radius, J, Dk, f, 200, 1e-6)
        end
    end
    return λ, δ
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
    strategy::St,
    J::AbstractMatrix{T},
    f::AbstractVector{T},
    radius::Real,
    cache::QRSubproblemCache,
) where {St<:Union{SVDSolve,QRrecursiveSolve},T<:Real}
    F = cache.factorization
    Dk = cache.scaling_matrix

    # Note: F \ -f might allocate if not careful, but usually acceptable for the check.
    # To be strictly zero-alloc, we would need to ldiv! into cache.p here.
    δgn = cache.p
    ldiv!(δgn, F, -f)
    # δgn = cache.p

    if norm(δgn) <= radius
        λ = zero(T)
        δ = δgn
    else
        # PASS CACHE HERE
        λ, δ = find_λ_scaled(strategy, cache, radius, J, Dk, f, 200, 1e-6)
    end
    return λ, δ
end



"""
    find_λ_scaled(strategy::QRSolve, F, Δ, J, D, f, maxiters, θ=1e-4)

Find the Lagrange multiplier λ for the scaled trust region subproblem using QR factorization.

This function solves for λ such that ‖D*p‖ = Δ where p solves:
    (JᵀJ + λDᵀD)p = -Jᵀf

# Arguments
- `strategy::QRSolve`: QR factorization strategy
- `F`: QR factorization object (currently not used, refactored in implementation)
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
Uses Newton's method with safeguarding to find λ. The search is constrained
between lower bound l₀ = 0 and upper bound u₀ = ‖D*(Jᵀf)‖/Δ.
"""
function find_λ_scaled(strategy::QRSolve, cache, Δ, J, D, f, maxiters, θ = 1e-4)
    l₀ = 0.0
    u₀ = norm(D*(J'f))/Δ
    λ₀ = max(1e-3*u₀, √(l₀*u₀))
    λ = λ₀
    uₖ = u₀
    lₖ = l₀
    p = cache.p
    dpdλ = cache.dpdλ
    b_aug = [-f; zeros(size(J, 2))]
    for i = 1:maxiters
        F = qr([J; √λ*D])
        ldiv!(p, F, b_aug)
        # p = solve_augmented(strategy, J, D, b_aug, -f, λ)
        if (1-θ)*Δ < norm(D*p) < (1+θ)*Δ
            break
        end
        ϕ = norm(D*p)-Δ
        if ϕ < 0
            uₖ = λ
        else
            lₖ = λ
        end
        solve_for_dp_dlambda_scaled!(strategy, dpdλ, F, p, D)
        λ = λ - (norm(D*p)-Δ)/Δ*((D*p)'*(D*p)/(p'*D'*(D*dpdλ)))
        if !(lₖ <= λ <= uₖ)
            λ = max(lₖ+0.01*(uₖ-lₖ), √(lₖ*uₖ))
        end
    end
    return λ, p
end



"""
    find_λ_scaled_underdetermined(strategy::QRSolve, F, Δ, J, D, f, maxiters, θ=1e-4)

Find the Lagrange multiplier λ for the scaled trust region subproblem using QR factorization.

This function solves for λ such that ‖D*p‖ = Δ where p solves:
    (JᵀJ + λDᵀD)p = -Jᵀf

# Arguments
- `strategy::QRSolve`: QR factorization strategy
- `F`: QR factorization object (currently not used, refactored in implementation)
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
Uses Newton's method with safeguarding to find λ. The search is constrained
between lower bound l₀ = 0 and upper bound u₀ = ‖D*(Jᵀf)‖/Δ.
"""
function find_λ_scaled_undetermined(
    strategy::QRSolve,
    cache,
    Δ,
    J,
    D,
    f,
    maxiters,
    θ = 1e-4,
)
    l₀ = 0.0
    u₀ = norm((J'f) ./ diag(D))/Δ
    λ₀ = max(1e-3*u₀, √(l₀*u₀))
    λ = λ₀
    uₖ = u₀
    lₖ = l₀
    p = cache.p
    dzdλ = cache.dzdλ
    dpdλ = cache.dpdλ
    z = cache.z
    b_aug = [zeros(size(J, 2)); -1/√λ*f]
    DJ = J' ./ diag(D) # Math: D⁻¹ Jᵀ
    for i = 1:maxiters
        F = qr([DJ; √λ*I(size(J, 1))])
        z .= -f
        Ru = UpperTriangular(F.R)
        ldiv!(Ru', z)
        ldiv!(Ru, z)
        # ldiv!(z, F, b_aug)
        mul!(p, J', z)
        p .= p ./ (diag(D) .^ 2) # D⁻²
        # p = solve_augmented(strategy, J, D, b_aug, -f, λ)
        if (1-θ)*Δ < norm(D*p) < (1+θ)*Δ
            break
        end
        ϕ = norm(D*p)-Δ
        if ϕ < 0
            uₖ = λ
        else
            lₖ = λ
        end
        solve_for_dz_dlambda_scaled!(strategy, dzdλ, F, z, D)
        mul!(dpdλ, J', dzdλ)
        dpdλ .= dpdλ ./ (diag(D) .^ 2) # D⁻²
        λ = λ - (norm(D*p)-Δ)/Δ*((D*p)'*(D*p)/(p'*D'*(D*dpdλ)))
        if !(lₖ ≤ λ ≤ uₖ)
            λ = max(lₖ+0.01*(uₖ-lₖ), √(lₖ*uₖ))
        end
    end
    return λ, p
end


"""
    find_λ_scaled(strategy::SVDSolve, F, Δ, J, D, f, maxiters, θ=1e-4)

Find the Lagrange multiplier λ for the scaled trust region subproblem using SVD.

This function solves for λ such that ‖D*p‖ = Δ where p is computed using
the SVD factorization to solve the regularized system.

# Arguments
- `strategy::SVDSolve`: SVD factorization strategy
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
function find_λ_scaled(strategy::SVDSolve, cache, Δ, J, D, f, maxiters, θ = 1e-4)
    F = cache.factorization
    l₀ = 0.0
    u₀ = norm(D*(J'f))/Δ
    λ₀ = max(1e-3*u₀, √(l₀*u₀))
    λ = λ₀
    uₖ = u₀
    lₖ = l₀
    p = cache.p
    dpdλ = cache.dpdλ
    for i = 1:maxiters
        p = solve_augmented(strategy::SVDSolve, F, J, D, -f, λ)
        if (1-θ)*Δ < norm(D*p) < (1+θ)*Δ
            break
        end
        ϕ = norm(D*p)-Δ
        if ϕ < 0
            uₖ = λ
        else
            lₖ = λ
        end
        dpdλ = solve_for_dp_dlambda_scaled(strategy::SVDSolve, F, D, λ, -f)
        λ = λ - (norm(D*p)-Δ)/Δ*((D*p)'*(D*p)/(p'*D'*(D*dpdλ)))
        if !(lₖ ≤ λ ≤ uₖ)
            λ = max(lₖ+0.01*(uₖ-lₖ), √(lₖ*uₖ))
        end
    end
    return λ, p
end


"""
    solve_augmented(::SVDSolve, svdls::LinearAlgebra.SVD, J::AbstractMatrix, D::Diagonal, b::AbstractVector, λ::Real)

Solve the regularized linear system using SVD factorization.

Computes the solution to:
    (JᵀJ + λDᵀD)δ = b

using the SVD factorization J = UΣVᵀ. The solution is computed as:
    δ = Σᵢ (σᵢ/(σᵢ² + λdᵢ²)) * (uᵢᵀb) * vᵢ

where σᵢ are singular values, dᵢ are diagonal elements of D, and uᵢ, vᵢ are
left and right singular vectors.

# Arguments
- `::SVDSolve`: SVD strategy indicator
- `svdls`: SVD factorization object containing U, Σ, V
- `J`: Jacobian matrix (not directly used, provided for interface consistency)
- `D`: Diagonal scaling matrix
- `b`: Right-hand side vector
- `λ`: Regularization parameter

# Returns
- `δ`: Solution vector to the regularized system
"""
function solve_augmented(
    ::SVDSolve,
    svdls::LinearAlgebra.SVD,
    J::AbstractMatrix,
    D::Diagonal,
    b::AbstractVector,
    λ::Real,
)
    U = svdls.U
    V = svdls.V
    σs = svdls.S
    n = length(σs)
    δ = zeros(size(J, 2))
    for i = 1:n
        δ += (σs[i]/(σs[i]^2 + λ*D[i, i]^2))*(U[:, i]'*b)*V[:, i]
    end
    return δ
end

"""
    solve_for_dp_dlambda_scaled(::QRSolve, qrf::Union{LinearAlgebra.QR, LinearAlgebra.QRCompactWY}, p::AbstractVector, D::AbstractMatrix)

Compute the derivative dp/dλ for the QR-based trust region subproblem.

This function computes how the step direction p changes with respect to the
Lagrange multiplier λ. This derivative is needed for Newton's method in
finding the optimal λ.

For the system (JᵀJ + λDᵀD)p = -Jᵀf, the derivative satisfies:
    (JᵀJ + λDᵀD)(dp/dλ) = -DᵀDp

Using the QR factorization, this is solved as:
    (RᵀR)(dp/dλ) = -DᵀDp

# Arguments
- `::QRSolve`: QR strategy indicator
- `qrf`: QR factorization object
- `p`: Current step direction
- `D`: Scaling matrix

# Returns
- `dp_dλ`: Derivative of step direction with respect to λ
"""
function solve_for_dp_dlambda_scaled!(
    ::QRSolve,
    dp_dλ,
    qrf::Union{LinearAlgebra.QR,LinearAlgebra.QRCompactWY},
    p::AbstractVector,
    D::AbstractMatrix,
)
    # Perform the QRSolve factorization to get the factors explicitly
    R = qrf.R
    dp_dλ .= .-(D'*(D*p))
    # Now solve (RᵀR) * (dp/dλ) = -p
    # This is done in two steps:
    # 1. Rᵀz = -p  =>  z = Rᵀ \ -p
    # 2. R(dp/dλ) = z  =>  dp/dλ = R \ z
    ldiv!(LowerTriangular(R'), dp_dλ)
    ldiv!(UpperTriangular(R), dp_dλ)
end

function solve_for_dz_dlambda_scaled!(
    ::QRSolve,
    dz_dλ,
    qrf::Union{LinearAlgebra.QR,LinearAlgebra.QRCompactWY},
    z::AbstractVector,
    D::AbstractMatrix,
)
    # Perform the QRSolve factorization to get the factors explicitly
    R = qrf.R
    dz_dλ .= .-z
    # Now solve (RᵀR) * (dp/dλ) = -p
    # This is done in two steps:
    # 1. Rᵀz = -p  =>  z = Rᵀ \ -p
    # 2. R(dp/dλ) = z  =>  dp/dλ = R \ z
    ldiv!(LowerTriangular(R'), dz_dλ)
    ldiv!(UpperTriangular(R), dz_dλ)
end

"""
    solve_for_dp_dlambda_scaled(::SVDSolve, svdls::LinearAlgebra.SVD, D::AbstractMatrix, λ, b)

Compute the derivative dp/dλ for the SVD-based trust region subproblem.

This function computes how the step direction p changes with respect to the
Lagrange multiplier λ using the SVD factorization. The derivative is computed
analytically using the SVD representation.

For p = Σᵢ (σᵢ/(σᵢ² + λdᵢ²)) * (uᵢᵀb) * vᵢ, the derivative is:
    dp/dλ = Σᵢ (-σᵢdᵢ²/(σᵢ² + λdᵢ²)²) * (uᵢᵀb) * vᵢ

# Arguments
- `::SVDSolve`: SVD strategy indicator
- `svdls`: SVD factorization object containing U, Σ, V
- `D`: Diagonal scaling matrix
- `λ`: Current Lagrange multiplier
- `b`: Right-hand side vector

# Returns
- `dpdλ`: Derivative of step direction with respect to λ
"""
function solve_for_dp_dlambda_scaled(
    ::SVDSolve,
    svdls::LinearAlgebra.SVD,
    D::AbstractMatrix,
    λ,
    b,
)
    # Perform the SVDSolve factorization to get the factors explicitly
    U = svdls.U
    V = svdls.V
    σs = svdls.S
    # Get diagonal elements of D
    d = diag(D)
    n = length(σs)
    # Calculate dδ/dλ using the derived formula
    dpdλ = zeros(size(V, 1))
    for i = 1:n
        coeff = -σs[i] * d[i]^2 / (σs[i]^2 + λ * d[i]^2)^2
        dpdλ += coeff * dot(U[:, i], b) * V[:, i]
    end
    return dpdλ
end

"""
    find_λ_scaled(strategy::QRrecursiveSolve, F, Δ, J, D, f, maxiters, θ=1e-4)

Find the Lagrange multiplier λ for the scaled trust region subproblem using Recursive QR.
"""
function find_λ_scaled(strategy::QRrecursiveSolve, cache, Δ, J, D, f, maxiters, θ = 1e-4)
    # Unpack buffers
    F = cache.factorization
    p = cache.p
    dpdλ = cache.p_newton
    R_buffer = cache.R_buffer
    rhs_buffer = cache.rhs_buffer
    qtf = cache.qtf_buffer
    v_row = cache.v_row
    perm_buffer = cache.perm_buffer

    m, n = size(J)

    # FIX 1: Type Stable Permutation
    # ensure perm is always Vector{Int}
    perm = hasproperty(F, :p) ? F.p : collect(1:n)

    # Lambda initialization
    l₀ = 0.0
    u₀ = norm(D*(J'f))/Δ
    λ₀ = max(1e-3*u₀, √(l₀*u₀))
    λ = λ₀
    uₖ = u₀
    lₖ = l₀

    # 1. Setup (One-time allocation for Q'*f allowed)
    qtf_src = F.Q' * f
    @inbounds for i = 1:m
        qtf[i] = qtf_src[i]
    end

    # 2. Main Loop
    for i = 1:maxiters
        # FIX 3: Efficient R copy
        # Access F.factors directly if possible to avoid UpperTriangular wrapper allocs
        # If F is QRPivoted or QR, it has :factors.
        src_R = hasproperty(F, :factors) ? F.factors : F.R

        if m >= n
            @inbounds for j = 1:n, k = 1:j

                R_buffer[k, j] = src_R[k, j]
            end
        else
            fill!(R_buffer, 0.0)
            @inbounds for j = 1:n, k = 1:min(j, m)

                R_buffer[k, j] = src_R[k, j]
            end
        end

        # Reset RHS
        @inbounds for k = 1:n
            rhs_buffer[k] = -qtf[k]
        end

        # Recursive Update
        solve_damped_system_recursive_inplace!(
            p,
            R_buffer,
            rhs_buffer,
            λ,
            n,
            D,
            perm,
            v_row,
        )

        norm_Dp = norm(D*p)
        if (1-θ)*Δ < norm_Dp < (1+θ)*Δ
            break
        end

        ϕ = norm_Dp - Δ
        if ϕ < 0

            uₖ = λ
        else

            lₖ = λ
        end

        # Derivative
        solve_for_dp_dlambda_scaled!(dpdλ, R_buffer, p, D, perm, perm_buffer)

        # Newton Update
        denominator = (D*p)' * (D * dpdλ)
        λ = λ - (norm_Dp - Δ)/Δ * ((norm_Dp^2) / denominator)

        if !(lₖ ≤ λ ≤ uₖ)
            λ = max(lₖ + 0.01*(uₖ - lₖ), √(lₖ * uₖ))
        end
    end
    return λ, p
end

function solve_damped_system_recursive_inplace!(
    p_cache,
    R_cache,
    QTr_cache,
    λ,
    n,
    D,
    perm,
    v_row,
)
    sqrt_λ = sqrt(λ)

    for c_idx = 1:n
        fill!(v_row, 0.0)

        var_idx = perm[c_idx]
        d_val = D isa Diagonal ? D[var_idx, var_idx] : D[var_idx]

        v_row[c_idx] = sqrt_λ * d_val
        v_rhs = 0.0

        for i = c_idx:n
            r_ii = R_cache[i, i]
            v_val = v_row[i]

            if abs(v_val) > 0 || i == c_idx
                c, s = compute_givens(r_ii, v_val)

                R_cache[i, i] = c * r_ii + s * v_val

                @inbounds for k = (i+1):n
                    val_R = R_cache[i, k]
                    val_v = v_row[k]
                    R_cache[i, k] = c * val_R + s * val_v
                    v_row[k] = -s * val_R + c * val_v
                end

                val_rhs_R = QTr_cache[i]
                QTr_cache[i] = c * val_rhs_R + s * v_rhs
                v_rhs = -s * val_rhs_R + c * v_rhs
            end
        end
    end

    # Use View for RHS to match dimension n
    ldiv!(p_cache, UpperTriangular(R_cache), view(QTr_cache, 1:n))

    copyto!(v_row, p_cache)
    @inbounds for i = 1:n
        p_cache[perm[i]] = v_row[i]
    end

    return p_cache
end

function solve_for_dp_dlambda_scaled!(dp_dλ, R_cache, p, D, perm, perm_buffer)
    # FIX 2: Zero-allocation D scaling
    # Replaces @. dp_dλ = -(D.diag^2) * dp_dλ
    if D isa Diagonal
        @inbounds for i = 1:length(p)
            d_val = D.diag[i]
            # Handle Bool/Number conversion implicitly by math
            val_sq = d_val * d_val
            dp_dλ[i] = -val_sq * p[i]
        end
    else
        # Fallback (allocating, but rare in this context)
        dp_dλ .= -(D' * (D * p))
    end

    # Permute RHS into perm_buffer
    @inbounds for i = 1:length(perm)
        perm_buffer[i] = dp_dλ[perm[i]]
    end

    # Solve
    ldiv!(LowerTriangular(R_cache'), perm_buffer)
    ldiv!(UpperTriangular(R_cache), perm_buffer)

    # Unpermute
    @inbounds for i = 1:length(perm)
        dp_dλ[perm[i]] = perm_buffer[i]
    end

    return dp_dλ
end

# Robust calculation of c, s for a Givens rotation
@inline function compute_givens(f::T, g::T) where {T<:Real}
    if g == 0
        return one(T), zero(T)
    end
    if f == 0
        return zero(T), one(T)
    end
    r = hypot(f, g)
    inv_r = 1 / r
    c = f * inv_r
    s = g * inv_r
    return c, s
end

"""
    update_cache!(cache::SubproblemCache, strategy, scaling_strat, J, x, lb, ub, g)

Updates the cache in-place with new Jacobian and scaling information.
"""
function update_cache!(cache::QRSubproblemCache, strategy, scaling_strat, J, x, lb, ub, g)
    factorize!(cache, strategy, J)

    # Update scaling
    # We handle the case where scaling returns a tuple (like ColemanandLiScaling)
    # by taking the first element, assuming it's the scaling matrix D.
    val = scaling(scaling_strat, J; x = x, lb = lb, ub = ub, g = g)
    if val isa Tuple
        cache.scaling_matrix = val[1]
    else
        cache.scaling_matrix = val
    end
    return cache
end
