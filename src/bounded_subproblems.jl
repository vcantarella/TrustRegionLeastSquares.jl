abstract type BoundedSubproblemCache end

mutable struct ColemanandLiCache{S<:SubProblemStrategy,F,D,JV,V,M,T} <:
               BoundedSubproblemCache
    factorization::F
    scaling_matrix::D
    Jv::JV
    v::V
    # Workspaces (matching SubproblemCache)
    J_buffer::M
    p::Vector{T}             # Step direction 
    p_newton::Vector{T}      # Gradient dp/dλ
    R_buffer::Matrix{T}      # n x n mutable R
    rhs_buffer::Vector{T}    # n mutable RHS
    qtf_buffer::Vector{T}    # Stores Q'f
    v_row::Vector{T}         # Row workspace
    perm_buffer::Vector{T}   # Buffer for permutation operations

    # --- NEW BUFFERS FOR STEP & ALGORITHM ---
    x_new::Vector{T}
    hits::Vector{Int}
    p_refl::Vector{T}
    dg::Vector{T}
    v_scaling::Vector{T} # For 1 ./ Dk^2
    J_v_buffer::Vector{T} # For J * v (size m)

    ColemanandLiCache(
        strategy::S,
        scaling_strat::Sc,
        J::AbstractMatrix{T};
        kwargs...,
    ) where {S<:SubProblemStrategy,Sc<:BoundedScalingStrategy,T} = begin
        m, n = size(J)
        F = factorize(strategy, J)
        Dk, A, v = scaling(scaling_strat, J; kwargs...)

        # Initialize workspaces
        if J isa StridedMatrix{T}
            J_buffer = similar(J)
        else
            J_buffer = nothing
        end

        p = zeros(T, n)
        p_newton = zeros(T, n)
        R_buffer = zeros(T, n, n)
        rhs_buffer = zeros(T, n) # Only needs to store size n for the solve
        qtf_buffer = zeros(T, m) # Stores Q'f (size m)
        v_row = zeros(T, n)
        perm_buffer = zeros(T, n)

        # New buffers
        x_new = zeros(T, n)
        hits = zeros(Int, n)
        p_refl = zeros(T, n)
        dg = zeros(T, n)
        v_scaling = zeros(T, n)
        J_v_buffer = zeros(T, m)

        new{S,typeof(F),typeof(Dk),typeof(A),typeof(v),typeof(J_buffer),T}(
            F,
            Dk,
            A,
            v,
            J_buffer,
            p,
            p_newton,
            R_buffer,
            rhs_buffer,
            qtf_buffer,
            v_row,
            perm_buffer,
            x_new,
            hits,
            p_refl,
            dg,
            v_scaling,
            J_v_buffer,
        )
    end
end

"""
    update_cache!(cache, strategy, scaling_strat, J, x, lb, ub, g)

Updates the cache in-place with new Jacobian and scaling information.
"""
function update_cache!(cache::ColemanandLiCache, strategy, scaling_strat, J, x, lb, ub, g)
    # 1. Update Factorization (reuse J_buffer if available via factorize!)
    # Note: If J changed significantly, factorize! might allocate, but usually it's optimized.
    # Assuming factorize! exists or we fall back to creating a new one.
    if hasmethod(factorize!, Tuple{typeof(cache),typeof(strategy),typeof(J)})
        factorize!(cache, strategy, J)
    else
        cache.factorization = factorize(strategy, J)
    end

    # 2. Update Scaling
    # Ideally scaling() should have an in-place version, but for now we re-assign.
    # If Dk, A, v are immutable structs or vectors, this is standard.
    # To be fully non-allocating, we would need update_scaling!(cache.Dk, ...).
    Dk, A, v = scaling(scaling_strat, J; x = x, lb = lb, ub = ub, g = g)
    cache.scaling_matrix = Dk
    cache.Jv = A
    cache.v = v
    return cache
end

"""
    solve_subproblem(strategy::SubProblemStrategy, J, f, radius, cache::ColemanandLiCache)

Solve the trust region subproblem for bounded problems.
"""
function solve_subproblem(
    strategy::SubProblemStrategy,
    J::AbstractMatrix{T},
    f::AbstractVector{T},
    radius::Real,
    cache::ColemanandLiCache,
) where {T<:Real}
    Dk = cache.scaling_matrix
    A = cache.Jv
    F = cache.factorization

    n = size(J, 2)
    m = size(J, 1)

    # 1. Prepare Q'f (Used for both δgn and find_λ)
    # Reuse qtf_buffer.
    # qtf_src = F.Q' * f. We use mul! if possible.
    # For standard QRPivoted, Q' * f allocates. 
    # Optimized way:
    if hasproperty(F, :Q)
        mul!(cache.qtf_buffer, F.Q', f)
    else
        # Fallback for generic factorizations
        cache.qtf_buffer .= F.Q' * f
    end

    # 2. Prepare R_buffer
    # Reuse code from find_λ to setup R
    src_R = hasproperty(F, :factors) ? F.factors : F.R
    if m >= n
        @inbounds for j = 1:n, k = 1:j

            cache.R_buffer[k, j] = src_R[k, j]
        end
    else
        fill!(cache.R_buffer, 0.0)
        @inbounds for j = 1:n, k = 1:min(j, m)

            cache.R_buffer[k, j] = src_R[k, j]
        end
    end

    # 3. Compute δgn (Gauss-Newton step, λ=0)
    # We use the recursive solver with λ=0 to avoid the [J; √A*Dk] allocation.

    # Reset RHS buffer for δgn solve
    @inbounds for k = 1:n
        cache.rhs_buffer[k] = -cache.qtf_buffer[k]
    end

    perm = hasproperty(F, :p) ? F.p : collect(1:n)

    # Solve with λ=0
    # Note: R_buffer is modified in-place! So we must RE-COPY it for find_λ later if we branch there.
    # However, since we likely branch, we should perhaps copy R_buffer to a temp or restore it.
    # Strategy: Use R_buffer for δgn. If rejected, find_λ will re-initialize R_buffer anyway.
    solve_damped_system_recursive_coleman!(
        cache.p,
        cache.R_buffer,
        cache.rhs_buffer,
        zero(T),
        n,
        Dk,
        A,
        perm,
        cache.v_row,
    )

    # Check Trust Region in SCALED space
    norm_scaled_gn = norm(Dk * cache.p) # allocation-free if Dk is Diagonal

    if norm_scaled_gn <= radius
        λ = zero(T)
        δ = cache.p # This aliases cache.p, which is fine as long as we copy/use it before next call
    # If the caller needs δ to persist, they should copy it. 
    # But wait, find_λ returns `p` which aliases cache.p too. 
    # So we stick to the pattern: return the buffer.
    else
        # R_buffer is dirty. find_λ_colemanandli must re-initialize it.
        # It does: loop 2 of find_λ starts by copying src_R -> R_buffer.
        λ, δ = find_λ_colemanandli(strategy, cache, radius, J, Dk, A, f, 200, 1e-6)
    end

    return λ, δ
end

"""
    find_λ_colemanandli(strategy::QRrecursiveSolve, F, Δ, J, D,A, f, maxiters, θ=1e-4)

Find the Lagrange multiplier λ for the scaled trust region subproblem using Recursive QR.
"""
function find_λ_colemanandli(
    strategy::QRrecursiveSolve,
    cache,
    Δ,
    J,
    D,
    A,
    f,
    maxiters,
    θ = 1e-4,
)
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
        solve_damped_system_recursive_coleman!(
            p,
            R_buffer,
            rhs_buffer,
            λ,
            n,
            D,
            A,
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

        if !(uₖ < λ <= lₖ)
            λ = max(lₖ + 0.01*(uₖ - lₖ), √(lₖ * uₖ))
        end
    end
    return λ, p
end


function solve_damped_system_recursive_coleman!(
    p_cache,
    R_cache,
    QTr_cache,
    λ,
    n,
    D,
    A,
    perm,
    v_row,
)

    for c_idx = 1:n
        fill!(v_row, 0.0)

        var_idx = perm[c_idx]
        d_val = D isa Diagonal ? D[var_idx, var_idx] : D[var_idx]
        a_val = A isa Diagonal ? A[var_idx, var_idx] : A[var_idx]
        v_row[c_idx] = v_row[c_idx] = d_val * sqrt(λ + a_val)
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
