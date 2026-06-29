mutable struct QRSubproblemCache{S<:SubProblemStrategy,F,D,T,M} <: AbstractSubproblemCache
    factorization::F
    scaling_matrix::D

    # --- Buffers ---
    J_buffer::M

    p::Vector{T}             # Step direction 
    p_newton::Vector{T}      # Gradient dp/dλ
    dpdλ::Vector{T}
    z::Vector{T}
    dzdλ::Vector{T}
    R_buffer::Matrix{T}      # n x n mutable R
    rhs_buffer::Vector{T}    # n mutable RHS
    qtf_buffer::Vector{T}    # Stores Q'f
    v_row::Vector{T}         # Row workspace
    perm_buffer::Vector{T}   # Buffer for permutation operations

    function QRSubproblemCache(
        strategy::S,
        scaling_strat::Sc,
        J::AbstractMatrix{T};
        kwargs...,
    ) where {S<:SubProblemStrategy,Sc<:ScalingStrategy,T}

        n, m = size(J)
        F = factorize(strategy, J)
        Dk = scaling(scaling_strat, J; kwargs)

        if J isa StridedMatrix{T}
            if strategy isa QRSolve
                if n ≥ m
                    J_buffer = similar(J)
                else
                    J_buffer = similar(J') # Transposed shape for underdetermined!
                end
            else
                J_buffer = similar(J)
            end
        else
            J_buffer = nothing
        end

        p = zeros(T, m)
        p_newton = zeros(T, m)
        dpdλ = zeros(T, m)
        z = zeros(T, n)
        dzdλ = zeros(T, n)
        R_buffer = zeros(T, m, m)
        rhs_buffer = zeros(T, max(m, n))
        qtf_buffer = zeros(T, max(m, n))
        v_row = zeros(T, m)
        perm_buffer = zeros(T, m)

        new{S,typeof(F),typeof(Dk),T,typeof(J_buffer)}(
            F,
            Dk,
            J_buffer,
            p,
            p_newton,
            dpdλ,
            z,
            dzdλ,
            R_buffer,
            rhs_buffer,
            qtf_buffer,
            v_row,
            perm_buffer,
        )
    end
end


mutable struct EVDSubproblemCache{S<:EVDSolve,F,D,T,M} <: AbstractSubproblemCache
    factorization::F
    scaling_matrix::D

    # --- Buffers ---
    J_buffer::M

    p::Vector{T}             # Step direction 
    p_newton::Vector{T}      # Gradient dp/dλ
    z::Vector{T}             # intermediate vector when system is undetermined.

    function EVDSubproblemCache(
        strategy::S,
        scaling_strat::Sc,
        J::AbstractMatrix{T};
        kwargs...,
    ) where {S<:EVDSolve,Sc<:ScalingStrategy,T}

        n, m = size(J)
        F = factorize(strategy, J)
        Dk = scaling(scaling_strat, J; kwargs)

        if J isa StridedMatrix{T}
            if n ≥ m
                J_buffer = similar(J'*J)
            else # m < n
                J_buffer = similar(J*J')
            end
        else
            J_buffer = nothing
        end

        p = zeros(T, m)
        p_newton = zeros(T, m)
        z = zeros(T, n)


        new{S,typeof(F),typeof(Dk),T,typeof(J_buffer)}(F, Dk, J_buffer, p, p_newton, z)
    end
end

# Defining the end point for the functions
SubproblemCache(
    subproblem_strategy::QRrecursiveSolve,
    scaling_strategy::ScalingStrategy,
    J::AbstractMatrix,
) = QRSubproblemCache(subproblem_strategy, scaling_strategy, J)

SubproblemCache(
    subproblem_strategy::QRSolve,
    scaling_strategy::ScalingStrategy,
    J::AbstractMatrix,
) = QRSubproblemCache(subproblem_strategy, scaling_strategy, J)

SubproblemCache(
    subproblem_strategy::SVDSolve,
    scaling_strategy::ScalingStrategy,
    J::AbstractMatrix,
) = QRSubproblemCache(subproblem_strategy, scaling_strategy, J)

# Map the EVD strategy to its specialized cache
SubproblemCache(
    subproblem_strategy::EVDSolve,
    scaling_strategy::ScalingStrategy,
    J::AbstractMatrix,
) = EVDSubproblemCache(subproblem_strategy, scaling_strategy, J)
