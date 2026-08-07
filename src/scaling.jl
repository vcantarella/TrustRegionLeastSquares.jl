abstract type ScalingStrategy end
abstract type BoundedScalingStrategy <: ScalingStrategy end
struct NoScaling <: ScalingStrategy end
struct JacobianScaling <: ScalingStrategy end
struct ColemanandLiScaling <: BoundedScalingStrategy end

function scaling(::NoScaling, J::AbstractMatrix; kwargs...)
    return I(size(J, 2))
end

function scaling(::JacobianScaling, J; τ = 1e-12, kwargs...)
    n = size(J, 2)
    Dk = Diagonal(ones(eltype(J), n))
    column_norms = [norm(J[:, i]) for i in axes(J, 2)]
    for i in axes(J, 2)
        Dk[i, i] = max(τ, column_norms[i])
    end
    return Dk
end

function scaling(::ColemanandLiScaling, J; x, lb, ub, g, τ = 1e-12)
    v = ones(length(x))
    jᵥ = zeros(length(x))
    for i in eachindex(x)
        if (g[i] < 0) & (ub[i] < Inf)
            v[i] = x[i] - ub[i]
            jᵥ[i] = sign(g[i])
        elseif (g[i] ≥ 0) & (lb[i] > -Inf)
            v[i] = x[i] - lb[i]
            jᵥ[i] = sign(g[i])
        elseif g[i] < 0 #ui = Infs
            v[i] = -1
            jᵥ[i] = 0
        else # g[i] > 0  & li = -Inf
            v[i] = 1
            jᵥ[i] = 0
        end
    end
    D = Diagonal(1 ./ sqrt.(abs.(v) .+ τ))
    Jᵥ = Diagonal(g) * Diagonal(jᵥ)
    return D, Jᵥ, v
end


# Plain loop on purpose: an assignment inside a comprehension (`[D[i,i] = ... for ...]`)
# triggers a JuliaFormatter v2 bug that rewrites the `=` into `in`, silently changing
# the code's meaning — and it kept the repo permanently failing the CI format check.
function scaling!(D::AbstractMatrix, scaling_strat::NoScaling; kwargs...)
    @inbounds for i in axes(D, 1)
        D[i, i] = one(eltype(D))
    end
    return D
end
