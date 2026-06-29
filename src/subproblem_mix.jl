

function factorize_aug!(cache::QRCholCache, A::AbstractMatrix)
    copyto!(cache.factorization_aug.factors, A)
    cache.factorization_aug = cholesky!(cache.factorization_aug.factors)
end


function solve_subproblem(
    J::AbstractMatrix{T},
    f::AbstractVector{T},
    Δ::Real,
    cache::QRCholCache;
    maxiters = 6,
    θ = 1e-4,
) where {T<:Real}
    F = cache.factorization
    D = cache.scaling_matrix
    n, m = size(J)
    p = cache.p
    ldiv!(p, F, -f)
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
        b_aug = cache.b_aug
        D²p = cache.D²p
        D² = cache.D²
        q = cache.q
        JᵀJ = cache.JᵀJ
        mul!(b_aug, -J', f)
        mul!(JᵀJ, J', J)
        mul!(D², D, D)
        niters = 0
        for i = 1:maxiters
            factorize_aug!(cache, (JᵀJ+λ*D²))
            Fa = cache.factorization_aug
            ldiv!(p, Fa, b_aug)
            mul!(D²p, D, p)
            lmul!(D, D²p)
            pᵀD²p = p'D²p
            norm_Dp = √pᵀD²p
            if (1-θ)*Δ < norm_Dp < (1+θ)*Δ
                break
            end
            ϕ = norm_Dp-Δ
            if ϕ < 0
                uₖ = λ
            else
                lₖ = λ
            end
            ldiv!(q, Fa, D²p)
            denom = -D²p'q
            λ = λ - ϕ/Δ*pᵀD²p/denom
            if !(lₖ ≤ λ ≤ uₖ)
                λ = max(lₖ+0.01*(uₖ-lₖ), √(lₖ*uₖ))
            end
            niters += 1
        end
        println("Iterations: $niters")
        return λ
    end
end
