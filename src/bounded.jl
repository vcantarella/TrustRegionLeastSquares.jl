# Bound constraints  lb ≤ x ≤ ub, following [MMP09] with the affine scaling of [CL96]:
# the trust-region step is computed in the scaled norm ‖D_s |v|^{-1/2} p‖ ≤ Δ, so that a variable
# close to the bound its gradient points at can barely move towards it; the step is then projected
# onto the box and made to achieve at least a fraction β₁ of the decrease of a generalized Cauchy
# step along the scaled steepest descent −|v| ⊙ g.

"""
    BoundedCache(inner::SolverCache)

Wraps an unconstrained cache with the buffers of the projected step: `D` (the base scaling matrix,
before the affine factor `|v|^{-1/2}` is applied to `inner.scaling_matrix`), `v` (Coleman–Li
distances), `p` (final step), `pC` (Cauchy step), `w = pC − p̄`, and their images under `J`.
"""
struct BoundedCache{C<:SolverCache,T}
    inner::C
    D::Diagonal{T,Vector{T}}
    v::Vector{T}
    p::Vector{T}
    pC::Vector{T}
    w::Vector{T}
    Jp::Vector{T}
    Jw::Vector{T}
end
function BoundedCache(inner::SolverCache)
    T, m, n = eltype(inner.p), length(inner.p), length(inner.Jp)
    return BoundedCache(
        inner,
        copy(inner.scaling_matrix),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, n),
        zeros(T, n),
    )
end
function update_cache!(cache::BoundedCache, J, scaling)
    cache.inner.scaling_matrix.diag .= cache.D.diag    # restore the base scaling before Moré's max-update
    update_cache!(cache.inner, J, scaling)
    cache.D.diag .= cache.inner.scaling_matrix.diag
end

"""
    coleman_li_distances!(v, x, g, lb, ub)

`|v_i|` of [CL96], eq. (1.4): the distance from `x_i` to the bound the negative gradient points
at, or `1` when that bound is infinite. Zero on an active bound with the gradient pushing outwards.
"""
function coleman_li_distances!(v, x, g, lb, ub)
    @inbounds for i in eachindex(x)
        v[i] =
            g[i] < 0 && isfinite(ub[i]) ? ub[i] - x[i] :
            g[i] >= 0 && isfinite(lb[i]) ? x[i] - lb[i] : one(eltype(x))
    end
    return v
end

"""
    affine_scaling!(cache, x, g, lb, ub) -> D

Set the Coleman–Li distances `cache.v` and, from them, the trust-region norm of the inner solver,
`D = D_base |v|^{-1/2}`: a variable close to the bound its negative gradient points at gets a large
`D_ii`, so the trust region allows it only a small move in that direction. `eps` keeps `D` finite on
an active bound. Call this before [`solve_subproblem`](@ref) and [`cauchy_step!`](@ref).
"""
function affine_scaling!(cache::BoundedCache, x, g, lb, ub)
    v, D = coleman_li_distances!(cache.v, x, g, lb, ub), cache.inner.scaling_matrix
    D.diag .= cache.D.diag ./ sqrt.(max.(v, eps(eltype(v))))
    return D
end

"""
    cauchy_step!(cache, J, g, x, Δ, lb, ub) -> decrease

Generalized Cauchy step `pC = ω d` ([MMP09] eq. 8; the unconstrained form is Algorithm 4.2 of
[NW06]) along the scaled steepest descent
`d = -|v| ⊙ g`, with `ω` the smallest of the model minimiser along `d`, the trust-region boundary
(in the scaled norm of the inner solver) and the ray–box intersection. Writes `pC` into `cache.pC`
and returns `m(0) − m(pC) = ω gᵀDg − ½ ω² ‖Jd‖²`, where `gᵀDg = -gᵀd`. Expects the trust-region
norm of `cache.inner` to be the affine-scaled one ([`affine_scaling!`](@ref)).
"""
function cauchy_step!(cache::BoundedCache, J, g, x, Δ, lb, ub)
    d, Jd, Dd = cache.pC, cache.Jw, cache.w              # Jd, Dd are scratch, rewritten in projected_step!
    d .= .-coleman_li_distances!(cache.v, x, g, lb, ub) .* g
    gᵀDg = -dot(g, d)                                    # = Σ |v_i| g_i² ≥ 0; zero only at a KKT point
    gᵀDg > 0 || return zero(gᵀDg)
    mul!(Jd, J, d)
    JdᵀJd = dot(Jd, Jd)
    mul!(Dd, cache.inner.scaling_matrix, d)
    ω = min(JdᵀJd > 0 ? gᵀDg / JdᵀJd : typemax(gᵀDg), Δ / norm(Dd))
    @inbounds for i in eachindex(d)                      # ray–box intersection; Inf for an infinite bound
        d[i] > 0 && (ω = min(ω, (ub[i] - x[i]) / d[i]))
        d[i] < 0 && (ω = min(ω, (lb[i] - x[i]) / d[i]))
    end
    d .*= ω
    return ω * gᵀDg - ω^2 / 2 * JdᵀJd
end

"""
    trust_region_step!(cache::BoundedCache, J, f, g, Δ, λ_old, x, lb, ub; β₁ = 0.1) -> (λ, predicted_reduction, ‖Dp‖)

Trust-region step in the Coleman–Li scaled norm `D = D_s |v|^{-1/2}`, projected onto the box,
`p̄ = P(x + p) − x`, and safeguarded by the Cauchy step: if `m(0) − m(p̄) < β₁ (m(0) − m(pC))`
([MMP09] condition 11) the step is moved along `p(t) = p̄ + t (pC − p̄)` to the first `t` where
the condition holds with equality. The step is written into `cache.p`; `x + p` stays in the box.
`‖Dp‖` is its length in the scaled norm.
"""
function trust_region_step!(cache::BoundedCache, J, f, g, Δ, λ_old, x, lb, ub; β₁ = 0.1)
    D = affine_scaling!(cache, x, g, lb, ub)
    λ = solve_subproblem(J, f, Δ, cache.inner, λ_old)
    decrease_cauchy = cauchy_step!(cache, J, g, x, Δ, lb, ub)
    predicted_reduction = projected_step!(cache, J, g, x, lb, ub, decrease_cauchy; β₁)
    return λ, predicted_reduction, norm(D * cache.p)
end

"""
    projected_step!(cache, J, g, x, lb, ub, decrease_cauchy; β₁ = 0.1) -> m(0) − m(p)

Project the inner step, `p̄ = P(x + p_LM) − x`, and if `m(0) − m(p̄) < β₁·decrease_cauchy` move
along `p(t) = p̄ + t (pC − p̄)` to the first `t` where equality holds. Writes `p` into `cache.p`.
"""
function projected_step!(cache::BoundedCache, J, g, x, lb, ub, decrease_cauchy; β₁ = 0.1)
    p, pC, w, Jp, Jw = cache.p, cache.pC, cache.w, cache.Jp, cache.Jw
    p .= clamp.(x .+ cache.inner.p, lb, ub) .- x         # p̄ = P(x + p_LM) − x
    mul!(Jp, J, p)
    decrease = -dot(g, p) - dot(Jp, Jp) / 2              # m(0) − m(p̄)
    decrease >= β₁ * decrease_cauchy && return decrease
    # m(0) − m(p̄ + t w) = a + b t + c t²  is concave in t, below the target β₁·decrease_cauchy at
    # t = 0 and above it at t = 1, so its smaller root lies in (0, 1].
    w .= pC .- p
    mul!(Jw, J, w)
    a, b, c = decrease, -(dot(g, w) + dot(Jp, Jw)), -dot(Jw, Jw) / 2
    r₀ = a - β₁ * decrease_cauchy
    t = -2r₀ / (b + sqrt(max(b^2 - 4c * r₀, zero(r₀))))   # smaller root in a cancellation-free form; = −r₀/b when c = 0
    t = clamp(t, zero(t), one(t))
    p .+= t .* w
    return a + t * (b + t * c)
end
