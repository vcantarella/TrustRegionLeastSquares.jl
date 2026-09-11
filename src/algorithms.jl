"""
    projected_gradient_norm(g, x, lb, ub)

`‖x − P(x − g)‖`, the first-order optimality measure for `lb ≤ x ≤ ub`; equals `‖g‖` exactly
when the bounds are infinite. Written component-wise as a `max`/`min` rather than
`x − clamp(x − g, lb, ub)`, which loses `g` to rounding when `|x| ≫ |g|`.
"""
projected_gradient_norm(g, x, lb, ub) = sqrt(
    sum(
        abs2(g[i] < 0 ? max(g[i], x[i] - ub[i]) : min(g[i], x[i] - lb[i])) for
        i in eachindex(x)
    ),
)

"""
    trust_region_step!(cache::SolverCache, J, f, g, Δ, λ_old, x, lb, ub) -> (λ, predicted_reduction, ‖Dp‖)

Unconstrained trust-region step into `cache.p`. The predicted reduction is `½‖Jp‖² + λ‖Dp‖²`,
the MINPACK form of [Mor78]: it equals `-gᵀp − ½‖Jp‖²` at a solution of `(JᵀJ + λD²) p = -Jᵀf`
([NW06] Theorem 4.1) but is free of that expression's cancellation for ill-conditioned `J`. The scaled step length `‖Dp‖`
drives the radius update.
"""
function trust_region_step!(cache::SolverCache, J, f, g, Δ, λ_old, x, lb, ub)
    λ = solve_subproblem(J, f, Δ, cache, λ_old)
    Jp = mul!(cache.Jp, J, cache.p)
    Dp = cache.scaling_matrix * cache.p
    return λ, dot(Jp, Jp) / 2 + λ * dot(Dp, Dp), norm(Dp)
end

"""
    lm_trust_region!(res!, jac!, x0, output_length,
                     strategy = QRCholStrategy(), scaling = NoScaling();
                     lb = -Inf, ub = Inf, kwargs...) -> (x, f, g, iter)

Minimize `½‖f(x)‖²` subject to `lb ≤ x ≤ ub` with a Levenberg–Marquardt trust-region method
([NW06] Algorithm 4.1 and §10.3; see the module docstring for the reference keys).

`res!(f, x)` writes the `output_length` residuals into `f`; `jac!(J, x)` writes the Jacobian into
the `output_length × length(x)` matrix `J`. Each iteration solves
`min ‖J p + f‖  s.t.  ‖D p‖ ≤ Δ` (see [`solve_subproblem`](@ref)); `strategy` picks the
factorization ([`QRCholStrategy`](@ref), [`QRStrategy`](@ref), [`LQStrategy`](@ref),
[`LQCholStrategy`](@ref)) and `scaling` the matrix `D` ([`NoScaling`](@ref),
[`JacobianScaling`](@ref)).

With finite bounds the step is projected onto the box and safeguarded by a generalized Cauchy
step ([MMP09], see [`trust_region_step!`](@ref)); `x0` is clamped into
the box, every iterate stays feasible and may sit on a bound, and convergence is measured by
the projected gradient `‖x − P(x − g)‖`.

# Keywords
- `lb`, `ub`: bounds, vectors of `length(x0)`; default `±Inf` (unconstrained).
- `initial_radius = 1.0`: trust-region radius Δ₀; replaced by `‖D x0‖` when
  `norm_overrides_initial_radius = true` (default) and `‖x0‖ > 1e-4`.
- `max_trust_radius = 1e12`, `min_trust_radius = 1e-8`: Δ bounds; the solver stops when Δ shrinks below the minimum.
- `step_threshold = 0.001`: accept the step when the ratio ρ of actual to predicted reduction exceeds this.
- `shrink_threshold = 0.25`, `shrink_factor = 0.25`: when ρ < shrink_threshold,
  Δ ← shrink_factor·min(Δ, 10‖Dp‖) — the radius follows the step that was tried, as in [Mor78]
  (MINPACK `lmder`), rather than the previous radius.
- `expand_threshold = 0.75`, `expand_factor = 2.0`: when ρ ≥ expand_threshold or the step was
  a full Gauss–Newton step (λ = 0), Δ ← expand_factor·‖Dp‖ — twice the step just taken, which
  equals expand_factor·Δ for a step on the boundary.
- `max_iter = 100`.
- `gtol = 1e-6`: stop when `‖x − P(x − g)‖ < gtol` (`‖g‖` unconstrained), with `g = Jᵀf`.
- `ftol = 1e-15`: stop when the cost reduction becomes relatively small, `actual < ftol·cost`, or
  the cost itself drops below `ftol` (a zero-residual solution).
- `verbose = false`: print one line per accepted step.

# Returns
`x` (solution), `f` (residuals at `x`), `g = Jᵀf`, and the iteration count.

# Example
```julia
rosen!(f, x) = (f[1] = 10(x[2] - x[1]^2); f[2] = 1 - x[1]; f)
rosen_jac!(J, x) = (J[1, 1] = -20x[1]; J[1, 2] = 10; J[2, 1] = -1; J[2, 2] = 0; J)
x, f, g, iter = lm_trust_region!(rosen!, rosen_jac!, [-1.2, 1.0], 2)                  # → (1, 1)
x, f, g, iter = lm_trust_region!(rosen!, rosen_jac!, [-1.2, 1.0], 2; ub = [0.5, 2.0])  # → (0.5, 0.25)
```
"""
function lm_trust_region!(
    res!,
    jac!,
    x0::AbstractVector{T},
    output_length::Int,
    strategy::Strategy = QRCholStrategy(),
    scaling::ScalingStrategy = NoScaling();
    lb::AbstractVector{<:Real} = fill(T(-Inf), length(x0)::Int),
    ub::AbstractVector{<:Real} = fill(T(Inf), length(x0)::Int),
    initial_radius::Real = 1.0,
    norm_overrides_initial_radius::Bool = true,
    kwargs...,
) where {T}
    all(lb .<= ub) || throw(ArgumentError("lb > ub"))
    # `length` of an abstract AbstractVector is not inferrable as an Int; assert it once, and own
    # `x` as a plain Vector, so the solve is type-stable whatever container x0 came in as (the
    # solver mutates x, so it must not alias or inherit the caller's storage).
    nvars = length(x0)::Int
    x = Vector{T}(undef, nvars)
    x .= clamp.(x0, lb, ub)                     # a feasible copy of the starting point
    f = zeros(T, output_length)
    J = zeros(T, (output_length, nvars))        # the tuple form fixes the rank of J at two for inference
    res!(f, x)
    jac!(J, x)
    cache = subproblem_cache_init(strategy, scaling, J)
    if norm_overrides_initial_radius && norm(x) > 1e-4
        initial_radius = norm(cache.scaling_matrix * x)
    end
    if any(isfinite, lb) || any(isfinite, ub)
        return trust_region_loop!(
            res!,
            jac!,
            x,
            f,
            J,
            BoundedCache(cache),
            lb,
            ub,
            initial_radius,
            scaling;
            kwargs...,
        )
    else
        return trust_region_loop!(
            res!,
            jac!,
            x,
            f,
            J,
            cache,
            lb,
            ub,
            initial_radius,
            scaling;
            kwargs...,
        )
    end
end

function trust_region_loop!(
    res!,
    jac!,
    x,
    f,
    J,
    cache,
    lb,
    ub,
    radius,
    scaling;
    max_trust_radius::Real = 1e12,
    min_trust_radius::Real = 1e-8,
    step_threshold::Real = 0.001,
    shrink_threshold::Real = 0.25,
    expand_threshold::Real = 0.75,
    shrink_factor::Real = 0.25,
    expand_factor::Real = 2.0,
    max_iter::Int = 100,
    gtol::Real = 1e-6,
    ftol::Real = 1e-15,
    verbose::Bool = false,
)
    T = eltype(x)
    x_trial, f_trial = similar(x), similar(f)
    g = J' * f
    cost = dot(f, f) / 2
    λ = zero(T)
    projected_gradient_norm(g, x, lb, ub) < gtol && return x, f, g, 0
    for iter = 1:max_iter
        λ, predicted_reduction, step_norm =
            trust_region_step!(cache, J, f, g, radius, λ, x, lb, ub)
        x_trial .= clamp.(x .+ cache.p, lb, ub)
        res!(f_trial, x_trial)
        cost_trial = dot(f_trial, f_trial) / 2
        actual_reduction = cost - cost_trial
        ρ = predicted_reduction > 0 ? actual_reduction / predicted_reduction : -one(T)
        if ρ < shrink_threshold                           # radius rules of [Mor78], MINPACK lmder
            radius = shrink_factor * min(radius, 10 * step_norm)
        elseif ρ >= expand_threshold || iszero(λ)
            radius = min(max_trust_radius, expand_factor * step_norm)
        end
        if ρ >= step_threshold
            x .= x_trial
            f .= f_trial
            cost = cost_trial
            jac!(J, x)
            mul!(g, J', f)
            gnorm = projected_gradient_norm(g, x, lb, ub)
            verbose && println("iter $iter  cost $cost  ‖∇‖ $gnorm  radius $radius")
            # ftol is a RELATIVE test: the reduction is compared against the cost itself, so it
            # does not fire merely because the cost has become small (comparing against
            # `max(cost, 1)` makes it absolute below cost 1 and stops far from a stationary point).
            # `cost < ftol` is the separate zero-residual exit.
            (gnorm < gtol || actual_reduction < ftol * cost || cost < ftol) &&
                return x, f, g, iter
            update_cache!(cache, J, scaling)
        else
            verbose && println("iter $iter  step rejected, ρ = $ρ")
        end
        radius < min_trust_radius && return x, f, g, iter
    end
    return x, f, g, max_iter
end
