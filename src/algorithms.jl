"""
    lm_trust_region(
        res::Function,
        jac::Function,
        x0::Array{T},
        subproblem_strategy::SubProblemStrategy = SVDSolve(),
        scaling_strategy::ScalingStrategy = NoScaling();
        initial_radius::Real = 1.0,
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
        norm_overrides_initial_radius::Bool = true,
    ) where {T}

Solves a nonlinear least squares problem using a Levenberg-Marquardt style trust-region algorithm.
Minimizes `0.5 * ||f(x)||^2`.

# Arguments
- `res::Function`: The residual function `f(x)` returning a vector.
- `jac::Function`: The Jacobian function `J(x)` returning the Jacobian matrix.
- `x0::Array{T}`: Initial guess for the solution.
- `subproblem_strategy`: Strategy for solving the subproblem (default: `SVDSolve()`).
                         Options: `SVDSolve()`, `QRSolve()`, `QRrecursiveSolve()`.
- `scaling_strategy`: Strategy for scaling variables (default: `NoScaling()`).

# Keywords
- `initial_radius`: Initial trust region radius (default: 1.0).
- `max_trust_radius`: Maximum allowed radius.
- `min_trust_radius`: Minimum radius before termination.
- `step_threshold`: Minimum relative improvement to accept a step.
- `max_iter`: Maximum number of iterations.
- `gtol`: Gradient tolerance for convergence (`norm(g) < gtol`).
- `ftol`: Function tolerance for convergence.
- `norm_overrides_initial_radius`: If true, `initial_radius` is scaled by the norm of the first step.

# Returns
- `x`: Optimized parameters.
- `f`: Final residuals.
- `g`: Final gradient.
- `iter`: Number of iterations performed.
"""
function lm_trust_region(
    res::Function,
    jac::Function,
    x0::Array{T},
    subproblem_strategy::SubProblemStrategy = SVDSolve(),
    scaling_strategy::ScalingStrategy = NoScaling();
    initial_radius::Real = 1.0,
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
    norm_overrides_initial_radius::Bool = true,
) where {T}

    # Initialize
    x = copy(x0)
    x_trial = copy(x0) # Buffer for candidate step
    f = res(x)
    J = jac(x)
    Jδ = Vector{T}(undef, length(f))
    cost = 0.5 * dot(f, f)
    g = J' * f
    λ_old = zero(T)
    cache = SubproblemCache(subproblem_strategy, scaling_strategy, J)
    if norm_overrides_initial_radius && norm(x0) > 1e-4
        initial_radius = norm(cache.scaling_matrix * x0)
    end
    radius = initial_radius
    # Check initial convergence
    if norm(g) < gtol
        println("Initial guess satisfies gradient tolerance")
        return x, f, g, 0
    end
    #iterations
    for iter = 1:max_iter
        # Compute step using QR-facorization
        λ, δ = solve_subproblem(subproblem_strategy, J, f, radius, cache, λ_old)
        λ_old = λ
        # Evaluate new point
        @. x_trial = x + δ
        f_new = res(x_trial)
        cost_new = 0.5 * dot(f_new, f_new)
        # Compute reduction ratio
        actual_reduction = cost - cost_new
        # Predicted reduction using QR factorization
        mul!(Jδ, J, δ)
        #predicted_reduction = -dot(g, δ) - 0.5 * dot(Jδ, Jδ)
        predicted_reduction = 0.5*dot(Jδ, Jδ)+λ*dot(δ, δ)
        if predicted_reduction <= 0 #this potentially means the δ is wrong but we leave some margin
            println("Non-positive predicted reduction, shrinking radius")
            radius *= shrink_factor
            continue
        end
        # the reduction ratio
        ρ = actual_reduction / predicted_reduction
        # Update trust region radius
        if (ρ >= expand_threshold) && (λ > 0)
            radius = min(max_trust_radius, expand_factor * radius)
        elseif ρ < shrink_threshold
            radius *= shrink_factor
        end
        # Accept or reject step
        if ρ >= step_threshold
            @. x = x_trial
            @. f = f_new
            cost = cost_new
            J = jac(x)
            mul!(g, J', f)
            println(
                "Iteration: $iter, cost: $cost, norm(g): $(norm(g, 2)), radius: $radius",
            )
            # Check convergence
            if norm(g, 2) < gtol
                println("Gradient convergence criterion reached")
                return x, f, g, iter
            end
            if actual_reduction < ftol * max(cost, 1.0)
                println("Function tolerance criterion reached")
                return x, f, g, iter
            end
            # update cache
            factorize!(cache, subproblem_strategy, J)
            cache.scaling_matrix .= scaling(scaling_strategy, J)
            # To verify allocations reduced, check cache.J_buffer
            if cache.J_buffer === nothing
                @warn "Optimized path NOT taken: J is $(typeof(J))"
            end
        else
            println("Step rejected, ρ = $ρ")
        end
        # Check trust region size
        if radius < min_trust_radius
            println("Trust region radius below minimum")
            return x, f, g, iter
        end
    end
    println("Maximum number of iterations reached")
    return x, f, g, max_iter
end

"""
    lm_trust_region(
        res::Function,
        jac::Function,
        x0::Array{T},
        subproblem_strategy::SubProblemStrategy = SVDSolve(),
        scaling_strategy::ScalingStrategy = NoScaling();
        initial_radius::Real = 1.0,
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
        norm_overrides_initial_radius::Bool = true,
    ) where {T}

Solves a nonlinear least squares problem using a Levenberg-Marquardt style trust-region algorithm.
Minimizes `0.5 * ||f(x)||^2`.

# Arguments
- `res::Function`: The residual function `f(x)` returning a vector.
- `jac::Function`: The Jacobian function `J(x)` returning the Jacobian matrix.
- `x0::Array{T}`: Initial guess for the solution.
- `subproblem_strategy`: Strategy for solving the subproblem (default: `SVDSolve()`).
                         Options: `SVDSolve()`, `QRSolve()`, `QRrecursiveSolve()`.
- `scaling_strategy`: Strategy for scaling variables (default: `NoScaling()`).

# Keywords
- `initial_radius`: Initial trust region radius (default: 1.0).
- `max_trust_radius`: Maximum allowed radius.
- `min_trust_radius`: Minimum radius before termination.
- `step_threshold`: Minimum relative improvement to accept a step.
- `max_iter`: Maximum number of iterations.
- `gtol`: Gradient tolerance for convergence (`norm(g) < gtol`).
- `ftol`: Function tolerance for convergence.
- `norm_overrides_initial_radius`: If true, `initial_radius` is scaled by the norm of the first step.
- `verbose`: Print per-iteration progress (default `false`; keep off when timing).
- `callback`: Optional function invoked once per iteration with a NamedTuple
  `(iter, x, x_trial, δ, λ, radius, ρ, accepted, cost)` for tracing/animation
  (default `nothing`; `ρ = NaN` when the predicted reduction was non-positive).

# Returns
- `x`: Optimized parameters.
- `f`: Final residuals.
- `g`: Final gradient.
- `iter`: Number of iterations performed.
"""
function lm_trust_region!(
    res!::Function,
    jac!::Function,
    x0::Array{T},
    output_length::Int,
    subproblem_strategy::SubProblemStrategy = SVDSolve(),
    scaling_strategy::ScalingStrategy = NoScaling();
    initial_radius::Real = 1.0,
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
    norm_overrides_initial_radius::Bool = true,
    verbose::Bool = false,
    callback = nothing,
) where {T}

    # Initialize
    x = copy(x0)
    x_trial = copy(x0) # Buffer for candidate step
    f = zeros(eltype(x0), output_length)
    f_new = copy(f)
    J = zeros(eltype(x0), (output_length, length(x0)))
    res!(f, x)
    jac!(J, x)
    Jδ = Vector{T}(undef, length(f))
    cost = 0.5 * dot(f, f)
    g = J' * f
    λ_old = 0.0
    cache = SubproblemCache(subproblem_strategy, scaling_strategy, J)
    if norm_overrides_initial_radius && norm(x0) > 1e-4
        initial_radius = norm(cache.scaling_matrix * x0)
    end
    radius = initial_radius
    # Check initial convergence
    if norm(g) < gtol
        verbose && println("Initial guess satisfies gradient tolerance")
        return x, f, g, 0
    end
    #iterations
    for iter = 1:max_iter
        # Compute step using QR-facorization
        λ, δ = solve_subproblem(subproblem_strategy, J, f, radius, cache, λ_old)
        λ_old = λ
        radius_used = radius # radius the step was computed with, for the callback
        # Evaluate new point
        @. x_trial = x + δ
        res!(f_new, x_trial)
        cost_new = 0.5 * dot(f_new, f_new)
        # Compute reduction ratio
        actual_reduction = cost - cost_new
        # Predicted reduction using QR factorization
        mul!(Jδ, J, δ)
        #predicted_reduction = -dot(g, δ) - 0.5 * dot(Jδ, Jδ)
        predicted_reduction = 0.5*dot(Jδ, Jδ)+λ*dot(δ, δ)
        if predicted_reduction <= 0 #this potentially means the δ is wrong but we leave some margin
            verbose && println("Non-positive predicted reduction, shrinking radius")
            callback === nothing || callback((;
                iter, x = copy(x), x_trial = copy(x_trial), δ = copy(δ),
                λ, radius = radius_used, ρ = NaN, accepted = false, cost,
            ))
            radius *= shrink_factor
            continue
        end
        # the reduction ratio
        ρ = actual_reduction / predicted_reduction
        # Update trust region radius
        if (ρ >= expand_threshold) && (λ > 0)
            radius = min(max_trust_radius, expand_factor * radius)
        elseif ρ < shrink_threshold
            radius *= shrink_factor
        end
        # Accept or reject step
        accepted = ρ >= step_threshold
        # Invoked before x is overwritten so `x` is the step's origin, and before the
        # convergence returns so the final step is still reported.
        callback === nothing || callback((;
            iter, x = copy(x), x_trial = copy(x_trial), δ = copy(δ),
            λ, radius = radius_used, ρ, accepted, cost,
        ))
        if accepted
            @. x = x_trial
            @. f = f_new
            cost = cost_new
            jac!(J, x)
            mul!(g, J', f)
            verbose && println(
                "Iteration: $iter, cost: $cost, norm(g): $(norm(g, 2)), radius: $radius",
            )
            # Check convergence
            if norm(g, 2) < gtol
                verbose && println("Gradient convergence criterion reached")
                return x, f, g, iter
            end
            if actual_reduction < ftol * max(cost, 1.0)
                verbose && println("Function tolerance criterion reached")
                return x, f, g, iter
            end
            # update cache
            factorize!(cache, subproblem_strategy, J)
            cache.scaling_matrix .= scaling(scaling_strategy, J)
            # To verify allocations reduced, check cache.J_buffer
            if cache.J_buffer === nothing
                @warn "Optimized path NOT taken: J is $(typeof(J))"
            end
        else
            verbose && println("Step rejected, ρ = $ρ")
        end
        # Check trust region size
        if radius < min_trust_radius
            verbose && println("Trust region radius below minimum")
            return x, f, g, iter
        end
    end
    verbose && println("Maximum number of iterations reached")
    return x, f, g, max_iter
end
