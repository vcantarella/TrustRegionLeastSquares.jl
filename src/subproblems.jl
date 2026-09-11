# Trust-region subproblem   min ‖J p + f‖²  s.t.  ‖D p‖ ≤ Δ.
#
# [NW06] Theorem 4.1: a step solves this iff (JᵀJ + λD²) p = -Jᵀf for some λ ≥ 0 with
# λ(Δ - ‖Dp‖) = 0. So if the Gauss–Newton step already fits, λ = 0; otherwise λ is the root of
# ‖Dp(λ)‖ = Δ, found by safeguarded Newton on the near-linear φ₂(λ) = 1/Δ - 1/‖Dp(λ)‖ of
# [NW06] §4.3, with the bracket and the initial guess of [Mor78] (MINPACK lmpar).
#
# Each strategy differs only in how the damped system is factorized: the QR strategies follow the
# augmented-matrix implementation of [NW06] §10.3, and the LQ strategies solve through J Jᵀ for
# underdetermined problems, following [CJK26] Appendix B and [SGJ26].

"""
    solve_subproblem(J, f, Δ, cache, λ_old; maxiters = 10, θ = 1e-4) -> λ

Solve the trust-region subproblem for the Jacobian `J` and residual `f`, writing the step into
`cache.p` and returning the damping parameter `λ` that produced it (`0` when the Gauss–Newton
step is inside the region). Stops once `(1−θ)Δ ≤ ‖Dp‖ ≤ (1+θ)Δ` or after `maxiters` Newton
iterations on λ, warm-started from `λ_old`. Well-conditioned problems need one to three
iterations; the cap of 10 (as in MINPACK `lmpar`) is what an ill-conditioned Jacobian needs.
"""
function solve_subproblem end

# Moré's bracket ([Mor78], MINPACK lmpar): λ* ∈ (0, u₀] with u₀ = ‖D⁻¹Jᵀf‖/Δ, since
# ‖Dp(λ)‖ ≤ ‖D⁻¹Jᵀf‖/λ.
initial_λ(λ_old, u₀) = (iszero(λ_old) ? 1e-3 * u₀ : min(λ_old, u₀), zero(u₀), u₀)

"""
    newton_update_λ(λ, ϕ, Δ, pᵀD²p, pᵀD²q, l, u) -> (λ, l, u)

One safeguarded Newton step on `φ₂(λ) = 1/Δ − 1/‖Dp‖`, the reformulation of `‖Dp(λ)‖ = Δ` that is
nearly linear in λ near the root and so suits Newton's method ([NW06] §4.3). Here
`ϕ = ‖Dp‖ − Δ` and `pᵀD²q = pᵀD²(JᵀJ + λD²)⁻¹D²p = −‖Dp‖ dϕ/dλ`. `[l, u]` brackets the root and
is tightened on the way, which is the safeguard of [Mor78] (MINPACK `lmpar`).
"""
function newton_update_λ(λ, ϕ, Δ, pᵀD²p, pᵀD²q, l, u)
    ϕ < 0 ? (u = λ) : (l = λ)
    λ += ϕ / Δ * pᵀD²p / pᵀD²q
    l <= λ <= u || (λ = max(l + 0.01 * (u - l), sqrt(l * u)))
    return λ, l, u
end

"Numerical rank of a pivoted QR from the diagonal of `R` (the rule `rank(::QRPivoted)` uses on Julia ≥ 1.12)."
function numerical_rank(F::QRPivoted)
    k = minimum(size(F.factors))
    tol = k * eps(real(eltype(F.factors))) * abs(F.factors[1, 1])
    return count(i -> abs(F.factors[i, i]) > tol, 1:k)
end

"""
    gauss_newton_min_norm!(p, F, y, f)

Minimum-norm solution of `J p = -f` for a wide `J`, given `F = qr(Jᵀ, ColumnNorm())` — the LQ
factorization of [CJK26] Appendix B, which keeps the condition number unsquared. That is,
`J = P Rᵀ Qᵀ`: solve `Rᵀ y = -Pᵀf`, then `p = Q [y; 0]`. When `R` is rank-deficient the leading
`r` rows of `R` get a second QR (complete orthogonal decomposition), which keeps `p` minimum-norm.
`y` is a rows-length workspace.
"""
function gauss_newton_min_norm!(p, F::QRPivoted, y, f)
    n = length(y)
    Pᵀf = view(f, F.p)
    y .= .-Pᵀf
    r = numerical_rank(F)
    fill!(p, 0)
    if r == n
        ldiv!(LowerTriangular(view(F.factors, 1:n, 1:n)'), y)    # Rᵀ y = -Pᵀf
        copyto!(p, y)
    else
        R₁ᵀ = Matrix(F.R[1:r, :]')                   # [R₁₁ R₁₂]ᵀ = Z Tᵀ   (qr! destroys its input)
        Fc = qr!(R₁ᵀ)
        Zᵀy = (Fc.Q'*y)[1:r]
        ldiv!(UpperTriangular(Fc.R), Zᵀy)            # Tᵀ v = Zᵀ y
        copyto!(p, Zᵀy)
    end
    return lmul!(F.Q, p)                             # p = Q [y; 0]
end

function solve_subproblem(J, f, Δ, cache::QRCholCache, λ_old; maxiters = 10, θ = 1e-4)
    D, p, Dp, D²p, q = cache.scaling_matrix, cache.p, cache.Dp, cache.D²p, cache.q
    JᵀJ, damped, Jᵀf = cache.JᵀJ, cache.damped, cache.Jᵀf
    ldiv!(p, cache.factorization, -f)                # Gauss–Newton step
    mul!(Dp, D, p)
    norm(Dp) <= Δ && return zero(eltype(p))
    mul!(Jᵀf, J', f)
    λ, l, u = initial_λ(λ_old, norm(Jᵀf ./ D.diag) / Δ)
    λ_of_p = λ
    for i = 1:maxiters
        # (JᵀJ + λD²) p = -Jᵀf. Damping only adds λd² to the diagonal, so the system is assembled
        # into the cache's own buffer and factorized there — no matrix is allocated per λ.
        copyto!(damped, JᵀJ)
        @inbounds for j in axes(damped, 1)
            damped[j, j] += λ * D.diag[j]^2
        end
        Fa = cholesky!(Hermitian(damped))
        p .= .-Jᵀf
        ldiv!(Fa, p)
        λ_of_p = λ
        mul!(Dp, D, p)
        ϕ = norm(Dp) - Δ
        (abs(ϕ) <= θ * Δ || i == maxiters) && break
        mul!(D²p, D, Dp)
        ldiv!(q, Fa, D²p)                            # q = (JᵀJ + λD²)⁻¹ D²p
        λ, l, u = newton_update_λ(λ, ϕ, Δ, dot(Dp, Dp), dot(D²p, q), l, u)
    end
    return λ_of_p
end

function solve_subproblem(J, f, Δ, cache::QRCache, λ_old; maxiters = 10, θ = 1e-4)
    D, p, Dp, D²p, q = cache.scaling_matrix, cache.p, cache.Dp, cache.D²p, cache.q
    A, rhs = cache.augmented, cache.rhs
    n, m = size(J)
    ldiv!(p, cache.factorization, -f)                # Gauss–Newton step
    mul!(Dp, D, p)
    norm(Dp) <= Δ && return zero(eltype(p))
    mul!(cache.Jᵀf, J', f)
    λ, l, u = initial_λ(λ_old, norm(cache.Jᵀf ./ D.diag) / Δ)
    λ_of_p = λ
    fill!(rhs, 0)
    @views rhs[1:n] .= .-f                           # [-f; 0], fixed across the λ-iteration
    for i = 1:maxiters
        # min ‖[J; √λ D] p − [-f; 0]‖ ⇔ (JᵀJ + λD²) p = -Jᵀf. `qr!` consumes the augmented matrix,
        # so both blocks are rewritten into the cache's buffer each time instead of allocated.
        copyto!(view(A, 1:n, :), J)
        fill!(view(A, (n+1):(n+m), :), 0)
        @inbounds for j = 1:m
            A[n+j, j] = √λ * D.diag[j]
        end
        Fa = qr!(A)
        ldiv!(p, Fa, rhs)
        λ_of_p = λ
        mul!(Dp, D, p)
        ϕ = norm(Dp) - Δ
        (abs(ϕ) <= θ * Δ || i == maxiters) && break
        mul!(D²p, D, Dp)
        # q = R⁻ᵀ D²p, so qᵀq = pᵀD²(JᵀJ + λD²)⁻¹D²p. `Fa.R` would copy the factor out; the view
        # aliases it, and the triangular wrapper ignores everything below the diagonal.
        R = UpperTriangular(view(Fa.factors, 1:m, 1:m))
        ldiv!(q, R', D²p)
        λ, l, u = newton_update_λ(λ, ϕ, Δ, dot(Dp, Dp), dot(q, q), l, u)
    end
    return λ_of_p
end

# Wide J ([CJK26] Appendix B, [SGJ26]): substitute p = D⁻² Jᵀ z, so that the damped system becomes
# (JᵀJ + λD²) p = -Jᵀf  ⇔  (J D⁻² Jᵀ + λI) z = -f, which is rows × rows rather than cols × cols and
# full rank even when J is not. The step it returns is the regularized minimum-norm one.
# Then ‖Dp‖² = zᵀ J D⁻² Jᵀ z and pᵀD²(JᵀJ + λD²)⁻¹D²p = zᵀz − λ zᵀ(J D⁻² Jᵀ + λI)⁻¹z.
function solve_subproblem(J, f, Δ, cache::LQCache, λ_old; maxiters = 10, θ = 1e-4)
    D, p, Dp, z, q = cache.scaling_matrix, cache.p, cache.Dp, cache.z, cache.q
    D⁻¹Jᵀ, A = cache.D⁻¹Jᵀ, cache.augmented
    n, m = size(J)
    gauss_newton_min_norm!(p, cache.factorization, z, f)
    mul!(Dp, D, p)
    norm(Dp) <= Δ && return zero(eltype(p))
    mul!(cache.Jᵀf, J', f)
    λ, l, u = initial_λ(λ_old, norm(cache.Jᵀf ./ D.diag) / Δ)
    λ_of_p = λ
    D⁻¹Jᵀ .= J' ./ D.diag
    for i = 1:maxiters
        # RᵀR = J D⁻² Jᵀ + λI, from a QR of [D⁻¹Jᵀ; √λ I] assembled in the cache's buffer.
        copyto!(view(A, 1:m, :), D⁻¹Jᵀ)
        fill!(view(A, (m+1):(m+n), :), 0)
        @inbounds for j = 1:n
            A[m+j, j] = √λ
        end
        Fa = qr!(A)
        R = UpperTriangular(view(Fa.factors, 1:n, 1:n))   # aliases the factor; Fa.R would copy it
        z .= .-f
        ldiv!(R', z)
        ldiv!(R, z)                                  # z = -(J D⁻² Jᵀ + λI)⁻¹ f
        mul!(p, J', z)
        p ./= D.diag .^ 2                            # p = D⁻² Jᵀ z
        λ_of_p = λ
        mul!(Dp, D, p)
        ϕ = norm(Dp) - Δ
        (abs(ϕ) <= θ * Δ || i == maxiters) && break
        ldiv!(q, R', z)                              # q = R⁻ᵀ z, so qᵀq = zᵀ(J D⁻² Jᵀ + λI)⁻¹ z
        λ, l, u = newton_update_λ(λ, ϕ, Δ, dot(Dp, Dp), dot(z, z) - λ * dot(q, q), l, u)
    end
    return λ_of_p
end

function solve_subproblem(J, f, Δ, cache::LQCholCache, λ_old; maxiters = 10, θ = 1e-4)
    D, p, Dp, z, q = cache.scaling_matrix, cache.p, cache.Dp, cache.z, cache.q
    D⁻²Jᵀ, gram, damped = cache.D⁻²Jᵀ, cache.gram, cache.damped
    gauss_newton_min_norm!(p, cache.factorization, z, f)
    mul!(Dp, D, p)
    norm(Dp) <= Δ && return zero(eltype(p))
    mul!(cache.Jᵀf, J', f)
    λ, l, u = initial_λ(λ_old, norm(cache.Jᵀf ./ D.diag) / Δ)
    λ_of_p = λ
    D⁻²Jᵀ .= J' ./ D.diag .^ 2
    mul!(gram, J, D⁻²Jᵀ)                             # J D⁻² Jᵀ, formed once per solve
    for i = 1:maxiters
        copyto!(damped, gram)
        @inbounds for j in axes(damped, 1)
            damped[j, j] += λ
        end
        Fa = cholesky!(Hermitian(damped))
        z .= .-f
        ldiv!(Fa, z)                                 # z = -(J D⁻² Jᵀ + λI)⁻¹ f
        mul!(p, D⁻²Jᵀ, z)
        λ_of_p = λ
        mul!(Dp, D, p)
        ϕ = norm(Dp) - Δ
        (abs(ϕ) <= θ * Δ || i == maxiters) && break
        ldiv!(q, Fa, z)                              # q = (J D⁻² Jᵀ + λI)⁻¹ z
        λ, l, u = newton_update_λ(λ, ϕ, Δ, dot(Dp, Dp), dot(z, z) - λ * dot(z, q), l, u)
    end
    return λ_of_p
end
