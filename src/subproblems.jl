# Trust-region subproblem   min ‖J p + f‖²  s.t.  ‖D p‖ ≤ Δ.
#
# If the Gauss–Newton step fits, λ = 0. Otherwise (JᵀJ + λD²) p = -Jᵀf with the λ that puts
# ‖Dp‖ on the boundary, found by Hebden/Moré safeguarded Newton on ψ(λ) = 1/Δ − 1/‖Dp(λ)‖
# (MINPACK lmpar). Each strategy differs only in how the damped system is factorized.

"""
    solve_subproblem(J, f, Δ, cache, λ_old; maxiters = 10, θ = 1e-4) -> λ

Solve the trust-region subproblem for the Jacobian `J` and residual `f`, writing the step into
`cache.p` and returning the damping parameter `λ` that produced it (`0` when the Gauss–Newton
step is inside the region). Stops once `(1−θ)Δ ≤ ‖Dp‖ ≤ (1+θ)Δ` or after `maxiters` Newton
iterations on λ, warm-started from `λ_old`. Well-conditioned problems need one to three
iterations; the cap of 10 (as in MINPACK `lmpar`) is what an ill-conditioned Jacobian needs.
"""
function solve_subproblem end

# Moré's bracket: λ* ∈ (0, u₀] with u₀ = ‖D⁻¹Jᵀf‖/Δ, since ‖Dp(λ)‖ ≤ ‖D⁻¹Jᵀf‖/λ.
initial_λ(λ_old, u₀) = (iszero(λ_old) ? 1e-3 * u₀ : min(λ_old, u₀), zero(u₀), u₀)

"""
    newton_update_λ(λ, ϕ, Δ, pᵀD²p, pᵀD²q, l, u) -> (λ, l, u)

One safeguarded Newton step on `ψ(λ) = 1/Δ − 1/‖Dp‖`, where `ϕ = ‖Dp‖ − Δ` and
`pᵀD²q = pᵀD²(JᵀJ + λD²)⁻¹D²p = −‖Dp‖ dϕ/dλ`. `[l, u]` brackets the root and is tightened
on the way (Moré 1978, MINPACK `lmpar`).
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

Minimum-norm solution of `J p = -f` for a wide `J`, given `F = qr(Jᵀ, ColumnNorm())`, i.e.
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
        ldiv!(LowerTriangular(F.R'), y)              # Rᵀ y = -Pᵀf
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

function solve_subproblem(
    J,
    f,
    Δ,
    cache::QRCache{QRCholStrategy},
    λ_old;
    maxiters = 10,
    θ = 1e-4,
)
    D, p, Dp, q = cache.scaling_matrix, cache.p, cache.Dp, cache.q
    ldiv!(p, cache.factorization, -f)                # Gauss–Newton step
    mul!(Dp, D, p)
    norm(Dp) <= Δ && return zero(eltype(p))
    Jᵀf = J' * f
    JᵀJ = J' * J
    λ, l, u = initial_λ(λ_old, norm(Jᵀf ./ D.diag) / Δ)
    λ_of_p = λ
    for i = 1:maxiters
        Fa = cholesky!(Hermitian(JᵀJ + λ * D^2))     # (JᵀJ + λD²) p = -Jᵀf
        p .= .-Jᵀf
        ldiv!(Fa, p)
        λ_of_p = λ
        mul!(Dp, D, p)
        ϕ = norm(Dp) - Δ
        (abs(ϕ) <= θ * Δ || i == maxiters) && break
        D²p = D * Dp
        ldiv!(q, Fa, D²p)                            # q = (JᵀJ + λD²)⁻¹ D²p
        λ, l, u = newton_update_λ(λ, ϕ, Δ, dot(Dp, Dp), dot(D²p, q), l, u)
    end
    return λ_of_p
end

function solve_subproblem(
    J,
    f,
    Δ,
    cache::QRCache{QRStrategy},
    λ_old;
    maxiters = 10,
    θ = 1e-4,
)
    D, p, Dp, q = cache.scaling_matrix, cache.p, cache.Dp, cache.q
    n, m = size(J)
    ldiv!(p, cache.factorization, -f)                # Gauss–Newton step
    mul!(Dp, D, p)
    norm(Dp) <= Δ && return zero(eltype(p))
    λ, l, u = initial_λ(λ_old, norm((J' * f) ./ D.diag) / Δ)
    λ_of_p = λ
    rhs = [-f; zeros(eltype(p), m)]
    for i = 1:maxiters
        Fa = qr!([J; √λ * D])                        # min ‖[J; √λD] p + [f; 0]‖  ⇔  (JᵀJ + λD²) p = -Jᵀf
        ldiv!(p, Fa, rhs)
        λ_of_p = λ
        mul!(Dp, D, p)
        ϕ = norm(Dp) - Δ
        (abs(ϕ) <= θ * Δ || i == maxiters) && break
        D²p = D * Dp
        ldiv!(q, LowerTriangular(Fa.R'), D²p)        # q = R⁻ᵀ D²p, so qᵀq = pᵀD²(JᵀJ + λD²)⁻¹D²p
        λ, l, u = newton_update_λ(λ, ϕ, Δ, dot(Dp, Dp), dot(q, q), l, u)
    end
    return λ_of_p
end

# Wide J: substitute p = D⁻² Jᵀ z, so that (JᵀJ + λD²) p = -Jᵀf  ⇔  (J D⁻² Jᵀ + λI) z = -f  (rows × rows).
# Then ‖Dp‖² = zᵀ J D⁻² Jᵀ z and pᵀD²(JᵀJ + λD²)⁻¹D²p = zᵀz − λ zᵀ(J D⁻² Jᵀ + λI)⁻¹z.
function solve_subproblem(
    J,
    f,
    Δ,
    cache::LQCache{LQStrategy},
    λ_old;
    maxiters = 10,
    θ = 1e-4,
)
    D, p, Dp, z, q = cache.scaling_matrix, cache.p, cache.Dp, cache.z, cache.q
    n, m = size(J)
    gauss_newton_min_norm!(p, cache.factorization, z, f)
    mul!(Dp, D, p)
    norm(Dp) <= Δ && return zero(eltype(p))
    λ, l, u = initial_λ(λ_old, norm((J' * f) ./ D.diag) / Δ)
    λ_of_p = λ
    D⁻¹Jᵀ = J' ./ D.diag
    for i = 1:maxiters
        Fa = qr!([D⁻¹Jᵀ; √λ * I(n)])                  # RᵀR = J D⁻² Jᵀ + λI
        R = UpperTriangular(Fa.R)
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

function solve_subproblem(
    J,
    f,
    Δ,
    cache::LQCache{LQCholStrategy},
    λ_old;
    maxiters = 10,
    θ = 1e-4,
)
    D, p, Dp, z, q = cache.scaling_matrix, cache.p, cache.Dp, cache.z, cache.q
    n, m = size(J)
    gauss_newton_min_norm!(p, cache.factorization, z, f)
    mul!(Dp, D, p)
    norm(Dp) <= Δ && return zero(eltype(p))
    λ, l, u = initial_λ(λ_old, norm((J' * f) ./ D.diag) / Δ)
    λ_of_p = λ
    D⁻²Jᵀ = J' ./ D.diag .^ 2
    JD⁻²Jᵀ = J * D⁻²Jᵀ
    for i = 1:maxiters
        Fa = cholesky!(Hermitian(JD⁻²Jᵀ + λ * I(n)))
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
