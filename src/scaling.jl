abstract type ScalingStrategy end

"""
    NoScaling()

Trust region in the Euclidean norm, `D = I`.
"""
struct NoScaling <: ScalingStrategy end

"""
    JacobianScaling()

Moré's (1978) diagonal scaling as in MINPACK `lmder`: `D_ii = max(D_ii, ‖J[:, i]‖)`, never
decreasing across iterations, so the trust region `‖Dp‖ ≤ Δ` is an ellipsoid that follows the
scale of each variable. A zero column keeps `D_ii = 1`.
"""
struct JacobianScaling <: ScalingStrategy end

# Plain loops on purpose: an assignment inside a comprehension triggers a JuliaFormatter v2
# bug that rewrites the `=` into `in`, silently changing the code's meaning.
scaling!(D::Diagonal, ::NoScaling, J) = (fill!(D.diag, one(eltype(D))); D)
function scaling!(D::Diagonal, ::JacobianScaling, J)
    @inbounds for i in axes(J, 2)
        D.diag[i] = max(D.diag[i], norm(@view J[:, i]))
        iszero(D.diag[i]) && (D.diag[i] = one(eltype(D)))
    end
    return D
end
