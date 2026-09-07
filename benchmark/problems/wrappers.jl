function create_cutest_functions(nlp)
    """Create Julia function wrappers for CUTEst NLLS problem"""
    # Get problem info
    #n = nlp.meta.nvar
    #m = nlp.meta.ncon  # Number of residuals (constraints in NLLS formulation)
    x0 = copy(nlp.meta.x0)
    bl = copy(nlp.meta.lvar)
    bu = copy(nlp.meta.uvar)
    # For CUTEst NLLS problems with objtype="none", the residuals are the constraints and the
    # residual Jacobian is the constraint Jacobian. Constraints are equalities c(x) = lcon, so
    # the residual is F(x) = cons(x) - lcon (subtracting the RHS matters when it is nonzero;
    # the Jacobian is unchanged since lcon is constant).
    lcon = copy(nlp.meta.lcon)
    residual_func(x) = NLPModels.cons(nlp, x) .- lcon
    jacobian_func(x) = Matrix(NLPModels.jac(nlp, x))
    n, m = size(jacobian_func(x0))

    # In-place, fairness-correct wrappers (parallel to create_nls_functions): evaluate the
    # residual via cons! and fill a dense Jacobian in place via the constraint coordinate
    # API (structure computed once), so the in-place solvers (NonlinearSolve, LSO) can run
    # CUTEst problems and every solver pays the same per-evaluation cost. Without these only
    # the out-of-place solvers (TRF, Scipy, PRIMA, LsqFit) could use CUTEst problems.
    residual_func!(r, x) = (NLPModels.cons!(nlp, x, r); r .-= lcon; r)
    jac_rows, jac_cols = NLPModels.jac_structure(nlp)
    jac_vals = zeros(eltype(x0), length(jac_rows))
    function jacobian_func!(J, x)
        NLPModels.jac_coord!(nlp, x, jac_vals)
        fill!(J, 0)
        @inbounds for k in eachindex(jac_rows)
            J[jac_rows[k], jac_cols[k]] = jac_vals[k]
        end
        return J
    end

    # Create objective as 0.5 * ||r||²
    obj_func(x) = 0.5 * dot(residual_func(x), residual_func(x))
    grad_func(x) = jacobian_func(x)' * residual_func(x)

    # Use Gauss-Newton approximation for Hessian
    hess_func(x) = begin
        J = jacobian_func(x)
        return J' * J
    end

    return (
        n = n,
        m = m,
        x0 = x0,
        bl = bl,
        bu = bu,
        initial_cost = obj_func(x0),
        residual_func = residual_func,
        jacobian_func = jacobian_func,
        residual_func! = residual_func!,
        jacobian_func! = jacobian_func!,
        obj_func = obj_func,
        grad_func = grad_func,
        hess_func = hess_func,
        problem = nlp.meta.name,
    )
end


function create_nls_functions(prob)
    """
    Create Julia function wrappers for NLSProblems with in-place support.

    Benchmark-fairness note: the in-place `residual_func!`/`jacobian_func!` returned here
    evaluate via NLPModels' native in-place coordinate API with buffers preallocated once,
    so every solver that consumes these closures pays the same per-evaluation cost. This
    matches how JSO-TRON evaluates the model (it consumes the NLSModel directly). One
    residual asymmetry remains and is intentional/unavoidable: TRON is handed the raw
    NLSModel while the others receive these extracted closures, so a thin call-indirection
    difference still exists between TRON and the rest. Interpret small TRON-vs-rest timing
    gaps with that in mind.
    """
    x0 = copy(prob.meta.x0)
    bl = copy(prob.meta.lvar)
    bu = copy(prob.meta.uvar)

    # Initial probe to get exact sizes
    r0 = residual(prob, x0)
    n = length(r0)
    m = length(x0)

    # 1. Out-of-place functions (Required for SciPy, PRIMA)
    residual_func(x) = residual(prob, x)
    jacobian_func(x) = Matrix(jac_residual(prob, x))

    # 2. In-place functions (Crucial for LeastSquaresOptim, SciML)
    residual_func!(r, x) = residual!(prob, x, r)

    # Truly in-place dense Jacobian fill. We compute the sparsity structure ONCE here,
    # then on each call fill the nonzeros via jac_coord_residual! (no allocation) and
    # scatter them into the caller's preallocated dense J. The previous version did
    # `J .= Matrix(jac_residual(prob, x))`, allocating a fresh dense matrix on EVERY
    # evaluation — a cost paid by every solver except JSO-TRON, which consumes the
    # NLSModel directly and uses NLPModels' native in-place operators. Equalizing this
    # path makes the benchmark compare algorithms rather than wrapper overhead.
    jac_rows, jac_cols = jac_structure_residual(prob)
    jac_vals = zeros(eltype(x0), length(jac_rows))
    function jacobian_func!(J, x)
        jac_coord_residual!(prob, x, jac_vals)
        fill!(J, 0)
        @inbounds for k in eachindex(jac_rows)
            J[jac_rows[k], jac_cols[k]] = jac_vals[k]
        end
        return J
    end

    obj_func(x) = obj(prob, x)
    grad_func(x) = grad(prob, x)

    hess_func(x) = begin
        J = jacobian_func(x)
        return J' * J
    end

    return (
        n = n,
        m = m,
        x0 = x0,
        bl = bl,
        bu = bu,
        initial_cost = obj_func(x0),
        # Return both versions!
        residual_func = residual_func,
        jacobian_func = jacobian_func,
        residual_func! = residual_func!,
        jacobian_func! = jacobian_func!,
        obj_func = obj_func,
        grad_func = grad_func,
        hess_func = hess_func,
        problem = prob.meta.name,
    )
end

"""
    crop_nls_functions(pd, k; rows)

Underdetermined variant of `pd` (a `create_*_functions` NamedTuple): keep only `k` of its
residual rows and the matching Jacobian rows. Default rows are evenly spaced over `1:pd.n`
so data-fitting problems keep their whole abscissa range and chained/banded problems keep
every variable touched (the FIRST k rows would leave zero Jacobian columns). The cropped
system has k equations in `pd.m > k` unknowns, so its solution set is generically an
(m-k)-dimensional manifold on which the cost is exactly 0. Every solver pays the same extra
row-copy per evaluation. Returns `pd` untouched when it already has at most `k` residuals
(the natively underdetermined NLSProblems).
"""
# range(1, n, length = 1) throws, so a single kept row is row 1.
function crop_nls_functions(
    pd,
    k;
    rows = k == 1 ? [1] : round.(Int, range(1, pd.n, length = k)),
)
    pd.n <= k && return pd
    rbuf = zeros(pd.n)
    Jbuf = zeros(pd.n, pd.m)
    residual_func(x) = pd.residual_func(x)[rows]
    jacobian_func(x) = pd.jacobian_func(x)[rows, :]
    residual_func!(r, x) = (pd.residual_func!(rbuf, x); r .= @view rbuf[rows]; r)
    jacobian_func!(J, x) = (pd.jacobian_func!(Jbuf, x); J .= @view Jbuf[rows, :]; J)
    obj_func(x) = 0.5 * sum(abs2, residual_func(x))
    grad_func(x) = jacobian_func(x)' * residual_func(x)
    hess_func(x) = (J = jacobian_func(x); J' * J)
    return merge(
        pd,
        (;
            n = length(rows),
            initial_cost = obj_func(pd.x0),
            residual_func,
            jacobian_func,
            residual_func!,
            jacobian_func!,
            obj_func,
            grad_func,
            hess_func,
            problem = "$(pd.problem)_crop$(length(rows))",
        ),
    )
end
