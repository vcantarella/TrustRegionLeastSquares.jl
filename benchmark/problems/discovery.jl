function find_cutest_nlls_problems(max_vars = 50)
    """Find CUTEst nonlinear least squares problems with obj='none' (residual form)"""
    # Find problems with objective="none" (these are NLLS problems)
    nlls_problems = CUTEst.select_sif_problems(objtype = "none", max_var = max_vars)
    # Test which ones actually have residuals
    valid_problems = []
    for prob_name in nlls_problems
        nlp = CUTEstModel(prob_name)
        # Check if it has constraints (residuals for NLLS)
        if nlp.meta.ncon > 0 && nlp.meta.nvar <= max_vars
            push!(valid_problems, prob_name)
        end
        finalize(nlp)
    end
    println("Found $(length(valid_problems)) valid NLLS problems")
    return valid_problems
end

function find_cutest_bounded_nlls_problems(max_vars = 50)
    """
    Find CUTEst BOUND-constrained nonlinear least squares problems.

    CUTEst encodes NLS problems with objtype="none": the residuals are the constraints
    (`cons(nlp, x)` -> F) and the residual Jacobian is the constraint Jacobian
    (`jac(nlp, x)` -> J). A bound-constrained NLS problem is therefore one of these that
    ALSO has at least one finite variable bound in `meta.lvar`/`meta.uvar`. The box is
    consumed directly by `create_cutest_functions` (which copies lvar/uvar).
    """
    candidates = CUTEst.select_sif_problems(objtype = "none", max_var = max_vars)
    valid_problems = String[]
    for prob_name in candidates
        nlp = CUTEstModel(prob_name)
        lvar, uvar = nlp.meta.lvar, nlp.meta.uvar
        # A genuinely BOUNDED variable has a finite bound AND is not FIXED (lvar < uvar).
        # Fixed variables also have finite lvar==uvar, so a naive `any(isfinite, ...)` test
        # wrongly treats fixed-variable equality systems (e.g. TRIGGER, NYSTROM5, AIRCRFTA,
        # which have only fixed/free vars and no bounds) as bound-constrained. Require a real
        # bound.
        has_real_bounds = any(
            i -> (isfinite(lvar[i]) || isfinite(uvar[i])) && lvar[i] < uvar[i],
            eachindex(lvar),
        )
        # The residuals-as-constraints interpretation (min ‖cons(x) - lcon‖ s.t. bounds) is
        # only valid when every constraint is an EQUALITY (lcon == ucon). Skip anything with
        # inequality/range constraints.
        equality_only = nlp.meta.ncon > 0 && all(nlp.meta.lcon .== nlp.meta.ucon)
        if equality_only && has_real_bounds && nlp.meta.nvar <= max_vars
            push!(valid_problems, prob_name)
        end
        finalize(nlp)
    end
    println(
        "Found $(length(valid_problems)) bound-constrained NLS problems (≤ $max_vars variables)",
    )
    return valid_problems
end


function find_nlls_problems(max_vars = 50)
    """Find NLS problems from NLSProblems.jl package"""

    # Get all available NLS problems
    all_problems = setdiff(names(NLSProblems), [:NLSProblems])

    valid_problems = []

    for prob_name in all_problems
        prob = eval(prob_name)()

        # Filter by size and check if it's a valid NLS problem
        if !unconstrained(prob)
            println("  Problem $prob_name is constrained, skipping")
            finalize(prob)
            continue
        elseif prob.meta.nvar <= max_vars #&& 
            #prob.meta.nequ > 0 &&  # Has residuals
            isa(prob, AbstractNLSModel)

            push!(valid_problems, prob_name)
            finalize(prob)
        else
            finalize(prob)
        end
    end

    println("Found $(length(valid_problems)) valid NLS problems (≤ $max_vars variables)")
    return valid_problems
end

function find_bounded_problems(max_vars = Inf)
    """Find bounded NLS problems from NLSProblems.jl package"""
    # Get all available NLS problems
    all_problems = setdiff(names(NLSProblems), [:NLSProblems])

    valid_problems = []

    for prob_name in all_problems
        prob = eval(prob_name)()

        # Filter by size and check if it's a valid NLS problem
        if !bound_constrained(prob)
            # println("  Problem $prob_name is not bound constrained, skipping")
            finalize(prob)
            continue
        elseif prob.meta.nvar <= max_vars
            isa(prob, AbstractNLSModel)

            push!(valid_problems, prob_name)
            finalize(prob)
        else
            finalize(prob)
        end
    end

    println(
        "Found $(length(valid_problems)) valid bounded NLS problems (≤ $max_vars variables)",
    )
    return valid_problems
end
