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
