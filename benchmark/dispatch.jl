# Generic wrapper for prob_data residual
# Parametric wrapper encodes sizes at the type level so NLLSsolver can use
# static sizes while we still carry a runtime `prob_data` instance.
struct ProbDataResidual{NB,M} <: NLLSsolver.AbstractResidual
    prob_data::Any
end

# Construct a properly-parameterized wrapper from runtime `prob_data`.
# NB is the number of variable blocks the residual depends on (we use 1).
ProbDataResidual(prob_data) = ProbDataResidual{1,prob_data.n}(prob_data)

Base.eltype(::ProbDataResidual) = Float64
NLLSsolver.ndeps(::ProbDataResidual{NB,M}) where {NB,M} = static(NB) # number of variable blocks
NLLSsolver.nres(::ProbDataResidual{NB,M}) where {NB,M} = static(M)   # residual length
NLLSsolver.varindices(::ProbDataResidual{NB,M}) where {NB,M} = SVector{NB}(1:NB)

function NLLSsolver.getvars(::ProbDataResidual{NB,M}, vars::Vector) where {NB,M}
    # Return the variable blocks this residual depends on as a tuple
    return (vars[1],)
end

function NLLSsolver.computeresidual(res::ProbDataResidual{NB,M}, vars...) where {NB,M}
    # `vars` may be passed as a tuple of variable blocks or explicit args
    vb = length(vars) == 1 ? vars[1] : vars
    x = collect(vb)
    return SVector{M}(res.prob_data.residual_func(x))
end

# Provide an analytic residual+jacobian implementation so the solver uses it
# rather than attempting autodiff through potentially non-Dual-friendly code.
function NLLSsolver.computeresjacstatic(
    varflags::StaticInt{NB},
    res::ProbDataResidual{NB,M},
    vars,
) where {NB,M}
    vb = isa(vars, Tuple) ? vars[1] : vars
    x = collect(vb)
    r = SVector{M}(res.prob_data.residual_func(x))
    J = res.prob_data.jacobian_func(x)
    return r, J
end

function test_solver_on_problem(solver_name, solver_func, prob_data, prob, max_iter = 100)
    """Test a single solver on a problem"""
    try
        if solver_name == "TRLS" || startswith(solver_name, "LM-")
            # TrustRegionLeastSquares. The label picks the variant: "TRLS" / "LM-QR" -> QRStrategy,
            # "LM-QRChol" -> QRCholStrategy, "LM-LQ" / "LM-LQChol" -> the LQ family (residuals <=
            # variables only); a "-scaled" suffix switches on JacobianScaling. Bounds are always
            # passed: infinite bounds take the unconstrained path, finite ones the projected step.
            scaling_strategy =
                contains(lowercase(solver_name), "scaled") ? TRLS.JacobianScaling() :
                TRLS.NoScaling()
            subproblem_strategy =
                contains(solver_name, "LQChol") ? TRLS.LQCholStrategy() :
                contains(solver_name, "QRChol") ? TRLS.QRCholStrategy() :
                contains(solver_name, "LQ") ? TRLS.LQStrategy() : TRLS.QRStrategy()
            solve_once() = solver_func(
                prob_data.residual_func!,
                prob_data.jacobian_func!,
                prob_data.x0,
                prob_data.n,
                subproblem_strategy,
                scaling_strategy;
                lb = prob_data.bl,
                ub = prob_data.bu,
                max_iter = max_iter,
                gtol = 1e-8,
                ftol = 1e-8,
            )
            x_opt, r_opt, g_opt, iterations = solve_once()
            t = minimum(@be solve_once()).time
            final_cost = 0.5 * dot(r_opt, r_opt)
            converged =
                TRLS.projected_gradient_norm(g_opt, x_opt, prob_data.bl, prob_data.bu) <
                1e-8
        elseif solver_name in ["PRIMA-NEWUOA", "PRIMA-BOBYQA"]
            # Use objective-only interface
            if solver_name == "PRIMA-NEWUOA"
                # because PRIMA counts function evaluations we count an iteration as running the
                #function model n times. I am evaluating time here once because they are the ones that take the longer.
                result = PRIMA.newuoa(
                    prob_data.obj_func,
                    prob_data.x0;
                    maxfun = max_iter*prob_data.m,
                    rhoend = 1e-8,
                )
                t = minimum(
                    @be PRIMA.newuoa(
                        prob_data.obj_func,
                        prob_data.x0;
                        maxfun = max_iter*prob_data.m,
                        rhoend = 1e-8,
                    )
                ).time
            else
                lb = prob_data.bl
                ub = prob_data.bu
                result = PRIMA.bobyqa(
                    prob_data.obj_func,
                    prob_data.x0;
                    xl = lb,
                    xu = ub,
                    maxfun = max_iter*prob_data.m,
                    rhoend = 1e-8,
                )
                t = minimum(
                    @be PRIMA.bobyqa(
                        prob_data.obj_func,
                        prob_data.x0;
                        xl = lb,
                        xu = ub,
                        maxfun = max_iter*prob_data.m,
                        rhoend = 1e-8,
                    )
                ).time
            end
            x_opt = result[1]
            final_cost = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = result[2].nf
            converged = PRIMA.issuccess(result[2])
        elseif contains(solver_name, "NonlinearSolve-")
            n_res!(du, u, p) = prob_data.residual_func!(du, u)
            nl_jac!(J, u, p) = prob_data.jacobian_func!(J, u)
            nl_func = NonlinearFunction(
                n_res!,
                jac = nl_jac!,
                resid_prototype = zeros(prob_data.n),
            )
            lb = prob_data.bl
            ub = prob_data.bu
            if any(isfinite, lb) || any(isfinite, ub)
                prob_nl = NonlinearLeastSquaresProblem(
                    nl_func,
                    copy(prob_data.x0);
                    lb = lb,
                    ub = ub,
                )
            else
                prob_nl = NonlinearLeastSquaresProblem(nl_func, copy(prob_data.x0))
            end
            # Fairness: NonlinearSolve's default abstol for Float64 is 3e-13 on ‖F‖ —
            # unreachable on nonzero-residual problems, so runs would only stop via the
            # 32-step stall window or maxiters, inflating iteration counts and time
            # relative to the 1e-8 tolerances given to the other solvers. abstol=1e-8
            # matches the suite-wide value (see README "Benchmark limitations"); its
            # stall detector then also fires at the 1e-8 step scale.
            sol = solve(prob_nl, solver_func(); maxiters = max_iter, abstol = 1e-8)
            t = minimum(
                @be solve(prob_nl, solver_func(); maxiters = max_iter, abstol = 1e-8)
            ).time
            x_opt = sol.u
            final_cost = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = sol.stats.nsteps
            converged = SciMLBase.successful_retcode(sol)
            # elseif contains(solver_name, "MINPACK-")
            #     n_res(u, p) = residual(nlp, u)
            #     nl_jac(u, p) = Matrix(jac_residual(prob, u))
            #     nl_func = NonlinearFunction(n_res, jac = nl_jac)
            #     prob_nl = NonlinearLeastSquaresProblem(nl_func, prob_data.x0)
            #     sol = solve(prob_nl, solver_func(); maxiters = max_iter)
            #     t = @elapsed solve(prob_nl, solver_func(); maxiters = max_iter)
            #     x_opt = sol.u
            #     final_cost = prob_data.obj_func(x_opt)
            #     g_opt = prob_data.grad_func(x_opt)
            #     iterations = sol.stats.nsteps
            #     converged = SciMLBase.successful_retcode(sol)
        elseif contains(solver_name, "JSO-")
            # Fairness: TRON's defaults stop at atol + rtol*‖g₀‖ with atol=rtol=√eps — a
            # per-problem-varying threshold — and cap wall-clock at max_time=30s, which no
            # other solver has. atol=1e-8, rtol=0 matches the suite-wide 1e-8 absolute
            # tolerance; max_time=Inf removes the hidden cap.
            stats = solver_func(
                prob;
                max_iter = max_iter,
                atol = 1e-8,
                rtol = 0.0,
                max_time = Inf,
            )
            t = minimum(
                @be solver_func(
                    prob;
                    max_iter = max_iter,
                    atol = 1e-8,
                    rtol = 0.0,
                    max_time = Inf,
                )
            ).time
            x_opt = stats.solution
            final_cost = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = stats.iter
            converged = stats.status == :first_order
        elseif contains(solver_name, "LSO-")
            # Fairness: without explicit kwargs LSO runs its own defaults — 1000
            # iterations, while everyone else gets max_iter. Pass the shared budget and
            # the suite-wide 1e-8 gradient tolerance (its x_tol/f_tol already default
            # to 1e-8, matching SciPy's xtol/ftol).
            res = LeastSquaresOptim.optimize!(
                LeastSquaresProblem(
                    x = copy(prob_data.x0),
                    f! = prob_data.residual_func!,
                    g! = prob_data.jacobian_func!,
                    output_length = prob_data.n,
                ),
                solver_func,
                lower = prob_data.bl,
                upper = prob_data.bu,
                iterations = max_iter,
                g_tol = 1e-8,
                x_tol = 0.0,
            )
            # Construct the LeastSquaresProblem in the Chairmarks SETUP phase (run per
            # sample but NOT timed) so we measure only the solve, matching the other
            # solvers. A fresh problem per sample is required because optimize! mutates
            # its `x`, so a single hoisted problem would start from the solved point.
            t = minimum(
                @be LeastSquaresProblem(
                    x = copy(prob_data.x0),
                    f! = prob_data.residual_func!,
                    g! = prob_data.jacobian_func!,
                    output_length = prob_data.n,
                ) p -> LeastSquaresOptim.optimize!(
                    p,
                    solver_func,
                    lower = prob_data.bl,
                    upper = prob_data.bu,
                    iterations = max_iter,
                    g_tol = 1e-8,
                    x_tol = 0.0,
                )
            ).time
            x_opt = res.minimizer
            iterations = res.iterations
            converged = res.converged
            final_cost = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
        elseif solver_name == "Scipy-LeastSquares"
            pyresult = scipy.optimize.least_squares(
                prob_data.residual_func,
                copy(prob_data.x0),
                jac = prob_data.jacobian_func,
                bounds = (prob_data.bl, prob_data.bu),
                xtol = nothing,
                gtol = 1e-8,
                max_nfev = 1000,
                verbose = 0,
            )
            t = minimum(
                @be scipy.optimize.least_squares(
                    prob_data.residual_func,
                    copy(prob_data.x0),
                    jac = prob_data.jacobian_func,
                    bounds = (prob_data.bl, prob_data.bu),
                    xtol = nothing,
                    gtol = 1e-8,
                    max_nfev = 1000,
                    verbose = 0,
                )
            ).time
            x_opt = pyconvert(Vector{Float64}, pyresult["x"])
            final_cost =
                0.5 * dot(prob_data.residual_func(x_opt), prob_data.residual_func(x_opt))
            converged = pyconvert(Bool, pyresult["success"])
            g_opt = prob_data.grad_func(x_opt)
            iterations = pyconvert(Int, pyresult["njev"])
            pyresult = nothing
            GC.gc()
        elseif solver_name == "Scipy-LSMR"
            pyresult = scipy.optimize.least_squares(
                prob_data.residual_func,
                copy(prob_data.x0),
                jac = prob_data.jacobian_func,
                bounds = (prob_data.bl, prob_data.bu),
                tr_solver = "lsmr",
                xtol = nothing,
                gtol = 1e-8,
                max_nfev = 1000,
                verbose = 0,
            )
            t = minimum(
                @be scipy.optimize.least_squares(
                    prob_data.residual_func,
                    copy(prob_data.x0),
                    jac = prob_data.jacobian_func,
                    bounds = (prob_data.bl, prob_data.bu),
                    tr_solver = "lsmr",
                    xtol = nothing,
                    gtol = 1e-8,
                    max_nfev = 1000,
                    verbose = 0,
                )
            ).time
            x_opt = pyconvert(Vector{Float64}, pyresult["x"])
            final_cost =
                0.5 * dot(prob_data.residual_func(x_opt), prob_data.residual_func(x_opt))
            converged = pyconvert(Bool, pyresult["success"])
            g_opt = prob_data.grad_func(x_opt)
            iterations = pyconvert(Int, pyresult["njev"])
            pyresult = nothing
            GC.gc()
        elseif contains(solver_name, "NLLSsolver-")
            # Instantiate concrete objects first (type stability) and build the problem.
            function build_nlls_problem()
                var_obj = NLLSsolver.EuclideanVector(copy(prob_data.x0)...)
                res_obj = ProbDataResidual(prob_data)
                problem = NLLSsolver.NLLSProblem(typeof(var_obj), typeof(res_obj))
                NLLSsolver.addvariable!(problem, var_obj)
                NLLSsolver.addcost!(problem, res_obj)
                return problem
            end
            # Fairness: NLLSOptions defaults to maxtime=30s — a hidden wall-clock cap no
            # other solver has. 1e6 s is effectively unlimited (maxtime is stored as
            # UInt64 nanoseconds, so Inf would throw an InexactError).
            options = NLLSsolver.NLLSOptions(
                reldcost = 1e-8,
                iterator = solver_func,
                maxiters = max_iter,
                maxtime = 1e6,
            )

            problem = build_nlls_problem()
            result = NLLSsolver.optimize!(problem, options)

            # Rebuild a fresh problem per sample in the Chairmarks SETUP phase (untimed):
            # optimize! mutates problem.variables, so reusing one problem would time solves
            # that start from the already-converged point.
            t = minimum(@be build_nlls_problem() p -> NLLSsolver.optimize!(p, options)).time

            x_opt = collect(problem.variables[1])
            final_cost = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = result.niterations
            converged = result.termination > 0
        elseif solver_name in ["Optim-BFGS", "Optim-L-BFGS"]
            # Quasi-Newton baselines on the scalar objective 0.5‖r‖² (Optim.jl).
            # Out-of-place obj/grad closures, hence inplace = false.
            method = solver_name == "Optim-BFGS" ? Optim.BFGS() : Optim.LBFGS()
            optim_opts = Optim.Options(iterations = max_iter, g_tol = 1e-8)
            res = Optim.optimize(
                prob_data.obj_func,
                prob_data.grad_func,
                copy(prob_data.x0),
                method,
                optim_opts;
                inplace = false,
            )
            t = minimum(
                @be Optim.optimize(
                    prob_data.obj_func,
                    prob_data.grad_func,
                    copy(prob_data.x0),
                    method,
                    optim_opts;
                    inplace = false,
                )
            ).time
            x_opt = Optim.minimizer(res)
            final_cost = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = Optim.iterations(res)
            converged = Optim.converged(res)
        elseif contains(solver_name, "LsqFit")
            # LsqFit wants: model(xdata, p) ≈ ydata
            # We want: r(p) ≈ 0

            # 1. Crelseif contains(solver_name, "LsqFit")
            # LsqFit wants: model(xdata, p) ≈ ydata
            # We want: r(p) ≈ 0

            # 1. Create dummy data
            ydata = zeros(prob_data.n)
            xdata = zeros(prob_data.n) # The model will ignore this

            # 2. Wrap the residual to accept (and ignore) xdata
            model(x, p) = prob_data.residual_func(p)

            # 3. Wrap the Jacobian to accept (and ignore) xdata
            jac_model(x, p) = prob_data.jacobian_func(p)

            # Run the solver
            fit = LsqFit.curve_fit(
                model,
                jac_model,
                xdata,
                ydata,
                copy(prob_data.x0);
                lower = prob_data.bl,
                upper = prob_data.bu,
                maxIter = max_iter,
                g_tol = 1e-8,
                x_tol = 0.0,
            )

            # Run again for timing
            t = minimum(
                @be LsqFit.curve_fit(
                    model,
                    jac_model,
                    xdata,
                    ydata,
                    copy(prob_data.x0);
                    lower = prob_data.bl,
                    upper = prob_data.bu,
                    maxIter = max_iter,
                    g_tol = 1e-8,
                    x_tol = 0.0,
                )
            ).time

            x_opt = fit.param

            # FIX: LsqFit does not expose iterations!
            iterations = 0

            converged = fit.converged
            final_cost = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)

        else
            error("Unknown solver: solver_name")
        end

        bounds_satisfied = all(prob_data.bl .<= x_opt .<= prob_data.bu)

        return (
            solver = solver_name,
            success = true,
            converged = converged,
            final_cost = final_cost,
            iterations = iterations,
            time = t,
            bounds_satisfied = bounds_satisfied,
            final_gradient_norm = norm(g_opt, 2),
            x_opt = x_opt,
        )
    catch e
        println("  $solver_name failed: $e")
        return (
            solver = solver_name,
            success = false,
            converged = false,
            final_cost = Inf,
            iterations = 0,
            time = Inf,
            bounds_satisfied = false,
            final_gradient_norm = Inf,
            x_opt = fill(NaN, prob_data.m),   # variable-length, like the success branch
        )
    end
end
