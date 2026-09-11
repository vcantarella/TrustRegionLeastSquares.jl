# The hot loop must stay free of runtime dispatch and inference errors: @test_opt flags dispatch,
# @test_call flags inference. Both analyse the whole solve, for every strategy × scaling, with and
# without bounds. Keywords are written out literally — splatting them makes JET analyse a dynamic
# `kwcall` instead, which costs an order of magnitude more time.
@testitem "JET: the entry point is type stable" tags = [:unit, :slow] setup =
    [Problems, Solver] begin
    using JET

    @testset "$(Problems.label(strategy)) / $(Problems.label(scaling))" for strategy in
                                                                            Problems.STRATEGIES,
        scaling in Problems.SCALINGS

        n = 6                                           # square, so the LQ strategies apply too
        A, b = randn(n, n), randn(n)
        res!(r, x) = (r .= A * x .- b; r)
        jac!(J, x) = (J .= A; J)
        lo, hi = fill(-1.0, n), fill(2.0, n)
        @test_opt target_modules = (TrustRegionLeastSquares,) TRLS.lm_trust_region!(
            res!,
            jac!,
            zeros(n),
            n,
            strategy,
            scaling;
            max_iter = 10,
        )
        @test_call target_modules = (TrustRegionLeastSquares,) TRLS.lm_trust_region!(
            res!,
            jac!,
            zeros(n),
            n,
            strategy,
            scaling;
            max_iter = 10,
        )
        @test_opt target_modules = (TrustRegionLeastSquares,) TRLS.lm_trust_region!(
            res!,
            jac!,
            zeros(n),
            n,
            strategy,
            scaling;
            max_iter = 10,
            lb = lo,
            ub = hi,
        )
        @test_call target_modules = (TrustRegionLeastSquares,) TRLS.lm_trust_region!(
            res!,
            jac!,
            zeros(n),
            n,
            strategy,
            scaling;
            max_iter = 10,
            lb = lo,
            ub = hi,
        )
    end
end

# Whole-package analysis: every method as declared, including generic signatures no test calls
# directly (an `x0` whose `length` is not inferrable as an Int, for instance). Asserted clean.
@testitem "JET: package-level report" tags = [:unit, :slow] begin
    using JET

    report =
        report_package(TrustRegionLeastSquares; target_modules = (TrustRegionLeastSquares,))
    issues = JET.get_reports(report)
    isempty(issues) || @info "JET report_package" report
    @test isempty(issues)
end

# Aqua's standard package-quality battery: method ambiguities, unbound type parameters, undefined
# exports, stale or missing dependencies, project-file consistency and piracy.
#
# persistent_tasks is off: it instantiates and precompiles a throwaway environment, which takes
# about 15 minutes here and dominated the whole suite. The check looks for a package leaving tasks
# or timers running after `using`, and this one's only dependency is LinearAlgebra.
@testitem "Aqua quality checks" tags = [:unit, :slow] begin
    using Aqua

    Aqua.test_all(TrustRegionLeastSquares; persistent_tasks = false)
end
