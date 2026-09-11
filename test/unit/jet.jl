using JET, Test, LinearAlgebra
isdefined(Main, :MGH) || include(joinpath(@__DIR__, "..", "problems.jl"))

# The hot loop must stay free of runtime dispatch and inference errors: @test_opt flags dispatch,
# @test_call flags inference. Both analyse the whole solve, for every strategy × scaling, with and
# without bounds. Keywords are written out literally — splatting them makes JET analyse a dynamic
# `kwcall` instead, which costs an order of magnitude more time.
@testset "JET: entry point is type stable — $(label(strategy)) / $(label(scaling))" for strategy in
                                                                                        STRATEGIES,
    scaling in SCALINGS

    n = 6                                           # square, so the LQ strategies apply too
    A, b = randn(n, n), randn(n)
    res!(r, x) = (r .= A * x .- b; r)
    jac!(J, x) = (J .= A; J)
    lo, hi = fill(-1.0, n), fill(2.0, n)
    @test_opt target_modules = (nonlinearlstr,) NL.lm_trust_region!(
        res!,
        jac!,
        zeros(n),
        n,
        strategy,
        scaling;
        max_iter = 10,
    )
    @test_call target_modules = (nonlinearlstr,) NL.lm_trust_region!(
        res!,
        jac!,
        zeros(n),
        n,
        strategy,
        scaling;
        max_iter = 10,
    )
    @test_opt target_modules = (nonlinearlstr,) NL.lm_trust_region!(
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
    @test_call target_modules = (nonlinearlstr,) NL.lm_trust_region!(
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

# Whole-package analysis: every method as declared, including generic signatures no test calls
# directly (an `x0` whose `length` is not inferrable as an Int, for instance). Asserted clean.
@testset "JET: package-level report" begin
    report = report_package(nonlinearlstr; target_modules = (nonlinearlstr,))
    issues = JET.get_reports(report)
    isempty(issues) || @info "JET report_package" report
    @test isempty(issues)
end
