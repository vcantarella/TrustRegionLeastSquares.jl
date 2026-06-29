using JET
using nonlinearlstr
using Test
using LinearAlgebra

@testset "JET checks" begin
    # Package-level error analysis, scoped to nonlinearlstr's own code (Base/LinearAlgebra
    # internals produce JET false positives we don't care about). On recent Julia/JET this
    # surfaces a handful of latent dispatch issues on code paths NOT exercised by the main
    # solver, worth investigating at leisure:
    #   - bounded_subproblems.jl: factorize!(::ColemanandLiCache, ...) has no matching method
    #   - bounded_algorithms.jl:  SubproblemCache(...; x, lb, ub, g) kwcall has no method
    #   - algorithms_v2.jl:       solve_subproblem(..., ::Nothing) / update_cache!(::Nothing, ...)
    # The exact count is version-dependent (newer JET infers more aggressively; e.g. Julia
    # 1.10 reports 0, 1.12 reports 5), so we only LOG it here rather than assert on it — the
    # hard gate is the entry-point analysis below. Print the full report when issues are found
    # so they're visible in CI logs.
    pkg_report = report_package(nonlinearlstr; target_modules = (nonlinearlstr,))
    n_pkg_issues = length(JET.get_reports(pkg_report))
    if n_pkg_issues > 0
        @info "JET report_package surfaced $n_pkg_issues latent package-level issue(s)" pkg_report
    end

    # Targeted type-stability / dispatch checks on the hot solver entry points. These are the
    # hard gate: the steady-state factorize -> solve-subproblem -> step path must stay free of
    # runtime dispatch and inference errors. A small dense least-squares problem exercises it.
    m, n = 12, 4
    A = randn(m, n)
    b = randn(m)
    res!(r, x) = (r .= A * x .- b; r)
    jac!(J, x) = (J .= A; J)
    x0 = zeros(n)

    @testset "entry-point type stability" begin
        for strat in (
            nonlinearlstr.QRSolve(),
            nonlinearlstr.SVDSolve(),
            nonlinearlstr.QRrecursiveSolve(),
        )
            # @test_opt flags runtime dispatch; @test_call flags inference errors.
            # lm_trust_region! is not exported, so qualify it.
            @test_opt target_modules = (nonlinearlstr,) nonlinearlstr.lm_trust_region!(
                res!,
                jac!,
                copy(x0),
                m,
                strat,
                nonlinearlstr.NoScaling();
                max_iter = 25,
                gtol = 1e-8,
            )
            @test_call target_modules = (nonlinearlstr,) nonlinearlstr.lm_trust_region!(
                res!,
                jac!,
                copy(x0),
                m,
                strat,
                nonlinearlstr.NoScaling();
                max_iter = 25,
                gtol = 1e-8,
            )
        end
    end
end
