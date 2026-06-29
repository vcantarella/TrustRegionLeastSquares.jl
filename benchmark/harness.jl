# Cross-package benchmark harness for nonlinearlstr.
#
# This is the single entry point the benchmark scripts include. It loads the heavy
# dependencies and the harness pieces in the right order.
#
# ONE-TIME SETUP (nonlinearlstr is a local dev package, not registered, so it must be
# dev'd into this env by path before instantiate can resolve it):
#
#   julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#
# Then run scripts against the benchmark environment, e.g.:
#
#   julia --project=benchmark benchmark/scripts/compare_unconstrained.jl
#
# Layout:
#   deps.jl              - all `using` statements + the scipy handle
#   problems/discovery.jl - find_{nlls,bounded,cutest_nlls}_problems
#   problems/wrappers.jl  - create_{nls,cutest}_functions (solver-agnostic problem wrappers)
#   dispatch.jl           - ProbDataResidual + test_solver_on_problem (per-solver adapters)
#   run.jl                - nlls_benchmark (the orchestration loop)
const _HERE = @__DIR__
include(joinpath(_HERE, "deps.jl"))
include(joinpath(_HERE, "problems", "discovery.jl"))
include(joinpath(_HERE, "problems", "wrappers.jl"))
include(joinpath(_HERE, "dispatch.jl"))
include(joinpath(_HERE, "run.jl"))
