```@meta
CurrentModule = TrustRegionLeastSquares
```

# [Bibliography](@id bibliography)

The solver is an implementation of published methods rather than a new one. This page lists what it
is built from; the source comments cite these by the keys below, with a section, algorithm or
equation number, so any line of the implementation can be traced to the statement it comes from.

## The trust-region framework

```@bibliography
Pages = []
Canonical = true

NW06
Mor78
```

Most of this package comes from Chapters 4 and 10 of [NW06](@cite). Specifically:

| Used for | Where |
|---|---|
| The iteration: trial step, ratio test, radius update | Algorithm 4.1 |
| The subproblem solution is `(B + λI)p = -g` for some `λ ≥ 0` with `λ(Δ - ‖p‖) = 0` | Theorem 4.1 |
| The Cauchy point, which the bound-constrained step falls back on | Algorithm 4.2 |
| Newton's method on `φ₂(λ) = 1/Δ - 1/‖p(λ)‖`, chosen because it is nearly linear in `λ` near the root where `‖p(λ)‖ = Δ` is not | §4.3 |
| Gauss–Newton, Levenberg–Marquardt, and the augmented-matrix implementation `[J; √λ I]` | §10.3 |

From [Mor78](@cite) come the practical details that make the λ-iteration converge in a handful of
steps, all as implemented in MINPACK's `lmder` and `lmpar`: the bracket `λ* ∈ (0, ‖D⁻¹Jᵀf‖/Δ]` and
its safeguard, the initial guess, the non-decreasing diagonal scaling of [`JacobianScaling`](@ref),
and the radius update that follows the step actually taken rather than the previous radius.

## Underdetermined problems

```@bibliography
Pages = []
Canonical = true

CJK26
Joh26
```

[CJK26; Appendix B](@cite), "Underdetermined Levenberg–Marquardt algorithm". When there are fewer
residuals than parameters the usual `JᵀJ` is large and rank-deficient, so the damped system is
solved through the small, full-rank `J Jᵀ` instead, giving a regularized minimum-norm step, with an
LQ factorization so that the condition number is not squared. This is what [`LQStrategy`](@ref) and
[`LQCholStrategy`](@ref) implement.

[Joh26](@cite) is the post that prompted these strategies, and which points at [CJK26](@cite):

> one can devise a variant of Levenberg–Marquardt that works well in such cases, based on the
> obvious idea of finding a regularized minimum-norm solution to the linearized overdetermined
> problem at each step, so that you use the small (m × m), full-rank matrix `J Jᵀ` instead (you can
> again use LQ to avoid squaring the condition number)

## Box constraints

```@bibliography
Pages = []
Canonical = true

MMP09
CL96
```

[MMP09](@cite) is the bound-constrained method implemented here: the generalized Cauchy step of
eq. (8), and the condition that the accepted step achieve a fixed fraction of its decrease
(condition 11), which is what makes projecting the Levenberg–Marquardt step onto the box globally
convergent rather than merely feasible.

From [CL96](@cite) comes the affine scaling `v(x)`: the distance from each variable to the bound its
negative gradient points at, which is what lets the trust region shrink a variable's step as it
approaches an active bound, without an active set. Only the scaling is taken from here, not the
interior-point framework — this solver's iterates are allowed to rest on a bound.

## Benchmarking

```@bibliography
Pages = []
Canonical = true

MGH81
DM02
Luk96
```

[MGH81](@cite) is the problem collection and the reference objective values the unit tests assert
against. [DM02](@cite) defines the performance profiles the benchmark figures use. [Luk96](@cite)
contributes the six hard exponential-fit problems in `benchmark/scripts/hard_luksan.jl`.
