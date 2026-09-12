```@meta
CurrentModule = TrustRegionLeastSquares
```

# [Bibliography](@id bibliography)

The solver is an implementation of published methods rather than a new one. This page lists what it
is built from; the source comments cite these by the keys below, with a section, algorithm or
equation number, so any line of the implementation can be traced to the statement it comes from.

## The trust-region framework

**[NW06]** J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer Series in
Operations Research and Financial Engineering, Springer, New York (2006).
[doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)

Most of this package comes from Chapters 4 and 10. Specifically:

| Used for | Where |
|---|---|
| The iteration: trial step, ratio test, radius update | Algorithm 4.1 |
| The subproblem solution is `(B + λI)p = -g` for some `λ ≥ 0` with `λ(Δ - ‖p‖) = 0` | Theorem 4.1 |
| The Cauchy point, which the bound-constrained step falls back on | Algorithm 4.2 |
| Newton's method on `φ₂(λ) = 1/Δ - 1/‖p(λ)‖`, chosen because it is nearly linear in `λ` near the root where `‖p(λ)‖ = Δ` is not | §4.3 |
| Gauss–Newton, Levenberg–Marquardt, and the augmented-matrix implementation `[J; √λ I]` | §10.3 |

**[Mor78]** J. J. Moré, "The Levenberg–Marquardt algorithm: implementation and theory", in
*Numerical Analysis*, G. A. Watson (ed.), Lecture Notes in Mathematics **630**, Springer, Berlin
(1978), pp. 105–116. [doi:10.1007/BFb0067700](https://doi.org/10.1007/BFb0067700)

The practical details that make the λ-iteration converge in a handful of steps, all as implemented
in MINPACK's `lmder` and `lmpar`: the bracket `λ* ∈ (0, ‖D⁻¹Jᵀf‖/Δ]` and its safeguard, the initial
guess, the non-decreasing diagonal scaling of [`JacobianScaling`](@ref), and the radius update that
follows the step actually taken rather than the previous radius.

## Underdetermined problems

**[CJK26]** M. Chen, S. G. Johnson and A. Karalis, "Inverse design of multiresonance filters via
quasi-normal mode theory", *Optics Express* **34**(4), 5729–5752 (2026).
[doi:10.1364/OE.579219](https://doi.org/10.1364/OE.579219) ·
[arXiv:2504.10219](https://arxiv.org/abs/2504.10219)

Appendix B, "Underdetermined Levenberg–Marquardt algorithm". When there are fewer residuals than
parameters the usual `JᵀJ` is large and rank-deficient, so the damped system is solved through the
small, full-rank `J Jᵀ` instead, giving a regularized minimum-norm step, with an LQ factorization so
that the condition number is not squared. This is what [`LQStrategy`](@ref) and
[`LQCholStrategy`](@ref) implement.

**[SGJ26]** S. G. Johnson, reply in
["Should NonlinearLeastSquaresProblem be used for deep learning?"](https://discourse.julialang.org/t/should-nonlinearleastsquaresproblem-be-used-for-deep-learning/135793/4),
Julia Discourse, 23 February 2026.

The post that prompted these strategies, and which points at [CJK26]:

> one can devise a variant of Levenberg–Marquardt that works well in such cases, based on the
> obvious idea of finding a regularized minimum-norm solution to the linearized overdetermined
> problem at each step, so that you use the small (m × m), full-rank matrix `J Jᵀ` instead (you can
> again use LQ to avoid squaring the condition number)

## Box constraints

**[MMP09]** M. Macconi, B. Morini and M. Porcelli, "A Gauss–Newton method for solving
bound-constrained underdetermined nonlinear systems", *Optimization Methods and Software* **24**(2),
219–235 (2009). [doi:10.1080/10556780902753031](https://doi.org/10.1080/10556780902753031)

The bound-constrained method implemented here: the generalized Cauchy step of eq. (8), and the
condition that the accepted step achieve a fixed fraction of its decrease (condition 11), which is
what makes projecting the Levenberg–Marquardt step onto the box globally convergent rather than
merely feasible.

**[CL96]** T. F. Coleman and Y. Li, "An interior trust region approach for nonlinear minimization
subject to bounds", *SIAM Journal on Optimization* **6**(2), 418–445 (1996).
[doi:10.1137/0806023](https://doi.org/10.1137/0806023)

The affine scaling `v(x)`: the distance from each variable to the bound its negative gradient points
at, which is what lets the trust region shrink a variable's step as it approaches an active bound,
without an active set. Only the scaling is taken from here, not the interior-point framework — this
solver's iterates are allowed to rest on a bound.

## Benchmarking

**[MGH81]** J. J. Moré, B. S. Garbow and K. E. Hillstrom, "Testing unconstrained optimization
software", *ACM Transactions on Mathematical Software* **7**(1), 17–41 (1981).
[doi:10.1145/355934.355936](https://doi.org/10.1145/355934.355936)

The problem collection and the reference objective values the unit tests assert against.

**[DM02]** E. D. Dolan and J. J. Moré, "Benchmarking optimization software with performance
profiles", *Mathematical Programming* **91**(2), 201–213 (2002).
[doi:10.1007/s101070100263](https://doi.org/10.1007/s101070100263)

The performance profiles the benchmark figures use.

**[Luk96]** L. Lukšan, "Hybrid methods for large sparse nonlinear least squares", *Journal of
Optimization Theory and Applications* **89**(3), 575–595 (1996).
[doi:10.1007/BF02275350](https://doi.org/10.1007/BF02275350)

The six hard exponential-fit problems in `benchmark/scripts/hard_luksan.jl`.
