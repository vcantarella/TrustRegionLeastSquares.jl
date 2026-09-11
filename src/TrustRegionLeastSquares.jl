"""
    TrustRegionLeastSquares

A Levenberg–Marquardt trust-region solver for `min ½‖f(x)‖²`, with optional box constraints,
minimum-norm steps for underdetermined and rank-deficient problems, and diagonal variable scaling.
The entry point is [`lm_trust_region!`](@ref).

# References

The trust-region framework, the characterization of the subproblem solution, and the Newton
iteration on the damping parameter all follow Nocedal & Wright; the bound-constrained step follows
Macconi, Morini & Porcelli; the underdetermined formulation follows Chen, Johnson & Karalis. Source
comments cite these by the keys below, with a section, algorithm or equation number.

- **[NW06]** J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer (2006).
  <https://doi.org/10.1007/978-0-387-40065-5>
  Algorithm 4.1 (the trust-region loop and its ratio test), Theorem 4.1 (a step solves the
  subproblem iff `(B + λI)p = -g` with `λ ≥ 0` and `λ(Δ - ‖p‖) = 0`), Algorithm 4.2 (the Cauchy
  point), §4.3 (the λ-iteration on `φ₂(λ) = 1/Δ - 1/‖p(λ)‖`, which is near-linear in λ and so
  suits Newton's method), and §10.3 (Gauss–Newton and Levenberg–Marquardt, including the
  augmented-matrix implementation).
- **[Mor78]** J. J. Moré, "The Levenberg–Marquardt algorithm: implementation and theory", in
  *Numerical Analysis*, G. A. Watson (ed.), Lecture Notes in Mathematics 630, Springer (1978),
  pp. 105–116. <https://doi.org/10.1007/BFb0067700>
  The safeguarded bracket for λ, the non-decreasing diagonal scaling, and the radius update, as
  implemented in MINPACK's `lmder`/`lmpar`.
- **[MMP09]** M. Macconi, B. Morini and M. Porcelli, "A Gauss–Newton method for solving
  bound-constrained underdetermined nonlinear systems", *Optimization Methods and Software*
  **24**(2), 219–235 (2009). <https://doi.org/10.1080/10556780902753031>
  The generalized Cauchy step (eq. 8) and the fraction-of-Cauchy-decrease condition (condition 11)
  that make the projected step globally convergent.
- **[CL96]** T. F. Coleman and Y. Li, "An interior trust region approach for nonlinear minimization
  subject to bounds", *SIAM Journal on Optimization* **6**(2), 418–445 (1996).
  <https://doi.org/10.1137/0806023>
  The affine scaling `|v(x)|` that measures the distance to the bound each gradient component
  points at.
- **[CJK26]** M. Chen, S. G. Johnson and A. Karalis, "Inverse design of multiresonance filters via
  quasi-normal mode theory", *Optics Express* **34**(4), 5729–5752 (2026).
  <https://doi.org/10.1364/OE.579219>, preprint <https://arxiv.org/abs/2504.10219>
  Appendix B, "Underdetermined Levenberg–Marquardt algorithm": for fewer residuals than parameters,
  solve the damped system through the small `J Jᵀ` matrix rather than `JᵀJ`, giving a regularized
  minimum-norm step, and use an LQ factorization so the condition number is not squared. This is
  what [`LQStrategy`](@ref) and [`LQCholStrategy`](@ref) implement.
- **[SGJ26]** S. G. Johnson, reply in "Should NonlinearLeastSquaresProblem be used for deep
  learning?", Julia Discourse, 23 February 2026.
  <https://discourse.julialang.org/t/should-nonlinearleastsquaresproblem-be-used-for-deep-learning/135793/4>
  The post that prompted the LQ strategies here: *"one can devise a variant of Levenberg–Marquardt
  that works well in such cases, based on the obvious idea of finding a regularized minimum-norm
  solution to the linearized overdetermined problem at each step, so that you use the small
  (m × m), full-rank matrix `J Jᵀ` instead (you can again use LQ to avoid squaring the condition
  number)"*.
"""
module TrustRegionLeastSquares
using LinearAlgebra
include("scaling.jl")
include("caches.jl")
include("subproblems.jl")
include("bounded.jl")
include("algorithms.jl")
export lm_trust_region!
end
