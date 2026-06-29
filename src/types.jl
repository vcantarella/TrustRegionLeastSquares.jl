"""
    SubProblemStrategy

Abstract type defining strategies for solving subproblems in trust region methods.
Implementations should provide methods for factorization and solving linear systems.
"""
abstract type SubProblemStrategy end

"""
    QRSolve <: SubProblemStrategy

Strategy that uses QR factorization with column pivoting for solving subproblems.
"""
struct QRSolve <: SubProblemStrategy end

"""
    QRSolve <: SubProblemStrategy

Strategy that reuses QR factorization with approximation.
"""
struct QRrecursiveSolve <: SubProblemStrategy end

"""
    SVDSolve <: SubProblemStrategy

Strategy that uses Singular Value Decomposition (SVD) for solving subproblems.
"""
struct SVDSolve <: SubProblemStrategy end

struct EVDSolve <: SubProblemStrategy end

abstract type AbstractSubproblemCache end
