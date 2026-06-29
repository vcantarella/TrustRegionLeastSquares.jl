using Test
using LinearAlgebra
using Statistics
using Random
using ForwardDiff
using nonlinearlstr

@testset "Unconstrained Subproblem Tests" begin

    # Setup test data - used across all test sets
    Random.seed!(123)  # For reproducible tests
    m, n = 10, 4
    J = randn(m, n)
    f_true = randn(n)
    p_true = randn(n)
    f = -J * p_true + 0.1 * randn(m)  # Add some noise
    λ = 0.8

    # Helper function for reference solution
    function get_p(J, f, λ)
        n = size(J, 2)
        J_aug = [J; sqrt(λ) * I(n)]
        b = [-f; zeros(n)]
        return J_aug \ b
    end

    p_ref = get_p(J, f, λ)
    size_p = norm(p_ref)

    @testset "Cache Construction" begin
        @testset "QR Cache" begin
            strat = nonlinearlstr.QRSolve()
            scaling = nonlinearlstr.NoScaling()
            cache = nonlinearlstr.SubproblemCache(strat, scaling, J)

            @test cache.scaling_matrix == I(n)
            @test typeof(cache.factorization) <:
                  Union{LinearAlgebra.QR,LinearAlgebra.QRCompactWY,LinearAlgebra.QRPivoted}
        end

        @testset "SVD Cache" begin
            strat = nonlinearlstr.SVDSolve()
            scaling = nonlinearlstr.NoScaling()
            cache = nonlinearlstr.SubproblemCache(strat, scaling, J)

            @test cache.scaling_matrix == I(n)
            @test typeof(cache.factorization) <: LinearAlgebra.SVD
        end

        @testset "QRrecursive Cache" begin
            strat = nonlinearlstr.QRrecursiveSolve()
            scaling = nonlinearlstr.NoScaling()
            cache = nonlinearlstr.SubproblemCache(strat, scaling, J)

            @test cache.scaling_matrix == I(n)
            @test typeof(cache.factorization) <:
                  Union{LinearAlgebra.QR,LinearAlgebra.QRCompactWY,LinearAlgebra.QRPivoted}
        end

        @testset "EVD Cache" begin
            strat = nonlinearlstr.EVDSolve()
            scaling = nonlinearlstr.NoScaling()
            cache = nonlinearlstr.SubproblemCache(strat, scaling, J)

            @test cache.scaling_matrix == I(n)
            @test typeof(cache.factorization) <: LinearAlgebra.Eigen
        end
    end

    @testset "Low-level Solving Functions" begin
        @testset "SVD Augmented System Solver" begin
            strat = nonlinearlstr.SVDSolve()
            svdls = svd(J)
            p_svd = nonlinearlstr.solve_augmented(strat, svdls, J, Diagonal(ones(n)), -f, λ)

            @test p_svd ≈ p_ref rtol=1e-10
        end

        @testset "Derivative dp/dλ - QR Method" begin
            strat = nonlinearlstr.QRSolve()
            J_aug = [J; sqrt(λ) * I(n)]
            qrf = qr(J_aug)
            Dk = I(n)
            dp_dλ = zeros(size(p_ref))


            # Analytical derivative
            nonlinearlstr.solve_for_dp_dlambda_scaled!(strat, dp_dλ, qrf, p_ref, Dk)

            # Finite difference reference
            dp_dλ_fd = ForwardDiff.derivative(λ -> get_p(J, f, λ), λ)

            @test dp_dλ ≈ dp_dλ_fd atol=1e-6
        end

        @testset "Derivative dp/dλ - SVD Method" begin
            strat = nonlinearlstr.SVDSolve()
            F = svd(J)
            Dk = I(n)

            # Analytical derivative
            dp_dλ_svd = nonlinearlstr.solve_for_dp_dlambda_scaled(strat, F, Dk, λ, -f)

            # Finite difference reference
            dp_dλ_fd = ForwardDiff.derivative(λ -> get_p(J, f, λ), λ)

            @test dp_dλ_svd ≈ dp_dλ_fd atol=1e-6
        end

        @testset "Derivative dp/dλ - QRrecursive Method" begin
            strat = nonlinearlstr.QRrecursiveSolve()
            F = qr(J, ColumnNorm())
            Dk = I(n)
            # 1. Extract Permutation and Caches
            perm = hasproperty(F, :p) ? F.p : (1:n)

            p = zeros(n)
            R_buffer = copy(F.R)
            # Precompute Q' * f. 
            # The RHS for the system R*y = ... is -Q'*f
            QTr_orig = F.Q' * f
            rhs_orig = -QTr_orig
            rhs_buffer = copy(rhs_orig)
            perm_buffer = copy(p)
            dpdλ = copy(p)

            # We need a workspace vector for the update. 
            # Ideally passed in cache, here we allocate if not available or assume safety.
            # For this snippet, I will allocate one locally to be safe.
            v_row = zeros(n)
            nonlinearlstr.solve_damped_system_recursive_inplace!(
                p,
                R_buffer,
                rhs_buffer,
                λ,
                n,
                Dk,
                perm,
                v_row,
            )

            # Analytical derivative
            nonlinearlstr.solve_for_dp_dlambda_scaled!(
                dpdλ,
                R_buffer,
                p,
                Dk,
                perm,
                perm_buffer,
            )
            # Finite difference reference
            dp_dλ_fd = ForwardDiff.derivative(λ -> get_p(J, f, λ), λ)

            @test dpdλ ≈ dp_dλ_fd atol=1e-6
        end

        @testset "Derivative dp/dλ - EVD Method" begin
            strat = nonlinearlstr.EVDSolve()
            F = eigen(J'J)
            Dk = I(n)

            # Analytical derivative
            p, dp_dλ_evd = nonlinearlstr.solve_augmented_with_derivative(strat, F, J, -f, λ)


            # Automatic difference reference
            dp_dλ_fd = ForwardDiff.derivative(λ -> get_p(J, f, λ), λ)

            @test dp_dλ_evd ≈ dp_dλ_fd atol=1e-6
        end


    end

    @testset "Trust Region Subproblem Solver" begin
        @testset "QR-based Solver" begin
            strat = nonlinearlstr.QRSolve()
            scaling = nonlinearlstr.NoScaling()
            cache = nonlinearlstr.SubproblemCache(strat, scaling, J)

            λ_qr, δ_qr = nonlinearlstr.solve_subproblem(strat, J, f, size_p, cache)

            @test abs(λ_qr - λ) < 1e-3
            @test norm(δ_qr) ≈ size_p atol=1e-3
            @test δ_qr ≈ p_ref atol=1e-3
        end

        @testset "SVD-based Solver" begin
            strat = nonlinearlstr.SVDSolve()
            scaling = nonlinearlstr.NoScaling()
            cache = nonlinearlstr.SubproblemCache(strat, scaling, J)

            λ_svd, δ_svd = nonlinearlstr.solve_subproblem(strat, J, f, size_p, cache)

            @test abs(λ_svd - λ) < 1e-3
            @test norm(δ_svd) ≈ size_p atol=1e-3
            @test δ_svd ≈ p_ref atol=1e-3
        end

        @testset "QR-recursivebased Solver" begin
            strat = nonlinearlstr.QRrecursiveSolve()
            scaling = nonlinearlstr.NoScaling()
            cache = nonlinearlstr.SubproblemCache(strat, scaling, J)

            λ_qr, δ_qr = nonlinearlstr.solve_subproblem(strat, J, f, size_p, cache)

            @test abs(λ_qr - λ) < 1e-3
            @test norm(δ_qr) ≈ size_p atol=1e-3
            @test δ_qr ≈ p_ref atol=1e-3
        end

        @testset "EVD Solver" begin
            strat = nonlinearlstr.EVDSolve()
            scaling = nonlinearlstr.NoScaling()
            cache = nonlinearlstr.SubproblemCache(strat, scaling, J)

            λ_evd, δ_evd = nonlinearlstr.solve_subproblem(strat, J, f, size_p, cache)

            @test abs(λ_evd - λ) < 1e-3
            @test norm(δ_evd) ≈ size_p atol=1e-3
            @test δ_evd ≈ p_ref atol=1e-3
        end

        @testset "Method Consistency" begin
            # Both methods should give similar results
            strat_qr = nonlinearlstr.QRSolve()
            strat_svd = nonlinearlstr.SVDSolve()
            scaling = nonlinearlstr.NoScaling()

            cache_qr = nonlinearlstr.SubproblemCache(strat_qr, scaling, J)
            cache_svd = nonlinearlstr.SubproblemCache(strat_svd, scaling, J)

            λ_qr, δ_qr = nonlinearlstr.solve_subproblem(strat_qr, J, f, size_p, cache_qr)
            λ_svd, δ_svd =
                nonlinearlstr.solve_subproblem(strat_svd, J, f, size_p, cache_svd)

            @test λ_qr ≈ λ_svd atol=1e-2
            @test δ_qr ≈ δ_svd atol=1e-2
        end
    end

    # @testset "Performance Benchmarks" begin
    #     # Note: These are for development/profiling, not automated testing
    #     println("\n=== Performance Benchmarks ===")

    #     println("Benchmarking QR-based solve:")
    #     strat_qr = nonlinearlstr.QRSolve()
    #     scaling = nonlinearlstr.NoScaling()
    #     cache_qr = nonlinearlstr.SubproblemCache(strat_qr, scaling, J)
    #     @benchmark nonlinearlstr.solve_subproblem($strat_qr, $J, $f, $size_p, $cache_qr)

    #     println("Benchmarking SVD-based solve:")
    #     strat_svd = nonlinearlstr.SVDSolve()
    #     cache_svd = nonlinearlstr.SubproblemCache(strat_svd, scaling, J)
    #     @benchmark nonlinearlstr.solve_subproblem($strat_svd, $J, $f, $size_p, $cache_svd)
    # end

    # @testset "Type Stability" begin
    #     # Check for type instabilities
    #     println("\n=== Type Stability Analysis ===")

    #     println("QR method type analysis:")
    #     strat_qr = nonlinearlstr.QRSolve()
    #     scaling = nonlinearlstr.NoScaling()
    #     cache_qr = nonlinearlstr.SubproblemCache(strat_qr, scaling, J)
    #     @code_warntype nonlinearlstr.solve_subproblem(strat_qr, J, f, size_p, cache_qr)

    #     println("SVD method type analysis:")
    #     strat_svd = nonlinearlstr.SVDSolve()
    #     cache_svd = nonlinearlstr.SubproblemCache(strat_svd, scaling, J)
    #     @code_warntype nonlinearlstr.solve_subproblem(strat_svd, J, f, size_p, cache_svd)
    # end
end
