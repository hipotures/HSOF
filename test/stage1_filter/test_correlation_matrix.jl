using Test
using CUDA
using Statistics
using LinearAlgebra
using Random

# Include the correlation computation module
include("../../src/stage1_filter/correlation_matrix.jl")

using .CorrelationComputation
using .CorrelationComputation.GPUMemoryLayout

@testset "Correlation Matrix Computation Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU correlation tests"
        return
    end
    
    @testset "CorrelationConfig Creation" begin
        config = create_correlation_config(100, 1000)
        
        @test config.n_features == 100
        @test config.n_samples == 1000
        @test config.use_cublas == true
        @test config.batch_size == 100000
        @test config.epsilon == Float32(1e-8)
        
        # Test with custom parameters
        config2 = create_correlation_config(50, 500,
                                          use_cublas=false,
                                          batch_size=50000,
                                          epsilon=Float32(1e-6))
        
        @test config2.n_features == 50
        @test config2.n_samples == 500
        @test config2.use_cublas == false
        @test config2.batch_size == 50000
        @test config2.epsilon == Float32(1e-6)
    end
    
    @testset "Feature Standardization" begin
        n_samples = 1000
        n_features = 10
        
        # Create test data with known mean and std
        feature_matrix = create_feature_matrix(n_samples, n_features)
        
        # Fill with normal distribution
        Random.seed!(42)
        test_data = randn(Float32, n_samples, n_features)
        
        # Make first feature have mean=5, std=2
        test_data[:, 1] = 2.0f0 * test_data[:, 1] .+ 5.0f0
        
        # Make second feature constant
        test_data[:, 2] .= 3.14f0
        
        # Copy test data to GPU feature matrix
        copyto!(view(feature_matrix.data, 1:n_samples, 1:n_features), test_data)
        
        # Create correlation matrix and config
        corr_matrix = create_correlation_matrix(n_features, n_samples)
        config = create_correlation_config(n_features, n_samples)
        
        # Compute just standardization
        threads = 256
        blocks = cld(n_features, threads)
        
        @cuda threads=threads blocks=blocks CorrelationComputation.compute_means_kernel!(
            corr_matrix.feature_means,
            feature_matrix.data,
            Int32(n_samples),
            Int32(n_features)
        )
        
        @cuda threads=threads blocks=blocks CorrelationComputation.compute_stds_kernel!(
            corr_matrix.feature_stds,
            feature_matrix.data,
            corr_matrix.feature_means,
            Int32(n_samples),
            Int32(n_features),
            config.epsilon
        )
        
        CUDA.synchronize()
        
        # Check means
        means_cpu = Array(corr_matrix.feature_means)
        @test isapprox(means_cpu[1], 5.0f0, atol=0.1)
        @test isapprox(means_cpu[2], 3.14f0, atol=0.01)
        
        # Check stds
        stds_cpu = Array(corr_matrix.feature_stds)
        @test isapprox(stds_cpu[1], 2.0f0, atol=0.1)
        @test stds_cpu[2] < 0.01  # Should be very small (epsilon effect)
        
        # Test standardization
        threads_x = 32
        threads_y = 8
        blocks_x = cld(n_samples, threads_x)
        blocks_y = cld(n_features, threads_y)
        
        @cuda threads=(threads_x, threads_y) blocks=(blocks_x, blocks_y) CorrelationComputation.standardize_features_kernel!(
            corr_matrix.standardized_data,
            feature_matrix.data,
            corr_matrix.feature_means,
            corr_matrix.feature_stds,
            Int32(n_samples),
            Int32(n_features)
        )
        
        CUDA.synchronize()
        
        # Check standardized data
        std_data_cpu = Array(corr_matrix.standardized_data[1:n_samples, 1:n_features])
        
        # First feature should have mean≈0, std≈1
        @test isapprox(mean(std_data_cpu[:, 1]), 0.0, atol=0.01)
        @test isapprox(std(std_data_cpu[:, 1]), 1.0, atol=0.01)
        
        # Second feature (constant) should be all zeros
        @test all(std_data_cpu[:, 2] .== 0.0f0)
    end
    
    @testset "Correlation Matrix Computation - Simple Case" begin
        n_samples = 1000
        n_features = 5
        
        # Create perfectly correlated features
        feature_matrix = create_feature_matrix(n_samples, n_features)
        corr_matrix = create_correlation_matrix(n_features, n_samples)
        
        # Generate test data
        Random.seed!(42)
        base = randn(Float32, n_samples)
        
        test_data = CUDA.zeros(Float32, n_samples, n_features)
        CUDA.@allowscalar begin
            # Feature 1: base data
            test_data[:, 1] = base
            # Feature 2: perfectly correlated with 1
            test_data[:, 2] = 2 * base .+ 3
            # Feature 3: negatively correlated with 1
            test_data[:, 3] = -base .+ 1
            # Feature 4: uncorrelated (random)
            test_data[:, 4] = randn(Float32, n_samples)
            # Feature 5: partially correlated
            test_data[:, 5] = 0.5 * base + 0.5 * randn(Float32, n_samples)
        end
        
        # Copy test data to GPU feature matrix
        copyto!(view(feature_matrix.data, 1:n_samples, 1:n_features), test_data)
        
        # Test cuBLAS version
        config = create_correlation_config(n_features, n_samples, use_cublas=true)
        compute_correlation_cublas!(corr_matrix, feature_matrix, config)
        
        # Check correlations
        @test isapprox(get_correlation(corr_matrix, 1, 2), 1.0, atol=0.01)  # Perfect positive
        @test isapprox(get_correlation(corr_matrix, 1, 3), -1.0, atol=0.01) # Perfect negative
        @test abs(get_correlation(corr_matrix, 1, 4)) < 0.1  # Near zero
        @test 0.4 < get_correlation(corr_matrix, 1, 5) < 0.6  # Partial correlation
        
        # Test symmetry
        @test get_correlation(corr_matrix, 2, 1) == get_correlation(corr_matrix, 1, 2)
        @test get_correlation(corr_matrix, 3, 1) == get_correlation(corr_matrix, 1, 3)
    end
    
    @testset "Correlation Matrix Computation - Direct vs cuBLAS" begin
        n_samples = 500
        n_features = 20
        
        # Create random data
        feature_matrix = create_feature_matrix(n_samples, n_features)
        Random.seed!(42)
        test_data = randn(Float32, n_samples, n_features)
        # Copy test data to GPU feature matrix
        copyto!(view(feature_matrix.data, 1:n_samples, 1:n_features), test_data)
        
        # Compute with cuBLAS
        corr_matrix_cublas = create_correlation_matrix(n_features, n_samples)
        config_cublas = create_correlation_config(n_features, n_samples, use_cublas=true)
        compute_correlation_cublas!(corr_matrix_cublas, feature_matrix, config_cublas)
        
        # Compute with direct method
        corr_matrix_direct = create_correlation_matrix(n_features, n_samples)
        config_direct = create_correlation_config(n_features, n_samples, use_cublas=false)
        compute_correlation_direct!(corr_matrix_direct, feature_matrix, config_direct)
        
        # Compare results
        corr_cublas_cpu = Array(corr_matrix_cublas.data)
        corr_direct_cpu = Array(corr_matrix_direct.data)
        
        @test all(isapprox.(corr_cublas_cpu, corr_direct_cpu, atol=1e-5))
    end
    
    @testset "Find Correlated Pairs" begin
        n_samples = 1000
        n_features = 10
        
        feature_matrix = create_feature_matrix(n_samples, n_features)
        corr_matrix = create_correlation_matrix(n_features, n_samples)
        
        # Create some highly correlated features
        Random.seed!(42)
        base1 = randn(Float32, n_samples)
        base2 = randn(Float32, n_samples)
        
        test_data = randn(Float32, n_samples, n_features)
        CUDA.@allowscalar begin
            # Make features 1 and 2 highly correlated
            test_data[:, 1] = base1
            test_data[:, 2] = 0.98 * base1 + 0.02 * randn(Float32, n_samples)
            
            # Make features 5 and 7 highly correlated
            test_data[:, 5] = base2
            test_data[:, 7] = 0.96 * base2 + 0.04 * randn(Float32, n_samples)
        end
        
        # Copy test data to GPU feature matrix
        copyto!(view(feature_matrix.data, 1:n_samples, 1:n_features), test_data)
        
        config = create_correlation_config(n_features, n_samples)
        compute_correlation_cublas!(corr_matrix, feature_matrix, config)
        
        # Find pairs with correlation > 0.95
        correlated_pairs = find_correlated_pairs(corr_matrix, Float32(0.95))
        
        @test length(correlated_pairs) >= 2
        
        # Check that our known pairs are found
        pair_indices = [(p[1], p[2]) for p in correlated_pairs]
        @test (1, 2) in pair_indices || (2, 1) in pair_indices
        @test (5, 7) in pair_indices || (7, 5) in pair_indices
    end
    
    @testset "Edge Cases" begin
        # Small dataset
        n_samples = 10
        n_features = 3
        
        feature_matrix = create_feature_matrix(n_samples, n_features)
        corr_matrix = create_correlation_matrix(n_features, n_samples)
        
        # All constant features
        test_data = CUDA.ones(Float32, n_samples, n_features)
        test_data[:, 1] .= 1.0f0
        test_data[:, 2] .= 2.0f0
        test_data[:, 3] .= 3.0f0
        
        # Copy test data to GPU feature matrix
        copyto!(view(feature_matrix.data, 1:n_samples, 1:n_features), test_data)
        
        config = create_correlation_config(n_features, n_samples)
        compute_correlation_cublas!(corr_matrix, feature_matrix, config)
        
        # Constant features should have NaN or 0 correlations
        corr_cpu = Array(corr_matrix.data)
        @test all(corr_cpu .== 0.0f0) || all(isnan.(corr_cpu))
    end
    
    @testset "Performance Benchmark" begin
        # Test with realistic size
        n_samples = 10000
        n_features = 1000
        
        feature_matrix = create_feature_matrix(n_samples, n_features)
        corr_matrix = create_correlation_matrix(n_features, n_samples)
        
        # Fill with random data
        Random.seed!(42)
        test_data = randn(Float32, n_samples, n_features)
        # Copy test data to GPU feature matrix
        copyto!(view(feature_matrix.data, 1:n_samples, 1:n_features), test_data)
        
        config = create_correlation_config(n_features, n_samples)
        
        # Warmup
        compute_correlation_cublas!(corr_matrix, feature_matrix, config)
        CUDA.synchronize()
        
        # Time the computation
        elapsed = CUDA.@elapsed begin
            compute_correlation_cublas!(corr_matrix, feature_matrix, config)
        end
        
        println("\nCorrelation matrix computation for $n_samples samples × $n_features features: $(round(elapsed * 1000, digits=2)) ms")
        
        # Should be fast
        @test elapsed < 5.0  # Less than 5 seconds for 1000 features
        
        # Verify some results
        n_pairs = div(n_features * (n_features - 1), 2)
        @test length(corr_matrix.data) == n_pairs
        
        # Check that correlations are in valid range
        corr_cpu = Array(corr_matrix.data)
        @test all(-1.0 .<= corr_cpu .<= 1.0)
    end
end

println("\n✅ Correlation matrix computation tests completed!")