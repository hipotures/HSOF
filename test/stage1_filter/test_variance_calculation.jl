using Test
using CUDA
using Statistics
using Random

# Include the variance calculation module
include("../../src/stage1_filter/variance_calculation.jl")

using .VarianceCalculation
using .VarianceCalculation.GPUMemoryLayout

@testset "Variance Calculation Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU variance tests"
        return
    end
    
    @testset "VarianceConfig Creation" begin
        config = create_variance_config(100, 1000)
        
        @test config.n_features == 100
        @test config.n_samples == 1000
        @test config.use_welford == true
        @test config.block_size == 256
        @test config.epsilon == Float32(1e-10)
        
        # Test with custom parameters
        config2 = create_variance_config(50, 500,
                                       use_welford=false,
                                       block_size=128,
                                       epsilon=Float32(1e-8))
        
        @test config2.n_features == 50
        @test config2.n_samples == 500
        @test config2.use_welford == false
        @test config2.block_size == 128
        @test config2.epsilon == Float32(1e-8)
    end
    
    @testset "Simple Variance Calculation" begin
        n_samples = 1000
        n_features = 10
        
        # Create test data with known variance
        feature_matrix = create_feature_matrix(n_samples, n_features)
        variance_buffers = create_variance_buffers(n_features)
        
        # Generate test data
        Random.seed!(42)
        test_data = zeros(Float32, n_samples, n_features)
        
        # Feature 1: constant (variance = 0)
        test_data[:, 1] .= 5.0f0
        
        # Feature 2: uniform distribution [0, 1] (variance ≈ 1/12)
        test_data[:, 2] = rand(Float32, n_samples)
        
        # Feature 3: normal distribution with std=2
        test_data[:, 3] = 2.0f0 * randn(Float32, n_samples) .+ 10.0f0
        
        # Other features: standard normal
        for j in 4:n_features
            test_data[:, j] = randn(Float32, n_samples)
        end
        
        # Copy to GPU
        copyto!(view(feature_matrix.data, 1:n_samples, 1:n_features), test_data)
        
        # Compute variances using Welford's algorithm
        config = create_variance_config(n_features, n_samples)
        compute_variances!(variance_buffers, feature_matrix, config)
        
        # Check results
        variances_cpu = Array(variance_buffers.variances)
        means_cpu = Array(variance_buffers.means)
        
        # Feature 1: constant
        @test variances_cpu[1] < 1e-6
        @test isapprox(means_cpu[1], 5.0f0, atol=0.01)
        
        # Feature 2: uniform [0,1]
        @test isapprox(variances_cpu[2], 1.0f0/12.0f0, atol=0.01)
        @test isapprox(means_cpu[2], 0.5f0, atol=0.05)
        
        # Feature 3: normal with std=2
        @test isapprox(variances_cpu[3], 4.0f0, atol=0.5)
        @test isapprox(means_cpu[3], 10.0f0, atol=0.2)
        
        # Other features: standard normal
        for j in 4:n_features
            @test isapprox(variances_cpu[j], 1.0f0, atol=0.2)
            @test isapprox(means_cpu[j], 0.0f0, atol=0.1)
        end
    end
    
    @testset "Two-pass vs Welford Algorithm" begin
        n_samples = 5000
        n_features = 20
        
        feature_matrix = create_feature_matrix(n_samples, n_features)
        
        # Generate random data
        Random.seed!(42)
        test_data = randn(Float32, n_samples, n_features)
        copyto!(view(feature_matrix.data, 1:n_samples, 1:n_features), test_data)
        
        # Compute with Welford's algorithm
        variance_buffers_welford = create_variance_buffers(n_features)
        config_welford = create_variance_config(n_features, n_samples, use_welford=true)
        compute_variances!(variance_buffers_welford, feature_matrix, config_welford)
        
        # Compute with two-pass algorithm
        variance_buffers_twopass = create_variance_buffers(n_features)
        config_twopass = create_variance_config(n_features, n_samples, use_welford=false)
        compute_variances!(variance_buffers_twopass, feature_matrix, config_twopass)
        
        # Compare results
        var_welford = Array(variance_buffers_welford.variances)
        var_twopass = Array(variance_buffers_twopass.variances)
        
        @test all(isapprox.(var_welford, var_twopass, rtol=1e-5))
        
        mean_welford = Array(variance_buffers_welford.means)
        mean_twopass = Array(variance_buffers_twopass.means)
        
        @test all(isapprox.(mean_welford, mean_twopass, rtol=1e-5))
    end
    
    @testset "Warp Shuffle Optimization" begin
        n_samples = 10000
        n_features = 32
        
        feature_matrix = create_feature_matrix(n_samples, n_features)
        
        # Generate test data
        Random.seed!(42)
        test_data = randn(Float32, n_samples, n_features) .* 3.0f0 .+ 2.0f0
        copyto!(view(feature_matrix.data, 1:n_samples, 1:n_features), test_data)
        
        # Compute with standard method
        variance_buffers_std = create_variance_buffers(n_features)
        config = create_variance_config(n_features, n_samples)
        compute_variances!(variance_buffers_std, feature_matrix, config)
        
        # Compute with warp shuffle
        variance_buffers_warp = create_variance_buffers(n_features)
        compute_variances_warp_shuffle!(variance_buffers_warp, feature_matrix, config)
        
        # Compare results
        var_std = Array(variance_buffers_std.variances)
        var_warp = Array(variance_buffers_warp.variances)
        
        @test all(isapprox.(var_std, var_warp, rtol=1e-4))
    end
    
    @testset "Low Variance Feature Detection" begin
        n_samples = 1000
        n_features = 20
        
        feature_matrix = create_feature_matrix(n_samples, n_features)
        variance_buffers = create_variance_buffers(n_features)
        
        # Create features with varying variances
        Random.seed!(42)
        test_data = randn(Float32, n_samples, n_features)
        
        # Make some features near-constant
        test_data[:, 1] .= 1.0f0  # Exactly constant
        test_data[:, 5] = 1.0f0 .+ Float32(1e-8) * randn(Float32, n_samples)  # Very low variance
        test_data[:, 10] = 1.0f0 .+ Float32(1e-4) * randn(Float32, n_samples)  # Low variance
        
        copyto!(view(feature_matrix.data, 1:n_samples, 1:n_features), test_data)
        
        # Compute variances
        config = create_variance_config(n_features, n_samples)
        compute_variances!(variance_buffers, feature_matrix, config)
        
        # Find low variance features
        low_var_threshold = Float32(1e-6)
        low_var_indices = find_low_variance_features(variance_buffers, low_var_threshold)
        
        @test 1 in low_var_indices  # Constant feature
        @test 5 in low_var_indices  # Very low variance
        @test !(10 in low_var_indices)  # Should not be flagged
        
        # Test GPU marking
        low_var_mask = CUDA.zeros(Bool, n_features)
        mark_low_variance_features!(low_var_mask, variance_buffers, low_var_threshold)
        
        low_var_mask_cpu = Array(low_var_mask)
        @test low_var_mask_cpu[1] == true
        @test low_var_mask_cpu[5] == true
        @test low_var_mask_cpu[10] == false
    end
    
    @testset "Edge Cases" begin
        # Single sample
        n_samples = 1
        n_features = 5
        
        feature_matrix = create_feature_matrix(n_samples, n_features)
        variance_buffers = create_variance_buffers(n_features)
        
        test_data = rand(Float32, n_samples, n_features)
        copyto!(view(feature_matrix.data, 1:n_samples, 1:n_features), test_data)
        
        config = create_variance_config(n_features, n_samples)
        compute_variances!(variance_buffers, feature_matrix, config)
        
        variances_cpu = Array(variance_buffers.variances)
        
        # Single sample should have epsilon variance
        @test all(variances_cpu .== config.epsilon)
        
        # Large values test (numerical stability)
        n_samples = 100
        feature_matrix = create_feature_matrix(n_samples, n_features)
        
        test_data = Float32(1e6) .+ randn(Float32, n_samples, n_features)
        copyto!(view(feature_matrix.data, 1:n_samples, 1:n_features), test_data)
        
        compute_variances!(variance_buffers, feature_matrix, config)
        
        variances_cpu = Array(variance_buffers.variances)
        means_cpu = Array(variance_buffers.means)
        
        # Should still compute reasonable variance despite large values
        @test all(isfinite.(variances_cpu))
        @test all(isapprox.(variances_cpu, 1.0f0, rtol=0.2))
        @test all(isapprox.(means_cpu, Float32(1e6), rtol=1e-5))
    end
    
    @testset "Performance Benchmark" begin
        # Test with realistic size
        n_samples = 100000
        n_features = 1000
        
        feature_matrix = create_feature_matrix(n_samples, n_features)
        variance_buffers = create_variance_buffers(n_features)
        
        # Fill with random data
        Random.seed!(42)
        test_data = randn(Float32, n_samples, n_features)
        copyto!(view(feature_matrix.data, 1:n_samples, 1:n_features), test_data)
        
        config = create_variance_config(n_features, n_samples)
        
        # Warmup
        compute_variances!(variance_buffers, feature_matrix, config)
        CUDA.synchronize()
        
        # Time Welford's algorithm
        elapsed_welford = CUDA.@elapsed begin
            compute_variances!(variance_buffers, feature_matrix, config)
        end
        
        # Time warp shuffle (if applicable)
        if n_features <= 1024  # Reasonable block limit
            elapsed_warp = CUDA.@elapsed begin
                compute_variances_warp_shuffle!(variance_buffers, feature_matrix, config)
            end
            println("\nVariance calculation (warp shuffle) for $n_samples samples × $n_features features: $(round(elapsed_warp * 1000, digits=2)) ms")
        end
        
        println("Variance calculation (Welford) for $n_samples samples × $n_features features: $(round(elapsed_welford * 1000, digits=2)) ms")
        
        # Should be fast
        @test elapsed_welford < 1.0  # Less than 1 second
        
        # Verify results are reasonable
        variances_cpu = Array(variance_buffers.variances)
        @test all(isfinite.(variances_cpu))
        @test all(variances_cpu .>= 0)
    end
    
    @testset "CPU Reference Comparison" begin
        n_samples = 1000
        n_features = 50
        
        # Generate test data
        Random.seed!(42)
        test_data_cpu = randn(Float32, n_samples, n_features)
        
        # CPU reference
        cpu_means = vec(mean(test_data_cpu, dims=1))
        cpu_vars = vec(var(test_data_cpu, dims=1))
        
        # GPU computation
        feature_matrix = create_feature_matrix(n_samples, n_features)
        variance_buffers = create_variance_buffers(n_features)
        copyto!(view(feature_matrix.data, 1:n_samples, 1:n_features), test_data_cpu)
        
        config = create_variance_config(n_features, n_samples)
        compute_variances!(variance_buffers, feature_matrix, config)
        
        gpu_means = Array(variance_buffers.means)
        gpu_vars = Array(variance_buffers.variances)
        
        # Compare (allowing for epsilon in GPU version)
        @test all(isapprox.(gpu_means, cpu_means, rtol=1e-5))
        @test all(isapprox.(gpu_vars .- config.epsilon, cpu_vars, rtol=1e-5))
    end
end

println("\n✅ Variance calculation tests completed!")