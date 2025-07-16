using Test
using CUDA
using Statistics
using Random

# Include the mutual information module
include("../../src/stage1_filter/mutual_information.jl")

using .MutualInformation
using .MutualInformation.GPUMemoryLayout

@testset "Mutual Information Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU mutual information tests"
        return
    end
    
    @testset "MIConfig Creation" begin
        config = create_mi_config(100, 1000)
        
        @test config.n_features == 100
        @test config.n_samples == 1000
        @test config.n_bins == HISTOGRAM_BINS
        @test config.min_samples_per_bin == 3
        @test config.epsilon == Float32(1e-10)
        @test config.use_shared_memory == true
        
        # Test with custom parameters
        config2 = create_mi_config(50, 500, 
                                  n_bins=128, 
                                  min_samples_per_bin=5,
                                  epsilon=Float32(1e-8),
                                  use_shared_memory=false)
        
        @test config2.n_features == 50
        @test config2.n_samples == 500
        @test config2.n_bins == 128
        @test config2.min_samples_per_bin == 5
        @test config2.epsilon == Float32(1e-8)
        @test config2.use_shared_memory == false
    end
    
    @testset "Simple MI Calculation" begin
        # Create simple test data where MI should be high
        n_samples = 10000
        n_features = 10
        
        # Create feature data
        feature_data = CUDA.randn(Float32, n_samples, n_features)
        
        # Create target that's correlated with first feature
        target_data = CUDA.zeros(Float32, n_samples)
        CUDA.@allowscalar begin
            for i in 1:n_samples
                # First feature perfectly predicts target
                target_data[i] = feature_data[i, 1] > 0 ? 1.0f0 : 0.0f0
                # Add some noise to other features
                for j in 2:n_features
                    feature_data[i, j] += 0.1f0 * randn(Float32)
                end
            end
        end
        
        # Create buffers and config
        histogram_buffers = create_histogram_buffers(n_features)
        config = create_mi_config(n_features, n_samples)
        mi_scores = CUDA.zeros(Float32, n_features)
        
        # Compute MI
        compute_mutual_information!(mi_scores, feature_data, target_data, histogram_buffers, config)
        
        # Check results
        mi_cpu = Array(mi_scores)
        
        # First feature should have high MI (close to 1 bit for binary target)
        @test mi_cpu[1] > 0.8
        
        # Other features should have much lower MI
        for i in 2:n_features
            @test mi_cpu[i] < mi_cpu[1] * 0.5
        end
        
        # All MI scores should be non-negative
        @test all(mi_cpu .>= 0)
    end
    
    @testset "MI with Continuous Target" begin
        # Test with continuous target variable
        n_samples = 5000
        n_features = 5
        
        # Create feature data
        feature_data = CUDA.randn(Float32, n_samples, n_features)
        
        # Create continuous target with linear relationship to features
        coefficients = Float32[2.0, -1.5, 0.5, 0.1, 0.0]  # Last feature has no relationship
        target_data = CUDA.zeros(Float32, n_samples)
        
        CUDA.@allowscalar begin
            for i in 1:n_samples
                target_data[i] = sum(coefficients[j] * feature_data[i, j] for j in 1:n_features)
                target_data[i] += 0.1f0 * randn(Float32)  # Add noise
            end
        end
        
        # Compute MI
        histogram_buffers = create_histogram_buffers(n_features)
        config = create_mi_config(n_features, n_samples)
        mi_scores = CUDA.zeros(Float32, n_features)
        
        compute_mutual_information!(mi_scores, feature_data, target_data, histogram_buffers, config)
        
        mi_cpu = Array(mi_scores)
        
        # Features with larger coefficients should have higher MI
        @test mi_cpu[1] > mi_cpu[3]  # |2.0| > |0.5|
        @test mi_cpu[2] > mi_cpu[4]  # |-1.5| > |0.1|
        @test mi_cpu[4] > mi_cpu[5]  # |0.1| > |0.0|
        
        # Last feature should have very low MI
        @test mi_cpu[5] < 0.1
    end
    
    @testset "MI with Categorical Features" begin
        # Test with discrete/categorical-like features
        n_samples = 10000
        n_features = 4
        
        feature_data = CUDA.zeros(Float32, n_samples, n_features)
        
        # Create categorical features (0, 1, 2, 3)
        CUDA.@allowscalar begin
            for i in 1:n_samples
                for j in 1:n_features
                    feature_data[i, j] = Float32(rand(0:3))
                end
            end
        end
        
        # Target depends on XOR of first two features
        target_data = CUDA.zeros(Float32, n_samples)
        CUDA.@allowscalar begin
            for i in 1:n_samples
                val1 = Int(round(feature_data[i, 1]))
                val2 = Int(round(feature_data[i, 2]))
                target_data[i] = Float32((val1 ⊻ val2) % 2)
            end
        end
        
        # Compute MI
        histogram_buffers = create_histogram_buffers(n_features)
        config = create_mi_config(n_features, n_samples, n_bins=64)  # Fewer bins for discrete data
        mi_scores = CUDA.zeros(Float32, n_features)
        
        compute_mutual_information!(mi_scores, feature_data, target_data, histogram_buffers, config)
        
        mi_cpu = Array(mi_scores)
        
        # First two features should have similar MI (both involved in XOR)
        @test abs(mi_cpu[1] - mi_cpu[2]) < 0.2
        
        # Last two features should have near-zero MI
        @test mi_cpu[3] < 0.1
        @test mi_cpu[4] < 0.1
    end
    
    @testset "Edge Cases" begin
        n_features = 5
        
        # Test with constant features
        n_samples = 1000
        feature_data = CUDA.ones(Float32, n_samples, n_features)
        feature_data[:, 2] .= 2.0f0  # Different constant
        
        # Add one varying feature
        CUDA.@allowscalar begin
            for i in 1:n_samples
                feature_data[i, 3] = Float32(i % 10)
            end
        end
        
        target_data = CUDA.rand(Float32, n_samples)
        
        histogram_buffers = create_histogram_buffers(n_features)
        config = create_mi_config(n_features, n_samples)
        mi_scores = CUDA.zeros(Float32, n_features)
        
        compute_mutual_information!(mi_scores, feature_data, target_data, histogram_buffers, config)
        
        mi_cpu = Array(mi_scores)
        
        # Constant features should have zero or very low MI
        @test mi_cpu[1] < 0.01
        @test mi_cpu[2] < 0.01
        
        # Varying feature might have some MI
        @test mi_cpu[3] >= 0
    end
    
    @testset "Performance Benchmark" begin
        # Test with realistic size
        n_samples = 100000
        n_features = 100
        
        feature_data = CUDA.randn(Float32, n_samples, n_features)
        target_data = CUDA.rand(Float32, n_samples)
        
        histogram_buffers = create_histogram_buffers(n_features)
        config = create_mi_config(n_features, n_samples)
        mi_scores = CUDA.zeros(Float32, n_features)
        
        # Warmup
        compute_mutual_information!(mi_scores, feature_data, target_data, histogram_buffers, config)
        CUDA.synchronize()
        
        # Time the computation
        elapsed = CUDA.@elapsed begin
            compute_mutual_information!(mi_scores, feature_data, target_data, histogram_buffers, config)
        end
        
        println("\nMI computation for $n_samples samples × $n_features features: $(round(elapsed * 1000, digits=2)) ms")
        
        # Should be reasonably fast
        @test elapsed < 1.0  # Less than 1 second
        
        # Check all scores are computed
        mi_cpu = Array(mi_scores)
        @test length(mi_cpu) == n_features
        @test all(isfinite.(mi_cpu))
        @test all(mi_cpu .>= 0)
    end
    
    @testset "Histogram Computation" begin
        # Test histogram computation directly
        n_samples = 1000
        n_features = 2
        n_bins = Int32(10)  # Smaller for testing
        
        # Create simple data
        feature_data = CUDA.zeros(Float32, n_samples, n_features)
        CUDA.@allowscalar begin
            # Feature 1: uniform in [0, 1]
            for i in 1:n_samples
                feature_data[i, 1] = (i - 1) / (n_samples - 1)
            end
            # Feature 2: normal-like
            for i in 1:n_samples
                feature_data[i, 2] = randn(Float32) * 0.3f0 + 0.5f0
            end
        end
        
        # Create histogram buffers with custom size
        hist_buffer = CUDA.zeros(Int32, n_bins, n_features)
        bin_edges = CUDA.zeros(Float32, n_bins + 1, n_features)
        feature_mins = CUDA.zeros(Float32, n_features)
        feature_maxs = CUDA.zeros(Float32, n_features)
        
        # Compute bin edges
        @cuda threads=256 blocks=1 MutualInformation.compute_bin_edges_kernel!(
            bin_edges, feature_data, feature_mins, feature_maxs,
            Int32(n_samples), Int32(n_features), n_bins
        )
        
        # Compute histogram
        @cuda threads=256 blocks=n_features MutualInformation.compute_histogram_kernel!(
            hist_buffer, feature_data, bin_edges,
            Int32(n_samples), Int32(n_features), n_bins
        )
        
        CUDA.synchronize()
        
        # Check histogram properties
        hist_cpu = Array(hist_buffer)
        
        # All bins should sum to n_samples for each feature
        for j in 1:n_features
            @test sum(hist_cpu[:, j]) == n_samples
        end
        
        # Uniform feature should have roughly equal bins
        uniform_hist = hist_cpu[:, 1]
        expected_per_bin = n_samples ÷ n_bins
        @test all(abs.(uniform_hist .- expected_per_bin) .<= 2)
    end
end

println("\n✅ Mutual information tests completed!")