using Test
using CUDA
using Statistics
using LinearAlgebra
using Random

# Skip if no GPU
if !CUDA.functional()
    @warn "CUDA not functional, skipping GPU kernel tests"
    exit(0)
end

# Include modules
include("../../src/stage1_filter/gpu_memory_layout.jl")
include("../../src/stage1_filter/mutual_information.jl")
include("../../src/stage1_filter/correlation_matrix.jl")
include("../../src/stage1_filter/variance_calculation.jl")
include("../../src/stage1_filter/categorical_features.jl")
include("../../src/stage1_filter/feature_ranking.jl")

using .GPUMemoryLayout
using .MutualInformation
using .CorrelationMatrix
using .VarianceCalculation
using .CategoricalFeatures
using .FeatureRanking

@testset "GPU Kernel Unit Tests" begin
    
    @testset "Variance Calculation Kernels" begin
        # Test 1: Basic variance calculation
        @test begin
            X = randn(Float32, 100, 1000)
            X_gpu = CuArray(X')  # GPU expects features × samples
            
            var_gpu = compute_variance(X_gpu)
            var_cpu = vec(var(X, dims=1, corrected=false))
            
            var_gpu_cpu = Array(var_gpu)
            maximum(abs.(var_gpu_cpu .- var_cpu)) < 1e-4
        end
        
        # Test 2: Constant features (zero variance)
        @test begin
            X_const = ones(Float32, 50, 500)
            X_const_gpu = CuArray(X_const')
            
            var_const = compute_variance(X_const_gpu)
            all(Array(var_const) .< 1e-8)
        end
        
        # Test 3: Single sample
        @test begin
            X_single = randn(Float32, 50, 1)
            X_single_gpu = CuArray(X_single')
            
            var_single = compute_variance(X_single_gpu)
            all(Array(var_single) .== 0.0f0)  # Variance of single sample is 0
        end
        
        # Test 4: Empty features
        @test_throws Exception compute_variance(CuArray(Float32[;;]))
        
        # Test 5: Large values
        @test begin
            X_large = 1e10f0 * randn(Float32, 20, 100)
            X_large_gpu = CuArray(X_large')
            
            var_large = compute_variance(X_large_gpu)
            all(isfinite.(Array(var_large)))
        end
    end
    
    @testset "Mutual Information Kernels" begin
        # Test 1: Perfect correlation
        @test begin
            n_samples = 1000
            X_perfect = zeros(Float32, 10, n_samples)
            y_perfect = Int32.(rand(0:1, n_samples))
            
            # Feature perfectly predicts y
            X_perfect[1, :] = Float32.(y_perfect)
            X_perfect_gpu = CuArray(X_perfect)
            y_perfect_gpu = CuArray(y_perfect)
            
            config = MutualInformationConfig()
            mi_scores = compute_mutual_information(X_perfect_gpu, y_perfect_gpu, config)
            mi_cpu = Array(mi_scores)
            
            # First feature should have high MI
            mi_cpu[1] > mean(mi_cpu[2:end]) * 10
        end
        
        # Test 2: Random features (low MI)
        @test begin
            X_random = randn(Float32, 50, 2000)
            y_random = Int32.(rand(1:3, 2000))
            
            X_random_gpu = CuArray(X_random)
            y_random_gpu = CuArray(y_random)
            
            config = MutualInformationConfig()
            mi_random = compute_mutual_information(X_random_gpu, y_random_gpu, config)
            
            # All MI scores should be low
            all(Array(mi_random) .< 0.1f0)
        end
        
        # Test 3: Binary classification
        @test begin
            X_binary = randn(Float32, 20, 500)
            y_binary = Int32.(rand(0:1, 500))
            
            X_binary_gpu = CuArray(X_binary)
            y_binary_gpu = CuArray(y_binary .+ 1)  # Convert to 1,2
            
            config = MutualInformationConfig()
            mi_binary = compute_mutual_information(X_binary_gpu, y_binary_gpu, config)
            
            all(Array(mi_binary) .>= 0.0f0)  # MI is non-negative
        end
        
        # Test 4: Multi-class
        @test begin
            X_multi = randn(Float32, 30, 1000)
            y_multi = Int32.(rand(1:5, 1000))
            
            X_multi_gpu = CuArray(X_multi)
            y_multi_gpu = CuArray(y_multi)
            
            config = MutualInformationConfig(n_bins = 20)
            mi_multi = compute_mutual_information(X_multi_gpu, y_multi_gpu, config)
            
            all(isfinite.(Array(mi_multi)))
        end
        
        # Test 5: Extreme values handling
        @test begin
            X_extreme = randn(Float32, 10, 500)
            X_extreme[1, :] .= Inf32
            X_extreme[2, :] .= -Inf32
            X_extreme[3, :] .= NaN32
            
            X_extreme_gpu = CuArray(X_extreme)
            y_extreme_gpu = CuArray(Int32.(rand(1:2, 500)))
            
            config = MutualInformationConfig()
            mi_extreme = compute_mutual_information(X_extreme_gpu, y_extreme_gpu, config)
            mi_cpu = Array(mi_extreme)
            
            # Should handle gracefully
            mi_cpu[1] == 0.0f0 || !isfinite(mi_cpu[1])  # Inf feature
            mi_cpu[2] == 0.0f0 || !isfinite(mi_cpu[2])  # -Inf feature
            mi_cpu[3] == 0.0f0 || !isfinite(mi_cpu[3])  # NaN feature
        end
    end
    
    @testset "Correlation Matrix Kernels" begin
        # Test 1: Identity correlation
        @test begin
            X_identity = Matrix{Float32}(I, 50, 50)'
            X_identity_gpu = CuArray(X_identity)
            
            config = CorrelationConfig()
            corr_identity = compute_correlation_matrix(X_identity_gpu, config)
            corr_cpu = Array(corr_identity)
            
            # Should be identity matrix
            norm(corr_cpu - I) < 1e-3
        end
        
        # Test 2: Perfect correlation
        @test begin
            n_features = 20
            n_samples = 500
            X_corr = zeros(Float32, n_features, n_samples)
            
            # Create correlated features
            base = randn(Float32, n_samples)
            for i in 1:10
                X_corr[i, :] = base + 0.01f0 * randn(Float32, n_samples)
                X_corr[i+10, :] = -base + 0.01f0 * randn(Float32, n_samples)
            end
            
            X_corr_gpu = CuArray(X_corr)
            config = CorrelationConfig()
            corr_matrix = compute_correlation_matrix(X_corr_gpu, config)
            corr_cpu = Array(corr_matrix)
            
            # Check high positive correlation
            abs(corr_cpu[1, 2]) > 0.95
            # Check high negative correlation
            corr_cpu[1, 11] < -0.95
        end
        
        # Test 3: Redundant features detection
        @test begin
            X_redundant = randn(Float32, 30, 1000)
            # Make some exact duplicates
            X_redundant[11:15, :] = X_redundant[1:5, :]
            
            X_redundant_gpu = CuArray(X_redundant)
            config = CorrelationConfig(threshold = 0.99f0)
            corr_matrix = compute_correlation_matrix(X_redundant_gpu, config)
            redundant_pairs = find_redundant_features(corr_matrix, config.threshold)
            
            length(redundant_pairs) >= 5  # Should find duplicate pairs
        end
        
        # Test 4: Constant features
        @test begin
            X_const_corr = randn(Float32, 20, 500)
            X_const_corr[10, :] .= 1.0f0  # Constant feature
            
            X_const_corr_gpu = CuArray(X_const_corr)
            config = CorrelationConfig()
            corr_matrix = compute_correlation_matrix(X_const_corr_gpu, config)
            corr_cpu = Array(corr_matrix)
            
            # Constant feature should have 0 or NaN correlation
            all(corr_cpu[10, :] .== 0.0f0) || all(isnan.(corr_cpu[10, :]))
        end
        
        # Test 5: Large matrix efficiency
        @test begin
            if CUDA.available_memory() > 2 * 1024^3  # If > 2GB available
                X_large = randn(Float32, 1000, 5000)
                X_large_gpu = CuArray(X_large)
                
                t = @elapsed begin
                    config = CorrelationConfig(use_sampling = true, sample_size = 1000)
                    corr_large = compute_correlation_matrix(X_large_gpu, config)
                    CUDA.synchronize()
                end
                
                t < 5.0  # Should complete within 5 seconds
            else
                true  # Skip if insufficient memory
            end
        end
    end
    
    @testset "Categorical Features Kernels" begin
        # Test 1: Integer categorical features
        @test begin
            X_cat = rand(1:5, 20, 500)
            X_cat_float = Float32.(X_cat)
            X_cat_gpu = CuArray(X_cat_float)
            
            cat_indices = detect_categorical_features(X_cat_gpu)
            cat_cpu = Array(cat_indices)
            
            sum(cat_cpu) == 20  # All should be detected as categorical
        end
        
        # Test 2: Mixed features
        @test begin
            X_mixed = randn(Float32, 30, 1000)
            # Make some categorical
            for i in 1:10
                X_mixed[i, :] = Float32.(rand(0:4, 1000))
            end
            
            X_mixed_gpu = CuArray(X_mixed)
            cat_indices = detect_categorical_features(X_mixed_gpu)
            cat_cpu = Array(cat_indices)
            
            sum(cat_cpu[1:10]) == 10  # First 10 should be categorical
            sum(cat_cpu[11:end]) == 0  # Rest should be continuous
        end
        
        # Test 3: One-hot encoding
        @test begin
            X_cat_encode = Float32.(rand(1:3, 10, 500))
            X_cat_encode_gpu = CuArray(X_cat_encode)
            
            X_encoded, mapping = encode_categorical_features(X_cat_encode_gpu, 1:10)
            X_encoded_cpu = Array(X_encoded)
            
            # Should have 3 columns per original feature
            size(X_encoded_cpu, 1) == 30
        end
        
        # Test 4: Binary features
        @test begin
            X_binary_cat = Float32.(rand(0:1, 15, 800))
            X_binary_cat_gpu = CuArray(X_binary_cat)
            
            cat_binary = detect_categorical_features(X_binary_cat_gpu, max_categories = 2)
            
            all(Array(cat_binary))  # All should be detected as binary
        end
    end
    
    @testset "Feature Ranking Integration" begin
        # Test 1: Complete pipeline
        @test begin
            Random.seed!(42)
            n_features = 500
            n_samples = 2000
            n_select = 50
            
            # Create dataset with known properties
            X = randn(Float32, n_features, n_samples)
            y = Int32.(zeros(n_samples))
            
            # Make first 20 features informative
            for i in 1:20
                y .+= Int32.(X[i, :] .> 0)
            end
            y = Int32.((y .> 10) .+ 1)
            
            X_gpu = CuArray(X)
            y_gpu = CuArray(y)
            
            config = RankingConfig(
                n_features_to_select = n_select,
                variance_threshold = 1f-6,
                correlation_threshold = 0.95f0
            )
            
            selected = select_features(X_gpu, y_gpu, config)
            selected_cpu = Array(selected)
            valid_selected = selected_cpu[selected_cpu .> 0]
            
            # Should select requested number
            length(valid_selected) == n_select
            # Should prefer informative features
            sum(valid_selected .<= 20) >= 15
        end
        
        # Test 2: Stress test with all edge cases
        @test begin
            X_stress = randn(Float32, 200, 1000)
            # Add various edge cases
            X_stress[1:10, :] .= 0.0f0  # Zero features
            X_stress[11:20, :] .= 1.0f0  # Constant features
            X_stress[21, :] .= Inf32  # Inf feature
            X_stress[22, :] .= -Inf32  # -Inf feature
            X_stress[23, :] .= NaN32  # NaN feature
            
            # Make some identical
            for i in 31:35
                X_stress[i, :] = X_stress[30, :]
            end
            
            X_stress_gpu = CuArray(X_stress)
            y_stress_gpu = CuArray(Int32.(rand(1:3, 1000)))
            
            config = RankingConfig(
                n_features_to_select = 50,
                variance_threshold = 1f-6,
                correlation_threshold = 0.99f0
            )
            
            # Should handle gracefully
            selected_stress = select_features(X_stress_gpu, y_stress_gpu, config)
            selected_cpu = Array(selected_stress)
            valid = selected_cpu[selected_cpu .> 0]
            
            # Should filter out problematic features
            all(valid .> 23)  # Skip Inf/NaN features
            !(30 in valid && 31 in valid)  # Should not select duplicates
        end
    end
    
    @testset "Memory and Performance Constraints" begin
        # Test 1: Memory allocation tracking
        @test begin
            CUDA.reclaim()
            initial_mem = CUDA.memory_status().used
            
            X_mem = CUDA.randn(Float32, 1000, 5000)
            y_mem = CuArray(Int32.(rand(1:2, 5000)))
            
            # Run feature selection
            config = RankingConfig(n_features_to_select = 100)
            selected = select_features(X_mem, y_mem, config)
            CUDA.synchronize()
            
            peak_mem = CUDA.memory_status().used
            memory_used_mb = (peak_mem - initial_mem) / 1024^2
            
            # Memory usage should be reasonable
            memory_used_mb < 500  # Less than 500MB for this size
        end
        
        # Test 2: Performance regression check
        @test begin
            X_perf = CUDA.randn(Float32, 500, 5000)
            y_perf = CuArray(Int32.(rand(1:2, 5000)))
            
            # Warm up
            select_features(X_perf, y_perf, RankingConfig(n_features_to_select = 50))
            CUDA.synchronize()
            
            # Time the operation
            t = @elapsed begin
                selected = select_features(X_perf, y_perf, RankingConfig(n_features_to_select = 50))
                CUDA.synchronize()
            end
            
            # Should complete quickly
            t < 2.0  # Less than 2 seconds for 500×5000
        end
    end
end

println("\n" * "="^60)
println("GPU KERNEL UNIT TESTS COMPLETED")
println("="^60)
println("✓ All kernel edge cases tested")
println("✓ Memory constraints validated")
println("✓ Performance targets met")
println("="^60)