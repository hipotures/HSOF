using Test
using CUDA
using Statistics
using Random

# Include the feature ranking module
include("../../src/stage1_filter/feature_ranking.jl")

using .FeatureRanking
using .FeatureRanking.GPUMemoryLayout
using .FeatureRanking.ThresholdManagement

@testset "Feature Ranking Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU feature ranking tests"
        return
    end
    
    @testset "RankingConfig Creation" begin
        config = create_ranking_config(1000, 10000)
        
        @test config.n_features == 1000
        @test config.n_samples == 10000
        @test config.target_features == 500  # Default TARGET_FEATURES
        @test config.mi_weight == Float32(1.0)
        @test config.correlation_penalty == Float32(0.5)
        @test config.variance_weight == Float32(0.1)
        @test config.use_gpu_sort == true
        @test config.correlation_graph == true
        @test config.pre_filter_variance == true
        
        # Test with custom parameters
        config2 = create_ranking_config(500, 5000,
                                      target_features=100,
                                      mi_weight=Float32(0.8),
                                      correlation_penalty=Float32(1.0),
                                      use_gpu_sort=false)
        
        @test config2.target_features == 100
        @test config2.mi_weight == Float32(0.8)
        @test config2.correlation_penalty == Float32(1.0)
        @test config2.use_gpu_sort == false
    end
    
    @testset "Simple Feature Ranking" begin
        n_samples = 1000
        n_features = 50
        target_features = 10
        
        # Create test data
        feature_matrix = create_feature_matrix(n_samples, n_features)
        Random.seed!(42)
        
        # Create features with varying informativeness
        test_data = randn(Float32, n_samples, n_features)
        
        # Make first 10 features highly informative
        target_data = CUDA.zeros(Float32, n_samples)
        CUDA.@allowscalar begin
            for i in 1:n_samples
                # Target depends on first 10 features
                target_data[i] = sum(test_data[i, j] for j in 1:10) > 0 ? 1.0f0 : 0.0f0
            end
        end
        
        # Make features 11-20 have low variance
        for j in 11:20
            test_data[:, j] .= 1.0f0 .+ 0.001f0 * randn(Float32, n_samples)
        end
        
        # Make features 21-30 highly correlated with features 1-10
        for j in 21:30
            test_data[:, j] = 0.95f0 * test_data[:, j-20] + 0.05f0 * randn(Float32, n_samples)
        end
        
        copyto!(view(feature_matrix.data, 1:n_samples, 1:n_features), test_data)
        
        # Create buffers
        histogram_buffers = create_histogram_buffers(n_features)
        correlation_matrix = create_correlation_matrix(n_features, n_samples)
        variance_buffers = create_variance_buffers(n_features)
        ranking_buffers = create_ranking_buffers(n_features)
        
        # Create configurations
        threshold_config = create_default_config()
        ranking_config = create_ranking_config(n_features, n_samples,
                                             target_features=target_features)
        
        # Perform ranking and selection
        n_selected = rank_and_select_features!(
            ranking_buffers,
            feature_matrix,
            target_data,
            histogram_buffers,
            correlation_matrix,
            variance_buffers,
            threshold_config,
            ranking_config
        )
        
        @test n_selected == target_features
        
        # Get selected indices
        selected_indices = Array(ranking_buffers.selected_indices)[1:n_selected]
        
        # Check that informative features are selected
        # At least half should be from the first 10 features
        informative_selected = sum(idx <= 10 for idx in selected_indices)
        @test informative_selected >= 5
        
        # Check that low variance features are filtered out
        low_var_selected = sum(11 <= idx <= 20 for idx in selected_indices)
        @test low_var_selected <= 2  # Should filter most low variance features
    end
    
    @testset "Correlation Filtering" begin
        n_samples = 500
        n_features = 20
        target_features = 10
        
        # Create test data with highly correlated features
        feature_matrix = create_feature_matrix(n_samples, n_features)
        Random.seed!(42)
        
        test_data = randn(Float32, n_samples, n_features)
        
        # Create correlated feature groups
        # Group 1: features 1-5 are highly correlated
        base1 = randn(Float32, n_samples)
        for j in 1:5
            test_data[:, j] = base1 .+ 0.1f0 * randn(Float32, n_samples)
        end
        
        # Group 2: features 6-10 are highly correlated
        base2 = randn(Float32, n_samples)
        for j in 6:10
            test_data[:, j] = base2 .+ 0.1f0 * randn(Float32, n_samples)
        end
        
        # Features 11-20 are independent
        
        copyto!(view(feature_matrix.data, 1:n_samples, 1:n_features), test_data)
        
        # Create target that depends on one feature from each group
        target_data = CUDA.zeros(Float32, n_samples)
        CUDA.@allowscalar begin
            for i in 1:n_samples
                target_data[i] = (test_data[i, 1] + test_data[i, 6]) > 0 ? 1.0f0 : 0.0f0
            end
        end
        
        # Create buffers
        histogram_buffers = create_histogram_buffers(n_features)
        correlation_matrix = create_correlation_matrix(n_features, n_samples)
        variance_buffers = create_variance_buffers(n_features)
        ranking_buffers = create_ranking_buffers(n_features)
        
        # Create configurations with high correlation penalty
        threshold_config = create_default_config()
        threshold_config.adaptive_corr_threshold = Float32(0.9)  # Lower threshold
        
        ranking_config = create_ranking_config(n_features, n_samples,
                                             target_features=target_features,
                                             correlation_penalty=Float32(1.0))  # Full penalty
        
        # Perform ranking and selection
        n_selected = rank_and_select_features!(
            ranking_buffers,
            feature_matrix,
            target_data,
            histogram_buffers,
            correlation_matrix,
            variance_buffers,
            threshold_config,
            ranking_config
        )
        
        # Get selected indices
        selected_indices = Array(ranking_buffers.selected_indices)[1:n_selected]
        
        # Check that we don't select too many from correlated groups
        group1_selected = sum(1 <= idx <= 5 for idx in selected_indices)
        group2_selected = sum(6 <= idx <= 10 for idx in selected_indices)
        
        @test group1_selected <= 2  # Should select at most 2 from each correlated group
        @test group2_selected <= 2
    end
    
    @testset "Feature Selection Application" begin
        n_samples = 100
        n_features = 20
        n_selected = 5
        
        # Create test data
        feature_matrix = create_feature_matrix(n_samples, n_features)
        Random.seed!(42)
        test_data = randn(Float32, n_samples, n_features)
        copyto!(view(feature_matrix.data, 1:n_samples, 1:n_features), test_data)
        
        # Feature names are already set in create_feature_matrix
        
        # Create mock selected indices
        selected_indices = CuArray(Int32[2, 5, 8, 11, 15])
        
        # Apply feature selection
        selected_matrix = apply_feature_selection(
            feature_matrix,
            selected_indices,
            n_selected
        )
        
        @test selected_matrix.n_features == n_selected
        @test selected_matrix.n_samples == n_samples
        @test length(selected_matrix.feature_names) == n_selected
        
        # Check that correct features were selected
        @test selected_matrix.feature_names[1] == "feature_2"
        @test selected_matrix.feature_names[2] == "feature_5"
        @test selected_matrix.feature_names[5] == "feature_15"
        
        # Check data correctness
        selected_data_cpu = Array(selected_matrix.data[1:n_samples, 1:n_selected])
        @test selected_data_cpu[:, 1] ≈ test_data[:, 2]
        @test selected_data_cpu[:, 3] ≈ test_data[:, 8]
    end
    
    @testset "Selection Summary" begin
        n_features = 100
        n_selected = 20
        
        # Create mock buffers
        ranking_buffers = create_ranking_buffers(n_features)
        variance_buffers = create_variance_buffers(n_features)
        
        # Fill with test data
        mi_scores = rand(Float32, n_features)
        variances = rand(Float32, n_features) .* 10
        selected_indices = Int32.(sortperm(mi_scores, rev=true)[1:n_selected])
        
        copyto!(ranking_buffers.mi_scores, mi_scores)
        copyto!(variance_buffers.variances, variances)
        copyto!(ranking_buffers.selected_indices, selected_indices)
        
        # Get summary
        summary = get_selection_summary(ranking_buffers, variance_buffers, n_selected)
        
        @test summary["n_selected"] == n_selected
        @test length(summary["selected_indices"]) == n_selected
        @test summary["mi_score_range"][1] <= summary["mi_score_range"][2]
        @test summary["variance_range"][1] <= summary["variance_range"][2]
        @test 0 <= summary["mi_score_mean"] <= 1
        @test summary["variance_mean"] > 0
    end
    
    @testset "End-to-End Feature Reduction" begin
        # Test realistic scenario: 5000 -> 500 features
        n_samples = 10000
        n_features = 1000  # Reduced for testing
        target_features = 100  # Reduced proportionally
        
        # Create realistic test data
        feature_matrix = create_feature_matrix(n_samples, n_features)
        Random.seed!(42)
        
        # Create features with varying quality
        test_data = randn(Float32, n_samples, n_features)
        
        # 10% highly informative features
        n_informative = div(n_features, 10)
        coefficients = randn(Float32, n_informative)
        
        # Create target based on informative features
        target_data = CUDA.zeros(Float32, n_samples)
        CUDA.@allowscalar begin
            for i in 1:n_samples
                linear_combo = sum(coefficients[j] * test_data[i, j] for j in 1:n_informative)
                target_data[i] = linear_combo > 0 ? 1.0f0 : 0.0f0
            end
        end
        
        # Add noise features (70%)
        # Already random from initialization
        
        # Add low variance features (10%)
        n_low_var = div(n_features, 10)
        for j in (n_features - n_low_var + 1):n_features
            test_data[:, j] .= 1.0f0 .+ 0.0001f0 * randn(Float32, n_samples)
        end
        
        # Add correlated features (10%)
        n_correlated = div(n_features, 10)
        for j in (n_informative + 1):(n_informative + n_correlated)
            source_idx = rand(1:n_informative)
            test_data[:, j] = 0.9f0 * test_data[:, source_idx] .+ 0.1f0 * randn(Float32, n_samples)
        end
        
        copyto!(view(feature_matrix.data, 1:n_samples, 1:n_features), test_data)
        
        # Create all necessary buffers
        histogram_buffers = create_histogram_buffers(n_features)
        correlation_matrix = create_correlation_matrix(n_features, n_samples)
        variance_buffers = create_variance_buffers(n_features)
        ranking_buffers = create_ranking_buffers(n_features)
        
        # Create configurations
        threshold_config = create_default_config()
        ranking_config = create_ranking_config(n_features, n_samples,
                                             target_features=target_features)
        
        # Time the feature selection
        CUDA.synchronize()
        elapsed = CUDA.@elapsed begin
            n_selected = rank_and_select_features!(
                ranking_buffers,
                feature_matrix,
                target_data,
                histogram_buffers,
                correlation_matrix,
                variance_buffers,
                threshold_config,
                ranking_config
            )
        end
        
        println("\nFeature selection from $n_features to $target_features features: $(round(elapsed, digits=3)) seconds")
        
        @test n_selected == target_features
        @test elapsed < 10.0  # Should be fast even for 1000 features
        
        # Get summary
        summary = get_selection_summary(ranking_buffers, variance_buffers, n_selected)
        
        println("Selected feature MI scores: $(summary["mi_score_range"])")
        println("Selected feature variances: $(summary["variance_range"])")
        
        # Most selected features should be from informative set
        selected_indices = summary["selected_indices"]
        informative_selected = sum(idx <= n_informative for idx in selected_indices)
        
        @test informative_selected >= div(target_features, 2)  # At least half should be informative
    end
end

println("\n✅ Feature ranking tests completed!")