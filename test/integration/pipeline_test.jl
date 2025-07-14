# HSOF Pipeline Integration Tests
# Tests for the complete feature selection pipeline

using Test
using HSOF
using CUDA
using Random
using Statistics
using LinearAlgebra

Random.seed!(42)  # Reproducibility

@testset "HSOF Pipeline Integration" begin
    
    # Helper function to generate test data
    function generate_test_data(n_samples, n_features, n_informative)
        # Generate informative features
        X_informative = randn(n_samples, n_informative)
        
        # Generate noise features
        X_noise = randn(n_samples, n_features - n_informative)
        
        # Combine features
        X = hcat(X_informative, X_noise)
        
        # Generate target based on informative features
        true_weights = randn(n_informative)
        y_continuous = X_informative * true_weights + 0.1 * randn(n_samples)
        y = Int.(y_continuous .> median(y_continuous))
        
        return X, y, 1:n_informative  # Return true informative indices
    end
    
    @testset "Stage 1: Statistical Filtering" begin
        @info "Testing Stage 1: Statistical Filtering"
        
        # Generate test data
        X, y, true_indices = generate_test_data(1000, 100, 10)
        
        # Mock stage 1 filtering
        function stage1_filter(X, y; variance_threshold=0.01)
            n_features = size(X, 2)
            
            # Calculate variance for each feature
            variances = vec(var(X, dims=1))
            
            # Filter by variance
            selected = findall(v -> v > variance_threshold, variances)
            
            @info "Stage 1: Selected $(length(selected)) features from $n_features"
            return selected
        end
        
        # Run stage 1
        stage1_indices = stage1_filter(X, y)
        
        @test length(stage1_indices) < size(X, 2)
        @test length(stage1_indices) > 0
        @test all(1 .<= stage1_indices .<= size(X, 2))
        
        # Check that most informative features are retained
        retained_informative = length(intersect(stage1_indices, true_indices))
        retention_rate = retained_informative / length(true_indices)
        @test retention_rate >= 0.8  # At least 80% of informative features retained
        
        @info "Stage 1 retained $retained_informative/$(length(true_indices)) informative features"
    end
    
    @testset "Stage 2: MCTS Selection" begin
        @info "Testing Stage 2: MCTS Selection"
        
        # Use smaller data for MCTS test
        X, y, true_indices = generate_test_data(500, 50, 5)
        
        # Mock MCTS selection
        function stage2_mcts(X, y, initial_indices; target_features=10)
            n_features = length(initial_indices)
            
            # Simulate MCTS by selecting features with highest correlation to target
            correlations = Float64[]
            for idx in initial_indices
                corr = abs(cor(X[:, idx], Float64.(y)))
                push!(correlations, corr)
            end
            
            # Select top features
            n_selected = min(target_features, n_features)
            selected_local = partialsortperm(correlations, 1:n_selected, rev=true)
            selected_global = initial_indices[selected_local]
            
            @info "Stage 2: Selected $n_selected features from $n_features"
            return selected_global
        end
        
        # Run stage 2
        stage1_indices = 1:size(X, 2)  # Use all features for this test
        stage2_indices = stage2_mcts(X, y, stage1_indices)
        
        @test length(stage2_indices) <= 10
        @test length(stage2_indices) > 0
        @test all(in(stage1_indices), stage2_indices)
        
        # Check selection quality
        selected_informative = length(intersect(stage2_indices, true_indices))
        @test selected_informative >= 3  # At least 3 of 5 informative features
        
        @info "Stage 2 selected $selected_informative/$(length(true_indices)) informative features"
    end
    
    @testset "Stage 3: Ensemble Refinement" begin
        @info "Testing Stage 3: Ensemble Refinement"
        
        # Generate data
        X, y, true_indices = generate_test_data(200, 20, 5)
        
        # Mock ensemble refinement
        function stage3_ensemble(X, y, initial_indices; min_features=3, max_features=8)
            n_features = length(initial_indices)
            X_subset = X[:, initial_indices]
            
            # Simulate ensemble scoring
            importance_scores = Float64[]
            for i in 1:n_features
                # Simple importance: correlation + random noise
                score = abs(cor(X_subset[:, i], Float64.(y))) + 0.1 * rand()
                push!(importance_scores, score)
            end
            
            # Select final features based on importance
            n_final = clamp(round(Int, n_features * 0.5), min_features, max_features)
            selected_local = partialsortperm(importance_scores, 1:n_final, rev=true)
            selected_global = initial_indices[selected_local]
            
            @info "Stage 3: Selected $n_final features from $n_features"
            return selected_global, importance_scores[selected_local]
        end
        
        # Run stage 3
        initial_indices = 1:size(X, 2)
        final_indices, importance_scores = stage3_ensemble(X, y, initial_indices)
        
        @test 3 <= length(final_indices) <= 8
        @test length(importance_scores) == length(final_indices)
        @test all(importance_scores .>= 0)
        @test issorted(importance_scores, rev=true)
        
        @info "Stage 3 completed with $(length(final_indices)) features"
    end
    
    @testset "Complete Pipeline" begin
        @info "Testing complete pipeline integration"
        
        # Generate larger dataset
        X, y, true_indices = generate_test_data(1000, 500, 20)
        
        # Run complete pipeline
        function run_pipeline(X, y)
            # Stage 1: Filtering
            stage1_indices = stage1_filter(X, y, variance_threshold=0.001)
            X_stage1 = X[:, stage1_indices]
            
            # Stage 2: MCTS
            stage2_indices_local = stage2_mcts(X_stage1, y, 1:length(stage1_indices), target_features=50)
            stage2_indices = stage1_indices[stage2_indices_local]
            X_stage2 = X[:, stage2_indices]
            
            # Stage 3: Ensemble
            stage3_indices_local, scores = stage3_ensemble(X_stage2, y, 1:length(stage2_indices))
            final_indices = stage2_indices[stage3_indices_local]
            
            return (
                selected_indices = final_indices,
                feature_scores = scores,
                stage1_count = length(stage1_indices),
                stage2_count = length(stage2_indices),
                stage3_count = length(final_indices)
            )
        end
        
        # Time the pipeline
        pipeline_time = @elapsed results = run_pipeline(X, y)
        
        @test length(results.selected_indices) >= 3
        @test length(results.selected_indices) <= 20
        @test results.stage1_count > results.stage2_count > results.stage3_count
        @test pipeline_time < 60.0  # Should complete in under 60 seconds
        
        # Check quality
        selected_informative = length(intersect(results.selected_indices, true_indices))
        precision = selected_informative / length(results.selected_indices)
        recall = selected_informative / length(true_indices)
        f1_score = 2 * (precision * recall) / (precision + recall + eps())
        
        @info "Pipeline results:" results.stage1_count results.stage2_count results.stage3_count
        @info "Selection quality:" precision recall f1_score
        
        @test precision >= 0.3  # At least 30% of selected features are informative
        @test recall >= 0.2     # At least 20% of informative features are selected
    end
    
    @testset "GPU Pipeline" begin
        if CUDA.functional()
            @info "Testing GPU-accelerated pipeline"
            
            # Generate GPU data
            X_cpu, y_cpu, true_indices = generate_test_data(1000, 100, 10)
            X_gpu = CuArray(Float32.(X_cpu))
            y_gpu = CuArray(Float32.(y_cpu))
            
            # Mock GPU filtering
            function gpu_variance_filter(X_gpu, threshold)
                # Calculate variance on GPU
                n_samples = size(X_gpu, 1)
                means = sum(X_gpu, dims=1) / n_samples
                centered = X_gpu .- means
                variances = sum(centered .^ 2, dims=1) / n_samples
                
                # Find high variance features
                var_cpu = Array(vec(variances))
                selected = findall(v -> v > threshold, var_cpu)
                
                return selected
            end
            
            # Run GPU filtering
            gpu_time = CUDA.@elapsed begin
                selected = gpu_variance_filter(X_gpu, 0.5)
                CUDA.synchronize()
            end
            
            @test length(selected) > 0
            @test length(selected) < size(X_gpu, 2)
            @test gpu_time < 1.0  # Should be fast
            
            @info "GPU filtering completed in $(round(gpu_time*1000, digits=2)) ms"
            
            # Test memory usage
            initial_memory = CUDA.available_memory()
            
            # Allocate large array
            large_gpu = CUDA.zeros(10_000, 10_000)
            used_memory = initial_memory - CUDA.available_memory()
            
            @test used_memory > 0
            @info "GPU memory allocated: $(round(used_memory/2^20, digits=1)) MB"
            
            # Clean up
            large_gpu = nothing
            GC.gc()
            CUDA.reclaim()
            
            final_memory = CUDA.available_memory()
            @test final_memory > initial_memory - 100_000_000
        else
            @test_skip "GPU tests require CUDA"
        end
    end
    
    @testset "Error Handling" begin
        @info "Testing error handling"
        
        # Test with invalid data
        @test_throws ArgumentError run_pipeline(zeros(10, 0), [1,2,3])  # No features
        @test_throws DimensionMismatch run_pipeline(randn(10, 5), [1,2])  # Wrong y size
        
        # Test with edge cases
        X_single = randn(100, 1)  # Single feature
        y_single = rand([0, 1], 100)
        
        results = run_pipeline(X_single, y_single)
        @test length(results.selected_indices) == 1
        
        # Test with constant features
        X_const = ones(100, 10)
        X_const[:, 1:2] = randn(100, 2)  # Only 2 variable features
        y_const = rand([0, 1], 100)
        
        results = run_pipeline(X_const, y_const)
        @test length(results.selected_indices) <= 2
    end
    
    @testset "Performance Benchmarks" begin
        @info "Running performance benchmarks"
        
        benchmark_results = DataFrame(
            size = String[],
            time_ms = Float64[],
            features_selected = Int[]
        )
        
        for (name, n_samples, n_features) in [
            ("Small", 100, 50),
            ("Medium", 1000, 500),
            ("Large", 5000, 1000)
        ]
            X, y, _ = generate_test_data(n_samples, n_features, 20)
            
            time_ms = @elapsed(run_pipeline(X, y)) * 1000
            results = run_pipeline(X, y)
            
            push!(benchmark_results, (
                size = name,
                time_ms = time_ms,
                features_selected = length(results.selected_indices)
            ))
            
            @info "Benchmark $name: $(round(time_ms, digits=1)) ms"
        end
        
        # Check performance scaling
        @test all(benchmark_results.time_ms .< 60_000)  # All under 60 seconds
        @test issorted(benchmark_results.time_ms)  # Time increases with size
    end
end

# Helper functions for the mock pipeline
function stage1_filter(X, y; variance_threshold=0.01)
    variances = vec(var(X, dims=1))
    return findall(v -> v > variance_threshold, variances)
end

function stage2_mcts(X, y, initial_indices; target_features=10)
    n_features = length(initial_indices)
    correlations = [abs(cor(X[:, idx], Float64.(y))) for idx in initial_indices]
    n_selected = min(target_features, n_features)
    selected_local = partialsortperm(correlations, 1:n_selected, rev=true)
    return initial_indices[selected_local]
end

function stage3_ensemble(X, y, initial_indices; min_features=3, max_features=8)
    n_features = length(initial_indices)
    X_subset = X[:, initial_indices]
    importance_scores = [abs(cor(X_subset[:, i], Float64.(y))) + 0.1 * rand() for i in 1:n_features]
    n_final = clamp(round(Int, n_features * 0.5), min_features, max_features)
    selected_local = partialsortperm(importance_scores, 1:n_final, rev=true)
    return initial_indices[selected_local], importance_scores[selected_local]
end

function run_pipeline(X, y)
    size(X, 2) > 0 || throw(ArgumentError("X must have at least one feature"))
    size(X, 1) == length(y) || throw(DimensionMismatch("X and y must have same number of samples"))
    
    stage1_indices = stage1_filter(X, y, variance_threshold=0.001)
    isempty(stage1_indices) && return (selected_indices=Int[], feature_scores=Float64[], stage1_count=0, stage2_count=0, stage3_count=0)
    
    X_stage1 = X[:, stage1_indices]
    stage2_indices_local = stage2_mcts(X_stage1, y, 1:length(stage1_indices), target_features=50)
    stage2_indices = stage1_indices[stage2_indices_local]
    
    X_stage2 = X[:, stage2_indices]
    stage3_indices_local, scores = stage3_ensemble(X_stage2, y, 1:length(stage2_indices))
    final_indices = stage2_indices[stage3_indices_local]
    
    return (
        selected_indices = final_indices,
        feature_scores = scores,
        stage1_count = length(stage1_indices),
        stage2_count = length(stage2_indices),
        stage3_count = length(final_indices)
    )
end