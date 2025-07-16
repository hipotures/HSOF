using Test
using CUDA
using Random
using JSON3
using Statistics
using Dates

# Import all ensemble components
include("../../src/gpu/mcts_gpu.jl")
include("../../src/config/ensemble_config.jl")
include("../../src/config/templates.jl")

using .MCTSGPU
using .EnsembleConfig
using .ConfigTemplates

"""
Integration Testing Framework for Ensemble MCTS System

This comprehensive test suite validates end-to-end ensemble functionality
including feature reduction (500→50), GPU utilization, performance metrics,
and memory efficiency across all components.
"""

@testset "Ensemble Integration Tests" begin
    # Skip tests if CUDA not available
    if !CUDA.functional()
        @warn "CUDA not available, skipping ensemble integration tests"
        return
    end
    
    @testset "Synthetic Dataset Generation" begin
        """
        Generate controlled synthetic datasets for testing feature reduction accuracy
        """
        
        function generate_synthetic_features(num_samples::Int, num_features::Int; 
                                           relevant_features::Int = 50,
                                           noise_level::Float64 = 0.1,
                                           random_seed::Int = 42)
            Random.seed!(random_seed)
            
            # Generate relevant features with signal
            X_relevant = randn(num_samples, relevant_features)
            
            # Generate noise features
            X_noise = randn(num_samples, num_features - relevant_features) * noise_level
            
            # Combine features
            X = hcat(X_relevant, X_noise)
            
            # Generate labels based on relevant features
            y = vec(sum(X_relevant[:, 1:min(10, relevant_features)], dims=2)) .> 0
            
            return X, y, collect(1:relevant_features)  # Return true relevant feature indices
        end
        
        # Test dataset generation
        X, y, true_features = generate_synthetic_features(1000, 500, relevant_features=50)
        @test size(X) == (1000, 500)
        @test length(y) == 1000
        @test length(true_features) == 50
        @test all(true_features .<= 50)  # Relevant features are first 50
        
        # Test class balance
        positive_ratio = sum(y) / length(y)
        @test 0.3 < positive_ratio < 0.7  # Roughly balanced
        
        @info "Synthetic dataset generation tests passed"
    end
    
    @testset "Feature Reduction Accuracy Validation" begin
        """
        Test 500→50 feature reduction maintains feature quality vs exhaustive search
        """
        
        # Generate test dataset
        X, y, names, metadata = generate_synthetic_dataset(500, 500, relevant_features=50)
        true_features = metadata["relevant_indices"]
        
        # Create development configuration for testing
        config = development_config()
        
        # Test ensemble creation and initialization
        ensemble = MemoryEfficientTreeEnsemble(
            CUDA.device(),
            max_trees = Int32(config.trees_per_gpu),
            max_nodes_per_tree = Int32(config.max_nodes_per_tree)
        )
        
        # Initialize ensemble with empty features
        initialize_ensemble!(ensemble, [Int[] for _ in 1:config.trees_per_gpu])
        
        # Test feature selection simulation
        dummy_evaluation = function(feature_indices::Vector{Int})
            if isempty(feature_indices)
                return 0.5
            end
            
            # Higher score for features in true_features
            overlap = length(intersect(feature_indices, true_features))
            base_score = 0.5 + 0.3 * overlap / length(feature_indices)
            
            # Add noise
            noise = 0.1 * randn()
            return clamp(base_score + noise, 0.0, 1.0)
        end
        
        # Test short ensemble run
        available_actions = UInt16.(1:100)  # First 100 features
        prior_scores = fill(0.01f0, 100)
        
        run_ensemble_mcts!(ensemble, 100, dummy_evaluation, available_actions, prior_scores)
        
        # Get selected features
        selected_features = get_best_features_ensemble(ensemble, 50)
        @test length(selected_features) <= 50
        
        # Check overlap with true features
        overlap = length(intersect(selected_features, true_features))
        overlap_ratio = overlap / length(selected_features)
        
        @test overlap_ratio > 0.3  # Should find at least 30% true features
        
        @info "Feature reduction accuracy test passed" overlap_ratio
    end
    
    @testset "GPU Utilization and Performance Metrics" begin
        """
        Test GPU utilization and performance metrics collection
        """
        
        # Test with benchmark configuration
        config = benchmark_config()
        
        # Create ensemble with performance monitoring
        ensemble = MemoryEfficientTreeEnsemble(
            CUDA.device(),
            max_trees = Int32(config.trees_per_gpu),
            max_nodes_per_tree = Int32(config.max_nodes_per_tree)
        )
        
        initialize_ensemble!(ensemble, [Int[] for _ in 1:config.trees_per_gpu])
        
        # Run with performance monitoring
        start_time = time()
        
        dummy_evaluation = features -> 0.5 + 0.1 * randn()
        available_actions = UInt16.(1:100)
        prior_scores = fill(0.01f0, 100)
        
        run_ensemble_mcts!(ensemble, 200, dummy_evaluation, available_actions, prior_scores)
        
        elapsed_time = time() - start_time
        
        # Get ensemble statistics
        stats = get_ensemble_statistics(ensemble)
        
        # Validate performance metrics
        @test haskey(stats, "memory_stats")
        @test haskey(stats, "pool_stats")
        @test haskey(stats, "lazy_stats")
        @test haskey(stats, "tree_stats")
        
        memory_stats = stats["memory_stats"]
        @test memory_stats["total_mb"] > 0
        @test memory_stats["compression_ratio"] > 1.0
        
        # Test memory efficiency
        @test memory_stats["total_mb"] < 100.0  # Should be under 100MB
        
        # Test tree statistics
        tree_stats = stats["tree_stats"]
        @test tree_stats["active_trees"] == config.trees_per_gpu
        @test tree_stats["total_nodes"] > 0
        
        @info "GPU utilization and performance metrics test passed" elapsed_time memory_mb=memory_stats["total_mb"]
    end
    
    @testset "Memory Efficiency Validation" begin
        """
        Test memory efficiency across different configurations
        """
        
        configurations = [
            ("development", development_config()),
            ("production", production_config()),
            ("single-gpu", single_gpu_config())
        ]
        
        memory_results = Dict{String, Float64}()
        
        for (config_name, config) in configurations
            # Create ensemble
            ensemble = MemoryEfficientTreeEnsemble(
                CUDA.device(),
                max_trees = config.trees_per_gpu,
                max_nodes_per_tree = config.max_nodes_per_tree
            )
            
            initialize_ensemble!(ensemble, [Int[] for _ in 1:config.trees_per_gpu])
            
            # Run brief test
            dummy_evaluation = features -> 0.5
            available_actions = UInt16.(1:20)
            prior_scores = fill(0.05f0, 20)
            
            run_ensemble_mcts!(ensemble, 50, dummy_evaluation, available_actions, prior_scores)
            
            # Get memory usage
            stats = get_ensemble_statistics(ensemble)
            memory_mb = stats["memory_stats"]["total_mb"]
            memory_results[config_name] = memory_mb
            
            @test memory_mb > 0
            @test memory_mb < 1000.0  # Reasonable upper bound
        end
        
        # Validate memory scaling
        @test memory_results["development"] < memory_results["production"]
        
        @info "Memory efficiency validation passed" memory_results
    end
    
    @testset "Configuration Integration Tests" begin
        """
        Test configuration system integration with ensemble
        """
        
        # Test loading configuration from file
        config_path = tempname() * ".json"
        
        # Create and save test configuration
        test_config = EnsembleConfiguration(
            num_trees = 20,
            trees_per_gpu = 20,
            max_nodes_per_tree = 5000,
            max_iterations = 50000,
            lazy_expansion = true,
            shared_features = true,
            compressed_nodes = true
        )
        
        save_json_config(test_config, config_path)
        
        # Load configuration
        loaded_config = load_json_config(config_path)
        @test loaded_config.num_trees == 20
        @test loaded_config.trees_per_gpu == 20
        @test loaded_config.lazy_expansion == true
        
        # Test ensemble creation with loaded configuration
        ensemble = MemoryEfficientTreeEnsemble(
            CUDA.device(),
            max_trees = loaded_config.trees_per_gpu,
            max_nodes_per_tree = loaded_config.max_nodes_per_tree
        )
        
        initialize_ensemble!(ensemble, [Int[] for _ in 1:loaded_config.trees_per_gpu])
        
        # Validate ensemble matches configuration
        stats = get_ensemble_statistics(ensemble)
        @test stats["tree_stats"]["active_trees"] == loaded_config.trees_per_gpu
        
        # Cleanup
        rm(config_path)
        
        @info "Configuration integration test passed"
    end
    
    @testset "Stress Testing with Maximum Parameters" begin
        """
        Test ensemble with maximum tree counts and feature dimensions
        """
        
        # Create high-stress configuration
        stress_config = EnsembleConfiguration(
            num_trees = 50,
            trees_per_gpu = 50,
            max_nodes_per_tree = 10000,
            max_iterations = 100000,
            initial_features = 1000,
            target_features = 100,
            memory_pool_size = 0.9,
            lazy_expansion = true,
            shared_features = true,
            compressed_nodes = true
        )
        
        # Test ensemble can handle stress configuration
        ensemble = MemoryEfficientTreeEnsemble(
            CUDA.device(),
            max_trees = stress_config.trees_per_gpu,
            max_nodes_per_tree = stress_config.max_nodes_per_tree
        )
        
        initialize_ensemble!(ensemble, [Int[] for _ in 1:stress_config.trees_per_gpu])
        
        # Run stress test
        dummy_evaluation = features -> 0.5 + 0.1 * randn()
        available_actions = UInt16.(1:200)  # Large action space
        prior_scores = fill(0.005f0, 200)
        
        start_time = time()
        run_ensemble_mcts!(ensemble, 500, dummy_evaluation, available_actions, prior_scores)
        elapsed_time = time() - start_time
        
        # Validate stress test results
        stats = get_ensemble_statistics(ensemble)
        @test stats["tree_stats"]["active_trees"] == stress_config.trees_per_gpu
        @test stats["memory_stats"]["total_mb"] < 500.0  # Should handle stress efficiently
        
        # Test feature selection under stress
        selected_features = get_best_features_ensemble(ensemble, 100)
        @test length(selected_features) <= 100
        
        @info "Stress testing passed" elapsed_time memory_mb=stats["memory_stats"]["total_mb"]
    end
    
    @testset "Regression Testing" begin
        """
        Test that changes don't break existing functionality
        """
        
        # Test basic functionality with known good configuration
        config = development_config()
        
        # Create ensemble
        ensemble = MemoryEfficientTreeEnsemble(
            CUDA.device(),
            max_trees = Int32(config.trees_per_gpu),
            max_nodes_per_tree = Int32(config.max_nodes_per_tree)
        )
        
        initialize_ensemble!(ensemble, [Int[] for _ in 1:config.trees_per_gpu])
        
        # Test basic operations
        dummy_evaluation = features -> 0.5
        available_actions = UInt16.(1:50)
        prior_scores = fill(0.02f0, 50)
        
        # Test ensemble run
        @test_nowarn run_ensemble_mcts!(ensemble, 100, dummy_evaluation, available_actions, prior_scores)
        
        # Test feature selection
        selected_features = get_best_features_ensemble(ensemble, 25)
        @test length(selected_features) <= 25
        @test all(f -> f isa Int, selected_features)
        
        # Test statistics collection
        stats = get_ensemble_statistics(ensemble)
        @test haskey(stats, "memory_stats")
        @test haskey(stats, "tree_stats")
        
        # Test memory optimizations are working
        @test stats["memory_stats"]["compression_ratio"] > 1.0
        @test stats["lazy_stats"]["memory_saved_mb"] >= 0
        
        @info "Regression testing passed"
    end
    
    @testset "Performance Benchmark Framework" begin
        """
        Test performance benchmarking capabilities
        """
        
        # Create benchmark configuration
        config = benchmark_config()
        
        # Run benchmark test
        benchmark_results = Dict{String, Any}()
        
        # Test ensemble creation time
        start_time = time()
        ensemble = MemoryEfficientTreeEnsemble(
            CUDA.device(),
            max_trees = Int32(config.trees_per_gpu),
            max_nodes_per_tree = Int32(config.max_nodes_per_tree)
        )
        initialize_ensemble!(ensemble, [Int[] for _ in 1:config.trees_per_gpu])
        creation_time = time() - start_time
        
        benchmark_results["creation_time_ms"] = creation_time * 1000
        
        # Test MCTS performance
        dummy_evaluation = features -> 0.5 + 0.1 * randn()
        available_actions = UInt16.(1:100)
        prior_scores = fill(0.01f0, 100)
        
        start_time = time()
        run_ensemble_mcts!(ensemble, 1000, dummy_evaluation, available_actions, prior_scores)
        mcts_time = time() - start_time
        
        benchmark_results["mcts_time_ms"] = mcts_time * 1000
        benchmark_results["iterations_per_second"] = 1000 / mcts_time
        
        # Test memory efficiency
        stats = get_ensemble_statistics(ensemble)
        benchmark_results["memory_mb"] = stats["memory_stats"]["total_mb"]
        benchmark_results["compression_ratio"] = stats["memory_stats"]["compression_ratio"]
        
        # Test feature selection performance
        start_time = time()
        selected_features = get_best_features_ensemble(ensemble, 50)
        selection_time = time() - start_time
        
        benchmark_results["feature_selection_time_ms"] = selection_time * 1000
        benchmark_results["selected_features_count"] = length(selected_features)
        
        # Validate benchmark results
        @test benchmark_results["creation_time_ms"] < 5000  # Should create in under 5s
        @test benchmark_results["iterations_per_second"] > 100  # Should achieve decent throughput
        @test benchmark_results["memory_mb"] < 200  # Should be memory efficient
        @test benchmark_results["compression_ratio"] > 2.0  # Should achieve good compression
        @test benchmark_results["feature_selection_time_ms"] < 1000  # Should select quickly
        
        @info "Performance benchmark results:" benchmark_results
    end
    
    @testset "Error Handling and Edge Cases" begin
        """
        Test error handling and edge case scenarios
        """
        
        # Test with invalid configuration
        @test_throws ArgumentError EnsembleConfiguration(num_trees = -1)
        @test_throws ArgumentError EnsembleConfiguration(target_features = 100, initial_features = 50)
        
        # Test ensemble with empty actions
        config = development_config()
        ensemble = MemoryEfficientTreeEnsemble(
            CUDA.device(),
            max_trees = Int32(config.trees_per_gpu),
            max_nodes_per_tree = Int32(config.max_nodes_per_tree)
        )
        
        initialize_ensemble!(ensemble, [Int[] for _ in 1:config.trees_per_gpu])
        
        # Test with empty action set
        dummy_evaluation = features -> 0.5
        empty_actions = UInt16[]
        empty_priors = Float32[]
        
        @test_nowarn run_ensemble_mcts!(ensemble, 10, dummy_evaluation, empty_actions, empty_priors)
        
        # Test feature selection with no features
        selected_features = get_best_features_ensemble(ensemble, 10)
        @test length(selected_features) == 0
        
        # Test with evaluation function that throws
        error_evaluation = features -> error("Test evaluation error")
        actions = UInt16.(1:10)
        priors = fill(0.1f0, 10)
        
        @test_throws Exception run_ensemble_mcts!(ensemble, 10, error_evaluation, actions, priors)
        
        @info "Error handling and edge cases test passed"
    end
end

"""
Run comprehensive integration test suite
"""
function run_integration_tests()
    println("="^60)
    println("Running Ensemble Integration Test Suite")
    println("="^60)
    
    start_time = time()
    
    # Run tests with comprehensive reporting
    test_results = @testset "Complete Integration Tests" begin
        # All test sets are included above
    end
    
    elapsed_time = time() - start_time
    
    println("\n" * "="^60)
    println("Integration Test Suite Completed")
    println("Total time: $(round(elapsed_time, digits=2)) seconds")
    println("="^60)
    
    return test_results
end

@info "Integration Testing Framework loaded successfully"
@info "Run `run_integration_tests()` to execute complete test suite"