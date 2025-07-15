"""
Test Suite for Diversity Mechanisms Implementation
Validates diversity-promoting mechanisms for MCTS ensemble forest including feature subsampling,
exploration variation, initial randomization, and feature masking diversity.
"""

using Test
using Random
using Statistics
using Dates

# Include the diversity mechanisms module
include("../../src/stage2/diversity_mechanisms.jl")
using .DiversityMechanisms

# Include ensemble forest for integration testing
include("../../src/stage2/ensemble_forest.jl")
using .EnsembleForest

@testset "Diversity Mechanisms Tests" begin
    
    Random.seed!(42)  # For reproducible tests
    
    @testset "Configuration Tests" begin
        # Test default configuration
        config = create_diversity_config()
        
        @test config.feature_subsample_rate == 0.8f0
        @test config.min_features_per_tree == 100
        @test config.enable_dynamic_subsampling == true
        @test config.exploration_constant_min == 0.5f0
        @test config.exploration_constant_max == 2.0f0
        @test config.exploration_strategy == "uniform"
        @test config.enable_initial_randomization == true
        @test config.initial_selection_rate == 0.1f0
        @test config.enable_feature_masking == true
        @test config.max_overlap_threshold == 0.7f0
        @test config.target_diversity_score == 0.6f0
        @test config.enable_adaptive_diversity == true
        
        # Test custom configuration
        custom_config = create_diversity_config(
            feature_subsample_rate = 0.6f0,
            min_features_per_tree = 50,
            exploration_constant_min = 1.0f0,
            exploration_constant_max = 3.0f0,
            exploration_strategy = "normal",
            enable_initial_randomization = false,
            max_overlap_threshold = 0.5f0
        )
        
        @test custom_config.feature_subsample_rate == 0.6f0
        @test custom_config.min_features_per_tree == 50
        @test custom_config.exploration_constant_min == 1.0f0
        @test custom_config.exploration_constant_max == 3.0f0
        @test custom_config.exploration_strategy == "normal"
        @test custom_config.enable_initial_randomization == false
        @test custom_config.max_overlap_threshold == 0.5f0
        
        println("  ✅ Configuration tests passed")
    end
    
    @testset "Diversity Strategy Enum Tests" begin
        # Test enum values
        @test Int(RANDOM_SUBSAMPLING) == 1
        @test Int(EXPLORATION_VARIATION) == 2
        @test Int(INITIAL_RANDOMIZATION) == 3
        @test Int(MASKING_DIVERSITY) == 4
        @test Int(COMBINED_DIVERSITY) == 5
        
        # Test enum operations
        @test RANDOM_SUBSAMPLING != EXPLORATION_VARIATION
        @test MASKING_DIVERSITY != COMBINED_DIVERSITY
        
        println("  ✅ Diversity strategy enum tests passed")
    end
    
    @testset "Diversity Metrics Initialization Tests" begin
        n_trees = 10
        n_features = 500
        
        metrics = DiversityMechanisms.initialize_diversity_metrics(n_trees, n_features)
        
        @test size(metrics.pairwise_overlap_matrix) == (n_trees, n_trees)
        @test metrics.average_overlap == 0.0f0
        @test metrics.max_overlap == 0.0f0
        @test metrics.min_overlap == 0.0f0
        @test metrics.feature_selection_entropy == 0.0f0
        @test metrics.unique_feature_combinations == 0
        @test length(metrics.feature_usage_distribution) == n_features
        @test all(metrics.feature_usage_distribution .== 0)
        @test metrics.exploration_constant_variance == 0.0f0
        @test metrics.score_diversity == 0.0f0
        @test metrics.overall_diversity_score == 0.0f0
        @test isempty(metrics.diversity_trend)
        @test metrics.update_count == 0
        @test metrics.computation_time == 0.0
        
        println("  ✅ Diversity metrics initialization tests passed")
    end
    
    @testset "Diversity Manager Initialization Tests" begin
        config = create_diversity_config()
        n_trees = 20
        n_features = 500
        
        manager = initialize_diversity_manager(config, n_trees, n_features)
        
        @test manager.config == config
        @test manager.n_trees == n_trees
        @test manager.n_features == n_features
        @test length(manager.tree_feature_masks) == n_trees
        @test length(manager.tree_exploration_constants) == n_trees
        @test length(manager.tree_random_seeds) == n_trees
        @test manager.manager_state == "active"
        @test manager.current_subsample_rate == config.feature_subsample_rate
        @test manager.adaptation_iterations == 0
        @test isempty(manager.error_log)
        
        # Test that all trees have feature masks
        for tree_id in 1:n_trees
            @test haskey(manager.tree_feature_masks, tree_id)
            mask = manager.tree_feature_masks[tree_id]
            @test length(mask) == n_features
            @test sum(mask) >= config.min_features_per_tree
        end
        
        # Test that all trees have exploration constants
        for tree_id in 1:n_trees
            @test haskey(manager.tree_exploration_constants, tree_id)
            constant = manager.tree_exploration_constants[tree_id]
            @test config.exploration_constant_min <= constant <= config.exploration_constant_max
        end
        
        # Test that all trees have unique random seeds
        seeds = collect(values(manager.tree_random_seeds))
        @test length(unique(seeds)) == n_trees  # All seeds should be unique
        
        println("  ✅ Diversity manager initialization tests passed")
    end
    
    @testset "Feature Subsampling Tests" begin
        config = create_diversity_config(
            feature_subsample_rate = 0.6f0,
            min_features_per_tree = 100
        )
        n_trees = 10
        n_features = 500
        
        manager = initialize_diversity_manager(config, n_trees, n_features)
        
        # Test feature counts
        expected_features = max(100, round(Int, 500 * 0.6))  # 300 features
        
        for tree_id in 1:n_trees
            mask = manager.tree_feature_masks[tree_id]
            n_selected = sum(mask)
            @test n_selected == expected_features
        end
        
        # Test diversity in feature selection
        all_masks = [manager.tree_feature_masks[i] for i in 1:n_trees]
        
        # Calculate pairwise overlaps
        overlaps = Float64[]
        for i in 1:n_trees
            for j in (i+1):n_trees
                mask_i = all_masks[i]
                mask_j = all_masks[j]
                intersection = sum(mask_i .& mask_j)
                union_size = sum(mask_i .| mask_j)
                overlap = union_size > 0 ? intersection / union_size : 0.0
                push!(overlaps, overlap)
            end
        end
        
        # Should have some diversity (not all trees identical)
        @test std(overlaps) > 0.0
        @test mean(overlaps) < 1.0  # Not all identical
        
        println("  ✅ Feature subsampling tests passed")
    end
    
    @testset "Exploration Constant Variation Tests" begin
        # Test uniform distribution
        config_uniform = create_diversity_config(
            exploration_constant_min = 0.5f0,
            exploration_constant_max = 2.0f0,
            exploration_strategy = "uniform"
        )
        n_trees = 50
        
        manager_uniform = initialize_diversity_manager(config_uniform, n_trees)
        constants_uniform = collect(values(manager_uniform.tree_exploration_constants))
        
        @test all(0.5f0 .<= constants_uniform .<= 2.0f0)
        @test length(unique(constants_uniform)) > 1  # Should have variation
        
        # Test normal distribution
        config_normal = create_diversity_config(
            exploration_constant_min = 0.5f0,
            exploration_constant_max = 2.0f0,
            exploration_strategy = "normal"
        )
        
        manager_normal = initialize_diversity_manager(config_normal, n_trees)
        constants_normal = collect(values(manager_normal.tree_exploration_constants))
        
        @test all(0.5f0 .<= constants_normal .<= 2.0f0)
        @test length(unique(constants_normal)) > 1
        
        # Test adaptive distribution
        config_adaptive = create_diversity_config(
            exploration_constant_min = 0.5f0,
            exploration_constant_max = 2.0f0,
            exploration_strategy = "adaptive"
        )
        
        manager_adaptive = initialize_diversity_manager(config_adaptive, n_trees)
        constants_adaptive = collect(values(manager_adaptive.tree_exploration_constants))
        
        @test all(0.5f0 .<= constants_adaptive .<= 2.0f0)
        @test constants_adaptive[1] ≈ 0.5f0  # First tree should have minimum
        @test constants_adaptive[end] ≈ 2.0f0  # Last tree should have maximum
        
        println("  ✅ Exploration constant variation tests passed")
    end
    
    @testset "Diversity Metrics Calculation Tests" begin
        config = create_diversity_config()
        n_trees = 5
        n_features = 100
        
        manager = initialize_diversity_manager(config, n_trees, n_features)
        
        # Test initial metrics calculation
        DiversityMechanisms.calculate_diversity_metrics!(manager)
        
        metrics = manager.diversity_metrics
        
        # Test pairwise overlap matrix
        @test size(metrics.pairwise_overlap_matrix) == (n_trees, n_trees)
        @test all(diag(metrics.pairwise_overlap_matrix) .== 1.0f0)  # Self-overlap should be 1
        
        # Test that metrics are calculated
        @test metrics.average_overlap >= 0.0f0
        @test metrics.max_overlap >= metrics.average_overlap
        @test metrics.min_overlap <= metrics.average_overlap
        @test metrics.overall_diversity_score >= 0.0f0
        @test metrics.overall_diversity_score <= 1.0f0
        @test metrics.update_count == 1
        @test metrics.computation_time > 0.0
        
        # Test feature usage distribution
        @test length(metrics.feature_usage_distribution) == n_features
        @test sum(metrics.feature_usage_distribution) > 0  # Some features should be used
        
        # Test that diversity trend is updated
        @test length(metrics.diversity_trend) == 1
        @test metrics.diversity_trend[1] == metrics.overall_diversity_score
        
        println("  ✅ Diversity metrics calculation tests passed")
    end
    
    @testset "Adaptive Diversity Tests" begin
        config = create_diversity_config(
            enable_adaptive_diversity = true,
            target_diversity_score = 0.7f0,
            diversity_adaptation_rate = 0.1f0,
            diversity_check_frequency = 0  # Always check
        )
        n_trees = 10
        
        manager = initialize_diversity_manager(config, n_trees)
        
        # Create mock forest manager
        forest_config = create_tree_pool_config(initial_trees = n_trees)
        forest_manager = initialize_forest_manager(forest_config)
        
        # Record initial subsample rate
        initial_rate = manager.current_subsample_rate
        
        # Simulate low diversity scenario
        manager.diversity_metrics.overall_diversity_score = 0.3f0  # Below target
        
        # Update diversity mechanisms
        update_diversity_mechanisms!(manager, forest_manager)
        
        # Should have adapted to increase diversity
        @test manager.adaptation_iterations == 1
        
        # Test that we can get status
        status = get_diversity_status(manager)
        @test haskey(status, "overall_diversity_score")
        @test haskey(status, "adaptation_iterations")
        @test status["adaptation_iterations"] == 1
        
        println("  ✅ Adaptive diversity tests passed")
    end
    
    @testset "Forest Integration Tests" begin
        config = create_diversity_config()
        n_trees = 8
        
        # Create diversity manager
        manager = initialize_diversity_manager(config, n_trees)
        
        # Create forest manager
        forest_config = create_tree_pool_config(initial_trees = n_trees)
        forest_manager = initialize_forest_manager(forest_config)
        
        # Apply diversity to forest
        apply_diversity_to_forest!(manager, forest_manager)
        
        # Verify diversity configurations were applied
        for tree_id in 1:n_trees
            if haskey(forest_manager.trees, tree_id)
                tree = forest_manager.trees[tree_id]
                
                # Check exploration constant was set
                @test haskey(tree.config, "exploration_constant")
                exploration_constant = tree.config["exploration_constant"]
                @test config.exploration_constant_min <= exploration_constant <= config.exploration_constant_max
                
                # Check feature mask was set
                @test haskey(tree.metadata, "available_features")
                available_features = tree.metadata["available_features"]
                @test length(available_features) == 500  # Default n_features
                @test sum(available_features) >= config.min_features_per_tree
                
                # Check random seed was set
                @test haskey(tree.config, "random_seed")
                @test tree.config["random_seed"] isa Int
            end
        end
        
        # Cleanup
        cleanup_forest!(forest_manager)
        
        println("  ✅ Forest integration tests passed")
    end
    
    @testset "Status and Reporting Tests" begin
        config = create_diversity_config()
        n_trees = 6
        
        manager = initialize_diversity_manager(config, n_trees)
        
        # Test status retrieval
        status = get_diversity_status(manager)
        
        @test haskey(status, "manager_state")
        @test haskey(status, "n_trees")
        @test haskey(status, "n_features")
        @test haskey(status, "current_subsample_rate")
        @test haskey(status, "overall_diversity_score")
        @test haskey(status, "average_overlap")
        @test haskey(status, "feature_selection_entropy")
        @test haskey(status, "unique_combinations")
        
        @test status["manager_state"] == "active"
        @test status["n_trees"] == n_trees
        @test status["n_features"] == 500
        @test status["error_count"] == 0
        
        # Test report generation
        report = generate_diversity_report(manager)
        
        @test contains(report, "Ensemble Diversity Report")
        @test contains(report, "Manager State: active")
        @test contains(report, "Feature Subsample Rate:")
        @test contains(report, "Overall Score:")
        @test contains(report, "Average Overlap:")
        @test contains(report, "Unique Combinations:")
        @test contains(report, "Performance:")
        
        println("  ✅ Status and reporting tests passed")
    end
    
    @testset "Error Handling Tests" begin
        config = create_diversity_config()
        
        # Test with invalid tree count
        @test_throws BoundsError initialize_diversity_manager(config, 0)
        
        # Test with very small feature count
        manager_small = initialize_diversity_manager(config, 5, 10)
        @test manager_small.n_features == 10
        @test manager_small.manager_state == "active"
        
        # Test metrics calculation with small setup
        DiversityMechanisms.calculate_diversity_metrics!(manager_small)
        @test manager_small.diversity_metrics.update_count == 1
        
        println("  ✅ Error handling tests passed")
    end
    
    @testset "Weighted Sampling Tests" begin
        rng = MersenneTwister(42)
        items = [1, 2, 3, 4, 5]
        weights = [0.1, 0.2, 0.3, 0.3, 0.1]
        
        # Test sampling fewer items than available
        selected = DiversityMechanisms.weighted_sample_without_replacement(rng, items, weights, 3)
        @test length(selected) == 3
        @test all(item in items for item in selected)
        @test length(unique(selected)) == 3  # No duplicates
        
        # Test sampling all items
        selected_all = DiversityMechanisms.weighted_sample_without_replacement(rng, items, weights, 5)
        @test length(selected_all) == 5
        @test sort(selected_all) == sort(items)
        
        # Test sampling more items than available
        selected_more = DiversityMechanisms.weighted_sample_without_replacement(rng, items, weights, 10)
        @test length(selected_more) == 5  # Should return all available
        @test sort(selected_more) == sort(items)
        
        println("  ✅ Weighted sampling tests passed")
    end
end

println("All Diversity Mechanisms tests completed!")
println("✅ Configuration and strategy validation")
println("✅ Diversity metrics initialization and calculation")
println("✅ Diversity manager setup and lifecycle")
println("✅ Feature subsampling with configurable rates")
println("✅ Exploration constant variation strategies")
println("✅ Adaptive diversity mechanisms")
println("✅ Forest integration and configuration application")
println("✅ Status reporting and comprehensive monitoring")
println("✅ Error handling and edge cases")
println("✅ Ready for ensemble MCTS diversity implementation")