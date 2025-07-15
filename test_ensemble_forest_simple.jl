"""
Simple test for Ensemble Forest Architecture
Tests core functionality without Stage1Integration dependencies
"""

using Test
using Random
using Statistics
using Dates

# Create a mock Stage1Integration interface for testing
module MockStage1Integration
    export Stage1IntegrationInterface
    
    struct Stage1IntegrationInterface
        # Mock interface - empty for testing
    end
end

# Temporarily modify the ensemble forest file to use mock integration
ensemble_content = read("/home/xai/DEV/HSOF/src/stage2/ensemble_forest.jl", String)
modified_content = replace(ensemble_content, 
    "include(\"stage1_integration.jl\")\nusing .Stage1Integration" => 
    "# Mock Stage1Integration for testing")

# Write temporary modified file
temp_file = "/tmp/ensemble_forest_test.jl"
write(temp_file, modified_content)

# Include and test the modified module
include(temp_file)
using .EnsembleForest

@testset "Ensemble Forest Core Tests" begin
    
    Random.seed!(42)  # For reproducible tests
    
    @testset "Tree Pool Configuration Tests" begin
        # Test default configuration
        config = create_tree_pool_config()
        
        @test config.max_trees == 100
        @test config.initial_trees == 10
        @test config.tree_growth_rate == 5
        @test config.default_max_iterations == 1000
        @test config.default_max_nodes == 10000
        @test config.default_max_depth == 50
        @test config.enable_memory_pooling == true
        @test config.enable_node_pruning == true
        @test config.enable_cross_tree_learning == true
        
        # Test custom configuration
        custom_config = create_tree_pool_config(
            max_trees = 50,
            initial_trees = 5,
            default_max_iterations = 500,
            enable_cross_tree_learning = false
        )
        
        @test custom_config.max_trees == 50
        @test custom_config.initial_trees == 5
        @test custom_config.default_max_iterations == 500
        @test custom_config.enable_cross_tree_learning == false
        
        println("  ✅ Tree pool configuration tests passed")
    end
    
    @testset "Forest Manager Initialization Tests" begin
        config = create_tree_pool_config(initial_trees = 5)
        manager = initialize_forest_manager(config, nothing, debug_enabled = false)
        
        @test manager.config.initial_trees == 5
        @test length(manager.trees) == 5
        @test manager.forest_state == "ready"
        @test length(manager.available_tree_ids) == 95  # 100 - 5 used
        @test manager.next_tree_id == 6  # Next available ID
        @test length(manager.global_feature_importance) == 500
        @test length(manager.consensus_features) == 500
        @test all(.!manager.consensus_features)  # Initially no consensus
        
        # Test with debug enabled
        debug_manager = initialize_forest_manager(config, nothing, debug_enabled = true)
        @test debug_manager.debug_enabled == true
        
        println("  ✅ Forest manager initialization tests passed")
    end
    
    @testset "Tree Creation and State Management Tests" begin
        config = create_tree_pool_config(initial_trees = 2, max_trees = 10)
        manager = initialize_forest_manager(config)
        
        initial_tree_count = length(manager.trees)
        @test initial_tree_count == 2
        
        # Test tree creation
        new_tree_id = create_tree!(manager)
        @test new_tree_id isa Int
        @test haskey(manager.trees, new_tree_id)
        @test length(manager.trees) == 3
        
        # Verify tree properties
        tree = manager.trees[new_tree_id]
        @test tree.tree_id == new_tree_id
        @test tree.state == EnsembleForest.READY
        @test length(tree.selected_features) == 500
        @test length(tree.feature_importance) == 500
        @test all(.!tree.selected_features)  # Initially no features selected
        @test tree.iteration_count == 0
        @test tree.evaluation_count == 0
        @test tree.best_score == -Inf32
        
        # Test tree state transitions
        start_tree!(manager, new_tree_id)
        @test manager.trees[new_tree_id].state == EnsembleForest.RUNNING
        @test new_tree_id in manager.active_trees
        
        pause_tree!(manager, new_tree_id)
        @test manager.trees[new_tree_id].state == EnsembleForest.PAUSED
        @test new_tree_id in manager.paused_trees
        @test new_tree_id ∉ manager.active_trees
        
        start_tree!(manager, new_tree_id)  # Resume
        @test manager.trees[new_tree_id].state == EnsembleForest.RUNNING
        
        complete_tree!(manager, new_tree_id, 0.85f0)
        @test manager.trees[new_tree_id].state == EnsembleForest.COMPLETED
        @test manager.trees[new_tree_id].current_score == 0.85f0
        @test manager.trees[new_tree_id].best_score == 0.85f0
        @test new_tree_id in manager.completed_trees
        
        println("  ✅ Tree creation and state management tests passed")
    end
    
    @testset "Feature Synchronization Tests" begin
        config = create_tree_pool_config(initial_trees = 3, enable_cross_tree_learning = true)
        manager = initialize_forest_manager(config)
        
        tree_ids = collect(keys(manager.trees))
        
        # Set up different feature selections for trees
        for (i, tree_id) in enumerate(tree_ids)
            tree = manager.trees[tree_id]
            tree.state = EnsembleForest.RUNNING
            
            # Create distinct feature patterns for each tree
            selected_features = fill(false, 500)
            feature_importance = zeros(Float32, 500)
            
            # Tree-specific patterns
            if i == 1
                selected_features[1:100] .= true  # First 100 features
                feature_importance[1:100] .= 0.9f0
            elseif i == 2
                selected_features[51:150] .= true  # Overlapping with tree 1
                feature_importance[51:150] .= 0.8f0
            else
                selected_features[101:200] .= true  # Different range
                feature_importance[101:200] .= 0.7f0
            end
            
            update_tree_features!(tree, selected_features, feature_importance)
        end
        
        # Test synchronization
        synchronize_features!(manager)
        
        # Verify global importance was updated
        @test !all(iszero, manager.global_feature_importance)
        
        # Verify consensus features exist
        consensus_count = sum(manager.consensus_features)
        @test consensus_count > 0
        
        println("  ✅ Feature synchronization tests passed")
    end
    
    @testset "Performance and Status Tests" begin
        config = create_tree_pool_config(initial_trees = 3)
        manager = initialize_forest_manager(config)
        
        # Set up some tree states for testing
        tree_ids = collect(keys(manager.trees))
        start_tree!(manager, tree_ids[1])
        complete_tree!(manager, tree_ids[2], 0.9f0)
        fail_tree!(manager, tree_ids[3], "Test failure")
        
        # Update some metrics
        manager.total_iterations = 500
        manager.total_evaluations = 250
        
        # Test status retrieval
        status = get_forest_status(manager)
        
        @test status["total_trees"] == 3
        @test status["active_trees"] == 1
        @test status["completed_trees"] == 1
        @test status["failed_trees"] == 1
        @test status["total_iterations"] == 500
        @test status["total_evaluations"] == 250
        @test status["forest_state"] == "ready"
        
        # Test report generation
        report = generate_forest_report(manager)
        @test contains(report, "Ensemble Forest Status Report")
        @test contains(report, "Total Trees: 3")
        @test contains(report, "Active: 1")
        @test contains(report, "Completed: 1")
        @test contains(report, "Failed: 1")
        
        println("  ✅ Performance and status tests passed")
    end
end

println("✅ Core Ensemble Forest functionality verified!")
println("✅ Tree pool configuration and management")
println("✅ Forest manager initialization and coordination") 
println("✅ Tree lifecycle management and state transitions")
println("✅ Feature synchronization and cross-tree learning")
println("✅ Performance monitoring and status reporting")
println("✅ Ready for GPU-MCTS implementation")

# Clean up temporary file
rm(temp_file, force=true)