"""
Test Suite for Ensemble Forest Architecture
Validates tree management, resource allocation, synchronization, and performance
"""

using Test
using Random
using Statistics
using Dates

# Include the ensemble forest module
include("../../src/stage2/ensemble_forest.jl")
using .EnsembleForest

@testset "Ensemble Forest Tests" begin
    
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
    
    @testset "Tree Creation and Management Tests" begin
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
        
        # Test tree creation with custom parameters
        custom_tree_id = create_tree!(manager, 
                                     max_iterations = 2000,
                                     max_nodes = 5000,
                                     priority = 0.8f0)
        custom_tree = manager.trees[custom_tree_id]
        @test custom_tree.max_iterations == 2000
        @test custom_tree.max_nodes == 5000
        @test custom_tree.priority == 0.8f0
        
        # Test capacity limit
        for i in 1:(config.max_trees - length(manager.trees))
            create_tree!(manager)
        end
        @test length(manager.trees) == config.max_trees
        @test_throws ErrorException create_tree!(manager)  # Should fail at capacity
        
        println("  ✅ Tree creation and management tests passed")
    end
    
    @testset "Tree State Management Tests" begin
        config = create_tree_pool_config(initial_trees = 3)
        manager = initialize_forest_manager(config)
        
        tree_ids = collect(keys(manager.trees))
        test_tree_id = tree_ids[1]
        
        # Test starting tree
        @test manager.trees[test_tree_id].state == EnsembleForest.READY
        start_tree!(manager, test_tree_id)
        @test manager.trees[test_tree_id].state == EnsembleForest.RUNNING
        @test test_tree_id in manager.active_trees
        
        # Test pausing tree
        pause_tree!(manager, test_tree_id)
        @test manager.trees[test_tree_id].state == EnsembleForest.PAUSED
        @test test_tree_id in manager.paused_trees
        @test test_tree_id ∉ manager.active_trees
        
        # Test resuming from pause
        start_tree!(manager, test_tree_id)
        @test manager.trees[test_tree_id].state == EnsembleForest.RUNNING
        @test test_tree_id in manager.active_trees
        @test test_tree_id ∉ manager.paused_trees
        
        # Test completing tree
        complete_tree!(manager, test_tree_id, 0.85f0)
        @test manager.trees[test_tree_id].state == EnsembleForest.COMPLETED
        @test manager.trees[test_tree_id].current_score == 0.85f0
        @test manager.trees[test_tree_id].best_score == 0.85f0
        @test test_tree_id in manager.completed_trees
        @test test_tree_id ∉ manager.active_trees
        
        # Test failing tree
        test_tree_id2 = tree_ids[2]
        start_tree!(manager, test_tree_id2)
        fail_tree!(manager, test_tree_id2, "Test error")
        @test manager.trees[test_tree_id2].state == EnsembleForest.FAILED
        @test test_tree_id2 in manager.failed_trees
        @test !isempty(manager.trees[test_tree_id2].error_log)
        @test !isempty(manager.error_log)
        
        println("  ✅ Tree state management tests passed")
    end
    
    @testset "Bulk Tree Operations Tests" begin
        config = create_tree_pool_config(initial_trees = 5)
        manager = initialize_forest_manager(config)
        
        # Test starting all trees
        started_count = start_all_trees!(manager)
        @test started_count == 5
        @test length(manager.active_trees) == 5
        @test all(tree.state == EnsembleForest.RUNNING for tree in values(manager.trees))
        
        # Test pausing all trees
        paused_count = pause_all_trees!(manager)
        @test paused_count == 5
        @test length(manager.paused_trees) == 5
        @test length(manager.active_trees) == 0
        @test all(tree.state == EnsembleForest.PAUSED for tree in values(manager.trees))
        
        println("  ✅ Bulk tree operations tests passed")
    end
    
    @testset "Tree Metrics and Feature Updates Tests" begin
        config = create_tree_pool_config(initial_trees = 1)
        manager = initialize_forest_manager(config)
        
        tree_id = first(keys(manager.trees))
        tree = manager.trees[tree_id]
        
        # Test metrics update
        update_tree_metrics!(tree, 100, 50, 0.75f0, 1024 * 1024)  # 1MB GPU memory
        @test tree.iteration_count == 100
        @test tree.evaluation_count == 50
        @test tree.current_score == 0.75f0
        @test tree.best_score == 0.75f0
        @test tree.gpu_memory_used == 1024 * 1024
        
        # Test feature update
        selected_features = rand(Bool, 500)
        feature_importance = rand(Float32, 500)
        
        initial_history_length = length(tree.selection_history)
        update_tree_features!(tree, selected_features, feature_importance)
        
        @test tree.selected_features == selected_features
        @test tree.feature_importance == feature_importance
        @test length(tree.selection_history) == initial_history_length + 1
        
        # Test invalid feature vector sizes
        @test_throws ErrorException update_tree_features!(tree, rand(Bool, 400), feature_importance)
        @test_throws ErrorException update_tree_features!(tree, selected_features, rand(Float32, 400))
        
        println("  ✅ Tree metrics and feature updates tests passed")
    end
    
    @testset "Feature Synchronization Tests" begin
        config = create_tree_pool_config(initial_trees = 4, enable_cross_tree_learning = true)
        manager = initialize_forest_manager(config)
        
        # Set up different feature selections for trees
        tree_ids = collect(keys(manager.trees))
        
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
            elseif i == 3
                selected_features[101:200] .= true  # Different range
                feature_importance[101:200] .= 0.7f0
            else
                selected_features[1:50] .= true  # Partial overlap with tree 1
                feature_importance[1:50] .= 1.0f0
            end
            
            update_tree_features!(tree, selected_features, feature_importance)
        end
        
        # Test synchronization
        synchronize_features!(manager)
        
        # Verify global importance was updated
        @test !all(iszero, manager.global_feature_importance)
        
        # Verify consensus features (majority vote from 4 trees)
        # Features 1-50 should be consensus (selected by trees 1, 2, and 4)
        consensus_count = sum(manager.consensus_features)
        @test consensus_count > 0
        
        # Test that cross-tree learning affected individual trees
        # Each tree's importance should be influenced by global importance
        original_importance = manager.trees[tree_ids[1]].feature_importance[1]
        @test original_importance != 0.9f0  # Should have changed due to learning
        
        println("  ✅ Feature synchronization tests passed")
    end
    
    @testset "Forest Status and Reporting Tests" begin
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
        @test status["consensus_features"] == 0  # No features selected yet
        @test status["error_count"] == 1  # One failed tree
        
        # Test report generation
        report = generate_forest_report(manager)
        @test contains(report, "Ensemble Forest Status Report")
        @test contains(report, "Total Trees: 3")
        @test contains(report, "Active: 1")
        @test contains(report, "Completed: 1")
        @test contains(report, "Failed: 1")
        @test contains(report, "Total Iterations: 500")
        @test contains(report, "Recent Errors:")
        
        println("  ✅ Forest status and reporting tests passed")
    end
    
    @testset "Tree Removal Tests" begin
        config = create_tree_pool_config(initial_trees = 3)
        manager = initialize_forest_manager(config)
        
        tree_ids = collect(keys(manager.trees))
        test_tree_id = tree_ids[1]
        
        initial_count = length(manager.trees)
        initial_available = length(manager.available_tree_ids)
        
        # Start tree and add to various tracking sets
        start_tree!(manager, test_tree_id)
        @test test_tree_id in manager.active_trees
        
        # Remove tree
        remove_tree!(manager, test_tree_id)
        
        @test !haskey(manager.trees, test_tree_id)
        @test length(manager.trees) == initial_count - 1
        @test length(manager.available_tree_ids) == initial_available + 1
        @test test_tree_id ∉ manager.active_trees
        @test test_tree_id in manager.available_tree_ids
        
        # Test removing non-existent tree
        @test_throws ErrorException remove_tree!(manager, 999)
        
        println("  ✅ Tree removal tests passed")
    end
    
    @testset "Tree Query Operations Tests" begin
        config = create_tree_pool_config(initial_trees = 5)
        manager = initialize_forest_manager(config)
        
        tree_ids = collect(keys(manager.trees))
        
        # Set up different tree states and scores
        start_tree!(manager, tree_ids[1])
        complete_tree!(manager, tree_ids[2], 0.9f0)
        complete_tree!(manager, tree_ids[3], 0.7f0)
        fail_tree!(manager, tree_ids[4], "Test error")
        # tree_ids[5] remains READY
        
        # Test get_tree
        tree = get_tree(manager, tree_ids[1])
        @test tree.tree_id == tree_ids[1]
        @test_throws ErrorException get_tree(manager, 999)  # Non-existent tree
        
        # Test get_trees_by_state
        running_trees = get_trees_by_state(manager, EnsembleForest.RUNNING)
        @test length(running_trees) == 1
        @test running_trees[1].tree_id == tree_ids[1]
        
        completed_trees = get_trees_by_state(manager, EnsembleForest.COMPLETED)
        @test length(completed_trees) == 2
        
        ready_trees = get_trees_by_state(manager, EnsembleForest.READY)
        @test length(ready_trees) == 1
        @test ready_trees[1].tree_id == tree_ids[5]
        
        failed_trees = get_trees_by_state(manager, EnsembleForest.FAILED)
        @test length(failed_trees) == 1
        @test failed_trees[1].tree_id == tree_ids[4]
        
        # Test get_top_trees
        top_trees = get_top_trees(manager, 3)
        @test length(top_trees) == 3
        @test top_trees[1].best_score >= top_trees[2].best_score  # Sorted by score
        @test top_trees[1].tree_id == tree_ids[2]  # Highest score (0.9)
        
        top_all = get_top_trees(manager, 10)  # More than available
        @test length(top_all) == 5  # Should return all trees
        
        println("  ✅ Tree query operations tests passed")
    end
    
    @testset "Resource Management Tests" begin
        config = create_tree_pool_config(
            initial_trees = 2,
            max_gpu_memory_per_tree = 100 * 1024 * 1024,  # 100MB
            max_cpu_memory_per_tree = 50 * 1024 * 1024     # 50MB
        )
        manager = initialize_forest_manager(config)
        
        # Test initial resource allocation
        expected_cpu_usage = config.initial_trees * config.max_cpu_memory_per_tree
        @test manager.cpu_memory_usage == expected_cpu_usage
        
        # Test resource tracking with tree creation
        initial_cpu = manager.cpu_memory_usage
        new_tree_id = create_tree!(manager)
        @test manager.cpu_memory_usage == initial_cpu + config.max_cpu_memory_per_tree
        
        # Test resource cleanup with tree removal
        remove_tree!(manager, new_tree_id)
        @test manager.cpu_memory_usage == initial_cpu
        
        println("  ✅ Resource management tests passed")
    end
    
    @testset "Forest Cleanup Tests" begin
        config = create_tree_pool_config(initial_trees = 3)
        manager = initialize_forest_manager(config)
        
        # Set up some active trees
        tree_ids = collect(keys(manager.trees))
        start_tree!(manager, tree_ids[1])
        start_tree!(manager, tree_ids[2])
        
        @test length(manager.active_trees) == 2
        @test manager.forest_state == "ready"
        
        # Test cleanup
        cleanup_forest!(manager)
        
        @test manager.forest_state == "terminated"
        @test isempty(manager.active_trees)
        @test isempty(manager.paused_trees)
        @test isempty(manager.completed_trees)
        @test isempty(manager.failed_trees)
        @test manager.cpu_memory_usage == 0
        @test all(usage == 0 for usage in values(manager.gpu_memory_usage))
        
        # Verify all trees are terminated
        @test all(tree.state == EnsembleForest.TERMINATED for tree in values(manager.trees))
        
        println("  ✅ Forest cleanup tests passed")
    end
    
    @testset "Error Handling Tests" begin
        config = create_tree_pool_config(initial_trees = 1)
        manager = initialize_forest_manager(config)
        
        tree_id = first(keys(manager.trees))
        
        # Test invalid state transitions
        @test_throws ErrorException pause_tree!(manager, tree_id)  # Can't pause READY tree
        
        start_tree!(manager, tree_id)
        @test_throws ErrorException start_tree!(manager, tree_id)  # Can't start RUNNING tree
        
        # Test operations on non-existent trees
        @test_throws ErrorException start_tree!(manager, 999)
        @test_throws ErrorException pause_tree!(manager, 999)
        @test_throws ErrorException complete_tree!(manager, 999, 0.5f0)
        @test_throws ErrorException fail_tree!(manager, 999, "error")
        @test_throws ErrorException remove_tree!(manager, 999)
        
        println("  ✅ Error handling tests passed")
    end
    
    @testset "Cross-Tree Learning Tests" begin
        config = create_tree_pool_config(
            initial_trees = 3,
            enable_cross_tree_learning = true,
            learning_rate = 0.5f0
        )
        manager = initialize_forest_manager(config)
        
        tree_ids = collect(keys(manager.trees))
        
        # Set up trees with different importance scores
        for (i, tree_id) in enumerate(tree_ids)
            tree = manager.trees[tree_id]
            tree.state = EnsembleForest.RUNNING
            
            # Create different importance patterns
            importance = zeros(Float32, 500)
            importance[1:100] .= Float32(i * 0.3)  # Different values per tree
            
            selected_features = fill(false, 500)
            update_tree_features!(tree, selected_features, importance)
        end
        
        # Store original importance for comparison
        original_importance = copy(manager.trees[tree_ids[1]].feature_importance)
        
        # Run synchronization with learning
        synchronize_features!(manager)
        
        # Verify that tree importance was updated due to cross-tree learning
        updated_importance = manager.trees[tree_ids[1]].feature_importance
        @test updated_importance != original_importance
        
        # Test with learning disabled
        config_no_learning = create_tree_pool_config(
            initial_trees = 2,
            enable_cross_tree_learning = false
        )
        manager_no_learning = initialize_forest_manager(config_no_learning)
        
        tree_ids_nl = collect(keys(manager_no_learning.trees))
        for tree_id in tree_ids_nl
            tree = manager_no_learning.trees[tree_id]
            tree.state = EnsembleForest.RUNNING
            importance = rand(Float32, 500)
            selected_features = fill(false, 500)
            update_tree_features!(tree, selected_features, importance)
        end
        
        original_nl = copy(manager_no_learning.trees[tree_ids_nl[1]].feature_importance)
        synchronize_features!(manager_no_learning)
        updated_nl = manager_no_learning.trees[tree_ids_nl[1]].feature_importance
        
        @test updated_nl == original_nl  # Should not change without learning
        
        println("  ✅ Cross-tree learning tests passed")
    end
end

println("All Ensemble Forest tests passed successfully!")
println("✅ Tree pool configuration and forest manager initialization")
println("✅ Tree creation, removal, and lifecycle management")
println("✅ Tree state transitions and bulk operations")
println("✅ Feature synchronization and cross-tree learning")
println("✅ Resource management and memory tracking")
println("✅ Performance monitoring and status reporting")
println("✅ Tree querying and ranking operations")
println("✅ Error handling and robustness validation")
println("✅ Forest cleanup and termination procedures")
println("✅ Ready for Stage 2 GPU-MCTS implementation")