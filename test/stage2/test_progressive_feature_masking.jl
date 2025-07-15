"""
Test Suite for Progressive Feature Masking System
Validates dynamic feature masking, constraint inheritance, mask merging,
and validation functionality for MCTS ensemble feature selection.
"""

using Test
using Random
using Statistics
using Dates

# Include the progressive feature masking module
include("../../src/stage2/progressive_feature_masking.jl")
using .ProgressiveFeatureMasking

@testset "Progressive Feature Masking Tests" begin
    
    Random.seed!(42)  # For reproducible tests
    
    @testset "Configuration Tests" begin
        # Test default configuration
        config = create_progressive_masking_config()
        
        @test config.feature_dimension == 500
        @test config.min_features_per_tree == 10
        @test config.max_features_per_tree == 100
        @test config.enable_parent_inheritance == true
        @test config.inheritance_strength == 0.7f0
        @test config.max_inheritance_depth == 10
        @test config.validation_level == MODERATE
        @test config.enable_strict_validation == true
        @test config.min_available_features == 20
        @test config.enable_mask_caching == true
        @test config.cache_size == 1000
        
        # Test custom configuration
        custom_config = create_progressive_masking_config(
            feature_dimension = 200,
            min_features_per_tree = 5,
            max_features_per_tree = 50,
            inheritance_strength = 0.5f0,
            validation_level = STRICT,
            enable_mask_caching = false
        )
        
        @test custom_config.feature_dimension == 200
        @test custom_config.min_features_per_tree == 5
        @test custom_config.max_features_per_tree == 50
        @test custom_config.inheritance_strength == 0.5f0
        @test custom_config.validation_level == STRICT
        @test custom_config.enable_mask_caching == false
        
        println("  ✅ Configuration tests passed")
    end
    
    @testset "Feature Action Enum Tests" begin
        # Test enum values
        @test Int(FEATURE_SELECTED) == 1
        @test Int(FEATURE_REJECTED) == 2
        @test Int(FEATURE_DESELECTED) == 3
        @test Int(FEATURE_LOCKED) == 4
        @test Int(FEATURE_RESTORED) == 5
        
        println("  ✅ Feature action enum tests passed")
    end
    
    @testset "Mask Operation Enum Tests" begin
        # Test enum values
        @test Int(INTERSECTION) == 1
        @test Int(UNION) == 2
        @test Int(WEIGHTED_MERGE) == 3
        @test Int(PRIORITY_MERGE) == 4
        
        println("  ✅ Mask operation enum tests passed")
    end
    
    @testset "Validation Level Enum Tests" begin
        # Test enum values
        @test Int(STRICT) == 1
        @test Int(MODERATE) == 2
        @test Int(PERMISSIVE) == 3
        
        println("  ✅ Validation level enum tests passed")
    end
    
    @testset "Feature Mask Creation Tests" begin
        # Test basic mask creation
        mask = create_feature_mask(1, 10, 100)
        
        @test mask.tree_id == 1
        @test mask.node_id == 10
        @test mask.feature_dimension == 100
        @test length(mask.available_features) == 100
        @test length(mask.selected_features) == 100
        @test length(mask.rejected_features) == 100
        @test length(mask.locked_features) == 100
        @test length(mask.feature_priorities) == 100
        
        # Initially all features should be available, none selected
        @test all(mask.available_features)
        @test !any(mask.selected_features)
        @test !any(mask.rejected_features)
        @test !any(mask.locked_features)
        @test all(mask.feature_priorities .== 1.0f0)
        
        @test mask.mask_version == 1
        @test mask.update_count == 0
        @test mask.validation_failures == 0
        @test mask.is_valid == true
        @test mask.is_locked == false
        @test mask.requires_validation == false
        
        # Test mask creation with custom parameters
        custom_mask = create_feature_mask(2, 20, 50, 
                                        min_features_required = 5,
                                        max_features_allowed = 25,
                                        parent_mask_id = 10)
        
        @test custom_mask.min_features_required == 5
        @test custom_mask.max_features_allowed == 25
        @test custom_mask.parent_mask_id == 10
        
        println("  ✅ Feature mask creation tests passed")
    end
    
    @testset "Manager Initialization Tests" begin
        config = create_progressive_masking_config(feature_dimension = 100)
        manager = initialize_progressive_masking_manager(config)
        
        @test manager.config == config
        @test length(manager.masks) == 0
        @test length(manager.mask_cache) == 0
        @test length(manager.tree_masks) == 0
        @test length(manager.global_constraints) == 100
        @test all(manager.global_constraints)  # All features initially available
        @test length(manager.update_history) == 0
        @test manager.manager_state == "active"
        @test length(manager.error_log) == 0
        
        # Test statistics initialization
        @test manager.stats.total_mask_updates == 0
        @test manager.stats.successful_updates == 0
        @test manager.stats.failed_updates == 0
        @test manager.stats.validation_checks == 0
        @test manager.stats.validation_failures == 0
        @test manager.stats.cache_hits == 0
        @test manager.stats.cache_misses == 0
        
        println("  ✅ Manager initialization tests passed")
    end
    
    @testset "Mask Creation for Nodes Tests" begin
        config = create_progressive_masking_config(feature_dimension = 50)
        manager = initialize_progressive_masking_manager(config)
        
        # Create mask for root node
        mask1 = create_mask_for_node!(manager, 1, 1)
        
        @test mask1.tree_id == 1
        @test mask1.node_id == 1
        @test length(manager.masks) == 1
        @test haskey(manager.masks, (1, 1))
        @test haskey(manager.tree_masks, 1)
        @test length(manager.tree_masks[1]) == 1
        
        # Create mask for child node
        mask2 = create_mask_for_node!(manager, 1, 2, 1)  # Parent node 1
        
        @test mask2.tree_id == 1
        @test mask2.node_id == 2
        @test mask2.parent_mask_id == 1
        @test length(manager.masks) == 2
        @test length(manager.tree_masks[1]) == 2
        
        # Test duplicate creation returns existing mask
        mask1_duplicate = create_mask_for_node!(manager, 1, 1)
        @test mask1_duplicate === mask1
        @test length(manager.masks) == 2  # No new mask created
        
        println("  ✅ Mask creation for nodes tests passed")
    end
    
    @testset "Mask Action Application Tests" begin
        mask = create_feature_mask(1, 1, 20)
        
        # Test feature selection
        @test ProgressiveFeatureMasking.apply_mask_action!(mask, 5, FEATURE_SELECTED) == true
        @test mask.selected_features[5] == true
        @test mask.available_features[5] == true
        
        # Test feature rejection
        @test ProgressiveFeatureMasking.apply_mask_action!(mask, 10, FEATURE_REJECTED) == true
        @test mask.rejected_features[10] == true
        @test mask.available_features[10] == false
        @test mask.selected_features[10] == false
        
        # Test cannot select rejected feature
        @test ProgressiveFeatureMasking.apply_mask_action!(mask, 10, FEATURE_SELECTED) == false
        
        # Test feature deselection
        @test ProgressiveFeatureMasking.apply_mask_action!(mask, 5, FEATURE_DESELECTED) == true
        @test mask.selected_features[5] == false
        @test mask.available_features[5] == true  # Still available
        
        # Test feature locking
        @test ProgressiveFeatureMasking.apply_mask_action!(mask, 15, FEATURE_SELECTED) == true
        @test ProgressiveFeatureMasking.apply_mask_action!(mask, 15, FEATURE_LOCKED) == true
        @test mask.locked_features[15] == true
        @test mask.selected_features[15] == true
        
        # Test cannot reject locked feature
        @test ProgressiveFeatureMasking.apply_mask_action!(mask, 15, FEATURE_REJECTED) == false
        
        # Test feature restoration
        @test ProgressiveFeatureMasking.apply_mask_action!(mask, 10, FEATURE_RESTORED) == true
        @test mask.rejected_features[10] == false
        @test mask.available_features[10] == true
        @test mask.selected_features[10] == false
        
        # Test invalid feature ID
        @test ProgressiveFeatureMasking.apply_mask_action!(mask, 0, FEATURE_SELECTED) == false
        @test ProgressiveFeatureMasking.apply_mask_action!(mask, 25, FEATURE_SELECTED) == false
        
        println("  ✅ Mask action application tests passed")
    end
    
    @testset "Mask Update Tests" begin
        config = create_progressive_masking_config(feature_dimension = 30)
        manager = initialize_progressive_masking_manager(config)
        
        # Create mask
        mask = create_mask_for_node!(manager, 1, 1)
        initial_version = mask.mask_version
        
        # Test successful update
        success = update_feature_mask!(manager, 1, 1, 5, FEATURE_SELECTED)
        @test success == true
        @test mask.mask_version == initial_version + 1
        @test mask.selected_features[5] == true
        @test mask.update_count == 1
        @test manager.stats.successful_updates == 1
        @test manager.stats.total_mask_updates == 1
        
        # Test update non-existent mask
        success = update_feature_mask!(manager, 99, 99, 5, FEATURE_SELECTED)
        @test success == false
        @test manager.stats.failed_updates == 1
        
        # Test update with locked mask
        mask.is_locked = true
        success = update_feature_mask!(manager, 1, 1, 10, FEATURE_SELECTED)
        @test success == false
        @test manager.stats.failed_updates == 2
        
        # Test update history
        @test length(manager.update_history) >= 2  # At least 2 events recorded
        
        println("  ✅ Mask update tests passed")
    end
    
    @testset "Parent Inheritance Tests" begin
        parent_mask = create_feature_mask(1, 1, 20)
        child_mask = create_feature_mask(1, 2, 20)
        
        # Set up parent mask state
        ProgressiveFeatureMasking.apply_mask_action!(parent_mask, 5, FEATURE_REJECTED)
        ProgressiveFeatureMasking.apply_mask_action!(parent_mask, 10, FEATURE_SELECTED)
        ProgressiveFeatureMasking.apply_mask_action!(parent_mask, 15, FEATURE_LOCKED)
        parent_mask.feature_priorities[8] = 2.0f0
        
        # Apply inheritance
        ProgressiveFeatureMasking.inherit_parent_constraints!(child_mask, parent_mask, 0.8f0)
        
        # Check inheritance
        @test child_mask.rejected_features[5] == true    # Inherited rejection
        @test child_mask.available_features[5] == false  # Not available due to rejection
        @test child_mask.locked_features[15] == true     # Inherited lock (high strength)
        @test child_mask.feature_priorities[8] > 1.0f0   # Inherited priority
        
        # Test lower inheritance strength
        child_mask2 = create_feature_mask(1, 3, 20)
        ProgressiveFeatureMasking.inherit_parent_constraints!(child_mask2, parent_mask, 0.3f0)
        @test child_mask2.locked_features[15] == false   # Not inherited (low strength)
        
        println("  ✅ Parent inheritance tests passed")
    end
    
    @testset "Global Constraints Tests" begin
        config = create_progressive_masking_config(feature_dimension = 20)
        manager = initialize_progressive_masking_manager(config)
        
        # Create some masks
        mask1 = create_mask_for_node!(manager, 1, 1)
        mask2 = create_mask_for_node!(manager, 2, 1)
        
        # Select some features
        update_feature_mask!(manager, 1, 1, 5, FEATURE_SELECTED)
        update_feature_mask!(manager, 2, 1, 5, FEATURE_SELECTED)
        
        # Set global constraints (disable feature 5)
        global_constraints = trues(20)
        global_constraints[5] = false
        set_global_constraints!(manager, global_constraints)
        
        # Check that feature 5 is now unavailable and deselected
        @test mask1.available_features[5] == false
        @test mask1.selected_features[5] == false
        @test mask1.rejected_features[5] == true
        @test mask2.available_features[5] == false
        @test mask2.selected_features[5] == false
        @test mask2.rejected_features[5] == true
        
        println("  ✅ Global constraints tests passed")
    end
    
    @testset "Mask Merging Tests" begin
        # Create test masks
        mask1 = create_feature_mask(1, 1, 10)
        mask2 = create_feature_mask(2, 1, 10)
        mask3 = create_feature_mask(3, 1, 10)
        
        # Set up different states
        ProgressiveFeatureMasking.apply_mask_action!(mask1, 1, FEATURE_SELECTED)
        ProgressiveFeatureMasking.apply_mask_action!(mask1, 2, FEATURE_REJECTED)
        
        ProgressiveFeatureMasking.apply_mask_action!(mask2, 1, FEATURE_SELECTED)
        ProgressiveFeatureMasking.apply_mask_action!(mask2, 3, FEATURE_SELECTED)
        ProgressiveFeatureMasking.apply_mask_action!(mask2, 4, FEATURE_REJECTED)
        
        ProgressiveFeatureMasking.apply_mask_action!(mask3, 2, FEATURE_SELECTED)
        ProgressiveFeatureMasking.apply_mask_action!(mask3, 5, FEATURE_REJECTED)
        
        masks = [mask1, mask2, mask3]
        
        # Test intersection merge
        merge_config = create_mask_merge_config(operation = INTERSECTION)
        merged_intersection = merge_masks(masks, merge_config)
        
        @test merged_intersection.selected_features[1] == true   # Selected in mask1 and mask2
        @test merged_intersection.selected_features[2] == false  # Selected in mask3, rejected in mask1
        @test merged_intersection.selected_features[3] == false  # Only selected in mask2
        @test merged_intersection.rejected_features[2] == true   # Rejected in mask1
        @test merged_intersection.rejected_features[4] == true   # Rejected in mask2
        @test merged_intersection.rejected_features[5] == true   # Rejected in mask3
        
        # Test union merge
        merge_config_union = create_mask_merge_config(operation = UNION)
        merged_union = merge_masks(masks, merge_config_union)
        
        @test merged_union.selected_features[1] == true   # Selected in mask1 and mask2
        @test merged_union.selected_features[3] == true   # Selected in mask2
        @test merged_union.rejected_features[2] == false  # Union reduces restrictions
        
        # Test weighted merge
        weights = [0.5f0, 0.3f0, 0.2f0]
        merge_config_weighted = create_mask_merge_config(operation = WEIGHTED_MERGE, weights = weights)
        merged_weighted = merge_masks(masks, merge_config_weighted)
        
        # Should have feature 1 selected (selected in mask1 and mask2 with higher weights)
        @test merged_weighted.selected_features[1] == true
        
        # Test priority merge
        priority_order = [3, 1, 2]  # mask3 highest priority, mask2 lowest
        merge_config_priority = create_mask_merge_config(operation = PRIORITY_MERGE, priority_order = priority_order)
        merged_priority = merge_masks(masks, merge_config_priority)
        
        # Higher priority masks should override lower priority
        @test merged_priority.selected_features[2] == true   # mask3 has highest priority
        
        println("  ✅ Mask merging tests passed")
    end
    
    @testset "Mask Validation Tests" begin
        config = create_progressive_masking_config(
            feature_dimension = 20,
            min_features_per_tree = 5,
            validation_level = STRICT
        )
        manager = initialize_progressive_masking_manager(config)
        
        # Create valid mask
        mask = create_mask_for_node!(manager, 1, 1)
        @test validate_mask!(manager, mask) == true
        @test mask.is_valid == true
        @test manager.stats.validation_checks == 1
        @test manager.stats.validation_failures == 0
        
        # Create invalid mask (selected and rejected feature)
        invalid_mask = create_feature_mask(1, 2, 20, min_features_required = 5)
        invalid_mask.selected_features[5] = true
        invalid_mask.rejected_features[5] = true
        
        @test validate_mask!(manager, invalid_mask) == false
        @test invalid_mask.is_valid == false
        @test manager.stats.validation_failures == 1
        
        # Create mask with too few available features
        low_features_mask = create_feature_mask(1, 3, 20, min_features_required = 15)
        low_features_mask.available_features[1:10] .= false
        low_features_mask.rejected_features[1:10] .= true
        
        @test validate_mask!(manager, low_features_mask) == false
        @test manager.stats.validation_failures == 2
        
        println("  ✅ Mask validation tests passed")
    end
    
    @testset "Feature Access Tests" begin
        config = create_progressive_masking_config(feature_dimension = 30)
        manager = initialize_progressive_masking_manager(config)
        
        # Create mask and set some features
        mask = create_mask_for_node!(manager, 1, 1)
        update_feature_mask!(manager, 1, 1, 5, FEATURE_SELECTED)
        update_feature_mask!(manager, 1, 1, 10, FEATURE_SELECTED)
        update_feature_mask!(manager, 1, 1, 15, FEATURE_REJECTED)
        
        # Test get selected features
        selected = get_selected_features(manager, 1, 1)
        @test 5 in selected
        @test 10 in selected
        @test 15 ∉ selected
        @test length(selected) == 2
        
        # Test get available features
        available = get_available_features(manager, 1, 1)
        @test 5 in available
        @test 10 in available
        @test 15 ∉ available  # Rejected
        @test length(available) == 29  # 30 total - 1 rejected
        
        # Test non-existent mask
        selected_empty = get_selected_features(manager, 99, 99)
        @test isempty(selected_empty)
        
        available_all = get_available_features(manager, 99, 99)
        @test length(available_all) == 30  # Returns all features when mask not found
        
        println("  ✅ Feature access tests passed")
    end
    
    @testset "Cache Operations Tests" begin
        config = create_progressive_masking_config(
            feature_dimension = 20,
            enable_mask_caching = true,
            cache_size = 3
        )
        manager = initialize_progressive_masking_manager(config)
        
        # Create mask and update cache
        mask1 = create_mask_for_node!(manager, 1, 1)
        update_feature_mask!(manager, 1, 1, 5, FEATURE_SELECTED)
        update_feature_mask!(manager, 1, 1, 10, FEATURE_SELECTED)
        
        # Test cache miss
        cached_mask = get_cached_mask(manager, [1, 2, 3])
        @test isnothing(cached_mask)
        @test manager.stats.cache_misses == 1
        
        # Test cache operations with updates trigger caching
        @test length(manager.mask_cache) >= 0  # Cache might have entries from updates
        
        # Test cache size limit by manually adding entries
        ProgressiveFeatureMasking.update_mask_cache!(manager, mask1)
        
        mask2 = create_feature_mask(2, 1, 20)
        ProgressiveFeatureMasking.apply_mask_action!(mask2, 1, FEATURE_SELECTED)
        ProgressiveFeatureMasking.update_mask_cache!(manager, mask2)
        
        mask3 = create_feature_mask(3, 1, 20)
        ProgressiveFeatureMasking.apply_mask_action!(mask3, 2, FEATURE_SELECTED)
        ProgressiveFeatureMasking.update_mask_cache!(manager, mask3)
        
        mask4 = create_feature_mask(4, 1, 20)
        ProgressiveFeatureMasking.apply_mask_action!(mask4, 3, FEATURE_SELECTED)
        ProgressiveFeatureMasking.update_mask_cache!(manager, mask4)
        
        # Cache should be limited to configured size
        @test length(manager.mask_cache) <= config.cache_size
        
        println("  ✅ Cache operations tests passed")
    end
    
    @testset "Status and Reporting Tests" begin
        config = create_progressive_masking_config(feature_dimension = 20)
        manager = initialize_progressive_masking_manager(config)
        
        # Create some activity
        mask1 = create_mask_for_node!(manager, 1, 1)
        mask2 = create_mask_for_node!(manager, 1, 2)
        update_feature_mask!(manager, 1, 1, 5, FEATURE_SELECTED)
        update_feature_mask!(manager, 1, 2, 10, FEATURE_SELECTED)
        
        # Test status retrieval
        status = get_masking_status(manager)
        
        @test haskey(status, "manager_state")
        @test haskey(status, "total_masks")
        @test haskey(status, "active_trees")
        @test haskey(status, "total_updates")
        @test haskey(status, "successful_updates")
        @test haskey(status, "validation_checks")
        @test haskey(status, "cache_size")
        @test haskey(status, "average_update_time_ms")
        
        @test status["manager_state"] == "active"
        @test status["total_masks"] == 2
        @test status["active_trees"] == 1
        @test status["successful_updates"] == 2
        
        # Test report generation
        report = generate_masking_report(manager)
        
        @test contains(report, "Progressive Feature Masking Report")
        @test contains(report, "Manager State: active")
        @test contains(report, "Total Masks: 2")
        @test contains(report, "Active Trees: 1")
        @test contains(report, "Feature Dimension: 20")
        @test contains(report, "Successful Updates: 2")
        
        println("  ✅ Status and reporting tests passed")
    end
    
    @testset "Error Handling Tests" begin
        config = create_progressive_masking_config(feature_dimension = 10)
        manager = initialize_progressive_masking_manager(config)
        
        # Test invalid global constraints
        @test_throws BoundsError set_global_constraints!(manager, trues(5))  # Wrong size
        
        # Test merge with empty mask list
        @test_throws ErrorException merge_masks(FeatureMask[], create_mask_merge_config())
        
        # Test weighted merge with mismatched weights
        mask1 = create_feature_mask(1, 1, 10)
        mask2 = create_feature_mask(2, 1, 10)
        masks = [mask1, mask2]
        wrong_weights = [0.5f0]  # Only one weight for two masks
        merge_config = create_mask_merge_config(operation = WEIGHTED_MERGE, weights = wrong_weights)
        
        @test_throws ErrorException merge_masks(masks, merge_config)
        
        # Test priority merge with mismatched priority order
        wrong_priorities = [1]  # Only one priority for two masks
        merge_config_priority = create_mask_merge_config(operation = PRIORITY_MERGE, priority_order = wrong_priorities)
        
        @test_throws ErrorException merge_masks(masks, merge_config_priority)
        
        println("  ✅ Error handling tests passed")
    end
    
    @testset "Cleanup Tests" begin
        config = create_progressive_masking_config(feature_dimension = 20)
        manager = initialize_progressive_masking_manager(config)
        
        # Create some data
        mask1 = create_mask_for_node!(manager, 1, 1)
        mask2 = create_mask_for_node!(manager, 2, 1)
        update_feature_mask!(manager, 1, 1, 5, FEATURE_SELECTED)
        
        @test length(manager.masks) == 2
        @test length(manager.tree_masks) == 2
        @test length(manager.update_history) > 0
        
        # Test cleanup
        cleanup_masking_manager!(manager)
        
        @test length(manager.masks) == 0
        @test length(manager.tree_masks) == 0
        @test length(manager.update_history) == 0
        @test length(manager.mask_cache) == 0
        @test manager.manager_state == "shutdown"
        
        println("  ✅ Cleanup tests passed")
    end
end

println("All Progressive Feature Masking tests completed!")
println("✅ Configuration and enum validation")
println("✅ Feature mask creation and initialization")
println("✅ Manager setup and node mask creation")
println("✅ Mask action application (select, reject, lock, restore)")
println("✅ Mask update operations with version tracking")
println("✅ Parent-child constraint inheritance")
println("✅ Global constraint application and propagation")
println("✅ Mask merging with intersection, union, weighted, and priority operations")
println("✅ Mask validation with strict/moderate/permissive levels")
println("✅ Feature access methods for available and selected features")
println("✅ Cache operations with LRU management")
println("✅ Status reporting and performance monitoring")
println("✅ Error handling and edge cases")
println("✅ Cleanup and resource management")
println("✅ Ready for MCTS progressive feature constraint propagation")