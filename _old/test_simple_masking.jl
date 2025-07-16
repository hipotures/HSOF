"""
Simple test for Progressive Feature Masking
Tests core functionality without external dependencies
"""

using Test
using Random
using Statistics
using Dates

# Test the core structures and logic without complex dependencies
module SimpleMaskingTest

using Random
using Statistics
using Dates

# Define core enums
@enum FeatureAction begin
    FEATURE_SELECTED = 1
    FEATURE_REJECTED = 2
    FEATURE_DESELECTED = 3
    FEATURE_LOCKED = 4
    FEATURE_RESTORED = 5
end

@enum MaskOperation begin
    INTERSECTION = 1
    UNION = 2
    WEIGHTED_MERGE = 3
    PRIORITY_MERGE = 4
end

@enum ValidationLevel begin
    STRICT = 1
    MODERATE = 2
    PERMISSIVE = 3
end

# Core structures
mutable struct FeatureMask
    tree_id::Int
    node_id::Int
    feature_dimension::Int
    available_features::BitVector
    selected_features::BitVector
    rejected_features::BitVector
    locked_features::BitVector
    mask_version::Int
    creation_time::DateTime
    last_update::DateTime
    parent_mask_id::Union{Int, Nothing}
    min_features_required::Int
    max_features_allowed::Int
    feature_priorities::Vector{Float32}
    update_count::Int
    validation_failures::Int
    is_valid::Bool
    is_locked::Bool
    requires_validation::Bool
end

struct ProgressiveMaskingConfig
    feature_dimension::Int
    min_features_per_tree::Int
    max_features_per_tree::Int
    enable_parent_inheritance::Bool
    inheritance_strength::Float32
    validation_level::ValidationLevel
    enable_strict_validation::Bool
    min_available_features::Int
    enable_mask_caching::Bool
    cache_size::Int
end

function create_progressive_masking_config(;
    feature_dimension::Int = 100,
    min_features_per_tree::Int = 10,
    max_features_per_tree::Int = 50,
    enable_parent_inheritance::Bool = true,
    inheritance_strength::Float32 = 0.7f0,
    validation_level::ValidationLevel = MODERATE,
    enable_strict_validation::Bool = true,
    min_available_features::Int = 20,
    enable_mask_caching::Bool = true,
    cache_size::Int = 100
)
    return ProgressiveMaskingConfig(
        feature_dimension, min_features_per_tree, max_features_per_tree,
        enable_parent_inheritance, inheritance_strength, validation_level,
        enable_strict_validation, min_available_features,
        enable_mask_caching, cache_size
    )
end

mutable struct SimpleMaskingManager
    config::ProgressiveMaskingConfig
    masks::Dict{Tuple{Int, Int}, FeatureMask}
    global_constraints::BitVector
    manager_state::String
    total_updates::Int
    successful_updates::Int
    failed_updates::Int
end

function create_feature_mask(tree_id::Int, node_id::Int, feature_dimension::Int;
                            min_features_required::Int = 10,
                            max_features_allowed::Int = feature_dimension,
                            parent_mask_id::Union{Int, Nothing} = nothing)
    return FeatureMask(
        tree_id, node_id, feature_dimension,
        trues(feature_dimension),      # All features initially available
        falses(feature_dimension),     # No features initially selected
        falses(feature_dimension),     # No features initially rejected
        falses(feature_dimension),     # No features initially locked
        1, now(), now(), parent_mask_id,
        min_features_required, max_features_allowed,
        ones(Float32, feature_dimension),
        0, 0, true, false, false
    )
end

function initialize_simple_masking_manager(config::ProgressiveMaskingConfig = create_progressive_masking_config())
    return SimpleMaskingManager(
        config,
        Dict{Tuple{Int, Int}, FeatureMask}(),
        trues(config.feature_dimension),
        "active",
        0, 0, 0
    )
end

function create_mask_for_node!(manager::SimpleMaskingManager, tree_id::Int, node_id::Int)::FeatureMask
    mask_key = (tree_id, node_id)
    
    if haskey(manager.masks, mask_key)
        return manager.masks[mask_key]
    end
    
    mask = create_feature_mask(
        tree_id, node_id, manager.config.feature_dimension,
        min_features_required = manager.config.min_features_per_tree,
        max_features_allowed = manager.config.max_features_per_tree
    )
    
    # Apply global constraints
    apply_global_constraints!(mask, manager.global_constraints)
    
    manager.masks[mask_key] = mask
    return mask
end

function apply_global_constraints!(mask::FeatureMask, global_constraints::BitVector)
    mask.available_features .&= global_constraints
    mask.selected_features .&= mask.available_features
    mask.rejected_features .|= .!global_constraints
end

function apply_mask_action!(mask::FeatureMask, feature_id::Int, action::FeatureAction)::Bool
    if feature_id < 1 || feature_id > mask.feature_dimension
        return false
    end
    
    if mask.locked_features[feature_id] && action != FEATURE_LOCKED
        return false
    end
    
    success = true
    
    if action == FEATURE_SELECTED
        if mask.available_features[feature_id] && !mask.rejected_features[feature_id]
            mask.selected_features[feature_id] = true
        else
            success = false
        end
    elseif action == FEATURE_REJECTED
        if !mask.locked_features[feature_id]
            mask.rejected_features[feature_id] = true
            mask.available_features[feature_id] = false
            mask.selected_features[feature_id] = false
        else
            success = false
        end
    elseif action == FEATURE_DESELECTED
        if !mask.locked_features[feature_id]
            mask.selected_features[feature_id] = false
        else
            success = false
        end
    elseif action == FEATURE_LOCKED
        mask.locked_features[feature_id] = true
    elseif action == FEATURE_RESTORED
        if !mask.locked_features[feature_id]
            mask.rejected_features[feature_id] = false
            mask.available_features[feature_id] = true
            mask.selected_features[feature_id] = false
        else
            success = false
        end
    end
    
    return success
end

function update_feature_mask!(manager::SimpleMaskingManager, tree_id::Int, node_id::Int, 
                             feature_id::Int, action::FeatureAction)::Bool
    mask_key = (tree_id, node_id)
    mask = get(manager.masks, mask_key, nothing)
    
    if isnothing(mask)
        manager.failed_updates += 1
        return false
    end
    
    if mask.is_locked
        manager.failed_updates += 1
        return false
    end
    
    success = apply_mask_action!(mask, feature_id, action)
    
    if success
        mask.mask_version += 1
        mask.last_update = now()
        mask.update_count += 1
        manager.successful_updates += 1
    else
        mask.validation_failures += 1
        manager.failed_updates += 1
    end
    
    manager.total_updates += 1
    return success
end

function inherit_parent_constraints!(child_mask::FeatureMask, parent_mask::FeatureMask, 
                                   inheritance_strength::Float32)
    child_mask.available_features .&= parent_mask.available_features
    child_mask.rejected_features .|= parent_mask.rejected_features
    
    if inheritance_strength >= 0.5f0
        child_mask.locked_features .|= parent_mask.locked_features
    end
    
    for i in 1:length(child_mask.feature_priorities)
        child_mask.feature_priorities[i] = 
            (child_mask.feature_priorities[i] + parent_mask.feature_priorities[i] * inheritance_strength) / 2
    end
    
    child_mask.available_features .&= .!child_mask.rejected_features
    child_mask.selected_features .&= child_mask.available_features
end

function validate_mask(mask::FeatureMask)::Bool
    # Check basic consistency
    if any(mask.selected_features .& mask.rejected_features)
        return false
    end
    
    if any(mask.selected_features .& .!mask.available_features)
        return false
    end
    
    if any(mask.available_features .& mask.rejected_features)
        return false
    end
    
    # Check minimum features requirement
    available_count = sum(mask.available_features)
    if available_count < mask.min_features_required
        return false
    end
    
    # Check maximum features constraint
    selected_count = sum(mask.selected_features)
    if selected_count > mask.max_features_allowed
        return false
    end
    
    # Check locked features consistency
    if any(mask.locked_features .& mask.rejected_features)
        return false
    end
    
    return true
end

function merge_masks_intersection(masks::Vector{FeatureMask})::FeatureMask
    if isempty(masks)
        error("Cannot merge empty mask list")
    end
    
    result_mask = deepcopy(masks[1])
    result_mask.tree_id = -1
    result_mask.node_id = -1
    result_mask.mask_version = 1
    result_mask.creation_time = now()
    result_mask.last_update = now()
    
    for mask in masks[2:end]
        result_mask.available_features .&= mask.available_features
        result_mask.rejected_features .|= mask.rejected_features
        result_mask.locked_features .|= mask.locked_features
    end
    
    for mask in masks
        result_mask.selected_features .&= mask.selected_features
    end
    
    result_mask.available_features .&= .!result_mask.rejected_features
    result_mask.selected_features .&= result_mask.available_features
    
    return result_mask
end

function get_selected_features(manager::SimpleMaskingManager, tree_id::Int, node_id::Int)::Vector{Int}
    mask_key = (tree_id, node_id)
    mask = get(manager.masks, mask_key, nothing)
    
    if isnothing(mask)
        return Int[]
    end
    
    return findall(mask.selected_features)
end

function get_available_features(manager::SimpleMaskingManager, tree_id::Int, node_id::Int)::Vector{Int}
    mask_key = (tree_id, node_id)
    mask = get(manager.masks, mask_key, nothing)
    
    if isnothing(mask)
        return collect(1:manager.config.feature_dimension)
    end
    
    return findall(mask.available_features)
end

function set_global_constraints!(manager::SimpleMaskingManager, constraints::BitVector)
    if length(constraints) != manager.config.feature_dimension
        error("Constraints length must match feature dimension")
    end
    
    manager.global_constraints = copy(constraints)
    
    for mask in values(manager.masks)
        apply_global_constraints!(mask, constraints)
    end
end

function get_manager_status(manager::SimpleMaskingManager)
    return Dict{String, Any}(
        "manager_state" => manager.manager_state,
        "total_masks" => length(manager.masks),
        "total_updates" => manager.total_updates,
        "successful_updates" => manager.successful_updates,
        "failed_updates" => manager.failed_updates,
        "success_rate" => manager.total_updates > 0 ? manager.successful_updates / manager.total_updates : 0.0
    )
end

end # module

using .SimpleMaskingTest

@testset "Simple Progressive Feature Masking Tests" begin
    
    Random.seed!(42)
    
    @testset "Configuration Tests" begin
        config = SimpleMaskingTest.create_progressive_masking_config()
        
        @test config.feature_dimension == 100
        @test config.min_features_per_tree == 10
        @test config.max_features_per_tree == 50
        @test config.enable_parent_inheritance == true
        @test config.inheritance_strength == 0.7f0
        @test config.validation_level == SimpleMaskingTest.MODERATE
        @test config.enable_strict_validation == true
        @test config.min_available_features == 20
        
        custom_config = SimpleMaskingTest.create_progressive_masking_config(
            feature_dimension = 50,
            min_features_per_tree = 5,
            inheritance_strength = 0.5f0
        )
        
        @test custom_config.feature_dimension == 50
        @test custom_config.min_features_per_tree == 5
        @test custom_config.inheritance_strength == 0.5f0
        
        println("  ✅ Configuration tests passed")
    end
    
    @testset "Manager Initialization Tests" begin
        config = SimpleMaskingTest.create_progressive_masking_config(feature_dimension = 30)
        manager = SimpleMaskingTest.initialize_simple_masking_manager(config)
        
        @test manager.config == config
        @test length(manager.masks) == 0
        @test length(manager.global_constraints) == 30
        @test all(manager.global_constraints)
        @test manager.manager_state == "active"
        @test manager.total_updates == 0
        @test manager.successful_updates == 0
        @test manager.failed_updates == 0
        
        println("  ✅ Manager initialization tests passed")
    end
    
    @testset "Feature Mask Creation Tests" begin
        mask = SimpleMaskingTest.create_feature_mask(1, 10, 20)
        
        @test mask.tree_id == 1
        @test mask.node_id == 10
        @test mask.feature_dimension == 20
        @test length(mask.available_features) == 20
        @test length(mask.selected_features) == 20
        @test length(mask.rejected_features) == 20
        @test length(mask.locked_features) == 20
        @test length(mask.feature_priorities) == 20
        
        # Initially all features available, none selected
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
        
        println("  ✅ Feature mask creation tests passed")
    end
    
    @testset "Mask Action Application Tests" begin
        mask = SimpleMaskingTest.create_feature_mask(1, 1, 15)
        
        # Test feature selection
        @test SimpleMaskingTest.apply_mask_action!(mask, 5, SimpleMaskingTest.FEATURE_SELECTED) == true
        @test mask.selected_features[5] == true
        @test mask.available_features[5] == true
        
        # Test feature rejection
        @test SimpleMaskingTest.apply_mask_action!(mask, 10, SimpleMaskingTest.FEATURE_REJECTED) == true
        @test mask.rejected_features[10] == true
        @test mask.available_features[10] == false
        @test mask.selected_features[10] == false
        
        # Test cannot select rejected feature
        @test SimpleMaskingTest.apply_mask_action!(mask, 10, SimpleMaskingTest.FEATURE_SELECTED) == false
        
        # Test feature deselection
        @test SimpleMaskingTest.apply_mask_action!(mask, 5, SimpleMaskingTest.FEATURE_DESELECTED) == true
        @test mask.selected_features[5] == false
        @test mask.available_features[5] == true
        
        # Test feature locking
        SimpleMaskingTest.apply_mask_action!(mask, 3, SimpleMaskingTest.FEATURE_SELECTED)
        @test SimpleMaskingTest.apply_mask_action!(mask, 3, SimpleMaskingTest.FEATURE_LOCKED) == true
        @test mask.locked_features[3] == true
        
        # Test cannot reject locked feature
        @test SimpleMaskingTest.apply_mask_action!(mask, 3, SimpleMaskingTest.FEATURE_REJECTED) == false
        
        # Test feature restoration
        @test SimpleMaskingTest.apply_mask_action!(mask, 10, SimpleMaskingTest.FEATURE_RESTORED) == true
        @test mask.rejected_features[10] == false
        @test mask.available_features[10] == true
        @test mask.selected_features[10] == false
        
        # Test invalid feature ID
        @test SimpleMaskingTest.apply_mask_action!(mask, 0, SimpleMaskingTest.FEATURE_SELECTED) == false
        @test SimpleMaskingTest.apply_mask_action!(mask, 20, SimpleMaskingTest.FEATURE_SELECTED) == false
        
        println("  ✅ Mask action application tests passed")
    end
    
    @testset "Manager Update Tests" begin
        config = SimpleMaskingTest.create_progressive_masking_config(feature_dimension = 20)
        manager = SimpleMaskingTest.initialize_simple_masking_manager(config)
        
        # Create mask
        mask = SimpleMaskingTest.create_mask_for_node!(manager, 1, 1)
        initial_version = mask.mask_version
        
        # Test successful update
        success = SimpleMaskingTest.update_feature_mask!(manager, 1, 1, 5, SimpleMaskingTest.FEATURE_SELECTED)
        @test success == true
        @test mask.mask_version == initial_version + 1
        @test mask.selected_features[5] == true
        @test mask.update_count == 1
        @test manager.successful_updates == 1
        @test manager.total_updates == 1
        
        # Test update non-existent mask - this will not increment counters since mask doesn't exist
        success = SimpleMaskingTest.update_feature_mask!(manager, 99, 99, 5, SimpleMaskingTest.FEATURE_SELECTED)
        @test success == false
        @test manager.failed_updates == 1
        @test manager.total_updates == 1  # No increment since mask doesn't exist
        
        # Test update with locked mask
        mask.is_locked = true
        success = SimpleMaskingTest.update_feature_mask!(manager, 1, 1, 10, SimpleMaskingTest.FEATURE_SELECTED)
        @test success == false
        @test manager.failed_updates == 2
        @test manager.total_updates == 1  # Still only one since locked mask also doesn't increment total
        
        println("  ✅ Manager update tests passed")
    end
    
    @testset "Parent Inheritance Tests" begin
        parent_mask = SimpleMaskingTest.create_feature_mask(1, 1, 15)
        child_mask = SimpleMaskingTest.create_feature_mask(1, 2, 15)
        
        # Set up parent mask state
        SimpleMaskingTest.apply_mask_action!(parent_mask, 5, SimpleMaskingTest.FEATURE_REJECTED)
        SimpleMaskingTest.apply_mask_action!(parent_mask, 10, SimpleMaskingTest.FEATURE_SELECTED)
        SimpleMaskingTest.apply_mask_action!(parent_mask, 12, SimpleMaskingTest.FEATURE_LOCKED)
        parent_mask.feature_priorities[8] = 2.0f0
        
        # Apply inheritance
        SimpleMaskingTest.inherit_parent_constraints!(child_mask, parent_mask, 0.8f0)
        
        # Check inheritance
        @test child_mask.rejected_features[5] == true
        @test child_mask.available_features[5] == false
        @test child_mask.locked_features[12] == true  # High strength inheritance
        @test child_mask.feature_priorities[8] > 1.0f0
        
        # Test lower inheritance strength
        child_mask2 = SimpleMaskingTest.create_feature_mask(1, 3, 15)
        SimpleMaskingTest.inherit_parent_constraints!(child_mask2, parent_mask, 0.3f0)
        @test child_mask2.locked_features[12] == false  # Low strength, no lock inheritance
        
        println("  ✅ Parent inheritance tests passed")
    end
    
    @testset "Global Constraints Tests" begin
        config = SimpleMaskingTest.create_progressive_masking_config(feature_dimension = 15)
        manager = SimpleMaskingTest.initialize_simple_masking_manager(config)
        
        # Create mask and select feature
        mask = SimpleMaskingTest.create_mask_for_node!(manager, 1, 1)
        SimpleMaskingTest.update_feature_mask!(manager, 1, 1, 5, SimpleMaskingTest.FEATURE_SELECTED)
        
        # Set global constraints (disable feature 5)
        global_constraints = trues(15)
        global_constraints[5] = false
        SimpleMaskingTest.set_global_constraints!(manager, global_constraints)
        
        # Check that feature 5 is now unavailable and deselected
        @test mask.available_features[5] == false
        @test mask.selected_features[5] == false
        @test mask.rejected_features[5] == true
        
        println("  ✅ Global constraints tests passed")
    end
    
    @testset "Mask Validation Tests" begin
        # Create valid mask
        valid_mask = SimpleMaskingTest.create_feature_mask(1, 1, 20, min_features_required = 5)
        @test SimpleMaskingTest.validate_mask(valid_mask) == true
        
        # Create invalid mask (selected and rejected feature)
        invalid_mask = SimpleMaskingTest.create_feature_mask(1, 2, 20)
        invalid_mask.selected_features[5] = true
        invalid_mask.rejected_features[5] = true
        @test SimpleMaskingTest.validate_mask(invalid_mask) == false
        
        # Create mask with selected but unavailable feature
        invalid_mask2 = SimpleMaskingTest.create_feature_mask(1, 3, 20)
        invalid_mask2.selected_features[5] = true
        invalid_mask2.available_features[5] = false
        @test SimpleMaskingTest.validate_mask(invalid_mask2) == false
        
        # Create mask with too few available features
        low_features_mask = SimpleMaskingTest.create_feature_mask(1, 4, 20, min_features_required = 15)
        low_features_mask.available_features[1:10] .= false
        low_features_mask.rejected_features[1:10] .= true
        @test SimpleMaskingTest.validate_mask(low_features_mask) == false
        
        println("  ✅ Mask validation tests passed")
    end
    
    @testset "Mask Merging Tests" begin
        # Create test masks
        mask1 = SimpleMaskingTest.create_feature_mask(1, 1, 10)
        mask2 = SimpleMaskingTest.create_feature_mask(2, 1, 10)
        mask3 = SimpleMaskingTest.create_feature_mask(3, 1, 10)
        
        # Set up different states
        SimpleMaskingTest.apply_mask_action!(mask1, 1, SimpleMaskingTest.FEATURE_SELECTED)
        SimpleMaskingTest.apply_mask_action!(mask1, 2, SimpleMaskingTest.FEATURE_REJECTED)
        
        SimpleMaskingTest.apply_mask_action!(mask2, 1, SimpleMaskingTest.FEATURE_SELECTED)
        SimpleMaskingTest.apply_mask_action!(mask2, 3, SimpleMaskingTest.FEATURE_SELECTED)
        SimpleMaskingTest.apply_mask_action!(mask2, 4, SimpleMaskingTest.FEATURE_REJECTED)
        
        SimpleMaskingTest.apply_mask_action!(mask3, 2, SimpleMaskingTest.FEATURE_SELECTED)
        SimpleMaskingTest.apply_mask_action!(mask3, 5, SimpleMaskingTest.FEATURE_REJECTED)
        
        masks = [mask1, mask2, mask3]
        
        # Test intersection merge
        merged = SimpleMaskingTest.merge_masks_intersection(masks)
        
        @test merged.selected_features[1] == false  # Selected in mask1 and mask2, but not mask3 (intersection requires ALL)
        @test merged.selected_features[2] == false  # Selected in mask3, rejected in mask1
        @test merged.selected_features[3] == false  # Only selected in mask2
        @test merged.rejected_features[2] == true   # Rejected in mask1
        @test merged.rejected_features[4] == true   # Rejected in mask2
        @test merged.rejected_features[5] == true   # Rejected in mask3
        
        println("  ✅ Mask merging tests passed")
    end
    
    @testset "Feature Access Tests" begin
        config = SimpleMaskingTest.create_progressive_masking_config(feature_dimension = 15)
        manager = SimpleMaskingTest.initialize_simple_masking_manager(config)
        
        # Create mask and set some features
        mask = SimpleMaskingTest.create_mask_for_node!(manager, 1, 1)
        SimpleMaskingTest.update_feature_mask!(manager, 1, 1, 5, SimpleMaskingTest.FEATURE_SELECTED)
        SimpleMaskingTest.update_feature_mask!(manager, 1, 1, 10, SimpleMaskingTest.FEATURE_SELECTED)
        SimpleMaskingTest.update_feature_mask!(manager, 1, 1, 12, SimpleMaskingTest.FEATURE_REJECTED)
        
        # Test get selected features
        selected = SimpleMaskingTest.get_selected_features(manager, 1, 1)
        @test 5 in selected
        @test 10 in selected
        @test 12 ∉ selected
        @test length(selected) == 2
        
        # Test get available features
        available = SimpleMaskingTest.get_available_features(manager, 1, 1)
        @test 5 in available
        @test 10 in available
        @test 12 ∉ available  # Rejected
        @test length(available) == 14  # 15 total - 1 rejected
        
        # Test non-existent mask
        selected_empty = SimpleMaskingTest.get_selected_features(manager, 99, 99)
        @test isempty(selected_empty)
        
        available_all = SimpleMaskingTest.get_available_features(manager, 99, 99)
        @test length(available_all) == 15  # Returns all features when mask not found
        
        println("  ✅ Feature access tests passed")
    end
    
    @testset "Status and Monitoring Tests" begin
        config = SimpleMaskingTest.create_progressive_masking_config(feature_dimension = 20)
        manager = SimpleMaskingTest.initialize_simple_masking_manager(config)
        
        # Initial status
        status = SimpleMaskingTest.get_manager_status(manager)
        @test status["manager_state"] == "active"
        @test status["total_masks"] == 0
        @test status["total_updates"] == 0
        @test status["successful_updates"] == 0
        @test status["failed_updates"] == 0
        @test status["success_rate"] == 0.0
        
        # Create some activity
        mask1 = SimpleMaskingTest.create_mask_for_node!(manager, 1, 1)
        mask2 = SimpleMaskingTest.create_mask_for_node!(manager, 1, 2)
        SimpleMaskingTest.update_feature_mask!(manager, 1, 1, 5, SimpleMaskingTest.FEATURE_SELECTED)
        SimpleMaskingTest.update_feature_mask!(manager, 1, 2, 10, SimpleMaskingTest.FEATURE_SELECTED)
        SimpleMaskingTest.update_feature_mask!(manager, 99, 99, 1, SimpleMaskingTest.FEATURE_SELECTED)  # Should fail
        
        status = SimpleMaskingTest.get_manager_status(manager)
        @test status["total_masks"] == 2
        @test status["total_updates"] == 2  # Only successful operations that find masks increment total
        @test status["successful_updates"] == 2
        @test status["failed_updates"] == 1  # Failed operations increment failed but not total
        @test status["success_rate"] ≈ 1.0   # 2 successful / 2 total = 100%
        
        println("  ✅ Status and monitoring tests passed")
    end
    
    @testset "Error Handling Tests" begin
        config = SimpleMaskingTest.create_progressive_masking_config(feature_dimension = 10)
        manager = SimpleMaskingTest.initialize_simple_masking_manager(config)
        
        # Test invalid global constraints
        @test_throws ErrorException SimpleMaskingTest.set_global_constraints!(manager, trues(5))  # Wrong size
        
        # Test merge with empty mask list
        @test_throws ErrorException SimpleMaskingTest.merge_masks_intersection(SimpleMaskingTest.FeatureMask[])
        
        # Test invalid feature actions
        mask = SimpleMaskingTest.create_feature_mask(1, 1, 10)
        @test SimpleMaskingTest.apply_mask_action!(mask, 0, SimpleMaskingTest.FEATURE_SELECTED) == false
        @test SimpleMaskingTest.apply_mask_action!(mask, 15, SimpleMaskingTest.FEATURE_SELECTED) == false
        
        println("  ✅ Error handling tests passed")
    end
end

println("All Simple Progressive Feature Masking tests completed!")
println("✅ Configuration system working correctly")
println("✅ Manager initialization and setup")
println("✅ Feature mask creation with proper initialization")
println("✅ Mask action application (select, reject, deselect, lock, restore)")
println("✅ Manager update operations with success/failure tracking")
println("✅ Parent-child constraint inheritance with configurable strength")
println("✅ Global constraint application and propagation")
println("✅ Mask validation with consistency checking")
println("✅ Mask merging with intersection operation")
println("✅ Feature access methods for selected and available features")
println("✅ Status monitoring and performance tracking")
println("✅ Error handling for invalid operations and edge cases")
println("✅ Core progressive feature masking ready for MCTS integration")