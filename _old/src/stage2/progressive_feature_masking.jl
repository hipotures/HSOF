"""
Progressive Feature Masking System for MCTS Ensemble Feature Selection
Implements dynamic feature masking as selections are made or rejected during MCTS exploration,
ensuring efficient feature space constraint propagation across tree nodes with inheritance,
merging, and validation mechanisms for optimal ensemble diversity.

This module provides tree-specific feature masking with parent-child constraint inheritance,
multi-tree constraint merging, and validation ensuring minimum feature availability.
"""

module ProgressiveFeatureMasking

using Random
using Statistics
using Dates
using Printf
using LinearAlgebra

# Import ensemble forest for tree management
include("ensemble_forest.jl")
using .EnsembleForest

# Import metamodel integration for evaluation
include("metamodel_integration.jl")
using .MetamodelIntegration

"""
Feature selection action types for mask updates
"""
@enum FeatureAction begin
    FEATURE_SELECTED = 1       # Feature added to current selection
    FEATURE_REJECTED = 2       # Feature permanently rejected
    FEATURE_DESELECTED = 3     # Feature removed from selection (still available)
    FEATURE_LOCKED = 4         # Feature permanently locked in selection
    FEATURE_RESTORED = 5       # Feature restored to available state
end

"""
Mask operation types for different masking strategies
"""
@enum MaskOperation begin
    INTERSECTION = 1           # Combine masks using AND operation
    UNION = 2                 # Combine masks using OR operation
    WEIGHTED_MERGE = 3        # Combine masks using weighted averaging
    PRIORITY_MERGE = 4        # Combine masks based on priority ordering
end

"""
Feature mask validation levels
"""
@enum ValidationLevel begin
    STRICT = 1                # Enforce all validation rules strictly
    MODERATE = 2              # Allow some flexibility in validation
    PERMISSIVE = 3            # Minimal validation for performance
end

"""
Feature mask state for individual tree nodes
"""
mutable struct FeatureMask
    tree_id::Int                    # Tree this mask belongs to
    node_id::Int                    # Node this mask applies to
    feature_dimension::Int          # Total number of features
    
    # Core mask data
    available_features::BitVector   # Features available for selection
    selected_features::BitVector    # Features currently selected
    rejected_features::BitVector    # Features permanently rejected
    locked_features::BitVector      # Features permanently locked in selection
    
    # Metadata
    mask_version::Int              # Version number for change tracking
    creation_time::DateTime        # When mask was created
    last_update::DateTime          # Last modification time
    parent_mask_id::Union{Int, Nothing}  # Parent mask reference
    
    # Constraint tracking
    min_features_required::Int     # Minimum features that must remain available
    max_features_allowed::Int      # Maximum features that can be selected
    feature_priorities::Vector{Float32}  # Priority weights for features
    
    # Performance tracking
    update_count::Int             # Number of mask updates
    validation_failures::Int     # Number of validation failures
    
    # State flags
    is_valid::Bool               # Whether mask passes validation
    is_locked::Bool              # Whether mask is locked from changes
    requires_validation::Bool    # Whether mask needs validation
end

"""
Create new feature mask
"""
function create_feature_mask(tree_id::Int, 
                            node_id::Int, 
                            feature_dimension::Int;
                            min_features_required::Int = 10,
                            max_features_allowed::Int = feature_dimension,
                            parent_mask_id::Union{Int, Nothing} = nothing)
    
    return FeatureMask(
        tree_id, node_id, feature_dimension,
        # All features initially available
        trues(feature_dimension),
        # No features initially selected
        falses(feature_dimension),
        # No features initially rejected
        falses(feature_dimension),
        # No features initially locked
        falses(feature_dimension),
        1,  # Initial version
        now(), now(),
        parent_mask_id,
        min_features_required,
        max_features_allowed,
        ones(Float32, feature_dimension),  # Equal priority initially
        0, 0,  # No updates or failures initially
        true, false, false  # Valid, not locked, no validation needed
    )
end

"""
Feature mask update event for tracking changes
"""
struct MaskUpdateEvent
    mask_id::Int
    feature_id::Int
    action::FeatureAction
    timestamp::DateTime
    success::Bool
    error_message::Union{String, Nothing}
end

"""
Mask merge configuration
"""
struct MaskMergeConfig
    operation::MaskOperation
    weights::Vector{Float32}       # Weights for weighted merge
    priority_order::Vector{Int}    # Priority order for masks
    inheritance_factor::Float32    # Factor for parent-child inheritance
    conflict_resolution::String    # How to resolve conflicts ("strict", "permissive", "weighted")
    preserve_locked_features::Bool # Whether to preserve locked features
end

"""
Create default mask merge configuration
"""
function create_mask_merge_config(;
    operation::MaskOperation = INTERSECTION,
    weights::Vector{Float32} = Float32[],
    priority_order::Vector{Int} = Int[],
    inheritance_factor::Float32 = 0.8f0,
    conflict_resolution::String = "strict",
    preserve_locked_features::Bool = true
)
    return MaskMergeConfig(
        operation, weights, priority_order,
        inheritance_factor, conflict_resolution,
        preserve_locked_features
    )
end

"""
Progressive feature masking configuration
"""
struct ProgressiveMaskingConfig
    # Feature management
    feature_dimension::Int
    min_features_per_tree::Int
    max_features_per_tree::Int
    
    # Mask inheritance
    enable_parent_inheritance::Bool
    inheritance_strength::Float32
    max_inheritance_depth::Int
    
    # Validation settings
    validation_level::ValidationLevel
    enable_strict_validation::Bool
    min_available_features::Int
    
    # Performance settings
    enable_mask_caching::Bool
    cache_size::Int
    enable_incremental_updates::Bool
    
    # Merge settings
    default_merge_operation::MaskOperation
    enable_priority_weighting::Bool
    priority_decay_rate::Float32
    
    # Constraint enforcement
    enforce_feature_limits::Bool
    allow_temporary_violations::Bool
    violation_recovery_attempts::Int
    
    # Monitoring
    enable_mask_monitoring::Bool
    log_mask_changes::Bool
    track_performance_metrics::Bool
end

"""
Create default progressive masking configuration
"""
function create_progressive_masking_config(;
    feature_dimension::Int = 500,
    min_features_per_tree::Int = 10,
    max_features_per_tree::Int = 100,
    enable_parent_inheritance::Bool = true,
    inheritance_strength::Float32 = 0.7f0,
    max_inheritance_depth::Int = 10,
    validation_level::ValidationLevel = MODERATE,
    enable_strict_validation::Bool = true,
    min_available_features::Int = 20,
    enable_mask_caching::Bool = true,
    cache_size::Int = 1000,
    enable_incremental_updates::Bool = true,
    default_merge_operation::MaskOperation = INTERSECTION,
    enable_priority_weighting::Bool = true,
    priority_decay_rate::Float32 = 0.95f0,
    enforce_feature_limits::Bool = true,
    allow_temporary_violations::Bool = false,
    violation_recovery_attempts::Int = 3,
    enable_mask_monitoring::Bool = true,
    log_mask_changes::Bool = true,
    track_performance_metrics::Bool = true
)
    return ProgressiveMaskingConfig(
        feature_dimension, min_features_per_tree, max_features_per_tree,
        enable_parent_inheritance, inheritance_strength, max_inheritance_depth,
        validation_level, enable_strict_validation, min_available_features,
        enable_mask_caching, cache_size, enable_incremental_updates,
        default_merge_operation, enable_priority_weighting, priority_decay_rate,
        enforce_feature_limits, allow_temporary_violations, violation_recovery_attempts,
        enable_mask_monitoring, log_mask_changes, track_performance_metrics
    )
end

"""
Mask performance statistics
"""
mutable struct MaskPerformanceStats
    total_mask_updates::Int
    successful_updates::Int
    failed_updates::Int
    validation_checks::Int
    validation_failures::Int
    cache_hits::Int
    cache_misses::Int
    merge_operations::Int
    inheritance_operations::Int
    average_update_time::Float64
    average_validation_time::Float64
    last_update_time::DateTime
end

"""
Initialize mask performance statistics
"""
function initialize_mask_stats()
    return MaskPerformanceStats(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, now()
    )
end

"""
Progressive feature masking manager
Coordinates mask operations across all trees and nodes
"""
mutable struct ProgressiveMaskingManager
    # Configuration
    config::ProgressiveMaskingConfig
    
    # Mask storage
    masks::Dict{Tuple{Int, Int}, FeatureMask}  # (tree_id, node_id) -> mask
    mask_cache::Dict{Vector{Int}, FeatureMask}  # Feature subset -> cached mask
    
    # Tree management
    tree_masks::Dict{Int, Vector{FeatureMask}}  # tree_id -> masks for that tree
    global_constraints::BitVector               # Global feature constraints
    
    # Update tracking
    update_history::Vector{MaskUpdateEvent}
    mask_versions::Dict{Tuple{Int, Int}, Int}  # Track mask versions
    
    # Performance monitoring
    stats::MaskPerformanceStats
    update_times::Vector{Float64}
    validation_times::Vector{Float64}
    
    # Synchronization
    mask_lock::ReentrantLock
    cache_lock::ReentrantLock
    
    # Status
    manager_state::String
    error_log::Vector{String}
    creation_time::DateTime
end

"""
Initialize progressive masking manager
"""
function initialize_progressive_masking_manager(config::ProgressiveMaskingConfig = create_progressive_masking_config())
    # Initialize global constraints (all features available initially)
    global_constraints = trues(config.feature_dimension)
    
    manager = ProgressiveMaskingManager(
        config,
        Dict{Tuple{Int, Int}, FeatureMask}(),
        Dict{Vector{Int}, FeatureMask}(),
        Dict{Int, Vector{FeatureMask}}(),
        global_constraints,
        MaskUpdateEvent[],
        Dict{Tuple{Int, Int}, Int}(),
        initialize_mask_stats(),
        Float64[],
        Float64[],
        ReentrantLock(),
        ReentrantLock(),
        "active",
        String[],
        now()
    )
    
    return manager
end

"""
Create feature mask for tree node
"""
function create_mask_for_node!(manager::ProgressiveMaskingManager, 
                              tree_id::Int, 
                              node_id::Int,
                              parent_node_id::Union{Int, Nothing} = nothing)::FeatureMask
    lock(manager.mask_lock) do
        # Check if mask already exists
        mask_key = (tree_id, node_id)
        if haskey(manager.masks, mask_key)
            return manager.masks[mask_key]
        end
        
        # Create new mask
        parent_mask_id = isnothing(parent_node_id) ? nothing : parent_node_id
        mask = create_feature_mask(
            tree_id, node_id, manager.config.feature_dimension;
            min_features_required = manager.config.min_features_per_tree,
            max_features_allowed = manager.config.max_features_per_tree,
            parent_mask_id = parent_mask_id
        )
        
        # Apply parent inheritance if enabled
        if manager.config.enable_parent_inheritance && !isnothing(parent_node_id)
            parent_mask = get(manager.masks, (tree_id, parent_node_id), nothing)
            if !isnothing(parent_mask)
                inherit_parent_constraints!(mask, parent_mask, manager.config.inheritance_strength)
            end
        end
        
        # Apply global constraints
        apply_global_constraints!(mask, manager.global_constraints)
        
        # Store mask
        manager.masks[mask_key] = mask
        manager.mask_versions[mask_key] = mask.mask_version
        
        # Add to tree masks
        if !haskey(manager.tree_masks, tree_id)
            manager.tree_masks[tree_id] = FeatureMask[]
        end
        push!(manager.tree_masks[tree_id], mask)
        
        # Validate mask
        if manager.config.enable_strict_validation
            validate_mask!(manager, mask)
        end
        
        @info "Created mask for tree $tree_id, node $node_id"
        return mask
    end
end

"""
Update feature mask with new action
"""
function update_feature_mask!(manager::ProgressiveMaskingManager,
                             tree_id::Int,
                             node_id::Int,
                             feature_id::Int,
                             action::FeatureAction)::Bool
    start_time = time()
    
    lock(manager.mask_lock) do
        mask_key = (tree_id, node_id)
        mask = get(manager.masks, mask_key, nothing)
        
        if isnothing(mask)
            error_msg = "Mask not found for tree $tree_id, node $node_id"
            push!(manager.error_log, error_msg)
            record_update_event!(manager, 0, feature_id, action, false, error_msg)
            return false
        end
        
        if mask.is_locked
            error_msg = "Mask is locked for tree $tree_id, node $node_id"
            push!(manager.error_log, error_msg)
            record_update_event!(manager, mask_key[1], feature_id, action, false, error_msg)
            return false
        end
        
        # Perform the update
        success = false
        error_message = nothing
        
        try
            success = apply_mask_action!(mask, feature_id, action)
            
            if success
                # Update version and timestamps
                mask.mask_version += 1
                mask.last_update = now()
                mask.update_count += 1
                mask.requires_validation = true
                
                # Update version tracking
                manager.mask_versions[mask_key] = mask.mask_version
                
                # Validate if required
                if manager.config.enable_strict_validation
                    validate_mask!(manager, mask)
                end
                
                # Update cache
                if manager.config.enable_mask_caching
                    update_mask_cache!(manager, mask)
                end
                
                manager.stats.successful_updates += 1
                
            else
                error_message = "Failed to apply action $action to feature $feature_id"
                mask.validation_failures += 1
                manager.stats.failed_updates += 1
            end
            
        catch e
            success = false
            error_message = "Exception during mask update: $e"
            push!(manager.error_log, error_message)
            manager.stats.failed_updates += 1
        end
        
        # Record update event
        record_update_event!(manager, mask_key[1], feature_id, action, success, error_message)
        
        # Update performance stats
        update_time = (time() - start_time) * 1000  # Convert to ms
        push!(manager.update_times, update_time)
        if length(manager.update_times) > 1000
            deleteat!(manager.update_times, 1)
        end
        manager.stats.average_update_time = mean(manager.update_times)
        manager.stats.total_mask_updates += 1
        manager.stats.last_update_time = now()
        
        return success
    end
end

"""
Apply mask action to feature
"""
function apply_mask_action!(mask::FeatureMask, feature_id::Int, action::FeatureAction)::Bool
    # Validate feature ID
    if feature_id < 1 || feature_id > mask.feature_dimension
        return false
    end
    
    # Check if feature is locked
    if mask.locked_features[feature_id] && action != FEATURE_LOCKED
        return false
    end
    
    # Apply action
    success = true
    
    if action == FEATURE_SELECTED
        # Select feature if available
        if mask.available_features[feature_id] && !mask.rejected_features[feature_id]
            mask.selected_features[feature_id] = true
        else
            success = false
        end
        
    elseif action == FEATURE_REJECTED
        # Reject feature permanently
        if !mask.locked_features[feature_id]
            mask.rejected_features[feature_id] = true
            mask.available_features[feature_id] = false
            mask.selected_features[feature_id] = false
        else
            success = false
        end
        
    elseif action == FEATURE_DESELECTED
        # Deselect feature but keep available
        if !mask.locked_features[feature_id]
            mask.selected_features[feature_id] = false
        else
            success = false
        end
        
    elseif action == FEATURE_LOCKED
        # Lock feature in current state
        mask.locked_features[feature_id] = true
        
    elseif action == FEATURE_RESTORED
        # Restore feature to available state
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

"""
Inherit constraints from parent mask
"""
function inherit_parent_constraints!(child_mask::FeatureMask, 
                                   parent_mask::FeatureMask,
                                   inheritance_strength::Float32)
    # Inherit available features (intersection with parent)
    child_mask.available_features .&= parent_mask.available_features
    
    # Inherit rejected features (union with parent)
    child_mask.rejected_features .|= parent_mask.rejected_features
    
    # Inherit locked features with strength factor
    if inheritance_strength >= 0.5f0
        child_mask.locked_features .|= parent_mask.locked_features
    end
    
    # Inherit feature priorities with decay
    for i in 1:length(child_mask.feature_priorities)
        child_mask.feature_priorities[i] = 
            (child_mask.feature_priorities[i] + parent_mask.feature_priorities[i] * inheritance_strength) / 2
    end
    
    # Update constraints
    child_mask.available_features .&= .!child_mask.rejected_features
    child_mask.selected_features .&= child_mask.available_features
    
    @info "Inherited constraints from parent mask (strength: $inheritance_strength)"
end

"""
Apply global constraints to mask
"""
function apply_global_constraints!(mask::FeatureMask, global_constraints::BitVector)
    # Apply global constraints to available features
    mask.available_features .&= global_constraints
    
    # Ensure selected features are still available
    mask.selected_features .&= mask.available_features
    
    # Mark globally unavailable features as rejected
    mask.rejected_features .|= .!global_constraints
    
    @info "Applied global constraints to mask"
end

"""
Merge multiple masks using specified operation
"""
function merge_masks(masks::Vector{FeatureMask}, 
                    merge_config::MaskMergeConfig)::FeatureMask
    if isempty(masks)
        error("Cannot merge empty mask list")
    end
    
    if length(masks) == 1
        return masks[1]
    end
    
    # Create result mask based on first mask
    result_mask = deepcopy(masks[1])
    result_mask.tree_id = -1  # Mark as merged mask
    result_mask.node_id = -1
    result_mask.mask_version = 1
    result_mask.creation_time = now()
    result_mask.last_update = now()
    
    # Apply merge operation
    if merge_config.operation == INTERSECTION
        merge_intersection!(result_mask, masks)
    elseif merge_config.operation == UNION
        merge_union!(result_mask, masks)
    elseif merge_config.operation == WEIGHTED_MERGE
        merge_weighted!(result_mask, masks, merge_config.weights)
    elseif merge_config.operation == PRIORITY_MERGE
        merge_priority!(result_mask, masks, merge_config.priority_order)
    end
    
    # Preserve locked features if specified
    if merge_config.preserve_locked_features
        for mask in masks
            result_mask.locked_features .|= mask.locked_features
        end
    end
    
    # Ensure consistency
    result_mask.available_features .&= .!result_mask.rejected_features
    result_mask.selected_features .&= result_mask.available_features
    
    return result_mask
end

"""
Merge masks using intersection (most restrictive)
"""
function merge_intersection!(result_mask::FeatureMask, masks::Vector{FeatureMask})
    for mask in masks[2:end]
        result_mask.available_features .&= mask.available_features
        result_mask.rejected_features .|= mask.rejected_features
        result_mask.locked_features .|= mask.locked_features
    end
    
    # Update selected features to be intersection
    for mask in masks
        result_mask.selected_features .&= mask.selected_features
    end
end

"""
Merge masks using union (least restrictive)
"""
function merge_union!(result_mask::FeatureMask, masks::Vector{FeatureMask})
    for mask in masks[2:end]
        result_mask.available_features .|= mask.available_features
        result_mask.rejected_features .&= mask.rejected_features
        result_mask.locked_features .&= mask.locked_features
    end
    
    # Update selected features to be union
    for mask in masks
        result_mask.selected_features .|= mask.selected_features
    end
end

"""
Merge masks using weighted combination
"""
function merge_weighted!(result_mask::FeatureMask, masks::Vector{FeatureMask}, weights::Vector{Float32})
    if length(weights) != length(masks)
        error("Weights length must match masks length")
    end
    
    # Normalize weights
    normalized_weights = weights ./ sum(weights)
    
    # Reset result mask
    result_mask.available_features = falses(result_mask.feature_dimension)
    result_mask.selected_features = falses(result_mask.feature_dimension)
    result_mask.rejected_features = falses(result_mask.feature_dimension)
    
    # Weighted combination
    for i in 1:result_mask.feature_dimension
        available_weight = sum(normalized_weights[j] * masks[j].available_features[i] for j in 1:length(masks))
        selected_weight = sum(normalized_weights[j] * masks[j].selected_features[i] for j in 1:length(masks))
        rejected_weight = sum(normalized_weights[j] * masks[j].rejected_features[i] for j in 1:length(masks))
        
        # Apply thresholds
        result_mask.available_features[i] = available_weight > 0.5
        result_mask.selected_features[i] = selected_weight > 0.5
        result_mask.rejected_features[i] = rejected_weight > 0.5
    end
    
    # Update priorities
    for i in 1:length(result_mask.feature_priorities)
        result_mask.feature_priorities[i] = 
            sum(normalized_weights[j] * masks[j].feature_priorities[i] for j in 1:length(masks))
    end
end

"""
Merge masks using priority order
"""
function merge_priority!(result_mask::FeatureMask, masks::Vector{FeatureMask}, priority_order::Vector{Int})
    if length(priority_order) != length(masks)
        error("Priority order length must match masks length")
    end
    
    # Sort masks by priority
    sorted_indices = sortperm(priority_order)
    
    # Apply masks in priority order (higher priority overwrites lower)
    for idx in sorted_indices
        mask = masks[idx]
        
        # Higher priority mask constraints override lower priority
        result_mask.available_features .|= mask.available_features
        result_mask.rejected_features .|= mask.rejected_features
        result_mask.selected_features .|= mask.selected_features
        result_mask.locked_features .|= mask.locked_features
    end
    
    # Ensure consistency
    result_mask.available_features .&= .!result_mask.rejected_features
    result_mask.selected_features .&= result_mask.available_features
end

"""
Validate feature mask
"""
function validate_mask!(manager::ProgressiveMaskingManager, mask::FeatureMask)::Bool
    start_time = time()
    
    validation_passed = true
    error_messages = String[]
    
    try
        # Check basic consistency
        if any(mask.selected_features .& mask.rejected_features)
            validation_passed = false
            push!(error_messages, "Selected features overlap with rejected features")
        end
        
        if any(mask.selected_features .& .!mask.available_features)
            validation_passed = false
            push!(error_messages, "Selected features not in available features")
        end
        
        if any(mask.available_features .& mask.rejected_features)
            validation_passed = false
            push!(error_messages, "Available features overlap with rejected features")
        end
        
        # Check minimum features requirement
        available_count = sum(mask.available_features)
        if available_count < mask.min_features_required
            if manager.config.validation_level == STRICT
                validation_passed = false
                push!(error_messages, "Available features ($available_count) below minimum ($(mask.min_features_required))")
            elseif manager.config.validation_level == MODERATE
                @warn "Available features ($available_count) below minimum ($(mask.min_features_required))"
            end
        end
        
        # Check maximum features constraint
        selected_count = sum(mask.selected_features)
        if selected_count > mask.max_features_allowed
            validation_passed = false
            push!(error_messages, "Selected features ($selected_count) exceeds maximum ($(mask.max_features_allowed))")
        end
        
        # Check locked features consistency
        if any(mask.locked_features .& mask.rejected_features)
            validation_passed = false
            push!(error_messages, "Locked features overlap with rejected features")
        end
        
        # Update mask state
        mask.is_valid = validation_passed
        mask.requires_validation = false
        
        if !validation_passed
            mask.validation_failures += 1
            for msg in error_messages
                push!(manager.error_log, msg)
            end
        end
        
    catch e
        validation_passed = false
        error_msg = "Validation exception: $e"
        push!(manager.error_log, error_msg)
        mask.validation_failures += 1
    end
    
    # Update performance stats
    validation_time = (time() - start_time) * 1000  # Convert to ms
    push!(manager.validation_times, validation_time)
    if length(manager.validation_times) > 1000
        deleteat!(manager.validation_times, 1)
    end
    manager.stats.average_validation_time = mean(manager.validation_times)
    manager.stats.validation_checks += 1
    
    if !validation_passed
        manager.stats.validation_failures += 1
    end
    
    return validation_passed
end

"""
Record mask update event
"""
function record_update_event!(manager::ProgressiveMaskingManager,
                             mask_id::Int,
                             feature_id::Int,
                             action::FeatureAction,
                             success::Bool,
                             error_message::Union{String, Nothing})
    if manager.config.log_mask_changes
        event = MaskUpdateEvent(
            mask_id, feature_id, action, now(), success, error_message
        )
        
        push!(manager.update_history, event)
        
        # Limit history size
        if length(manager.update_history) > 10000
            deleteat!(manager.update_history, 1)
        end
    end
end

"""
Update mask cache
"""
function update_mask_cache!(manager::ProgressiveMaskingManager, mask::FeatureMask)
    lock(manager.cache_lock) do
        # Create cache key from selected features
        selected_indices = findall(mask.selected_features)
        
        # Check cache size limit
        if length(manager.mask_cache) >= manager.config.cache_size
            # Remove oldest entry (simple FIFO)
            first_key = first(keys(manager.mask_cache))
            delete!(manager.mask_cache, first_key)
        end
        
        # Cache the mask
        manager.mask_cache[selected_indices] = deepcopy(mask)
    end
end

"""
Get mask from cache
"""
function get_cached_mask(manager::ProgressiveMaskingManager, selected_features::Vector{Int})::Union{FeatureMask, Nothing}
    lock(manager.cache_lock) do
        cached_mask = get(manager.mask_cache, selected_features, nothing)
        
        if !isnothing(cached_mask)
            manager.stats.cache_hits += 1
            return cached_mask
        else
            manager.stats.cache_misses += 1
            return nothing
        end
    end
end

"""
Get available features for tree node
"""
function get_available_features(manager::ProgressiveMaskingManager, 
                               tree_id::Int, 
                               node_id::Int)::Vector{Int}
    mask_key = (tree_id, node_id)
    mask = get(manager.masks, mask_key, nothing)
    
    if isnothing(mask)
        @warn "Mask not found for tree $tree_id, node $node_id"
        return collect(1:manager.config.feature_dimension)
    end
    
    return findall(mask.available_features)
end

"""
Get selected features for tree node
"""
function get_selected_features(manager::ProgressiveMaskingManager,
                              tree_id::Int,
                              node_id::Int)::Vector{Int}
    mask_key = (tree_id, node_id)
    mask = get(manager.masks, mask_key, nothing)
    
    if isnothing(mask)
        @warn "Mask not found for tree $tree_id, node $node_id"
        return Int[]
    end
    
    return findall(mask.selected_features)
end

"""
Set global feature constraints
"""
function set_global_constraints!(manager::ProgressiveMaskingManager, constraints::BitVector)
    if length(constraints) != manager.config.feature_dimension
        error("Constraints length must match feature dimension")
    end
    
    lock(manager.mask_lock) do
        manager.global_constraints = copy(constraints)
        
        # Apply to all existing masks
        for mask in values(manager.masks)
            apply_global_constraints!(mask, constraints)
            
            if manager.config.enable_strict_validation
                validate_mask!(manager, mask)
            end
        end
    end
    
    @info "Updated global constraints"
end

"""
Get masking manager status
"""
function get_masking_status(manager::ProgressiveMaskingManager)
    lock(manager.mask_lock) do
        return Dict{String, Any}(
            "manager_state" => manager.manager_state,
            "total_masks" => length(manager.masks),
            "active_trees" => length(manager.tree_masks),
            "total_updates" => manager.stats.total_mask_updates,
            "successful_updates" => manager.stats.successful_updates,
            "failed_updates" => manager.stats.failed_updates,
            "validation_checks" => manager.stats.validation_checks,
            "validation_failures" => manager.stats.validation_failures,
            "cache_hits" => manager.stats.cache_hits,
            "cache_misses" => manager.stats.cache_misses,
            "cache_hit_rate" => manager.stats.cache_hits > 0 ? 
                manager.stats.cache_hits / (manager.stats.cache_hits + manager.stats.cache_misses) : 0.0,
            "average_update_time_ms" => manager.stats.average_update_time,
            "average_validation_time_ms" => manager.stats.average_validation_time,
            "cache_size" => length(manager.mask_cache),
            "max_cache_size" => manager.config.cache_size,
            "update_history_size" => length(manager.update_history),
            "error_count" => length(manager.error_log),
            "last_update" => manager.stats.last_update_time
        )
    end
end

"""
Generate masking performance report
"""
function generate_masking_report(manager::ProgressiveMaskingManager)
    status = get_masking_status(manager)
    
    report = String[]
    
    push!(report, "=== Progressive Feature Masking Report ===")
    push!(report, "Generated: $(now())")
    push!(report, "Manager State: $(status["manager_state"])")
    push!(report, "")
    
    # Overview
    push!(report, "Overview:")
    push!(report, "  Total Masks: $(status["total_masks"])")
    push!(report, "  Active Trees: $(status["active_trees"])")
    push!(report, "  Feature Dimension: $(manager.config.feature_dimension)")
    push!(report, "  Min Features/Tree: $(manager.config.min_features_per_tree)")
    push!(report, "  Max Features/Tree: $(manager.config.max_features_per_tree)")
    push!(report, "")
    
    # Performance metrics
    push!(report, "Performance Metrics:")
    push!(report, "  Total Updates: $(status["total_updates"])")
    push!(report, "  Successful Updates: $(status["successful_updates"])")
    push!(report, "  Failed Updates: $(status["failed_updates"])")
    push!(report, "  Success Rate: $(round(status["successful_updates"] / max(1, status["total_updates"]) * 100, digits=1))%")
    push!(report, "  Average Update Time: $(round(status["average_update_time_ms"], digits=2))ms")
    push!(report, "")
    
    # Validation statistics
    push!(report, "Validation Statistics:")
    push!(report, "  Validation Checks: $(status["validation_checks"])")
    push!(report, "  Validation Failures: $(status["validation_failures"])")
    push!(report, "  Validation Success Rate: $(round((status["validation_checks"] - status["validation_failures"]) / max(1, status["validation_checks"]) * 100, digits=1))%")
    push!(report, "  Average Validation Time: $(round(status["average_validation_time_ms"], digits=2))ms")
    push!(report, "")
    
    # Cache performance
    push!(report, "Cache Performance:")
    push!(report, "  Cache Size: $(status["cache_size"])/$(status["max_cache_size"])")
    push!(report, "  Cache Hits: $(status["cache_hits"])")
    push!(report, "  Cache Misses: $(status["cache_misses"])")
    push!(report, "  Cache Hit Rate: $(round(status["cache_hit_rate"] * 100, digits=1))%")
    push!(report, "")
    
    # Configuration
    push!(report, "Configuration:")
    push!(report, "  Parent Inheritance: $(manager.config.enable_parent_inheritance)")
    push!(report, "  Inheritance Strength: $(manager.config.inheritance_strength)")
    push!(report, "  Validation Level: $(manager.config.validation_level)")
    push!(report, "  Strict Validation: $(manager.config.enable_strict_validation)")
    push!(report, "  Mask Caching: $(manager.config.enable_mask_caching)")
    push!(report, "  Incremental Updates: $(manager.config.enable_incremental_updates)")
    push!(report, "")
    
    # Error reporting
    if status["error_count"] > 0
        push!(report, "Recent Errors:")
        recent_errors = manager.error_log[max(1, length(manager.error_log)-3):end]
        for error in recent_errors
            push!(report, "  - $error")
        end
        push!(report, "")
    end
    
    push!(report, "=== End Masking Report ===")
    
    return join(report, "\n")
end

"""
Cleanup masking manager
"""
function cleanup_masking_manager!(manager::ProgressiveMaskingManager)
    lock(manager.mask_lock) do
        empty!(manager.masks)
        empty!(manager.tree_masks)
        empty!(manager.update_history)
        empty!(manager.mask_versions)
    end
    
    lock(manager.cache_lock) do
        empty!(manager.mask_cache)
    end
    
    manager.manager_state = "shutdown"
    @info "Progressive masking manager cleaned up"
end

# Export main types and functions
export FeatureAction, MaskOperation, ValidationLevel
export FEATURE_SELECTED, FEATURE_REJECTED, FEATURE_DESELECTED, FEATURE_LOCKED, FEATURE_RESTORED
export INTERSECTION, UNION, WEIGHTED_MERGE, PRIORITY_MERGE
export STRICT, MODERATE, PERMISSIVE
export FeatureMask, MaskMergeConfig, ProgressiveMaskingConfig, ProgressiveMaskingManager
export create_feature_mask, create_mask_merge_config, create_progressive_masking_config
export initialize_progressive_masking_manager, create_mask_for_node!
export update_feature_mask!, merge_masks, validate_mask!
export get_available_features, get_selected_features, set_global_constraints!
export get_masking_status, generate_masking_report, cleanup_masking_manager!

end # module ProgressiveFeatureMasking