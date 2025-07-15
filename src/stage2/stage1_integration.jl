"""
Stage 1 Output Integration and Feature Indexing System
Provides interface layer to receive 500-feature output from Stage 1 and builds efficient indexing
system for MCTS tree operations in Stage 2.

This module handles the critical transition between Stage 1 feature selection and Stage 2 MCTS,
maintaining feature metadata, creating efficient lookup systems, and ensuring data integrity.
"""

module Stage1Integration

using Random
using Statistics
using JSON3
using HDF5
using JLD2
using Dates

"""
Feature metadata structure from Stage 1
Contains all information about a selected feature
"""
struct FeatureMetadata
    original_id::Int                     # Original feature ID in full dataset
    stage1_rank::Int                     # Ranking position in Stage 1 selection (1-500)
    name::String                         # Feature name/identifier
    feature_type::String                 # "numeric", "categorical", "binary", etc.
    importance_score::Float64            # Importance score from Stage 1 (0.0-1.0)
    selection_confidence::Float64        # Confidence in selection decision (0.0-1.0)
    correlation_group::Union{Int, Nothing}  # Group ID for correlated features
    preprocessing_info::Dict{String, Any}   # Normalization, encoding details
    source_stage::String                 # Source: "xgboost", "random_forest", "correlation"
    timestamp_selected::DateTime         # When this feature was selected
end

"""
Stage 1 output container holding all selected features and metadata
"""
struct Stage1Output
    features::Vector{FeatureMetadata}    # 500 selected features with metadata
    selection_summary::Dict{String, Any} # Summary statistics from Stage 1
    dataset_info::Dict{String, Any}      # Original dataset information
    stage1_performance::Dict{String, Any} # Stage 1 selection performance metrics
    feature_matrix::Union{Matrix{Float64}, Nothing} # Optional: actual feature data
    validation_results::Dict{String, Any} # Validation of selection quality
    timestamp_created::DateTime          # When this output was generated
    version::String                      # Stage 1 output format version
end

"""
Feature indexing system for efficient MCTS operations
Maps between different feature identification schemes
"""
mutable struct FeatureIndexer
    # Core mappings
    original_to_stage2::Dict{Int, Int}    # Original ID -> Stage 2 index (1-500)
    stage2_to_original::Dict{Int, Int}    # Stage 2 index -> Original ID
    name_to_stage2::Dict{String, Int}     # Feature name -> Stage 2 index
    stage2_to_name::Dict{Int, String}     # Stage 2 index -> Feature name
    
    # Grouping and organization
    importance_ranking::Vector{Int}       # Stage 2 indices sorted by importance
    correlation_groups::Dict{Int, Vector{Int}} # Group ID -> Stage 2 indices
    type_groups::Dict{String, Vector{Int}}     # Type -> Stage 2 indices
    
    # Lookup optimization
    feature_lookup_cache::Dict{String, Any}   # Cache for frequent lookups
    index_validation::Dict{Int, Bool}          # Validation status per feature
    
    # Metadata
    total_features::Int                   # Should be 500
    creation_timestamp::DateTime
    last_update::DateTime
end

"""
Feature state tracker for MCTS operations
Maintains current selection/deselection status across all trees
"""
mutable struct FeatureStateTracker
    # Current states
    feature_states::Vector{Bool}          # True = selected, False = deselected (length 500)
    selection_history::Vector{Vector{Bool}} # History of state changes
    state_timestamps::Vector{DateTime}    # When each state change occurred
    
    # MCTS integration
    current_tree_id::Union{Int, Nothing}  # Currently active MCTS tree
    tree_feature_maps::Dict{Int, Vector{Bool}} # Tree ID -> feature states
    node_feature_cache::Dict{String, Vector{Bool}} # Node hash -> feature states
    
    # Performance tracking
    state_change_count::Int               # Total number of state changes
    cache_hit_count::Int                  # Cache performance metrics
    cache_miss_count::Int
    
    # Validation
    state_consistency_checks::Int         # Number of consistency validations
    last_validation_time::DateTime
    validation_errors::Vector{String}     # Any validation issues found
end

"""
Main Stage 1 integration interface
Coordinates feature reception, indexing, and state tracking
"""
mutable struct Stage1IntegrationInterface
    # Core components
    stage1_output::Union{Stage1Output, Nothing}
    feature_indexer::Union{FeatureIndexer, Nothing}
    state_tracker::Union{FeatureStateTracker, Nothing}
    
    # Configuration
    config::Dict{String, Any}
    validation_enabled::Bool
    caching_enabled::Bool
    performance_monitoring::Bool
    
    # Status and monitoring
    initialization_status::String         # "uninitialized", "ready", "error"
    last_operation::String               # Description of last operation
    operation_count::Int                 # Total operations performed
    error_log::Vector{String}            # Error messages
    performance_metrics::Dict{String, Float64} # Timing and performance data
    
    # Thread safety (for multi-GPU operations)
    access_lock::ReentrantLock
    modification_count::Int              # For detecting concurrent modifications
end

"""
Create default configuration for Stage 1 integration
"""
function create_stage1_integration_config(;
    validation_enabled::Bool = true,
    caching_enabled::Bool = true,
    performance_monitoring::Bool = true,
    cache_size_limit::Int = 10000,
    state_history_limit::Int = 1000,
    consistency_check_frequency::Int = 100,
    error_log_limit::Int = 500
)
    return Dict{String, Any}(
        "validation_enabled" => validation_enabled,
        "caching_enabled" => caching_enabled,
        "performance_monitoring" => performance_monitoring,
        "cache_size_limit" => cache_size_limit,
        "state_history_limit" => state_history_limit,
        "consistency_check_frequency" => consistency_check_frequency,
        "error_log_limit" => error_log_limit,
        "expected_feature_count" => 500,
        "supported_feature_types" => ["numeric", "categorical", "binary", "ordinal"],
        "minimum_importance_score" => 0.0,
        "maximum_importance_score" => 1.0
    )
end

"""
Initialize Stage 1 integration interface
"""
function initialize_stage1_interface(config::Dict{String, Any} = create_stage1_integration_config())
    interface = Stage1IntegrationInterface(
        nothing,  # stage1_output
        nothing,  # feature_indexer
        nothing,  # state_tracker
        config,
        config["validation_enabled"],
        config["caching_enabled"],
        config["performance_monitoring"],
        "uninitialized",
        "initialization",
        0,
        String[],
        Dict{String, Float64}(),
        ReentrantLock(),
        0
    )
    
    interface.initialization_status = "ready"
    interface.last_operation = "interface_initialized"
    
    return interface
end

"""
Load Stage 1 output from file
Supports JSON, HDF5, and JLD2 formats
"""
function load_stage1_output(interface::Stage1IntegrationInterface, filepath::String)
    lock(interface.access_lock) do
        try
            start_time = time()
            
            # Determine file format and load accordingly
            if endswith(filepath, ".json")
                stage1_data = load_stage1_from_json(filepath)
            elseif endswith(filepath, ".h5") || endswith(filepath, ".hdf5")
                stage1_data = load_stage1_from_hdf5(filepath)
            elseif endswith(filepath, ".jld2")
                stage1_data = load_stage1_from_jld2(filepath)
            else
                error("Unsupported file format. Supported: .json, .h5/.hdf5, .jld2")
            end
            
            # Validate loaded data
            if interface.validation_enabled
                validate_stage1_output(stage1_data, interface.config)
            end
            
            # Store the output
            interface.stage1_output = stage1_data
            interface.operation_count += 1
            interface.last_operation = "stage1_output_loaded"
            
            # Record performance metrics
            if interface.performance_monitoring
                load_time = time() - start_time
                interface.performance_metrics["last_load_time"] = load_time
                interface.performance_metrics["total_load_time"] = 
                    get(interface.performance_metrics, "total_load_time", 0.0) + load_time
            end
            
            return true
            
        catch e
            error_msg = "Failed to load Stage 1 output: $e"
            push!(interface.error_log, error_msg)
            interface.initialization_status = "error"
            interface.last_operation = "load_failed"
            
            # Limit error log size
            if length(interface.error_log) > interface.config["error_log_limit"]
                deleteat!(interface.error_log, 1)
            end
            
            rethrow(e)
        end
    end
end

"""
Load Stage 1 output from JSON format
"""
function load_stage1_from_json(filepath::String)
    data = JSON3.read(read(filepath, String))
    
    # Parse features
    features = FeatureMetadata[]
    for feature_data in data["features"]
        feature = FeatureMetadata(
            feature_data["original_id"],
            feature_data["stage1_rank"],
            feature_data["name"],
            feature_data["feature_type"],
            feature_data["importance_score"],
            feature_data["selection_confidence"],
            get(feature_data, "correlation_group", nothing),
            get(feature_data, "preprocessing_info", Dict{String, Any}()),
            feature_data["source_stage"],
            DateTime(feature_data["timestamp_selected"])
        )
        push!(features, feature)
    end
    
    return Stage1Output(
        features,
        data["selection_summary"],
        data["dataset_info"],
        data["stage1_performance"],
        nothing,  # Feature matrix not stored in JSON
        data["validation_results"],
        DateTime(data["timestamp_created"]),
        data["version"]
    )
end

"""
Load Stage 1 output from HDF5 format (with optional feature matrix)
"""
function load_stage1_from_hdf5(filepath::String)
    h5open(filepath, "r") do file
        # Load metadata
        metadata = JSON3.read(read(file["metadata"]))
        
        # Load feature matrix if present
        feature_matrix = nothing
        if haskey(file, "feature_matrix")
            feature_matrix = read(file["feature_matrix"])
        end
        
        # Parse features from metadata
        features = FeatureMetadata[]
        for feature_data in metadata["features"]
            feature = FeatureMetadata(
                feature_data["original_id"],
                feature_data["stage1_rank"],
                feature_data["name"],
                feature_data["feature_type"],
                feature_data["importance_score"],
                feature_data["selection_confidence"],
                get(feature_data, "correlation_group", nothing),
                get(feature_data, "preprocessing_info", Dict{String, Any}()),
                feature_data["source_stage"],
                DateTime(feature_data["timestamp_selected"])
            )
            push!(features, feature)
        end
        
        return Stage1Output(
            features,
            metadata["selection_summary"],
            metadata["dataset_info"],
            metadata["stage1_performance"],
            feature_matrix,
            metadata["validation_results"],
            DateTime(metadata["timestamp_created"]),
            metadata["version"]
        )
    end
end

"""
Load Stage 1 output from JLD2 format (native Julia serialization)
"""
function load_stage1_from_jld2(filepath::String)
    return JLD2.load(filepath, "stage1_output")
end

"""
Validate Stage 1 output data integrity and format
"""
function validate_stage1_output(output::Stage1Output, config::Dict{String, Any})
    # Check feature count
    expected_count = config["expected_feature_count"]
    actual_count = length(output.features)
    if actual_count != expected_count
        error("Expected $expected_count features, got $actual_count")
    end
    
    # Validate each feature
    for (i, feature) in enumerate(output.features)
        # Check stage1_rank sequence
        if feature.stage1_rank != i
            error("Feature $i has incorrect stage1_rank: $(feature.stage1_rank), expected $i")
        end
        
        # Check importance score range
        if !(config["minimum_importance_score"] <= feature.importance_score <= config["maximum_importance_score"])
            error("Feature $(feature.name) has invalid importance score: $(feature.importance_score)")
        end
        
        # Check feature type
        if !(feature.feature_type in config["supported_feature_types"])
            error("Feature $(feature.name) has unsupported type: $(feature.feature_type)")
        end
        
        # Check selection confidence
        if !(0.0 <= feature.selection_confidence <= 1.0)
            error("Feature $(feature.name) has invalid selection confidence: $(feature.selection_confidence)")
        end
    end
    
    # Check for duplicate original IDs
    original_ids = [f.original_id for f in output.features]
    if length(unique(original_ids)) != length(original_ids)
        error("Duplicate original feature IDs found")
    end
    
    # Check for duplicate names
    names = [f.name for f in output.features]
    if length(unique(names)) != length(names)
        error("Duplicate feature names found")
    end
    
    return true
end

"""
Build feature indexing system from Stage 1 output
"""
function build_feature_indexer(interface::Stage1IntegrationInterface)
    if isnothing(interface.stage1_output)
        error("Stage 1 output must be loaded before building indexer")
    end
    
    lock(interface.access_lock) do
        try
            start_time = time()
            output = interface.stage1_output
            
            # Initialize mappings
            original_to_stage2 = Dict{Int, Int}()
            stage2_to_original = Dict{Int, Int}()
            name_to_stage2 = Dict{String, Int}()
            stage2_to_name = Dict{Int, String}()
            
            # Build core mappings
            for (stage2_idx, feature) in enumerate(output.features)
                original_to_stage2[feature.original_id] = stage2_idx
                stage2_to_original[stage2_idx] = feature.original_id
                name_to_stage2[feature.name] = stage2_idx
                stage2_to_name[stage2_idx] = feature.name
            end
            
            # Build importance ranking (Stage 2 indices sorted by importance)
            importance_ranking = sortperm([f.importance_score for f in output.features], rev=true)
            
            # Build correlation groups
            correlation_groups = Dict{Int, Vector{Int}}()
            for (stage2_idx, feature) in enumerate(output.features)
                if !isnothing(feature.correlation_group)
                    group_id = feature.correlation_group
                    if !haskey(correlation_groups, group_id)
                        correlation_groups[group_id] = Int[]
                    end
                    push!(correlation_groups[group_id], stage2_idx)
                end
            end
            
            # Build type groups
            type_groups = Dict{String, Vector{Int}}()
            for (stage2_idx, feature) in enumerate(output.features)
                feature_type = feature.feature_type
                if !haskey(type_groups, feature_type)
                    type_groups[feature_type] = Int[]
                end
                push!(type_groups[feature_type], stage2_idx)
            end
            
            # Initialize validation status
            index_validation = Dict{Int, Bool}()
            for i in 1:length(output.features)
                index_validation[i] = true  # All valid initially
            end
            
            # Create indexer
            indexer = FeatureIndexer(
                original_to_stage2,
                stage2_to_original,
                name_to_stage2,
                stage2_to_name,
                importance_ranking,
                correlation_groups,
                type_groups,
                Dict{String, Any}(),  # Empty cache initially
                index_validation,
                length(output.features),
                now(),
                now()
            )
            
            interface.feature_indexer = indexer
            interface.operation_count += 1
            interface.last_operation = "indexer_built"
            
            # Record performance metrics
            if interface.performance_monitoring
                build_time = time() - start_time
                interface.performance_metrics["indexer_build_time"] = build_time
            end
            
            return indexer
            
        catch e
            error_msg = "Failed to build feature indexer: $e"
            push!(interface.error_log, error_msg)
            interface.last_operation = "indexer_build_failed"
            rethrow(e)
        end
    end
end

"""
Initialize feature state tracker
"""
function initialize_state_tracker(interface::Stage1IntegrationInterface)
    if isnothing(interface.feature_indexer)
        error("Feature indexer must be built before initializing state tracker")
    end
    
    lock(interface.access_lock) do
        try
            feature_count = interface.feature_indexer.total_features
            
            # Initialize all features as unselected
            initial_states = fill(false, feature_count)
            
            tracker = FeatureStateTracker(
                initial_states,
                [copy(initial_states)],  # Initial history entry
                [now()],
                nothing,  # No current tree
                Dict{Int, Vector{Bool}}(),
                Dict{String, Vector{Bool}}(),
                0,  # No state changes yet
                0,  # No cache hits
                0,  # No cache misses
                0,  # No consistency checks
                now(),
                String[]  # No validation errors
            )
            
            interface.state_tracker = tracker
            interface.operation_count += 1
            interface.last_operation = "state_tracker_initialized"
            
            return tracker
            
        catch e
            error_msg = "Failed to initialize state tracker: $e"
            push!(interface.error_log, error_msg)
            interface.last_operation = "state_tracker_init_failed"
            rethrow(e)
        end
    end
end

"""
Get Stage 2 index from original feature ID
"""
function get_stage2_index(interface::Stage1IntegrationInterface, original_id::Int)
    if isnothing(interface.feature_indexer)
        error("Feature indexer not initialized")
    end
    
    indexer = interface.feature_indexer
    
    # Check cache first
    cache_key = "orig_to_stage2_$original_id"
    if interface.caching_enabled && haskey(indexer.feature_lookup_cache, cache_key)
        interface.state_tracker.cache_hit_count += 1
        return indexer.feature_lookup_cache[cache_key]
    end
    
    # Lookup in mapping
    if haskey(indexer.original_to_stage2, original_id)
        stage2_idx = indexer.original_to_stage2[original_id]
        
        # Cache the result
        if interface.caching_enabled
            indexer.feature_lookup_cache[cache_key] = stage2_idx
            
            # Limit cache size
            if length(indexer.feature_lookup_cache) > interface.config["cache_size_limit"]
                # Remove oldest entries (simple FIFO)
                keys_to_remove = collect(keys(indexer.feature_lookup_cache))[1:100]
                for key in keys_to_remove
                    delete!(indexer.feature_lookup_cache, key)
                end
            end
        end
        
        interface.state_tracker.cache_miss_count += 1
        return stage2_idx
    else
        error("Original feature ID $original_id not found in Stage 1 output")
    end
end

"""
Get original feature ID from Stage 2 index
"""
function get_original_id(interface::Stage1IntegrationInterface, stage2_index::Int)
    if isnothing(interface.feature_indexer)
        error("Feature indexer not initialized")
    end
    
    indexer = interface.feature_indexer
    
    if haskey(indexer.stage2_to_original, stage2_index)
        return indexer.stage2_to_original[stage2_index]
    else
        error("Stage 2 index $stage2_index not valid (should be 1-$(indexer.total_features))")
    end
end

"""
Get Stage 2 index from feature name
"""
function get_stage2_index_by_name(interface::Stage1IntegrationInterface, feature_name::String)
    if isnothing(interface.feature_indexer)
        error("Feature indexer not initialized")
    end
    
    indexer = interface.feature_indexer
    
    if haskey(indexer.name_to_stage2, feature_name)
        return indexer.name_to_stage2[feature_name]
    else
        error("Feature name '$feature_name' not found in Stage 1 output")
    end
end

"""
Get feature metadata by Stage 2 index
"""
function get_feature_metadata(interface::Stage1IntegrationInterface, stage2_index::Int)
    if isnothing(interface.stage1_output)
        error("Stage 1 output not loaded")
    end
    
    if stage2_index < 1 || stage2_index > length(interface.stage1_output.features)
        error("Stage 2 index $stage2_index out of range")
    end
    
    return interface.stage1_output.features[stage2_index]
end

"""
Update feature state (select/deselect) for MCTS operations
"""
function update_feature_state!(interface::Stage1IntegrationInterface, stage2_index::Int, selected::Bool; tree_id::Union{Int, Nothing} = nothing)
    if isnothing(interface.state_tracker)
        error("State tracker not initialized")
    end
    
    lock(interface.access_lock) do
        tracker = interface.state_tracker
        
        # Validate index
        if stage2_index < 1 || stage2_index > length(tracker.feature_states)
            error("Stage 2 index $stage2_index out of range")
        end
        
        # Update state if changed
        if tracker.feature_states[stage2_index] != selected
            tracker.feature_states[stage2_index] = selected
            tracker.state_change_count += 1
            interface.modification_count += 1
            
            # Record history (with limits)
            push!(tracker.selection_history, copy(tracker.feature_states))
            push!(tracker.state_timestamps, now())
            
            # Limit history size
            history_limit = interface.config["state_history_limit"]
            if length(tracker.selection_history) > history_limit
                deleteat!(tracker.selection_history, 1)
                deleteat!(tracker.state_timestamps, 1)
            end
            
            # Update tree-specific state if tree_id provided
            if !isnothing(tree_id)
                tracker.current_tree_id = tree_id
                tracker.tree_feature_maps[tree_id] = copy(tracker.feature_states)
            end
            
            interface.last_operation = "feature_state_updated_$(stage2_index)_$(selected)"
        end
    end
end

"""
Get current feature selection state
"""
function get_feature_states(interface::Stage1IntegrationInterface; tree_id::Union{Int, Nothing} = nothing)
    if isnothing(interface.state_tracker)
        error("State tracker not initialized")
    end
    
    tracker = interface.state_tracker
    
    if !isnothing(tree_id) && haskey(tracker.tree_feature_maps, tree_id)
        # Return tree-specific state
        return copy(tracker.tree_feature_maps[tree_id])
    else
        # Return current global state
        return copy(tracker.feature_states)
    end
end

"""
Get features by importance ranking (top N most important)
"""
function get_top_features(interface::Stage1IntegrationInterface, n::Int = 50)
    if isnothing(interface.feature_indexer)
        error("Feature indexer not initialized")
    end
    
    indexer = interface.feature_indexer
    top_indices = indexer.importance_ranking[1:min(n, length(indexer.importance_ranking))]
    
    return [(idx, get_feature_metadata(interface, idx)) for idx in top_indices]
end

"""
Get features by correlation group
"""
function get_correlation_group(interface::Stage1IntegrationInterface, group_id::Int)
    if isnothing(interface.feature_indexer)
        error("Feature indexer not initialized")
    end
    
    indexer = interface.feature_indexer
    
    if haskey(indexer.correlation_groups, group_id)
        indices = indexer.correlation_groups[group_id]
        return [(idx, get_feature_metadata(interface, idx)) for idx in indices]
    else
        return Tuple{Int, FeatureMetadata}[]
    end
end

"""
Get features by type
"""
function get_features_by_type(interface::Stage1IntegrationInterface, feature_type::String)
    if isnothing(interface.feature_indexer)
        error("Feature indexer not initialized")
    end
    
    indexer = interface.feature_indexer
    
    if haskey(indexer.type_groups, feature_type)
        indices = indexer.type_groups[feature_type]
        return [(idx, get_feature_metadata(interface, idx)) for idx in indices]
    else
        return Tuple{Int, FeatureMetadata}[]
    end
end

"""
Validate feature state consistency
"""
function validate_feature_consistency!(interface::Stage1IntegrationInterface)
    if isnothing(interface.state_tracker) || isnothing(interface.feature_indexer)
        error("State tracker and indexer must be initialized")
    end
    
    lock(interface.access_lock) do
        tracker = interface.state_tracker
        indexer = interface.feature_indexer
        
        tracker.state_consistency_checks += 1
        tracker.last_validation_time = now()
        
        # Check state vector length
        expected_length = indexer.total_features
        actual_length = length(tracker.feature_states)
        
        if actual_length != expected_length
            error_msg = "Feature state length mismatch: expected $expected_length, got $actual_length"
            push!(tracker.validation_errors, error_msg)
            error(error_msg)
        end
        
        # Check tree state consistency
        for (tree_id, tree_states) in tracker.tree_feature_maps
            if length(tree_states) != expected_length
                error_msg = "Tree $tree_id state length mismatch: expected $expected_length, got $(length(tree_states))"
                push!(tracker.validation_errors, error_msg)
                error(error_msg)
            end
        end
        
        # Validate indexer mappings
        for stage2_idx in 1:indexer.total_features
            if !haskey(indexer.stage2_to_original, stage2_idx)
                error_msg = "Missing stage2_to_original mapping for index $stage2_idx"
                push!(tracker.validation_errors, error_msg)
                error(error_msg)
            end
            
            original_id = indexer.stage2_to_original[stage2_idx]
            if !haskey(indexer.original_to_stage2, original_id)
                error_msg = "Missing original_to_stage2 mapping for original ID $original_id"
                push!(tracker.validation_errors, error_msg)
                error(error_msg)
            end
        end
        
        interface.last_operation = "consistency_validated"
        return true
    end
end

"""
Generate comprehensive status report
"""
function generate_status_report(interface::Stage1IntegrationInterface)
    report = String[]
    
    push!(report, "=== Stage 1 Integration Interface Status Report ===")
    push!(report, "Generated: $(now())")
    push!(report, "")
    
    # Basic status
    push!(report, "Initialization Status: $(interface.initialization_status)")
    push!(report, "Last Operation: $(interface.last_operation)")
    push!(report, "Total Operations: $(interface.operation_count)")
    
    # Component status
    stage1_status = isnothing(interface.stage1_output) ? "Not Loaded" : "Loaded"
    indexer_status = isnothing(interface.feature_indexer) ? "Not Built" : "Ready"
    tracker_status = isnothing(interface.state_tracker) ? "Not Initialized" : "Active"
    
    push!(report, "")
    push!(report, "Component Status:")
    push!(report, "  Stage 1 Output: $stage1_status")
    push!(report, "  Feature Indexer: $indexer_status")
    push!(report, "  State Tracker: $tracker_status")
    
    # Feature information
    if !isnothing(interface.stage1_output)
        output = interface.stage1_output
        push!(report, "")
        push!(report, "Feature Information:")
        push!(report, "  Total Features: $(length(output.features))")
        push!(report, "  Output Version: $(output.version)")
        push!(report, "  Created: $(output.timestamp_created)")
        
        # Feature type distribution
        if !isnothing(interface.feature_indexer)
            type_counts = Dict{String, Int}()
            for feature in output.features
                type_counts[feature.feature_type] = get(type_counts, feature.feature_type, 0) + 1
            end
            
            push!(report, "  Feature Types:")
            for (type, count) in sort(collect(type_counts))
                push!(report, "    $type: $count")
            end
        end
    end
    
    # State tracking information
    if !isnothing(interface.state_tracker)
        tracker = interface.state_tracker
        selected_count = sum(tracker.feature_states)
        
        push!(report, "")
        push!(report, "State Tracking:")
        push!(report, "  Currently Selected: $selected_count / $(length(tracker.feature_states))")
        push!(report, "  State Changes: $(tracker.state_change_count)")
        push!(report, "  Active Trees: $(length(tracker.tree_feature_maps))")
        push!(report, "  Cache Hits: $(tracker.cache_hit_count)")
        push!(report, "  Cache Misses: $(tracker.cache_miss_count)")
        push!(report, "  Consistency Checks: $(tracker.state_consistency_checks)")
        push!(report, "  Last Validation: $(tracker.last_validation_time)")
        
        if !isempty(tracker.validation_errors)
            push!(report, "  Validation Errors: $(length(tracker.validation_errors))")
        end
    end
    
    # Performance metrics
    if !isempty(interface.performance_metrics)
        push!(report, "")
        push!(report, "Performance Metrics:")
        for (metric, value) in sort(collect(interface.performance_metrics))
            push!(report, "  $metric: $(round(value, digits=4))s")
        end
    end
    
    # Error information
    if !isempty(interface.error_log)
        push!(report, "")
        push!(report, "Recent Errors (last 5):")
        for error in interface.error_log[max(1, length(interface.error_log)-4):end]
            push!(report, "  - $error")
        end
    end
    
    push!(report, "")
    push!(report, "=== End Status Report ===")
    
    return join(report, "\n")
end

"""
Save current interface state to file
"""
function save_interface_state(interface::Stage1IntegrationInterface, filepath::String)
    lock(interface.access_lock) do
        state_data = Dict{String, Any}(
            "timestamp" => now(),
            "initialization_status" => interface.initialization_status,
            "last_operation" => interface.last_operation,
            "operation_count" => interface.operation_count,
            "modification_count" => interface.modification_count,
            "config" => interface.config,
            "performance_metrics" => interface.performance_metrics,
            "error_log" => interface.error_log
        )
        
        # Add component data if available
        if !isnothing(interface.state_tracker)
            tracker = interface.state_tracker
            state_data["state_tracker"] = Dict(
                "feature_states" => tracker.feature_states,
                "state_change_count" => tracker.state_change_count,
                "cache_hit_count" => tracker.cache_hit_count,
                "cache_miss_count" => tracker.cache_miss_count,
                "tree_count" => length(tracker.tree_feature_maps),
                "consistency_checks" => tracker.state_consistency_checks
            )
        end
        
        # Save to file
        if endswith(filepath, ".json")
            open(filepath, "w") do f
                JSON3.pretty(f, state_data)
            end
        elseif endswith(filepath, ".jld2")
            JLD2.save(filepath, "interface_state", state_data)
        else
            error("Unsupported save format. Use .json or .jld2")
        end
        
        interface.last_operation = "state_saved"
    end
end

# Export main types and functions
export FeatureMetadata, Stage1Output, FeatureIndexer, FeatureStateTracker, Stage1IntegrationInterface
export create_stage1_integration_config, initialize_stage1_interface
export load_stage1_output, build_feature_indexer, initialize_state_tracker
export get_stage2_index, get_original_id, get_stage2_index_by_name, get_feature_metadata
export update_feature_state!, get_feature_states
export get_top_features, get_correlation_group, get_features_by_type
export validate_feature_consistency!, generate_status_report, save_interface_state

end # module Stage1Integration