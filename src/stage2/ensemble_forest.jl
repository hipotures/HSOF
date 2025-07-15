"""
Ensemble Forest Architecture for 100+ MCTS Trees
Core ensemble forest structure supporting parallel MCTS tree execution with efficient memory management
and GPU optimization for Stage 2 feature selection optimization.

This module provides the foundational architecture for managing, coordinating, and synchronizing
100+ MCTS trees in a memory-efficient manner suitable for GPU acceleration.
"""

module EnsembleForest

using Random
using Statistics
using Dates
using Printf
using CUDA
using Base.Threads

# Import Stage 1 integration for feature management
include("stage1_integration.jl")
using .Stage1Integration

"""
Tree state enumeration for lifecycle management
"""
@enum TreeState begin
    UNINITIALIZED = 1
    READY = 2
    RUNNING = 3
    PAUSED = 4
    COMPLETED = 5
    FAILED = 6
    TERMINATED = 7
end

"""
Memory-efficient MCTS tree representation
Designed for GPU memory optimization and high-performance parallel execution
"""
mutable struct MCTSTree
    # Core identification
    tree_id::Int                        # Unique identifier within forest
    creation_time::DateTime             # When tree was created
    last_update::DateTime               # Last activity timestamp
    
    # State management
    state::TreeState                    # Current lifecycle state
    iteration_count::Int                # Number of MCTS iterations completed
    max_iterations::Int                 # Maximum iterations before termination
    
    # Feature selection state
    selected_features::Vector{Bool}     # Current feature selection (500 elements)
    feature_importance::Vector{Float32} # Local importance scores per feature
    selection_history::Vector{Vector{Bool}} # History of feature selections
    
    # Performance metrics
    best_score::Float32                 # Best evaluation score achieved
    current_score::Float32              # Current evaluation score
    evaluation_count::Int               # Total evaluations performed
    gpu_memory_used::Int               # GPU memory allocated (bytes)
    
    # Tree structure (memory-efficient representation)
    node_count::Int                     # Total nodes in tree
    max_nodes::Int                      # Maximum nodes before pruning
    tree_depth::Int                     # Current maximum depth
    max_depth::Int                      # Maximum allowed depth
    
    # Execution context
    gpu_device_id::Int                  # Assigned GPU device (-1 for CPU)
    thread_id::Int                      # Assigned thread ID
    priority::Float32                   # Execution priority (0.0-1.0)
    
    # Metadata and configuration
    config::Dict{String, Any}          # Tree-specific configuration
    metadata::Dict{String, Any}        # Runtime metadata
    error_log::Vector{String}           # Error messages
end

"""
Tree pool configuration for ensemble management
"""
struct TreePoolConfig
    # Capacity settings
    max_trees::Int                      # Maximum number of trees in pool
    initial_trees::Int                  # Number of trees to create initially
    tree_growth_rate::Int               # Trees to add when expanding
    
    # Resource management
    max_gpu_memory_per_tree::Int        # Maximum GPU memory per tree (bytes)
    max_cpu_memory_per_tree::Int        # Maximum CPU memory per tree (bytes)
    gpu_device_allocation::Vector{Int}  # GPU device assignment strategy
    
    # Performance settings
    default_max_iterations::Int         # Default max iterations per tree
    default_max_nodes::Int              # Default max nodes per tree
    default_max_depth::Int              # Default max tree depth
    default_priority::Float32           # Default execution priority
    
    # Memory optimization
    enable_memory_pooling::Bool         # Use memory pooling for efficiency
    enable_node_pruning::Bool           # Enable automatic node pruning
    pruning_threshold::Float32          # Score threshold for pruning
    garbage_collection_frequency::Int   # How often to run GC
    
    # Synchronization settings
    sync_frequency::Int                 # How often to sync feature states
    enable_cross_tree_learning::Bool    # Enable learning between trees
    learning_rate::Float32              # Cross-tree learning rate
end

"""
Create default tree pool configuration
"""
function create_tree_pool_config(;
    max_trees::Int = 100,
    initial_trees::Int = 10,
    tree_growth_rate::Int = 5,
    max_gpu_memory_per_tree::Int = 100 * 1024 * 1024,  # 100MB
    max_cpu_memory_per_tree::Int = 50 * 1024 * 1024,   # 50MB
    gpu_device_allocation::Vector{Int} = CUDA.functional() ? collect(0:CUDA.ndevices()-1) : Int[],
    default_max_iterations::Int = 1000,
    default_max_nodes::Int = 10000,
    default_max_depth::Int = 50,
    default_priority::Float32 = 0.5f0,
    enable_memory_pooling::Bool = true,
    enable_node_pruning::Bool = true,
    pruning_threshold::Float32 = 0.01f0,
    garbage_collection_frequency::Int = 100,
    sync_frequency::Int = 10,
    enable_cross_tree_learning::Bool = true,
    learning_rate::Float32 = 0.1f0
)
    return TreePoolConfig(
        max_trees, initial_trees, tree_growth_rate,
        max_gpu_memory_per_tree, max_cpu_memory_per_tree, gpu_device_allocation,
        default_max_iterations, default_max_nodes, default_max_depth, default_priority,
        enable_memory_pooling, enable_node_pruning, pruning_threshold, garbage_collection_frequency,
        sync_frequency, enable_cross_tree_learning, learning_rate
    )
end

"""
Forest manager for coordinating ensemble of MCTS trees
Handles tree lifecycle, resource allocation, and synchronization
"""
mutable struct ForestManager
    # Core configuration
    config::TreePoolConfig              # Pool configuration
    stage1_interface::Union{Stage1IntegrationInterface, Nothing} # Stage 1 integration
    
    # Tree management
    trees::Dict{Int, MCTSTree}          # Active trees by ID
    available_tree_ids::Vector{Int}     # Pool of available tree IDs
    next_tree_id::Int                   # Next ID to assign
    
    # Resource management
    gpu_memory_usage::Dict{Int, Int}    # GPU memory usage by device
    cpu_memory_usage::Int               # Total CPU memory usage
    thread_pool::Vector{Int}            # Available thread IDs
    
    # Execution coordination
    active_trees::Set{Int}              # Currently executing tree IDs
    paused_trees::Set{Int}              # Paused tree IDs
    completed_trees::Set{Int}           # Completed tree IDs
    failed_trees::Set{Int}              # Failed tree IDs
    
    # Performance monitoring
    total_iterations::Int               # Total iterations across all trees
    total_evaluations::Int              # Total evaluations performed
    total_execution_time::Float64       # Total execution time (seconds)
    creation_time::DateTime             # Forest creation timestamp
    last_sync_time::DateTime            # Last synchronization timestamp
    
    # Synchronization and coordination
    global_feature_importance::Vector{Float32} # Global feature importance scores
    consensus_features::Vector{Bool}    # Consensus feature selection
    sync_lock::ReentrantLock           # Thread synchronization
    
    # Status and monitoring
    forest_state::String               # Overall forest state
    performance_metrics::Dict{String, Float64} # Performance tracking
    error_log::Vector{String}          # Forest-level errors
    debug_enabled::Bool                # Enable debug logging
end

"""
Initialize forest manager with configuration and Stage 1 integration
"""
function initialize_forest_manager(config::TreePoolConfig, 
                                  stage1_interface::Union{Stage1IntegrationInterface, Nothing} = nothing;
                                  debug_enabled::Bool = false)
    
    # Initialize GPU memory tracking
    gpu_memory_usage = Dict{Int, Int}()
    if CUDA.functional()
        for device_id in config.gpu_device_allocation
            gpu_memory_usage[device_id] = 0
        end
    end
    
    # Initialize thread pool
    thread_pool = collect(1:Threads.nthreads())
    
    # Initialize global feature state (500 features from Stage 1)
    n_features = 500
    global_feature_importance = zeros(Float32, n_features)
    consensus_features = fill(false, n_features)
    
    # Create forest manager
    manager = ForestManager(
        config,
        stage1_interface,
        Dict{Int, MCTSTree}(),
        collect(1:config.max_trees),
        1,
        gpu_memory_usage,
        0,
        thread_pool,
        Set{Int}(),
        Set{Int}(),
        Set{Int}(),
        Set{Int}(),
        0, 0, 0.0,
        now(),
        now(),
        global_feature_importance,
        consensus_features,
        ReentrantLock(),
        "initialized",
        Dict{String, Float64}(),
        String[],
        debug_enabled
    )
    
    # Create initial trees
    for i in 1:config.initial_trees
        create_tree!(manager)
    end
    
    manager.forest_state = "ready"
    
    if debug_enabled
        @info "Forest manager initialized with $(length(manager.trees)) trees"
    end
    
    return manager
end

"""
Create new MCTS tree in the forest
"""
function create_tree!(manager::ForestManager; 
                     max_iterations::Union{Int, Nothing} = nothing,
                     max_nodes::Union{Int, Nothing} = nothing,
                     max_depth::Union{Int, Nothing} = nothing,
                     priority::Union{Float32, Nothing} = nothing,
                     gpu_device_id::Union{Int, Nothing} = nothing)
    
    lock(manager.sync_lock) do
        # Check capacity
        if length(manager.trees) >= manager.config.max_trees
            error("Forest at maximum capacity ($(manager.config.max_trees) trees)")
        end
        
        # Get next available tree ID
        if isempty(manager.available_tree_ids)
            error("No available tree IDs")
        end
        
        tree_id = popfirst!(manager.available_tree_ids)
        
        # Assign GPU device
        assigned_gpu = if isnothing(gpu_device_id)
            if !isempty(manager.config.gpu_device_allocation)
                # Find GPU with least memory usage
                min_usage = minimum(values(manager.gpu_memory_usage))
                candidates = [dev for (dev, usage) in manager.gpu_memory_usage if usage == min_usage]
                rand(candidates)
            else
                -1  # CPU execution
            end
        else
            gpu_device_id
        end
        
        # Assign thread
        assigned_thread = if !isempty(manager.thread_pool)
            rand(manager.thread_pool)
        else
            1  # Fallback to main thread
        end
        
        # Create tree
        tree = MCTSTree(
            tree_id,
            now(),
            now(),
            READY,
            0,
            isnothing(max_iterations) ? manager.config.default_max_iterations : max_iterations,
            fill(false, 500),  # Initial feature selection (all unselected)
            zeros(Float32, 500),  # Feature importance scores
            Vector{Vector{Bool}}(),  # Selection history
            -Inf32,  # Best score
            -Inf32,  # Current score
            0,       # Evaluation count
            0,       # GPU memory used
            0,       # Node count
            isnothing(max_nodes) ? manager.config.default_max_nodes : max_nodes,
            0,       # Tree depth
            isnothing(max_depth) ? manager.config.default_max_depth : max_depth,
            assigned_gpu,
            assigned_thread,
            isnothing(priority) ? manager.config.default_priority : priority,
            Dict{String, Any}(),  # Config
            Dict{String, Any}(),  # Metadata
            String[]  # Error log
        )
        
        # Store tree
        manager.trees[tree_id] = tree
        
        # Update resource tracking
        if assigned_gpu >= 0
            manager.gpu_memory_usage[assigned_gpu] += manager.config.max_gpu_memory_per_tree
        end
        manager.cpu_memory_usage += manager.config.max_cpu_memory_per_tree
        
        if manager.debug_enabled
            @info "Created tree $tree_id on GPU $assigned_gpu, thread $assigned_thread"
        end
        
        return tree_id
    end
end

"""
Remove tree from forest and clean up resources
"""
function remove_tree!(manager::ForestManager, tree_id::Int)
    lock(manager.sync_lock) do
        if !haskey(manager.trees, tree_id)
            error("Tree $tree_id not found")
        end
        
        tree = manager.trees[tree_id]
        
        # Update resource tracking
        if tree.gpu_device_id >= 0
            manager.gpu_memory_usage[tree.gpu_device_id] -= tree.gpu_memory_used
        end
        manager.cpu_memory_usage -= manager.config.max_cpu_memory_per_tree
        
        # Remove from tracking sets
        delete!(manager.active_trees, tree_id)
        delete!(manager.paused_trees, tree_id)
        delete!(manager.completed_trees, tree_id)
        delete!(manager.failed_trees, tree_id)
        
        # Remove tree and return ID to pool
        delete!(manager.trees, tree_id)
        push!(manager.available_tree_ids, tree_id)
        
        if manager.debug_enabled
            @info "Removed tree $tree_id"
        end
        
        return true
    end
end

"""
Start execution of specific tree
"""
function start_tree!(manager::ForestManager, tree_id::Int)
    lock(manager.sync_lock) do
        if !haskey(manager.trees, tree_id)
            error("Tree $tree_id not found")
        end
        
        tree = manager.trees[tree_id]
        
        if tree.state != READY && tree.state != PAUSED
            error("Tree $tree_id not in startable state (current: $(tree.state))")
        end
        
        # Update state
        tree.state = RUNNING
        tree.last_update = now()
        
        # Add to active set
        push!(manager.active_trees, tree_id)
        delete!(manager.paused_trees, tree_id)
        
        if manager.debug_enabled
            @info "Started tree $tree_id"
        end
        
        return true
    end
end

"""
Pause execution of specific tree
"""
function pause_tree!(manager::ForestManager, tree_id::Int)
    lock(manager.sync_lock) do
        if !haskey(manager.trees, tree_id)
            error("Tree $tree_id not found")
        end
        
        tree = manager.trees[tree_id]
        
        if tree.state != RUNNING
            error("Tree $tree_id not running (current: $(tree.state))")
        end
        
        # Update state
        tree.state = PAUSED
        tree.last_update = now()
        
        # Move to paused set
        delete!(manager.active_trees, tree_id)
        push!(manager.paused_trees, tree_id)
        
        if manager.debug_enabled
            @info "Paused tree $tree_id"
        end
        
        return true
    end
end

"""
Mark tree as completed
"""
function complete_tree!(manager::ForestManager, tree_id::Int, final_score::Float32)
    lock(manager.sync_lock) do
        if !haskey(manager.trees, tree_id)
            error("Tree $tree_id not found")
        end
        
        tree = manager.trees[tree_id]
        
        # Update tree state
        tree.state = COMPLETED
        tree.current_score = final_score
        tree.best_score = max(tree.best_score, final_score)
        tree.last_update = now()
        
        # Move to completed set
        delete!(manager.active_trees, tree_id)
        delete!(manager.paused_trees, tree_id)
        push!(manager.completed_trees, tree_id)
        
        # Update global statistics
        manager.total_iterations += tree.iteration_count
        manager.total_evaluations += tree.evaluation_count
        
        if manager.debug_enabled
            @info "Completed tree $tree_id with score $final_score"
        end
        
        return true
    end
end

"""
Mark tree as failed with error information
"""
function fail_tree!(manager::ForestManager, tree_id::Int, error_message::String)
    lock(manager.sync_lock) do
        if !haskey(manager.trees, tree_id)
            error("Tree $tree_id not found")
        end
        
        tree = manager.trees[tree_id]
        
        # Update tree state
        tree.state = FAILED
        tree.last_update = now()
        push!(tree.error_log, "$(now()): $error_message")
        
        # Move to failed set
        delete!(manager.active_trees, tree_id)
        delete!(manager.paused_trees, tree_id)
        push!(manager.failed_trees, tree_id)
        
        # Log forest-level error
        push!(manager.error_log, "Tree $tree_id failed: $error_message")
        
        if manager.debug_enabled
            @warn "Tree $tree_id failed: $error_message"
        end
        
        return true
    end
end

"""
Start all ready trees in the forest
"""
function start_all_trees!(manager::ForestManager)
    started_count = 0
    
    for (tree_id, tree) in manager.trees
        if tree.state == READY
            try
                start_tree!(manager, tree_id)
                started_count += 1
            catch e
                fail_tree!(manager, tree_id, "Failed to start: $e")
            end
        end
    end
    
    if manager.debug_enabled
        @info "Started $started_count trees"
    end
    
    return started_count
end

"""
Pause all running trees in the forest
"""
function pause_all_trees!(manager::ForestManager)
    paused_count = 0
    
    for tree_id in copy(manager.active_trees)
        try
            pause_tree!(manager, tree_id)
            paused_count += 1
        catch e
            fail_tree!(manager, tree_id, "Failed to pause: $e")
        end
    end
    
    if manager.debug_enabled
        @info "Paused $paused_count trees"
    end
    
    return paused_count
end

"""
Synchronize feature states across all trees
"""
function synchronize_features!(manager::ForestManager)
    lock(manager.sync_lock) do
        if isempty(manager.trees)
            return
        end
        
        n_features = 500
        feature_counts = zeros(Int, n_features)
        importance_sum = zeros(Float32, n_features)
        active_tree_count = 0
        
        # Aggregate feature information from all trees
        for (tree_id, tree) in manager.trees
            if tree.state == RUNNING || tree.state == COMPLETED
                active_tree_count += 1
                
                # Count feature selections
                for i in 1:n_features
                    if tree.selected_features[i]
                        feature_counts[i] += 1
                    end
                end
                
                # Sum importance scores
                importance_sum .+= tree.feature_importance
            end
        end
        
        if active_tree_count > 0
            # Update global feature importance (average across trees)
            manager.global_feature_importance .= importance_sum ./ active_tree_count
            
            # Update consensus features (majority vote)
            consensus_threshold = max(1, active_tree_count ÷ 2)
            for i in 1:n_features
                manager.consensus_features[i] = feature_counts[i] >= consensus_threshold
            end
            
            # Apply cross-tree learning if enabled
            if manager.config.enable_cross_tree_learning
                apply_cross_tree_learning!(manager)
            end
        end
        
        manager.last_sync_time = now()
        
        if manager.debug_enabled
            selected_consensus = sum(manager.consensus_features)
            @info "Synchronized features: $selected_consensus consensus features from $active_tree_count trees"
        end
    end
end

"""
Apply cross-tree learning to share knowledge between trees
"""
function apply_cross_tree_learning!(manager::ForestManager)
    if isempty(manager.trees)
        return
    end
    
    learning_rate = manager.config.learning_rate
    global_importance = manager.global_feature_importance
    
    # Update each tree's importance scores based on global knowledge
    for (tree_id, tree) in manager.trees
        if tree.state == RUNNING
            # Blend local and global importance
            for i in 1:length(tree.feature_importance)
                tree.feature_importance[i] = (1.0f0 - learning_rate) * tree.feature_importance[i] + 
                                           learning_rate * global_importance[i]
            end
        end
    end
end

"""
Get forest statistics and status information
"""
function get_forest_status(manager::ForestManager)
    lock(manager.sync_lock) do
        total_trees = length(manager.trees)
        active_count = length(manager.active_trees)
        paused_count = length(manager.paused_trees)
        completed_count = length(manager.completed_trees)
        failed_count = length(manager.failed_trees)
        
        # Calculate resource usage
        total_gpu_memory = sum(values(manager.gpu_memory_usage))
        avg_gpu_memory = total_trees > 0 ? total_gpu_memory / total_trees : 0
        
        # Calculate performance metrics
        runtime = (now() - manager.creation_time).value / 1000.0  # seconds
        avg_iterations_per_tree = total_trees > 0 ? manager.total_iterations / total_trees : 0
        avg_evaluations_per_tree = total_trees > 0 ? manager.total_evaluations / total_trees : 0
        
        # Feature selection statistics
        consensus_count = sum(manager.consensus_features)
        avg_importance = mean(manager.global_feature_importance)
        max_importance = maximum(manager.global_feature_importance)
        
        return Dict{String, Any}(
            "forest_state" => manager.forest_state,
            "total_trees" => total_trees,
            "active_trees" => active_count,
            "paused_trees" => paused_count,
            "completed_trees" => completed_count,
            "failed_trees" => failed_count,
            "total_iterations" => manager.total_iterations,
            "total_evaluations" => manager.total_evaluations,
            "runtime_seconds" => runtime,
            "avg_iterations_per_tree" => avg_iterations_per_tree,
            "avg_evaluations_per_tree" => avg_evaluations_per_tree,
            "total_gpu_memory_mb" => total_gpu_memory / (1024 * 1024),
            "avg_gpu_memory_mb" => avg_gpu_memory / (1024 * 1024),
            "cpu_memory_mb" => manager.cpu_memory_usage / (1024 * 1024),
            "consensus_features" => consensus_count,
            "avg_feature_importance" => avg_importance,
            "max_feature_importance" => max_importance,
            "last_sync_time" => manager.last_sync_time,
            "error_count" => length(manager.error_log)
        )
    end
end

"""
Generate comprehensive forest status report
"""
function generate_forest_report(manager::ForestManager)
    status = get_forest_status(manager)
    
    report = String[]
    
    push!(report, "=== Ensemble Forest Status Report ===")
    push!(report, "Generated: $(now())")
    push!(report, "Forest State: $(status["forest_state"])")
    push!(report, "")
    
    # Tree statistics
    push!(report, "Tree Statistics:")
    push!(report, "  Total Trees: $(status["total_trees"])")
    push!(report, "  Active: $(status["active_trees"])")
    push!(report, "  Paused: $(status["paused_trees"])")
    push!(report, "  Completed: $(status["completed_trees"])")
    push!(report, "  Failed: $(status["failed_trees"])")
    push!(report, "")
    
    # Performance metrics
    push!(report, "Performance Metrics:")
    push!(report, "  Runtime: $(round(status["runtime_seconds"], digits=1)) seconds")
    push!(report, "  Total Iterations: $(status["total_iterations"])")
    push!(report, "  Total Evaluations: $(status["total_evaluations"])")
    push!(report, "  Avg Iterations/Tree: $(round(status["avg_iterations_per_tree"], digits=1))")
    push!(report, "  Avg Evaluations/Tree: $(round(status["avg_evaluations_per_tree"], digits=1))")
    push!(report, "")
    
    # Resource usage
    push!(report, "Resource Usage:")
    push!(report, "  GPU Memory: $(round(status["total_gpu_memory_mb"], digits=1)) MB total")
    push!(report, "  Avg GPU Memory/Tree: $(round(status["avg_gpu_memory_mb"], digits=1)) MB")
    push!(report, "  CPU Memory: $(round(status["cpu_memory_mb"], digits=1)) MB")
    push!(report, "")
    
    # Feature selection
    push!(report, "Feature Selection:")
    push!(report, "  Consensus Features: $(status["consensus_features"])/500")
    push!(report, "  Avg Feature Importance: $(round(status["avg_feature_importance"], digits=4))")
    push!(report, "  Max Feature Importance: $(round(status["max_feature_importance"], digits=4))")
    push!(report, "  Last Sync: $(status["last_sync_time"])")
    push!(report, "")
    
    # Error information
    if status["error_count"] > 0
        push!(report, "Recent Errors:")
        for error in manager.error_log[max(1, length(manager.error_log)-4):end]
            push!(report, "  - $error")
        end
        push!(report, "")
    end
    
    # Individual tree summary
    if !isempty(manager.trees)
        push!(report, "Tree Summary (Top 5 by Score):")
        
        # Sort trees by best score
        sorted_trees = sort(collect(manager.trees), by = x -> x[2].best_score, rev = true)
        
        for (i, (tree_id, tree)) in enumerate(sorted_trees[1:min(5, length(sorted_trees))])
            push!(report, @sprintf("  %d. Tree %d: score=%.4f, iter=%d, state=%s", 
                                 i, tree_id, tree.best_score, tree.iteration_count, tree.state))
        end
    end
    
    push!(report, "")
    push!(report, "=== End Forest Report ===")
    
    return join(report, "\n")
end

"""
Cleanup forest resources and terminate all trees
"""
function cleanup_forest!(manager::ForestManager)
    lock(manager.sync_lock) do
        # Terminate all trees
        for tree_id in keys(manager.trees)
            try
                tree = manager.trees[tree_id]
                tree.state = TERMINATED
                tree.last_update = now()
            catch e
                @warn "Error terminating tree $tree_id: $e"
            end
        end
        
        # Clear tracking sets
        empty!(manager.active_trees)
        empty!(manager.paused_trees)
        empty!(manager.completed_trees)
        empty!(manager.failed_trees)
        
        # Reset resource usage
        for device_id in keys(manager.gpu_memory_usage)
            manager.gpu_memory_usage[device_id] = 0
        end
        manager.cpu_memory_usage = 0
        
        # Update state
        manager.forest_state = "terminated"
        
        if manager.debug_enabled
            @info "Forest cleanup completed"
        end
    end
end

"""
Get tree by ID with error checking
"""
function get_tree(manager::ForestManager, tree_id::Int)
    if !haskey(manager.trees, tree_id)
        error("Tree $tree_id not found in forest")
    end
    return manager.trees[tree_id]
end

"""
Update tree performance metrics
"""
function update_tree_metrics!(tree::MCTSTree, iteration_count::Int, evaluation_count::Int, 
                              current_score::Float32, gpu_memory_used::Int = 0)
    tree.iteration_count = iteration_count
    tree.evaluation_count = evaluation_count
    tree.current_score = current_score
    tree.best_score = max(tree.best_score, current_score)
    tree.gpu_memory_used = gpu_memory_used
    tree.last_update = now()
end

"""
Update tree feature selection and importance
"""
function update_tree_features!(tree::MCTSTree, selected_features::Vector{Bool}, 
                               feature_importance::Vector{Float32})
    if length(selected_features) != 500 || length(feature_importance) != 500
        error("Feature vectors must have exactly 500 elements")
    end
    
    # Store previous selection in history
    push!(tree.selection_history, copy(tree.selected_features))
    
    # Update current selection
    tree.selected_features .= selected_features
    tree.feature_importance .= feature_importance
    tree.last_update = now()
    
    # Limit history size
    max_history = 100
    if length(tree.selection_history) > max_history
        deleteat!(tree.selection_history, 1)
    end
end

"""
Get trees by state
"""
function get_trees_by_state(manager::ForestManager, state::TreeState)
    return [tree for tree in values(manager.trees) if tree.state == state]
end

"""
Get top performing trees
"""
function get_top_trees(manager::ForestManager, n::Int = 10)
    sorted_trees = sort(collect(values(manager.trees)), by = x -> x.best_score, rev = true)
    return sorted_trees[1:min(n, length(sorted_trees))]
end

# Export main types and functions
export TreeState, MCTSTree, TreePoolConfig, ForestManager
export create_tree_pool_config, initialize_forest_manager
export create_tree!, remove_tree!, start_tree!, pause_tree!, complete_tree!, fail_tree!
export start_all_trees!, pause_all_trees!
export synchronize_features!, get_forest_status, generate_forest_report
export cleanup_forest!, get_tree, update_tree_metrics!, update_tree_features!
export get_trees_by_state, get_top_trees

end # module EnsembleForest