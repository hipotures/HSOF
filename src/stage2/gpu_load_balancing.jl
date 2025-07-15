"""
Dynamic Load Balancing Across GPUs for MCTS Ensemble Feature Selection
Implements load balancing system maintaining optimal GPU utilization throughout ensemble execution,
including GPU monitoring, work stealing, tree migration, load prediction, and adaptive batch sizing
for dual RTX 4090 configuration.

This module provides intelligent workload distribution across multiple GPUs with real-time monitoring,
predictive load balancing, and dynamic work redistribution to maximize utilization efficiency.
"""

module GPULoadBalancing

using Random
using Statistics
using Dates
using Printf
using LinearAlgebra

# Import ensemble forest for tree management
include("ensemble_forest.jl")
using .EnsembleForest

# Import diversity mechanisms for metrics
include("diversity_mechanisms.jl")
using .DiversityMechanisms

"""
GPU device information and capabilities
"""
struct GPUDevice
    device_id::Int                    # GPU device ID (0, 1, etc.)
    device_name::String              # GPU model name
    total_memory::Int                # Total GPU memory in bytes
    compute_capability::Tuple{Int, Int} # CUDA compute capability (major, minor)
    multiprocessor_count::Int        # Number of streaming multiprocessors
    max_threads_per_block::Int       # Maximum threads per block
    max_shared_memory::Int           # Maximum shared memory per block
    memory_bandwidth::Float64        # Memory bandwidth in GB/s
    is_available::Bool              # Whether GPU is available for use
end

"""
Create GPU device information
"""
function create_gpu_device(device_id::Int; 
                          device_name::String = "RTX 4090",
                          total_memory::Int = 24 * 1024^3,  # 24GB
                          compute_capability::Tuple{Int, Int} = (8, 9),
                          multiprocessor_count::Int = 128,
                          max_threads_per_block::Int = 1024,
                          max_shared_memory::Int = 49152,
                          memory_bandwidth::Float64 = 1008.0,  # GB/s
                          is_available::Bool = true)
    return GPUDevice(
        device_id, device_name, total_memory, compute_capability,
        multiprocessor_count, max_threads_per_block, max_shared_memory,
        memory_bandwidth, is_available
    )
end

"""
Real-time GPU utilization metrics
"""
mutable struct GPUMetrics
    device_id::Int                   # GPU device ID
    utilization_gpu::Float32         # GPU utilization percentage (0-100)
    utilization_memory::Float32      # Memory utilization percentage (0-100)
    memory_used::Int                 # Used memory in bytes
    memory_free::Int                 # Free memory in bytes
    temperature::Float32             # GPU temperature in Celsius
    power_draw::Float32              # Power consumption in watts
    fan_speed::Float32               # Fan speed percentage
    
    # Performance metrics
    sm_clock_mhz::Float32           # Streaming multiprocessor clock MHz
    memory_clock_mhz::Float32       # Memory clock MHz
    throughput_samples_per_sec::Float64  # Computational throughput
    
    # Workload metrics
    active_trees::Int               # Currently active trees on this GPU
    pending_operations::Int         # Queued operations
    completed_operations::Int       # Completed operations since last reset
    
    # Timing metrics
    last_update::DateTime           # Last metrics update time
    update_count::Int               # Number of metrics updates
    average_update_interval::Float64 # Average time between updates (ms)
end

"""
Create new GPU metrics tracker
"""
function create_gpu_metrics(device_id::Int)
    return GPUMetrics(
        device_id, 0.0f0, 0.0f0, 0, 0, 0.0f0, 0.0f0, 0.0f0,
        0.0f0, 0.0f0, 0.0,
        0, 0, 0,
        now(), 0, 0.0
    )
end

"""
Work stealing strategy for load balancing
"""
@enum WorkStealingStrategy begin
    GREEDY_STEALING = 1             # Steal from most overloaded GPU
    ROUND_ROBIN_STEALING = 2        # Steal in round-robin fashion
    ADAPTIVE_STEALING = 3           # Adaptive stealing based on performance
    PREDICTIVE_STEALING = 4         # Steal based on predicted workload
end

"""
Tree migration safety level
"""
@enum MigrationSafety begin
    SAFE_MIGRATION = 1              # Only migrate trees at safe points
    CHECKPOINT_MIGRATION = 2        # Migrate with checkpointing
    IMMEDIATE_MIGRATION = 3         # Migrate immediately if needed
end

"""
Load prediction model type
"""
@enum LoadPredictionModel begin
    LINEAR_PREDICTION = 1           # Linear trend prediction
    EXPONENTIAL_PREDICTION = 2      # Exponential smoothing
    NEURAL_PREDICTION = 3           # Neural network prediction
    ENSEMBLE_PREDICTION = 4         # Ensemble of prediction models
end

"""
Tree workload estimation
"""
mutable struct TreeWorkload
    tree_id::Int                    # Tree identifier
    estimated_iterations::Int       # Estimated remaining iterations
    computational_complexity::Float64 # Computational complexity score
    memory_requirements::Int        # Memory requirements in bytes
    io_intensity::Float64          # I/O intensity score
    last_performance::Float64       # Last measured performance
    prediction_confidence::Float32  # Confidence in workload prediction
    migration_cost::Float64         # Cost of migrating this tree
    last_update::DateTime           # Last workload update
end

"""
Create tree workload estimation
"""
function create_tree_workload(tree_id::Int; 
                             estimated_iterations::Int = 1000,
                             computational_complexity::Float64 = 1.0,
                             memory_requirements::Int = 1024^3,  # 1GB default
                             io_intensity::Float64 = 1.0,
                             last_performance::Float64 = 1.0,
                             prediction_confidence::Float32 = 0.5f0,
                             migration_cost::Float64 = 100.0)
    return TreeWorkload(
        tree_id, estimated_iterations, computational_complexity,
        memory_requirements, io_intensity, last_performance,
        prediction_confidence, migration_cost, now()
    )
end

"""
Load balancing configuration
"""
struct LoadBalancingConfig
    # Monitoring configuration
    monitoring_interval_ms::Int
    metrics_history_size::Int
    enable_detailed_monitoring::Bool
    
    # Load balancing thresholds
    utilization_threshold_high::Float32
    utilization_threshold_low::Float32
    memory_threshold_high::Float32
    imbalance_threshold::Float32
    
    # Work stealing configuration
    work_stealing_strategy::WorkStealingStrategy
    enable_work_stealing::Bool
    stealing_aggressiveness::Float32
    min_trees_for_stealing::Int
    
    # Tree migration configuration
    migration_safety::MigrationSafety
    enable_tree_migration::Bool
    migration_cooldown_ms::Int
    max_concurrent_migrations::Int
    
    # Load prediction configuration
    prediction_model::LoadPredictionModel
    enable_load_prediction::Bool
    prediction_window_size::Int
    prediction_confidence_threshold::Float32
    
    # Adaptive batch sizing
    enable_adaptive_batching::Bool
    min_batch_size::Int
    max_batch_size::Int
    batch_size_adjustment_factor::Float32
    
    # Performance optimization
    enable_async_monitoring::Bool
    enable_predictive_scheduling::Bool
    enable_workload_caching::Bool
    cache_size::Int
    
    # Fault tolerance
    enable_fault_tolerance::Bool
    gpu_failure_timeout_ms::Int
    automatic_rebalancing::Bool
    fallback_single_gpu::Bool
end

"""
Create default load balancing configuration
"""
function create_load_balancing_config(;
    monitoring_interval_ms::Int = 100,
    metrics_history_size::Int = 600,  # 60 seconds at 100ms intervals
    enable_detailed_monitoring::Bool = true,
    utilization_threshold_high::Float32 = 85.0f0,
    utilization_threshold_low::Float32 = 30.0f0,
    memory_threshold_high::Float32 = 80.0f0,
    imbalance_threshold::Float32 = 20.0f0,
    work_stealing_strategy::WorkStealingStrategy = ADAPTIVE_STEALING,
    enable_work_stealing::Bool = true,
    stealing_aggressiveness::Float32 = 0.7f0,
    min_trees_for_stealing::Int = 5,
    migration_safety::MigrationSafety = CHECKPOINT_MIGRATION,
    enable_tree_migration::Bool = true,
    migration_cooldown_ms::Int = 5000,
    max_concurrent_migrations::Int = 3,
    prediction_model::LoadPredictionModel = EXPONENTIAL_PREDICTION,
    enable_load_prediction::Bool = true,
    prediction_window_size::Int = 50,
    prediction_confidence_threshold::Float32 = 0.7f0,
    enable_adaptive_batching::Bool = true,
    min_batch_size::Int = 1,
    max_batch_size::Int = 100,
    batch_size_adjustment_factor::Float32 = 1.2f0,
    enable_async_monitoring::Bool = true,
    enable_predictive_scheduling::Bool = true,
    enable_workload_caching::Bool = true,
    cache_size::Int = 1000,
    enable_fault_tolerance::Bool = true,
    gpu_failure_timeout_ms::Int = 30000,
    automatic_rebalancing::Bool = true,
    fallback_single_gpu::Bool = true
)
    return LoadBalancingConfig(
        monitoring_interval_ms, metrics_history_size, enable_detailed_monitoring,
        utilization_threshold_high, utilization_threshold_low, memory_threshold_high, imbalance_threshold,
        work_stealing_strategy, enable_work_stealing, stealing_aggressiveness, min_trees_for_stealing,
        migration_safety, enable_tree_migration, migration_cooldown_ms, max_concurrent_migrations,
        prediction_model, enable_load_prediction, prediction_window_size, prediction_confidence_threshold,
        enable_adaptive_batching, min_batch_size, max_batch_size, batch_size_adjustment_factor,
        enable_async_monitoring, enable_predictive_scheduling, enable_workload_caching, cache_size,
        enable_fault_tolerance, gpu_failure_timeout_ms, automatic_rebalancing, fallback_single_gpu
    )
end

"""
Load balancing statistics
"""
mutable struct LoadBalancingStats
    total_rebalancing_operations::Int
    successful_migrations::Int
    failed_migrations::Int
    work_stealing_events::Int
    load_prediction_accuracy::Float64
    average_gpu_utilization::Float64
    utilization_variance::Float64
    total_monitoring_updates::Int
    last_rebalancing_time::DateTime
    performance_improvement_ratio::Float64
end

"""
Initialize load balancing statistics
"""
function initialize_load_balancing_stats()
    return LoadBalancingStats(
        0, 0, 0, 0, 0.0, 0.0, 0.0, 0, now(), 1.0
    )
end

"""
Dynamic GPU load balancing manager
Coordinates load balancing across multiple GPUs with monitoring and optimization
"""
mutable struct GPULoadBalancer
    # Configuration
    config::LoadBalancingConfig
    
    # GPU management
    gpu_devices::Dict{Int, GPUDevice}        # Device ID -> GPU info
    gpu_metrics::Dict{Int, GPUMetrics}       # Device ID -> current metrics
    metrics_history::Dict{Int, Vector{GPUMetrics}} # Device ID -> metrics history
    
    # Workload tracking
    tree_assignments::Dict{Int, Int}         # Tree ID -> GPU device ID
    tree_workloads::Dict{Int, TreeWorkload}  # Tree ID -> workload estimation
    gpu_workloads::Dict{Int, Vector{Int}}    # Device ID -> assigned tree IDs
    
    # Load prediction
    load_predictions::Dict{Int, Vector{Float64}} # Device ID -> predicted loads
    prediction_accuracy::Dict{Int, Float64}   # Device ID -> prediction accuracy
    
    # Migration tracking
    active_migrations::Dict{Int, DateTime}   # Tree ID -> migration start time
    migration_history::Vector{Tuple{Int, Int, Int, DateTime}} # (tree_id, from_gpu, to_gpu, timestamp)
    last_rebalancing::DateTime               # Last rebalancing operation time
    
    # Performance monitoring
    stats::LoadBalancingStats
    monitoring_task::Union{Task, Nothing}   # Background monitoring task
    
    # Synchronization
    balancer_lock::ReentrantLock
    metrics_lock::ReentrantLock
    migration_lock::ReentrantLock
    
    # State management
    manager_state::String
    error_log::Vector{String}
    creation_time::DateTime
end

"""
Initialize GPU load balancer
"""
function initialize_gpu_load_balancer(gpu_count::Int = 2, 
                                     config::LoadBalancingConfig = create_load_balancing_config())
    # Create GPU devices
    gpu_devices = Dict{Int, GPUDevice}()
    gpu_metrics = Dict{Int, GPUMetrics}()
    metrics_history = Dict{Int, Vector{GPUMetrics}}()
    gpu_workloads = Dict{Int, Vector{Int}}()
    load_predictions = Dict{Int, Vector{Float64}}()
    prediction_accuracy = Dict{Int, Float64}()
    
    for device_id in 0:(gpu_count-1)
        gpu_devices[device_id] = create_gpu_device(device_id)
        gpu_metrics[device_id] = create_gpu_metrics(device_id)
        metrics_history[device_id] = GPUMetrics[]
        gpu_workloads[device_id] = Int[]
        load_predictions[device_id] = Float64[]
        prediction_accuracy[device_id] = 0.0
    end
    
    balancer = GPULoadBalancer(
        config,
        gpu_devices,
        gpu_metrics,
        metrics_history,
        Dict{Int, Int}(),
        Dict{Int, TreeWorkload}(),
        gpu_workloads,
        load_predictions,
        prediction_accuracy,
        Dict{Int, DateTime}(),
        Tuple{Int, Int, Int, DateTime}[],
        now(),
        initialize_load_balancing_stats(),
        nothing,
        ReentrantLock(),
        ReentrantLock(),
        ReentrantLock(),
        "active",
        String[],
        now()
    )
    
    @info "GPU load balancer initialized with $gpu_count GPUs"
    return balancer
end

"""
Start GPU monitoring task
"""
function start_monitoring!(balancer::GPULoadBalancer)
    if !isnothing(balancer.monitoring_task) && !istaskdone(balancer.monitoring_task)
        @warn "Monitoring task already running"
        return
    end
    
    balancer.monitoring_task = @async begin
        try
            while balancer.manager_state == "active"
                update_gpu_metrics!(balancer)
                
                if balancer.config.enable_load_prediction
                    update_load_predictions!(balancer)
                end
                
                if balancer.config.automatic_rebalancing
                    check_and_rebalance!(balancer)
                end
                
                sleep(balancer.config.monitoring_interval_ms / 1000.0)
            end
        catch e
            push!(balancer.error_log, "Monitoring task error: $e")
            @error "GPU monitoring task failed" exception=e
        end
    end
    
    @info "GPU monitoring started with $(balancer.config.monitoring_interval_ms)ms intervals"
end

"""
Update GPU metrics by sampling hardware
"""
function update_gpu_metrics!(balancer::GPULoadBalancer)
    lock(balancer.metrics_lock) do
        for (device_id, metrics) in balancer.gpu_metrics
            start_time = time()
            
            # Mock GPU metrics update (in real implementation, use CUDA/NVML)
            update_mock_gpu_metrics!(metrics, balancer.gpu_workloads[device_id])
            
            # Update timing
            metrics.last_update = now()
            metrics.update_count += 1
            
            update_time = (time() - start_time) * 1000
            if metrics.update_count > 1
                metrics.average_update_interval = (metrics.average_update_interval * (metrics.update_count - 1) + update_time) / metrics.update_count
            else
                metrics.average_update_interval = update_time
            end
            
            # Store in history
            if length(balancer.metrics_history[device_id]) >= balancer.config.metrics_history_size
                deleteat!(balancer.metrics_history[device_id], 1)
            end
            push!(balancer.metrics_history[device_id], deepcopy(metrics))
        end
        
        balancer.stats.total_monitoring_updates += 1
        
        # Update aggregated statistics
        update_aggregated_stats!(balancer)
    end
end

"""
Mock GPU metrics update (replace with real NVML/CUDA calls in production)
"""
function update_mock_gpu_metrics!(metrics::GPUMetrics, assigned_trees::Vector{Int})
    # Simulate realistic GPU metrics based on workload
    tree_count = length(assigned_trees)
    base_utilization = min(100.0f0, tree_count * 2.0f0)  # 2% per tree
    
    # Add some realistic noise
    noise = (rand() - 0.5) * 10.0f0
    metrics.utilization_gpu = max(0.0f0, min(100.0f0, base_utilization + noise))
    
    # Memory utilization roughly correlates with tree count
    metrics.utilization_memory = min(100.0f0, tree_count * 1.5f0 + (rand() - 0.5) * 5.0f0)
    
    # Temperature and power scale with utilization
    metrics.temperature = 30.0f0 + metrics.utilization_gpu * 0.6f0
    metrics.power_draw = 50.0f0 + metrics.utilization_gpu * 4.0f0
    metrics.fan_speed = max(0.0f0, min(100.0f0, (metrics.temperature - 40.0f0) * 2.0f0))
    
    # Clock speeds
    metrics.sm_clock_mhz = 1500.0f0 + metrics.utilization_gpu * 10.0f0
    metrics.memory_clock_mhz = 9000.0f0 + metrics.utilization_gpu * 50.0f0
    
    # Throughput
    metrics.throughput_samples_per_sec = tree_count * 100.0 * (1.0 + rand() * 0.2)
    
    # Workload metrics
    metrics.active_trees = tree_count
    metrics.pending_operations = max(0, tree_count - 10 + round(Int, (rand() - 0.5) * 20))
    metrics.completed_operations += tree_count + round(Int, rand() * 10)
    
    # Memory usage (mock values)
    total_memory = 24 * 1024^3  # 24GB
    metrics.memory_used = round(Int, total_memory * metrics.utilization_memory / 100.0)
    metrics.memory_free = total_memory - metrics.memory_used
end

"""
Update aggregated statistics
"""
function update_aggregated_stats!(balancer::GPULoadBalancer)
    utilizations = [metrics.utilization_gpu for metrics in values(balancer.gpu_metrics)]
    
    if !isempty(utilizations)
        balancer.stats.average_gpu_utilization = mean(utilizations)
        balancer.stats.utilization_variance = var(utilizations)
    end
end

"""
Assign tree to GPU device
"""
function assign_tree_to_gpu!(balancer::GPULoadBalancer, tree_id::Int, device_id::Int)
    lock(balancer.balancer_lock) do
        # Remove from current GPU if assigned
        current_gpu = get(balancer.tree_assignments, tree_id, nothing)
        if !isnothing(current_gpu)
            filter!(id -> id != tree_id, balancer.gpu_workloads[current_gpu])
        end
        
        # Assign to new GPU
        balancer.tree_assignments[tree_id] = device_id
        push!(balancer.gpu_workloads[device_id], tree_id)
        
        # Create workload estimation if not exists
        if !haskey(balancer.tree_workloads, tree_id)
            balancer.tree_workloads[tree_id] = create_tree_workload(tree_id)
        end
        
        @debug "Assigned tree $tree_id to GPU $device_id"
    end
end

"""
Update tree workload estimation
"""
function update_tree_workload!(balancer::GPULoadBalancer, 
                              tree_id::Int, 
                              iterations_remaining::Int,
                              computational_complexity::Float64 = 1.0,
                              memory_usage::Int = 1024^3,
                              performance::Float64 = 1.0)
    workload = get(balancer.tree_workloads, tree_id, create_tree_workload(tree_id))
    
    # Update workload parameters
    workload.estimated_iterations = iterations_remaining
    workload.computational_complexity = computational_complexity
    workload.memory_requirements = memory_usage
    workload.last_performance = performance
    workload.last_update = now()
    
    # Update prediction confidence based on recent accuracy
    device_id = get(balancer.tree_assignments, tree_id, 0)
    accuracy = get(balancer.prediction_accuracy, device_id, 0.5)
    workload.prediction_confidence = Float32(accuracy)
    
    balancer.tree_workloads[tree_id] = workload
    @debug "Updated workload for tree $tree_id: $iterations_remaining iterations, complexity $computational_complexity"
end

"""
Check if rebalancing is needed and perform if necessary
"""
function check_and_rebalance!(balancer::GPULoadBalancer)
    if !should_rebalance(balancer)
        return
    end
    
    # Check cooldown period
    time_since_last = (now() - balancer.last_rebalancing).value
    if time_since_last < balancer.config.migration_cooldown_ms
        return
    end
    
    lock(balancer.balancer_lock) do
        if balancer.config.enable_work_stealing
            perform_work_stealing!(balancer)
        end
        
        if balancer.config.enable_tree_migration
            perform_tree_migration!(balancer)
        end
        
        balancer.last_rebalancing = now()
        balancer.stats.total_rebalancing_operations += 1
    end
end

"""
Determine if rebalancing is needed
"""
function should_rebalance(balancer::GPULoadBalancer)::Bool
    utilizations = [metrics.utilization_gpu for metrics in values(balancer.gpu_metrics)]
    
    if length(utilizations) < 2
        return false
    end
    
    max_util = maximum(utilizations)
    min_util = minimum(utilizations)
    imbalance = max_util - min_util
    
    # Check if imbalance exceeds threshold
    if imbalance > balancer.config.imbalance_threshold
        @debug "GPU imbalance detected: max=$max_util%, min=$min_util%, diff=$imbalance%"
        return true
    end
    
    # Check if any GPU is overloaded
    if max_util > balancer.config.utilization_threshold_high
        @debug "GPU overload detected: $max_util% > $(balancer.config.utilization_threshold_high)%"
        return true
    end
    
    return false
end

"""
Perform work stealing between GPUs
"""
function perform_work_stealing!(balancer::GPULoadBalancer)
    # Find source and target GPUs
    source_gpu, target_gpu = select_gpus_for_stealing(balancer)
    
    if isnothing(source_gpu) || isnothing(target_gpu)
        return
    end
    
    # Select trees to steal
    trees_to_steal = select_trees_for_stealing(balancer, source_gpu, target_gpu)
    
    if isempty(trees_to_steal)
        return
    end
    
    # Perform the stealing
    for tree_id in trees_to_steal
        if can_migrate_tree(balancer, tree_id)
            migrate_tree!(balancer, tree_id, source_gpu, target_gpu)
            balancer.stats.work_stealing_events += 1
            @info "Work stealing: moved tree $tree_id from GPU $source_gpu to GPU $target_gpu"
        end
    end
end

"""
Select source and target GPUs for work stealing
"""
function select_gpus_for_stealing(balancer::GPULoadBalancer)
    utilizations = [(device_id, metrics.utilization_gpu) for (device_id, metrics) in balancer.gpu_metrics]
    sort!(utilizations, by=x->x[2], rev=true)
    
    if length(utilizations) < 2
        return nothing, nothing
    end
    
    source_gpu = utilizations[1][1]  # Most loaded
    target_gpu = utilizations[end][1]  # Least loaded
    
    # Check if stealing makes sense
    source_util = utilizations[1][2]
    target_util = utilizations[end][2]
    
    if source_util - target_util < balancer.config.imbalance_threshold
        return nothing, nothing
    end
    
    # Check minimum trees requirement
    source_trees = length(balancer.gpu_workloads[source_gpu])
    if source_trees < balancer.config.min_trees_for_stealing
        return nothing, nothing
    end
    
    return source_gpu, target_gpu
end

"""
Select trees for stealing based on strategy
"""
function select_trees_for_stealing(balancer::GPULoadBalancer, source_gpu::Int, target_gpu::Int)::Vector{Int}
    source_trees = balancer.gpu_workloads[source_gpu]
    
    if isempty(source_trees)
        return Int[]
    end
    
    # Calculate how many trees to steal
    source_util = balancer.gpu_metrics[source_gpu].utilization_gpu
    target_util = balancer.gpu_metrics[target_gpu].utilization_gpu
    imbalance = source_util - target_util
    
    # Steal proportional to imbalance and aggressiveness
    trees_to_steal_count = max(1, round(Int, length(source_trees) * (imbalance / 100.0) * balancer.config.stealing_aggressiveness))
    trees_to_steal_count = min(trees_to_steal_count, length(source_trees) ÷ 2)  # Don't steal more than half
    
    if trees_to_steal_count <= 0
        return Int[]
    end
    
    # Select trees based on strategy
    if balancer.config.work_stealing_strategy == GREEDY_STEALING
        # Steal trees with highest estimated workload
        tree_workloads = [(tree_id, get(balancer.tree_workloads, tree_id, create_tree_workload(tree_id)).computational_complexity) 
                         for tree_id in source_trees]
        sort!(tree_workloads, by=x->x[2], rev=true)
        return [pair[1] for pair in tree_workloads[1:trees_to_steal_count]]
        
    elseif balancer.config.work_stealing_strategy == ADAPTIVE_STEALING
        # Steal trees with best performance-to-cost ratio
        tree_scores = Float64[]
        for tree_id in source_trees
            workload = get(balancer.tree_workloads, tree_id, create_tree_workload(tree_id))
            score = workload.last_performance / max(1.0, workload.migration_cost)
            push!(tree_scores, score)
        end
        
        sorted_indices = sortperm(tree_scores, rev=true)
        return source_trees[sorted_indices[1:trees_to_steal_count]]
        
    else  # ROUND_ROBIN_STEALING or default
        # Simple round-robin selection
        return source_trees[1:trees_to_steal_count]
    end
end

"""
Check if tree can be migrated safely
"""
function can_migrate_tree(balancer::GPULoadBalancer, tree_id::Int)::Bool
    # Check if tree is already being migrated
    if haskey(balancer.active_migrations, tree_id)
        return false
    end
    
    # Check concurrent migration limit
    if length(balancer.active_migrations) >= balancer.config.max_concurrent_migrations
        return false
    end
    
    # Check migration safety level
    if balancer.config.migration_safety == SAFE_MIGRATION
        # In real implementation, check if tree is at a safe migration point
        return true  # Mock: always safe
    end
    
    return true
end

"""
Migrate tree between GPUs
"""
function migrate_tree!(balancer::GPULoadBalancer, tree_id::Int, from_gpu::Int, to_gpu::Int)
    lock(balancer.migration_lock) do
        # Record migration start
        balancer.active_migrations[tree_id] = now()
        
        try
            # Perform actual migration (mock implementation)
            migration_successful = perform_tree_migration(balancer, tree_id, from_gpu, to_gpu)
            
            if migration_successful
                # Update assignments
                assign_tree_to_gpu!(balancer, tree_id, to_gpu)
                
                # Record successful migration
                push!(balancer.migration_history, (tree_id, from_gpu, to_gpu, now()))
                balancer.stats.successful_migrations += 1
                
                @info "Successfully migrated tree $tree_id from GPU $from_gpu to GPU $to_gpu"
                
            else
                balancer.stats.failed_migrations += 1
                @warn "Failed to migrate tree $tree_id from GPU $from_gpu to GPU $to_gpu"
            end
            
        catch e
            balancer.stats.failed_migrations += 1
            push!(balancer.error_log, "Migration error for tree $tree_id: $e")
            @error "Tree migration failed" tree_id=tree_id from_gpu=from_gpu to_gpu=to_gpu exception=e
            
        finally
            # Remove from active migrations
            delete!(balancer.active_migrations, tree_id)
        end
    end
end

"""
Perform actual tree migration (mock implementation)
"""
function perform_tree_migration(balancer::GPULoadBalancer, tree_id::Int, from_gpu::Int, to_gpu::Int)::Bool
    # Mock migration - in real implementation:
    # 1. Checkpoint tree state
    # 2. Transfer data between GPUs
    # 3. Resume tree on target GPU
    # 4. Verify migration success
    
    # Simulate migration time
    sleep(0.01)  # 10ms migration time
    
    # Mock success rate (95%)
    return rand() < 0.95
end

"""
Perform tree migration for load balancing
"""
function perform_tree_migration!(balancer::GPULoadBalancer)
    # Similar to work stealing but more conservative
    # Focus on moving trees that will significantly improve balance
    
    utilizations = [(device_id, metrics.utilization_gpu) for (device_id, metrics) in balancer.gpu_metrics]
    sort!(utilizations, by=x->x[2], rev=true)
    
    if length(utilizations) < 2
        return
    end
    
    overloaded_gpus = [gpu for (gpu, util) in utilizations if util > balancer.config.utilization_threshold_high]
    underloaded_gpus = [gpu for (gpu, util) in utilizations if util < balancer.config.utilization_threshold_low]
    
    # Migrate trees from overloaded to underloaded GPUs
    for source_gpu in overloaded_gpus
        for target_gpu in underloaded_gpus
            # Find best tree to migrate
            candidate_trees = balancer.gpu_workloads[source_gpu]
            
            if length(candidate_trees) <= 1
                continue  # Don't leave GPU empty
            end
            
            # Select tree with highest migration benefit
            best_tree = select_best_migration_candidate(balancer, candidate_trees, source_gpu, target_gpu)
            
            if !isnothing(best_tree) && can_migrate_tree(balancer, best_tree)
                migrate_tree!(balancer, best_tree, source_gpu, target_gpu)
                break  # One migration per rebalancing round
            end
        end
    end
end

"""
Select best tree for migration
"""
function select_best_migration_candidate(balancer::GPULoadBalancer, 
                                       candidate_trees::Vector{Int}, 
                                       source_gpu::Int, 
                                       target_gpu::Int)::Union{Int, Nothing}
    if isempty(candidate_trees)
        return nothing
    end
    
    best_tree = nothing
    best_score = -Inf
    
    for tree_id in candidate_trees
        workload = get(balancer.tree_workloads, tree_id, create_tree_workload(tree_id))
        
        # Calculate migration benefit score
        benefit = workload.computational_complexity * workload.prediction_confidence
        cost = workload.migration_cost
        score = benefit / max(1.0, cost)
        
        if score > best_score
            best_score = score
            best_tree = tree_id
        end
    end
    
    return best_tree
end

"""
Update load predictions for all GPUs
"""
function update_load_predictions!(balancer::GPULoadBalancer)
    for device_id in keys(balancer.gpu_devices)
        update_gpu_load_prediction!(balancer, device_id)
    end
end

"""
Update load prediction for specific GPU
"""
function update_gpu_load_prediction!(balancer::GPULoadBalancer, device_id::Int)
    history = balancer.metrics_history[device_id]
    
    if length(history) < 5  # Need minimum history for prediction
        return
    end
    
    # Extract utilization history
    utilization_history = [metrics.utilization_gpu for metrics in history[max(1, end-balancer.config.prediction_window_size):end]]
    
    # Predict next values
    predicted_load = predict_load(balancer.config.prediction_model, utilization_history)
    
    # Store prediction
    if length(balancer.load_predictions[device_id]) >= balancer.config.prediction_window_size
        deleteat!(balancer.load_predictions[device_id], 1)
    end
    push!(balancer.load_predictions[device_id], predicted_load)
    
    # Update prediction accuracy
    if length(history) > 1
        actual = history[end].utilization_gpu
        if length(balancer.load_predictions[device_id]) > 1
            predicted = balancer.load_predictions[device_id][end-1]
            error = abs(actual - predicted) / max(1.0, actual)
            accuracy = max(0.0, 1.0 - error)
            
            # Update running average of accuracy
            current_accuracy = get(balancer.prediction_accuracy, device_id, 0.5)
            balancer.prediction_accuracy[device_id] = 0.9 * current_accuracy + 0.1 * accuracy
        end
    end
end

"""
Predict load using specified model
"""
function predict_load(model::LoadPredictionModel, history::Vector{Float32})::Float64
    if isempty(history)
        return 0.0
    end
    
    if model == LINEAR_PREDICTION
        # Simple linear trend
        if length(history) >= 2
            trend = history[end] - history[end-1]
            return Float64(history[end] + trend)
        else
            return Float64(history[end])
        end
        
    elseif model == EXPONENTIAL_PREDICTION
        # Exponential smoothing
        alpha = 0.3  # Smoothing factor
        prediction = Float64(history[1])
        for i in 2:length(history)
            prediction = alpha * history[i] + (1 - alpha) * prediction
        end
        return prediction
        
    else  # Default to mean
        return mean(history)
    end
end

"""
Calculate optimal batch size for GPU
"""
function calculate_optimal_batch_size(balancer::GPULoadBalancer, device_id::Int)::Int
    if !balancer.config.enable_adaptive_batching
        return balancer.config.max_batch_size ÷ 2  # Default middle value
    end
    
    metrics = balancer.gpu_metrics[device_id]
    base_batch_size = balancer.config.max_batch_size ÷ 2
    
    # Adjust based on GPU utilization
    utilization_factor = 1.0 - (metrics.utilization_gpu / 100.0)  # Higher utilization = smaller batches
    memory_factor = 1.0 - (metrics.utilization_memory / 100.0)    # Higher memory usage = smaller batches
    
    adjustment_factor = min(utilization_factor, memory_factor) * balancer.config.batch_size_adjustment_factor
    optimal_size = round(Int, base_batch_size * (1.0 + adjustment_factor))
    
    return clamp(optimal_size, balancer.config.min_batch_size, balancer.config.max_batch_size)
end

"""
Get load balancing status
"""
function get_load_balancing_status(balancer::GPULoadBalancer)
    lock(balancer.balancer_lock) do
        return Dict{String, Any}(
            "manager_state" => balancer.manager_state,
            "total_gpus" => length(balancer.gpu_devices),
            "available_gpus" => sum(device.is_available for device in values(balancer.gpu_devices)),
            "total_trees" => length(balancer.tree_assignments),
            "tree_assignments" => copy(balancer.tree_assignments),
            "gpu_utilizations" => Dict(id => metrics.utilization_gpu for (id, metrics) in balancer.gpu_metrics),
            "gpu_memory_usage" => Dict(id => metrics.utilization_memory for (id, metrics) in balancer.gpu_metrics),
            "average_utilization" => balancer.stats.average_gpu_utilization,
            "utilization_variance" => balancer.stats.utilization_variance,
            "total_rebalancing_ops" => balancer.stats.total_rebalancing_operations,
            "successful_migrations" => balancer.stats.successful_migrations,
            "failed_migrations" => balancer.stats.failed_migrations,
            "work_stealing_events" => balancer.stats.work_stealing_events,
            "active_migrations" => length(balancer.active_migrations),
            "last_rebalancing" => balancer.last_rebalancing,
            "monitoring_active" => !isnothing(balancer.monitoring_task) && !istaskdone(balancer.monitoring_task)
        )
    end
end

"""
Generate load balancing report
"""
function generate_load_balancing_report(balancer::GPULoadBalancer)
    status = get_load_balancing_status(balancer)
    
    report = String[]
    
    push!(report, "=== GPU Load Balancing Report ===")
    push!(report, "Generated: $(now())")
    push!(report, "Manager State: $(status["manager_state"])")
    push!(report, "")
    
    # GPU overview
    push!(report, "GPU Overview:")
    push!(report, "  Total GPUs: $(status["total_gpus"])")
    push!(report, "  Available GPUs: $(status["available_gpus"])")
    push!(report, "  Total Trees: $(status["total_trees"])")
    push!(report, "")
    
    # Load distribution
    push!(report, "Load Distribution:")
    for (gpu_id, utilization) in status["gpu_utilizations"]
        memory_usage = status["gpu_memory_usage"][gpu_id]
        tree_count = count(gpu -> gpu == gpu_id, values(status["tree_assignments"]))
        push!(report, "  GPU $gpu_id: $(round(utilization, digits=1))% util, $(round(memory_usage, digits=1))% mem, $tree_count trees")
    end
    push!(report, "")
    
    # Performance metrics
    push!(report, "Performance Metrics:")
    push!(report, "  Average Utilization: $(round(status["average_utilization"], digits=1))%")
    push!(report, "  Utilization Variance: $(round(status["utilization_variance"], digits=2))")
    push!(report, "  Total Rebalancing Operations: $(status["total_rebalancing_ops"])")
    push!(report, "  Successful Migrations: $(status["successful_migrations"])")
    push!(report, "  Failed Migrations: $(status["failed_migrations"])")
    push!(report, "  Work Stealing Events: $(status["work_stealing_events"])")
    push!(report, "")
    
    # Configuration
    push!(report, "Configuration:")
    push!(report, "  Work Stealing: $(balancer.config.enable_work_stealing)")
    push!(report, "  Tree Migration: $(balancer.config.enable_tree_migration)")
    push!(report, "  Load Prediction: $(balancer.config.enable_load_prediction)")
    push!(report, "  Adaptive Batching: $(balancer.config.enable_adaptive_batching)")
    push!(report, "  Monitoring Interval: $(balancer.config.monitoring_interval_ms)ms")
    push!(report, "")
    
    push!(report, "=== End Load Balancing Report ===")
    
    return join(report, "\n")
end

"""
Stop GPU monitoring and cleanup
"""
function stop_monitoring!(balancer::GPULoadBalancer)
    balancer.manager_state = "stopping"
    
    if !isnothing(balancer.monitoring_task)
        # Wait for monitoring task to finish
        try
            wait(balancer.monitoring_task)
        catch
            # Task may have already finished
        end
    end
    
    balancer.manager_state = "stopped"
    @info "GPU monitoring stopped"
end

"""
Cleanup load balancer
"""
function cleanup_load_balancer!(balancer::GPULoadBalancer)
    stop_monitoring!(balancer)
    
    lock(balancer.balancer_lock) do
        empty!(balancer.tree_assignments)
        empty!(balancer.tree_workloads)
        for workloads in values(balancer.gpu_workloads)
            empty!(workloads)
        end
        empty!(balancer.active_migrations)
        empty!(balancer.migration_history)
        empty!(balancer.error_log)
    end
    
    @info "Load balancer cleaned up"
end

# Export main types and functions
export WorkStealingStrategy, MigrationSafety, LoadPredictionModel
export GREEDY_STEALING, ROUND_ROBIN_STEALING, ADAPTIVE_STEALING, PREDICTIVE_STEALING
export SAFE_MIGRATION, CHECKPOINT_MIGRATION, IMMEDIATE_MIGRATION
export LINEAR_PREDICTION, EXPONENTIAL_PREDICTION, NEURAL_PREDICTION, ENSEMBLE_PREDICTION
export GPUDevice, GPUMetrics, TreeWorkload, LoadBalancingConfig, GPULoadBalancer
export create_gpu_device, create_gpu_metrics, create_tree_workload, create_load_balancing_config
export initialize_gpu_load_balancer, start_monitoring!, stop_monitoring!
export assign_tree_to_gpu!, update_tree_workload!, calculate_optimal_batch_size
export get_load_balancing_status, generate_load_balancing_report, cleanup_load_balancer!

end # module GPULoadBalancing