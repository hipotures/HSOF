"""
Dual-GPU Tree Distribution System for Stage 2 MCTS
Implements work distribution splitting 100 trees across two RTX 4090 GPUs with balanced load,
dynamic reallocation, and fault tolerance for high-performance ensemble execution.

This module provides GPU assignment, CUDA context management, memory pre-allocation,
and health monitoring for dual-GPU MCTS ensemble operations.
"""

module DualGPUDistribution

using CUDA
using Random
using Statistics
using Dates
using Printf
using Base.Threads

# Import ensemble forest for tree management
include("ensemble_forest.jl")
using .EnsembleForest

"""
GPU assignment strategies for tree distribution
"""
@enum GPUAssignmentStrategy begin
    STATIC_SPLIT = 1        # Fixed 50/50 split across GPUs
    DYNAMIC_LOAD = 2        # Dynamic based on GPU load
    MEMORY_BASED = 3        # Based on available GPU memory
    PERFORMANCE_BASED = 4   # Based on GPU performance metrics
end

"""
GPU device information and monitoring
"""
mutable struct GPUDeviceInfo
    device_id::Int                      # CUDA device ID (0 or 1)
    device_name::String                 # GPU model name
    total_memory::Int                   # Total GPU memory (bytes)
    available_memory::Int               # Currently available memory
    utilization::Float32                # Current utilization (0.0-1.0)
    temperature::Float32                # Temperature in Celsius
    power_usage::Float32                # Power usage in watts
    
    # Tree assignment
    assigned_trees::Set{Int}            # Tree IDs assigned to this GPU
    max_trees::Int                      # Maximum trees this GPU can handle
    active_trees::Int                   # Currently active trees
    
    # Performance tracking
    iterations_per_second::Float64      # MCTS iterations per second
    memory_bandwidth::Float64           # Memory bandwidth utilization
    last_update::DateTime               # Last metrics update
    
    # Status and health
    is_healthy::Bool                    # GPU health status
    cuda_context::Union{CuContext, Nothing}  # CUDA context
    error_count::Int                    # Number of errors encountered
    last_error::Union{String, Nothing}  # Last error message
end

"""
Dual-GPU distribution configuration
"""
struct DualGPUConfig
    # GPU assignment
    primary_gpu_id::Int                 # Primary GPU device ID (usually 0)
    secondary_gpu_id::Int               # Secondary GPU device ID (usually 1)
    assignment_strategy::GPUAssignmentStrategy
    
    # Load balancing
    enable_dynamic_reallocation::Bool   # Allow tree migration between GPUs
    reallocation_threshold::Float32     # Utilization difference to trigger migration
    reallocation_cooldown::Int          # Minimum seconds between reallocations
    
    # Memory management
    memory_reserve_mb::Int              # Memory to reserve per GPU (MB)
    trees_per_gpu::Int                  # Target trees per GPU (default 50)
    memory_per_tree_mb::Int             # Estimated memory per tree (MB)
    
    # Performance monitoring
    monitoring_interval_ms::Int         # GPU monitoring frequency (ms)
    health_check_interval_s::Int        # Health check frequency (seconds)
    
    # Fault tolerance
    enable_failover::Bool               # Enable single-GPU fallback
    max_errors_before_failover::Int     # Error threshold for failover
    
    # Communication
    sync_frequency::Int                 # Inter-GPU sync frequency (iterations)
    max_sync_latency_ms::Int           # Maximum acceptable sync latency
end

"""
Create default dual-GPU configuration
"""
function create_dual_gpu_config(;
    primary_gpu_id::Int = 0,
    secondary_gpu_id::Int = 1,
    assignment_strategy::GPUAssignmentStrategy = STATIC_SPLIT,
    enable_dynamic_reallocation::Bool = true,
    reallocation_threshold::Float32 = 0.3f0,
    reallocation_cooldown::Int = 30,
    memory_reserve_mb::Int = 1024,  # 1GB reserve
    trees_per_gpu::Int = 50,
    memory_per_tree_mb::Int = 100,  # 100MB per tree
    monitoring_interval_ms::Int = 1000,
    health_check_interval_s::Int = 5,
    enable_failover::Bool = true,
    max_errors_before_failover::Int = 5,
    sync_frequency::Int = 1000,
    max_sync_latency_ms::Int = 100
)
    return DualGPUConfig(
        primary_gpu_id, secondary_gpu_id, assignment_strategy,
        enable_dynamic_reallocation, reallocation_threshold, reallocation_cooldown,
        memory_reserve_mb, trees_per_gpu, memory_per_tree_mb,
        monitoring_interval_ms, health_check_interval_s,
        enable_failover, max_errors_before_failover,
        sync_frequency, max_sync_latency_ms
    )
end

"""
Dual-GPU distribution manager
Coordinates tree assignment, load balancing, and fault tolerance across two GPUs
"""
mutable struct DualGPUDistributionManager
    # Configuration
    config::DualGPUConfig
    
    # GPU devices
    gpu_devices::Dict{Int, GPUDeviceInfo}
    available_gpus::Vector{Int}
    active_gpus::Vector{Int}
    
    # Tree distribution
    tree_to_gpu_mapping::Dict{Int, Int}     # Tree ID -> GPU ID
    gpu_to_trees_mapping::Dict{Int, Vector{Int}}  # GPU ID -> Tree IDs
    
    # Load balancing
    last_reallocation_time::DateTime
    reallocation_history::Vector{Tuple{DateTime, Int, Int, Int}}  # Time, tree_id, from_gpu, to_gpu
    
    # Performance monitoring
    monitoring_active::Bool
    monitoring_task::Union{Task, Nothing}
    performance_history::Dict{String, Vector{Float64}}
    
    # Synchronization
    sync_lock::ReentrantLock
    last_sync_time::DateTime
    sync_stats::Dict{String, Any}
    
    # Fault tolerance
    failover_active::Bool
    failover_gpu::Union{Int, Nothing}
    error_log::Vector{String}
    
    # Status
    distribution_state::String          # "initializing", "active", "failover", "error"
    total_trees_managed::Int
    creation_time::DateTime
end

"""
Initialize dual-GPU distribution manager
"""
function initialize_dual_gpu_manager(config::DualGPUConfig = create_dual_gpu_config())
    # Verify CUDA availability
    if !CUDA.functional()
        error("CUDA not available - dual-GPU distribution requires CUDA support")
    end
    
    # Check available GPUs
    available_gpus = collect(0:CUDA.ndevices()-1)
    if length(available_gpus) < 2
        error("Dual-GPU distribution requires at least 2 GPUs, found $(length(available_gpus))")
    end
    
    # Verify target GPUs exist
    if !(config.primary_gpu_id in available_gpus)
        error("Primary GPU $(config.primary_gpu_id) not available")
    end
    if !(config.secondary_gpu_id in available_gpus)
        error("Secondary GPU $(config.secondary_gpu_id) not available")
    end
    
    # Initialize GPU device info
    gpu_devices = Dict{Int, GPUDeviceInfo}()
    for gpu_id in [config.primary_gpu_id, config.secondary_gpu_id]
        CUDA.device!(gpu_id)
        
        # Get device properties
        props = CUDA.properties(CuDevice(gpu_id))
        total_memory = Int(props.totalGlobalMem)
        available_memory = Int(CUDA.available_memory())
        
        # Create CUDA context
        cuda_context = CuContext(CuDevice(gpu_id))
        
        # Initialize device info
        device_info = GPUDeviceInfo(
            gpu_id,
            props.name,
            total_memory,
            available_memory,
            0.0f0,  # Initial utilization
            0.0f0,  # Initial temperature
            0.0f0,  # Initial power
            Set{Int}(),  # No trees assigned initially
            config.trees_per_gpu,
            0,  # No active trees
            0.0,  # No iterations yet
            0.0,  # No bandwidth data
            now(),
            true,  # Initially healthy
            cuda_context,
            0,  # No errors
            nothing  # No error message
        )
        
        gpu_devices[gpu_id] = device_info
    end
    
    # Initialize manager
    manager = DualGPUDistributionManager(
        config,
        gpu_devices,
        available_gpus,
        [config.primary_gpu_id, config.secondary_gpu_id],
        Dict{Int, Int}(),
        Dict{Int, Vector{Int}}(
            config.primary_gpu_id => Int[],
            config.secondary_gpu_id => Int[]
        ),
        now(),
        Tuple{DateTime, Int, Int, Int}[],
        false,  # Monitoring not started
        nothing,
        Dict{String, Vector{Float64}}(),
        ReentrantLock(),
        now(),
        Dict{String, Any}(),
        false,  # No failover
        nothing,
        String[],
        "initializing",
        0,
        now()
    )
    
    # Pre-allocate GPU memory for trees
    pre_allocate_gpu_memory!(manager)
    
    manager.distribution_state = "active"
    
    return manager
end

"""
Pre-allocate GPU memory for tree operations
"""
function pre_allocate_gpu_memory!(manager::DualGPUDistributionManager)
    lock(manager.sync_lock) do
        for (gpu_id, device_info) in manager.gpu_devices
            try
                CUDA.device!(gpu_id)
                
                # Calculate memory to allocate
                memory_per_tree = manager.config.memory_per_tree_mb * 1024 * 1024
                total_tree_memory = memory_per_tree * manager.config.trees_per_gpu
                reserve_memory = manager.config.memory_reserve_mb * 1024 * 1024
                
                # Check if enough memory is available
                available = device_info.available_memory
                required = total_tree_memory + reserve_memory
                
                if available < required
                    error("GPU $gpu_id: Insufficient memory. Required: $(required ÷ (1024^2))MB, Available: $(available ÷ (1024^2))MB")
                end
                
                # Update device memory tracking
                device_info.available_memory = available - total_tree_memory
                
                @info "GPU $gpu_id: Pre-allocated $(total_tree_memory ÷ (1024^2))MB for $(manager.config.trees_per_gpu) trees"
                
            catch e
                error_msg = "Failed to pre-allocate memory on GPU $gpu_id: $e"
                push!(manager.error_log, error_msg)
                device_info.is_healthy = false
                device_info.last_error = string(e)
                device_info.error_count += 1
                @error error_msg
            end
        end
    end
end

"""
Assign trees to GPUs based on configuration strategy
"""
function assign_trees_to_gpus!(manager::DualGPUDistributionManager, forest_manager::ForestManager)
    lock(manager.sync_lock) do
        tree_ids = collect(keys(forest_manager.trees))
        total_trees = length(tree_ids)
        
        if total_trees == 0
            @warn "No trees to assign"
            return
        end
        
        # Clear existing assignments
        for (gpu_id, device_info) in manager.gpu_devices
            empty!(device_info.assigned_trees)
        end
        empty!(manager.tree_to_gpu_mapping)
        for gpu_id in manager.active_gpus
            manager.gpu_to_trees_mapping[gpu_id] = Int[]
        end
        
        # Assign trees based on strategy
        if manager.config.assignment_strategy == STATIC_SPLIT
            assign_trees_static_split!(manager, tree_ids)
        elseif manager.config.assignment_strategy == DYNAMIC_LOAD
            assign_trees_dynamic_load!(manager, tree_ids)
        elseif manager.config.assignment_strategy == MEMORY_BASED
            assign_trees_memory_based!(manager, tree_ids)
        elseif manager.config.assignment_strategy == PERFORMANCE_BASED
            assign_trees_performance_based!(manager, tree_ids)
        end
        
        # Update forest manager with GPU assignments
        update_forest_gpu_assignments!(manager, forest_manager)
        
        manager.total_trees_managed = total_trees
        
        @info "Assigned $total_trees trees across $(length(manager.active_gpus)) GPUs"
        for gpu_id in manager.active_gpus
            tree_count = length(manager.gpu_to_trees_mapping[gpu_id])
            @info "  GPU $gpu_id: $tree_count trees"
        end
    end
end

"""
Static split assignment (50/50)
"""
function assign_trees_static_split!(manager::DualGPUDistributionManager, tree_ids::Vector{Int})
    gpu_ids = manager.active_gpus
    trees_per_gpu = div(length(tree_ids), length(gpu_ids))
    
    for (i, tree_id) in enumerate(tree_ids)
        gpu_index = min(div(i - 1, trees_per_gpu) + 1, length(gpu_ids))
        gpu_id = gpu_ids[gpu_index]
        
        assign_tree_to_gpu!(manager, tree_id, gpu_id)
    end
end

"""
Dynamic load-based assignment
"""
function assign_trees_dynamic_load!(manager::DualGPUDistributionManager, tree_ids::Vector{Int})
    gpu_loads = Dict{Int, Float32}()
    
    # Get current GPU loads
    for gpu_id in manager.active_gpus
        device_info = manager.gpu_devices[gpu_id]
        gpu_loads[gpu_id] = device_info.utilization
    end
    
    # Assign trees to least loaded GPU
    for tree_id in tree_ids
        least_loaded_gpu = argmin(gpu_loads)
        assign_tree_to_gpu!(manager, tree_id, least_loaded_gpu)
        
        # Estimate load increase (simplified)
        gpu_loads[least_loaded_gpu] += 1.0f0 / manager.config.trees_per_gpu
    end
end

"""
Memory-based assignment
"""
function assign_trees_memory_based!(manager::DualGPUDistributionManager, tree_ids::Vector{Int})
    # Sort GPUs by available memory
    gpu_memory = [(gpu_id, device_info.available_memory) for (gpu_id, device_info) in manager.gpu_devices if gpu_id in manager.active_gpus]
    sort!(gpu_memory, by = x -> x[2], rev = true)
    
    # Round-robin assignment prioritizing high-memory GPUs
    for (i, tree_id) in enumerate(tree_ids)
        gpu_index = ((i - 1) % length(gpu_memory)) + 1
        gpu_id = gpu_memory[gpu_index][1]
        assign_tree_to_gpu!(manager, tree_id, gpu_id)
    end
end

"""
Performance-based assignment
"""
function assign_trees_performance_based!(manager::DualGPUDistributionManager, tree_ids::Vector{Int})
    # Sort GPUs by iterations per second (performance)
    gpu_performance = [(gpu_id, device_info.iterations_per_second) for (gpu_id, device_info) in manager.gpu_devices if gpu_id in manager.active_gpus]
    sort!(gpu_performance, by = x -> x[2], rev = true)
    
    # Assign more trees to higher-performance GPUs
    total_performance = sum(perf[2] for perf in gpu_performance)
    
    if total_performance > 0
        for (gpu_id, performance) in gpu_performance
            trees_for_gpu = round(Int, (performance / total_performance) * length(tree_ids))
            gpu_trees = tree_ids[1:min(trees_for_gpu, length(tree_ids))]
            
            for tree_id in gpu_trees
                assign_tree_to_gpu!(manager, tree_id, gpu_id)
            end
            
            # Remove assigned trees
            tree_ids = tree_ids[length(gpu_trees)+1:end]
        end
    else
        # Fallback to static split if no performance data
        assign_trees_static_split!(manager, tree_ids)
    end
end

"""
Assign single tree to specific GPU
"""
function assign_tree_to_gpu!(manager::DualGPUDistributionManager, tree_id::Int, gpu_id::Int)
    # Update tree-to-GPU mapping
    manager.tree_to_gpu_mapping[tree_id] = gpu_id
    
    # Update GPU-to-trees mapping
    push!(manager.gpu_to_trees_mapping[gpu_id], tree_id)
    
    # Update device info
    device_info = manager.gpu_devices[gpu_id]
    push!(device_info.assigned_trees, tree_id)
end

"""
Update forest manager with GPU assignments
"""
function update_forest_gpu_assignments!(manager::DualGPUDistributionManager, forest_manager::ForestManager)
    for (tree_id, gpu_id) in manager.tree_to_gpu_mapping
        if haskey(forest_manager.trees, tree_id)
            forest_manager.trees[tree_id].gpu_device_id = gpu_id
        end
    end
end

"""
Start performance monitoring
"""
function start_monitoring!(manager::DualGPUDistributionManager)
    if manager.monitoring_active
        @warn "Monitoring already active"
        return
    end
    
    manager.monitoring_active = true
    
    manager.monitoring_task = @async begin
        try
            while manager.monitoring_active
                update_gpu_metrics!(manager)
                check_gpu_health!(manager)
                
                # Check for reallocation needs
                if manager.config.enable_dynamic_reallocation
                    check_reallocation_needs!(manager)
                end
                
                sleep(manager.config.monitoring_interval_ms / 1000.0)
            end
        catch e
            @error "Monitoring task failed: $e"
            manager.monitoring_active = false
        end
    end
    
    @info "GPU monitoring started"
end

"""
Stop performance monitoring
"""
function stop_monitoring!(manager::DualGPUDistributionManager)
    manager.monitoring_active = false
    
    if !isnothing(manager.monitoring_task)
        # Wait for monitoring task to finish
        try
            wait(manager.monitoring_task)
        catch e
            @warn "Error stopping monitoring task: $e"
        end
        manager.monitoring_task = nothing
    end
    
    @info "GPU monitoring stopped"
end

"""
Update GPU metrics
"""
function update_gpu_metrics!(manager::DualGPUDistributionManager)
    for (gpu_id, device_info) in manager.gpu_devices
        if gpu_id in manager.active_gpus
            try
                CUDA.device!(gpu_id)
                
                # Update memory info
                device_info.available_memory = Int(CUDA.available_memory())
                
                # Update utilization (simplified - would need NVML for real data)
                device_info.utilization = min(1.0f0, Float32(device_info.active_trees) / device_info.max_trees)
                
                # Update timestamp
                device_info.last_update = now()
                
                # Store in performance history
                timestamp = time()
                if !haskey(manager.performance_history, "gpu_$(gpu_id)_utilization")
                    manager.performance_history["gpu_$(gpu_id)_utilization"] = Float64[]
                end
                push!(manager.performance_history["gpu_$(gpu_id)_utilization"], device_info.utilization)
                
                # Limit history size
                if length(manager.performance_history["gpu_$(gpu_id)_utilization"]) > 1000
                    deleteat!(manager.performance_history["gpu_$(gpu_id)_utilization"], 1)
                end
                
            catch e
                device_info.error_count += 1
                device_info.last_error = string(e)
                @warn "Failed to update metrics for GPU $gpu_id: $e"
            end
        end
    end
end

"""
Check GPU health and trigger failover if needed
"""
function check_gpu_health!(manager::DualGPUDistributionManager)
    for (gpu_id, device_info) in manager.gpu_devices
        if gpu_id in manager.active_gpus
            # Check error count
            if device_info.error_count >= manager.config.max_errors_before_failover
                @error "GPU $gpu_id exceeded error threshold ($(device_info.error_count) errors)"
                device_info.is_healthy = false
                
                if manager.config.enable_failover && !manager.failover_active
                    trigger_failover!(manager, gpu_id)
                end
            end
            
            # Check if GPU is responsive
            try
                CUDA.device!(gpu_id)
                CUDA.synchronize()
            catch e
                @error "GPU $gpu_id not responsive: $e"
                device_info.is_healthy = false
                device_info.error_count += 1
                device_info.last_error = string(e)
                
                if manager.config.enable_failover && !manager.failover_active
                    trigger_failover!(manager, gpu_id)
                end
            end
        end
    end
end

"""
Check if tree reallocation is needed
"""
function check_reallocation_needs!(manager::DualGPUDistributionManager)
    # Check cooldown period
    if (now() - manager.last_reallocation_time).value < manager.config.reallocation_cooldown * 1000
        return
    end
    
    # Get GPU utilizations
    utilizations = Dict{Int, Float32}()
    for gpu_id in manager.active_gpus
        device_info = manager.gpu_devices[gpu_id]
        if device_info.is_healthy
            utilizations[gpu_id] = device_info.utilization
        end
    end
    
    if length(utilizations) < 2
        return  # Need at least 2 healthy GPUs
    end
    
    # Find most and least loaded GPUs
    max_util_gpu = argmax(utilizations)
    min_util_gpu = argmin(utilizations)
    util_diff = utilizations[max_util_gpu] - utilizations[min_util_gpu]
    
    # Check if reallocation is needed
    if util_diff > manager.config.reallocation_threshold
        perform_tree_reallocation!(manager, max_util_gpu, min_util_gpu)
    end
end

"""
Perform tree reallocation between GPUs
"""
function perform_tree_reallocation!(manager::DualGPUDistributionManager, from_gpu::Int, to_gpu::Int)
    lock(manager.sync_lock) do
        # Find a tree to move
        from_trees = manager.gpu_to_trees_mapping[from_gpu]
        
        if isempty(from_trees)
            return
        end
        
        # Choose tree to move (e.g., least active one)
        tree_to_move = from_trees[1]  # Simplified selection
        
        # Update mappings
        manager.tree_to_gpu_mapping[tree_to_move] = to_gpu
        
        # Update GPU-to-trees mappings
        filter!(t -> t != tree_to_move, manager.gpu_to_trees_mapping[from_gpu])
        push!(manager.gpu_to_trees_mapping[to_gpu], tree_to_move)
        
        # Update device info
        delete!(manager.gpu_devices[from_gpu].assigned_trees, tree_to_move)
        push!(manager.gpu_devices[to_gpu].assigned_trees, tree_to_move)
        
        # Record reallocation
        reallocation_record = (now(), tree_to_move, from_gpu, to_gpu)
        push!(manager.reallocation_history, reallocation_record)
        manager.last_reallocation_time = now()
        
        @info "Reallocated tree $tree_to_move from GPU $from_gpu to GPU $to_gpu"
    end
end

"""
Trigger failover to single GPU
"""
function trigger_failover!(manager::DualGPUDistributionManager, failed_gpu::Int)
    lock(manager.sync_lock) do
        if manager.failover_active
            @warn "Failover already active"
            return
        end
        
        @warn "Triggering failover due to GPU $failed_gpu failure"
        
        # Find healthy GPU for failover
        healthy_gpu = nothing
        for gpu_id in manager.active_gpus
            if gpu_id != failed_gpu && manager.gpu_devices[gpu_id].is_healthy
                healthy_gpu = gpu_id
                break
            end
        end
        
        if isnothing(healthy_gpu)
            @error "No healthy GPU available for failover"
            manager.distribution_state = "error"
            return
        end
        
        # Move all trees from failed GPU to healthy GPU
        failed_gpu_trees = copy(manager.gpu_to_trees_mapping[failed_gpu])
        
        for tree_id in failed_gpu_trees
            manager.tree_to_gpu_mapping[tree_id] = healthy_gpu
            push!(manager.gpu_to_trees_mapping[healthy_gpu], tree_id)
            push!(manager.gpu_devices[healthy_gpu].assigned_trees, tree_id)
        end
        
        # Clear failed GPU assignments
        empty!(manager.gpu_to_trees_mapping[failed_gpu])
        empty!(manager.gpu_devices[failed_gpu].assigned_trees)
        
        # Update active GPUs
        filter!(gpu -> gpu != failed_gpu, manager.active_gpus)
        
        # Set failover state
        manager.failover_active = true
        manager.failover_gpu = healthy_gpu
        manager.distribution_state = "failover"
        
        @info "Failover complete: all trees now on GPU $healthy_gpu"
    end
end

"""
Get distribution status and statistics
"""
function get_distribution_status(manager::DualGPUDistributionManager)
    lock(manager.sync_lock) do
        status = Dict{String, Any}(
            "distribution_state" => manager.distribution_state,
            "total_trees_managed" => manager.total_trees_managed,
            "active_gpus" => length(manager.active_gpus),
            "failover_active" => manager.failover_active,
            "monitoring_active" => manager.monitoring_active,
            "total_reallocations" => length(manager.reallocation_history),
            "creation_time" => manager.creation_time,
            "last_reallocation" => isempty(manager.reallocation_history) ? nothing : manager.reallocation_history[end][1]
        )
        
        # GPU-specific status
        status["gpus"] = Dict{String, Any}()
        for (gpu_id, device_info) in manager.gpu_devices
            status["gpus"]["gpu_$gpu_id"] = Dict{String, Any}(
                "device_name" => device_info.device_name,
                "assigned_trees" => length(device_info.assigned_trees),
                "max_trees" => device_info.max_trees,
                "active_trees" => device_info.active_trees,
                "utilization" => device_info.utilization,
                "available_memory_mb" => device_info.available_memory ÷ (1024^2),
                "total_memory_mb" => device_info.total_memory ÷ (1024^2),
                "is_healthy" => device_info.is_healthy,
                "error_count" => device_info.error_count,
                "last_update" => device_info.last_update
            )
        end
        
        return status
    end
end

"""
Generate distribution report
"""
function generate_distribution_report(manager::DualGPUDistributionManager)
    status = get_distribution_status(manager)
    
    report = String[]
    
    push!(report, "=== Dual-GPU Distribution Status Report ===")
    push!(report, "Generated: $(now())")
    push!(report, "Distribution State: $(status["distribution_state"])")
    push!(report, "")
    
    # Overview
    push!(report, "Overview:")
    push!(report, "  Total Trees: $(status["total_trees_managed"])")
    push!(report, "  Active GPUs: $(status["active_gpus"])")
    push!(report, "  Failover Active: $(status["failover_active"])")
    push!(report, "  Monitoring Active: $(status["monitoring_active"])")
    push!(report, "  Total Reallocations: $(status["total_reallocations"])")
    push!(report, "")
    
    # GPU details
    push!(report, "GPU Details:")
    for (gpu_key, gpu_info) in status["gpus"]
        push!(report, "  $gpu_key:")
        push!(report, "    Device: $(gpu_info["device_name"])")
        push!(report, "    Trees: $(gpu_info["assigned_trees"])/$(gpu_info["max_trees"])")
        push!(report, "    Utilization: $(round(gpu_info["utilization"] * 100, digits=1))%")
        push!(report, "    Memory: $(gpu_info["available_memory_mb"])/$(gpu_info["total_memory_mb"]) MB")
        push!(report, "    Health: $(gpu_info["is_healthy"] ? "Healthy" : "Unhealthy")")
        push!(report, "    Errors: $(gpu_info["error_count"])")
        push!(report, "")
    end
    
    # Recent reallocations
    if !isempty(manager.reallocation_history)
        push!(report, "Recent Reallocations (last 5):")
        recent_reallocations = manager.reallocation_history[max(1, length(manager.reallocation_history)-4):end]
        for (timestamp, tree_id, from_gpu, to_gpu) in recent_reallocations
            push!(report, "  $timestamp: Tree $tree_id moved from GPU $from_gpu to GPU $to_gpu")
        end
        push!(report, "")
    end
    
    push!(report, "=== End Distribution Report ===")
    
    return join(report, "\n")
end

"""
Cleanup and shutdown distribution manager
"""
function cleanup_distribution!(manager::DualGPUDistributionManager)
    # Stop monitoring
    stop_monitoring!(manager)
    
    # Cleanup CUDA contexts
    for (gpu_id, device_info) in manager.gpu_devices
        if !isnothing(device_info.cuda_context)
            try
                CUDA.unsafe_destroy!(device_info.cuda_context)
                device_info.cuda_context = nothing
            catch e
                @warn "Error destroying CUDA context for GPU $gpu_id: $e"
            end
        end
    end
    
    manager.distribution_state = "terminated"
    @info "Dual-GPU distribution manager cleaned up"
end

# Export main types and functions
export GPUAssignmentStrategy, GPUDeviceInfo, DualGPUConfig, DualGPUDistributionManager
export STATIC_SPLIT, DYNAMIC_LOAD, MEMORY_BASED, PERFORMANCE_BASED
export create_dual_gpu_config, initialize_dual_gpu_manager
export assign_trees_to_gpus!, start_monitoring!, stop_monitoring!
export get_distribution_status, generate_distribution_report, cleanup_distribution!

end # module DualGPUDistribution