module DynamicRebalancing

using CUDA
using Dates
using Printf
using Statistics
using Base.Threads: Atomic, @spawn, ReentrantLock

export RebalancingManager, RebalancingDecision, MigrationPlan, RebalancingMetrics
export create_rebalancing_manager, start_rebalancing!, stop_rebalancing!
export check_imbalance, should_rebalance, create_migration_plan
export execute_migration!, update_workload_metrics!
export get_rebalancing_history, get_current_distribution
export set_rebalancing_threshold!, enable_auto_rebalancing!

# Rebalancing state
@enum RebalancingState begin
    BALANCED = 0
    MONITORING = 1
    PLANNING = 2
    MIGRATING = 3
    STABILIZING = 4
end

"""
Workload metrics for a single GPU
"""
mutable struct GPUWorkload
    gpu_id::Int
    tree_ids::Vector{Int}
    total_trees::Int
    
    # Performance metrics
    avg_utilization::Float32
    current_utilization::Float32
    memory_usage::Float32
    throughput::Float32  # trees/second
    
    # Timing metrics
    avg_tree_time::Float64  # milliseconds
    total_compute_time::Float64
    idle_time::Float64
    
    # Historical data for stability
    utilization_history::Vector{Float32}
    max_history::Int
    
    function GPUWorkload(gpu_id::Int, tree_ids::Vector{Int} = Int[])
        new(
            gpu_id,
            tree_ids,
            length(tree_ids),
            0.0f0, 0.0f0, 0.0f0, 0.0f0,
            0.0, 0.0, 0.0,
            Float32[],
            20  # Keep last 20 measurements
        )
    end
end

"""
Migration plan for moving trees between GPUs
"""
struct MigrationPlan
    source_gpu::Int
    target_gpu::Int
    tree_ids::Vector{Int}
    estimated_cost::Float64  # milliseconds
    expected_improvement::Float32  # percentage
    reason::String
end

"""
Rebalancing decision with justification
"""
struct RebalancingDecision
    timestamp::DateTime
    should_rebalance::Bool
    imbalance_ratio::Float32
    migration_plans::Vector{MigrationPlan}
    total_trees_to_migrate::Int
    estimated_total_cost::Float64
    decision_reason::String
end

"""
Historical rebalancing metrics
"""
struct RebalancingMetrics
    timestamp::DateTime
    state::RebalancingState
    imbalance_before::Float32
    imbalance_after::Float32
    trees_migrated::Int
    migration_time::Float64
    success::Bool
end

"""
Cost model for rebalancing decisions
"""
mutable struct CostModel
    # Migration costs (milliseconds)
    per_tree_migration_cost::Float64
    state_transfer_cost::Float64
    synchronization_cost::Float64
    
    # Thresholds
    min_migration_benefit::Float32  # Minimum expected improvement
    max_migration_ratio::Float32   # Max % of trees to migrate at once
    
    # Factors
    memory_pressure_factor::Float32
    throughput_weight::Float32
    
    function CostModel()
        new(
            10.0,   # 10ms per tree migration
            50.0,   # 50ms state transfer
            20.0,   # 20ms synchronization
            5.0f0,  # 5% minimum improvement
            0.3f0,  # Max 30% trees at once
            1.5f0,  # Memory pressure multiplier
            0.7f0   # Throughput importance
        )
    end
end

"""
Main rebalancing manager
"""
mutable struct RebalancingManager
    # Workload tracking
    gpu_workloads::Dict{Int, GPUWorkload}
    num_gpus::Int
    total_trees::Int
    
    # Rebalancing configuration
    imbalance_threshold::Float32  # Default 10%
    check_interval::Float64       # seconds
    auto_rebalancing::Bool
    
    # Hysteresis mechanism
    hysteresis_factor::Float32    # 0.8 = 80% of threshold to re-trigger
    cooldown_period::Float64      # seconds between rebalancing
    last_rebalancing::DateTime
    consecutive_triggers::Int
    trigger_threshold::Int        # Number of consecutive triggers needed
    
    # State management
    current_state::RebalancingState
    state_lock::ReentrantLock
    
    # Cost model
    cost_model::CostModel
    
    # Monitoring
    monitoring_active::Atomic{Bool}
    monitor_task::Union{Nothing, Task}
    
    # History
    rebalancing_history::Vector{RebalancingMetrics}
    max_history::Int
    
    # Callbacks
    pre_migration_callbacks::Vector{Function}
    post_migration_callbacks::Vector{Function}
    
    # Statistics
    total_rebalancings::Atomic{Int}
    total_migrations::Atomic{Int}
    total_migration_time::Atomic{Float64}
    
    function RebalancingManager(;
        num_gpus::Int = 2,
        total_trees::Int = 100,
        imbalance_threshold::Float32 = 0.1f0,  # 10%
        check_interval::Float64 = 5.0,
        auto_rebalancing::Bool = true,
        hysteresis_factor::Float32 = 0.8f0,
        cooldown_period::Float64 = 30.0,
        trigger_threshold::Int = 3
    )
        # Initialize GPU workloads
        workloads = Dict{Int, GPUWorkload}()
        
        # Default distribution (even split)
        trees_per_gpu = total_trees ÷ num_gpus
        remaining = total_trees % num_gpus
        
        tree_id = 1
        for gpu_id in 0:(num_gpus-1)
            num_trees = trees_per_gpu + (gpu_id < remaining ? 1 : 0)
            tree_ids = collect(tree_id:(tree_id + num_trees - 1))
            workloads[gpu_id] = GPUWorkload(gpu_id, tree_ids)
            tree_id += num_trees
        end
        
        new(
            workloads,
            num_gpus,
            total_trees,
            imbalance_threshold,
            check_interval,
            auto_rebalancing,
            hysteresis_factor,
            cooldown_period,
            now() - Second(Int(cooldown_period)),
            0,
            trigger_threshold,
            BALANCED,
            ReentrantLock(),
            CostModel(),
            Atomic{Bool}(false),
            nothing,
            RebalancingMetrics[],
            100,
            Function[],
            Function[],
            Atomic{Int}(0),
            Atomic{Int}(0),
            Atomic{Float64}(0.0)
        )
    end
end

"""
Create and initialize rebalancing manager
"""
function create_rebalancing_manager(;kwargs...)
    return RebalancingManager(;kwargs...)
end

"""
Start automatic rebalancing monitoring
"""
function start_rebalancing!(manager::RebalancingManager)
    if manager.monitoring_active[]
        @warn "Rebalancing monitor already active"
        return
    end
    
    manager.monitoring_active[] = true
    
    manager.monitor_task = @spawn begin
        monitor_workload_balance(manager)
    end
    
    @info "Dynamic rebalancing started"
end

"""
Stop rebalancing monitoring
"""
function stop_rebalancing!(manager::RebalancingManager)
    manager.monitoring_active[] = false
    
    if !isnothing(manager.monitor_task)
        wait(manager.monitor_task)
        manager.monitor_task = nothing
    end
    
    @info "Dynamic rebalancing stopped"
end

"""
Monitor workload balance continuously
"""
function monitor_workload_balance(manager::RebalancingManager)
    while manager.monitoring_active[]
        try
            # Check imbalance
            imbalance = check_imbalance(manager)
            
            # Update state
            lock(manager.state_lock) do
                if manager.current_state == BALANCED && imbalance > manager.imbalance_threshold
                    manager.current_state = MONITORING
                    manager.consecutive_triggers = 1
                elseif manager.current_state == MONITORING
                    if imbalance > manager.imbalance_threshold
                        manager.consecutive_triggers += 1
                        
                        # Check if we should trigger rebalancing
                        if manager.consecutive_triggers >= manager.trigger_threshold
                            if manager.auto_rebalancing && should_rebalance(manager)
                                manager.current_state = PLANNING
                                
                                # Create and execute migration plan
                                decision = create_rebalancing_decision(manager)
                                if decision.should_rebalance
                                    execute_rebalancing!(manager, decision)
                                end
                            end
                        end
                    else
                        # Imbalance resolved
                        manager.current_state = BALANCED
                        manager.consecutive_triggers = 0
                    end
                end
            end
            
            sleep(manager.check_interval)
            
        catch e
            @error "Error in rebalancing monitor" exception=e
        end
    end
end

"""
Check current workload imbalance
"""
function check_imbalance(manager::RebalancingManager)::Float32
    if manager.num_gpus < 2
        return 0.0f0
    end
    
    # Calculate utilization-weighted workload for each GPU
    workloads = Float32[]
    
    for gpu_id in 0:(manager.num_gpus-1)
        if haskey(manager.gpu_workloads, gpu_id)
            workload = manager.gpu_workloads[gpu_id]
            
            # Weighted score combining utilization and tree count
            util_weight = 0.7f0
            tree_weight = 0.3f0
            
            weighted_load = util_weight * workload.avg_utilization + 
                          tree_weight * (workload.total_trees / manager.total_trees * 100)
            
            push!(workloads, weighted_load)
        end
    end
    
    if isempty(workloads)
        return 0.0f0
    end
    
    # Calculate imbalance as (max - min) / avg
    max_load = maximum(workloads)
    min_load = minimum(workloads)
    avg_load = mean(workloads)
    
    if avg_load > 0
        imbalance = (max_load - min_load) / avg_load
    else
        imbalance = 0.0f0
    end
    
    return imbalance
end

"""
Determine if rebalancing should occur
"""
function should_rebalance(manager::RebalancingManager)::Bool
    # Check cooldown period
    time_since_last = Dates.value(now() - manager.last_rebalancing) / 1000  # seconds
    if time_since_last < manager.cooldown_period
        return false
    end
    
    # Check if we're already rebalancing
    lock(manager.state_lock) do
        if manager.current_state == MIGRATING || manager.current_state == STABILIZING
            return false
        end
    end
    
    # Check memory pressure on all GPUs
    for (gpu_id, workload) in manager.gpu_workloads
        if workload.memory_usage > 0.9f0  # 90% memory usage
            @warn "High memory pressure on GPU $gpu_id, deferring rebalancing"
            return false
        end
    end
    
    return true
end

"""
Create rebalancing decision with migration plans
"""
function create_rebalancing_decision(manager::RebalancingManager)::RebalancingDecision
    imbalance = check_imbalance(manager)
    
    if imbalance <= manager.imbalance_threshold * manager.hysteresis_factor
        return RebalancingDecision(
            now(), false, imbalance, MigrationPlan[], 0, 0.0,
            "Imbalance below hysteresis threshold"
        )
    end
    
    # Find overloaded and underloaded GPUs
    gpu_loads = []
    for gpu_id in 0:(manager.num_gpus-1)
        workload = manager.gpu_workloads[gpu_id]
        load_score = workload.avg_utilization * 0.7f0 + 
                    (workload.total_trees / manager.total_trees * 100) * 0.3f0
        push!(gpu_loads, (gpu_id, load_score, workload))
    end
    
    sort!(gpu_loads, by=x->x[2], rev=true)
    
    # Create migration plans
    migration_plans = MigrationPlan[]
    total_cost = 0.0
    
    # Migrate from most loaded to least loaded
    source_idx = 1
    target_idx = length(gpu_loads)
    
    while source_idx < target_idx
        source_gpu, source_load, source_workload = gpu_loads[source_idx]
        target_gpu, target_load, target_workload = gpu_loads[target_idx]
        
        load_diff = Float32(source_load - target_load)
        if load_diff < manager.imbalance_threshold * 50  # Not worth migrating
            break
        end
        
        # Calculate how many trees to migrate
        trees_to_migrate = calculate_trees_to_migrate(
            manager, source_workload, target_workload, load_diff
        )
        
        if !isempty(trees_to_migrate)
            # Create migration plan
            cost = estimate_migration_cost(manager, length(trees_to_migrate))
            improvement = estimate_improvement(manager, Float32(source_load), Float32(target_load), length(trees_to_migrate))
            
            plan = MigrationPlan(
                source_gpu, target_gpu, trees_to_migrate,
                cost, improvement,
                "Load balancing: $(round(source_load, digits=1))% -> $(round(target_load, digits=1))%"
            )
            
            push!(migration_plans, plan)
            total_cost += cost
            
            # Update projected loads
            trees_moved_ratio = length(trees_to_migrate) / source_workload.total_trees
            gpu_loads[source_idx] = (source_gpu, source_load * (1 - trees_moved_ratio * 0.8), source_workload)
            gpu_loads[target_idx] = (target_gpu, target_load + source_load * trees_moved_ratio * 0.8, target_workload)
        end
        
        # Move to next pair if loads are balanced enough
        if abs(gpu_loads[source_idx][2] - gpu_loads[target_idx][2]) < manager.imbalance_threshold * 25
            source_idx += 1
            target_idx -= 1
        end
    end
    
    # Decide if the migration is worth it
    total_trees = sum(length(plan.tree_ids) for plan in migration_plans)
    avg_improvement = isempty(migration_plans) ? 0.0f0 : 
                     mean(plan.expected_improvement for plan in migration_plans)
    
    should_execute = !isempty(migration_plans) && 
                    avg_improvement >= manager.cost_model.min_migration_benefit &&
                    total_cost < 1000.0  # Max 1 second migration
    
    reason = if !should_execute
        if isempty(migration_plans)
            "No beneficial migrations found"
        elseif avg_improvement < manager.cost_model.min_migration_benefit
            "Expected improvement too low: $(round(avg_improvement, digits=1))%"
        else
            "Migration cost too high: $(round(total_cost, digits=1))ms"
        end
    else
        "Rebalancing $(total_trees) trees for $(round(avg_improvement, digits=1))% improvement"
    end
    
    return RebalancingDecision(
        now(), should_execute, imbalance,
        migration_plans, total_trees, total_cost, reason
    )
end

"""
Calculate trees to migrate for load balancing
"""
function calculate_trees_to_migrate(
    manager::RebalancingManager,
    source::GPUWorkload,
    target::GPUWorkload,
    load_diff::Float32
)::Vector{Int}
    # Don't migrate if source has too few trees
    if source.total_trees <= 5
        return Int[]
    end
    
    # Calculate target number of trees to migrate
    ideal_migration = Int(round(source.total_trees * load_diff / 200))  # Gradual approach
    max_migration = Int(floor(source.total_trees * manager.cost_model.max_migration_ratio))
    
    trees_to_migrate = min(ideal_migration, max_migration)
    trees_to_migrate = max(1, trees_to_migrate)  # At least 1 tree
    
    # Select trees to migrate (prefer recently added trees)
    if trees_to_migrate >= length(source.tree_ids)
        return Int[]  # Safety check
    end
    
    # Take trees from the end (most recently added)
    return source.tree_ids[end-trees_to_migrate+1:end]
end

"""
Estimate migration cost in milliseconds
"""
function estimate_migration_cost(manager::RebalancingManager, num_trees::Int)::Float64
    base_cost = manager.cost_model.per_tree_migration_cost * num_trees
    state_cost = manager.cost_model.state_transfer_cost
    sync_cost = manager.cost_model.synchronization_cost
    
    return base_cost + state_cost + sync_cost
end

"""
Estimate performance improvement percentage
"""
function estimate_improvement(
    manager::RebalancingManager,
    source_load::Float32,
    target_load::Float32,
    num_trees::Int
)::Float32
    source_workload = manager.gpu_workloads[0]  # Placeholder
    
    # Estimate load reduction on source
    trees_ratio = num_trees / manager.total_trees
    load_reduction = source_load * trees_ratio * 0.8f0  # 80% efficiency
    
    # Calculate balance improvement
    new_diff = abs((source_load - load_reduction) - (target_load + load_reduction))
    old_diff = abs(source_load - target_load)
    
    improvement = (old_diff - new_diff) / old_diff * 100
    
    return max(0.0f0, improvement)
end

"""
Execute rebalancing by migrating trees
"""
function execute_rebalancing!(manager::RebalancingManager, decision::RebalancingDecision)
    start_time = time()
    
    lock(manager.state_lock) do
        manager.current_state = MIGRATING
    end
    
    # Call pre-migration callbacks
    for callback in manager.pre_migration_callbacks
        try
            callback(decision)
        catch e
            @error "Pre-migration callback error" exception=e
        end
    end
    
    success = true
    trees_migrated = 0
    
    # Execute each migration plan
    for plan in decision.migration_plans
        try
            @info "Migrating $(length(plan.tree_ids)) trees from GPU $(plan.source_gpu) to GPU $(plan.target_gpu)"
            
            # Execute migration
            execute_migration!(manager, plan)
            trees_migrated += length(plan.tree_ids)
            
        catch e
            @error "Migration failed" plan exception=e
            success = false
            break
        end
    end
    
    # Update state
    lock(manager.state_lock) do
        manager.current_state = STABILIZING
        manager.last_rebalancing = now()
    end
    
    # Let system stabilize
    sleep(2.0)
    
    # Final state
    lock(manager.state_lock) do
        manager.current_state = BALANCED
        manager.consecutive_triggers = 0
    end
    
    # Record metrics
    migration_time = (time() - start_time) * 1000
    imbalance_after = check_imbalance(manager)
    
    metrics = RebalancingMetrics(
        decision.timestamp,
        BALANCED,
        decision.imbalance_ratio,
        imbalance_after,
        trees_migrated,
        migration_time,
        success
    )
    
    push!(manager.rebalancing_history, metrics)
    if length(manager.rebalancing_history) > manager.max_history
        popfirst!(manager.rebalancing_history)
    end
    
    # Update statistics
    if success
        manager.total_rebalancings[] += 1
        manager.total_migrations[] += trees_migrated
        manager.total_migration_time[] += migration_time
    end
    
    # Call post-migration callbacks
    for callback in manager.post_migration_callbacks
        try
            callback(metrics)
        catch e
            @error "Post-migration callback error" exception=e
        end
    end
    
    @info "Rebalancing completed" success trees_migrated time_ms=round(migration_time, digits=1) new_imbalance=round(imbalance_after, digits=3)
end

"""
Execute a single migration plan
"""
function execute_migration!(manager::RebalancingManager, plan::MigrationPlan)
    source_workload = manager.gpu_workloads[plan.source_gpu]
    target_workload = manager.gpu_workloads[plan.target_gpu]
    
    # Remove trees from source
    filter!(id -> !(id in plan.tree_ids), source_workload.tree_ids)
    source_workload.total_trees = length(source_workload.tree_ids)
    
    # Add trees to target
    append!(target_workload.tree_ids, plan.tree_ids)
    target_workload.total_trees = length(target_workload.tree_ids)
    
    # Sort tree IDs for consistency
    sort!(target_workload.tree_ids)
    
    # In a real implementation, this would trigger actual GPU work redistribution
    # For now, we just update the bookkeeping
    
    @info "Migrated trees" trees=plan.tree_ids from=plan.source_gpu to=plan.target_gpu
end

"""
Update workload metrics for a GPU
"""
function update_workload_metrics!(
    manager::RebalancingManager,
    gpu_id::Int;
    utilization::Float32,
    memory_usage::Float32,
    throughput::Float32,
    avg_tree_time::Float64
)
    if !haskey(manager.gpu_workloads, gpu_id)
        return
    end
    
    workload = manager.gpu_workloads[gpu_id]
    
    # Update current metrics
    workload.current_utilization = utilization
    workload.memory_usage = memory_usage
    workload.throughput = throughput
    workload.avg_tree_time = avg_tree_time
    
    # Update history
    push!(workload.utilization_history, utilization)
    if length(workload.utilization_history) > workload.max_history
        popfirst!(workload.utilization_history)
    end
    
    # Update average
    if !isempty(workload.utilization_history)
        workload.avg_utilization = mean(workload.utilization_history)
    end
end

"""
Get current tree distribution across GPUs
"""
function get_current_distribution(manager::RebalancingManager)
    distribution = Dict{Int, Vector{Int}}()
    
    for (gpu_id, workload) in manager.gpu_workloads
        distribution[gpu_id] = copy(workload.tree_ids)
    end
    
    return distribution
end

"""
Get rebalancing history
"""
function get_rebalancing_history(manager::RebalancingManager)
    return copy(manager.rebalancing_history)
end

"""
Set rebalancing threshold
"""
function set_rebalancing_threshold!(manager::RebalancingManager, threshold::Float32)
    if 0.0f0 < threshold < 1.0f0
        manager.imbalance_threshold = threshold
        @info "Rebalancing threshold set to $(threshold * 100)%"
    else
        @error "Invalid threshold: must be between 0 and 1"
    end
end

"""
Enable or disable automatic rebalancing
"""
function enable_auto_rebalancing!(manager::RebalancingManager, enabled::Bool)
    manager.auto_rebalancing = enabled
    @info "Automatic rebalancing $(enabled ? "enabled" : "disabled")"
end

"""
Create migration plan manually
"""
function create_migration_plan(
    source_gpu::Int,
    target_gpu::Int,
    tree_ids::Vector{Int},
    reason::String = "Manual migration"
)::MigrationPlan
    # Simple cost estimate
    cost = 10.0 * length(tree_ids) + 70.0  # Base costs
    improvement = 10.0f0  # Assumed improvement
    
    return MigrationPlan(
        source_gpu, target_gpu, tree_ids,
        cost, improvement, reason
    )
end

end # module