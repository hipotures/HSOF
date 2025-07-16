module WorkDistribution

using CUDA
using Dates
using Statistics

export WorkDistributor, GPUWorkAssignment, WorkloadMetrics
export create_work_distributor, assign_tree_work, assign_metamodel_work
export get_gpu_for_tree, get_tree_range, get_load_balance_ratio
export update_metrics!, rebalance_if_needed!, get_work_summary
export set_gpu_affinity, execute_on_gpu

"""
Work assignment for a specific GPU
"""
struct GPUWorkAssignment
    gpu_id::Int
    tree_range::UnitRange{Int}
    metamodel_role::Symbol  # :training, :inference, or :none
    is_primary::Bool
end

"""
Workload metrics for tracking GPU utilization
"""
mutable struct WorkloadMetrics
    gpu_id::Int
    trees_processed::Int
    metamodel_operations::Int
    total_time_ms::Float64
    utilization_percent::Float64
    memory_used_mb::Float64
    last_update::DateTime
    
    function WorkloadMetrics(gpu_id::Int)
        new(gpu_id, 0, 0, 0.0, 0.0, 0.0, now())
    end
end

"""
Main work distribution manager for dual-GPU setup
"""
mutable struct WorkDistributor
    gpu_assignments::Dict{Int, GPUWorkAssignment}
    workload_metrics::Dict{Int, WorkloadMetrics}
    total_trees::Int
    rebalance_threshold::Float64  # Trigger rebalancing if imbalance > threshold
    num_gpus::Int
    
    function WorkDistributor(;
        total_trees::Int = 100,
        rebalance_threshold::Float64 = 0.05,  # 5% imbalance threshold
        num_gpus::Int = -1  # -1 means auto-detect
    )
        # Initialize GPU assignments
        assignments = Dict{Int, GPUWorkAssignment}()
        metrics = Dict{Int, WorkloadMetrics}()
        
        # Detect available GPUs
        if num_gpus == -1
            num_gpus = CUDA.functional() ? length(CUDA.devices()) : 0
        end
        
        if num_gpus == 0
            error("No GPUs available for work distribution")
        elseif num_gpus == 1
            # Single GPU handles all work
            assignments[0] = GPUWorkAssignment(0, 1:total_trees, :both, true)
            metrics[0] = WorkloadMetrics(0)
        else
            # Dual GPU setup: split trees evenly
            trees_per_gpu = div(total_trees, 2)
            
            # GPU 0: Trees 1-50 + metamodel training
            assignments[0] = GPUWorkAssignment(
                0, 
                1:trees_per_gpu, 
                :training, 
                true
            )
            metrics[0] = WorkloadMetrics(0)
            
            # GPU 1: Trees 51-100 + metamodel inference
            assignments[1] = GPUWorkAssignment(
                1, 
                (trees_per_gpu + 1):total_trees, 
                :inference, 
                false
            )
            metrics[1] = WorkloadMetrics(1)
        end
        
        new(assignments, metrics, total_trees, rebalance_threshold, num_gpus)
    end
end

"""
Create a work distributor with automatic GPU detection
"""
function create_work_distributor(;
    total_trees::Int = 100,
    rebalance_threshold::Float64 = 0.05
)
    return WorkDistributor(
        total_trees = total_trees,
        rebalance_threshold = rebalance_threshold
    )
end

"""
Assign tree work to appropriate GPU
"""
function assign_tree_work(distributor::WorkDistributor, tree_indices::Vector{Int})
    assignments = Dict{Int, Vector{Int}}()
    
    for tree_idx in tree_indices
        gpu_id = get_gpu_for_tree(distributor, tree_idx)
        if !haskey(assignments, gpu_id)
            assignments[gpu_id] = Int[]
        end
        push!(assignments[gpu_id], tree_idx)
    end
    
    return assignments
end

"""
Get GPU ID for a specific tree index
"""
function get_gpu_for_tree(distributor::WorkDistributor, tree_idx::Int)
    for (gpu_id, assignment) in distributor.gpu_assignments
        if tree_idx in assignment.tree_range
            return gpu_id
        end
    end
    
    # Default to GPU 0 if not found
    return 0
end

"""
Get tree range for a specific GPU
"""
function get_tree_range(distributor::WorkDistributor, gpu_id::Int)
    if haskey(distributor.gpu_assignments, gpu_id)
        return distributor.gpu_assignments[gpu_id].tree_range
    else
        return 1:0  # Empty range
    end
end

"""
Assign metamodel work based on operation type
"""
function assign_metamodel_work(distributor::WorkDistributor, operation::Symbol)
    # Find GPU assigned to this metamodel operation
    for (gpu_id, assignment) in distributor.gpu_assignments
        if assignment.metamodel_role == operation || assignment.metamodel_role == :both
            return gpu_id
        end
    end
    
    # Default to GPU 0 if not found
    return 0
end

"""
Update workload metrics for a GPU
"""
function update_metrics!(
    distributor::WorkDistributor, 
    gpu_id::Int;
    trees_processed::Int = 0,
    metamodel_operations::Int = 0,
    time_ms::Float64 = 0.0,
    utilization::Float64 = 0.0,
    memory_mb::Float64 = 0.0
)
    if haskey(distributor.workload_metrics, gpu_id)
        metrics = distributor.workload_metrics[gpu_id]
        metrics.trees_processed += trees_processed
        metrics.metamodel_operations += metamodel_operations
        metrics.total_time_ms += time_ms
        metrics.utilization_percent = utilization
        metrics.memory_used_mb = memory_mb
        metrics.last_update = now()
    end
end

"""
Calculate load balance ratio between GPUs
"""
function get_load_balance_ratio(distributor::WorkDistributor)
    if length(distributor.gpu_assignments) < 2
        return 1.0  # Perfect balance for single GPU
    end
    
    # Calculate total work per GPU
    workloads = Float64[]
    for (gpu_id, metrics) in distributor.workload_metrics
        # Weighted workload: trees + metamodel operations
        workload = metrics.trees_processed + 0.5 * metrics.metamodel_operations
        push!(workloads, workload)
    end
    
    if isempty(workloads) || all(w -> w == 0, workloads)
        return 1.0
    end
    
    # Return ratio of min to max workload
    return minimum(workloads) / maximum(workloads)
end

"""
Check if rebalancing is needed based on current metrics
"""
function rebalance_if_needed!(distributor::WorkDistributor)
    balance_ratio = get_load_balance_ratio(distributor)
    
    if balance_ratio < (1.0 - distributor.rebalance_threshold)
        # Rebalancing needed
        _perform_rebalancing!(distributor)
        return true
    end
    
    return false
end

"""
Perform actual rebalancing of work distribution
"""
function _perform_rebalancing!(distributor::WorkDistributor)
    if length(distributor.gpu_assignments) < 2
        return  # No rebalancing for single GPU
    end
    
    # Calculate new distribution based on performance metrics
    total_capability = 0.0
    gpu_capabilities = Dict{Int, Float64}()
    
    for (gpu_id, metrics) in distributor.workload_metrics
        # Capability based on processing rate
        if metrics.total_time_ms > 0
            rate = metrics.trees_processed / (metrics.total_time_ms / 1000.0)
            gpu_capabilities[gpu_id] = rate
            total_capability += rate
        else
            gpu_capabilities[gpu_id] = 1.0
            total_capability += 1.0
        end
    end
    
    # Redistribute trees proportionally
    trees_assigned = 0
    sorted_gpus = sort(collect(keys(gpu_capabilities)))
    
    for (idx, gpu_id) in enumerate(sorted_gpus)
        proportion = gpu_capabilities[gpu_id] / total_capability
        trees_for_gpu = round(Int, proportion * distributor.total_trees)
        
        # Handle last GPU to ensure all trees are assigned
        if idx == length(sorted_gpus)
            trees_for_gpu = distributor.total_trees - trees_assigned
        end
        
        start_tree = trees_assigned + 1
        end_tree = trees_assigned + trees_for_gpu
        
        # Update assignment
        old_assignment = distributor.gpu_assignments[gpu_id]
        distributor.gpu_assignments[gpu_id] = GPUWorkAssignment(
            gpu_id,
            start_tree:end_tree,
            old_assignment.metamodel_role,
            old_assignment.is_primary
        )
        
        trees_assigned += trees_for_gpu
    end
end

"""
Get summary of current work distribution
"""
function get_work_summary(distributor::WorkDistributor)
    summary = Dict{String, Any}()
    summary["total_trees"] = distributor.total_trees
    summary["num_gpus"] = length(distributor.gpu_assignments)
    summary["assignments"] = Dict{Int, Dict{String, Any}}()
    
    for (gpu_id, assignment) in distributor.gpu_assignments
        gpu_summary = Dict{String, Any}()
        gpu_summary["tree_range"] = "$(first(assignment.tree_range))-$(last(assignment.tree_range))"
        gpu_summary["tree_count"] = length(assignment.tree_range)
        gpu_summary["metamodel_role"] = string(assignment.metamodel_role)
        gpu_summary["is_primary"] = assignment.is_primary
        
        # Add metrics if available
        if haskey(distributor.workload_metrics, gpu_id)
            metrics = distributor.workload_metrics[gpu_id]
            gpu_summary["trees_processed"] = metrics.trees_processed
            gpu_summary["metamodel_ops"] = metrics.metamodel_operations
            gpu_summary["utilization"] = round(metrics.utilization_percent, digits=1)
            gpu_summary["memory_mb"] = round(metrics.memory_used_mb, digits=1)
        end
        
        summary["assignments"][gpu_id] = gpu_summary
    end
    
    summary["load_balance_ratio"] = round(get_load_balance_ratio(distributor), digits=3)
    
    return summary
end

"""
Set GPU affinity for current thread
"""
function set_gpu_affinity(gpu_id::Int)
    if CUDA.functional()
        devices = collect(CUDA.devices())
        if gpu_id + 1 <= length(devices)
            CUDA.device!(devices[gpu_id + 1])  # Convert 0-based to 1-based
        end
    end
end

"""
Execute work on assigned GPU with affinity
"""
function execute_on_gpu(gpu_id::Int, work_fn::Function, args...)
    set_gpu_affinity(gpu_id)
    return work_fn(args...)
end

end # module