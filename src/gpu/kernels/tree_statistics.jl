module TreeStatsAnalysis

using CUDA
using Statistics

# Import required types - will be loaded by including module
include("mcts_types.jl")
using .MCTSTypes: MCTSTreeSoA, MAX_NODES, NODE_ACTIVE, NODE_EXPANDED, NODE_TERMINAL

"""
Real-time tree statistics collection for MCTS
"""
struct TreeStatsCollector
    # Depth histogram - tracks nodes at each depth level
    depth_histogram::CuArray{Int32, 1}      # [max_depth]
    max_depth_observed::CuArray{Int32, 1}   # [1]
    
    # Branching factor statistics
    branching_factors::CuArray{Float32, 1}  # [max_nodes] - per node branching
    avg_branching_factor::CuArray{Float32, 1} # [1]
    
    # Visit count distribution
    visit_buckets::CuArray{Int32, 1}        # [num_buckets] - histogram of visits
    max_visits::CuArray{Int32, 1}           # [1]
    
    # Node state counters
    active_nodes::CuArray{Int32, 1}         # [1]
    expanded_nodes::CuArray{Int32, 1}       # [1]
    terminal_nodes::CuArray{Int32, 1}       # [1]
    recycled_nodes::CuArray{Int32, 1}       # [1]
    
    # Path statistics
    avg_path_length::CuArray{Float32, 1}    # [1]
    path_count::CuArray{Int32, 1}           # [1]
    
    # Update counters
    stats_version::CuArray{Int32, 1}        # [1] - incremented on update
end

"""
Create a new tree statistics collector
"""
function TreeStatsCollector(max_depth::Int32 = Int32(100), num_visit_buckets::Int32 = Int32(20))
    TreeStatsCollector(
        CUDA.zeros(Int32, max_depth),
        CUDA.zeros(Int32, 1),
        CUDA.zeros(Float32, MAX_NODES),
        CUDA.zeros(Float32, 1),
        CUDA.zeros(Int32, num_visit_buckets),
        CUDA.zeros(Int32, 1),
        CUDA.zeros(Int32, 1),
        CUDA.zeros(Int32, 1),
        CUDA.zeros(Int32, 1),
        CUDA.zeros(Int32, 1),
        CUDA.zeros(Float32, 1),
        CUDA.zeros(Int32, 1),
        CUDA.zeros(Int32, 1)
    )
end

"""
Collect tree statistics kernel - runs in parallel across all nodes
"""
function collect_statistics_kernel!(
    node_states::CuDeviceArray{UInt8, 1},
    parent_ids::CuDeviceArray{Int32, 1},
    visit_counts::CuDeviceArray{Int32, 1},
    num_children::CuDeviceArray{Int32, 1},
    depth_histogram::CuDeviceArray{Int32, 1},
    visit_buckets::CuDeviceArray{Int32, 1},
    branching_factors::CuDeviceArray{Float32, 1},
    active_nodes::CuDeviceArray{Int32, 1},
    expanded_nodes::CuDeviceArray{Int32, 1},
    terminal_nodes::CuDeviceArray{Int32, 1},
    max_depth_observed::CuDeviceArray{Int32, 1},
    max_visits::CuDeviceArray{Int32, 1},
    total_nodes::Int32
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    # Shared memory for local reductions
    shared_mem = @cuDynamicSharedMem(Int32, 4)
    shared_active = view(shared_mem, 1:1)
    shared_expanded = view(shared_mem, 2:2)
    shared_terminal = view(shared_mem, 3:3)
    shared_max_depth = view(shared_mem, 4:4)
    
    # Initialize shared memory
    if threadIdx().x == 1
        shared_active[1] = 0
        shared_expanded[1] = 0
        shared_terminal[1] = 0
        shared_max_depth[1] = 0
    end
    sync_threads()
    
    # Process nodes in grid-stride loop
    stride = blockDim().x * gridDim().x
    
    for node_idx in tid:stride:total_nodes
        if node_idx <= total_nodes
            state = node_states[node_idx]
            
            # Count node states
            if state == NODE_ACTIVE
                CUDA.atomic_add!(pointer(shared_active), Int32(1))
            elseif state == NODE_EXPANDED
                CUDA.atomic_add!(pointer(shared_expanded), Int32(1))
                
                # Calculate branching factor for this node
                num_kids = num_children[node_idx]
                branching_factors[node_idx] = Float32(num_kids)
            elseif state == NODE_TERMINAL
                CUDA.atomic_add!(pointer(shared_terminal), Int32(1))
            end
            
            # Calculate node depth
            depth = calculate_node_depth(parent_ids, node_idx)
            if depth > 0 && depth <= length(depth_histogram)
                CUDA.atomic_add!(pointer(depth_histogram, depth), Int32(1))
                CUDA.atomic_max!(pointer(shared_max_depth), depth)
            end
            
            # Visit count distribution
            visits = visit_counts[node_idx]
            if visits > 0
                bucket = min(visit_count_to_bucket(visits), length(visit_buckets))
                CUDA.atomic_add!(pointer(visit_buckets, bucket), Int32(1))
                CUDA.atomic_max!(pointer(max_visits), visits)
            end
        end
    end
    
    sync_threads()
    
    # Reduce block results to global counters
    if threadIdx().x == 1
        CUDA.atomic_add!(pointer(active_nodes), shared_active[1])
        CUDA.atomic_add!(pointer(expanded_nodes), shared_expanded[1])
        CUDA.atomic_add!(pointer(terminal_nodes), shared_terminal[1])
        CUDA.atomic_max!(pointer(max_depth_observed), shared_max_depth[1])
    end
    
    return nothing
end

"""
Calculate average statistics kernel
"""
function calculate_averages_kernel!(
    branching_factors::CuDeviceArray{Float32, 1},
    avg_branching_factor::CuDeviceArray{Float32, 1},
    total_expanded::Int32,
    max_nodes::Int32
)
    tid = threadIdx().x
    
    if tid == 1 && total_expanded > 0
        # Calculate average branching factor
        total_branches = 0.0f0
        for i in 1:max_nodes
            total_branches += branching_factors[i]
        end
        avg_branching_factor[1] = total_branches / Float32(total_expanded)
    end
    
    return nothing
end

"""
Collect path statistics from leaf to root
"""
function collect_path_stats_kernel!(
    parent_ids::CuDeviceArray{Int32, 1},
    avg_path_length::CuDeviceArray{Float32, 1},
    path_count::CuDeviceArray{Int32, 1},
    leaf_nodes::CuDeviceArray{Int32, 1},
    num_leaves::Int32
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= num_leaves
        leaf_idx = leaf_nodes[tid]
        if leaf_idx > 0 && leaf_idx <= MAX_NODES
            # Trace path from leaf to root
            path_length = trace_path_to_root(parent_ids, leaf_idx)
            
            # Update average path length
            CUDA.atomic_add!(pointer(avg_path_length), Float32(path_length))
            CUDA.atomic_add!(pointer(path_count), Int32(1))
        end
    end
    
    return nothing
end

"""
Helper function to calculate node depth
"""
@inline function calculate_node_depth(parent_ids::CuDeviceArray{Int32, 1}, node_idx::Int32)
    depth = Int32(0)
    current = node_idx
    max_iterations = 100  # Prevent infinite loops
    
    while current > 0 && depth < max_iterations
        parent = parent_ids[current]
        if parent <= 0
            break
        end
        current = parent
        depth += 1
    end
    
    return depth
end

"""
Helper function to trace path length from node to root
"""
@inline function trace_path_to_root(parent_ids::CuDeviceArray{Int32, 1}, node_idx::Int32)
    length = Int32(0)
    current = node_idx
    max_iterations = 100
    
    while current > 0 && length < max_iterations
        parent = parent_ids[current]
        if parent <= 0
            break
        end
        current = parent
        length += 1
    end
    
    return length
end

"""
Convert visit count to histogram bucket
"""
@inline function visit_count_to_bucket(visits::Int32)
    if visits <= 10
        return visits
    elseif visits <= 100
        return 10 + (visits - 10) ÷ 10
    elseif visits <= 1000
        return 19 + (visits - 100) ÷ 100
    else
        return 20  # Last bucket for 1000+
    end
end

"""
Statistics summary structure for host access
"""
struct TreeStatsSummary
    total_nodes::Int32
    active_nodes::Int32
    expanded_nodes::Int32
    terminal_nodes::Int32
    recycled_nodes::Int32
    max_depth::Int32
    avg_branching_factor::Float32
    avg_path_length::Float32
    max_visits::Int32
    depth_distribution::Vector{Int32}
    visit_distribution::Vector{Int32}
end

"""
Collect comprehensive tree statistics
"""
function collect_tree_statistics(
    tree::MCTSTreeSoA,
    stats_collector::TreeStatsCollector
)
    # Reset counters
    fill!(stats_collector.active_nodes, 0)
    fill!(stats_collector.expanded_nodes, 0)
    fill!(stats_collector.terminal_nodes, 0)
    fill!(stats_collector.depth_histogram, 0)
    fill!(stats_collector.visit_buckets, 0)
    fill!(stats_collector.max_depth_observed, 0)
    fill!(stats_collector.max_visits, 0)
    
    # Get total nodes
    total_nodes = CUDA.@allowscalar tree.total_nodes[1]
    if total_nodes == 0
        return TreeStatsSummary(
            0, 0, 0, 0, 0, 0, 0.0f0, 0.0f0, 0,
            zeros(Int32, 0), zeros(Int32, 0)
        )
    end
    
    # Launch statistics collection kernel
    threads = 256
    blocks = cld(total_nodes, threads)
    shmem = 4 * sizeof(Int32)  # 4 Int32 values for shared memory
    
    @cuda threads=threads blocks=blocks shmem=shmem collect_statistics_kernel!(
        tree.node_states,
        tree.parent_ids,
        tree.visit_counts,
        tree.num_children,
        stats_collector.depth_histogram,
        stats_collector.visit_buckets,
        stats_collector.branching_factors,
        stats_collector.active_nodes,
        stats_collector.expanded_nodes,
        stats_collector.terminal_nodes,
        stats_collector.max_depth_observed,
        stats_collector.max_visits,
        total_nodes
    )
    
    # Calculate averages
    CUDA.@allowscalar begin
        expanded = stats_collector.expanded_nodes[1]
        if expanded > 0
            @cuda threads=1 calculate_averages_kernel!(
                stats_collector.branching_factors,
                stats_collector.avg_branching_factor,
                expanded,
                total_nodes
            )
        end
        
        # Calculate average path length
        if stats_collector.path_count[1] > 0
            stats_collector.avg_path_length[1] /= Float32(stats_collector.path_count[1])
        end
    end
    
    # Increment version
    CUDA.@allowscalar stats_collector.stats_version[1] += 1
    
    # Create summary
    CUDA.@allowscalar TreeStatsSummary(
        total_nodes,
        stats_collector.active_nodes[1],
        stats_collector.expanded_nodes[1],
        stats_collector.terminal_nodes[1],
        stats_collector.recycled_nodes[1],
        stats_collector.max_depth_observed[1],
        stats_collector.avg_branching_factor[1],
        stats_collector.avg_path_length[1],
        stats_collector.max_visits[1],
        Array(stats_collector.depth_histogram),
        Array(stats_collector.visit_buckets)
    )
end

"""
Generate statistics report
"""
function generate_stats_report(summary::TreeStatsSummary)
    report = Dict{String, Any}()
    
    # Basic counts
    report["node_counts"] = Dict(
        "total" => summary.total_nodes,
        "active" => summary.active_nodes,
        "expanded" => summary.expanded_nodes,
        "terminal" => summary.terminal_nodes,
        "recycled" => summary.recycled_nodes
    )
    
    # Tree structure metrics
    report["tree_structure"] = Dict(
        "max_depth" => summary.max_depth,
        "avg_branching_factor" => round(summary.avg_branching_factor, digits=2),
        "avg_path_length" => round(summary.avg_path_length, digits=2)
    )
    
    # Visit statistics
    report["visit_stats"] = Dict(
        "max_visits" => summary.max_visits,
        "distribution" => summary.visit_distribution
    )
    
    # Depth distribution
    report["depth_distribution"] = summary.depth_distribution
    
    # Tree balance metric
    if summary.max_depth > 0 && summary.expanded_nodes > 0
        theoretical_complete = (summary.avg_branching_factor^summary.max_depth - 1) / 
                              (summary.avg_branching_factor - 1)
        balance_ratio = summary.total_nodes / theoretical_complete
        report["tree_balance"] = round(balance_ratio, digits=3)
    end
    
    return report
end

# Export functions
export TreeStatsCollector, TreeStatsSummary
export collect_tree_statistics, generate_stats_report

end # module TreeStatsAnalysis