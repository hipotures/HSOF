module MCTSTypes

using CUDA

# Constants for MCTS configuration
const MAX_NODES = 1_000_000  # 1M nodes capacity
const MAX_FEATURES = 5000    # Maximum features to track
const FEATURE_CHUNKS = div(MAX_FEATURES + 63, 64)  # Number of UInt64 chunks for bitfield
const WARP_SIZE = 32
const MAX_CHILDREN = 32      # Maximum children per node

# Node states
const NODE_EMPTY = UInt8(0)
const NODE_ACTIVE = UInt8(1)
const NODE_EXPANDED = UInt8(2)
const NODE_TERMINAL = UInt8(3)

"""
Structure of Arrays (SoA) layout for MCTS tree nodes.
Designed for coalesced memory access on GPU.
"""
struct MCTSTreeSoA
    # Core node data (aligned for coalesced access)
    node_ids::CuArray{Int32, 1}          # Unique node identifier
    parent_ids::CuArray{Int32, 1}       # Parent node ID (-1 for root)
    node_states::CuArray{UInt8, 1}      # Node state flags
    
    # MCTS statistics (frequently accessed together)
    visit_counts::CuArray{Int32, 1}     # Number of visits
    total_scores::CuArray{Float32, 1}   # Sum of backpropagated scores
    prior_scores::CuArray{Float32, 1}   # Prior probability from metamodel
    
    # Children information (variable length, stored separately)
    first_child_idx::CuArray{Int32, 1}  # Index of first child (-1 if none)
    num_children::CuArray{Int32, 1}     # Number of children
    
    # Feature selection (bitfield representation)
    # Each node has FEATURE_CHUNKS x UInt64 values
    feature_masks::CuArray{UInt64, 2}   # [FEATURE_CHUNKS x MAX_NODES]
    
    # Memory management
    next_free_node::CuArray{Int32, 1}   # Next available node index (atomic)
    free_list::CuArray{Int32, 1}        # Stack of recycled node indices
    free_list_size::CuArray{Int32, 1}   # Size of free list (atomic)
    
    # Tree statistics (single values)
    total_nodes::CuArray{Int32, 1}      # Total allocated nodes
    max_depth::CuArray{Int32, 1}        # Maximum tree depth
    
    # Constructor
    function MCTSTreeSoA(device::CuDevice)
        # Allocate all arrays on specified device
        CUDA.device!(device) do
            node_ids = CUDA.zeros(Int32, MAX_NODES)
            parent_ids = CUDA.fill(Int32(-1), MAX_NODES)
            node_states = CUDA.zeros(UInt8, MAX_NODES)
            
            visit_counts = CUDA.zeros(Int32, MAX_NODES)
            total_scores = CUDA.zeros(Float32, MAX_NODES)
            prior_scores = CUDA.zeros(Float32, MAX_NODES)
            
            first_child_idx = CUDA.fill(Int32(-1), MAX_NODES)
            num_children = CUDA.zeros(Int32, MAX_NODES)
            
            feature_masks = CUDA.zeros(UInt64, FEATURE_CHUNKS, MAX_NODES)
            
            next_free_node = CUDA.ones(Int32, 1)  # Start at index 1 (0 is root)
            free_list = CUDA.zeros(Int32, MAX_NODES)
            free_list_size = CUDA.zeros(Int32, 1)
            
            total_nodes = CUDA.zeros(Int32, 1)
            max_depth = CUDA.zeros(Int32, 1)
            
            new(node_ids, parent_ids, node_states,
                visit_counts, total_scores, prior_scores,
                first_child_idx, num_children,
                feature_masks,
                next_free_node, free_list, free_list_size,
                total_nodes, max_depth)
        end
    end
end

"""
Configuration for persistent kernel execution
"""
struct PersistentKernelConfig
    block_size::Int32
    grid_size::Int32
    shared_mem_size::Int32
    max_iterations::Int32
    exploration_constant::Float32
    virtual_loss::Int32
    batch_size::Int32
    
    function PersistentKernelConfig(;
        block_size = 256,
        grid_size = 108,  # Optimized for RTX 4090 (128 SMs)
        shared_mem_size = 49152,  # 48KB shared memory
        max_iterations = 1000000,
        exploration_constant = 1.414f0,
        virtual_loss = 10,
        batch_size = 1024
    )
        new(block_size, grid_size, shared_mem_size, max_iterations,
            exploration_constant, virtual_loss, batch_size)
    end
end

"""
Work item for tree operations
"""
struct TreeWorkItem
    operation::UInt8  # 0=select, 1=expand, 2=evaluate, 3=backup
    node_idx::Int32
    thread_id::Int32
    priority::Float32
end

"""
Batch evaluation request
"""
struct BatchEvalRequest
    node_indices::CuArray{Int32, 1}
    feature_masks::CuArray{UInt64, 2}
    request_size::Int32
    request_id::Int32
end

"""
Tree statistics for monitoring
"""
mutable struct TreeStatistics
    nodes_allocated::Int64
    nodes_recycled::Int64
    max_depth_reached::Int32
    avg_branching_factor::Float32
    selection_time_ms::Float32
    expansion_time_ms::Float32
    evaluation_time_ms::Float32
    backup_time_ms::Float32
    gpu_utilization::Float32
    memory_bandwidth_gbps::Float32
    
    TreeStatistics() = new(0, 0, 0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
end

# Helper functions for feature mask operations
@inline function set_feature!(masks::CuArray{UInt64, 2}, node_idx::Int32, feature_idx::Int32)
    chunk_idx = div(feature_idx - 1, 64) + 1
    bit_idx = mod(feature_idx - 1, 64)
    @inbounds CUDA.atomic_or!(pointer(masks, chunk_idx + (node_idx - 1) * FEATURE_CHUNKS), UInt64(1) << bit_idx)
end

@inline function unset_feature!(masks::CuArray{UInt64, 2}, node_idx::Int32, feature_idx::Int32)
    chunk_idx = div(feature_idx - 1, 64) + 1
    bit_idx = mod(feature_idx - 1, 64)
    @inbounds CUDA.atomic_and!(pointer(masks, chunk_idx + (node_idx - 1) * FEATURE_CHUNKS), ~(UInt64(1) << bit_idx))
end

@inline function has_feature(masks::CuArray{UInt64, 2}, node_idx::Int32, feature_idx::Int32)
    chunk_idx = div(feature_idx - 1, 64) + 1
    bit_idx = mod(feature_idx - 1, 64)
    @inbounds return (masks[chunk_idx, node_idx] & (UInt64(1) << bit_idx)) != 0
end

@inline function count_features(masks::CuArray{UInt64, 2}, node_idx::Int32)
    count = Int32(0)
    for i in 1:FEATURE_CHUNKS
        @inbounds count += CUDA.popc(masks[i, node_idx])
    end
    return count
end

# UCB1 score calculation
@inline function ucb1_score(
    total_score::Float32,
    visit_count::Int32,
    parent_visits::Int32,
    exploration_constant::Float32,
    prior_score::Float32
)
    if visit_count == 0
        return Inf32
    end
    
    exploitation = total_score / Float32(visit_count)
    exploration = exploration_constant * sqrt(log(Float32(parent_visits)) / Float32(visit_count))
    prior_bonus = prior_score / Float32(1 + visit_count)
    
    return exploitation + exploration + prior_bonus
end

export MCTSTreeSoA, PersistentKernelConfig, TreeWorkItem, BatchEvalRequest, TreeStatistics
export MAX_NODES, MAX_FEATURES, FEATURE_CHUNKS, WARP_SIZE, MAX_CHILDREN
export NODE_EMPTY, NODE_ACTIVE, NODE_EXPANDED, NODE_TERMINAL
export set_feature!, unset_feature!, has_feature, count_features, ucb1_score

end # module