module CompressedNodeStorage

using CUDA
using ..MCTSTypes

"""
Compressed node representation using bit-packed fields to minimize memory usage.
Reduces node size from ~256 bytes to ~64 bytes for 50 trees per GPU.
"""

# Compressed node field bit layouts
const VISIT_COUNT_BITS = 8      # 0-255 direct, >255 logarithmic
const Q_VALUE_BITS = 16         # Fixed-point Q-values
const PRIOR_BITS = 16           # Fixed-point prior scores  
const PARENT_BITS = 16          # Up to 65K nodes per tree
const CHILD_BITS = 16           # First child index
const FEATURE_HASH_BITS = 32    # Hash of feature set

# Fixed-point scaling factors
const Q_VALUE_SCALE = Float32(1000.0)      # 3 decimal places
const PRIOR_SCALE = Float32(10000.0)       # 4 decimal places
const VISIT_COUNT_LOG_THRESHOLD = 255

"""
Compressed node data structure using bit-packed fields.
Each node is exactly 64 bytes for optimal cache line utilization.
"""
struct CompressedNode
    # Packed fields in 64-bit chunks for coalesced access
    chunk1::UInt64  # visit_count(8) + q_value(16) + prior(16) + parent_id(16) + flags(8)
    chunk2::UInt64  # child_count(8) + first_child(16) + feature_hash(32) + depth(8)
    chunk3::UInt64  # Reserved for future use
    chunk4::UInt64  # Reserved for future use
    chunk5::UInt64  # Reserved for future use
    chunk6::UInt64  # Reserved for future use
    chunk7::UInt64  # Reserved for future use
    chunk8::UInt64  # Reserved for future use
end

"""
Compressed tree using Structure-of-Arrays with compressed nodes.
Each tree pre-allocates exactly 20,000 nodes (20K * 64B = 1.28MB per tree).
"""
struct CompressedTreeSoA
    # Compressed node storage (50 trees × 20K nodes × 64B = 64MB per GPU)
    nodes::CuArray{CompressedNode, 2}    # [MAX_NODES_PER_TREE, MAX_TREES_PER_GPU]
    
    # Tree-level metadata
    tree_states::CuArray{UInt8, 1}       # Tree state flags
    root_nodes::CuArray{UInt16, 1}       # Root node index per tree
    next_free_nodes::CuArray{UInt32, 1}  # Next free node per tree (atomic)
    total_nodes::CuArray{UInt32, 1}      # Total allocated nodes per tree
    
    # Memory management per tree
    free_lists::CuArray{UInt16, 2}       # Free node lists [FREE_LIST_SIZE, MAX_TREES_PER_GPU]
    free_list_sizes::CuArray{UInt16, 1}  # Free list sizes per tree (atomic)
    
    # Shared feature storage (described in separate module)
    feature_pool_ids::CuArray{UInt32, 2} # Feature pool references [MAX_NODES_PER_TREE, MAX_TREES_PER_GPU]
    
    # Constants
    max_nodes_per_tree::Int32
    max_trees_per_gpu::Int32
    
    function CompressedTreeSoA(device::CuDevice, max_trees::Int32 = 50, max_nodes::Int32 = 20000)
        CUDA.device!(device) do
            # Pre-allocate compressed node storage
            nodes = CUDA.zeros(CompressedNode, max_nodes, max_trees)
            
            # Tree metadata
            tree_states = CUDA.zeros(UInt8, max_trees)
            root_nodes = CUDA.zeros(UInt16, max_trees)
            next_free_nodes = CUDA.ones(UInt32, max_trees)  # Start at 1 (0 is invalid)
            total_nodes = CUDA.zeros(UInt32, max_trees)
            
            # Memory management
            free_lists = CUDA.zeros(UInt16, max_nodes, max_trees)
            free_list_sizes = CUDA.zeros(UInt16, max_trees)
            
            # Feature pool references
            feature_pool_ids = CUDA.zeros(UInt32, max_nodes, max_trees)
            
            new(nodes, tree_states, root_nodes, next_free_nodes, total_nodes,
                free_lists, free_list_sizes, feature_pool_ids,
                max_nodes, max_trees)
        end
    end
end

# Bit manipulation helpers for compressed fields
@inline function pack_chunk1(visit_count::UInt8, q_value::UInt16, prior::UInt16, parent_id::UInt16, flags::UInt8)
    return (UInt64(visit_count) << 56) | (UInt64(q_value) << 40) | (UInt64(prior) << 24) | (UInt64(parent_id) << 8) | UInt64(flags)
end

@inline function unpack_chunk1(chunk::UInt64)
    visit_count = UInt8((chunk >> 56) & 0xFF)
    q_value = UInt16((chunk >> 40) & 0xFFFF)
    prior = UInt16((chunk >> 24) & 0xFFFF)
    parent_id = UInt16((chunk >> 8) & 0xFFFF)
    flags = UInt8(chunk & 0xFF)
    return visit_count, q_value, prior, parent_id, flags
end

@inline function pack_chunk2(child_count::UInt8, first_child::UInt16, feature_hash::UInt32, depth::UInt8)
    return (UInt64(child_count) << 56) | (UInt64(first_child) << 40) | (UInt64(feature_hash) << 8) | UInt64(depth)
end

@inline function unpack_chunk2(chunk::UInt64)
    child_count = UInt8((chunk >> 56) & 0xFF)
    first_child = UInt16((chunk >> 40) & 0xFFFF)
    feature_hash = UInt32((chunk >> 8) & 0xFFFFFFFF)
    depth = UInt8(chunk & 0xFF)
    return child_count, first_child, feature_hash, depth
end

# Compressed value encoders/decoders
@inline function encode_visit_count(count::Int32)
    if count <= VISIT_COUNT_LOG_THRESHOLD
        return UInt8(count)
    else
        # Logarithmic encoding for values > 255
        log_val = min(255, Int32(floor(log2(Float32(count)))) + VISIT_COUNT_LOG_THRESHOLD - 8)
        return UInt8(log_val)
    end
end

@inline function decode_visit_count(encoded::UInt8)
    if encoded <= VISIT_COUNT_LOG_THRESHOLD
        return Int32(encoded)
    else
        # Decode logarithmic values
        log_offset = encoded - VISIT_COUNT_LOG_THRESHOLD + 8
        return Int32(1 << log_offset)
    end
end

@inline function encode_q_value(value::Float32)
    # Fixed-point encoding with saturation
    scaled = value * Q_VALUE_SCALE
    clamped = max(-32768.0f0, min(32767.0f0, scaled))
    return UInt16(Int16(round(clamped)) + 32768)  # Offset to make unsigned
end

@inline function decode_q_value(encoded::UInt16)
    signed_val = Int16(encoded) - 32768
    return Float32(signed_val) / Q_VALUE_SCALE
end

@inline function encode_prior(value::Float32)
    # Fixed-point encoding for priors [0, 1]
    scaled = value * PRIOR_SCALE
    clamped = max(0.0f0, min(65535.0f0, scaled))
    return UInt16(round(clamped))
end

@inline function decode_prior(encoded::UInt16)
    return Float32(encoded) / PRIOR_SCALE
end

# Node field accessors
@inline function get_visit_count(node::CompressedNode)
    visit_count, _, _, _, _ = unpack_chunk1(node.chunk1)
    return decode_visit_count(visit_count)
end

@inline function set_visit_count!(tree::CompressedTreeSoA, tree_id::Int32, node_id::Int32, count::Int32)
    @inbounds begin
        node = tree.nodes[node_id, tree_id]
        _, q_value, prior, parent_id, flags = unpack_chunk1(node.chunk1)
        new_chunk1 = pack_chunk1(encode_visit_count(count), q_value, prior, parent_id, flags)
        tree.nodes[node_id, tree_id] = CompressedNode(new_chunk1, node.chunk2, node.chunk3, node.chunk4, node.chunk5, node.chunk6, node.chunk7, node.chunk8)
    end
end

@inline function get_q_value(node::CompressedNode)
    _, q_value, _, _, _ = unpack_chunk1(node.chunk1)
    return decode_q_value(q_value)
end

@inline function set_q_value!(tree::CompressedTreeSoA, tree_id::Int32, node_id::Int32, value::Float32)
    @inbounds begin
        node = tree.nodes[node_id, tree_id]
        visit_count, _, prior, parent_id, flags = unpack_chunk1(node.chunk1)
        new_chunk1 = pack_chunk1(visit_count, encode_q_value(value), prior, parent_id, flags)
        tree.nodes[node_id, tree_id] = CompressedNode(new_chunk1, node.chunk2, node.chunk3, node.chunk4, node.chunk5, node.chunk6, node.chunk7, node.chunk8)
    end
end

@inline function get_prior(node::CompressedNode)
    _, _, prior, _, _ = unpack_chunk1(node.chunk1)
    return decode_prior(prior)
end

@inline function set_prior!(tree::CompressedTreeSoA, tree_id::Int32, node_id::Int32, value::Float32)
    @inbounds begin
        node = tree.nodes[node_id, tree_id]
        visit_count, q_value, _, parent_id, flags = unpack_chunk1(node.chunk1)
        new_chunk1 = pack_chunk1(visit_count, q_value, encode_prior(value), parent_id, flags)
        tree.nodes[node_id, tree_id] = CompressedNode(new_chunk1, node.chunk2, node.chunk3, node.chunk4, node.chunk5, node.chunk6, node.chunk7, node.chunk8)
    end
end

@inline function get_parent_id(node::CompressedNode)
    _, _, _, parent_id, _ = unpack_chunk1(node.chunk1)
    return parent_id
end

@inline function set_parent_id!(tree::CompressedTreeSoA, tree_id::Int32, node_id::Int32, parent_id::UInt16)
    @inbounds begin
        node = tree.nodes[node_id, tree_id]
        visit_count, q_value, prior, _, flags = unpack_chunk1(node.chunk1)
        new_chunk1 = pack_chunk1(visit_count, q_value, prior, parent_id, flags)
        tree.nodes[node_id, tree_id] = CompressedNode(new_chunk1, node.chunk2, node.chunk3, node.chunk4, node.chunk5, node.chunk6, node.chunk7, node.chunk8)
    end
end

@inline function get_child_count(node::CompressedNode)
    child_count, _, _, _ = unpack_chunk2(node.chunk2)
    return child_count
end

@inline function set_child_count!(tree::CompressedTreeSoA, tree_id::Int32, node_id::Int32, count::UInt8)
    @inbounds begin
        node = tree.nodes[node_id, tree_id]
        _, first_child, feature_hash, depth = unpack_chunk2(node.chunk2)
        new_chunk2 = pack_chunk2(count, first_child, feature_hash, depth)
        tree.nodes[node_id, tree_id] = CompressedNode(node.chunk1, new_chunk2, node.chunk3, node.chunk4, node.chunk5, node.chunk6, node.chunk7, node.chunk8)
    end
end

@inline function get_first_child(node::CompressedNode)
    _, first_child, _, _ = unpack_chunk2(node.chunk2)
    return first_child
end

@inline function set_first_child!(tree::CompressedTreeSoA, tree_id::Int32, node_id::Int32, first_child::UInt16)
    @inbounds begin
        node = tree.nodes[node_id, tree_id]
        child_count, _, feature_hash, depth = unpack_chunk2(node.chunk2)
        new_chunk2 = pack_chunk2(child_count, first_child, feature_hash, depth)
        tree.nodes[node_id, tree_id] = CompressedNode(node.chunk1, new_chunk2, node.chunk3, node.chunk4, node.chunk5, node.chunk6, node.chunk7, node.chunk8)
    end
end

@inline function get_feature_hash(node::CompressedNode)
    _, _, feature_hash, _ = unpack_chunk2(node.chunk2)
    return feature_hash
end

@inline function set_feature_hash!(tree::CompressedTreeSoA, tree_id::Int32, node_id::Int32, hash::UInt32)
    @inbounds begin
        node = tree.nodes[node_id, tree_id]
        child_count, first_child, _, depth = unpack_chunk2(node.chunk2)
        new_chunk2 = pack_chunk2(child_count, first_child, hash, depth)
        tree.nodes[node_id, tree_id] = CompressedNode(node.chunk1, new_chunk2, node.chunk3, node.chunk4, node.chunk5, node.chunk6, node.chunk7, node.chunk8)
    end
end

@inline function get_depth(node::CompressedNode)
    _, _, _, depth = unpack_chunk2(node.chunk2)
    return depth
end

@inline function set_depth!(tree::CompressedTreeSoA, tree_id::Int32, node_id::Int32, depth::UInt8)
    @inbounds begin
        node = tree.nodes[node_id, tree_id]
        child_count, first_child, feature_hash, _ = unpack_chunk2(node.chunk2)
        new_chunk2 = pack_chunk2(child_count, first_child, feature_hash, depth)
        tree.nodes[node_id, tree_id] = CompressedNode(node.chunk1, new_chunk2, node.chunk3, node.chunk4, node.chunk5, node.chunk6, node.chunk7, node.chunk8)
    end
end

# Node flags (stored in chunk1)
const NODE_FLAG_ACTIVE = UInt8(1)
const NODE_FLAG_EXPANDED = UInt8(2)
const NODE_FLAG_TERMINAL = UInt8(4)
const NODE_FLAG_LAZY_CHILDREN = UInt8(8)

@inline function get_flags(node::CompressedNode)
    _, _, _, _, flags = unpack_chunk1(node.chunk1)
    return flags
end

@inline function set_flags!(tree::CompressedTreeSoA, tree_id::Int32, node_id::Int32, flags::UInt8)
    @inbounds begin
        node = tree.nodes[node_id, tree_id]
        visit_count, q_value, prior, parent_id, _ = unpack_chunk1(node.chunk1)
        new_chunk1 = pack_chunk1(visit_count, q_value, prior, parent_id, flags)
        tree.nodes[node_id, tree_id] = CompressedNode(new_chunk1, node.chunk2, node.chunk3, node.chunk4, node.chunk5, node.chunk6, node.chunk7, node.chunk8)
    end
end

@inline function has_flag(node::CompressedNode, flag::UInt8)
    flags = get_flags(node)
    return (flags & flag) != 0
end

@inline function add_flag!(tree::CompressedTreeSoA, tree_id::Int32, node_id::Int32, flag::UInt8)
    @inbounds begin
        node = tree.nodes[node_id, tree_id]
        current_flags = get_flags(node)
        set_flags!(tree, tree_id, node_id, current_flags | flag)
    end
end

@inline function remove_flag!(tree::CompressedTreeSoA, tree_id::Int32, node_id::Int32, flag::UInt8)
    @inbounds begin
        node = tree.nodes[node_id, tree_id]
        current_flags = get_flags(node)
        set_flags!(tree, tree_id, node_id, current_flags & ~flag)
    end
end

# Memory allocation for compressed nodes
@inline function allocate_compressed_node!(tree::CompressedTreeSoA, tree_id::Int32)
    # Try free list first
    free_size = @inbounds tree.free_list_sizes[tree_id]
    if free_size > 0
        old_size = CUDA.atomic_sub!(pointer(tree.free_list_sizes, tree_id), UInt16(1))
        if old_size > 0
            node_id = @inbounds tree.free_lists[old_size, tree_id]
            CUDA.atomic_add!(pointer(tree.total_nodes, tree_id), UInt32(1))
            return node_id
        else
            # Restore counter
            CUDA.atomic_add!(pointer(tree.free_list_sizes, tree_id), UInt16(1))
        end
    end
    
    # Allocate from sequential pool
    node_id = CUDA.atomic_add!(pointer(tree.next_free_nodes, tree_id), UInt32(1))
    
    if node_id > tree.max_nodes_per_tree
        # Pool exhausted
        CUDA.atomic_sub!(pointer(tree.next_free_nodes, tree_id), UInt32(1))
        return UInt16(0)  # Invalid node ID
    end
    
    CUDA.atomic_add!(pointer(tree.total_nodes, tree_id), UInt32(1))
    return UInt16(node_id)
end

@inline function free_compressed_node!(tree::CompressedTreeSoA, tree_id::Int32, node_id::UInt16)
    if node_id == 0 || node_id > tree.max_nodes_per_tree
        return
    end
    
    # Add to free list
    pos = CUDA.atomic_add!(pointer(tree.free_list_sizes, tree_id), UInt16(1)) + 1
    if pos <= tree.max_nodes_per_tree
        @inbounds tree.free_lists[pos, tree_id] = node_id
        CUDA.atomic_sub!(pointer(tree.total_nodes, tree_id), UInt32(1))
    else
        # Free list full, restore counter
        CUDA.atomic_sub!(pointer(tree.free_list_sizes, tree_id), UInt16(1))
    end
end

# Initialize compressed node with default values
@inline function init_compressed_node!(tree::CompressedTreeSoA, tree_id::Int32, node_id::UInt16, parent_id::UInt16 = UInt16(0))
    # Initialize with default values
    chunk1 = pack_chunk1(UInt8(0), UInt16(0), UInt16(0), parent_id, NODE_FLAG_ACTIVE)
    chunk2 = pack_chunk2(UInt8(0), UInt16(0), UInt32(0), UInt8(0))
    
    @inbounds tree.nodes[node_id, tree_id] = CompressedNode(chunk1, chunk2, UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0))
    @inbounds tree.feature_pool_ids[node_id, tree_id] = UInt32(0)  # No feature set initially
end

export CompressedTreeSoA, CompressedNode
export encode_visit_count, decode_visit_count, encode_q_value, decode_q_value, encode_prior, decode_prior
export get_visit_count, set_visit_count!, get_q_value, set_q_value!, get_prior, set_prior!
export get_parent_id, set_parent_id!, get_child_count, set_child_count!
export get_first_child, set_first_child!, get_feature_hash, set_feature_hash!, get_depth, set_depth!
export get_flags, set_flags!, has_flag, add_flag!, remove_flag!
export NODE_FLAG_ACTIVE, NODE_FLAG_EXPANDED, NODE_FLAG_TERMINAL, NODE_FLAG_LAZY_CHILDREN
export allocate_compressed_node!, free_compressed_node!, init_compressed_node!

end # module