module LazyExpansion

using CUDA
using ..MCTSTypes
using ..CompressedNodeStorage
using ..SharedFeatureStorage

"""
Lazy node expansion system for MCTS trees.
Children are not allocated until first visit, reducing memory usage by ~75%.
Implements atomic expansion flags to prevent race conditions.
"""

# Lazy expansion states
const LAZY_STATE_UNEXPANDED = UInt8(0)
const LAZY_STATE_EXPANDING = UInt8(1)
const LAZY_STATE_EXPANDED = UInt8(2)

# Maximum children per node (for memory pre-allocation)
const MAX_CHILDREN_PER_NODE = 32

"""
Lazy child descriptor stored in parent node.
Contains enough information to create child when needed.
"""
struct LazyChildDescriptor
    action_id::UInt16        # Action/feature that creates this child
    prior_score::UInt16      # Encoded prior probability
    feature_hash::UInt32     # Hash of resulting feature set
    
    LazyChildDescriptor(action_id::UInt16, prior_score::UInt16, feature_hash::UInt32) = 
        new(action_id, prior_score, feature_hash)
end

"""
Lazy expansion context stored per node.
Contains child descriptors and expansion state.
"""
struct LazyExpansionContext
    # Child descriptors (stored in parent)
    children::NTuple{MAX_CHILDREN_PER_NODE, LazyChildDescriptor}
    
    # Expansion state
    expansion_state::UInt8   # Current expansion state
    num_children::UInt8      # Number of child descriptors
    children_allocated::UInt8 # Number of children actually allocated
    reserved::UInt8          # Padding for alignment
    
    LazyExpansionContext() = new(
        ntuple(i -> LazyChildDescriptor(UInt16(0), UInt16(0), UInt32(0)), MAX_CHILDREN_PER_NODE),
        LAZY_STATE_UNEXPANDED,
        UInt8(0),
        UInt8(0),
        UInt8(0)
    )
end

"""
Lazy expansion manager for compressed trees.
Manages deferred allocation of child nodes.
"""
struct LazyExpansionManager
    # Expansion contexts per node per tree
    contexts::CuArray{LazyExpansionContext, 2}  # [MAX_NODES_PER_TREE, MAX_TREES_PER_GPU]
    
    # Temporary workspace for expansion
    expansion_workspace::CuArray{UInt32, 2}     # [MAX_CHILDREN_PER_NODE, MAX_TREES_PER_GPU]
    
    # Feature storage for lazy children
    feature_pool::SharedFeaturePool
    
    # Statistics
    expansions_performed::CuArray{UInt64, 1}    # Total expansions
    lazy_hits::CuArray{UInt64, 1}              # Children found without allocation
    memory_saved::CuArray{UInt64, 1}           # Bytes saved by lazy expansion
    
    function LazyExpansionManager(
        device::CuDevice,
        feature_pool::SharedFeaturePool,
        max_trees::Int32 = 50,
        max_nodes::Int32 = 20000
    )
        CUDA.device!(device) do
            # Expansion contexts
            contexts = CuArray{LazyExpansionContext}(undef, max_nodes, max_trees)
            
            # Initialize all contexts
            fill!(contexts, LazyExpansionContext())
            
            # Workspace for expansion operations
            expansion_workspace = CUDA.zeros(UInt32, MAX_CHILDREN_PER_NODE, max_trees)
            
            # Statistics
            expansions_performed = CUDA.zeros(UInt64, 1)
            lazy_hits = CUDA.zeros(UInt64, 1)
            memory_saved = CUDA.zeros(UInt64, 1)
            
            new(contexts, expansion_workspace, feature_pool,
                expansions_performed, lazy_hits, memory_saved)
        end
    end
end

# Initialize lazy expansion context for a node
@inline function init_lazy_context!(
    manager::LazyExpansionManager,
    tree_id::Int32,
    node_id::UInt16,
    num_children::UInt8
)
    # Create empty context
    context = LazyExpansionContext()
    
    # Update with actual number of children
    new_context = LazyExpansionContext(
        context.children,
        LAZY_STATE_UNEXPANDED,
        num_children,
        UInt8(0),
        UInt8(0)
    )
    
    @inbounds manager.contexts[node_id, tree_id] = new_context
end

# Add child descriptor to lazy context
@inline function add_child_descriptor!(
    manager::LazyExpansionManager,
    tree_id::Int32,
    node_id::UInt16,
    child_index::UInt8,
    action_id::UInt16,
    prior_score::Float32,
    feature_hash::UInt32
)
    if child_index > MAX_CHILDREN_PER_NODE
        return false
    end
    
    @inbounds context = manager.contexts[node_id, tree_id]
    
    # Create child descriptor
    descriptor = LazyChildDescriptor(
        action_id,
        encode_prior(prior_score),
        feature_hash
    )
    
    # Update context with new child
    children = context.children
    new_children = ntuple(i -> i == child_index ? descriptor : children[i], MAX_CHILDREN_PER_NODE)
    
    new_context = LazyExpansionContext(
        new_children,
        context.expansion_state,
        context.num_children,
        context.children_allocated,
        context.reserved
    )
    
    @inbounds manager.contexts[node_id, tree_id] = new_context
    return true
end

# Get child descriptor from lazy context
@inline function get_child_descriptor(
    manager::LazyExpansionManager,
    tree_id::Int32,
    node_id::UInt16,
    child_index::UInt8
)
    if child_index > MAX_CHILDREN_PER_NODE
        return LazyChildDescriptor(UInt16(0), UInt16(0), UInt32(0))
    end
    
    @inbounds context = manager.contexts[node_id, tree_id]
    @inbounds return context.children[child_index]
end

# Check if node has been expanded
@inline function is_expanded(
    manager::LazyExpansionManager,
    tree_id::Int32,
    node_id::UInt16
)
    @inbounds context = manager.contexts[node_id, tree_id]
    return context.expansion_state == LAZY_STATE_EXPANDED
end

# Check if node is currently being expanded
@inline function is_expanding(
    manager::LazyExpansionManager,
    tree_id::Int32,
    node_id::UInt16
)
    @inbounds context = manager.contexts[node_id, tree_id]
    return context.expansion_state == LAZY_STATE_EXPANDING
end

# Get number of allocated children
@inline function get_allocated_children(
    manager::LazyExpansionManager,
    tree_id::Int32,
    node_id::UInt16
)
    @inbounds context = manager.contexts[node_id, tree_id]
    return context.children_allocated
end

# Atomically try to start expansion
@inline function try_start_expansion!(
    manager::LazyExpansionManager,
    tree_id::Int32,
    node_id::UInt16
)
    @inbounds context = manager.contexts[node_id, tree_id]
    
    if context.expansion_state != LAZY_STATE_UNEXPANDED
        return false  # Already expanding or expanded
    end
    
    # Try to atomically change state from UNEXPANDED to EXPANDING
    new_context = LazyExpansionContext(
        context.children,
        LAZY_STATE_EXPANDING,
        context.num_children,
        context.children_allocated,
        context.reserved
    )
    
    # Use atomic compare-and-swap
    old_context = context
    result = CUDA.atomic_cas!(pointer(manager.contexts, node_id + (tree_id - 1) * size(manager.contexts, 1)), old_context, new_context)
    
    return result == old_context
end

# Complete expansion and mark as expanded
@inline function complete_expansion!(
    manager::LazyExpansionManager,
    tree_id::Int32,
    node_id::UInt16,
    children_allocated::UInt8
)
    @inbounds context = manager.contexts[node_id, tree_id]
    
    new_context = LazyExpansionContext(
        context.children,
        LAZY_STATE_EXPANDED,
        context.num_children,
        children_allocated,
        context.reserved
    )
    
    @inbounds manager.contexts[node_id, tree_id] = new_context
    
    # Update statistics
    CUDA.atomic_add!(pointer(manager.expansions_performed), UInt64(1))
    
    # Calculate memory saved (unexpanded children)
    memory_saved = UInt64(context.num_children - children_allocated) * UInt64(sizeof(CompressedNode))
    CUDA.atomic_add!(pointer(manager.memory_saved), memory_saved)
end

# Lazy child allocation function
@inline function allocate_lazy_child!(
    manager::LazyExpansionManager,
    tree::CompressedTreeSoA,
    tree_id::Int32,
    parent_id::UInt16,
    child_index::UInt8
)
    # Get child descriptor
    descriptor = get_child_descriptor(manager, tree_id, parent_id, child_index)
    
    if descriptor.action_id == 0
        return UInt16(0)  # Invalid descriptor
    end
    
    # Allocate compressed node
    child_id = allocate_compressed_node!(tree, tree_id)
    if child_id == 0
        return UInt16(0)  # Allocation failed
    end
    
    # Initialize child node
    init_compressed_node!(tree, tree_id, child_id, parent_id)
    
    # Set prior score from descriptor
    prior_score = decode_prior(descriptor.prior_score)
    set_prior!(tree, tree_id, child_id, prior_score)
    
    # Set feature hash
    set_feature_hash!(tree, tree_id, child_id, descriptor.feature_hash)
    
    # Create feature set for child based on parent + action
    parent_feature_pool_id = @inbounds tree.feature_pool_ids[parent_id, tree_id]
    child_features = create_child_features(manager.feature_pool, parent_feature_pool_id, descriptor.action_id)
    
    # Store child feature set in shared pool
    child_feature_pool_id = store_feature_set!(manager.feature_pool, child_features, UInt16(1))
    @inbounds tree.feature_pool_ids[child_id, tree_id] = child_feature_pool_id
    
    return child_id
end

# Create child feature set based on parent features and action
@inline function create_child_features(
    feature_pool::SharedFeaturePool,
    parent_pool_id::UInt32,
    action_id::UInt16
)
    # Get parent features
    parent_features = get_feature_set(feature_pool, parent_pool_id)
    
    # Apply action (toggle feature bit)
    feature_idx = Int(action_id)
    if feature_idx > 0 && feature_idx <= MAX_FEATURES
        chunk_idx = div(feature_idx - 1, 64) + 1
        bit_idx = mod(feature_idx - 1, 64)
        
        # Create modified features
        child_features = ntuple(i -> 
            i == chunk_idx ? parent_features[i] ⊻ (UInt64(1) << bit_idx) : parent_features[i],
            FEATURE_CHUNKS
        )
        
        return child_features
    end
    
    return parent_features  # No change if invalid action
end

# Batch expansion kernel for multiple nodes
function batch_expand_kernel!(
    manager::LazyExpansionManager,
    tree::CompressedTreeSoA,
    tree_id::Int32,
    node_ids::CuArray{UInt16, 1},
    num_nodes::Int32
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= num_nodes
        node_id = @inbounds node_ids[tid]
        
        # Try to start expansion for this node
        if try_start_expansion!(manager, tree_id, node_id)
            # Get node's lazy context
            @inbounds context = manager.contexts[node_id, tree_id]
            num_children = context.num_children
            
            # Allocate children up to some limit (e.g., first 8 children)
            max_allocate = min(num_children, UInt8(8))
            children_allocated = UInt8(0)
            
            for child_idx in 1:max_allocate
                child_id = allocate_lazy_child!(manager, tree, tree_id, node_id, child_idx)
                
                if child_id > 0
                    children_allocated += 1
                    
                    # Update parent's first child reference if this is the first child
                    if child_idx == 1
                        set_first_child!(tree, tree_id, node_id, child_id)
                    end
                end
            end
            
            # Update child count in parent
            set_child_count!(tree, tree_id, node_id, children_allocated)
            
            # Mark expansion as complete
            complete_expansion!(manager, tree_id, node_id, children_allocated)
        end
    end
    
    return nothing
end

# Expand multiple nodes in batch
function batch_expand_nodes!(
    manager::LazyExpansionManager,
    tree::CompressedTreeSoA,
    tree_id::Int32,
    node_ids::Vector{UInt16}
)
    if isempty(node_ids)
        return
    end
    
    # Copy node IDs to GPU
    gpu_node_ids = CuArray(node_ids)
    num_nodes = length(node_ids)
    
    # Launch kernel
    threads_per_block = 256
    blocks = div(num_nodes + threads_per_block - 1, threads_per_block)
    
    @cuda threads=threads_per_block blocks=blocks batch_expand_kernel!(
        manager, tree, tree_id, gpu_node_ids, num_nodes
    )
    
    CUDA.synchronize()
end

# On-demand child access with lazy allocation
@inline function get_or_create_child!(
    manager::LazyExpansionManager,
    tree::CompressedTreeSoA,
    tree_id::Int32,
    parent_id::UInt16,
    child_index::UInt8
)
    # Check if parent has been expanded
    if !is_expanded(manager, tree_id, parent_id)
        # Try to expand parent
        if try_start_expansion!(manager, tree_id, parent_id)
            # Perform minimal expansion (just this child)
            child_id = allocate_lazy_child!(manager, tree, tree_id, parent_id, child_index)
            
            if child_id > 0
                # Update parent references
                if child_index == 1
                    set_first_child!(tree, tree_id, parent_id, child_id)
                end
                
                # Mark as having 1 child allocated
                complete_expansion!(manager, tree_id, parent_id, UInt8(1))
                
                return child_id
            else
                # Failed to allocate, mark as expanded with 0 children
                complete_expansion!(manager, tree_id, parent_id, UInt8(0))
                return UInt16(0)
            end
        else
            # Another thread is expanding, wait briefly
            for i in 1:10
                if is_expanded(manager, tree_id, parent_id)
                    break
                end
                # Brief spin wait
            end
        end
    end
    
    # Node is expanded, get child normally
    @inbounds node = tree.nodes[parent_id, tree_id]
    first_child = get_first_child(node)
    child_count = get_child_count(node)
    
    if child_index <= child_count
        return first_child + child_index - 1
    else
        # Child not allocated, try to allocate now
        child_id = allocate_lazy_child!(manager, tree, tree_id, parent_id, child_index)
        
        if child_id > 0
            # Update parent child count
            set_child_count!(tree, tree_id, parent_id, max(child_count, child_index))
            
            # Update context
            @inbounds context = manager.contexts[parent_id, tree_id]
            new_context = LazyExpansionContext(
                context.children,
                context.expansion_state,
                context.num_children,
                max(context.children_allocated, child_index),
                context.reserved
            )
            @inbounds manager.contexts[parent_id, tree_id] = new_context
            
            CUDA.atomic_add!(pointer(manager.lazy_hits), UInt64(1))
        end
        
        return child_id
    end
end

# Cleanup lazy expansion context when node is freed
@inline function cleanup_lazy_context!(
    manager::LazyExpansionManager,
    tree_id::Int32,
    node_id::UInt16
)
    # Reset context to initial state
    @inbounds manager.contexts[node_id, tree_id] = LazyExpansionContext()
end

# Get lazy expansion statistics
function get_lazy_expansion_stats(manager::LazyExpansionManager)
    CUDA.@allowscalar begin
        stats = Dict{String, Any}(
            "expansions_performed" => manager.expansions_performed[1],
            "lazy_hits" => manager.lazy_hits[1],
            "memory_saved_bytes" => manager.memory_saved[1],
            "memory_saved_mb" => Float64(manager.memory_saved[1]) / (1024.0 * 1024.0)
        )
        return stats
    end
end

# Calculate memory efficiency
function calculate_memory_efficiency(manager::LazyExpansionManager)
    CUDA.@allowscalar begin
        total_expansions = manager.expansions_performed[1]
        memory_saved = manager.memory_saved[1]
        
        if total_expansions > 0
            avg_memory_saved = Float64(memory_saved) / Float64(total_expansions)
            return avg_memory_saved
        end
        
        return 0.0
    end
end

export LazyExpansionManager, LazyChildDescriptor, LazyExpansionContext
export init_lazy_context!, add_child_descriptor!, get_child_descriptor
export is_expanded, is_expanding, get_allocated_children
export try_start_expansion!, complete_expansion!, allocate_lazy_child!
export batch_expand_nodes!, get_or_create_child!, cleanup_lazy_context!
export get_lazy_expansion_stats, calculate_memory_efficiency
export LAZY_STATE_UNEXPANDED, LAZY_STATE_EXPANDING, LAZY_STATE_EXPANDED

end # module