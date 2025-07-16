module MemoryPool

using CUDA
using ..MCTSTypes

"""
GPU memory pool allocator for MCTS nodes with lock-free operations
"""

# Allocate a new node from the pool
@inline function allocate_node!(tree::MCTSTreeSoA)
    # Try to get from free list first
    free_size = @inbounds tree.free_list_size[1]
    if free_size > 0
        # Attempt to pop from free list
        old_size = CUDA.atomic_sub!(pointer(tree.free_list_size), Int32(1))
        if old_size > 0
            # Successfully claimed a slot
            node_idx = @inbounds tree.free_list[old_size]
            
            # Clear node state
            @inbounds tree.node_states[node_idx] = NODE_ACTIVE
            @inbounds tree.visit_counts[node_idx] = Int32(0)
            @inbounds tree.total_scores[node_idx] = 0.0f0
            
            CUDA.atomic_add!(pointer(tree.total_nodes), Int32(1))
            return node_idx
        else
            # Failed to get from free list, restore counter
            CUDA.atomic_add!(pointer(tree.free_list_size), Int32(1))
        end
    end
    
    # Allocate from next_free_node
    node_idx = CUDA.atomic_add!(pointer(tree.next_free_node), Int32(1))
    
    if node_idx >= MAX_NODES
        # Pool exhausted, restore counter and return failure
        CUDA.atomic_sub!(pointer(tree.next_free_node), Int32(1))
        return Int32(-1)
    end
    
    # Initialize new node
    @inbounds begin
        tree.node_ids[node_idx] = node_idx
        tree.node_states[node_idx] = NODE_ACTIVE
        tree.parent_ids[node_idx] = Int32(-1)
        tree.visit_counts[node_idx] = Int32(0)
        tree.total_scores[node_idx] = 0.0f0
        tree.prior_scores[node_idx] = 0.0f0
        tree.first_child_idx[node_idx] = Int32(-1)
        tree.num_children[node_idx] = Int32(0)
        
        # Clear feature mask
        for i in 1:FEATURE_CHUNKS
            tree.feature_masks[i, node_idx] = UInt64(0)
        end
    end
    
    CUDA.atomic_add!(pointer(tree.total_nodes), Int32(1))
    return node_idx
end

# Free a node back to the pool
@inline function free_node!(tree::MCTSTreeSoA, node_idx::Int32)
    if node_idx <= 0 || node_idx >= MAX_NODES
        return
    end
    
    # Mark as empty
    @inbounds tree.node_states[node_idx] = NODE_EMPTY
    
    # Add to free list
    pos = CUDA.atomic_add!(pointer(tree.free_list_size), Int32(1)) + 1
    if pos <= MAX_NODES
        @inbounds tree.free_list[pos] = node_idx
        CUDA.atomic_sub!(pointer(tree.total_nodes), Int32(1))
    else
        # Free list full, just decrement counter
        CUDA.atomic_sub!(pointer(tree.free_list_size), Int32(1))
    end
end

# Batch allocate nodes for efficiency
function batch_allocate_nodes!(tree::MCTSTreeSoA, count::Int32, output_indices::CuArray{Int32, 1})
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= count
        @inbounds output_indices[tid] = allocate_node!(tree)
    end
    
    return nothing
end

# Recycle an entire subtree
function recycle_subtree_kernel!(tree::MCTSTreeSoA, root_idx::Int32, work_queue::CuArray{Int32, 1}, queue_size::CuArray{Int32, 1})
    tid = threadIdx().x
    bid = blockIdx().x
    
    # Shared memory for local work queue
    shared_queue = @cuDynamicSharedMem(Int32, 512)
    shared_count = @cuDynamicSharedMem(Int32, 1, sizeof(Int32) * 512)
    
    if tid == 1
        shared_count[1] = Int32(0)
    end
    sync_threads()
    
    # Process nodes in BFS order
    if tid == 1 && bid == 1
        @inbounds work_queue[1] = root_idx
        @inbounds queue_size[1] = Int32(1)
    end
    
    sync_threads()
    
    while true
        # Get work from global queue
        work_idx = Int32(-1)
        if tid == 1
            old_size = @inbounds queue_size[1]
            if old_size > 0
                new_size = CUDA.atomic_sub!(pointer(queue_size), Int32(1))
                if new_size > 0
                    work_idx = @inbounds work_queue[new_size]
                else
                    CUDA.atomic_add!(pointer(queue_size), Int32(1))
                end
            end
        end
        
        # Broadcast work item to all threads
        work_idx = CUDA.shfl_sync(0xffffffff, work_idx, 1)
        
        if work_idx < 0
            break  # No more work
        end
        
        # Process children
        if @inbounds tree.node_states[work_idx] != NODE_EMPTY
            first_child = @inbounds tree.first_child_idx[work_idx]
            num_children = @inbounds tree.num_children[work_idx]
            
            # Each thread processes different children
            for i in tid:WARP_SIZE:num_children
                child_idx = first_child + i - 1
                if child_idx > 0 && child_idx <= MAX_NODES
                    # Add to local queue
                    pos = CUDA.atomic_add!(pointer(shared_count), Int32(1)) + 1
                    if pos <= 512
                        @inbounds shared_queue[pos] = child_idx
                    end
                end
            end
            
            sync_threads()
            
            # Free the current node
            if tid == 1
                free_node!(tree, work_idx)
            end
            
            # Move local queue to global queue
            local_count = @inbounds shared_count[1]
            if local_count > 0
                for i in tid:blockDim().x:local_count
                    if i <= local_count
                        node = @inbounds shared_queue[i]
                        pos = CUDA.atomic_add!(pointer(queue_size), Int32(1)) + 1
                        if pos <= MAX_NODES
                            @inbounds work_queue[pos] = node
                        end
                    end
                end
                
                if tid == 1
                    shared_count[1] = Int32(0)
                end
            end
        end
        
        sync_threads()
    end
    
    return nothing
end

# Memory defragmentation kernel
function defragment_memory_kernel!(
    tree::MCTSTreeSoA,
    compaction_map::CuArray{Int32, 1},
    new_positions::CuArray{Int32, 1}
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= MAX_NODES
        node_state = @inbounds tree.node_states[tid]
        
        if node_state != NODE_EMPTY
            # Calculate new position
            new_pos = CUDA.atomic_add!(pointer(new_positions), Int32(1)) + 1
            @inbounds compaction_map[tid] = new_pos
            
            # Copy node data to new position if different
            if new_pos != tid && new_pos <= MAX_NODES
                @inbounds begin
                    # Copy all node data
                    tree.node_ids[new_pos] = tree.node_ids[tid]
                    tree.parent_ids[new_pos] = tree.parent_ids[tid]
                    tree.node_states[new_pos] = tree.node_states[tid]
                    tree.visit_counts[new_pos] = tree.visit_counts[tid]
                    tree.total_scores[new_pos] = tree.total_scores[tid]
                    tree.prior_scores[new_pos] = tree.prior_scores[tid]
                    tree.first_child_idx[new_pos] = tree.first_child_idx[tid]
                    tree.num_children[new_pos] = tree.num_children[tid]
                    
                    # Copy feature mask
                    for i in 1:FEATURE_CHUNKS
                        tree.feature_masks[i, new_pos] = tree.feature_masks[i, tid]
                    end
                    
                    # Clear old position
                    tree.node_states[tid] = NODE_EMPTY
                end
            end
        else
            @inbounds compaction_map[tid] = Int32(-1)
        end
    end
    
    return nothing
end

# Update parent-child relationships after defragmentation
function update_relationships_kernel!(tree::MCTSTreeSoA, compaction_map::CuArray{Int32, 1})
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= MAX_NODES
        new_pos = @inbounds compaction_map[tid]
        
        if new_pos > 0
            # Update parent reference
            old_parent = @inbounds tree.parent_ids[new_pos]
            if old_parent > 0
                new_parent = @inbounds compaction_map[old_parent]
                if new_parent > 0
                    @inbounds tree.parent_ids[new_pos] = new_parent
                end
            end
            
            # Update first child reference
            old_first_child = @inbounds tree.first_child_idx[new_pos]
            if old_first_child > 0
                new_first_child = @inbounds compaction_map[old_first_child]
                if new_first_child > 0
                    @inbounds tree.first_child_idx[new_pos] = new_first_child
                end
            end
        end
    end
    
    return nothing
end

# Host-side memory pool management
struct MemoryPoolManager
    tree::MCTSTreeSoA
    work_queue::CuArray{Int32, 1}
    queue_size::CuArray{Int32, 1}
    compaction_map::CuArray{Int32, 1}
    defrag_threshold::Float32
    
    function MemoryPoolManager(tree::MCTSTreeSoA; defrag_threshold=0.5f0)
        work_queue = CUDA.zeros(Int32, MAX_NODES)
        queue_size = CUDA.zeros(Int32, 1)
        compaction_map = CUDA.zeros(Int32, MAX_NODES)
        
        new(tree, work_queue, queue_size, compaction_map, defrag_threshold)
    end
end

function should_defragment(manager::MemoryPoolManager)
    CUDA.@allowscalar begin
        total = manager.tree.total_nodes[1]
        next_free = manager.tree.next_free_node[1]
        
        if next_free <= 1 || total <= 0
            return false
        end
        
        fragmentation = 1.0f0 - (Float32(total) / Float32(next_free - 1))
        return fragmentation > manager.defrag_threshold
    end
end

function defragment!(manager::MemoryPoolManager)
    # Reset new positions counter
    fill!(manager.queue_size, 0)
    
    # Phase 1: Build compaction map
    @cuda threads=256 blocks=div(MAX_NODES + 255, 256) defragment_memory_kernel!(
        manager.tree, manager.compaction_map, manager.queue_size
    )
    
    # Phase 2: Update relationships
    @cuda threads=256 blocks=div(MAX_NODES + 255, 256) update_relationships_kernel!(
        manager.tree, manager.compaction_map
    )
    
    # Update next_free_node
    new_total = Array(manager.queue_size)[1]
    manager.tree.next_free_node[1] = new_total + 1
    
    # Clear free list as all nodes are now compacted
    manager.tree.free_list_size[1] = 0
    
    return new_total
end

function recycle_subtree!(manager::MemoryPoolManager, root_idx::Int32)
    if root_idx <= 0 || root_idx >= MAX_NODES
        return
    end
    
    # Initialize work queue
    manager.work_queue[1] = root_idx
    manager.queue_size[1] = 1
    
    # Launch recycling kernel
    @cuda threads=32 blocks=32 shmem=sizeof(Int32)*513 recycle_subtree_kernel!(
        manager.tree, root_idx, manager.work_queue, manager.queue_size
    )
    
    return nothing
end

export allocate_node!, free_node!, batch_allocate_nodes!
export MemoryPoolManager, should_defragment, defragment!, recycle_subtree!

end # module