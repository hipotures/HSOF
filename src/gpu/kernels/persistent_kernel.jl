module PersistentKernel

using CUDA
using ..MCTSTypes
using ..MemoryPool

# Include synchronization module
include("synchronization.jl")
using .Synchronization

# Include batch evaluation module
include("batch_evaluation.jl")
using .BatchEvaluation

# Kernel state flags
const KERNEL_RUNNING = UInt32(1)
const KERNEL_STOPPING = UInt32(2)
const KERNEL_STOPPED = UInt32(0)

# Work queue operations
const WORK_SELECT = UInt8(0)
const WORK_EXPAND = UInt8(1)
const WORK_EVALUATE = UInt8(2)
const WORK_BACKUP = UInt8(3)

"""
Persistent kernel state structure in constant memory
"""
struct KernelState
    status::CuArray{UInt32, 1}        # Kernel running status
    iteration::CuArray{Int32, 1}      # Current iteration
    work_counter::CuArray{Int32, 1}   # Work items processed
    phase::CuArray{UInt8, 1}          # Current phase (select/expand/eval/backup)
    
    # Performance counters
    selections_count::CuArray{Int64, 1}
    expansions_count::CuArray{Int64, 1}
    evaluations_count::CuArray{Int64, 1}
    backups_count::CuArray{Int64, 1}
    
    # Synchronization components
    grid_barrier::GridBarrier
    phase_sync::PhaseSynchronizer
    tree_sync::TreeSynchronizer
end

"""
Work queue for thread blocks
"""
struct WorkQueue
    items::CuArray{Int32, 2}      # [4 x max_items] - operation, node_idx, thread_id, priority
    head::CuArray{Int32, 1}       # Queue head (atomic)
    tail::CuArray{Int32, 1}       # Queue tail (atomic)
    size::CuArray{Int32, 1}       # Current size (atomic)
    max_size::Int32
end

"""
Main persistent MCTS kernel using grid-stride loops
"""
function persistent_mcts_kernel!(
    tree::MCTSTreeSoA,
    kernel_state::KernelState,
    work_queue::WorkQueue,
    config::PersistentKernelConfig,
    batch_buffer::EvalBatchBuffer,
    eval_config::BatchEvalConfig
)
    # Thread and block indices
    tid = threadIdx().x
    bid = blockIdx().x
    gid = tid + (bid - 1) * blockDim().x
    
    # Shared memory for block-local work
    shared_work = @cuDynamicSharedMem(Int32, 256)
    shared_count = @cuDynamicSharedMem(Int32, 1, sizeof(Int32) * 256)
    
    # Initialize shared memory
    if tid == 1
        shared_count[1] = 0
    end
    sync_threads()
    
    # Main persistent loop
    while @inbounds kernel_state.status[1] == KERNEL_RUNNING
        # Grid-stride loop for work distribution
        stride = blockDim().x * gridDim().x
        
        # Phase-based execution
        current_phase = @inbounds kernel_state.phase[1]
        
        if current_phase == WORK_SELECT
            # Selection phase - traverse tree to find leaf nodes
            perform_selection_phase!(tree, work_queue, config, gid, stride)
            
        elseif current_phase == WORK_EXPAND
            # Expansion phase - create new nodes
            perform_expansion_phase!(tree, work_queue, kernel_state, config, gid, stride)
            
        elseif current_phase == WORK_EVALUATE
            # Evaluation phase - prepare batch for neural network
            batch_eval_pipeline_kernel!(tree, work_queue, batch_buffer, eval_config, current_phase)
            
        elseif current_phase == WORK_BACKUP
            # Backup phase - propagate values up the tree
            batch_eval_pipeline_kernel!(tree, work_queue, batch_buffer, eval_config, current_phase)
            perform_backup_phase!(tree, work_queue, config, gid, stride)
        end
        
        # Synchronize all blocks at phase boundaries using new barrier
        grid_barrier_sync!(
            kernel_state.grid_barrier.arrival_count,
            kernel_state.grid_barrier.release_count,
            kernel_state.grid_barrier.generation,
            kernel_state.grid_barrier.target_count,
            bid, tid
        )
        
        # Phase transition with proper synchronization
        phase_barrier_sync!(
            kernel_state.phase_sync.current_phase,
            kernel_state.phase_sync.phase_counters,
            kernel_state.phase_sync.phase_ready,
            kernel_state.phase_sync.total_blocks,
            current_phase, bid, tid
        )
        
        # Only master thread updates iteration counter
        if bid == 1 && tid == 1
            # Check if we completed a full cycle
            if current_phase == WORK_BACKUP
                CUDA.atomic_add!(pointer(kernel_state.iteration), Int32(1))
                
                # Check termination condition
                if @inbounds kernel_state.iteration[1] >= config.max_iterations
                    @inbounds kernel_state.status[1] = KERNEL_STOPPING
                end
            end
        end
    end
    
    # Kernel cleanup on exit
    if bid == 1 && tid == 1
        @inbounds kernel_state.status[1] = KERNEL_STOPPED
    end
    
    return nothing
end

"""
Selection phase - traverse tree using UCB1 to find leaf nodes
"""
function perform_selection_phase!(
    tree::MCTSTreeSoA,
    work_queue::WorkQueue,
    config::PersistentKernelConfig,
    gid::Int32,
    stride::Int32
)
    # Each thread group processes different root paths
    for start_idx in gid:stride:MAX_NODES
        if @inbounds tree.node_states[start_idx] != NODE_ACTIVE
            continue
        end
        
        current_idx = start_idx
        path_length = 0
        
        # Traverse down to leaf
        while current_idx > 0 && path_length < 100  # Prevent infinite loops
            node_state = @inbounds tree.node_states[current_idx]
            
            if node_state == NODE_EMPTY || node_state == NODE_TERMINAL
                break
            end
            
            num_children = @inbounds tree.num_children[current_idx]
            
            if num_children == 0
                # Found leaf node - add to expansion queue
                add_work_item!(work_queue, WORK_EXPAND, current_idx, gid)
                break
            else
                # Select best child using UCB1
                best_child_idx = select_best_child_ucb1(tree, current_idx, config)
                
                if best_child_idx > 0
                    # Apply virtual loss
                    CUDA.atomic_add!(pointer(tree.visit_counts, best_child_idx), config.virtual_loss)
                    current_idx = best_child_idx
                    path_length += 1
                else
                    break
                end
            end
        end
    end
    
    return nothing
end

"""
Expansion phase - create new child nodes
"""
function perform_expansion_phase!(
    tree::MCTSTreeSoA,
    work_queue::WorkQueue,
    kernel_state::KernelState,
    config::PersistentKernelConfig,
    gid::Int32,
    stride::Int32
)
    # Process expansion work items
    work_size = @inbounds work_queue.size[1]
    
    for idx in gid:stride:work_size
        # Get work item
        operation = @inbounds work_queue.items[1, idx]
        node_idx = @inbounds work_queue.items[2, idx]
        
        if operation != WORK_EXPAND || node_idx <= 0
            continue
        end
        
        # Check if node can be expanded
        if @inbounds tree.node_states[node_idx] != NODE_ACTIVE
            continue
        end
        
        # Acquire write lock for node expansion
        acquire_write_lock!(
            kernel_state.tree_sync.read_locks,
            kernel_state.tree_sync.write_locks,
            node_idx
        )
        
        # Double-check after acquiring lock
        if @inbounds tree.node_states[node_idx] != NODE_ACTIVE
            release_write_lock!(kernel_state.tree_sync.write_locks, node_idx)
            continue
        end
        
        # Allocate child nodes (simplified - normally would use game logic)
        num_new_children = min(MAX_CHILDREN, 10)  # Example: expand 10 children
        first_child = allocate_node!(tree)
        
        if first_child > 0
            @inbounds tree.first_child_idx[node_idx] = first_child
            
            # Allocate remaining children
            for i in 2:num_new_children
                child_idx = allocate_node!(tree)
                if child_idx <= 0
                    num_new_children = i - 1
                    break
                end
                
                # Set parent relationship
                @inbounds tree.parent_ids[child_idx] = node_idx
                
                # Copy and modify parent's feature mask (example logic)
                for j in 1:FEATURE_CHUNKS
                    @inbounds tree.feature_masks[j, child_idx] = tree.feature_masks[j, node_idx]
                end
                
                # Add random feature (simplified)
                feature_to_add = Int32(1 + (child_idx % MAX_FEATURES))
                set_feature!(tree.feature_masks, child_idx, feature_to_add)
            end
            
            @inbounds tree.num_children[node_idx] = num_new_children
            @inbounds tree.node_states[node_idx] = NODE_EXPANDED
            
            # Add to evaluation queue
            add_work_item!(work_queue, WORK_EVALUATE, node_idx, gid)
        end
        
        # Release write lock
        release_write_lock!(kernel_state.tree_sync.write_locks, node_idx)
    end
    
    return nothing
end


"""
Backup phase - propagate values from leaves to root
"""
function perform_backup_phase!(
    tree::MCTSTreeSoA,
    work_queue::WorkQueue,
    config::PersistentKernelConfig,
    gid::Int32,
    stride::Int32
)
    # Process backup work items
    work_size = @inbounds work_queue.size[1]
    
    for idx in gid:stride:work_size
        operation = @inbounds work_queue.items[1, idx]
        node_idx = @inbounds work_queue.items[2, idx]
        
        if operation != WORK_BACKUP || node_idx <= 0
            continue
        end
        
        # Simulate evaluation result (normally from NN)
        eval_score = 0.5f0 + 0.5f0 * sin(Float32(node_idx))
        
        # Backup value to ancestors
        current_idx = node_idx
        backup_count = 0
        
        while current_idx > 0 && backup_count < 100
            # Update node statistics
            CUDA.atomic_add!(pointer(tree.total_scores, current_idx), eval_score)
            
            # Remove virtual loss and add real visit
            old_visits = CUDA.atomic_sub!(pointer(tree.visit_counts, current_idx), 
                                         config.virtual_loss - 1)
            
            # Move to parent
            parent_idx = @inbounds tree.parent_ids[current_idx]
            if parent_idx <= 0
                break  # Reached root
            end
            
            current_idx = parent_idx
            backup_count += 1
        end
    end
    
    return nothing
end

"""
Select best child using UCB1 formula with warp-level primitives
"""
function select_best_child_ucb1(
    tree::MCTSTreeSoA,
    parent_idx::Int32,
    config::PersistentKernelConfig
)
    first_child = @inbounds tree.first_child_idx[parent_idx]
    num_children = @inbounds tree.num_children[parent_idx]
    
    if first_child <= 0 || num_children <= 0
        return Int32(-1)
    end
    
    parent_visits = @inbounds tree.visit_counts[parent_idx]
    best_score = -Inf32
    best_idx = Int32(-1)
    
    # Warp-level parallel search
    lane_id = (threadIdx().x - 1) % WARP_SIZE + 1
    
    for offset in lane_id:WARP_SIZE:num_children
        child_idx = first_child + offset - 1
        
        if child_idx > 0 && child_idx <= MAX_NODES
            child_visits = @inbounds tree.visit_counts[child_idx]
            child_total = @inbounds tree.total_scores[child_idx]
            child_prior = @inbounds tree.prior_scores[child_idx]
            
            score = ucb1_score(child_total, child_visits, parent_visits,
                             config.exploration_constant, child_prior)
            
            if score > best_score
                best_score = score
                best_idx = child_idx
            end
        end
    end
    
    # Warp-level reduction to find global best
    for offset in 16:-1:1
        other_score = CUDA.shfl_down_sync(0xffffffff, best_score, offset)
        other_idx = CUDA.shfl_down_sync(0xffffffff, best_idx, offset)
        
        if other_score > best_score
            best_score = other_score
            best_idx = other_idx
        end
    end
    
    # Lane 0 has the best result
    return CUDA.shfl_sync(0xffffffff, best_idx, 1)
end

"""
Add work item to queue
"""
@inline function add_work_item!(
    work_queue::WorkQueue,
    operation::UInt8,
    node_idx::Int32,
    thread_id::Int32
)
    # Atomic increment tail
    pos = CUDA.atomic_add!(pointer(work_queue.tail), Int32(1))
    
    if pos < work_queue.max_size
        idx = mod(pos, work_queue.max_size) + 1
        @inbounds begin
            work_queue.items[1, idx] = Int32(operation)
            work_queue.items[2, idx] = node_idx
            work_queue.items[3, idx] = thread_id
            work_queue.items[4, idx] = 0  # priority (unused for now)
        end
        
        CUDA.atomic_add!(pointer(work_queue.size), Int32(1))
    end
    
    return nothing
end


# Host-side kernel launcher
function launch_persistent_kernel!(
    tree::MCTSTreeSoA,
    config::PersistentKernelConfig;
    device::CuDevice = CUDA.device()
)
    # Create synchronization components
    grid_barrier = GridBarrier(config.grid_size)
    phase_sync = PhaseSynchronizer(config.grid_size)
    tree_sync = TreeSynchronizer(MAX_NODES)
    
    # Allocate kernel state
    kernel_state = CUDA.device!(device) do
        KernelState(
            CUDA.ones(UInt32, 1) .* KERNEL_RUNNING,
            CUDA.zeros(Int32, 1),
            CUDA.zeros(Int32, 1),
            CUDA.zeros(UInt8, 1),
            CUDA.zeros(Int64, 1),
            CUDA.zeros(Int64, 1),
            CUDA.zeros(Int64, 1),
            CUDA.zeros(Int64, 1),
            grid_barrier,
            phase_sync,
            tree_sync
        )
    end
    
    # Allocate work queue
    work_queue = CUDA.device!(device) do
        WorkQueue(
            CUDA.zeros(Int32, 4, config.batch_size * 10),
            CUDA.zeros(Int32, 1),
            CUDA.zeros(Int32, 1),
            CUDA.zeros(Int32, 1),
            config.batch_size * 10
        )
    end
    
    # Create batch evaluation configuration
    eval_config = BatchEvalConfig(
        config.batch_size,
        true,  # double buffering
        256,   # eval threads
        0.7f0  # coalesce threshold
    )
    
    # Create batch evaluation buffer
    batch_buffer = EvalBatchBuffer(
        config.batch_size,
        MAX_FEATURES,
        32  # max actions
    )
    
    # Launch kernel
    kernel = @cuda threads=config.block_size blocks=config.grid_size shmem=config.shared_mem_size persistent_mcts_kernel!(
        tree, kernel_state, work_queue, config, batch_buffer, eval_config
    )
    
    return kernel, kernel_state, work_queue, batch_buffer
end

function stop_kernel!(kernel_state::KernelState)
    kernel_state.status[1] = KERNEL_STOPPING
    return nothing
end

export KernelState, WorkQueue, persistent_mcts_kernel!
export launch_persistent_kernel!, stop_kernel!
export KERNEL_RUNNING, KERNEL_STOPPING, KERNEL_STOPPED

end # module