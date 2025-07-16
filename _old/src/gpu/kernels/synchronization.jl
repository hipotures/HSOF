module Synchronization

using CUDA

"""
Barrier types for different synchronization levels
"""
const BARRIER_BLOCK = UInt8(0)
const BARRIER_GRID = UInt8(1)
const BARRIER_TREE = UInt8(2)
const BARRIER_PHASE = UInt8(3)

"""
Grid-wide synchronization barrier using atomic counters
"""
struct GridBarrier
    arrival_count::CuArray{Int32, 1}    # Threads arrived at barrier
    release_count::CuArray{Int32, 1}    # Threads released from barrier
    generation::CuArray{Int32, 1}       # Barrier generation number
    target_count::Int32                  # Expected thread count
end

"""
Phase synchronization for coordinating MCTS phases
"""
struct PhaseSynchronizer
    current_phase::CuArray{UInt8, 1}    # Current execution phase
    phase_counters::CuArray{Int32, 2}   # [4 x num_blocks] counters per phase
    phase_ready::CuArray{Bool, 1}       # Phase ready flags
    total_blocks::Int32
end

"""
Tree operation synchronization for consistency
"""
struct TreeSynchronizer
    read_locks::CuArray{Int32, 1}       # Read lock counters per node
    write_locks::CuArray{Int32, 1}      # Write lock flags per node
    global_lock::CuArray{Int32, 1}      # Global tree lock
    operation_counter::CuArray{Int64, 1} # Global operation counter
end

"""
Create a grid-wide barrier for specified number of thread blocks
"""
function GridBarrier(num_blocks::Int32)
    arrival_count = CUDA.zeros(Int32, 1)
    release_count = CUDA.zeros(Int32, 1) 
    generation = CUDA.zeros(Int32, 1)
    target_count = num_blocks
    
    GridBarrier(arrival_count, release_count, generation, target_count)
end

"""
Synchronize all blocks at a grid-wide barrier
"""
function grid_barrier_sync!(
    arrival_count::CuDeviceArray{Int32, 1},
    release_count::CuDeviceArray{Int32, 1},
    generation::CuDeviceArray{Int32, 1},
    target_count::Int32,
    block_id::Int32,
    thread_id::Int32
)
    # Only thread 0 of each block participates
    if thread_id != 1
        sync_threads()
        return
    end
    
    # Get current generation
    local_gen = @inbounds generation[1]
    
    # Increment arrival counter
    arrived = CUDA.atomic_add!(pointer(arrival_count), Int32(1)) + 1
    
    if arrived == target_count
        # Last block to arrive, reset counters and advance generation
        @inbounds arrival_count[1] = 0
        @inbounds release_count[1] = 0
        CUDA.atomic_add!(pointer(generation), Int32(1))
    else
        # Wait for generation to advance
        while @inbounds generation[1] == local_gen
            # Yield to prevent spinning
            CUDA.sync_warp()
        end
    end
    
    sync_threads()
end

"""
Phase synchronizer for MCTS phases (select, expand, evaluate, backup)
"""
function PhaseSynchronizer(num_blocks::Int32)
    current_phase = CUDA.zeros(UInt8, 1)
    phase_counters = CUDA.zeros(Int32, 4, num_blocks)
    phase_ready = CUDA.zeros(Bool, 4)
    
    PhaseSynchronizer(current_phase, phase_counters, phase_ready, num_blocks)
end

"""
Synchronize phase transitions across all blocks
"""
function phase_barrier_sync!(
    current_phase::CuDeviceArray{UInt8, 1},
    phase_counters::CuDeviceArray{Int32, 2},
    phase_ready::CuDeviceArray{Bool, 1},
    total_blocks::Int32,
    phase::UInt8,
    block_id::Int32,
    thread_id::Int32
)
    # Only thread 0 participates in phase sync
    if thread_id == 1
        # Mark this block as ready for next phase
        @inbounds phase_counters[phase + 1, block_id] = 1
        
        # Check if all blocks are ready
        if block_id == 1
            all_ready = true
            for bid in 1:total_blocks
                if @inbounds phase_counters[phase + 1, bid] == 0
                    all_ready = false
                    break
                end
            end
            
            if all_ready
                # Clear counters for this phase
                for bid in 1:total_blocks
                    @inbounds phase_counters[phase + 1, bid] = 0
                end
                
                # Advance to next phase
                next_phase = (phase + 1) % 4
                @inbounds current_phase[1] = next_phase
                @inbounds phase_ready[next_phase + 1] = true
            end
        end
        
        # Wait for phase transition
        target_phase = (phase + 1) % 4
        while @inbounds current_phase[1] != target_phase
            CUDA.sync_warp()
        end
    end
    
    sync_threads()
end

"""
Tree synchronizer for safe concurrent tree operations
"""
function TreeSynchronizer(max_nodes::Int32)
    read_locks = CUDA.zeros(Int32, max_nodes)
    write_locks = CUDA.zeros(Int32, max_nodes)
    global_lock = CUDA.zeros(Int32, 1)
    operation_counter = CUDA.zeros(Int64, 1)
    
    TreeSynchronizer(read_locks, write_locks, global_lock, operation_counter)
end

"""
Acquire read lock on tree node
"""
@inline function acquire_read_lock!(
    read_locks::CuDeviceArray{Int32, 1},
    write_locks::CuDeviceArray{Int32, 1},
    node_idx::Int32
)
    # Wait if write lock is held
    while @inbounds write_locks[node_idx] != 0
        CUDA.sync_warp()
    end
    
    # Increment read counter
    CUDA.atomic_add!(pointer(read_locks, node_idx), Int32(1))
end

"""
Release read lock on tree node
"""
@inline function release_read_lock!(
    read_locks::CuDeviceArray{Int32, 1},
    node_idx::Int32
)
    CUDA.atomic_sub!(pointer(read_locks, node_idx), Int32(1))
end

"""
Acquire write lock on tree node
"""
@inline function acquire_write_lock!(
    read_locks::CuDeviceArray{Int32, 1},
    write_locks::CuDeviceArray{Int32, 1},
    node_idx::Int32
)
    # Try to acquire write lock
    while CUDA.atomic_cas!(
        pointer(write_locks, node_idx),
        Int32(0),
        Int32(1)
    ) != 0
        CUDA.sync_warp()
    end
    
    # Wait for all readers to finish
    while @inbounds read_locks[node_idx] != 0
        CUDA.sync_warp()
    end
end

"""
Release write lock on tree node  
"""
@inline function release_write_lock!(
    write_locks::CuDeviceArray{Int32, 1},
    node_idx::Int32
)
    @inbounds write_locks[node_idx] = 0
    CUDA.threadfence()
end

"""
Global tree lock for structural modifications
"""
function acquire_global_tree_lock!(sync::TreeSynchronizer)
    while CUDA.atomic_cas!(
        pointer(sync.global_lock),
        Int32(0),
        Int32(1)
    ) != 0
        CUDA.sync_warp()
    end
end

function release_global_tree_lock!(sync::TreeSynchronizer)
    @inbounds sync.global_lock[1] = 0
    CUDA.threadfence_system()
end

"""
Deadlock detection mechanism
"""
struct DeadlockDetector
    wait_counters::CuArray{Int32, 2}    # [num_blocks x num_resources]
    deadlock_flag::CuArray{Bool, 1}
    timeout_cycles::Int64
end

function check_deadlock!(
    detector::DeadlockDetector,
    block_id::Int32,
    resource_id::Int32,
    waiting::Bool
)
    if waiting
        count = CUDA.atomic_add!(
            pointer(detector.wait_counters, resource_id + (block_id - 1) * size(detector.wait_counters, 1)),
            Int32(1)
        )
        
        if count > detector.timeout_cycles
            @inbounds detector.deadlock_flag[1] = true
        end
    else
        @inbounds detector.wait_counters[resource_id, block_id] = 0
    end
    
    return @inbounds detector.deadlock_flag[1]
end

"""
Cooperative group synchronization wrapper
"""
function cooperative_sync!(sync_level::UInt8)
    if sync_level == BARRIER_BLOCK
        sync_threads()
    elseif sync_level == BARRIER_GRID
        # Grid-wide sync requires cooperative kernel launch
        CUDA.grid_sync()
    elseif sync_level == BARRIER_TREE
        # Tree-wide sync through global barrier
        CUDA.threadfence_system()
    end
end

"""
Warp-level synchronization utilities
"""
@inline function sync_warp_ballot(predicate::Bool)
    return CUDA.vote_ballot_sync(0xffffffff, predicate)
end

@inline function sync_warp_any(predicate::Bool)
    return CUDA.vote_any_sync(0xffffffff, predicate)
end

@inline function sync_warp_all(predicate::Bool)
    return CUDA.vote_all_sync(0xffffffff, predicate)
end

export GridBarrier, PhaseSynchronizer, TreeSynchronizer, DeadlockDetector
export grid_barrier_sync!, phase_barrier_sync!, cooperative_sync!
export acquire_read_lock!, release_read_lock!
export acquire_write_lock!, release_write_lock!
export acquire_global_tree_lock!, release_global_tree_lock!
export check_deadlock!
export sync_warp_ballot, sync_warp_any, sync_warp_all
export BARRIER_BLOCK, BARRIER_GRID, BARRIER_TREE, BARRIER_PHASE

end # module