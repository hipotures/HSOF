module GPUSynchronization

using Base.Threads
using Dates
using CUDA

export SyncManager, SyncState, SyncPhase, SyncBarrier, SyncEvent
export create_sync_manager, set_phase!, wait_for_phase, signal_event!
export wait_for_event, enter_barrier!, reset_barrier!, get_sync_state
export acquire_lock!, release_lock!, with_lock, is_phase_complete
export set_gpu_result!, get_gpu_result, get_all_results, clear_results!
export register_gpu!, unregister_gpu!, get_active_gpus
export reset_event!, update_sync_stats!
# Export phase enum values
export PHASE_INIT, PHASE_READY, PHASE_RUNNING, PHASE_SYNCING, PHASE_DONE, PHASE_ERROR

# Synchronization phases for GPU coordination
@enum SyncPhase begin
    PHASE_INIT      = 0  # Initial state
    PHASE_READY     = 1  # GPUs ready to start
    PHASE_RUNNING   = 2  # GPUs executing work
    PHASE_SYNCING   = 3  # Synchronization in progress
    PHASE_DONE      = 4  # Execution complete
    PHASE_ERROR     = 5  # Error state
end

"""
Synchronization state for a single GPU
"""
mutable struct GPUState
    gpu_id::Int
    current_phase::SyncPhase
    last_update::DateTime
    is_active::Bool
    error_message::Union{Nothing, String}
    
    GPUState(gpu_id::Int) = new(gpu_id, PHASE_INIT, now(), true, nothing)
end

"""
Shared results storage for GPU outputs
"""
mutable struct SharedResults
    data::Dict{Int, Any}  # GPU ID -> Result data
    lock::ReentrantLock
    
    SharedResults() = new(Dict{Int, Any}(), ReentrantLock())
end

"""
Synchronization barrier for coordinating multiple GPUs
"""
mutable struct SyncBarrier
    required_count::Int
    current_count::Int
    generation::Int  # To handle barrier reuse
    lock::ReentrantLock
    condition::Condition
    
    function SyncBarrier(n::Integer)
        @assert n > 0 "Barrier count must be positive"
        new(Int(n), 0, 0, ReentrantLock(), Condition())
    end
end

"""
Event-based synchronization primitive
"""
mutable struct SyncEvent
    is_set::Bool
    lock::ReentrantLock
    condition::Condition
    
    SyncEvent() = new(false, ReentrantLock(), Condition())
end

"""
Global synchronization state shared across GPUs
"""
mutable struct SyncState
    gpu_states::Dict{Int, GPUState}
    global_phase::SyncPhase
    start_time::DateTime
    last_sync_time::DateTime
    sync_count::Int
    results::SharedResults
    
    function SyncState()
        new(
            Dict{Int, GPUState}(),
            PHASE_INIT,
            now(),
            now(),
            0,
            SharedResults()
        )
    end
end

"""
Main synchronization manager for GPU coordination
"""
mutable struct SyncManager
    state::SyncState
    state_lock::ReentrantLock
    phase_events::Dict{SyncPhase, SyncEvent}
    barriers::Dict{Symbol, SyncBarrier}
    timeout_ms::Int
    
    function SyncManager(;timeout_ms::Int = 30000)
        # Create phase events for each phase
        phase_events = Dict{SyncPhase, SyncEvent}()
        for phase in instances(SyncPhase)
            phase_events[phase] = SyncEvent()
        end
        
        new(
            SyncState(),
            ReentrantLock(),
            phase_events,
            Dict{Symbol, SyncBarrier}(),
            timeout_ms
        )
    end
end

"""
Create a new synchronization manager
"""
function create_sync_manager(;num_gpus::Int = -1, timeout_ms::Int = 30000)
    manager = SyncManager(timeout_ms=timeout_ms)
    
    # Auto-detect GPUs if not specified
    if num_gpus == -1
        num_gpus = CUDA.functional() ? length(CUDA.devices()) : 0
    end
    
    # Register GPUs
    for gpu_id in 0:(num_gpus-1)
        register_gpu!(manager, gpu_id)
    end
    
    # Create default barriers
    if num_gpus > 0
        manager.barriers[:compute] = SyncBarrier(num_gpus)
        manager.barriers[:sync] = SyncBarrier(num_gpus)
    end
    
    return manager
end

"""
Register a GPU with the synchronization manager
"""
function register_gpu!(manager::SyncManager, gpu_id::Int)
    lock(manager.state_lock) do
        manager.state.gpu_states[gpu_id] = GPUState(gpu_id)
    end
end

"""
Unregister a GPU (e.g., on failure)
"""
function unregister_gpu!(manager::SyncManager, gpu_id::Int)
    lock(manager.state_lock) do
        if haskey(manager.state.gpu_states, gpu_id)
            manager.state.gpu_states[gpu_id].is_active = false
        end
    end
end

"""
Get list of active GPU IDs
"""
function get_active_gpus(manager::SyncManager)
    lock(manager.state_lock) do
        [gpu_id for (gpu_id, state) in manager.state.gpu_states if state.is_active]
    end
end

"""
Set the phase for a specific GPU
"""
function set_phase!(manager::SyncManager, gpu_id::Int, phase::SyncPhase;
                    error_msg::Union{Nothing, String} = nothing)
    lock(manager.state_lock) do
        if haskey(manager.state.gpu_states, gpu_id)
            gpu_state = manager.state.gpu_states[gpu_id]
            gpu_state.current_phase = phase
            gpu_state.last_update = now()
            
            if phase == PHASE_ERROR
                gpu_state.error_message = error_msg
                gpu_state.is_active = false
            end
            
            # Check if all active GPUs reached this phase
            if all_gpus_in_phase(manager, phase)
                manager.state.global_phase = phase
                
                # Signal the phase event
                if haskey(manager.phase_events, phase)
                    signal_event!(manager.phase_events[phase])
                end
            end
        end
    end
end

"""
Check if all active GPUs are in the specified phase
"""
function all_gpus_in_phase(manager::SyncManager, phase::SyncPhase)
    for (gpu_id, state) in manager.state.gpu_states
        if state.is_active && state.current_phase != phase
            return false
        end
    end
    return true
end

"""
Wait for all GPUs to reach a specific phase
"""
function wait_for_phase(manager::SyncManager, phase::SyncPhase;
                       timeout_ms::Union{Nothing, Int} = nothing)
    timeout = something(timeout_ms, manager.timeout_ms)
    start_time = time()
    
    if haskey(manager.phase_events, phase)
        event = manager.phase_events[phase]
        
        while true
            # Check if phase already reached
            lock(manager.state_lock) do
                if manager.state.global_phase == phase
                    return true
                end
            end
            
            # Wait for event with timeout
            elapsed_ms = (time() - start_time) * 1000
            remaining_ms = max(0, timeout - round(Int, elapsed_ms))
            if remaining_ms <= 0
                return false  # Timeout
            end
            
            if !wait_for_event(event, timeout_ms=remaining_ms)
                return false  # Timeout
            end
        end
    end
    
    return false
end

"""
Check if a phase is complete
"""
function is_phase_complete(manager::SyncManager, phase::SyncPhase)
    lock(manager.state_lock) do
        manager.state.global_phase >= phase
    end
end

"""
Signal an event
"""
function signal_event!(event::SyncEvent)
    lock(event.lock) do
        event.is_set = true
        notify(event.condition, all=true)
    end
end

"""
Wait for an event with optional timeout
"""
function wait_for_event(event::SyncEvent; timeout_ms::Int = 30000)
    deadline = time() + timeout_ms / 1000.0
    
    lock(event.lock) do
        while !event.is_set
            remaining = deadline - time()
            if remaining <= 0
                return false  # Timeout
            end
            
            # Wait with timeout using condition  
            wait_start = time()
            while time() - wait_start < remaining
                if event.is_set
                    break
                end
                # Use yield and sleep instead of wait with timeout
                yield()
                sleep(0.001)  # Sleep 1ms
            end
        end
        
        return true
    end
end

"""
Reset an event
"""
function reset_event!(event::SyncEvent)
    lock(event.lock) do
        event.is_set = false
    end
end

"""
Enter a synchronization barrier
"""
function enter_barrier!(barrier::SyncBarrier; timeout_ms::Int = 30000)
    deadline = time() + timeout_ms / 1000.0
    local_generation = 0
    
    lock(barrier.lock) do
        # Increment count
        barrier.current_count += 1
        local_generation = barrier.generation
        
        if barrier.current_count == barrier.required_count
            # Last thread to arrive - release all waiting threads
            barrier.current_count = 0
            barrier.generation += 1
            notify(barrier.condition, all=true)
            return true
        else
            # Wait for other threads
            while local_generation == barrier.generation
                remaining = deadline - time()
                if remaining <= 0
                    # Timeout - decrement count
                    barrier.current_count -= 1
                    return false
                end
                
                # Wait with timeout
                wait_start = time()
                while time() - wait_start < remaining && local_generation == barrier.generation
                    yield()
                    sleep(0.001)  # Sleep 1ms
                end
                
                if local_generation == barrier.generation
                    # Still waiting - timeout
                    barrier.current_count -= 1
                    return false
                end
            end
            
            return true
        end
    end
end

"""
Reset a barrier (use with caution)
"""
function reset_barrier!(barrier::SyncBarrier)
    lock(barrier.lock) do
        barrier.current_count = 0
        barrier.generation += 1
        notify(barrier.condition, all=true)
    end
end

"""
Acquire lock with timeout
"""
function acquire_lock!(lock::ReentrantLock; timeout_ms::Int = 30000)
    deadline = time() + timeout_ms / 1000.0
    
    while time() < deadline
        if trylock(lock)
            return true
        end
        yield()  # Give other threads a chance
        sleep(0.001)  # Small sleep to prevent busy waiting
    end
    
    return false
end

"""
Release lock
"""
function release_lock!(lock::ReentrantLock)
    unlock(lock)
end

"""
Execute function with lock
"""
function with_lock(f::Function, lock::ReentrantLock; timeout_ms::Int = 30000)
    if acquire_lock!(lock, timeout_ms=timeout_ms)
        try
            return f()
        finally
            release_lock!(lock)
        end
    else
        error("Failed to acquire lock within timeout")
    end
end

"""
Store GPU computation result
"""
function set_gpu_result!(manager::SyncManager, gpu_id::Int, result::Any)
    lock(manager.state.results.lock) do
        manager.state.results.data[gpu_id] = result
    end
end

"""
Get GPU computation result
"""
function get_gpu_result(manager::SyncManager, gpu_id::Int)
    lock(manager.state.results.lock) do
        get(manager.state.results.data, gpu_id, nothing)
    end
end

"""
Get all GPU results
"""
function get_all_results(manager::SyncManager)
    lock(manager.state.results.lock) do
        copy(manager.state.results.data)
    end
end

"""
Clear all results
"""
function clear_results!(manager::SyncManager)
    lock(manager.state.results.lock) do
        empty!(manager.state.results.data)
    end
end

"""
Get current synchronization state
"""
function get_sync_state(manager::SyncManager)
    lock(manager.state_lock) do
        state_info = Dict{String, Any}()
        
        # Global state
        state_info["global_phase"] = string(manager.state.global_phase)
        state_info["start_time"] = manager.state.start_time
        state_info["last_sync_time"] = manager.state.last_sync_time
        state_info["sync_count"] = manager.state.sync_count
        
        # Per-GPU state
        gpu_states = Dict{String, Any}()
        for (gpu_id, gpu_state) in manager.state.gpu_states
            gpu_info = Dict(
                "phase" => string(gpu_state.current_phase),
                "active" => gpu_state.is_active,
                "last_update" => gpu_state.last_update,
                "error" => gpu_state.error_message
            )
            gpu_states["GPU$gpu_id"] = gpu_info
        end
        state_info["gpu_states"] = gpu_states
        
        # Barrier states
        barrier_states = Dict{String, Any}()
        for (name, barrier) in manager.barriers
            barrier_info = Dict(
                "waiting" => barrier.current_count,
                "required" => barrier.required_count,
                "generation" => barrier.generation
            )
            barrier_states[string(name)] = barrier_info
        end
        state_info["barriers"] = barrier_states
        
        return state_info
    end
end

"""
Update sync statistics
"""
function update_sync_stats!(manager::SyncManager)
    lock(manager.state_lock) do
        manager.state.last_sync_time = now()
        manager.state.sync_count += 1
    end
end

end # module