module KernelStateManagement

using CUDA
using Dates
using Serialization
using UUIDs

"""
Kernel execution state
"""
@enum KernelExecutionState begin
    KERNEL_UNINITIALIZED = 0
    KERNEL_INITIALIZING = 1
    KERNEL_RUNNING = 2
    KERNEL_PAUSED = 3
    KERNEL_STOPPING = 4
    KERNEL_STOPPED = 5
    KERNEL_ERROR = 6
    KERNEL_CHECKPOINTING = 7
    KERNEL_RESTORING = 8
end

"""
Kernel error codes
"""
@enum KernelErrorCode begin
    ERROR_NONE = 0
    ERROR_OUT_OF_MEMORY = 1
    ERROR_INVALID_STATE = 2
    ERROR_CHECKPOINT_FAILED = 3
    ERROR_RESTORE_FAILED = 4
    ERROR_TIMEOUT = 5
    ERROR_CUDA_ERROR = 6
    ERROR_USER_ABORT = 7
end

"""
Kernel state structure for persistent execution
"""
mutable struct KernelState
    # State information
    execution_state::CuArray{Int32, 1}     # Current execution state
    error_code::CuArray{Int32, 1}          # Last error code
    error_count::CuArray{Int32, 1}         # Total error count
    
    # Progress tracking
    iteration::CuArray{Int64, 1}           # Current iteration
    total_iterations::CuArray{Int64, 1}    # Total iterations target
    last_checkpoint_iter::CuArray{Int64, 1} # Last checkpoint iteration
    
    # Timing information
    start_time::CuArray{Float64, 1}        # Kernel start time
    last_update_time::CuArray{Float64, 1}  # Last state update
    total_runtime::CuArray{Float64, 1}     # Total runtime in seconds
    
    # Control flags
    should_stop::CuArray{Bool, 1}          # Stop signal
    should_pause::CuArray{Bool, 1}         # Pause signal
    should_checkpoint::CuArray{Bool, 1}    # Checkpoint signal
    
    # State validation
    state_hash::CuArray{UInt64, 1}         # Hash for state validation
    version::CuArray{Int32, 1}             # State version number
end

"""
State transition rules
"""
const VALID_TRANSITIONS = Dict{KernelExecutionState, Set{KernelExecutionState}}(
    KERNEL_UNINITIALIZED => Set([KERNEL_INITIALIZING]),
    KERNEL_INITIALIZING => Set([KERNEL_RUNNING, KERNEL_ERROR, KERNEL_STOPPED, KERNEL_RESTORING]),
    KERNEL_RUNNING => Set([KERNEL_PAUSED, KERNEL_STOPPING, KERNEL_ERROR, KERNEL_CHECKPOINTING]),
    KERNEL_PAUSED => Set([KERNEL_RUNNING, KERNEL_STOPPING, KERNEL_ERROR]),
    KERNEL_STOPPING => Set([KERNEL_STOPPED, KERNEL_ERROR]),
    KERNEL_STOPPED => Set([KERNEL_INITIALIZING, KERNEL_RESTORING]),
    KERNEL_ERROR => Set([KERNEL_INITIALIZING, KERNEL_STOPPED, KERNEL_RESTORING]),
    KERNEL_CHECKPOINTING => Set([KERNEL_RUNNING, KERNEL_ERROR]),
    KERNEL_RESTORING => Set([KERNEL_RUNNING, KERNEL_PAUSED, KERNEL_ERROR])
)

"""
Checkpoint data structure
"""
struct KernelCheckpoint
    # Metadata
    checkpoint_id::String
    timestamp::DateTime
    iteration::Int64
    state_version::Int32
    
    # State data
    execution_state::KernelExecutionState
    error_code::KernelErrorCode
    total_runtime::Float64
    
    # Application data (serialized)
    app_data::Dict{String, Any}
    
    # Validation
    checksum::UInt64
end

"""
State manager for kernel lifecycle
"""
mutable struct StateManager
    kernel_state::KernelState
    checkpoint_dir::String
    checkpoint_interval::Int64
    max_checkpoints::Int
    auto_recovery::Bool
    
    # Checkpoint history
    checkpoints::Vector{KernelCheckpoint}
    
    # Callbacks
    state_change_callback::Union{Nothing, Function}
    error_callback::Union{Nothing, Function}
end

"""
Create a new kernel state
"""
function KernelState(; version::Int32 = Int32(1))
    KernelState(
        CUDA.fill(Int32(KERNEL_UNINITIALIZED), 1),
        CUDA.zeros(Int32, 1),
        CUDA.zeros(Int32, 1),
        CUDA.zeros(Int64, 1),
        CUDA.zeros(Int64, 1),
        CUDA.zeros(Int64, 1),
        CUDA.zeros(Float64, 1),
        CUDA.zeros(Float64, 1),
        CUDA.zeros(Float64, 1),
        CUDA.zeros(Bool, 1),
        CUDA.zeros(Bool, 1),
        CUDA.zeros(Bool, 1),
        CUDA.zeros(UInt64, 1),
        CUDA.fill(version, 1)
    )
end

"""
Create a new state manager
"""
function StateManager(;
    checkpoint_dir::String = "checkpoints",
    checkpoint_interval::Int64 = 10000,
    max_checkpoints::Int = 5,
    auto_recovery::Bool = true
)
    kernel_state = KernelState()
    
    # Create checkpoint directory if needed
    if !isdir(checkpoint_dir)
        mkpath(checkpoint_dir)
    end
    
    StateManager(
        kernel_state,
        checkpoint_dir,
        checkpoint_interval,
        max_checkpoints,
        auto_recovery,
        KernelCheckpoint[],
        nothing,
        nothing
    )
end

"""
State transition kernel
"""
function transition_state_kernel!(
    execution_state::CuDeviceArray{Int32, 1},
    error_code::CuDeviceArray{Int32, 1},
    state_hash::CuDeviceArray{UInt64, 1},
    last_update_time::CuDeviceArray{Float64, 1},
    new_state::Int32,
    current_time::Float64
)
    tid = threadIdx().x
    
    if tid == 1
        # Update state
        execution_state[1] = new_state
        last_update_time[1] = current_time
        
        # Update state hash
        hash = UInt64(new_state) * UInt64(0x517cc1b727220a95)
        hash = hash ⊻ (hash >> 32)
        hash = hash * UInt64(0x85ebca6b)
        hash = hash ⊻ (hash >> 13)
        state_hash[1] = hash
        
        # Clear error on successful transition
        if new_state != Int32(KERNEL_ERROR)
            error_code[1] = Int32(ERROR_NONE)
        end
    end
    
    return nothing
end

"""
State validation kernel
"""
function validate_state_kernel!(
    execution_state::CuDeviceArray{Int32, 1},
    error_code::CuDeviceArray{Int32, 1},
    error_count::CuDeviceArray{Int32, 1},
    state_hash::CuDeviceArray{UInt64, 1},
    iteration::CuDeviceArray{Int64, 1},
    total_iterations::CuDeviceArray{Int64, 1},
    is_valid::CuDeviceArray{Bool, 1}
)
    tid = threadIdx().x
    
    if tid == 1
        valid = true
        
        # Check state bounds
        state = execution_state[1]
        if state < 0 || state > Int32(KERNEL_RESTORING)
            valid = false
            error_code[1] = Int32(ERROR_INVALID_STATE)
        end
        
        # Check iteration bounds
        iter = iteration[1]
        total = total_iterations[1]
        if iter < 0 || (total > 0 && iter > total)
            valid = false
        end
        
        # Verify state hash
        expected_hash = UInt64(state) * UInt64(0x517cc1b727220a95)
        expected_hash = expected_hash ⊻ (expected_hash >> 32)
        expected_hash = expected_hash * UInt64(0x85ebca6b)
        expected_hash = expected_hash ⊻ (expected_hash >> 13)
        
        if state_hash[1] != expected_hash
            valid = false
            error_code[1] = Int32(ERROR_INVALID_STATE)
        end
        
        is_valid[1] = valid
        
        if !valid
            CUDA.atomic_add!(pointer(error_count), Int32(1))
        end
    end
    
    return nothing
end

"""
Progress update kernel
"""
function update_progress_kernel!(
    iteration::CuDeviceArray{Int64, 1},
    total_runtime::CuDeviceArray{Float64, 1},
    last_update_time::CuDeviceArray{Float64, 1},
    should_checkpoint::CuDeviceArray{Bool, 1},
    last_checkpoint_iter::CuDeviceArray{Int64, 1},
    checkpoint_interval::Int64,
    current_time::Float64,
    delta_iterations::Int64
)
    tid = threadIdx().x
    
    if tid == 1
        # Update iteration count
        old_iter = iteration[1]
        new_iter = old_iter + delta_iterations
        iteration[1] = new_iter
        
        # Update runtime
        last_time = last_update_time[1]
        if last_time > 0.0
            delta_time = current_time - last_time
            total_runtime[1] += delta_time
        end
        last_update_time[1] = current_time
        
        # Check if checkpoint needed
        last_checkpoint = last_checkpoint_iter[1]
        if new_iter - last_checkpoint >= checkpoint_interval
            should_checkpoint[1] = true
        end
    end
    
    return nothing
end

"""
Try to transition to a new state
"""
function try_transition!(manager::StateManager, new_state::KernelExecutionState)
    current_state = get_current_state(manager)
    
    # Check if transition is valid
    if !is_valid_transition(current_state, new_state)
        @warn "Invalid state transition" from=current_state to=new_state
        return false
    end
    
    # Perform transition
    current_time = time()
    @cuda threads=1 transition_state_kernel!(
        manager.kernel_state.execution_state,
        manager.kernel_state.error_code,
        manager.kernel_state.state_hash,
        manager.kernel_state.last_update_time,
        Int32(new_state),
        current_time
    )
    
    # Call state change callback if set
    if !isnothing(manager.state_change_callback)
        manager.state_change_callback(current_state, new_state)
    end
    
    return true
end

"""
Check if state transition is valid
"""
function is_valid_transition(from::KernelExecutionState, to::KernelExecutionState)
    return haskey(VALID_TRANSITIONS, from) && to in VALID_TRANSITIONS[from]
end

"""
Get current execution state
"""
function get_current_state(manager::StateManager)
    state_value = CUDA.@allowscalar manager.kernel_state.execution_state[1]
    return KernelExecutionState(state_value)
end

"""
Set error state
"""
function set_error!(manager::StateManager, error_code::KernelErrorCode, message::String = "")
    CUDA.@allowscalar begin
        manager.kernel_state.error_code[1] = Int32(error_code)
        manager.kernel_state.error_count[1] += 1
    end
    
    try_transition!(manager, KERNEL_ERROR)
    
    # Call error callback if set
    if !isnothing(manager.error_callback)
        manager.error_callback(error_code, message)
    end
    
    @error "Kernel error" code=error_code message=message
end

"""
Update progress
"""
function update_progress!(manager::StateManager, delta_iterations::Int64 = Int64(1))
    current_time = time()
    
    @cuda threads=1 update_progress_kernel!(
        manager.kernel_state.iteration,
        manager.kernel_state.total_runtime,
        manager.kernel_state.last_update_time,
        manager.kernel_state.should_checkpoint,
        manager.kernel_state.last_checkpoint_iter,
        manager.checkpoint_interval,
        current_time,
        delta_iterations
    )
    
    # Check if checkpoint is needed
    if CUDA.@allowscalar manager.kernel_state.should_checkpoint[1]
        perform_checkpoint(manager)
    end
end

"""
Validate kernel state
"""
function validate_state(manager::StateManager)
    is_valid = CUDA.zeros(Bool, 1)
    
    @cuda threads=1 validate_state_kernel!(
        manager.kernel_state.execution_state,
        manager.kernel_state.error_code,
        manager.kernel_state.error_count,
        manager.kernel_state.state_hash,
        manager.kernel_state.iteration,
        manager.kernel_state.total_iterations,
        is_valid
    )
    
    return CUDA.@allowscalar is_valid[1]
end

"""
Create checkpoint
"""
function create_checkpoint(manager::StateManager, app_data::Dict{String, Any} = Dict{String, Any}())
    # Get current state
    CUDA.@allowscalar begin
        checkpoint = KernelCheckpoint(
            string(uuid4()),
            now(),
            manager.kernel_state.iteration[1],
            manager.kernel_state.version[1],
            KernelExecutionState(manager.kernel_state.execution_state[1]),
            KernelErrorCode(manager.kernel_state.error_code[1]),
            manager.kernel_state.total_runtime[1],
            app_data,
            UInt64(0)  # Calculate checksum
        )
        
        # Calculate checksum
        checksum = hash(checkpoint.iteration)
        checksum = hash(checkpoint.state_version, checksum)
        checksum = hash(Int(checkpoint.execution_state), checksum)
        checksum = hash(checkpoint.app_data, checksum)
        
        # Create checkpoint with checksum
        checkpoint = KernelCheckpoint(
            checkpoint.checkpoint_id,
            checkpoint.timestamp,
            checkpoint.iteration,
            checkpoint.state_version,
            checkpoint.execution_state,
            checkpoint.error_code,
            checkpoint.total_runtime,
            checkpoint.app_data,
            checksum
        )
        
        return checkpoint
    end
end

"""
Save checkpoint to disk
"""
function save_checkpoint(manager::StateManager, checkpoint::KernelCheckpoint)
    filename = joinpath(
        manager.checkpoint_dir,
        "checkpoint_$(checkpoint.checkpoint_id)_iter$(checkpoint.iteration).jld2"
    )
    
    try
        serialize(filename, checkpoint)
        push!(manager.checkpoints, checkpoint)
        
        # Maintain max checkpoints limit
        if length(manager.checkpoints) > manager.max_checkpoints
            oldest = manager.checkpoints[1]
            oldest_file = joinpath(
                manager.checkpoint_dir,
                "checkpoint_$(oldest.checkpoint_id)_iter$(oldest.iteration).jld2"
            )
            rm(oldest_file, force=true)
            popfirst!(manager.checkpoints)
        end
        
        return true
    catch e
        @error "Failed to save checkpoint" exception=e
        return false
    end
end

"""
Load checkpoint from disk
"""
function load_checkpoint(filename::String)
    try
        checkpoint = deserialize(filename)
        
        # Validate checksum
        checksum = hash(checkpoint.iteration)
        checksum = hash(checkpoint.state_version, checksum)
        checksum = hash(Int(checkpoint.execution_state), checksum)
        checksum = hash(checkpoint.app_data, checksum)
        
        if checksum != checkpoint.checksum
            @error "Checkpoint checksum mismatch"
            return nothing
        end
        
        return checkpoint
    catch e
        @error "Failed to load checkpoint" exception=e
        return nothing
    end
end

"""
Perform checkpoint operation
"""
function perform_checkpoint(manager::StateManager, app_data::Dict{String, Any} = Dict{String, Any}())
    # Can only checkpoint from running state
    current_state = get_current_state(manager)
    if current_state != KERNEL_RUNNING
        @warn "Cannot checkpoint from state $current_state"
        return false
    end
    
    # Store the original state before transitioning
    original_state = current_state
    
    # Transition to checkpointing state
    if !try_transition!(manager, KERNEL_CHECKPOINTING)
        return false
    end
    
    try
        # Override the state in checkpoint to be the original state
        # since we want to capture the state before checkpointing
        CUDA.@allowscalar begin
            saved_exec_state = manager.kernel_state.execution_state[1]
            manager.kernel_state.execution_state[1] = Int32(original_state)
        end
        
        # Create checkpoint
        checkpoint = create_checkpoint(manager, app_data)
        
        # Restore checkpointing state
        CUDA.@allowscalar manager.kernel_state.execution_state[1] = saved_exec_state
        
        # Save to disk
        if save_checkpoint(manager, checkpoint)
            # Update last checkpoint iteration
            CUDA.@allowscalar begin
                manager.kernel_state.last_checkpoint_iter[1] = checkpoint.iteration
                manager.kernel_state.should_checkpoint[1] = false
            end
            
            @info "Checkpoint saved" id=checkpoint.checkpoint_id iteration=checkpoint.iteration
            
            # Transition back to running if we were running
            if get_current_state(manager) == KERNEL_CHECKPOINTING
                try_transition!(manager, KERNEL_RUNNING)
            end
            return true
        else
            set_error!(manager, ERROR_CHECKPOINT_FAILED)
            return false
        end
    catch e
        @error "Checkpoint failed" exception=e
        set_error!(manager, ERROR_CHECKPOINT_FAILED)
        return false
    end
end

"""
Restore from checkpoint
"""
function restore_checkpoint!(manager::StateManager, checkpoint::KernelCheckpoint)
    # Valid states to restore from
    current_state = get_current_state(manager)
    valid_restore_states = Set([KERNEL_INITIALIZING, KERNEL_STOPPED, KERNEL_ERROR])
    
    if !(current_state in valid_restore_states)
        @warn "Cannot restore from state $current_state"
        return false, Dict{String, Any}()
    end
    
    # Transition to restoring state
    if !try_transition!(manager, KERNEL_RESTORING)
        return false, Dict{String, Any}()
    end
    
    try
        # Restore kernel state
        CUDA.@allowscalar begin
            manager.kernel_state.iteration[1] = checkpoint.iteration
            manager.kernel_state.total_runtime[1] = checkpoint.total_runtime
            manager.kernel_state.error_code[1] = Int32(checkpoint.error_code)
            manager.kernel_state.last_checkpoint_iter[1] = checkpoint.iteration
            manager.kernel_state.version[1] = checkpoint.state_version
        end
        
        @info "Checkpoint restored" id=checkpoint.checkpoint_id iteration=checkpoint.iteration
        
        # Transition to appropriate state
        target_state = checkpoint.execution_state == KERNEL_RUNNING ? KERNEL_PAUSED : checkpoint.execution_state
        try_transition!(manager, target_state)
        
        return true, checkpoint.app_data
    catch e
        @error "Restore failed" exception=e
        set_error!(manager, ERROR_RESTORE_FAILED)
        return false, Dict{String, Any}()
    end
end

"""
Find latest checkpoint
"""
function find_latest_checkpoint(manager::StateManager)
    checkpoint_files = readdir(manager.checkpoint_dir, join=true)
    checkpoint_files = filter(f -> endswith(f, ".jld2"), checkpoint_files)
    
    if isempty(checkpoint_files)
        return nothing
    end
    
    # Sort by modification time
    sort!(checkpoint_files, by=mtime, rev=true)
    
    # Try to load the most recent valid checkpoint
    for file in checkpoint_files
        checkpoint = load_checkpoint(file)
        if !isnothing(checkpoint)
            return checkpoint
        end
    end
    
    return nothing
end

"""
Initialize kernel execution
"""
function initialize_kernel!(manager::StateManager, total_iterations::Int64 = Int64(0))
    if !try_transition!(manager, KERNEL_INITIALIZING)
        return false
    end
    
    # Set initial values
    CUDA.@allowscalar begin
        manager.kernel_state.iteration[1] = 0
        manager.kernel_state.total_iterations[1] = total_iterations
        manager.kernel_state.start_time[1] = time()
        manager.kernel_state.last_update_time[1] = time()
        manager.kernel_state.total_runtime[1] = 0.0
        manager.kernel_state.error_count[1] = 0
    end
    
    # Check for auto-recovery
    if manager.auto_recovery
        latest_checkpoint = find_latest_checkpoint(manager)
        if !isnothing(latest_checkpoint)
            @info "Found checkpoint for auto-recovery" iteration=latest_checkpoint.iteration
            success, _ = restore_checkpoint!(manager, latest_checkpoint)
            if success
                return true
            end
        end
    end
    
    # Transition to running
    return try_transition!(manager, KERNEL_RUNNING)
end

"""
Set control signals
"""
function request_stop!(manager::StateManager)
    CUDA.@allowscalar manager.kernel_state.should_stop[1] = true
end

function request_pause!(manager::StateManager)
    CUDA.@allowscalar manager.kernel_state.should_pause[1] = true
end

function request_resume!(manager::StateManager)
    CUDA.@allowscalar manager.kernel_state.should_pause[1] = false
    try_transition!(manager, KERNEL_RUNNING)
end

"""
Get kernel statistics
"""
function get_kernel_stats(manager::StateManager)
    stats = Dict{String, Any}()
    
    CUDA.@allowscalar begin
        stats["state"] = KernelExecutionState(manager.kernel_state.execution_state[1])
        stats["error_code"] = KernelErrorCode(manager.kernel_state.error_code[1])
        stats["error_count"] = manager.kernel_state.error_count[1]
        stats["iteration"] = manager.kernel_state.iteration[1]
        stats["total_iterations"] = manager.kernel_state.total_iterations[1]
        stats["progress"] = manager.kernel_state.total_iterations[1] > 0 ? 
            Float64(manager.kernel_state.iteration[1]) / Float64(manager.kernel_state.total_iterations[1]) : 0.0
        stats["total_runtime"] = manager.kernel_state.total_runtime[1]
        stats["checkpoints_saved"] = length(manager.checkpoints)
        stats["last_checkpoint_iter"] = manager.kernel_state.last_checkpoint_iter[1]
    end
    
    return stats
end

# Export types and functions
export KernelExecutionState, KernelErrorCode, KernelState, StateManager, KernelCheckpoint
export KERNEL_UNINITIALIZED, KERNEL_INITIALIZING, KERNEL_RUNNING, KERNEL_PAUSED
export KERNEL_STOPPING, KERNEL_STOPPED, KERNEL_ERROR, KERNEL_CHECKPOINTING, KERNEL_RESTORING
export ERROR_NONE, ERROR_OUT_OF_MEMORY, ERROR_INVALID_STATE, ERROR_CHECKPOINT_FAILED
export ERROR_RESTORE_FAILED, ERROR_TIMEOUT, ERROR_CUDA_ERROR, ERROR_USER_ABORT
export try_transition!, get_current_state, set_error!, update_progress!
export validate_state, perform_checkpoint, restore_checkpoint!
export initialize_kernel!, request_stop!, request_pause!, request_resume!
export get_kernel_stats, find_latest_checkpoint

end # module KernelStateManagement