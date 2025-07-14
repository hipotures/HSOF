module ProgressTracking

using CUDA
using Dates
using Printf

# Include the GPUMemoryLayout module
include("gpu_memory_layout.jl")
using .GPUMemoryLayout: WARP_SIZE

"""
Progress tracking configuration
"""
struct ProgressConfig
    update_interval_ms::Int32      # Milliseconds between progress updates
    callback_enabled::Bool         # Enable progress callbacks
    persist_to_disk::Bool          # Save progress for resumability
    monitor_memory::Bool           # Track GPU memory usage
    show_eta::Bool                 # Calculate and show ETA
    progress_file::String          # Path to progress persistence file
end

"""
Create default progress configuration
"""
function create_progress_config(;
                              update_interval_ms::Integer = 100,
                              callback_enabled::Bool = true,
                              persist_to_disk::Bool = false,
                              monitor_memory::Bool = true,
                              show_eta::Bool = true,
                              progress_file::String = ".progress.json")
    return ProgressConfig(
        Int32(update_interval_ms),
        callback_enabled,
        persist_to_disk,
        monitor_memory,
        show_eta,
        progress_file
    )
end

"""
Progress state structure
"""
mutable struct ProgressState
    total_work::Int64              # Total units of work
    completed_work::Int64          # Completed units
    start_time::DateTime           # When processing started
    last_update_time::DateTime     # Last progress update
    current_phase::String          # Current processing phase
    throughput::Float64            # Work units per second
    gpu_memory_used::Int64         # Current GPU memory usage
    gpu_memory_total::Int64        # Total GPU memory
    is_cancelled::Bool             # Cancellation flag
end

"""
GPU progress counters using atomic operations
"""
struct GPUProgressCounters
    features_processed::CuArray{Int64, 1}     # Atomic counter for features
    samples_processed::CuArray{Int64, 1}      # Atomic counter for samples
    operations_completed::CuArray{Int64, 1}   # Generic operation counter
    error_count::CuArray{Int64, 1}            # Error counter
end

"""
Create GPU progress counters
"""
function create_gpu_counters()
    return GPUProgressCounters(
        CUDA.zeros(Int64, 1),
        CUDA.zeros(Int64, 1),
        CUDA.zeros(Int64, 1),
        CUDA.zeros(Int64, 1)
    )
end

"""
Reset GPU counters
"""
function reset_counters!(counters::GPUProgressCounters)
    fill!(counters.features_processed, 0)
    fill!(counters.samples_processed, 0)
    fill!(counters.operations_completed, 0)
    fill!(counters.error_count, 0)
    CUDA.synchronize()
end

"""
GPU kernel for atomic progress increment
"""
function increment_progress_kernel!(
    counter::CuDeviceArray{Int64, 1},
    increment::Int64
)
    # Single thread updates the counter
    if threadIdx().x == 1 && blockIdx().x == 1
        CUDA.atomic_add!(pointer(counter, 1), increment)
    end
    return nothing
end

"""
GPU kernel for batch progress update
"""
function batch_progress_update_kernel!(
    features_counter::CuDeviceArray{Int64, 1},
    samples_counter::CuDeviceArray{Int64, 1},
    feature_increments::CuDeviceArray{Int32, 1},
    n_blocks::Int32
)
    block_id = blockIdx().x
    tid = threadIdx().x
    
    # Shared memory for block-level reduction
    shared_features = @cuDynamicSharedMem(Int64, WARP_SIZE)
    shared_samples = @cuDynamicSharedMem(Int64, WARP_SIZE, WARP_SIZE * sizeof(Int64))
    
    if block_id <= n_blocks
        # Each thread in block accumulates its portion
        local_features = Int64(0)
        local_samples = Int64(0)
        
        if tid == 1
            local_features = Int64(feature_increments[block_id])
            local_samples = Int64(1000)  # Example: 1000 samples per feature
        end
        
        # Warp-level reduction
        warp_id = (tid - 1) ÷ WARP_SIZE
        lane_id = (tid - 1) % WARP_SIZE
        
        # Reduce within warp using shuffle
        for offset in (16, 8, 4, 2, 1)
            local_features += shfl_down_sync(0xffffffff, local_features, offset)
            local_samples += shfl_down_sync(0xffffffff, local_samples, offset)
        end
        
        # First lane stores to shared memory
        if lane_id == 0 && warp_id < WARP_SIZE
            shared_features[warp_id + 1] = local_features
            shared_samples[warp_id + 1] = local_samples
        end
        
        sync_threads()
        
        # Final reduction by first warp
        if tid <= WARP_SIZE
            local_features = tid <= blockDim().x ÷ WARP_SIZE ? shared_features[tid] : Int64(0)
            local_samples = tid <= blockDim().x ÷ WARP_SIZE ? shared_samples[tid] : Int64(0)
            
            for offset in (16, 8, 4, 2, 1)
                local_features += shfl_down_sync(0xffffffff, local_features, offset)
                local_samples += shfl_down_sync(0xffffffff, local_samples, offset)
            end
            
            # Thread 1 updates global counters
            if tid == 1
                CUDA.atomic_add!(pointer(features_counter, 1), local_features)
                CUDA.atomic_add!(pointer(samples_counter, 1), local_samples)
            end
        end
    end
    
    return nothing
end

"""
Progress callback function type
"""
const ProgressCallback = Function

"""
Create default progress callback that prints to console
"""
function create_console_callback()
    return function(state::ProgressState)
        percentage = state.completed_work / state.total_work * 100
        
        # Calculate ETA
        elapsed = Dates.value(state.last_update_time - state.start_time) / 1000.0  # seconds
        if elapsed > 0 && state.completed_work > 0
            rate = state.completed_work / elapsed
            remaining = state.total_work - state.completed_work
            eta_seconds = remaining / rate
            eta_str = format_duration(eta_seconds)
        else
            eta_str = "calculating..."
        end
        
        # Memory info
        mem_used_gb = state.gpu_memory_used / (1024^3)
        mem_total_gb = state.gpu_memory_total / (1024^3)
        mem_percentage = state.gpu_memory_used / state.gpu_memory_total * 100
        
        # Print progress line
        @printf("\r[%s] %s: %.1f%% (%d/%d) | ETA: %s | GPU Mem: %.1f/%.1f GB (%.1f%%) | %.0f items/s",
                Dates.format(now(), "HH:MM:SS"),
                state.current_phase,
                percentage,
                state.completed_work,
                state.total_work,
                eta_str,
                mem_used_gb,
                mem_total_gb,
                mem_percentage,
                state.throughput)
        
        # Flush output
        flush(stdout)
    end
end

"""
Format duration in seconds to human-readable string
"""
function format_duration(seconds::Float64)
    if seconds < 60
        return @sprintf("%.0fs", seconds)
    elseif seconds < 3600
        minutes = floor(seconds / 60)
        secs = seconds % 60
        return @sprintf("%.0fm %.0fs", minutes, secs)
    else
        hours = floor(seconds / 3600)
        minutes = floor((seconds % 3600) / 60)
        return @sprintf("%.0fh %.0fm", hours, minutes)
    end
end

"""
Update progress state from GPU counters
"""
function update_progress_state!(
    state::ProgressState,
    counters::GPUProgressCounters,
    phase::String = ""
)
    # Get current counter values
    features = CUDA.@allowscalar counters.features_processed[1]
    samples = CUDA.@allowscalar counters.samples_processed[1]
    operations = CUDA.@allowscalar counters.operations_completed[1]
    
    # Update completed work (use most relevant counter)
    state.completed_work = max(features, operations)
    
    # Update phase if provided
    if !isempty(phase)
        state.current_phase = phase
    end
    
    # Calculate throughput
    current_time = now()
    time_delta = Dates.value(current_time - state.start_time) / 1000.0  # seconds since start
    if time_delta > 0 && state.completed_work > 0
        state.throughput = state.completed_work / time_delta
    end
    
    state.last_update_time = current_time
    
    # Update GPU memory if monitoring
    if state.gpu_memory_used >= 0
        state.gpu_memory_used = CUDA.used_memory()
        state.gpu_memory_total = CUDA.total_memory()
    end
end

"""
Progress tracker with automatic updates
"""
mutable struct ProgressTracker
    state::ProgressState
    counters::GPUProgressCounters
    config::ProgressConfig
    callback::ProgressCallback
    update_event::CuEvent
    last_callback_time::DateTime
    active::Bool
end

"""
Create progress tracker
"""
function create_progress_tracker(
    total_work::Integer,
    config::ProgressConfig = create_progress_config();
    callback::ProgressCallback = create_console_callback(),
    phase::String = "Processing"
)
    # Get initial GPU memory info
    gpu_used = CUDA.used_memory()
    gpu_total = CUDA.total_memory()
    
    state = ProgressState(
        Int64(total_work),
        0,
        now(),
        now(),
        phase,
        0.0,
        gpu_used,
        gpu_total,
        false
    )
    
    counters = create_gpu_counters()
    update_event = CuEvent()
    
    return ProgressTracker(
        state,
        counters,
        config,
        callback,
        update_event,
        now(),
        true
    )
end

"""
Check if progress update is needed and perform callback
"""
function check_progress_update!(tracker::ProgressTracker)
    if !tracker.active || !tracker.config.callback_enabled
        return
    end
    
    current_time = now()
    time_since_update = Dates.value(current_time - tracker.last_callback_time)
    
    if time_since_update >= tracker.config.update_interval_ms
        # Update state from GPU counters
        update_progress_state!(tracker.state, tracker.counters)
        
        # Call the callback
        tracker.callback(tracker.state)
        
        tracker.last_callback_time = current_time
        
        # Persist if enabled
        if tracker.config.persist_to_disk
            save_progress(tracker)
        end
    end
end

"""
Mark progress tracker as complete
"""
function complete_progress!(tracker::ProgressTracker)
    tracker.state.completed_work = tracker.state.total_work
    update_progress_state!(tracker.state, tracker.counters, "Complete")
    tracker.callback(tracker.state)
    tracker.active = false
    println()  # New line after progress
end

"""
Cancel progress tracking
"""
function cancel_progress!(tracker::ProgressTracker)
    tracker.state.is_cancelled = true
    tracker.active = false
    println("\nProgress cancelled")
end

"""
Save progress state to disk for resumability
"""
function save_progress(tracker::ProgressTracker)
    if !tracker.config.persist_to_disk
        return
    end
    
    # Create progress data dictionary
    progress_data = Dict(
        "total_work" => tracker.state.total_work,
        "completed_work" => tracker.state.completed_work,
        "current_phase" => tracker.state.current_phase,
        "start_time" => string(tracker.state.start_time),
        "last_update_time" => string(tracker.state.last_update_time),
        "is_cancelled" => tracker.state.is_cancelled
    )
    
    # Write to file (simplified - in practice use JSON)
    open(tracker.config.progress_file, "w") do f
        for (k, v) in progress_data
            println(f, "$k=$v")
        end
    end
end

"""
Load progress state from disk
"""
function load_progress(config::ProgressConfig)
    if !isfile(config.progress_file)
        return nothing
    end
    
    progress_data = Dict{String, Any}()
    
    # Read from file (simplified)
    open(config.progress_file, "r") do f
        for line in eachline(f)
            if contains(line, "=")
                k, v = split(line, "=", limit=2)
                progress_data[k] = v
            end
        end
    end
    
    # Convert back to ProgressState
    return progress_data
end

"""
Macro for progress-tracked GPU kernel launch
"""
macro progress_kernel(tracker, kernel_call)
    return quote
        # Launch kernel
        $(esc(kernel_call))
        
        # Record event for timing
        CUDA.record($(esc(tracker)).update_event)
        
        # Check if update needed
        check_progress_update!($(esc(tracker)))
    end
end

"""
Example usage function for progress tracking
"""
function example_gpu_computation_with_progress(n_features::Integer, n_samples::Integer)
    # Create progress tracker
    config = create_progress_config(update_interval_ms=100)
    tracker = create_progress_tracker(n_features, config, phase="Feature Processing")
    
    # Simulated GPU work
    threads = 256
    blocks = cld(n_features, threads)
    
    feature_increments = CUDA.ones(Int32, blocks)
    
    # Launch kernels with progress tracking
    for batch in 1:10
        @progress_kernel tracker begin
            @cuda threads=threads blocks=blocks shmem=2*WARP_SIZE*sizeof(Int64) batch_progress_update_kernel!(
                tracker.counters.features_processed,
                tracker.counters.samples_processed,
                feature_increments,
                Int32(blocks)
            )
        end
        
        # Simulate some work
        sleep(0.1)
        
        # Manual progress check
        check_progress_update!(tracker)
    end
    
    # Mark complete
    complete_progress!(tracker)
end

# Export types and functions
export ProgressConfig, create_progress_config
export ProgressState, GPUProgressCounters, create_gpu_counters
export ProgressTracker, create_progress_tracker
export check_progress_update!, complete_progress!, cancel_progress!
export @progress_kernel
export increment_progress_kernel!, batch_progress_update_kernel!
export save_progress, load_progress
export reset_counters!, update_progress_state!

end # module ProgressTracking