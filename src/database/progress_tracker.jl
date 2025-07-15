module DBProgressTracker

using Dates
using Printf
using Base.Threads: Atomic, atomic_add!

export ProgressTracker, ProgressInfo, update_progress!, finish_progress!,
       estimate_eta, get_throughput, cancel_loading, format_progress

"""
Progress information snapshot
"""
struct ProgressInfo
    rows_processed::Int
    total_rows::Int
    chunks_processed::Int
    total_chunks::Int
    percentage::Float64
    elapsed_seconds::Float64
    eta_seconds::Union{Float64, Nothing}
    throughput_rows_per_sec::Float64
    throughput_mb_per_sec::Float64
    memory_used_gb::Float64
    status::String
    message::String
end

"""
Progress tracker for large dataset operations
"""
mutable struct ProgressTracker
    # Basic counts
    total_rows::Int
    total_chunks::Int
    rows_processed::Atomic{Int}
    chunks_processed::Atomic{Int}
    bytes_processed::Atomic{Int}
    
    # Timing
    start_time::Float64
    last_update_time::Float64
    update_interval::Float64
    
    # Throughput tracking
    throughput_window::Vector{Tuple{Float64, Int, Int}}  # (time, rows, bytes)
    window_size::Int
    
    # Callbacks
    progress_callback::Union{Function, Nothing}
    
    # Control
    cancelled::Atomic{Bool}
    finished::Bool
    
    # Memory monitoring
    initial_memory::Float64
    memory_limit_gb::Float64
    
    # Lock for thread safety
    lock::ReentrantLock
end

"""
    ProgressTracker(total_rows::Int, total_chunks::Int; kwargs...)

Create a new progress tracker for dataset loading operations.

# Arguments
- `total_rows`: Total number of rows to process
- `total_chunks`: Total number of chunks
- `progress_callback`: Optional callback function(::ProgressInfo)
- `update_interval`: Minimum seconds between updates (default: 0.5)
- `memory_limit_gb`: Memory limit for warnings (default: 8.0)
"""
function ProgressTracker(
    total_rows::Int,
    total_chunks::Int;
    progress_callback::Union{Function, Nothing} = nothing,
    update_interval::Float64 = 0.5,
    memory_limit_gb::Float64 = 8.0
)
    current_time = time()
    
    return ProgressTracker(
        total_rows,
        total_chunks,
        Atomic{Int}(0),
        Atomic{Int}(0),
        Atomic{Int}(0),
        current_time,
        current_time,
        update_interval,
        Vector{Tuple{Float64, Int, Int}}(),
        20,  # Keep last 20 measurements for throughput
        progress_callback,
        Atomic{Bool}(false),
        false,
        get_memory_usage(),
        memory_limit_gb,
        ReentrantLock()
    )
end

"""
    update_progress!(tracker::ProgressTracker, rows::Int, bytes::Int = 0; 
                    chunk_completed::Bool = false, message::String = "")

Update progress with processed rows and bytes.
"""
function update_progress!(
    tracker::ProgressTracker, 
    rows::Int, 
    bytes::Int = 0;
    chunk_completed::Bool = false,
    message::String = ""
)
    if tracker.finished || tracker.cancelled[]
        return
    end
    
    # Update atomic counters
    atomic_add!(tracker.rows_processed, rows)
    atomic_add!(tracker.bytes_processed, bytes)
    
    if chunk_completed
        atomic_add!(tracker.chunks_processed, 1)
    end
    
    current_time = time()
    
    # Check if we should emit an update
    should_update = false
    
    lock(tracker.lock) do
        if current_time - tracker.last_update_time >= tracker.update_interval ||
           chunk_completed ||
           tracker.rows_processed[] == tracker.total_rows
            should_update = true
            tracker.last_update_time = current_time
            
            # Update throughput window
            push!(tracker.throughput_window, 
                  (current_time, tracker.rows_processed[], tracker.bytes_processed[]))
            
            # Keep window size limited
            if length(tracker.throughput_window) > tracker.window_size
                popfirst!(tracker.throughput_window)
            end
        end
    end
    
    if should_update
        info = create_progress_info(tracker, message)
        
        # Call callback if provided
        if !isnothing(tracker.progress_callback)
            try
                tracker.progress_callback(info)
            catch e
                @warn "Progress callback error" exception=e
            end
        end
        
        # Check memory usage
        if info.memory_used_gb > tracker.memory_limit_gb
            @warn "Memory usage exceeds limit" used=info.memory_used_gb limit=tracker.memory_limit_gb
        end
    end
end

"""
Create a progress info snapshot
"""
function create_progress_info(tracker::ProgressTracker, message::String)
    rows = tracker.rows_processed[]
    chunks = tracker.chunks_processed[]
    bytes = tracker.bytes_processed[]
    
    percentage = tracker.total_rows > 0 ? 100.0 * rows / tracker.total_rows : 0.0
    elapsed = time() - tracker.start_time
    
    # Calculate throughput
    rows_per_sec, mb_per_sec = calculate_throughput(tracker)
    
    # Estimate ETA
    eta = estimate_eta(tracker, rows, elapsed, rows_per_sec)
    
    # Get memory usage
    current_memory = get_memory_usage()
    memory_used = current_memory - tracker.initial_memory
    
    # Determine status
    status = if tracker.cancelled[]
        "cancelled"
    elseif rows >= tracker.total_rows
        "completed"
    else
        "processing"
    end
    
    return ProgressInfo(
        rows,
        tracker.total_rows,
        chunks,
        tracker.total_chunks,
        percentage,
        elapsed,
        eta,
        rows_per_sec,
        mb_per_sec,
        memory_used,
        status,
        message
    )
end

"""
Calculate throughput from recent measurements
"""
function calculate_throughput(tracker::ProgressTracker)
    lock(tracker.lock) do
        if length(tracker.throughput_window) < 2
            return 0.0, 0.0
        end
        
        # Use recent window for calculation
        start_idx = max(1, length(tracker.throughput_window) - 5)
        
        start_time, start_rows, start_bytes = tracker.throughput_window[start_idx]
        end_time, end_rows, end_bytes = tracker.throughput_window[end]
        
        time_diff = end_time - start_time
        
        if time_diff > 0
            rows_per_sec = (end_rows - start_rows) / time_diff
            mb_per_sec = (end_bytes - start_bytes) / time_diff / 1e6
            return rows_per_sec, mb_per_sec
        else
            return 0.0, 0.0
        end
    end
end

"""
    estimate_eta(tracker::ProgressTracker) -> Union{Float64, Nothing}

Estimate time remaining in seconds.
"""
function estimate_eta(tracker::ProgressTracker, rows::Int, elapsed::Float64, rows_per_sec::Float64)
    if rows_per_sec <= 0 || rows >= tracker.total_rows
        return nothing
    end
    
    remaining_rows = tracker.total_rows - rows
    eta_seconds = remaining_rows / rows_per_sec
    
    # Sanity check - ETA shouldn't be more than 100x elapsed time
    if eta_seconds > elapsed * 100
        return nothing
    end
    
    return eta_seconds
end

"""
    finish_progress!(tracker::ProgressTracker; success::Bool = true, message::String = "")

Mark progress as finished.
"""
function finish_progress!(tracker::ProgressTracker; success::Bool = true, message::String = "")
    tracker.finished = true
    
    # Force final update
    info = create_progress_info(tracker, isempty(message) ? (success ? "Completed" : "Failed") : message)
    
    if !isnothing(tracker.progress_callback)
        try
            tracker.progress_callback(info)
        catch e
            @warn "Progress callback error" exception=e
        end
    end
    
    return info
end

"""
    cancel_loading(tracker::ProgressTracker)

Cancel the loading operation.
"""
function cancel_loading(tracker::ProgressTracker)
    tracker.cancelled[] = true
end

"""
    format_progress(info::ProgressInfo; width::Int = 80) -> String

Format progress info as a string for display.
"""
function format_progress(info::ProgressInfo; width::Int = 80)
    # Progress bar
    bar_width = min(width - 30, 50)
    filled = round(Int, bar_width * info.percentage / 100)
    bar = "[" * "=" ^ filled * " " ^ (bar_width - filled) * "]"
    
    # Format numbers
    rows_str = format_number(info.rows_processed) * "/" * format_number(info.total_rows)
    
    # Format ETA
    eta_str = if isnothing(info.eta_seconds)
        "ETA: --:--"
    else
        "ETA: " * format_duration(info.eta_seconds)
    end
    
    # Format throughput
    throughput_str = @sprintf("%.1fk rows/s", info.throughput_rows_per_sec / 1000)
    
    # Build output
    lines = String[]
    
    # Line 1: Progress bar with percentage
    push!(lines, @sprintf("%s %5.1f%%", bar, info.percentage))
    
    # Line 2: Rows and chunks
    push!(lines, @sprintf("Rows: %s | Chunks: %d/%d", 
                         rows_str, info.chunks_processed, info.total_chunks))
    
    # Line 3: Speed and ETA
    push!(lines, @sprintf("Speed: %s | %s | Mem: %.1f GB", 
                         throughput_str, eta_str, info.memory_used_gb))
    
    # Line 4: Message (if any)
    if !isempty(info.message)
        push!(lines, "Status: " * info.message)
    end
    
    return join(lines, "\n")
end

"""
Format number with thousand separators
"""
function format_number(n::Int)
    str = string(n)
    len = length(str)
    
    if len <= 3
        return str
    end
    
    # Add commas
    parts = String[]
    for i in len:-3:1
        start_idx = max(1, i - 2)
        push!(parts, str[start_idx:i])
    end
    
    return join(reverse(parts), ",")
end

"""
Format duration in seconds to human readable format
"""
function format_duration(seconds::Float64)
    if seconds < 60
        return @sprintf("%ds", round(Int, seconds))
    elseif seconds < 3600
        minutes = floor(Int, seconds / 60)
        secs = round(Int, seconds % 60)
        return @sprintf("%dm%ds", minutes, secs)
    else
        hours = floor(Int, seconds / 3600)
        minutes = floor(Int, (seconds % 3600) / 60)
        return @sprintf("%dh%dm", hours, minutes)
    end
end

"""
Get current memory usage in GB
"""
function get_memory_usage()
    # This is a simplified version - would need proper implementation
    # based on the platform
    if Sys.islinux()
        try
            pid = getpid()
            status = read("/proc/$pid/status", String)
            for line in split(status, '\n')
                if startswith(line, "VmRSS:")
                    kb = parse(Int, split(line)[2])
                    return kb / 1e6  # Convert to GB
                end
            end
        catch
        end
    end
    
    # Fallback - estimate based on Julia's GC
    return Base.gc_num().allocd / 1e9
end

"""
Create a simple console progress callback
"""
function console_progress_callback(width::Int = 80)
    last_output_lines = 0
    
    return function(info::ProgressInfo)
        # Clear previous output
        if last_output_lines > 0
            for _ in 1:last_output_lines
                print("\033[1A\033[2K")  # Move up and clear line
            end
        end
        
        # Format and print new progress
        output = format_progress(info, width=width)
        print(output)
        
        last_output_lines = count(c -> c == '\n', output) + 1
        
        # Add newline if finished
        if info.status in ["completed", "cancelled", "failed"]
            println()
            last_output_lines = 0
        end
        
        flush(stdout)
    end
end

"""
Create a file logging progress callback
"""
function file_progress_callback(filepath::String)
    return function(info::ProgressInfo)
        open(filepath, "a") do f
            timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
            println(f, "$timestamp | $(info.percentage)% | $(info.rows_processed)/$(info.total_rows) rows | " *
                      "$(info.throughput_rows_per_sec) rows/s | $(info.status) | $(info.message)")
        end
    end
end

"""
Create a combined progress callback
"""
function combined_progress_callback(callbacks::Vector{Function})
    return function(info::ProgressInfo)
        for cb in callbacks
            try
                cb(info)
            catch e
                @warn "Callback error" exception=e
            end
        end
    end
end

end # module DBProgressTracker