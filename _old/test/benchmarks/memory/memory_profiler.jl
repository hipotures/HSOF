module MemoryProfiler

using CUDA
using Statistics
using Dates

export MemoryProfile, MemorySnapshot, MemoryAllocation
export profile_memory, track_allocations, memory_summary
export HostMemoryTracker, DeviceMemoryTracker

"""
Memory allocation event
"""
struct MemoryAllocation
    timestamp::DateTime
    allocation_type::Symbol  # :host or :device
    size_bytes::Int64
    operation::String
    backtrace::Vector{Base.StackTraces.StackFrame}
end

"""
Memory snapshot at a point in time
"""
struct MemorySnapshot
    timestamp::DateTime
    host_used_mb::Float64
    host_available_mb::Float64
    device_used_mb::Float64
    device_available_mb::Float64
    device_reserved_mb::Float64
    gc_count::Int64
    gc_time_ms::Float64
end

"""
Complete memory profile
"""
struct MemoryProfile
    operation_name::String
    start_snapshot::MemorySnapshot
    end_snapshot::MemorySnapshot
    peak_host_mb::Float64
    peak_device_mb::Float64
    allocations::Vector{MemoryAllocation}
    snapshots::Vector{MemorySnapshot}
    total_allocated_host_mb::Float64
    total_allocated_device_mb::Float64
    gc_events::Int
end

"""
Track host memory allocations
"""
mutable struct HostMemoryTracker
    enabled::Bool
    allocations::Vector{MemoryAllocation}
    initial_memory::Float64
    peak_memory::Float64
end

"""
Track device memory allocations
"""
mutable struct DeviceMemoryTracker
    enabled::Bool
    device_id::Int
    allocations::Vector{MemoryAllocation}
    initial_memory::Float64
    peak_memory::Float64
end

# Global trackers
const HOST_TRACKER = HostMemoryTracker(false, [], 0.0, 0.0)
const DEVICE_TRACKERS = Dict{Int, DeviceMemoryTracker}()

"""
Get current memory snapshot
"""
function get_memory_snapshot()::MemorySnapshot
    # Host memory
    gc_stats = Base.gc_num()
    host_used = Base.gc_live_bytes() / 1024^2
    
    # Estimate available host memory (Linux-specific)
    host_available = if Sys.islinux()
        try
            meminfo = read("/proc/meminfo", String)
            available_line = match(r"MemAvailable:\s+(\d+)", meminfo)
            if !isnothing(available_line)
                parse(Float64, available_line.captures[1]) / 1024  # KB to MB
            else
                0.0
            end
        catch
            0.0
        end
    else
        0.0
    end
    
    # Device memory
    if CUDA.functional()
        device_stats = CUDA.memory_stats()
        device_used = device_stats.live_bytes / 1024^2
        device_reserved = device_stats.reserved_bytes / 1024^2
        device_available = CUDA.available_memory() / 1024^2
    else
        device_used = 0.0
        device_reserved = 0.0
        device_available = 0.0
    end
    
    return MemorySnapshot(
        now(),
        host_used,
        host_available,
        device_used,
        device_available,
        device_reserved,
        gc_stats.total_time / 1e6,  # Convert to ms
        gc_stats.pause
    )
end

"""
Start memory tracking
"""
function start_memory_tracking(; device_id::Int = 0)
    # Reset trackers
    HOST_TRACKER.enabled = true
    HOST_TRACKER.allocations = []
    HOST_TRACKER.initial_memory = Base.gc_live_bytes() / 1024^2
    HOST_TRACKER.peak_memory = HOST_TRACKER.initial_memory
    
    if CUDA.functional()
        device = CuDevice(device_id)
        tracker = DeviceMemoryTracker(
            true,
            device_id,
            [],
            CUDA.memory_stats().live_bytes / 1024^2,
            CUDA.memory_stats().live_bytes / 1024^2
        )
        DEVICE_TRACKERS[device_id] = tracker
    end
end

"""
Stop memory tracking
"""
function stop_memory_tracking(; device_id::Int = 0)
    HOST_TRACKER.enabled = false
    
    if haskey(DEVICE_TRACKERS, device_id)
        DEVICE_TRACKERS[device_id].enabled = false
    end
end

"""
Track a memory allocation
"""
function track_allocation(
    allocation_type::Symbol,
    size_bytes::Int64,
    operation::String;
    device_id::Int = 0
)
    bt = stacktrace()[3:end]  # Skip internal frames
    
    alloc = MemoryAllocation(
        now(),
        allocation_type,
        size_bytes,
        operation,
        bt[1:min(10, length(bt))]  # Keep only top 10 frames
    )
    
    if allocation_type == :host && HOST_TRACKER.enabled
        push!(HOST_TRACKER.allocations, alloc)
        current_memory = Base.gc_live_bytes() / 1024^2
        HOST_TRACKER.peak_memory = max(HOST_TRACKER.peak_memory, current_memory)
    elseif allocation_type == :device && haskey(DEVICE_TRACKERS, device_id)
        tracker = DEVICE_TRACKERS[device_id]
        if tracker.enabled
            push!(tracker.allocations, alloc)
            if CUDA.functional()
                current_memory = CUDA.memory_stats().live_bytes / 1024^2
                tracker.peak_memory = max(tracker.peak_memory, current_memory)
            end
        end
    end
end

"""
Profile memory usage of an operation
"""
function profile_memory(
    operation_name::String,
    operation::Function;
    device_id::Int = 0,
    sample_interval_ms::Int = 100
)::MemoryProfile
    
    # Get initial state
    start_snapshot = get_memory_snapshot()
    initial_gc_count = Base.gc_num().pause
    
    # Start tracking
    start_memory_tracking(device_id=device_id)
    
    # Collect snapshots during execution
    snapshots = MemorySnapshot[]
    monitoring = true
    
    # Start monitoring task
    monitor_task = @async begin
        while monitoring
            push!(snapshots, get_memory_snapshot())
            sleep(sample_interval_ms / 1000.0)
        end
    end
    
    # Run operation
    try
        result = operation()
    finally
        monitoring = false
        wait(monitor_task)
        stop_memory_tracking(device_id=device_id)
    end
    
    # Get final state
    end_snapshot = get_memory_snapshot()
    final_gc_count = Base.gc_num().pause
    
    # Calculate peaks
    peak_host_mb = maximum([s.host_used_mb for s in snapshots])
    peak_device_mb = maximum([s.device_used_mb for s in snapshots])
    
    # Calculate total allocations
    host_allocations = HOST_TRACKER.allocations
    total_host_allocated = sum(a.size_bytes for a in host_allocations) / 1024^2
    
    device_allocations = if haskey(DEVICE_TRACKERS, device_id)
        DEVICE_TRACKERS[device_id].allocations
    else
        MemoryAllocation[]
    end
    total_device_allocated = sum(a.size_bytes for a in device_allocations) / 1024^2
    
    # Combine allocations
    all_allocations = vcat(host_allocations, device_allocations)
    sort!(all_allocations, by=a->a.timestamp)
    
    return MemoryProfile(
        operation_name,
        start_snapshot,
        end_snapshot,
        peak_host_mb,
        peak_device_mb,
        all_allocations,
        snapshots,
        total_host_allocated,
        total_device_allocated,
        final_gc_count - initial_gc_count
    )
end

"""
Analyze memory allocations by operation
"""
function analyze_allocations(allocations::Vector{MemoryAllocation})
    operation_stats = Dict{String, Dict{Symbol, Any}}()
    
    for alloc in allocations
        op = alloc.operation
        if !haskey(operation_stats, op)
            operation_stats[op] = Dict(
                :count => 0,
                :total_mb => 0.0,
                :host_mb => 0.0,
                :device_mb => 0.0,
                :sizes => Float64[]
            )
        end
        
        stats = operation_stats[op]
        stats[:count] += 1
        size_mb = alloc.size_bytes / 1024^2
        stats[:total_mb] += size_mb
        push!(stats[:sizes], size_mb)
        
        if alloc.allocation_type == :host
            stats[:host_mb] += size_mb
        else
            stats[:device_mb] += size_mb
        end
    end
    
    # Calculate statistics
    for (op, stats) in operation_stats
        sizes = stats[:sizes]
        stats[:mean_mb] = mean(sizes)
        stats[:median_mb] = median(sizes)
        stats[:max_mb] = maximum(sizes)
        stats[:min_mb] = minimum(sizes)
        delete!(stats, :sizes)  # Remove raw data
    end
    
    return operation_stats
end

"""
Generate memory usage summary
"""
function memory_summary(profile::MemoryProfile; verbose::Bool = true)
    if verbose
        println("\nMemory Profile: $(profile.operation_name)")
        println("=" ^ 80)
        
        # Overall statistics
        println("\nOverall Statistics:")
        println("  Host Memory:")
        println("    Initial: $(round(profile.start_snapshot.host_used_mb, digits=2)) MB")
        println("    Final: $(round(profile.end_snapshot.host_used_mb, digits=2)) MB")
        println("    Peak: $(round(profile.peak_host_mb, digits=2)) MB")
        println("    Total Allocated: $(round(profile.total_allocated_host_mb, digits=2)) MB")
        
        if CUDA.functional()
            println("  Device Memory:")
            println("    Initial: $(round(profile.start_snapshot.device_used_mb, digits=2)) MB")
            println("    Final: $(round(profile.end_snapshot.device_used_mb, digits=2)) MB")
            println("    Peak: $(round(profile.peak_device_mb, digits=2)) MB")
            println("    Total Allocated: $(round(profile.total_allocated_device_mb, digits=2)) MB")
        end
        
        println("  GC Events: $(profile.gc_events)")
        
        # Allocation analysis
        if !isempty(profile.allocations)
            println("\nAllocation Analysis:")
            alloc_stats = analyze_allocations(profile.allocations)
            
            println("  By Operation:")
            for (op, stats) in alloc_stats
                println("    $op:")
                println("      Count: $(stats[:count])")
                println("      Total: $(round(stats[:total_mb], digits=2)) MB")
                println("      Mean: $(round(stats[:mean_mb], digits=2)) MB")
                println("      Max: $(round(stats[:max_mb], digits=2)) MB")
            end
        end
        
        # Memory growth
        if length(profile.snapshots) > 1
            host_growth = profile.end_snapshot.host_used_mb - profile.start_snapshot.host_used_mb
            device_growth = profile.end_snapshot.device_used_mb - profile.start_snapshot.device_used_mb
            
            println("\nMemory Growth:")
            println("  Host: $(round(host_growth, digits=2)) MB")
            if CUDA.functional()
                println("  Device: $(round(device_growth, digits=2)) MB")
            end
        end
    end
    
    # Return summary dict
    return Dict(
        "operation" => profile.operation_name,
        "host_peak_mb" => profile.peak_host_mb,
        "device_peak_mb" => profile.peak_device_mb,
        "host_allocated_mb" => profile.total_allocated_host_mb,
        "device_allocated_mb" => profile.total_allocated_device_mb,
        "gc_events" => profile.gc_events,
        "allocation_count" => length(profile.allocations)
    )
end

"""
Compare memory profiles
"""
function compare_profiles(profiles::Dict{String, MemoryProfile})
    println("\nMemory Profile Comparison")
    println("=" ^ 80)
    
    # Header
    println("Operation            | Host Peak | Dev Peak | Host Alloc | Dev Alloc | GC Events")
    println("-" * 80)
    
    # Results
    for (name, profile) in profiles
        @printf("%-19s | %9.1f | %8.1f | %10.1f | %9.1f | %9d\n",
            name,
            profile.peak_host_mb,
            profile.peak_device_mb,
            profile.total_allocated_host_mb,
            profile.total_allocated_device_mb,
            profile.gc_events
        )
    end
    
    println("-" * 80)
end

"""
Memory leak detection
"""
function detect_memory_leaks(profile::MemoryProfile; threshold_mb::Float64 = 10.0)
    host_leak = profile.end_snapshot.host_used_mb - profile.start_snapshot.host_used_mb
    device_leak = profile.end_snapshot.device_used_mb - profile.start_snapshot.device_used_mb
    
    leaks_detected = false
    
    if abs(host_leak) > threshold_mb
        println("⚠️  Potential host memory leak detected: $(round(host_leak, digits=2)) MB")
        leaks_detected = true
    end
    
    if abs(device_leak) > threshold_mb
        println("⚠️  Potential device memory leak detected: $(round(device_leak, digits=2)) MB")
        leaks_detected = true
    end
    
    return leaks_detected
end

end # module