module MemoryProfiling

using CUDA
using Printf

export MemoryTracker, MemorySnapshot, BandwidthMeasurement
export create_memory_tracker, track_allocation!, track_deallocation!
export track_transfer!, get_memory_snapshot, calculate_bandwidth
export get_bandwidth_stats, reset_tracker!

"""
Memory allocation record
"""
struct AllocationRecord
    ptr::Ptr{Nothing}
    size::Int
    timestamp::Float64
    operation::Symbol  # :alloc, :free
end

"""
Memory transfer record
"""
struct TransferRecord
    size::Int
    direction::Symbol  # :h2d, :d2h, :d2d
    elapsed_ms::Float32
    timestamp::Float64
end

"""
Memory snapshot at a point in time
"""
struct MemorySnapshot
    timestamp::Float64
    allocated_bytes::Int
    free_bytes::Int
    total_bytes::Int
    peak_allocated::Int
    allocation_count::Int
    deallocation_count::Int
end

"""
Bandwidth measurement result
"""
struct BandwidthMeasurement
    direction::Symbol
    total_bytes::Int
    total_time_ms::Float32
    bandwidth_gbps::Float32
    transfer_count::Int
    avg_transfer_size::Float32
end

"""
Memory usage and bandwidth tracker
"""
mutable struct MemoryTracker
    allocations::Dict{Ptr{Nothing}, Int}
    allocation_history::Vector{AllocationRecord}
    transfer_history::Vector{TransferRecord}
    current_allocated::Int
    peak_allocated::Int
    total_allocated::Int
    total_deallocated::Int
    start_time::Float64
    enabled::Bool
end

"""
Create a new memory tracker
"""
function create_memory_tracker(; enabled::Bool = true)
    return MemoryTracker(
        Dict{Ptr{Nothing}, Int}(),
        AllocationRecord[],
        TransferRecord[],
        0,
        0,
        0,
        0,
        time(),
        enabled
    )
end

"""
Track memory allocation
"""
function track_allocation!(tracker::MemoryTracker, ptr::Ptr{Nothing}, size::Int)
    if !tracker.enabled
        return
    end
    
    tracker.allocations[ptr] = size
    tracker.current_allocated += size
    tracker.total_allocated += size
    
    if tracker.current_allocated > tracker.peak_allocated
        tracker.peak_allocated = tracker.current_allocated
    end
    
    # Record in history
    push!(tracker.allocation_history, AllocationRecord(
        ptr, size, time() - tracker.start_time, :alloc
    ))
end

"""
Track memory deallocation
"""
function track_deallocation!(tracker::MemoryTracker, ptr::Ptr{Nothing})
    if !tracker.enabled || !haskey(tracker.allocations, ptr)
        return
    end
    
    size = tracker.allocations[ptr]
    delete!(tracker.allocations, ptr)
    tracker.current_allocated -= size
    tracker.total_deallocated += size
    
    # Record in history
    push!(tracker.allocation_history, AllocationRecord(
        ptr, size, time() - tracker.start_time, :free
    ))
end

"""
Track memory transfer
"""
function track_transfer!(
    tracker::MemoryTracker,
    size::Int,
    direction::Symbol,
    elapsed_ms::Float32
)
    if !tracker.enabled
        return
    end
    
    push!(tracker.transfer_history, TransferRecord(
        size, direction, elapsed_ms, time() - tracker.start_time
    ))
end

"""
Get current memory snapshot
"""
function get_memory_snapshot(tracker::MemoryTracker)
    mem_info = CUDA.memory_status()
    
    return MemorySnapshot(
        time() - tracker.start_time,
        tracker.current_allocated,
        mem_info.free,
        mem_info.total,
        tracker.peak_allocated,
        count(r -> r.operation == :alloc, tracker.allocation_history),
        count(r -> r.operation == :free, tracker.allocation_history)
    )
end

"""
Calculate bandwidth from transfer records
"""
function calculate_bandwidth(
    size_bytes::Int,
    elapsed_ms::Float32
)::Float32
    if elapsed_ms <= 0
        return 0.0f0
    end
    
    # Convert to GB/s
    return (size_bytes / 1e9) / (elapsed_ms / 1000)
end

"""
Get bandwidth statistics by direction
"""
function get_bandwidth_stats(tracker::MemoryTracker)
    stats = Dict{Symbol, BandwidthMeasurement}()
    
    for direction in [:h2d, :d2h, :d2d]
        transfers = filter(t -> t.direction == direction, tracker.transfer_history)
        
        if !isempty(transfers)
            total_bytes = sum(t.size for t in transfers)
            total_time = sum(t.elapsed_ms for t in transfers)
            
            stats[direction] = BandwidthMeasurement(
                direction,
                total_bytes,
                total_time,
                calculate_bandwidth(total_bytes, total_time),
                length(transfers),
                total_bytes / length(transfers)
            )
        end
    end
    
    return stats
end

"""
Reset tracker state
"""
function reset_tracker!(tracker::MemoryTracker)
    empty!(tracker.allocations)
    empty!(tracker.allocation_history)
    empty!(tracker.transfer_history)
    tracker.current_allocated = 0
    tracker.peak_allocated = 0
    tracker.total_allocated = 0
    tracker.total_deallocated = 0
    tracker.start_time = time()
end

"""
Pretty print memory snapshot
"""
function Base.show(io::IO, snapshot::MemorySnapshot)
    println(io, "Memory Snapshot @ $(round(snapshot.timestamp, digits=2))s:")
    println(io, "  Currently allocated: $(format_bytes(snapshot.allocated_bytes))")
    println(io, "  Free memory: $(format_bytes(snapshot.free_bytes))")
    println(io, "  Total memory: $(format_bytes(snapshot.total_bytes))")
    println(io, "  Peak allocated: $(format_bytes(snapshot.peak_allocated))")
    println(io, "  Allocations: $(snapshot.allocation_count)")
    println(io, "  Deallocations: $(snapshot.deallocation_count)")
end

"""
Pretty print bandwidth measurement
"""
function Base.show(io::IO, bw::BandwidthMeasurement)
    direction_str = Dict(:h2d => "Host→Device", :d2h => "Device→Host", :d2d => "Device→Device")[bw.direction]
    println(io, "$(direction_str):")
    println(io, "  Total: $(format_bytes(bw.total_bytes)) in $(round(bw.total_time_ms, digits=2)) ms")
    println(io, "  Bandwidth: $(round(bw.bandwidth_gbps, digits=2)) GB/s")
    println(io, "  Transfers: $(bw.transfer_count)")
    println(io, "  Avg size: $(format_bytes(Int(round(bw.avg_transfer_size))))")
end

"""
Format bytes to human-readable string
"""
function format_bytes(bytes::Int)
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 1
    value = Float64(bytes)
    
    while value >= 1024 && unit_idx < length(units)
        value /= 1024
        unit_idx += 1
    end
    
    return @sprintf("%.2f %s", value, units[unit_idx])
end

"""
Memory tracking hooks for CuArray
"""
function track_cuarray_allocation(tracker::MemoryTracker, arr::CuArray)
    ptr = pointer(arr)
    size = sizeof(arr)
    track_allocation!(tracker, ptr, size)
end

function track_cuarray_deallocation(tracker::MemoryTracker, arr::CuArray)
    ptr = pointer(arr)
    track_deallocation!(tracker, ptr)
end

export track_cuarray_allocation, track_cuarray_deallocation

end # module