# GPU Memory Manager Module
module MemoryManager

using CUDA
using Logging

# Memory pool information per device
mutable struct MemoryPool
    device_id::Int
    limit_gb::Float64
    allocated::Int
    peak_allocated::Int
    allocations::Dict{Ptr{Nothing}, Int}  # Track allocations
end

# Global memory pools
const MEMORY_POOLS = Dict{Int, MemoryPool}()

"""
    initialize()

Initialize memory manager for all available GPUs.
"""
function initialize()
    @info "Initializing Memory Manager..."
    
    # Create memory pool for each device
    for i in 0:length(CUDA.devices())-1
        MEMORY_POOLS[i] = MemoryPool(
            i,
            22.0,  # Default 22GB limit
            0,
            0,
            Dict{Ptr{Nothing}, Int}()
        )
    end
    
    @info "Memory Manager initialized for $(length(MEMORY_POOLS)) device(s)"
end

"""
    set_memory_limit(limit_gb::Float64, device_id::Int = CUDA.device())

Set memory limit for a specific device.
"""
function set_memory_limit(limit_gb::Float64, device_id::Int = CUDA.device())
    if haskey(MEMORY_POOLS, device_id)
        MEMORY_POOLS[device_id].limit_gb = limit_gb
        @info "Set memory limit for GPU $device_id to $limit_gb GB"
    else
        @warn "GPU $device_id not found in memory pools"
    end
end

"""
    allocate(T::Type, dims...; device_id::Int = -1)

Allocate GPU memory with tracking.
"""
function allocate(T::Type, dims...; device_id::Int = -1)
    # Get current device if not specified
    if device_id == -1
        device_id = CUDA.device()
    end
    
    # Calculate size
    size_bytes = prod(dims) * sizeof(T)
    
    # Check against limit
    pool = get(MEMORY_POOLS, device_id, nothing)
    if pool !== nothing
        limit_bytes = pool.limit_gb * 1024^3
        if pool.allocated + size_bytes > limit_bytes
            error("Memory allocation would exceed limit: $(pool.allocated + size_bytes) > $limit_bytes")
        end
    end
    
    # Allocate memory
    CUDA.device!(device_id)
    arr = CUDA.zeros(T, dims...)
    
    # Track allocation
    if pool !== nothing
        ptr = pointer(arr)
        pool.allocations[ptr] = size_bytes
        pool.allocated += size_bytes
        pool.peak_allocated = max(pool.peak_allocated, pool.allocated)
    end
    
    return arr
end

"""
    free(arr::CuArray)

Free GPU memory with tracking.
"""
function free(arr::CuArray)
    device_id = CUDA.device()
    pool = get(MEMORY_POOLS, device_id, nothing)
    
    if pool !== nothing
        ptr = pointer(arr)
        if haskey(pool.allocations, ptr)
            size_bytes = pool.allocations[ptr]
            delete!(pool.allocations, ptr)
            pool.allocated -= size_bytes
        end
    end
    
    # CUDA.jl handles actual deallocation via GC
    finalize(arr)
end

"""
    get_memory_stats(device_id::Int = -1)

Get memory statistics for a device.
"""
function get_memory_stats(device_id::Int = -1)
    # Get current device if not specified
    if device_id == -1
        device_id = CUDA.device()
    end
    CUDA.device!(device_id)
    
    total = CUDA.totalmem(CUDA.device())
    free = CUDA.available_memory()
    used = total - free
    
    stats = Dict{String, Any}(
        "total_gb" => total / 1024^3,
        "free_gb" => free / 1024^3,
        "used_gb" => used / 1024^3,
        "usage_percent" => 100 * used / total
    )
    
    # Add pool tracking info if available
    pool = get(MEMORY_POOLS, device_id, nothing)
    if pool !== nothing
        stats["tracked_gb"] = pool.allocated / 1024^3
        stats["peak_tracked_gb"] = pool.peak_allocated / 1024^3
        stats["limit_gb"] = pool.limit_gb
        stats["allocations_count"] = length(pool.allocations)
    end
    
    return stats
end

"""
    get_pool_info()

Get information about all memory pools.
"""
function get_pool_info()
    info = Dict{Int, Dict{String, Any}}()
    
    for (device_id, pool) in MEMORY_POOLS
        info[device_id] = Dict(
            "limit_gb" => pool.limit_gb,
            "allocated_gb" => pool.allocated / 1024^3,
            "peak_allocated_gb" => pool.peak_allocated / 1024^3,
            "allocations" => length(pool.allocations)
        )
    end
    
    return info
end

"""
    cleanup()

Cleanup memory manager resources.
"""
function cleanup()
    @info "Cleaning up Memory Manager..."
    
    # Clear all tracked allocations
    for pool in values(MEMORY_POOLS)
        empty!(pool.allocations)
        pool.allocated = 0
    end
    
    # Force garbage collection
    GC.gc()
    CUDA.reclaim()
    
    @info "Memory Manager cleaned up"
end

"""
    monitor_memory_usage(callback::Function, interval::Float64 = 1.0)

Monitor memory usage periodically.
"""
function monitor_memory_usage(callback::Function, interval::Float64 = 1.0)
    @async begin
        while true
            for device_id in keys(MEMORY_POOLS)
                stats = get_memory_stats(device_id)
                callback(device_id, stats)
            end
            sleep(interval)
        end
    end
end

end # module