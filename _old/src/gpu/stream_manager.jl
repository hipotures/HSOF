# CUDA Stream Manager Module
module StreamManager

using CUDA
using Logging

# Stream pool for each device
struct StreamPool
    device_id::Int
    streams::Vector{CuStream}
    default_stream::CuStream
end

# Global stream pools
const STREAM_POOLS = Dict{Int, StreamPool}()

"""
    initialize_streams(device_id::Int, count::Int = 4)

Initialize CUDA streams for a device.
"""
function initialize_streams(device_id::Int, count::Int = 4)
    CUDA.device!(device_id)
    
    streams = CuStream[]
    for i in 1:count
        push!(streams, CuStream())
    end
    
    STREAM_POOLS[device_id] = StreamPool(
        device_id,
        streams,
        CUDA.default_stream()
    )
    
    @info "Initialized $count streams for GPU $device_id"
    return STREAM_POOLS[device_id]
end

"""
    get_stream(device_id::Int, stream_id::Int)

Get a specific stream for a device.
"""
function get_stream(device_id::Int, stream_id::Int)
    if !haskey(STREAM_POOLS, device_id)
        error("No streams initialized for GPU $device_id")
    end
    
    pool = STREAM_POOLS[device_id]
    if stream_id < 1 || stream_id > length(pool.streams)
        error("Invalid stream ID $stream_id for GPU $device_id")
    end
    
    return pool.streams[stream_id]
end

"""
    get_stream_count(device_id::Int)

Get the number of streams for a device.
"""
function get_stream_count(device_id::Int)
    if haskey(STREAM_POOLS, device_id)
        return length(STREAM_POOLS[device_id].streams)
    else
        return 0
    end
end

"""
    with_stream(f::Function, device_id::Int, stream_id::Int)

Execute a function in the context of a specific stream.
"""
function with_stream(f::Function, device_id::Int, stream_id::Int)
    stream = get_stream(device_id, stream_id)
    CUDA.device!(device_id)
    
    # Execute in stream context
    CUDA.stream!(stream) do
        f()
    end
end

"""
    synchronize_stream(device_id::Int, stream_id::Int)

Synchronize a specific stream.
"""
function synchronize_stream(device_id::Int, stream_id::Int)
    stream = get_stream(device_id, stream_id)
    CUDA.synchronize(stream)
end

"""
    synchronize_all_streams(device_id::Int)

Synchronize all streams for a device.
"""
function synchronize_all_streams(device_id::Int)
    if haskey(STREAM_POOLS, device_id)
        pool = STREAM_POOLS[device_id]
        for stream in pool.streams
            CUDA.synchronize(stream)
        end
    end
end

"""
    cleanup_streams(device_id::Int)

Cleanup streams for a device.
"""
function cleanup_streams(device_id::Int)
    if haskey(STREAM_POOLS, device_id)
        synchronize_all_streams(device_id)
        delete!(STREAM_POOLS, device_id)
        @info "Cleaned up streams for GPU $device_id"
    end
end

"""
    cleanup_all()

Cleanup all stream pools.
"""
function cleanup_all()
    @info "Cleaning up all CUDA streams..."
    
    for device_id in keys(STREAM_POOLS)
        cleanup_streams(device_id)
    end
    
    @info "All streams cleaned up"
end

"""
    launch_kernel_on_stream(kernel::Function, args, config; 
                           device_id::Int, stream_id::Int)

Launch a CUDA kernel on a specific stream.
"""
function launch_kernel_on_stream(kernel::Function, args, config; 
                                device_id::Int, stream_id::Int)
    with_stream(device_id, stream_id) do
        # Unpack config tuple
        threads = config[1]
        blocks = config[2]
        @cuda threads=threads blocks=blocks kernel(args...)
    end
end

"""
    parallel_streams_execute(f::Function, device_id::Int; sync::Bool = true)

Execute function across all streams for parallel work.
"""
function parallel_streams_execute(f::Function, device_id::Int; sync::Bool = true)
    if !haskey(STREAM_POOLS, device_id)
        error("No streams initialized for GPU $device_id")
    end
    
    pool = STREAM_POOLS[device_id]
    
    # Launch work on each stream
    for (i, stream) in enumerate(pool.streams)
        CUDA.stream!(stream) do
            f(i, stream)
        end
    end
    
    # Synchronize if requested
    if sync
        synchronize_all_streams(device_id)
    end
end

end # module