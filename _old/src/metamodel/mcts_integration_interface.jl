"""
MCTS Integration Interface
Seamless API for MCTS to query metamodel predictions without GPU kernel interruption
maintaining sub-millisecond latency
"""
module MCTSIntegrationInterface

using CUDA
using Statistics

# Include required types from GPU kernels
include("../gpu/kernels/mcts_types.jl")
using .MCTSTypes: MAX_FEATURES, FEATURE_CHUNKS, MAX_NODES, NODE_ACTIVE, NODE_EXPANDED

"""
Configuration for metamodel evaluation (simplified version for interface)
"""
struct MetamodelConfig
    batch_size::Int32               # Max batch size for inference
    feature_dim::Int32              # Input feature dimension
    output_dim::Int32               # Output dimension (scores)
    max_queue_size::Int32           # Maximum evaluation queue size
    timeout_ms::Float32             # Timeout for batch collection
    cache_size::Int32               # Result cache size
    fallback_score::Float32         # Score to use on metamodel failure
    use_mixed_precision::Bool       # Use FP16 for inference
end

"""
Zero-copy interface configuration
"""
struct InterfaceConfig
    # Performance constraints
    max_latency_us::Float32         # Maximum allowed latency in microseconds
    batch_timeout_us::Float32       # Batch collection timeout
    max_concurrent_requests::Int32  # Maximum concurrent requests
    
    # Memory management
    shared_buffer_size::Int32       # Size of shared memory buffer
    request_ring_size::Int32        # Ring buffer size for requests
    result_ring_size::Int32         # Ring buffer size for results
    
    # Priority scheduling
    priority_levels::Int32          # Number of priority levels
    high_priority_threshold::Float32  # Score threshold for high priority
    
    # Callback configuration
    callback_buffer_size::Int32     # Callback buffer size
    max_callback_delay_us::Float32  # Maximum callback delay
end

"""
Default configuration for sub-millisecond performance
"""
function default_interface_config()
    InterfaceConfig(
        500.0f0,    # 500 microseconds max latency
        100.0f0,    # 100 microseconds batch timeout
        1000,       # 1000 concurrent requests
        4096,       # 4KB shared buffer
        2048,       # 2048 request ring buffer
        2048,       # 2048 result ring buffer
        4,          # 4 priority levels
        0.8f0,      # 80% score threshold for high priority
        512,        # 512 callback buffer
        50.0f0      # 50 microseconds max callback delay
    )
end

"""
Request entry in ring buffer
"""
struct MCTSRequest
    request_id::UInt64              # Unique request ID
    node_id::Int32                  # MCTS node identifier
    feature_mask_ptr::CuPtr{UInt64} # Pointer to feature mask
    priority::Int32                 # Priority level (0=highest)
    timestamp_us::UInt64            # Request timestamp in microseconds
    callback_ptr::CuPtr{Cvoid}      # Callback function pointer
    user_data::UInt64               # User data for callback
end

# Define zero for CUDA arrays
Base.zero(::Type{MCTSRequest}) = MCTSRequest(
    UInt64(0), Int32(0), CuPtr{UInt64}(0), Int32(0), 
    UInt64(0), CuPtr{Cvoid}(0), UInt64(0)
)

"""
Result entry in ring buffer
"""
struct MCTSResult
    request_id::UInt64              # Matching request ID
    prediction_score::Float32       # Metamodel prediction
    confidence::Float32             # Prediction confidence
    inference_time_us::Float32      # Actual inference time
    error_code::Int32               # Error code (0=success)
end

# Define zero for CUDA arrays
Base.zero(::Type{MCTSResult}) = MCTSResult(
    UInt64(0), Float32(0), Float32(0), Float32(0), Int32(0)
)

"""
Zero-copy ring buffer for requests
"""
mutable struct RequestRingBuffer
    buffer::CuArray{MCTSRequest, 1}     # Ring buffer storage
    head::CuArray{UInt32, 1}            # Head pointer (atomic)
    tail::CuArray{UInt32, 1}            # Tail pointer (atomic)
    count::CuArray{UInt32, 1}           # Current count (atomic)
    capacity::UInt32                    # Buffer capacity
    
    # Performance counters
    total_enqueued::CuArray{UInt64, 1}  # Total requests enqueued
    total_dropped::CuArray{UInt64, 1}   # Total requests dropped
    max_latency_us::CuArray{Float32, 1} # Maximum observed latency
end

"""
Zero-copy ring buffer for results
"""
mutable struct ResultRingBuffer
    buffer::CuArray{MCTSResult, 1}      # Ring buffer storage
    head::CuArray{UInt32, 1}            # Head pointer (atomic)
    tail::CuArray{UInt32, 1}            # Tail pointer (atomic)
    count::CuArray{UInt32, 1}           # Current count (atomic)
    capacity::UInt32                    # Buffer capacity
    
    # Callback tracking
    pending_callbacks::CuArray{UInt64, 1}  # Pending callback count
    completed_callbacks::CuArray{UInt64, 1} # Completed callback count
end

"""
Priority scheduler for requests
"""
mutable struct PriorityScheduler
    # Priority queues (one per level)
    priority_queues::Vector{CuArray{UInt32, 1}}  # Indices into request buffer
    queue_heads::CuArray{UInt32, 1}              # Head pointers for each queue
    queue_tails::CuArray{UInt32, 1}              # Tail pointers for each queue
    queue_counts::CuArray{UInt32, 1}             # Counts for each queue
    
    # Scheduling policy
    weights::CuArray{Float32, 1}                 # Weight for each priority level
    last_served::CuArray{UInt32, 1}              # Last served queue
    round_robin_counter::CuArray{UInt32, 1}      # Round-robin counter
end

"""
MCTS Integration Interface Manager
"""
mutable struct MCTSInterface
    config::InterfaceConfig
    metamodel_config::MetamodelConfig
    
    # Ring buffers
    request_ring::RequestRingBuffer
    result_ring::ResultRingBuffer
    
    # Priority scheduling
    scheduler::PriorityScheduler
    
    # Shared memory regions
    shared_features::CuArray{Float32, 2}         # Shared feature tensor
    shared_predictions::CuArray{Float32, 1}      # Shared prediction results
    
    # Synchronization
    request_semaphore::CuArray{UInt32, 1}        # Request availability semaphore
    result_semaphore::CuArray{UInt32, 1}         # Result availability semaphore
    processing_lock::CuArray{UInt32, 1}          # Processing lock
    
    # Performance monitoring
    total_requests::UInt64
    total_processed::UInt64
    avg_latency_us::Float32
    max_throughput_rps::Float32
    
    # Status
    is_active::Bool
    last_batch_time_us::UInt64
end

"""
Create request ring buffer
"""
function RequestRingBuffer(capacity::UInt32)
    RequestRingBuffer(
        CUDA.zeros(MCTSRequest, capacity),
        CUDA.zeros(UInt32, 1),
        CUDA.zeros(UInt32, 1),
        CUDA.zeros(UInt32, 1),
        capacity,
        CUDA.zeros(UInt64, 1),
        CUDA.zeros(UInt64, 1),
        CUDA.zeros(Float32, 1)
    )
end

"""
Create result ring buffer
"""
function ResultRingBuffer(capacity::UInt32)
    ResultRingBuffer(
        CUDA.zeros(MCTSResult, capacity),
        CUDA.zeros(UInt32, 1),
        CUDA.zeros(UInt32, 1),
        CUDA.zeros(UInt32, 1),
        capacity,
        CUDA.zeros(UInt64, 1),
        CUDA.zeros(UInt64, 1)
    )
end

"""
Create priority scheduler
"""
function PriorityScheduler(num_levels::Int32, queue_size::UInt32)
    priority_queues = [CUDA.zeros(UInt32, queue_size) for _ in 1:num_levels]
    
    PriorityScheduler(
        priority_queues,
        CUDA.zeros(UInt32, num_levels),
        CUDA.zeros(UInt32, num_levels),
        CUDA.zeros(UInt32, num_levels),
        CUDA.ones(Float32, num_levels) ./ Float32(num_levels),  # Equal weights initially
        CUDA.zeros(UInt32, 1),
        CUDA.zeros(UInt32, 1)
    )
end

"""
Create MCTS Integration Interface
"""
function MCTSInterface(
    interface_config::InterfaceConfig,
    metamodel_config::MetamodelConfig
)
    
    request_ring = RequestRingBuffer(UInt32(interface_config.request_ring_size))
    result_ring = ResultRingBuffer(UInt32(interface_config.result_ring_size))
    
    scheduler = PriorityScheduler(
        interface_config.priority_levels,
        UInt32(interface_config.request_ring_size ÷ interface_config.priority_levels)
    )
    
    shared_features = CUDA.zeros(Float32, 
        metamodel_config.feature_dim, 
        metamodel_config.batch_size)
    shared_predictions = CUDA.zeros(Float32, metamodel_config.batch_size)
    
    MCTSInterface(
        interface_config,
        metamodel_config,
        request_ring,
        result_ring,
        scheduler,
        shared_features,
        shared_predictions,
        CUDA.zeros(UInt32, 1),
        CUDA.zeros(UInt32, 1),
        CUDA.zeros(UInt32, 1),
        0,
        0,
        0.0f0,
        0.0f0,
        true,
        0
    )
end

"""
Atomic enqueue operation kernel
"""
function enqueue_request_kernel!(
    request_buffer::CuDeviceArray{MCTSRequest, 1},
    tail_ptr::CuDeviceArray{UInt32, 1},
    count_ptr::CuDeviceArray{UInt32, 1},
    capacity::UInt32,
    request::MCTSRequest,
    success_flag::CuDeviceArray{Bool, 1}
)
    if threadIdx().x == 1
        # Atomic increment of tail
        old_tail = CUDA.atomic_add!(pointer(tail_ptr), UInt32(1))
        
        # Check if buffer is full
        if old_tail < capacity
            # Store request
            request_buffer[old_tail + 1] = request
            
            # Increment count
            CUDA.atomic_add!(pointer(count_ptr), UInt32(1))
            success_flag[1] = true
        else
            # Buffer full, rollback tail
            CUDA.atomic_sub!(pointer(tail_ptr), UInt32(1))
            success_flag[1] = false
        end
    end
    
    return nothing
end

"""
Zero-copy request submission
"""
function submit_request(
    interface::MCTSInterface,
    node_id::Int32,
    feature_mask_ptr::CuPtr{UInt64},
    priority::Int32 = Int32(0),
    callback_ptr::CuPtr{Cvoid} = CuPtr{Cvoid}(0),
    user_data::UInt64 = UInt64(0)
)::UInt64
    # Generate unique request ID
    request_id = UInt64(time_ns())
    
    # Create request
    request = MCTSRequest(
        request_id,
        node_id,
        feature_mask_ptr,
        priority,
        UInt64(time_ns() ÷ 1000),  # Convert to microseconds
        callback_ptr,
        user_data
    )
    
    # Enqueue request
    success_flag = CUDA.zeros(Bool, 1)
    @cuda threads=1 enqueue_request_kernel!(
        interface.request_ring.buffer,
        interface.request_ring.tail,
        interface.request_ring.count,
        interface.request_ring.capacity,
        request,
        success_flag
    )
    
    # Check success
    success = CUDA.@allowscalar success_flag[1]
    if success
        interface.total_requests += UInt64(1)
        return request_id
    else
        # Buffer full, request dropped
        CUDA.@allowscalar interface.request_ring.total_dropped[1] += 1
        return UInt64(0)  # Invalid request ID
    end
end

"""
Dequeue result kernel
"""
function dequeue_result_kernel!(
    result_buffer::CuDeviceArray{MCTSResult, 1},
    head_ptr::CuDeviceArray{UInt32, 1},
    count_ptr::CuDeviceArray{UInt32, 1},
    capacity::UInt32,
    result_out::CuDeviceArray{MCTSResult, 1},
    success_flag::CuDeviceArray{Bool, 1}
)
    if threadIdx().x == 1
        current_count = count_ptr[1]
        
        if current_count > 0
            # Get current head
            head_pos = head_ptr[1]
            
            # Copy result
            result_out[1] = result_buffer[head_pos + 1]
            
            # Atomic increment head
            CUDA.atomic_add!(pointer(head_ptr), UInt32(1))
            if head_ptr[1] >= capacity
                head_ptr[1] = UInt32(0)  # Wrap around
            end
            
            # Decrement count
            CUDA.atomic_sub!(pointer(count_ptr), UInt32(1))
            success_flag[1] = true
        else
            success_flag[1] = false
        end
    end
    
    return nothing
end

"""
Poll for result
"""
function poll_result(interface::MCTSInterface, request_id::UInt64)::Union{MCTSResult, Nothing}
    result_buffer = CUDA.zeros(MCTSResult, 1)
    success_flag = CUDA.zeros(Bool, 1)
    
    @cuda threads=1 dequeue_result_kernel!(
        interface.result_ring.buffer,
        interface.result_ring.head,
        interface.result_ring.count,
        interface.result_ring.capacity,
        result_buffer,
        success_flag
    )
    
    success = CUDA.@allowscalar success_flag[1]
    if success
        result = CUDA.@allowscalar result_buffer[1]
        if result.request_id == request_id
            return result
        else
            # Put back result (different request ID)
            # This is simplified - proper implementation would need better matching
            return nothing
        end
    else
        return nothing
    end
end

"""
Batch processing kernel for high throughput
"""
function process_batch_kernel!(
    request_buffer::CuDeviceArray{MCTSRequest, 1},
    result_buffer::CuDeviceArray{MCTSResult, 1},
    shared_features::CuDeviceArray{Float32, 2},
    shared_predictions::CuDeviceArray{Float32, 1},
    batch_indices::CuDeviceArray{UInt32, 1},
    batch_size::Int32,
    feature_dim::Int32
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    # Prepare features for this batch
    if tid <= batch_size
        request_idx = batch_indices[tid]
        request = request_buffer[request_idx]
        
        # Copy features from feature mask to shared buffer
        # This is a simplified version - real implementation would 
        # decode the feature mask properly
        for feat_idx in 1:feature_dim
            shared_features[feat_idx, tid] = Float32(feat_idx % 2)  # Placeholder
        end
    end
    
    return nothing
end

"""
High-throughput batch processing
"""
function process_pending_requests!(interface::MCTSInterface)
    start_time = time_ns()
    
    # Check if there are pending requests
    current_count = CUDA.@allowscalar interface.request_ring.count[1]
    if current_count == 0
        return 0
    end
    
    # Collect batch
    batch_size = min(current_count, interface.metamodel_config.batch_size)
    batch_indices = CUDA.zeros(UInt32, batch_size)
    
    # Simple batch collection (could be optimized with priority scheduling)
    CUDA.@allowscalar begin
        head_pos = interface.request_ring.head[1]
        for i in 1:batch_size
            batch_indices[i] = (head_pos + i - 1) % interface.request_ring.capacity
        end
    end
    
    # Process batch using metamodel
    try
        # This would call the actual metamodel inference
        # For now, we'll use a placeholder
        predictions = fill(0.5f0, batch_size)  # Placeholder predictions
        
        # Create results
        CUDA.@allowscalar begin
            for i in 1:batch_size
                request_idx = batch_indices[i]
                request = interface.request_ring.buffer[request_idx + 1]
                
                result = MCTSResult(
                    request.request_id,
                    predictions[i],
                    0.8f0,  # Placeholder confidence
                    Float32((time_ns() - start_time) ÷ 1000),  # Latency in microseconds
                    0  # Success
                )
                
                # Enqueue result (simplified)
                result_tail = interface.result_ring.tail[1]
                if result_tail < interface.result_ring.capacity
                    interface.result_ring.buffer[result_tail + 1] = result
                    interface.result_ring.tail[1] += 1
                    interface.result_ring.count[1] += 1
                end
            end
            
            # Update request ring
            interface.request_ring.head[1] = 
                (interface.request_ring.head[1] + batch_size) % interface.request_ring.capacity
            interface.request_ring.count[1] -= batch_size
        end
        
        interface.total_processed += UInt64(batch_size)
        
        # Update performance metrics
        processing_time_us = Float32((time_ns() - start_time) ÷ 1000)
        interface.avg_latency_us = 
            0.9f0 * interface.avg_latency_us + 0.1f0 * processing_time_us
        
        return batch_size
        
    catch e
        @warn "Batch processing failed" exception=e
        return 0
    end
end

"""
Asynchronous processing loop
"""
function async_processing_loop(interface::MCTSInterface)
    while interface.is_active
        processed = process_pending_requests!(interface)
        
        if processed == 0
            # No work, brief sleep to avoid busy waiting
            sleep(0.001)  # 1ms sleep
        end
        
        # Update last batch time
        interface.last_batch_time_us = UInt64(time_ns() ÷ 1000)
    end
end

"""
Get interface statistics
"""
function get_interface_statistics(interface::MCTSInterface)
    stats = Dict{String, Any}()
    
    CUDA.@allowscalar begin
        # Request statistics
        stats["total_requests"] = interface.total_requests
        stats["total_processed"] = interface.total_processed
        stats["pending_requests"] = interface.request_ring.count[1]
        stats["dropped_requests"] = interface.request_ring.total_dropped[1]
        
        # Performance metrics
        stats["avg_latency_us"] = interface.avg_latency_us
        stats["max_latency_us"] = interface.request_ring.max_latency_us[1]
        stats["throughput_rps"] = interface.max_throughput_rps
        
        # Buffer utilization
        stats["request_buffer_utilization"] = 
            Float32(interface.request_ring.count[1]) / Float32(interface.request_ring.capacity)
        stats["result_buffer_utilization"] = 
            Float32(interface.result_ring.count[1]) / Float32(interface.result_ring.capacity)
        
        # Status
        stats["is_active"] = interface.is_active
        stats["last_batch_time_us"] = interface.last_batch_time_us
    end
    
    return stats
end

"""
Shutdown interface gracefully
"""
function shutdown!(interface::MCTSInterface)
    interface.is_active = false
    
    # Process remaining requests
    while CUDA.@allowscalar(interface.request_ring.count[1]) > 0
        process_pending_requests!(interface)
        sleep(0.001)
    end
    
    @info "MCTS Integration Interface shutdown complete"
end

# Export main types and functions
export InterfaceConfig, MCTSInterface, MCTSRequest, MCTSResult
export RequestRingBuffer, ResultRingBuffer, PriorityScheduler
export MetamodelConfig
export default_interface_config
export submit_request, poll_result, process_pending_requests!
export async_processing_loop, get_interface_statistics, shutdown!

end # module MCTSIntegrationInterface