module MetamodelIntegration

using CUDA
using Statistics

# Include required types
include("mcts_types.jl")
using .MCTSTypes: MAX_FEATURES, FEATURE_CHUNKS, MAX_NODES, NODE_ACTIVE, NODE_EXPANDED

"""
Configuration for metamodel evaluation
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
Evaluation request for metamodel
"""
struct EvalRequest
    node_idx::Int32                 # Node index in tree
    request_id::Int32               # Unique request ID
    timestamp::Float32              # Request timestamp
    priority::Int32                 # Priority level
end

"""
Evaluation result from metamodel
"""
struct EvalResult
    node_idx::Int32                 # Node index in tree
    score::Float32                  # Evaluation score
    confidence::Float32             # Confidence level
    inference_time_ms::Float32      # Inference time
end

"""
Asynchronous evaluation queue for metamodel requests
"""
mutable struct EvalQueue
    requests::CuArray{EvalRequest, 1}      # Queue of pending requests
    queue_head::CuArray{Int32, 1}          # Head pointer
    queue_tail::CuArray{Int32, 1}          # Tail pointer
    queue_size::CuArray{Int32, 1}          # Current queue size
    
    # Batch staging area
    batch_indices::CuArray{Int32, 1}       # Node indices for current batch
    batch_features::CuArray{Float32, 2}    # Prepared feature tensors
    batch_ready::CuArray{Bool, 1}          # Batch ready flag
    batch_size::CuArray{Int32, 1}          # Current batch size
    
    # Result storage
    result_scores::CuArray{Float32, 1}     # Evaluation scores
    result_ready::CuArray{Bool, 1}         # Result ready flags
    
    # Statistics
    total_requests::CuArray{Int64, 1}      # Total requests processed
    total_batches::CuArray{Int64, 1}       # Total batches processed
    avg_batch_size::CuArray{Float32, 1}    # Average batch size
    avg_wait_time_ms::CuArray{Float32, 1}  # Average wait time
end

"""
Create a new evaluation queue
"""
function EvalQueue(config::MetamodelConfig)
    max_size = config.max_queue_size
    batch_size = config.batch_size
    feature_dim = config.feature_dim
    
    EvalQueue(
        CUDA.fill(EvalRequest(0, 0, 0.0f0, 0), max_size),
        CUDA.ones(Int32, 1),   # queue_head starts at 1
        CUDA.ones(Int32, 1),   # queue_tail starts at 1
        CUDA.zeros(Int32, 1),
        CUDA.zeros(Int32, batch_size),
        CUDA.zeros(Float32, feature_dim, batch_size),
        CUDA.zeros(Bool, 1),
        CUDA.zeros(Int32, 1),
        CUDA.zeros(Float32, MAX_NODES),
        CUDA.zeros(Bool, MAX_NODES),
        CUDA.zeros(Int64, 1),
        CUDA.zeros(Int64, 1),
        CUDA.zeros(Float32, 1),
        CUDA.zeros(Float32, 1)
    )
end

"""
Result cache for metamodel evaluations
"""
mutable struct ResultCache
    cache_keys::CuArray{UInt64, 1}         # Feature hash keys
    cache_scores::CuArray{Float32, 1}      # Cached scores
    cache_hits::CuArray{Int32, 1}          # Hit counts
    cache_timestamps::CuArray{Float32, 1}  # Timestamps
    cache_valid::CuArray{Bool, 1}          # Valid flags
    
    # Statistics
    total_hits::CuArray{Int64, 1}          # Total cache hits
    total_misses::CuArray{Int64, 1}        # Total cache misses
    hit_rate::CuArray{Float32, 1}          # Hit rate
end

"""
Create a new result cache
"""
function ResultCache(cache_size::Int32)
    ResultCache(
        CUDA.zeros(UInt64, cache_size),
        CUDA.zeros(Float32, cache_size),
        CUDA.zeros(Int32, cache_size),
        CUDA.zeros(Float32, cache_size),
        CUDA.zeros(Bool, cache_size),
        CUDA.zeros(Int64, 1),
        CUDA.zeros(Int64, 1),
        CUDA.zeros(Float32, 1)
    )
end

"""
Metamodel evaluation manager
"""
mutable struct MetamodelManager
    config::MetamodelConfig
    eval_queue::EvalQueue
    result_cache::ResultCache
    
    # Tensor preparation buffers
    feature_buffer::CuArray{Float32, 2}    # Reusable feature buffer
    output_buffer::CuArray{Float32, 2}     # Reusable output buffer
    
    # Mixed precision buffers (if enabled)
    feature_buffer_fp16::Union{Nothing, CuArray{Float16, 2}}
    output_buffer_fp16::Union{Nothing, CuArray{Float16, 2}}
    
    # Status
    is_active::Bool
    last_inference_time::Float32
    total_inferences::Int64
end

"""
Create a new metamodel manager
"""
function MetamodelManager(config::MetamodelConfig)
    eval_queue = EvalQueue(config)
    result_cache = ResultCache(config.cache_size)
    
    feature_buffer = CUDA.zeros(Float32, config.feature_dim, config.batch_size)
    output_buffer = CUDA.zeros(Float32, config.output_dim, config.batch_size)
    
    # Create mixed precision buffers if enabled
    feature_buffer_fp16 = config.use_mixed_precision ? 
        CUDA.zeros(Float16, config.feature_dim, config.batch_size) : nothing
    output_buffer_fp16 = config.use_mixed_precision ? 
        CUDA.zeros(Float16, config.output_dim, config.batch_size) : nothing
    
    MetamodelManager(
        config,
        eval_queue,
        result_cache,
        feature_buffer,
        output_buffer,
        feature_buffer_fp16,
        output_buffer_fp16,
        true,
        0.0f0,
        0
    )
end

"""
Enqueue evaluation request kernel
"""
function enqueue_request_kernel!(
    queue_requests::CuDeviceArray{EvalRequest, 1},
    queue_tail::CuDeviceArray{Int32, 1},
    queue_size::CuDeviceArray{Int32, 1},
    node_idx::Int32,
    request_id::Int32,
    timestamp::Float32,
    priority::Int32,
    max_size::Int32
)
    tid = threadIdx().x
    
    if tid == 1
        current_size = queue_size[1]
        
        if current_size < max_size
            # Get current tail position
            tail_pos = queue_tail[1]
            
            # Create and store request
            request = EvalRequest(node_idx, request_id, timestamp, priority)
            queue_requests[tail_pos] = request
            
            # Update tail and size
            queue_tail[1] = (tail_pos % max_size) + 1
            CUDA.atomic_add!(pointer(queue_size), Int32(1))
        end
    end
    
    return nothing
end

"""
Prepare feature tensor kernel
"""
function prepare_features_kernel!(
    feature_buffer::CuDeviceArray{Float32, 2},
    feature_masks::CuDeviceArray{UInt64, 2},
    node_indices::CuDeviceArray{Int32, 1},
    batch_size::Int32,
    feature_dim::Int32
)
    tid = threadIdx().x
    bid = blockIdx().x
    
    if bid <= batch_size
        node_idx = node_indices[bid]
        
        # Each thread handles a subset of features
        features_per_thread = cld(feature_dim, blockDim().x)
        start_feature = (tid - 1) * features_per_thread + 1
        end_feature = min(tid * features_per_thread, feature_dim)
        
        for feat_idx in start_feature:end_feature
            if feat_idx <= feature_dim
                # Check if feature is set
                chunk_idx = div(feat_idx - 1, 64) + 1
                bit_idx = mod(feat_idx - 1, 64)
                
                if chunk_idx <= FEATURE_CHUNKS
                    mask = feature_masks[chunk_idx, node_idx]
                    is_set = (mask >> bit_idx) & UInt64(1)
                    
                    # Convert binary feature to float
                    feature_buffer[feat_idx, bid] = Float32(is_set)
                end
            end
        end
    end
    
    return nothing
end

"""
Cache lookup kernel
"""
function cache_lookup_kernel!(
    cache_keys::CuDeviceArray{UInt64, 1},
    cache_scores::CuDeviceArray{Float32, 1},
    cache_valid::CuDeviceArray{Bool, 1},
    cache_hits::CuDeviceArray{Int32, 1},
    result_scores::CuDeviceArray{Float32, 1},
    result_ready::CuDeviceArray{Bool, 1},
    feature_hash::UInt64,
    node_idx::Int32,
    cache_size::Int32,
    total_hits::CuDeviceArray{Int64, 1},
    total_misses::CuDeviceArray{Int64, 1}
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    found = false
    
    # Linear search through cache (could be optimized with hash table)
    for i in tid:blockDim().x * gridDim().x:cache_size
        if i <= cache_size && cache_valid[i]
            if cache_keys[i] == feature_hash
                # Cache hit
                result_scores[node_idx] = cache_scores[i]
                result_ready[node_idx] = true
                CUDA.atomic_add!(pointer(cache_hits, i), Int32(1))
                
                if tid == 1
                    CUDA.atomic_add!(pointer(total_hits), Int64(1))
                end
                
                found = true
                break
            end
        end
    end
    
    # Record miss if not found
    if tid == 1 && !found
        CUDA.atomic_add!(pointer(total_misses), Int64(1))
    end
    
    return nothing
end

"""
Update cache kernel
"""
function update_cache_kernel!(
    cache_keys::CuDeviceArray{UInt64, 1},
    cache_scores::CuDeviceArray{Float32, 1},
    cache_timestamps::CuDeviceArray{Float32, 1},
    cache_valid::CuDeviceArray{Bool, 1},
    feature_hash::UInt64,
    score::Float32,
    timestamp::Float32,
    cache_size::Int32
)
    tid = threadIdx().x
    
    if tid == 1
        # Find oldest entry (simple LRU)
        oldest_time = Inf32
        oldest_idx = 1
        
        for i in 1:cache_size
            if !cache_valid[i] || cache_timestamps[i] < oldest_time
                oldest_time = cache_timestamps[i]
                oldest_idx = i
            end
        end
        
        # Update cache entry
        cache_keys[oldest_idx] = feature_hash
        cache_scores[oldest_idx] = score
        cache_timestamps[oldest_idx] = timestamp
        cache_valid[oldest_idx] = true
    end
    
    return nothing
end

"""
Compute feature hash for cache key
"""
@inline function compute_feature_hash(
    feature_masks::CuDeviceArray{UInt64, 2},
    node_idx::Int32
)
    hash = UInt64(0)
    
    # Simple XOR-based hash of feature chunks
    for chunk_idx in 1:FEATURE_CHUNKS
        mask = feature_masks[chunk_idx, node_idx]
        hash = hash ⊻ (mask * UInt64(chunk_idx))
        hash = (hash << 1) | (hash >> 63)  # Rotate
    end
    
    return hash
end

"""
Process batch collection kernel
"""
function collect_batch_kernel!(
    queue_requests::CuDeviceArray{EvalRequest, 1},
    queue_head::CuDeviceArray{Int32, 1},
    queue_size::CuDeviceArray{Int32, 1},
    batch_indices::CuDeviceArray{Int32, 1},
    batch_size_out::CuDeviceArray{Int32, 1},
    batch_ready::CuDeviceArray{Bool, 1},
    max_batch_size::Int32,
    max_queue_size::Int32,
    current_time::Float32,
    timeout_ms::Float32
)
    tid = threadIdx().x
    
    if tid == 1
        current_size = queue_size[1]
        if current_size > 0
            # Collect up to max_batch_size requests
            batch_count = min(current_size, max_batch_size)
            
            head_pos = queue_head[1]
            
            # Check if oldest request has timed out
            oldest_request = queue_requests[head_pos]
            time_waited = current_time - oldest_request.timestamp
            
            if batch_count >= max_batch_size ÷ 2 || time_waited >= timeout_ms
                # Process batch
                for i in 0:(batch_count-1)
                    actual_pos = ((head_pos + i - 1) % max_queue_size) + 1
                    request = queue_requests[actual_pos]
                    batch_indices[i + 1] = request.node_idx
                end
                
                # Update queue pointers
                queue_head[1] = ((head_pos + batch_count - 1) % max_queue_size) + 1
                CUDA.atomic_sub!(pointer(queue_size), batch_count)
                
                batch_size_out[1] = batch_count
                batch_ready[1] = true
            end
        end
    end
    
    return nothing
end

"""
Enqueue evaluation request
"""
function enqueue_evaluation!(
    manager::MetamodelManager,
    node_idx::Int32,
    priority::Int32 = Int32(0)
)
    timestamp = Float32(time())
    request_id = Int32(manager.total_inferences + 1)
    
    @cuda threads=1 enqueue_request_kernel!(
        manager.eval_queue.requests,
        manager.eval_queue.queue_tail,
        manager.eval_queue.queue_size,
        node_idx,
        request_id,
        timestamp,
        priority,
        manager.config.max_queue_size
    )
    
    return request_id
end

"""
Check if results are ready
"""
function check_results(
    manager::MetamodelManager,
    node_indices::Vector{Int32}
)
    results = Dict{Int32, Float32}()
    
    CUDA.@allowscalar begin
        for node_idx in node_indices
            if manager.eval_queue.result_ready[node_idx]
                results[node_idx] = manager.eval_queue.result_scores[node_idx]
                # Reset ready flag
                manager.eval_queue.result_ready[node_idx] = false
            end
        end
    end
    
    return results
end

"""
Process evaluation batch
"""
function process_batch!(
    manager::MetamodelManager,
    metamodel_fn::Function  # User-provided inference function
)
    current_time = Float32(time())
    
    # Collect batch
    @cuda threads=1 collect_batch_kernel!(
        manager.eval_queue.requests,
        manager.eval_queue.queue_head,
        manager.eval_queue.queue_size,
        manager.eval_queue.batch_indices,
        manager.eval_queue.batch_size,
        manager.eval_queue.batch_ready,
        manager.config.batch_size,
        manager.config.max_queue_size,
        current_time,
        manager.config.timeout_ms
    )
    
    # Check if batch is ready
    batch_ready = CUDA.@allowscalar manager.eval_queue.batch_ready[1]
    if !batch_ready
        return 0
    end
    
    batch_size = CUDA.@allowscalar manager.eval_queue.batch_size[1]
    
    # Prepare features
    # Note: This would need the tree's feature_masks, which we can't pass due to struct limitation
    # For now, this is a placeholder for the interface
    
    try
        # Call user-provided metamodel function
        start_time = time()
        scores = metamodel_fn(manager.feature_buffer, batch_size)
        inference_time = Float32((time() - start_time) * 1000)
        
        # Store results
        CUDA.@allowscalar begin
            for i in 1:batch_size
                node_idx = manager.eval_queue.batch_indices[i]
                if node_idx > 0 && node_idx <= length(manager.eval_queue.result_scores)
                    manager.eval_queue.result_scores[node_idx] = scores[i]
                    manager.eval_queue.result_ready[node_idx] = true
                end
            end
        end
        
        # Update statistics
        manager.last_inference_time = inference_time
        manager.total_inferences += batch_size
        
        # Update queue statistics
        CUDA.@allowscalar begin
            manager.eval_queue.total_requests[1] += batch_size
            manager.eval_queue.total_batches[1] += 1
            
            # Update running averages
            α = 0.1f0  # Exponential moving average factor
            current_avg_batch = manager.eval_queue.avg_batch_size[1]
            manager.eval_queue.avg_batch_size[1] = 
                α * Float32(batch_size) + (1 - α) * current_avg_batch
        end
        
    catch e
        # Fallback on error
        @warn "Metamodel evaluation failed, using fallback scores" exception=e
        
        CUDA.@allowscalar begin
            for i in 1:batch_size
                node_idx = manager.eval_queue.batch_indices[i]
                if node_idx > 0 && node_idx <= length(manager.eval_queue.result_scores)
                    manager.eval_queue.result_scores[node_idx] = manager.config.fallback_score
                    manager.eval_queue.result_ready[node_idx] = true
                end
            end
        end
    end
    
    # Reset batch ready flag
    CUDA.@allowscalar manager.eval_queue.batch_ready[1] = false
    
    return batch_size
end

"""
Get evaluation statistics
"""
function get_eval_statistics(manager::MetamodelManager)
    stats = Dict{String, Any}()
    
    CUDA.@allowscalar begin
        # Queue statistics
        stats["queue_size"] = manager.eval_queue.queue_size[1]
        stats["total_requests"] = manager.eval_queue.total_requests[1]
        stats["total_batches"] = manager.eval_queue.total_batches[1]
        stats["avg_batch_size"] = manager.eval_queue.avg_batch_size[1]
        stats["avg_wait_time_ms"] = manager.eval_queue.avg_wait_time_ms[1]
        
        # Cache statistics
        total_hits = manager.result_cache.total_hits[1]
        total_misses = manager.result_cache.total_misses[1]
        total_lookups = total_hits + total_misses
        
        stats["cache_hits"] = total_hits
        stats["cache_misses"] = total_misses
        stats["cache_hit_rate"] = total_lookups > 0 ? 
            Float32(total_hits) / Float32(total_lookups) : 0.0f0
        
        # Manager statistics
        stats["total_inferences"] = manager.total_inferences
        stats["last_inference_time_ms"] = manager.last_inference_time
        stats["is_active"] = manager.is_active
    end
    
    return stats
end

# Export types and functions
export MetamodelConfig, MetamodelManager, EvalRequest, EvalResult
export EvalQueue, ResultCache
export enqueue_evaluation!, check_results, process_batch!, get_eval_statistics

end # module MetamodelIntegration