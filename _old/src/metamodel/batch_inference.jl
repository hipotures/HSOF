module BatchInference

using CUDA
using Statistics
using LinearAlgebra
using Printf

# Include dependencies
include("../gpu/kernels/metamodel_integration.jl")
using .MetamodelIntegration

export BatchInferenceConfig, BatchInferenceEngine, InferenceBatch
export PersistentKernel, FusedInferenceKernel, BatchCache
export create_batch_inference_engine, submit_inference_request!
export process_inference_batch!, get_inference_results, flush_cache!
export start_persistent_kernels!, stop_persistent_kernels!

"""
Configuration for batch inference system
"""
struct BatchInferenceConfig
    max_batch_size::Int32           # Maximum features per batch (target: 1000+)
    target_latency_ms::Float32      # Target latency (target: <1ms)
    cache_size::Int32               # Number of cached results
    persistent_kernels::Int32       # Number of persistent kernels
    feature_dim::Int32              # Input feature dimension
    hidden_dims::Vector{Int32}      # Hidden layer dimensions
    output_dim::Int32               # Output dimension
    use_mixed_precision::Bool       # Use FP16 for inference
    enable_profiling::Bool          # Enable detailed profiling
    prefetch_batches::Int32         # Number of batches to prefetch
    min_batch_timeout_us::Int32     # Minimum batch collection timeout (microseconds)
end

"""
Default configuration optimized for <1ms inference
"""
function BatchInferenceConfig(;
    max_batch_size::Int32 = Int32(1024),
    target_latency_ms::Float32 = 0.8f0,
    cache_size::Int32 = Int32(8192),
    persistent_kernels::Int32 = Int32(4),
    feature_dim::Int32 = Int32(500),
    hidden_dims::Vector{Int32} = Int32[256, 128, 64],
    output_dim::Int32 = Int32(1),
    use_mixed_precision::Bool = true,
    enable_profiling::Bool = true,
    prefetch_batches::Int32 = Int32(2),
    min_batch_timeout_us::Int32 = Int32(50)
)
    BatchInferenceConfig(
        max_batch_size, target_latency_ms, cache_size, persistent_kernels,
        feature_dim, hidden_dims, output_dim, use_mixed_precision,
        enable_profiling, prefetch_batches, min_batch_timeout_us
    )
end

"""
Individual inference request
"""
struct InferenceRequest
    request_id::UInt64              # Unique request ID
    feature_hash::UInt64            # Hash of feature combination
    node_idx::Int32                 # MCTS node index
    priority::Int32                 # Request priority
    timestamp_us::UInt64            # Request timestamp (microseconds)
end

"""
Batch of inference requests
"""
mutable struct InferenceBatch
    requests::CuArray{InferenceRequest, 1}    # Batch requests
    features::CuArray{Float32, 2}             # Feature matrix [feature_dim x batch_size]
    features_fp16::Union{Nothing, CuArray{Float16, 2}}  # FP16 features if enabled
    batch_size::CuArray{Int32, 1}             # Current batch size
    ready::CuArray{Bool, 1}                   # Batch ready flag
    processed::CuArray{Bool, 1}               # Batch processed flag
    
    # Results
    outputs::CuArray{Float32, 1}              # Inference outputs
    latencies_us::CuArray{UInt64, 1}          # Per-request latencies
    
    # Timing
    batch_start_time::CuArray{UInt64, 1}      # Batch processing start time
    batch_end_time::CuArray{UInt64, 1}        # Batch processing end time
end

"""
Create new inference batch
"""
function InferenceBatch(config::BatchInferenceConfig)
    max_batch = config.max_batch_size
    feature_dim = config.feature_dim
    
    features_fp16 = config.use_mixed_precision ? 
        CUDA.zeros(Float16, feature_dim, max_batch) : nothing
    
    InferenceBatch(
        CUDA.fill(InferenceRequest(0, 0, 0, 0, 0), max_batch),
        CUDA.zeros(Float32, feature_dim, max_batch),
        features_fp16,
        CUDA.zeros(Int32, 1),
        CUDA.zeros(Bool, 1),
        CUDA.zeros(Bool, 1),
        CUDA.zeros(Float32, max_batch),
        CUDA.zeros(UInt64, max_batch),
        CUDA.zeros(UInt64, 1),
        CUDA.zeros(UInt64, 1)
    )
end

"""
Cache entry for storing inference results
"""
struct CacheEntry
    feature_hash::UInt64            # Hash key
    output::Float32                 # Cached output
    hit_count::Int32                # Number of cache hits
    timestamp_us::UInt64            # Last access time
    valid::Bool                     # Entry validity flag
end

"""
High-performance cache for inference results
"""
mutable struct BatchCache
    entries::CuArray{CacheEntry, 1}           # Cache entries
    hash_table::CuArray{Int32, 1}             # Hash table for O(1) lookup
    size::Int32                               # Cache size
    capacity::Int32                           # Cache capacity
    
    # Statistics
    total_lookups::CuArray{UInt64, 1}         # Total cache lookups
    total_hits::CuArray{UInt64, 1}            # Total cache hits
    hit_rate::CuArray{Float32, 1}             # Current hit rate
end

"""
Create new batch cache
"""
function BatchCache(capacity::Int32)
    # Use power of 2 for efficient hashing
    table_size = nextpow(2, capacity * 2)
    
    BatchCache(
        CUDA.fill(CacheEntry(0, 0.0f0, 0, 0, false), capacity),
        CUDA.fill(Int32(-1), table_size),  # -1 indicates empty slot
        0,
        capacity,
        CUDA.zeros(UInt64, 1),
        CUDA.zeros(UInt64, 1),
        CUDA.zeros(Float32, 1)
    )
end

"""
Persistent kernel for continuous batch processing
"""
mutable struct PersistentKernel
    kernel_id::Int32                          # Kernel identifier
    stream::CuStream                          # Dedicated CUDA stream
    is_running::CuArray{Bool, 1}              # Kernel running flag
    work_available::CuArray{Bool, 1}          # Work available flag
    batch_queue::CuArray{Int32, 1}            # Queue of batch indices
    queue_head::CuArray{Int32, 1}             # Queue head pointer
    queue_tail::CuArray{Int32, 1}             # Queue tail pointer
    queue_size::CuArray{Int32, 1}             # Current queue size
    
    # Performance metrics
    batches_processed::CuArray{UInt64, 1}     # Total batches processed
    total_latency_us::CuArray{UInt64, 1}      # Cumulative latency
    avg_latency_us::CuArray{Float32, 1}       # Average latency
end

"""
Create persistent kernel
"""
function PersistentKernel(kernel_id::Int32, queue_capacity::Int32)
    PersistentKernel(
        kernel_id,
        CuStream(),
        CUDA.ones(Bool, 1),    # Start running
        CUDA.zeros(Bool, 1),
        CUDA.zeros(Int32, queue_capacity),
        CUDA.ones(Int32, 1),   # Start at position 1
        CUDA.ones(Int32, 1),
        CUDA.zeros(Int32, 1),
        CUDA.zeros(UInt64, 1),
        CUDA.zeros(UInt64, 1),
        CUDA.zeros(Float32, 1)
    )
end

"""
Fused inference kernel combining multiple neural network layers
"""
struct FusedInferenceKernel
    weights::Vector{CuArray{Float32, 2}}      # Layer weights
    biases::Vector{CuArray{Float32, 1}}       # Layer biases
    weights_fp16::Vector{Union{Nothing, CuArray{Float16, 2}}}  # FP16 weights
    biases_fp16::Vector{Union{Nothing, CuArray{Float16, 1}}}   # FP16 biases
    layer_dims::Vector{Int32}                 # Layer dimensions
    activation_type::Symbol                   # Activation function (:relu, :tanh, :sigmoid)
end

"""
Create fused inference kernel
"""
function FusedInferenceKernel(
    config::BatchInferenceConfig,
    weights::Vector{Matrix{Float32}},
    biases::Vector{Vector{Float32}},
    activation_type::Symbol = :relu
)
    # Convert to GPU arrays
    gpu_weights = [CuArray(w) for w in weights]
    gpu_biases = [CuArray(b) for b in biases]
    
    # Create FP16 versions if enabled
    gpu_weights_fp16 = config.use_mixed_precision ? 
        [CuArray(Float16.(w)) for w in weights] : [nothing for _ in weights]
    gpu_biases_fp16 = config.use_mixed_precision ?
        [CuArray(Float16.(b)) for b in biases] : [nothing for _ in biases]
    
    layer_dims = Int32[config.feature_dim; config.hidden_dims; config.output_dim]
    
    FusedInferenceKernel(
        gpu_weights, gpu_biases, gpu_weights_fp16, gpu_biases_fp16,
        layer_dims, activation_type
    )
end

"""
Main batch inference engine
"""
mutable struct BatchInferenceEngine
    config::BatchInferenceConfig
    cache::BatchCache
    persistent_kernels::Vector{PersistentKernel}
    fused_kernel::FusedInferenceKernel
    
    # Batch management
    active_batches::Vector{InferenceBatch}
    request_queue::Channel{InferenceRequest}
    result_queue::Channel{Tuple{UInt64, Float32, UInt64}}  # (request_id, output, latency)
    
    # Synchronization
    batch_ready_event::CuEvent
    processing_complete_event::CuEvent
    
    # Performance tracking
    total_requests::UInt64
    total_batches::UInt64
    avg_batch_size::Float32
    avg_latency_us::Float32
    throughput_reqs_per_sec::Float32
    
    # Control
    is_running::Bool
end

"""
Create batch inference engine
"""
function create_batch_inference_engine(
    config::BatchInferenceConfig,
    weights::Vector{Matrix{Float32}},
    biases::Vector{Vector{Float32}}
)
    # Create cache
    cache = BatchCache(config.cache_size)
    
    # Create persistent kernels
    queue_capacity = config.max_batch_size * 2  # Double buffer
    persistent_kernels = [PersistentKernel(Int32(i), Int32(queue_capacity)) 
                         for i in 1:config.persistent_kernels]
    
    # Create fused kernel
    fused_kernel = FusedInferenceKernel(config, weights, biases)
    
    # Create batches
    n_batches = config.persistent_kernels + config.prefetch_batches
    active_batches = [InferenceBatch(config) for _ in 1:n_batches]
    
    # Create queues
    request_queue = Channel{InferenceRequest}(config.max_batch_size * 4)
    result_queue = Channel{Tuple{UInt64, Float32, UInt64}}(config.max_batch_size * 4)
    
    # Create events
    batch_ready_event = CuEvent()
    processing_complete_event = CuEvent()
    
    BatchInferenceEngine(
        config, cache, persistent_kernels, fused_kernel,
        active_batches, request_queue, result_queue,
        batch_ready_event, processing_complete_event,
        0, 0, 0.0f0, 0.0f0, 0.0f0,
        false
    )
end

"""
Cache lookup kernel - optimized for high throughput
"""
function cache_lookup_kernel!(
    cache_entries::CuDeviceArray{CacheEntry, 1},
    hash_table::CuDeviceArray{Int32, 1},
    feature_hash::UInt64,
    result::CuDeviceArray{Float32, 1},
    found::CuDeviceArray{Bool, 1},
    table_mask::Int32
)
    # Fast hash table lookup using linear probing
    hash_idx = (feature_hash % UInt64(table_mask)) + 1
    
    for probe in 0:7  # Max 8 probes for good performance
        idx = ((hash_idx + probe - 1) % table_mask) + 1
        entry_idx = hash_table[idx]
        
        if entry_idx == -1  # Empty slot
            found[1] = false
            return nothing
        end
        
        entry = cache_entries[entry_idx]
        if entry.valid && entry.feature_hash == feature_hash
            result[1] = entry.output
            found[1] = true
            
            # Atomic increment hit count
            # Note: This is a simplified version - real implementation would use atomic operations
            return nothing
        end
    end
    
    found[1] = false
    return nothing
end

"""
Batch assembly kernel - collect requests into processing batches
"""
function batch_assembly_kernel!(
    request_buffer::CuDeviceArray{InferenceRequest, 1},
    feature_buffer::CuDeviceArray{Float32, 2},
    feature_masks::CuDeviceArray{UInt64, 2},  # From MCTS tree
    batch_size::CuDeviceArray{Int32, 1},
    max_batch_size::Int32,
    feature_dim::Int32,
    current_time_us::UInt64,
    timeout_us::Int32
)
    tid = threadIdx().x
    bid = blockIdx().x
    
    if bid == 1 && tid == 1
        # Single thread handles batch assembly for simplicity
        # Real implementation could parallelize this further
        
        current_batch_size = batch_size[1]
        if current_batch_size > 0
            # Check timeout condition
            oldest_request = request_buffer[1]
            time_waited = current_time_us - oldest_request.timestamp_us
            
            if current_batch_size >= max_batch_size ÷ 2 || time_waited >= timeout_us
                # Prepare features for this batch
                for i in 1:current_batch_size
                    request = request_buffer[i]
                    node_idx = request.node_idx
                    
                    # Convert feature mask to dense features
                    for feat_idx in 1:feature_dim
                        chunk_idx = div(feat_idx - 1, 64) + 1
                        bit_idx = mod(feat_idx - 1, 64)
                        
                        if chunk_idx <= size(feature_masks, 1)
                            mask = feature_masks[chunk_idx, node_idx]
                            is_set = (mask >> bit_idx) & UInt64(1)
                            feature_buffer[feat_idx, i] = Float32(is_set)
                        else
                            feature_buffer[feat_idx, i] = 0.0f0
                        end
                    end
                end
            end
        end
    end
    
    return nothing
end

"""
Fused multi-layer perceptron kernel with activation functions
"""
function fused_mlp_kernel!(
    input::CuDeviceArray{T, 2},
    output::CuDeviceArray{T, 1},
    weights::Tuple{Vararg{CuDeviceArray{T, 2}}},
    biases::Tuple{Vararg{CuDeviceArray{T, 1}}},
    batch_size::Int32,
    activation_type::Symbol
) where T
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    bid_y = blockIdx().y
    
    if bid_y <= batch_size
        # Shared memory for intermediate activations
        shared_size = 512  # Adjust based on largest hidden layer
        shmem = CuDynamicSharedArray(T, shared_size)
        
        n_layers = length(weights)
        sample_idx = bid_y
        
        # Input layer to first hidden layer
        if tid <= size(weights[1], 1)  # First hidden layer size
            val = biases[1][tid]
            
            # Compute dot product with input
            for i in 1:size(input, 1)
                val += weights[1][tid, i] * input[i, sample_idx]
            end
            
            # Apply activation
            if activation_type == :relu
                val = max(val, T(0))
            elseif activation_type == :tanh
                val = tanh(val)
            elseif activation_type == :sigmoid
                val = T(1) / (T(1) + exp(-val))
            end
            
            shmem[tid] = val
        end
        
        __syncthreads()
        
        # Process remaining hidden layers
        current_input = shmem
        current_size = size(weights[1], 1)
        
        for layer in 2:n_layers
            output_size = size(weights[layer], 1)
            
            if tid <= output_size
                val = biases[layer][tid]
                
                # Compute dot product
                for i in 1:current_size
                    val += weights[layer][tid, i] * current_input[i]
                end
                
                # Apply activation (except for output layer)
                if layer < n_layers
                    if activation_type == :relu
                        val = max(val, T(0))
                    elseif activation_type == :tanh
                        val = tanh(val)
                    elseif activation_type == :sigmoid
                        val = T(1) / (T(1) + exp(-val))
                    end
                end
                
                # Store in shared memory or output
                if layer == n_layers
                    output[sample_idx] = val  # Final output
                else
                    # Offset in shared memory for next layer
                    offset = current_size
                    shmem[offset + tid] = val
                end
            end
            
            __syncthreads()
            
            if layer < n_layers
                current_input = @view shmem[(current_size + 1):(current_size + output_size)]
                current_size = output_size
            end
        end
    end
    
    return nothing
end

"""
Persistent kernel for continuous batch processing
"""
function persistent_inference_kernel!(
    kernel_id::Int32,
    is_running::CuDeviceArray{Bool, 1},
    work_available::CuDeviceArray{Bool, 1},
    batch_queue::CuDeviceArray{Int32, 1},
    queue_head::CuDeviceArray{Int32, 1},
    queue_size::CuDeviceArray{Int32, 1},
    batches_processed::CuDeviceArray{UInt64, 1}
)
    tid = threadIdx().x
    
    # Persistent loop - kernel stays alive
    while is_running[1]
        if tid == 1
            # Check for available work
            if queue_size[1] > 0
                # Dequeue batch
                head_pos = queue_head[1]
                batch_idx = batch_queue[head_pos]
                
                # Update queue
                queue_head[1] = (head_pos % length(batch_queue)) + 1
                CUDA.atomic_sub!(pointer(queue_size), Int32(1))
                
                # Signal work processing
                work_available[1] = true
                
                # Increment processed count
                CUDA.atomic_add!(pointer(batches_processed), UInt64(1))
            else
                # No work available, brief spin
                work_available[1] = false
            end
        end
        
        __syncthreads()
        
        # Brief pause to avoid excessive GPU utilization
        if !work_available[1]
            # Cooperative yielding - let other work proceed
            for _ in 1:1000
                # Simple delay loop
            end
        end
    end
    
    return nothing
end

"""
Submit inference request to the engine
"""
function submit_inference_request!(
    engine::BatchInferenceEngine,
    node_idx::Int32,
    feature_hash::UInt64,
    priority::Int32 = Int32(0)
)
    request_id = UInt64(engine.total_requests + 1)
    timestamp_us = UInt64(time() * 1_000_000)
    
    request = InferenceRequest(request_id, feature_hash, node_idx, priority, timestamp_us)
    
    # Non-blocking put to request queue
    try
        put!(engine.request_queue, request)
        engine.total_requests += 1
        return request_id
    catch
        @warn "Request queue full, dropping request"
        return UInt64(0)
    end
end

"""
Process inference batch - main inference execution
"""
function process_inference_batch!(
    engine::BatchInferenceEngine,
    batch::InferenceBatch
)
    config = engine.config
    
    # Record batch start time
    start_time = UInt64(time() * 1_000_000)
    CUDA.@allowscalar batch.batch_start_time[1] = start_time
    
    # Get batch size
    batch_size = CUDA.@allowscalar batch.batch_size[1]
    
    if batch_size == 0
        return 0
    end
    
    # Check cache for each request
    cache_hits = 0
    remaining_requests = Int32[]
    
    for i in 1:batch_size
        request = CUDA.@allowscalar batch.requests[i]
        
        # Simple cache lookup (optimized version would use GPU kernel)
        cached_result = lookup_cache(engine.cache, request.feature_hash)
        if cached_result !== nothing
            CUDA.@allowscalar batch.outputs[i] = cached_result
            cache_hits += 1
        else
            push!(remaining_requests, i)
        end
    end
    
    # Process remaining requests through neural network
    if !isempty(remaining_requests)
        remaining_size = length(remaining_requests)
        
        # Extract features for remaining requests
        remaining_features = batch.features[:, remaining_requests]
        remaining_outputs = batch.outputs[remaining_requests]
        
        # Run fused MLP inference
        run_fused_inference!(
            engine.fused_kernel,
            remaining_features,
            remaining_outputs,
            remaining_size,
            config.use_mixed_precision
        )
        
        # Update cache with new results
        for (idx, original_idx) in enumerate(remaining_requests)
            request = CUDA.@allowscalar batch.requests[original_idx]
            output = CUDA.@allowscalar batch.outputs[original_idx]
            update_cache!(engine.cache, request.feature_hash, output, start_time)
        end
    end
    
    # Record batch end time and calculate latencies
    end_time = UInt64(time() * 1_000_000)
    CUDA.@allowscalar batch.batch_end_time[1] = end_time
    
    batch_latency = end_time - start_time
    
    # Update per-request latencies
    for i in 1:batch_size
        CUDA.@allowscalar batch.latencies_us[i] = batch_latency
    end
    
    # Mark batch as processed
    CUDA.@allowscalar batch.processed[1] = true
    
    # Update engine statistics
    engine.total_batches += 1
    engine.avg_batch_size = (engine.avg_batch_size * (engine.total_batches - 1) + batch_size) / engine.total_batches
    engine.avg_latency_us = (engine.avg_latency_us * (engine.total_requests - batch_size) + batch_latency * batch_size) / engine.total_requests
    
    return batch_size
end

"""
Simple cache lookup (CPU version for now)
"""
function lookup_cache(cache::BatchCache, feature_hash::UInt64)
    # This is a simplified CPU version
    # Production version would use GPU kernels
    
    CUDA.@allowscalar begin
        table_size = length(cache.hash_table)
        table_mask = table_size - 1
        
        hash_idx = (feature_hash % UInt64(table_mask)) + 1
        
        for probe in 0:7
            idx = ((hash_idx + probe - 1) % table_mask) + 1
            entry_idx = cache.hash_table[idx]
            
            if entry_idx == -1
                cache.total_lookups[1] += 1
                # total_misses = total_lookups - total_hits (calculated when needed)
                return nothing
            end
            
            entry = cache.entries[entry_idx]
            if entry.valid && entry.feature_hash == feature_hash
                cache.total_lookups[1] += 1
                cache.total_hits[1] += 1
                cache.hit_rate[1] = cache.total_hits[1] / cache.total_lookups[1]
                return entry.output
            end
        end
        
        cache.total_lookups[1] += 1
        # total_misses = total_lookups - total_hits (calculated when needed)
        return nothing
    end
end

"""
Update cache with new result
"""
function update_cache!(cache::BatchCache, feature_hash::UInt64, output::Float32, timestamp_us::UInt64)
    CUDA.@allowscalar begin
        if cache.size < cache.capacity
            # Find empty slot
            entry_idx = cache.size + 1
            cache.size += 1
        else
            # Find LRU entry
            oldest_time = typemax(UInt64)
            entry_idx = 1
            
            for i in 1:cache.capacity
                entry = cache.entries[i]
                if entry.timestamp_us < oldest_time
                    oldest_time = entry.timestamp_us
                    entry_idx = i
                end
            end
        end
        
        # Update entry
        new_entry = CacheEntry(feature_hash, output, 0, timestamp_us, true)
        cache.entries[entry_idx] = new_entry
        
        # Update hash table
        table_size = length(cache.hash_table)
        table_mask = table_size - 1
        hash_idx = (feature_hash % UInt64(table_mask)) + 1
        
        for probe in 0:7
            idx = ((hash_idx + probe - 1) % table_mask) + 1
            if cache.hash_table[idx] == -1
                cache.hash_table[idx] = entry_idx
                break
            end
        end
    end
end

"""
Run fused inference using optimized kernels
"""
function run_fused_inference!(
    fused_kernel::FusedInferenceKernel,
    input_features::CuArray{Float32, 2},
    outputs::CuArray{Float32, 1},
    batch_size::Int,
    use_mixed_precision::Bool
)
    if use_mixed_precision && !isnothing(fused_kernel.weights_fp16[1])
        # Use FP16 inference for higher throughput
        input_fp16 = CuArray{Float16}(input_features)
        outputs_fp16 = CUDA.zeros(Float16, batch_size)
        
        # Run FP16 inference (simplified for this implementation)
        run_simple_mlp!(input_fp16, outputs_fp16, fused_kernel.weights_fp16, fused_kernel.biases_fp16)
        
        # Convert back to FP32
        outputs .= Float32.(outputs_fp16)
    else
        # Use FP32 inference
        run_simple_mlp!(input_features, outputs, fused_kernel.weights, fused_kernel.biases)
    end
end

"""
Simplified MLP inference (placeholder for full fused kernel)
"""
function run_simple_mlp!(
    input::CuArray{T, 2},
    output::CuArray{T, 1},
    weights::Vector{<:Union{Nothing, CuArray{T, 2}}},
    biases::Vector{<:Union{Nothing, CuArray{T, 1}}}
) where T
    current_activation = input
    
    for (i, (W, b)) in enumerate(zip(weights, biases))
        if W === nothing || b === nothing
            continue
        end
        
        # Linear transformation
        next_activation = W * current_activation .+ b
        
        # Apply activation (except for last layer)
        if i < length(weights)
            next_activation = max.(next_activation, T(0))  # ReLU
        end
        
        current_activation = next_activation
    end
    
    output .= vec(current_activation)
end

"""
Get inference results from the engine
"""
function get_inference_results(engine::BatchInferenceEngine, timeout_ms::Float32 = 1.0f0)
    results = Dict{UInt64, Tuple{Float32, UInt64}}()  # request_id => (output, latency_us)
    
    timeout_time = time() + timeout_ms / 1000
    
    while time() < timeout_time && isready(engine.result_queue)
        try
            request_id, output, latency = take!(engine.result_queue)
            results[request_id] = (output, latency)
        catch
            break
        end
    end
    
    return results
end

"""
Start persistent kernels for continuous processing
"""
function start_persistent_kernels!(engine::BatchInferenceEngine)
    engine.is_running = true
    
    for (i, kernel) in enumerate(engine.persistent_kernels)
        CUDA.stream!(kernel.stream) do
            # Launch persistent kernel
            threads = 128
            blocks = 1
            shared_mem = 0
            
            @cuda threads=threads blocks=blocks shmem=shared_mem persistent_inference_kernel!(
                kernel.kernel_id,
                kernel.is_running,
                kernel.work_available,
                kernel.batch_queue,
                kernel.queue_head,
                kernel.queue_size,
                kernel.batches_processed
            )
        end
    end
    
    println("Started $(length(engine.persistent_kernels)) persistent inference kernels")
end

"""
Stop persistent kernels
"""
function stop_persistent_kernels!(engine::BatchInferenceEngine)
    engine.is_running = false
    
    # Signal all kernels to stop
    for kernel in engine.persistent_kernels
        CUDA.@allowscalar kernel.is_running[1] = false
    end
    
    # Wait for kernels to finish
    for kernel in engine.persistent_kernels
        CUDA.synchronize(kernel.stream)
    end
    
    println("Stopped all persistent inference kernels")
end

"""
Flush inference cache
"""
function flush_cache!(engine::BatchInferenceEngine)
    # Reset cache
    CUDA.fill!(engine.cache.hash_table, Int32(-1))
    CUDA.fill!(engine.cache.entries, CacheEntry(0, 0.0f0, 0, 0, false))
    engine.cache.size = 0
    
    # Reset statistics
    CUDA.@allowscalar begin
        engine.cache.total_lookups[1] = 0
        engine.cache.total_hits[1] = 0
        engine.cache.hit_rate[1] = 0.0f0
    end
    
    println("Cache flushed")
end

"""
Get performance statistics
"""
function get_performance_stats(engine::BatchInferenceEngine)
    stats = Dict{String, Any}()
    
    stats["total_requests"] = engine.total_requests
    stats["total_batches"] = engine.total_batches
    stats["avg_batch_size"] = engine.avg_batch_size
    stats["avg_latency_us"] = engine.avg_latency_us
    stats["avg_latency_ms"] = engine.avg_latency_us / 1000
    
    # Calculate throughput
    if engine.avg_latency_us > 0
        stats["throughput_reqs_per_sec"] = 1_000_000 / engine.avg_latency_us * engine.avg_batch_size
    else
        stats["throughput_reqs_per_sec"] = 0.0
    end
    
    # Cache statistics
    CUDA.@allowscalar begin
        stats["cache_size"] = engine.cache.size
        stats["cache_capacity"] = engine.cache.capacity
        stats["cache_hit_rate"] = engine.cache.hit_rate[1]
        stats["cache_total_lookups"] = engine.cache.total_lookups[1]
        stats["cache_total_hits"] = engine.cache.total_hits[1]
    end
    
    # Persistent kernel statistics
    kernel_stats = []
    for (i, kernel) in enumerate(engine.persistent_kernels)
        CUDA.@allowscalar begin
            push!(kernel_stats, Dict(
                "kernel_id" => kernel.kernel_id,
                "batches_processed" => kernel.batches_processed[1],
                "avg_latency_us" => kernel.avg_latency_us[1],
                "queue_size" => kernel.queue_size[1]
            ))
        end
    end
    stats["persistent_kernels"] = kernel_stats
    
    return stats
end

end # module BatchInference