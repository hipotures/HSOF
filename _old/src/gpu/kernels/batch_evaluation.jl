module BatchEvaluation

using CUDA
using ..MCTSTypes

# Work queue constants from PersistentKernel
const WORK_SELECT = UInt8(0)
const WORK_EXPAND = UInt8(1)
const WORK_EVALUATE = UInt8(2)
const WORK_BACKUP = UInt8(3)

# Import WorkQueue structure
struct WorkQueue
    items::CuArray{Int32, 2}      # [4 x max_items] - operation, node_idx, thread_id, priority
    head::CuArray{Int32, 1}       # Queue head (atomic)
    tail::CuArray{Int32, 1}       # Queue tail (atomic)
    size::CuArray{Int32, 1}       # Current size (atomic)
    max_size::Int32
end

"""
Batch evaluation configuration
"""
struct BatchEvalConfig
    batch_size::Int32            # Maximum nodes per batch
    double_buffer::Bool          # Use double buffering
    eval_threads::Int32          # Threads per evaluation
    coalesce_threshold::Float32  # Minimum occupancy to trigger eval
end

"""
Evaluation batch buffer with double buffering support
"""
struct EvalBatchBuffer
    # Node indices for current batch
    node_indices_a::CuArray{Int32, 1}
    node_indices_b::CuArray{Int32, 1}
    
    # Feature data prepared for NN (dense format)
    features_a::CuArray{Float32, 2}      # [features x batch_size]
    features_b::CuArray{Float32, 2}
    
    # Evaluation results
    scores_a::CuArray{Float32, 1}
    scores_b::CuArray{Float32, 1}
    priors_a::CuArray{Float32, 2}       # [max_actions x batch_size]
    priors_b::CuArray{Float32, 2}
    
    # Batch management
    batch_size_a::CuArray{Int32, 1}     # Current batch size
    batch_size_b::CuArray{Int32, 1}
    active_buffer::CuArray{Int32, 1}    # 0 or 1
    ready_flag::CuArray{Bool, 1}        # Batch ready for eval
    
    # Synchronization
    producer_lock::CuArray{Int32, 1}
    consumer_lock::CuArray{Int32, 1}
end

"""
Create evaluation batch buffer
"""
function EvalBatchBuffer(batch_size::Int32, num_features::Int32, max_actions::Int32)
    # Allocate double buffers
    node_indices_a = CUDA.zeros(Int32, batch_size)
    node_indices_b = CUDA.zeros(Int32, batch_size)
    
    features_a = CUDA.zeros(Float32, num_features, batch_size)
    features_b = CUDA.zeros(Float32, num_features, batch_size)
    
    scores_a = CUDA.zeros(Float32, batch_size)
    scores_b = CUDA.zeros(Float32, batch_size)
    priors_a = CUDA.zeros(Float32, max_actions, batch_size)
    priors_b = CUDA.zeros(Float32, max_actions, batch_size)
    
    batch_size_a = CUDA.zeros(Int32, 1)
    batch_size_b = CUDA.zeros(Int32, 1)
    active_buffer = CUDA.zeros(Int32, 1)
    ready_flag = CUDA.zeros(Bool, 2)  # One per buffer
    
    producer_lock = CUDA.zeros(Int32, 1)
    consumer_lock = CUDA.zeros(Int32, 1)
    
    EvalBatchBuffer(
        node_indices_a, node_indices_b,
        features_a, features_b,
        scores_a, scores_b,
        priors_a, priors_b,
        batch_size_a, batch_size_b,
        active_buffer, ready_flag,
        producer_lock, consumer_lock
    )
end

"""
Collect nodes for batch evaluation with coalesced memory access
"""
function collect_eval_batch!(
    tree::MCTSTreeSoA,
    work_queue::WorkQueue,
    batch_buffer::EvalBatchBuffer,
    config::BatchEvalConfig,
    gid::Int32,
    stride::Int32
)
    # Determine which buffer to use
    buffer_idx = @inbounds batch_buffer.active_buffer[1]
    
    # Select appropriate arrays based on active buffer
    node_indices = buffer_idx == 0 ? batch_buffer.node_indices_a : batch_buffer.node_indices_b
    features = buffer_idx == 0 ? batch_buffer.features_a : batch_buffer.features_b
    batch_size_ptr = buffer_idx == 0 ? batch_buffer.batch_size_a : batch_buffer.batch_size_b
    
    # Process work queue items
    work_size = @inbounds work_queue.size[1]
    
    for idx in gid:stride:work_size
        operation = @inbounds work_queue.items[1, idx]
        node_idx = @inbounds work_queue.items[2, idx]
        
        if operation != WORK_EVALUATE || node_idx <= 0
            continue
        end
        
        # Try to add to batch
        batch_pos = CUDA.atomic_add!(pointer(batch_size_ptr), Int32(1)) + 1
        
        if batch_pos <= config.batch_size
            # Add node to batch
            @inbounds node_indices[batch_pos] = node_idx
            
            # Convert sparse feature mask to dense representation
            convert_features_to_dense!(
                tree.feature_masks,
                features,
                node_idx,
                batch_pos
            )
        else
            # Batch full, restore counter
            CUDA.atomic_sub!(pointer(batch_size_ptr), Int32(1))
        end
    end
    
    return nothing
end

"""
Convert sparse feature mask to dense feature vector
"""
@inline function convert_features_to_dense!(
    feature_masks::CuArray{UInt64, 2},
    dense_features::CuArray{Float32, 2},
    node_idx::Int32,
    batch_pos::Int32
)
    # Coalesced write pattern - threads handle consecutive features
    tid = threadIdx().x
    num_features = size(dense_features, 1)
    
    # Each thread processes multiple features
    for feature_idx in tid:blockDim().x:num_features
        if feature_idx <= MAX_FEATURES
            # Check if feature is set
            has_feat = has_feature(feature_masks, node_idx, feature_idx)
            @inbounds dense_features[feature_idx, batch_pos] = has_feat ? 1.0f0 : 0.0f0
        else
            # Padding for alignment
            @inbounds dense_features[feature_idx, batch_pos] = 0.0f0
        end
    end
end

"""
Dynamic batch size determination based on GPU occupancy
"""
function compute_dynamic_batch_size(
    current_nodes::Int32,
    gpu_occupancy::Float32,
    config::BatchEvalConfig
)
    # Adjust batch size based on GPU utilization
    if gpu_occupancy < config.coalesce_threshold
        # Low occupancy - increase batch size
        return min(current_nodes, config.batch_size)
    else
        # High occupancy - use smaller batches for lower latency
        return min(current_nodes, div(config.batch_size, 2))
    end
end

"""
Dispatch batch for neural network evaluation
"""
function dispatch_eval_batch!(
    batch_buffer::EvalBatchBuffer,
    buffer_idx::Int32,
    batch_size::Int32
)
    # Mark batch as ready for evaluation
    @inbounds batch_buffer.ready_flag[buffer_idx + 1] = true
    
    # Memory fence to ensure visibility
    CUDA.threadfence_system()
    
    # Switch to other buffer for continued collection
    new_buffer = 1 - buffer_idx
    @inbounds batch_buffer.active_buffer[1] = new_buffer
    
    # Reset new buffer's counter
    if new_buffer == 0
        @inbounds batch_buffer.batch_size_a[1] = 0
    else
        @inbounds batch_buffer.batch_size_b[1] = 0
    end
    
    return nothing
end

"""
Scatter evaluation results back to tree nodes
"""
function scatter_eval_results!(
    tree::MCTSTreeSoA,
    batch_buffer::EvalBatchBuffer,
    buffer_idx::Int32,
    gid::Int32,
    stride::Int32
)
    # Select appropriate arrays
    node_indices = buffer_idx == 0 ? batch_buffer.node_indices_a : batch_buffer.node_indices_b
    scores = buffer_idx == 0 ? batch_buffer.scores_a : batch_buffer.scores_b
    batch_size = buffer_idx == 0 ? batch_buffer.batch_size_a[1] : batch_buffer.batch_size_b[1]
    
    # Process results in parallel
    for idx in gid:stride:batch_size
        node_idx = @inbounds node_indices[idx]
        score = @inbounds scores[idx]
        
        if node_idx > 0 && node_idx <= MAX_NODES
            # Update node with evaluation result
            @inbounds tree.prior_scores[node_idx] = score
            
            # Mark node as evaluated
            old_state = CUDA.atomic_cas!(
                pointer(tree.node_states, node_idx),
                NODE_ACTIVE,
                NODE_EXPANDED
            )
        end
    end
    
    # Clear ready flag
    @inbounds batch_buffer.ready_flag[buffer_idx + 1] = false
    
    return nothing
end

"""
Batch evaluation pipeline kernel
"""
function batch_eval_pipeline_kernel!(
    tree::MCTSTreeSoA,
    work_queue::WorkQueue,
    batch_buffer::EvalBatchBuffer,
    config::BatchEvalConfig,
    phase::UInt8
)
    tid = threadIdx().x
    bid = blockIdx().x
    gid = tid + (bid - 1) * blockDim().x
    stride = blockDim().x * gridDim().x
    
    if phase == WORK_EVALUATE
        # Collection phase - gather nodes for evaluation
        collect_eval_batch!(tree, work_queue, batch_buffer, config, gid, stride)
        
        # Check if batch is ready (only thread 0)
        if tid == 1 && bid == 1
            buffer_idx = @inbounds batch_buffer.active_buffer[1]
            batch_size_ptr = buffer_idx == 0 ? batch_buffer.batch_size_a : batch_buffer.batch_size_b
            current_size = @inbounds batch_size_ptr[1]
            
            # Dispatch if batch is full or timeout
            if current_size >= config.batch_size || current_size >= compute_dynamic_batch_size(current_size, 0.7f0, config)
                dispatch_eval_batch!(batch_buffer, buffer_idx, current_size)
            end
        end
        
    elseif phase == WORK_BACKUP
        # Check for completed evaluations
        for buffer_idx in 0:1
            if @inbounds batch_buffer.ready_flag[buffer_idx + 1]
                # Scatter results back
                scatter_eval_results!(tree, batch_buffer, buffer_idx, gid, stride)
            end
        end
    end
    
    return nothing
end

"""
Host-side batch evaluation manager
"""
mutable struct BatchEvalManager
    buffer::EvalBatchBuffer
    config::BatchEvalConfig
    eval_stream::CuStream        # Separate stream for NN evaluation
    eval_event::CuEvent          # Synchronization event
    
    function BatchEvalManager(;
        batch_size = 1024,
        num_features = MAX_FEATURES,
        max_actions = 32,
        double_buffer = true,
        eval_threads = 256,
        coalesce_threshold = 0.7f0
    )
        config = BatchEvalConfig(
            batch_size,
            double_buffer,
            eval_threads,
            coalesce_threshold
        )
        
        buffer = EvalBatchBuffer(batch_size, num_features, max_actions)
        eval_stream = CuStream()
        eval_event = CuEvent()
        
        new(buffer, config, eval_stream, eval_event)
    end
end

"""
Process pending evaluation batches (called from host)
"""
function process_eval_batches!(manager::BatchEvalManager, eval_fn::Function)
    # Check both buffers for ready batches
    for buffer_idx in 0:1
        ready = CUDA.@allowscalar manager.buffer.ready_flag[buffer_idx + 1]
        
        if ready
            # Get batch data
            node_indices = buffer_idx == 0 ? manager.buffer.node_indices_a : manager.buffer.node_indices_b
            features = buffer_idx == 0 ? manager.buffer.features_a : manager.buffer.features_b
            scores = buffer_idx == 0 ? manager.buffer.scores_a : manager.buffer.scores_b
            batch_size = CUDA.@allowscalar (buffer_idx == 0 ? manager.buffer.batch_size_a[1] : manager.buffer.batch_size_b[1])
            
            if batch_size > 0
                # Launch evaluation on separate stream
                CUDA.stream!(manager.eval_stream) do
                    # Call user-provided evaluation function
                    eval_fn(features, scores, batch_size)
                    
                    # Record completion event
                    CUDA.record(manager.eval_event)
                end
            end
        end
    end
end

export BatchEvalConfig, EvalBatchBuffer, BatchEvalManager
export collect_eval_batch!, dispatch_eval_batch!, scatter_eval_results!
export batch_eval_pipeline_kernel!, process_eval_batches!
export compute_dynamic_batch_size

end # module