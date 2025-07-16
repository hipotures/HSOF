module ExperienceReplay

using CUDA
using Random
using Statistics
using LinearAlgebra

"""
Configuration for experience replay buffer
"""
struct ReplayConfig
    buffer_size::Int32            # Maximum number of experiences
    max_features::Int32           # Maximum features per combination
    priority_alpha::Float32       # Priority exponent (0 = uniform, 1 = full priority)
    priority_beta::Float32        # Importance sampling exponent
    priority_epsilon::Float32     # Small constant to ensure non-zero priorities
    batch_size::Int32             # Default batch size for sampling
    use_memory_pool::Bool         # Whether to use memory pooling
end

"""
Create default replay configuration
"""
function create_replay_config(;
    buffer_size::Int = 1000,
    max_features::Int = 100,
    priority_alpha::Float32 = 0.6f0,
    priority_beta::Float32 = 0.4f0,
    priority_epsilon::Float32 = 1f-6,
    batch_size::Int = 32,
    use_memory_pool::Bool = true
)
    return ReplayConfig(
        Int32(buffer_size),
        Int32(max_features),
        priority_alpha,
        priority_beta,
        priority_epsilon,
        Int32(batch_size),
        use_memory_pool
    )
end

"""
Experience structure for GPU storage
"""
struct Experience
    feature_indices::CuArray{Int32, 2}    # (max_features, buffer_size) - padded with -1
    n_features::CuArray{Int32, 1}         # (buffer_size,) - actual feature count
    predicted_scores::CuArray{Float32, 1} # (buffer_size,) - metamodel predictions
    actual_scores::CuArray{Float32, 1}    # (buffer_size,) - true scores
    td_errors::CuArray{Float32, 1}        # (buffer_size,) - temporal difference errors
    priorities::CuArray{Float32, 1}       # (buffer_size,) - sampling priorities
    timestamps::CuArray{Int64, 1}         # (buffer_size,) - insertion timestamps
    valid::CuArray{Bool, 1}               # (buffer_size,) - valid entries
end

"""
GPU-accelerated circular replay buffer
"""
mutable struct ReplayBuffer
    config::ReplayConfig
    experience::Experience
    current_idx::CuArray{Int32, 1}       # Single element array for atomic operations
    total_count::CuArray{Int64, 1}       # Total experiences added
    max_priority::CuArray{Float32, 1}    # Maximum priority for new experiences
    sum_tree::CuArray{Float32, 1}        # Sum tree for efficient priority sampling
    memory_pool::Union{Nothing, CuArray} # Optional memory pool
end

"""
Create GPU replay buffer
"""
function create_replay_buffer(config::ReplayConfig = create_replay_config())
    # Allocate GPU arrays
    feature_indices = CUDA.fill(Int32(-1), config.max_features, config.buffer_size)
    n_features = CUDA.zeros(Int32, config.buffer_size)
    predicted_scores = CUDA.zeros(Float32, config.buffer_size)
    actual_scores = CUDA.zeros(Float32, config.buffer_size)
    td_errors = CUDA.zeros(Float32, config.buffer_size)
    priorities = CUDA.ones(Float32, config.buffer_size) * config.priority_epsilon
    timestamps = CUDA.zeros(Int64, config.buffer_size)
    valid = CUDA.zeros(Bool, config.buffer_size)
    
    experience = Experience(
        feature_indices,
        n_features,
        predicted_scores,
        actual_scores,
        td_errors,
        priorities,
        timestamps,
        valid
    )
    
    # Initialize state
    current_idx = CUDA.zeros(Int32, 1)
    total_count = CUDA.zeros(Int64, 1)
    max_priority = CUDA.ones(Float32, 1)
    
    # Sum tree for priority sampling (size = 2 * buffer_size for binary tree)
    sum_tree = CUDA.zeros(Float32, 2 * config.buffer_size)
    
    # Optional memory pool
    memory_pool = config.use_memory_pool ? 
        CUDA.zeros(UInt8, 10 * 1024 * 1024) : # 10MB pool
        nothing
    
    return ReplayBuffer(
        config,
        experience,
        current_idx,
        total_count,
        max_priority,
        sum_tree,
        memory_pool
    )
end

"""
GPU kernel for inserting single experience
"""
function insert_experience_kernel!(
    feature_indices::CuDeviceArray{Int32, 2},
    n_features::CuDeviceArray{Int32, 1},
    predicted_scores::CuDeviceArray{Float32, 1},
    actual_scores::CuDeviceArray{Float32, 1},
    td_errors::CuDeviceArray{Float32, 1},
    priorities::CuDeviceArray{Float32, 1},
    timestamps::CuDeviceArray{Int64, 1},
    valid::CuDeviceArray{Bool, 1},
    current_idx::CuDeviceArray{Int32, 1},
    total_count::CuDeviceArray{Int64, 1},
    max_priority::CuDeviceArray{Float32, 1},
    new_features::CuDeviceArray{Int32, 1},
    new_n_features::Int32,
    new_predicted::Float32,
    new_actual::Float32,
    timestamp::Int64,
    buffer_size::Int32,
    max_features::Int32,
    priority_alpha::Float32,
    priority_epsilon::Float32
)
    # Single thread handles insertion
    if threadIdx().x == 1 && blockIdx().x == 1
        # Get insertion index atomically
        idx = CUDA.atomic_add!(pointer(current_idx, 1), Int32(1)) % buffer_size + 1
        
        # Copy feature indices
        for i in 1:new_n_features
            if i <= max_features
                @inbounds feature_indices[i, idx] = new_features[i]
            end
        end
        
        # Clear remaining features
        for i in (new_n_features + 1):max_features
            @inbounds feature_indices[i, idx] = Int32(-1)
        end
        
        # Store experience data
        @inbounds n_features[idx] = new_n_features
        @inbounds predicted_scores[idx] = new_predicted
        @inbounds actual_scores[idx] = new_actual
        
        # Calculate TD error
        td_error = abs(new_predicted - new_actual)
        @inbounds td_errors[idx] = td_error
        
        # Calculate priority: (|td_error| + epsilon)^alpha
        priority = (td_error + priority_epsilon)^priority_alpha
        @inbounds priorities[idx] = priority
        
        # Update max priority - store directly, will be updated in host code
        @inbounds max_priority[1] = priority
        
        # Store metadata
        @inbounds timestamps[idx] = timestamp
        @inbounds valid[idx] = true
        
        # Increment total count
        CUDA.atomic_add!(pointer(total_count, 1), Int64(1))
    end
    
    return nothing
end

"""
GPU kernel for batch insertion
"""
function batch_insert_kernel!(
    feature_indices::CuDeviceArray{Int32, 2},
    n_features::CuDeviceArray{Int32, 1},
    predicted_scores::CuDeviceArray{Float32, 1},
    actual_scores::CuDeviceArray{Float32, 1},
    td_errors::CuDeviceArray{Float32, 1},
    priorities::CuDeviceArray{Float32, 1},
    timestamps::CuDeviceArray{Int64, 1},
    valid::CuDeviceArray{Bool, 1},
    current_idx::CuDeviceArray{Int32, 1},
    total_count::CuDeviceArray{Int64, 1},
    max_priority::CuDeviceArray{Float32, 1},
    batch_features::CuDeviceArray{Int32, 2},
    batch_n_features::CuDeviceArray{Int32, 1},
    batch_predicted::CuDeviceArray{Float32, 1},
    batch_actual::CuDeviceArray{Float32, 1},
    batch_size::Int32,
    buffer_size::Int32,
    max_features::Int32,
    priority_alpha::Float32,
    priority_epsilon::Float32
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= batch_size
        # Get insertion index for this item - one atomic add per thread
        my_idx = CUDA.atomic_add!(pointer(current_idx, 1), Int32(1))
        idx = (my_idx - 1) % buffer_size + 1
        
        # Get item data
        item_n_features = batch_n_features[tid]
        item_predicted = batch_predicted[tid]
        item_actual = batch_actual[tid]
        
        # Copy features
        for i in 1:min(item_n_features, max_features)
            @inbounds feature_indices[i, idx] = batch_features[i, tid]
        end
        
        # Clear remaining
        for i in (item_n_features + 1):max_features
            @inbounds feature_indices[i, idx] = Int32(-1)
        end
        
        # Store data
        @inbounds n_features[idx] = item_n_features
        @inbounds predicted_scores[idx] = item_predicted
        @inbounds actual_scores[idx] = item_actual
        
        # TD error and priority
        td_error = abs(item_predicted - item_actual)
        @inbounds td_errors[idx] = td_error
        priority = (td_error + priority_epsilon)^priority_alpha
        @inbounds priorities[idx] = priority
        
        # Update max priority - will be handled in host code
        # For now just store locally
        
        # Metadata
        @inbounds timestamps[idx] = Int64(tid) * 1000  # Use thread ID as placeholder
        @inbounds valid[idx] = true
        
        # Count - each thread increments by 1
        CUDA.atomic_add!(pointer(total_count, 1), Int64(1))
    end
    
    return nothing
end

"""
GPU kernel for updating sum tree
"""
function update_sum_tree_kernel!(
    sum_tree::CuDeviceArray{Float32, 1},
    priorities::CuDeviceArray{Float32, 1},
    valid::CuDeviceArray{Bool, 1},
    buffer_size::Int32
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= buffer_size
        # Leaf nodes start at buffer_size
        leaf_idx = buffer_size + tid - 1
        
        # Update leaf with priority if valid
        if valid[tid]
            @inbounds sum_tree[leaf_idx + 1] = priorities[tid]
        else
            @inbounds sum_tree[leaf_idx + 1] = 0.0f0
        end
        
        # Propagate up the tree
        idx = leaf_idx + 1
        while idx > 1
            parent_idx = idx ÷ 2
            
            # Atomically add to parent
            left_child = 2 * parent_idx - 1
            right_child = 2 * parent_idx
            
            if idx == left_child + 1
                # We're the left child
                sibling_val = idx < 2 * buffer_size ? sum_tree[right_child + 1] : 0.0f0
            else
                # We're the right child
                sibling_val = sum_tree[left_child + 1]
            end
            
            # Update parent
            CUDA.atomic_add!(pointer(sum_tree, parent_idx), priorities[tid])
            
            idx = parent_idx
        end
    end
    
    return nothing
end

"""
Insert single experience into buffer
"""
function insert_experience!(
    buffer::ReplayBuffer,
    feature_indices::Vector{Int32},
    predicted_score::Float32,
    actual_score::Float32;
    timestamp::Int64 = round(Int64, time() * 1000)
)
    # Transfer to GPU
    gpu_features = CuArray(feature_indices)
    
    # Launch insertion kernel
    @cuda threads=1 blocks=1 insert_experience_kernel!(
        buffer.experience.feature_indices,
        buffer.experience.n_features,
        buffer.experience.predicted_scores,
        buffer.experience.actual_scores,
        buffer.experience.td_errors,
        buffer.experience.priorities,
        buffer.experience.timestamps,
        buffer.experience.valid,
        buffer.current_idx,
        buffer.total_count,
        buffer.max_priority,
        gpu_features,
        Int32(length(feature_indices)),
        predicted_score,
        actual_score,
        timestamp,
        buffer.config.buffer_size,
        buffer.config.max_features,
        buffer.config.priority_alpha,
        buffer.config.priority_epsilon
    )
    
    CUDA.synchronize()
    
    # Update max priority on host
    current_max = CUDA.@allowscalar buffer.max_priority[1]
    td_error = abs(predicted_score - actual_score)
    new_priority = (td_error + buffer.config.priority_epsilon)^buffer.config.priority_alpha
    CUDA.@allowscalar buffer.max_priority[1] = max(current_max, new_priority)
    
    # Don't update sum tree here - let caller decide when to update
end

"""
Batch insert experiences
"""
function batch_insert!(
    buffer::ReplayBuffer,
    feature_indices_batch::Matrix{Int32},
    predicted_scores::Vector{Float32},
    actual_scores::Vector{Float32}
)
    batch_size = length(predicted_scores)
    
    # Insert experiences one by one using the single insertion method
    for i in 1:batch_size
        features = feature_indices_batch[:, i]
        # Count non-zero features
        n_features = count(x -> x > 0, features)
        if n_features > 0
            # Trim to actual features
            actual_features = features[1:n_features]
            insert_experience!(
                buffer,
                actual_features,
                predicted_scores[i],
                actual_scores[i]
            )
        end
    end
    
    # Update sum tree once at the end
    update_sum_tree!(buffer)
end

"""
Update sum tree for priority sampling
"""
function update_sum_tree!(buffer::ReplayBuffer)
    threads = 256
    blocks = cld(buffer.config.buffer_size, threads)
    
    @cuda threads=threads blocks=blocks update_sum_tree_kernel!(
        buffer.sum_tree,
        buffer.experience.priorities,
        buffer.experience.valid,
        buffer.config.buffer_size
    )
    
    CUDA.synchronize()
end

"""
GPU kernel for priority sampling
"""
function sample_indices_kernel!(
    sampled_indices::CuDeviceArray{Int32, 1},
    sampled_weights::CuDeviceArray{Float32, 1},
    sum_tree::CuDeviceArray{Float32, 1},
    priorities::CuDeviceArray{Float32, 1},
    valid::CuDeviceArray{Bool, 1},
    random_values::CuDeviceArray{Float32, 1},
    batch_size::Int32,
    buffer_size::Int32,
    total_priority::Float32,
    priority_beta::Float32
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= batch_size
        # Sample from sum tree
        target_sum = random_values[tid] * total_priority
        
        # Binary search in sum tree
        idx = 1
        while idx < buffer_size
            left_child = 2 * idx
            left_sum = sum_tree[left_child]
            
            if target_sum <= left_sum
                idx = left_child
            else
                target_sum -= left_sum
                idx = left_child + 1
            end
        end
        
        # Get actual buffer index
        buffer_idx = idx - buffer_size + 1
        
        # Store sampled index
        @inbounds sampled_indices[tid] = buffer_idx
        
        # Calculate importance sampling weight
        if valid[buffer_idx] && priorities[buffer_idx] > 0
            # w_i = (1/N * 1/P(i))^beta
            prob = priorities[buffer_idx] / total_priority
            weight = (1.0f0 / (Float32(buffer_size) * prob))^priority_beta
            @inbounds sampled_weights[tid] = weight
        else
            @inbounds sampled_weights[tid] = 0.0f0
        end
    end
    
    return nothing
end

"""
Sample batch of experiences with priority
"""
function sample_batch(
    buffer::ReplayBuffer,
    batch_size::Int = Int(buffer.config.batch_size)
)
    # Get total priority
    total_priority = CUDA.@allowscalar buffer.sum_tree[1]
    
    if total_priority == 0
        # No valid experiences
        return nothing
    end
    
    # Generate random values
    random_values = CUDA.rand(Float32, batch_size)
    
    # Allocate output
    sampled_indices = CUDA.zeros(Int32, batch_size)
    sampled_weights = CUDA.zeros(Float32, batch_size)
    
    # Sample indices
    threads = 256
    blocks = cld(batch_size, threads)
    
    @cuda threads=threads blocks=blocks sample_indices_kernel!(
        sampled_indices,
        sampled_weights,
        buffer.sum_tree,
        buffer.experience.priorities,
        buffer.experience.valid,
        random_values,
        Int32(batch_size),
        buffer.config.buffer_size,
        total_priority,
        buffer.config.priority_beta
    )
    
    CUDA.synchronize()
    
    # Filter out invalid samples (with zero weights)
    valid_mask = sampled_weights .> 0
    n_valid = sum(valid_mask)
    
    if n_valid == 0
        return nothing
    end
    
    # Extract valid indices
    valid_indices = sampled_indices[valid_mask]
    valid_weights = sampled_weights[valid_mask]
    
    # Normalize weights
    max_weight = maximum(valid_weights)
    valid_weights ./= max_weight
    
    # Extract experiences
    indices_cpu = Array(valid_indices)
    
    # Gather data
    features = buffer.experience.feature_indices[:, indices_cpu]
    n_features = buffer.experience.n_features[indices_cpu]
    predicted = buffer.experience.predicted_scores[indices_cpu]
    actual = buffer.experience.actual_scores[indices_cpu]
    
    return (
        indices = valid_indices,
        features = features,
        n_features = n_features,
        predicted = predicted,
        actual = actual,
        weights = valid_weights
    )
end

"""
Update priorities after training
"""
function update_priorities!(
    buffer::ReplayBuffer,
    indices::CuArray{Int32, 1},
    new_td_errors::CuArray{Float32, 1}
)
    # Update TD errors and priorities
    indices_cpu = Array(indices)
    td_errors_cpu = Array(new_td_errors)
    
    for (i, idx) in enumerate(indices_cpu)
        if 1 <= idx <= buffer.config.buffer_size
            td_error = td_errors_cpu[i]
            priority = (abs(td_error) + buffer.config.priority_epsilon)^buffer.config.priority_alpha
            
            CUDA.@allowscalar buffer.experience.td_errors[idx] = td_error
            CUDA.@allowscalar buffer.experience.priorities[idx] = priority
            
            # Update max priority
            current_max = CUDA.@allowscalar buffer.max_priority[1]
            CUDA.@allowscalar buffer.max_priority[1] = max(current_max, priority)
        end
    end
    
    # Update sum tree
    update_sum_tree!(buffer)
end

"""
Get buffer statistics
"""
function get_buffer_stats(buffer::ReplayBuffer)
    n_valid = sum(buffer.experience.valid)
    total_count = CUDA.@allowscalar buffer.total_count[1]
    current_idx = CUDA.@allowscalar buffer.current_idx[1]
    max_priority = CUDA.@allowscalar buffer.max_priority[1]
    
    if n_valid > 0
        avg_td_error = mean(buffer.experience.td_errors[buffer.experience.valid])
        avg_priority = mean(buffer.experience.priorities[buffer.experience.valid])
    else
        avg_td_error = 0.0f0
        avg_priority = 0.0f0
    end
    
    return (
        n_valid = n_valid,
        total_count = total_count,
        current_idx = current_idx,
        buffer_utilization = n_valid / buffer.config.buffer_size,
        avg_td_error = avg_td_error,
        avg_priority = avg_priority,
        max_priority = max_priority
    )
end

"""
Clear buffer
"""
function clear_buffer!(buffer::ReplayBuffer)
    buffer.experience.valid .= false
    buffer.experience.priorities .= buffer.config.priority_epsilon
    buffer.sum_tree .= 0.0f0
    CUDA.@allowscalar buffer.current_idx[1] = 0
    CUDA.@allowscalar buffer.total_count[1] = 0
    CUDA.@allowscalar buffer.max_priority[1] = 1.0f0
end

# Export types and functions
export ReplayConfig, create_replay_config
export ReplayBuffer, create_replay_buffer
export insert_experience!, batch_insert!
export sample_batch, update_priorities!
export get_buffer_stats, clear_buffer!
export update_sum_tree!

end # module ExperienceReplay