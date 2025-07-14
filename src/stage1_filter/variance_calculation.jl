module VarianceCalculation

using CUDA
using Statistics

# Include the GPUMemoryLayout module
include("gpu_memory_layout.jl")
using .GPUMemoryLayout: VarianceBuffers, FeatureMatrix, WARP_SIZE, TILE_SIZE

"""
Configuration for variance calculation
"""
struct VarianceConfig
    n_features::Int32
    n_samples::Int32
    use_welford::Bool          # Use Welford's algorithm for numerical stability
    block_size::Int32          # Thread block size
    epsilon::Float32           # Small value for numerical stability
end

"""
Create default variance configuration
"""
function create_variance_config(n_features::Integer, n_samples::Integer;
                               use_welford::Bool = true,
                               block_size::Integer = 256,
                               epsilon::Float32 = Float32(1e-10))
    return VarianceConfig(
        Int32(n_features),
        Int32(n_samples),
        use_welford,
        Int32(block_size),
        epsilon
    )
end

"""
GPU kernel for computing feature variances using Welford's algorithm (single-pass)
"""
function compute_variance_welford_kernel!(
    variances::CuDeviceArray{Float32, 1},
    means::CuDeviceArray{Float32, 1},
    feature_data::CuDeviceArray{Float32, 2},
    n_samples::Int32,
    n_features::Int32,
    epsilon::Float32
)
    feature_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if feature_idx <= n_features
        # Welford's online algorithm for variance
        mean = Float32(0)
        m2 = Float32(0)
        count = Int32(0)
        
        for sample_idx in 1:n_samples
            count += 1
            value = feature_data[sample_idx, feature_idx]
            delta = value - mean
            mean += delta / Float32(count)
            delta2 = value - mean
            m2 += delta * delta2
        end
        
        # Store results
        means[feature_idx] = mean
        if count > 1
            variances[feature_idx] = m2 / Float32(count - 1) + epsilon
        else
            variances[feature_idx] = epsilon
        end
    end
    
    return nothing
end

"""
GPU kernel for parallel reduction within blocks (first stage)
"""
function reduce_variance_block_kernel!(
    block_sums::CuDeviceArray{Float32, 2},
    block_counts::CuDeviceArray{Int32, 2},
    block_m2::CuDeviceArray{Float32, 2},
    feature_data::CuDeviceArray{Float32, 2},
    n_samples::Int32,
    n_features::Int32,
    samples_per_block::Int32
)
    # Shared memory for reduction
    shared_sum = @cuDynamicSharedMem(Float32, blockDim().x)
    shared_m2 = @cuDynamicSharedMem(Float32, blockDim().x, blockDim().x * sizeof(Float32))
    
    tid = threadIdx().x
    feature_idx = blockIdx().y
    block_idx = blockIdx().x
    
    if feature_idx <= n_features
        # Calculate sample range for this block
        start_sample = (block_idx - 1) * samples_per_block + 1
        end_sample = min(block_idx * samples_per_block, n_samples)
        
        # Thread-local accumulation
        local_sum = Float32(0)
        local_m2 = Float32(0)
        local_count = Int32(0)
        
        # Each thread processes multiple samples
        sample_idx = start_sample + tid - 1
        while sample_idx <= end_sample
            if sample_idx <= n_samples
                value = feature_data[sample_idx, feature_idx]
                local_sum += value
                local_count += 1
            end
            sample_idx += blockDim().x
        end
        
        # Store in shared memory
        shared_sum[tid] = local_sum
        sync_threads()
        
        # Parallel reduction in shared memory
        stride = blockDim().x ÷ 2
        while stride > 0
            if tid <= stride && tid + stride <= blockDim().x
                shared_sum[tid] += shared_sum[tid + stride]
            end
            sync_threads()
            stride ÷= 2
        end
        
        # First thread stores block result
        if tid == 1
            block_sums[block_idx, feature_idx] = shared_sum[1]
            block_counts[block_idx, feature_idx] = local_count
            
            # For m2, we need mean first, so store raw sum for now
            block_m2[block_idx, feature_idx] = Float32(0)  # Placeholder
        end
    end
    
    return nothing
end

"""
GPU kernel for computing variance using warp shuffle operations
"""
function compute_variance_warp_shuffle_kernel!(
    variances::CuDeviceArray{Float32, 1},
    means::CuDeviceArray{Float32, 1},
    feature_data::CuDeviceArray{Float32, 2},
    n_samples::Int32,
    n_features::Int32,
    epsilon::Float32
)
    feature_idx = blockIdx().x
    tid = threadIdx().x
    warp_id = tid ÷ WARP_SIZE
    lane_id = tid % WARP_SIZE
    
    # Shared memory for warp reductions
    shared_sums = @cuDynamicSharedMem(Float32, blockDim().x ÷ WARP_SIZE)
    shared_counts = @cuDynamicSharedMem(Int32, blockDim().x ÷ WARP_SIZE, 
                                       (blockDim().x ÷ WARP_SIZE) * sizeof(Float32))
    
    if feature_idx <= n_features
        # Each thread accumulates its portion
        local_sum = Float32(0)
        local_count = Int32(0)
        
        sample_idx = tid
        while sample_idx <= n_samples
            local_sum += feature_data[sample_idx, feature_idx]
            local_count += 1
            sample_idx += blockDim().x
        end
        
        # Warp-level reduction using shuffle
        for offset in (16, 8, 4, 2, 1)
            local_sum += shfl_down_sync(0xffffffff, local_sum, offset)
            local_count += shfl_down_sync(0xffffffff, local_count, offset)
        end
        
        # First thread in warp stores result
        if lane_id == 0
            shared_sums[warp_id + 1] = local_sum
            shared_counts[warp_id + 1] = local_count
        end
        
        sync_threads()
        
        # Final reduction across warps
        if tid == 1
            total_sum = Float32(0)
            total_count = Int32(0)
            num_warps = blockDim().x ÷ WARP_SIZE
            
            for i in 1:num_warps
                total_sum += shared_sums[i]
                total_count += shared_counts[i]
            end
            
            # Compute mean
            mean = total_sum / Float32(total_count)
            means[feature_idx] = mean
        end
        
        sync_threads()
        
        # Second pass for variance
        mean = means[feature_idx]
        local_m2 = Float32(0)
        
        sample_idx = tid
        while sample_idx <= n_samples
            diff = feature_data[sample_idx, feature_idx] - mean
            local_m2 += diff * diff
            sample_idx += blockDim().x
        end
        
        # Warp-level reduction for m2
        for offset in (16, 8, 4, 2, 1)
            local_m2 += shfl_down_sync(0xffffffff, local_m2, offset)
        end
        
        if lane_id == 0
            shared_sums[warp_id + 1] = local_m2  # Reuse shared memory
        end
        
        sync_threads()
        
        # Final reduction
        if tid == 1
            total_m2 = Float32(0)
            num_warps = blockDim().x ÷ WARP_SIZE
            
            for i in 1:num_warps
                total_m2 += shared_sums[i]
            end
            
            # Store variance
            if n_samples > 1
                variances[feature_idx] = total_m2 / Float32(n_samples - 1) + epsilon
            else
                variances[feature_idx] = epsilon
            end
        end
    end
    
    return nothing
end

"""
GPU kernel for simple two-pass variance calculation
"""
function compute_variance_twopass_kernel!(
    variances::CuDeviceArray{Float32, 1},
    means::CuDeviceArray{Float32, 1},
    feature_data::CuDeviceArray{Float32, 2},
    n_samples::Int32,
    n_features::Int32,
    epsilon::Float32
)
    feature_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if feature_idx <= n_features
        # First pass: compute mean
        sum = Float32(0)
        for sample_idx in 1:n_samples
            sum += feature_data[sample_idx, feature_idx]
        end
        mean = sum / Float32(n_samples)
        means[feature_idx] = mean
        
        # Second pass: compute variance
        sum_sq = Float32(0)
        for sample_idx in 1:n_samples
            diff = feature_data[sample_idx, feature_idx] - mean
            sum_sq += diff * diff
        end
        
        if n_samples > 1
            variances[feature_idx] = sum_sq / Float32(n_samples - 1) + epsilon
        else
            variances[feature_idx] = epsilon
        end
    end
    
    return nothing
end

"""
Compute feature variances using the configured method
"""
function compute_variances!(
    variance_buffers::VarianceBuffers,
    feature_matrix::FeatureMatrix,
    config::VarianceConfig
)
    n_features = config.n_features
    n_samples = config.n_samples
    
    if config.use_welford
        # Use Welford's algorithm for numerical stability
        threads = min(config.block_size, n_features)
        blocks = cld(n_features, threads)
        
        @cuda threads=threads blocks=blocks compute_variance_welford_kernel!(
            variance_buffers.variances,
            variance_buffers.means,
            feature_matrix.data,
            n_samples,
            n_features,
            config.epsilon
        )
    else
        # Use simple two-pass algorithm
        threads = min(config.block_size, n_features)
        blocks = cld(n_features, threads)
        
        @cuda threads=threads blocks=blocks compute_variance_twopass_kernel!(
            variance_buffers.variances,
            variance_buffers.means,
            feature_matrix.data,
            n_samples,
            n_features,
            config.epsilon
        )
    end
    
    CUDA.synchronize()
end

"""
Compute variances using warp shuffle operations (optimized for newer GPUs)
"""
function compute_variances_warp_shuffle!(
    variance_buffers::VarianceBuffers,
    feature_matrix::FeatureMatrix,
    config::VarianceConfig
)
    n_features = config.n_features
    n_samples = config.n_samples
    
    # Use one block per feature with multiple warps
    threads = config.block_size
    blocks = n_features
    
    # Calculate shared memory size
    warps_per_block = threads ÷ WARP_SIZE
    shmem_size = warps_per_block * (sizeof(Float32) + sizeof(Int32))
    
    @cuda threads=threads blocks=blocks shmem=shmem_size compute_variance_warp_shuffle_kernel!(
        variance_buffers.variances,
        variance_buffers.means,
        feature_matrix.data,
        n_samples,
        n_features,
        config.epsilon
    )
    
    CUDA.synchronize()
end

"""
Find features with low variance (near-constant features)
"""
function find_low_variance_features(
    variance_buffers::VarianceBuffers,
    threshold::Float32 = Float32(1e-6)
)
    variances_cpu = Array(variance_buffers.variances)
    low_var_indices = Int32[]
    
    for (idx, var) in enumerate(variances_cpu)
        if var < threshold
            push!(low_var_indices, Int32(idx))
        end
    end
    
    return low_var_indices
end

"""
GPU kernel to mark low variance features
"""
function mark_low_variance_kernel!(
    low_var_mask::CuDeviceArray{Bool, 1},
    variances::CuDeviceArray{Float32, 1},
    threshold::Float32,
    n_features::Int32
)
    feature_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if feature_idx <= n_features
        low_var_mask[feature_idx] = variances[feature_idx] < threshold
    end
    
    return nothing
end

"""
Mark features with variance below threshold
"""
function mark_low_variance_features!(
    low_var_mask::CuArray{Bool, 1},
    variance_buffers::VarianceBuffers,
    threshold::Float32 = Float32(1e-6)
)
    n_features = length(variance_buffers.variances)
    
    threads = 256
    blocks = cld(n_features, threads)
    
    @cuda threads=threads blocks=blocks mark_low_variance_kernel!(
        low_var_mask,
        variance_buffers.variances,
        threshold,
        Int32(n_features)
    )
    
    CUDA.synchronize()
end

"""
Update variances online for streaming data
"""
function update_variances_online!(
    variance_buffers::VarianceBuffers,
    new_data::CuArray{Float32, 2},
    config::VarianceConfig
)
    # TODO: Implement online variance update using Welford's algorithm
    # This would update existing statistics with new batch of data
    @warn "Online variance update not yet implemented"
end

# Export functions
export VarianceConfig, create_variance_config
export compute_variances!, compute_variances_warp_shuffle!
export find_low_variance_features, mark_low_variance_features!
export update_variances_online!

end # module VarianceCalculation