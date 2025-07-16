module MemoryOptimizedKernelsV2

using CUDA
using Statistics

export optimized_variance_kernel_v2!, optimized_mi_kernel_v2!
export optimized_correlation_kernel_v2!, benchmark_memory_patterns_v2

"""
Optimized variance calculation with vectorized loads
"""
function optimized_variance_kernel_v2!(
    variances::CuDeviceVector{Float32},
    X::CuDeviceMatrix{Float32},
    n_features::Int32,
    n_samples::Int32
)
    feat_idx = Int32(blockIdx().x)
    tid = Int32(threadIdx().x)
    block_size = Int32(blockDim().x)
    
    if feat_idx > n_features
        return
    end
    
    # Shared memory for reduction
    shared_sum = @cuDynamicSharedMem(Float32, block_size)
    shared_sum_sq = @cuDynamicSharedMem(Float32, block_size, block_size * sizeof(Float32))
    
    # Initialize local accumulators
    local_sum = 0.0f0
    local_sum_sq = 0.0f0
    
    # Process 4 elements per iteration for better memory throughput
    # Each thread handles multiple chunks
    elements_per_thread = 4
    stride = block_size * elements_per_thread
    start_idx = (tid - 1) * elements_per_thread + 1
    
    # Main loop processing 4 elements at a time
    idx = start_idx
    while idx + 3 <= n_samples
        # Load 4 consecutive values
        val1 = X[feat_idx, idx]
        val2 = X[feat_idx, idx + 1]
        val3 = X[feat_idx, idx + 2]
        val4 = X[feat_idx, idx + 3]
        
        # Accumulate
        local_sum += val1 + val2 + val3 + val4
        local_sum_sq += val1*val1 + val2*val2 + val3*val3 + val4*val4
        
        idx += stride
    end
    
    # Handle remaining elements
    while idx <= n_samples
        val = X[feat_idx, idx]
        local_sum += val
        local_sum_sq += val * val
        idx += 1
    end
    
    # Store in shared memory
    shared_sum[tid] = local_sum
    shared_sum_sq[tid] = local_sum_sq
    sync_threads()
    
    # Reduction in shared memory
    stride_red = block_size ÷ 2
    while stride_red > 32
        if tid <= stride_red
            shared_sum[tid] += shared_sum[tid + stride_red]
            shared_sum_sq[tid] += shared_sum_sq[tid + stride_red]
        end
        sync_threads()
        stride_red ÷= 2
    end
    
    # Final warp reduction without sync
    if tid <= 32
        # Unroll last warp
        if block_size >= 64
            shared_sum[tid] += shared_sum[tid + 32]
            shared_sum_sq[tid] += shared_sum_sq[tid + 32]
        end
        if tid <= 16
            shared_sum[tid] += shared_sum[tid + 16]
            shared_sum_sq[tid] += shared_sum_sq[tid + 16]
        end
        if tid <= 8
            shared_sum[tid] += shared_sum[tid + 8]
            shared_sum_sq[tid] += shared_sum_sq[tid + 8]
        end
        if tid <= 4
            shared_sum[tid] += shared_sum[tid + 4]
            shared_sum_sq[tid] += shared_sum_sq[tid + 4]
        end
        if tid <= 2
            shared_sum[tid] += shared_sum[tid + 2]
            shared_sum_sq[tid] += shared_sum_sq[tid + 2]
        end
        if tid == 1
            shared_sum[tid] += shared_sum[tid + 1]
            shared_sum_sq[tid] += shared_sum_sq[tid + 1]
            
            # Calculate variance
            mean_val = shared_sum[1] / Float32(n_samples)
            var_val = (shared_sum_sq[1] / Float32(n_samples)) - mean_val * mean_val
            variances[feat_idx] = max(var_val, 0.0f0)
        end
    end
    
    return nothing
end

"""
Optimized MI kernel with bank conflict avoidance
"""
function optimized_mi_kernel_v2!(
    mi_scores::CuDeviceVector{Float32},
    X::CuDeviceMatrix{Float32},
    y::CuDeviceVector{Int32},
    n_features::Int32,
    n_samples::Int32,
    n_bins::Int32,
    n_classes::Int32
)
    feat_idx = Int32(blockIdx().x)
    tid = Int32(threadIdx().x)
    block_size = Int32(blockDim().x)
    
    if feat_idx > n_features
        return
    end
    
    # Shared memory for histogram with padding
    hist_size = n_bins * n_classes
    # Add padding to avoid bank conflicts (stride of 33 for 32 banks)
    hist_stride = n_bins + 1
    shared_hist = @cuDynamicSharedMem(Int32, hist_stride * n_classes)
    
    # Initialize histogram
    for i in tid:block_size:hist_stride * n_classes
        shared_hist[i] = Int32(0)
    end
    
    # Shared memory for min/max
    shared_stats = @cuDynamicSharedMem(Float32, block_size * 2, 
                                       hist_stride * n_classes * sizeof(Int32))
    
    sync_threads()
    
    # Find min/max
    local_min = Inf32
    local_max = -Inf32
    
    for idx in tid:block_size:n_samples
        if idx <= n_samples
            val = X[feat_idx, idx]
            local_min = min(local_min, val)
            local_max = max(local_max, val)
        end
    end
    
    # Store for reduction
    shared_stats[tid] = local_min
    shared_stats[block_size + tid] = local_max
    sync_threads()
    
    # Reduce min/max
    stride = block_size ÷ 2
    while stride > 0
        if tid <= stride
            shared_stats[tid] = min(shared_stats[tid], shared_stats[tid + stride])
            shared_stats[block_size + tid] = max(shared_stats[block_size + tid], 
                                                  shared_stats[block_size + tid + stride])
        end
        sync_threads()
        stride ÷= 2
    end
    
    feat_min = shared_stats[1]
    feat_max = shared_stats[block_size + 1]
    
    # Add small epsilon to avoid division by zero
    if feat_max - feat_min < 1e-8
        if tid == 1
            mi_scores[feat_idx] = 0.0f0
        end
        return
    end
    
    bin_width = (feat_max - feat_min) / Float32(n_bins)
    
    sync_threads()
    
    # Build histogram
    for idx in tid:block_size:n_samples
        if idx <= n_samples
            val = X[feat_idx, idx]
            class_idx = y[idx]
            
            # Compute bin
            normalized = (val - feat_min) / bin_width
            bin_idx = min(max(Int32(floor(normalized)) + 1, 1), n_bins)
            
            # Update histogram with strided access
            hist_idx = (class_idx - 1) * hist_stride + bin_idx
            CUDA.@atomic shared_hist[hist_idx] += Int32(1)
        end
    end
    
    sync_threads()
    
    # Calculate MI (single thread)
    if tid == 1
        mi = 0.0f0
        
        # Count samples per class (use local array)
        class_counts = @cuStaticSharedMem(Float32, 10)  # Support up to 10 classes
        
        # Initialize counts
        for c in 1:min(n_classes, 10)
            class_counts[c] = 0.0f0
        end
        
        # Count samples per class
        for c in 1:n_classes
            count = 0.0f0
            for b in 1:n_bins
                count += Float32(shared_hist[(c-1) * hist_stride + b])
            end
            if c <= 10
                class_counts[c] = count
            end
        end
        
        # Calculate MI
        for b in 1:n_bins
            p_x = 0.0f0
            for c in 1:n_classes
                p_x += Float32(shared_hist[(c-1) * hist_stride + b])
            end
            p_x /= Float32(n_samples)
            
            if p_x > 0
                for c in 1:n_classes
                    joint_count = shared_hist[(c-1) * hist_stride + b]
                    if joint_count > 0
                        p_joint = Float32(joint_count) / Float32(n_samples)
                        # For simplicity, only support up to 10 classes
                        if c <= 10
                            p_y = class_counts[c] / Float32(n_samples)
                        else
                            # Fallback: recount for classes > 10
                            class_count = 0.0f0
                            for b2 in 1:n_bins
                                class_count += Float32(shared_hist[(c-1) * hist_stride + b2])
                            end
                            p_y = class_count / Float32(n_samples)
                        end
                        
                        if p_y > 0
                            mi += p_joint * log2(p_joint / (p_x * p_y))
                        end
                    end
                end
            end
        end
        
        mi_scores[feat_idx] = mi
    end
    
    return nothing
end

"""
Optimized tiled correlation kernel
"""
function optimized_correlation_kernel_v2!(
    corr_matrix::CuDeviceMatrix{Float32},
    X_standardized::CuDeviceMatrix{Float32},
    n_features::Int32,
    n_samples::Int32,
    tile_size::Int32
)
    tx = Int32(threadIdx().x)
    ty = Int32(threadIdx().y)
    bx = Int32(blockIdx().x)
    by = Int32(blockIdx().y)
    
    # Global indices
    row = (by - 1) * tile_size + ty
    col = (bx - 1) * tile_size + tx
    
    # Shared memory tiles
    tile_A = @cuDynamicSharedMem(Float32, (tile_size, tile_size))
    tile_B = @cuDynamicSharedMem(Float32, (tile_size, tile_size), 
                                  tile_size * tile_size * sizeof(Float32))
    
    # Accumulator
    sum = 0.0f0
    
    # Number of tiles to process
    n_tiles = cld(n_samples, tile_size)
    
    for tile_idx in 1:n_tiles
        # Calculate sample index
        sample_idx = (tile_idx - 1) * tile_size + tx
        
        # Load tile A (row of X)
        if sample_idx <= n_samples && row <= n_features
            tile_A[ty, tx] = X_standardized[row, sample_idx]
        else
            tile_A[ty, tx] = 0.0f0
        end
        
        # Load tile B (row of X for correlation)
        if sample_idx <= n_samples && col <= n_features
            tile_B[ty, tx] = X_standardized[col, sample_idx]
        else
            tile_B[ty, tx] = 0.0f0
        end
        
        sync_threads()
        
        # Compute partial dot product
        if row <= n_features && col <= n_features
            for k in 1:tile_size
                sum += tile_A[ty, k] * tile_B[ty, k]
            end
        end
        
        sync_threads()
    end
    
    # Write result
    if row <= n_features && col <= n_features
        corr_matrix[row, col] = sum / Float32(n_samples)
    end
    
    return nothing
end

"""
Benchmark memory patterns with improved kernels
"""
function benchmark_memory_patterns_v2(X::CuArray{Float32, 2}, y::CuArray{Int32, 1})
    n_features, n_samples = size(X)
    
    println("Benchmarking Optimized Memory Patterns V2")
    println("Dataset: $n_features features × $n_samples samples")
    println("="^60)
    
    # Test variance with vectorized loads
    println("\n1. Variance Calculation (Vectorized):")
    
    variances = CUDA.zeros(Float32, n_features)
    shared_mem = 2 * 256 * sizeof(Float32)
    
    t_var = @elapsed begin
        @cuda threads=256 blocks=n_features shmem=shared_mem optimized_variance_kernel_v2!(
            variances, X, Int32(n_features), Int32(n_samples)
        )
        CUDA.synchronize()
    end
    
    # Calculate bandwidth
    bytes_read = n_features * n_samples * sizeof(Float32)
    bytes_written = n_features * sizeof(Float32)
    total_bytes = bytes_read + bytes_written
    bandwidth_gb_s = total_bytes / t_var / 1e9
    
    println("  Time: $(round(t_var*1000, digits=2))ms")
    println("  Throughput: $(round(n_features/t_var, digits=0)) features/sec")
    println("  Bandwidth: $(round(bandwidth_gb_s, digits=1)) GB/s")
    
    # Test MI with bank conflict avoidance
    println("\n2. Mutual Information (Bank Conflict Free):")
    
    mi_scores = CUDA.zeros(Float32, n_features)
    n_bins = Int32(10)
    n_classes = Int32(length(unique(Array(y))))
    
    # Calculate shared memory
    hist_stride = n_bins + 1
    hist_mem = hist_stride * n_classes * sizeof(Int32)
    stats_mem = 2 * 256 * sizeof(Float32)
    total_shmem = hist_mem + stats_mem
    
    t_mi = @elapsed begin
        @cuda threads=256 blocks=n_features shmem=total_shmem optimized_mi_kernel_v2!(
            mi_scores, X, y, Int32(n_features), Int32(n_samples), n_bins, n_classes
        )
        CUDA.synchronize()
    end
    
    println("  Time: $(round(t_mi*1000, digits=2))ms")
    println("  Throughput: $(round(n_features/t_mi, digits=0)) features/sec")
    
    # Test tiled correlation
    println("\n3. Correlation Matrix (Tiled):")
    
    # Standardize data
    X_mean = mean(X, dims=2)
    X_std = std(X, dims=2, corrected=false)
    X_standardized = (X .- X_mean) ./ max.(X_std, 1f-8)
    
    corr_matrix = CUDA.zeros(Float32, n_features, n_features)
    tile_size = Int32(16)  # Use smaller tile for better occupancy
    
    threads = (tile_size, tile_size)
    blocks = (cld(n_features, tile_size), cld(n_features, tile_size))
    shmem = 2 * tile_size * tile_size * sizeof(Float32)
    
    t_corr = @elapsed begin
        @cuda threads=threads blocks=blocks shmem=shmem optimized_correlation_kernel_v2!(
            corr_matrix, X_standardized, Int32(n_features), Int32(n_samples), tile_size
        )
        CUDA.synchronize()
    end
    
    gflops = 2 * n_features^2 * n_samples / t_corr / 1e9
    
    println("  Time: $(round(t_corr*1000, digits=2))ms")
    println("  GFLOPS: $(round(gflops, digits=1))")
    
    return Dict(
        "variance_time_ms" => t_var * 1000,
        "variance_bandwidth_gb_s" => bandwidth_gb_s,
        "mi_time_ms" => t_mi * 1000,
        "mi_throughput" => n_features / t_mi,
        "corr_time_ms" => t_corr * 1000,
        "corr_gflops" => gflops
    )
end

end # module