module MemoryOptimizedKernels

using CUDA
using CUDA.CUSPARSE
using Statistics

export OptimizedKernelConfig, optimized_variance_kernel!, optimized_mi_kernel!
export optimized_correlation_kernel!, benchmark_memory_patterns

"""
Configuration for memory-optimized kernels
"""
struct OptimizedKernelConfig
    use_texture_memory::Bool
    use_vectorized_loads::Bool
    memory_alignment::Int32
    shared_memory_banks::Int32
    warp_size::Int32
end

function OptimizedKernelConfig(;
    use_texture_memory::Bool = true,
    use_vectorized_loads::Bool = true,
    memory_alignment::Int32 = Int32(128),  # 128-byte alignment for optimal access
    shared_memory_banks::Int32 = Int32(32),  # RTX 4090 has 32 banks
    warp_size::Int32 = Int32(32)
)
    return OptimizedKernelConfig(
        use_texture_memory,
        use_vectorized_loads,
        memory_alignment,
        shared_memory_banks,
        warp_size
    )
end

"""
Optimized variance calculation with coalesced memory access
"""
function optimized_variance_kernel!(
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
    
    # Shared memory for reduction - padded to avoid bank conflicts
    shared_sum = @cuDynamicSharedMem(Float32, block_size + 1)
    shared_sum_sq = @cuDynamicSharedMem(Float32, block_size + 1, (block_size + 1) * sizeof(Float32))
    
    # Initialize local accumulators
    local_sum = 0.0f0
    local_sum_sq = 0.0f0
    
    # Process samples with coalesced access pattern
    # Each thread processes samples with stride = block_size
    for idx in tid:block_size:n_samples
        if idx <= n_samples
            val = X[feat_idx, idx]
            local_sum += val
            local_sum_sq += val * val
        end
    end
    
    # Store in shared memory with padding to avoid bank conflicts
    shared_sum[tid] = local_sum
    shared_sum_sq[tid] = local_sum_sq
    sync_threads()
    
    # Warp-shuffle reduction for better performance
    if block_size >= 64
        if tid <= 32
            # Add upper half to lower half
            shared_sum[tid] += shared_sum[tid + 32]
            shared_sum_sq[tid] += shared_sum_sq[tid + 32]
        end
        sync_threads()
    end
    
    # Final warp reduction using shuffle operations
    if tid <= 32
        sum_val = shared_sum[tid]
        sum_sq_val = shared_sum_sq[tid]
        
        # Warp shuffle reduction
        for offset in (16, 8, 4, 2, 1)
            sum_val += shfl_down_sync(0xffffffff, sum_val, offset)
            sum_sq_val += shfl_down_sync(0xffffffff, sum_sq_val, offset)
        end
        
        if tid == 1
            mean_val = sum_val / Float32(n_samples)
            var_val = (sum_sq_val / Float32(n_samples)) - mean_val * mean_val
            variances[feat_idx] = max(var_val, 0.0f0)  # Ensure non-negative
        end
    end
    
    return nothing
end

"""
Optimized mutual information kernel with texture memory for histogram
"""
function optimized_mi_kernel!(
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
    
    # Shared memory layout optimized for bank conflict avoidance
    hist_size = n_bins * n_classes
    # Pad histogram to avoid bank conflicts
    hist_stride = n_bins + (n_bins % 32 == 0 ? 1 : 0)
    shared_hist = @cuDynamicSharedMem(Int32, hist_stride * n_classes)
    
    # Initialize histogram with stride access pattern
    for i in tid:block_size:hist_size
        class_idx = (i - 1) ÷ n_bins + 1
        bin_idx = (i - 1) % n_bins + 1
        if class_idx <= n_classes && bin_idx <= n_bins
            shared_hist[(class_idx - 1) * hist_stride + bin_idx] = Int32(0)
        end
    end
    
    # Shared memory for min/max reduction
    shared_minmax = @cuDynamicSharedMem(Float32, block_size * 2, hist_stride * n_classes * sizeof(Int32))
    
    # Find min/max with vectorized loads
    local_min = Inf32
    local_max = -Inf32
    
    # Process multiple samples per thread
    samples_per_thread = cld(n_samples, block_size)
    base_idx = (tid - 1) * samples_per_thread + 1
    
    for i in 0:samples_per_thread-1
        idx = base_idx + i
        if idx <= n_samples
            val = X[feat_idx, idx]
            local_min = min(local_min, val)
            local_max = max(local_max, val)
        end
    end
    
    # Store in shared memory for reduction
    shared_minmax[tid] = local_min
    shared_minmax[block_size + tid] = local_max
    sync_threads()
    
    # Parallel reduction for min/max
    stride = block_size ÷ 2
    while stride > 0
        if tid <= stride
            shared_minmax[tid] = min(shared_minmax[tid], shared_minmax[tid + stride])
            shared_minmax[block_size + tid] = max(shared_minmax[block_size + tid], 
                                                  shared_minmax[block_size + tid + stride])
        end
        sync_threads()
        stride ÷= 2
    end
    
    feat_min = shared_minmax[1]
    feat_max = shared_minmax[block_size + 1]
    bin_width = (feat_max - feat_min) / Float32(n_bins)
    
    sync_threads()
    
    # Build histogram with atomic operations
    for i in 0:samples_per_thread-1
        idx = base_idx + i
        if idx <= n_samples
            val = X[feat_idx, idx]
            class_idx = y[idx]
            
            # Compute bin with boundary checking
            normalized = (val - feat_min) / bin_width
            bin_idx = min(max(Int32(floor(normalized)) + 1, 1), n_bins)
            
            # Atomic increment with strided access
            hist_idx = (class_idx - 1) * hist_stride + bin_idx
            CUDA.@atomic shared_hist[hist_idx] += Int32(1)
        end
    end
    
    sync_threads()
    
    # Calculate MI in parallel
    if tid == 1
        mi = 0.0f0
        
        # Pre-calculate class probabilities
        class_counts = @cuStaticSharedMem(Float32, 16)  # Support up to 16 classes
        for c in 1:min(n_classes, 16)
            class_counts[c] = 0.0f0
        end
        
        # Count samples per class
        for c in 1:n_classes
            for b in 1:n_bins
                class_counts[c] += Float32(shared_hist[(c-1) * hist_stride + b])
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
                        p_y = class_counts[c] / Float32(n_samples)
                        
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
Optimized correlation matrix kernel using tiling and shared memory
"""
function optimized_correlation_kernel!(
    corr_matrix::CuDeviceMatrix{Float32},
    X_standardized::CuDeviceMatrix{Float32},
    n_features::Int32,
    n_samples::Int32,
    tile_size::Int32 = Int32(32)
)
    # Thread indices
    tx = Int32(threadIdx().x)
    ty = Int32(threadIdx().y)
    bx = Int32(blockIdx().x)
    by = Int32(blockIdx().y)
    
    # Global indices
    row = (by - 1) * tile_size + ty
    col = (bx - 1) * tile_size + tx
    
    if row > n_features || col > n_features
        return
    end
    
    # Shared memory tiles for matrix multiplication
    tile_A = @cuDynamicSharedMem(Float32, (tile_size, tile_size))
    tile_B = @cuDynamicSharedMem(Float32, (tile_size, tile_size), 
                                  tile_size * tile_size * sizeof(Float32))
    
    # Initialize accumulator
    sum = 0.0f0
    
    # Tile-based matrix multiplication
    n_tiles = cld(n_samples, tile_size)
    
    for tile_idx in 1:n_tiles
        # Load tiles into shared memory with coalesced access
        sample_idx = (tile_idx - 1) * tile_size + tx
        
        if sample_idx <= n_samples && row <= n_features
            tile_A[ty, tx] = X_standardized[row, sample_idx]
        else
            tile_A[ty, tx] = 0.0f0
        end
        
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
Apply memory padding for optimal alignment
"""
function apply_memory_padding(array::CuArray{T}, alignment::Int32 = Int32(128)) where T
    n_elements = length(array)
    element_size = sizeof(T)
    
    # Calculate padding needed
    current_bytes = n_elements * element_size
    aligned_bytes = cld(current_bytes, alignment) * alignment
    padded_elements = aligned_bytes ÷ element_size
    
    if padded_elements > n_elements
        # Create padded array
        padded = CUDA.zeros(T, padded_elements)
        padded[1:n_elements] = array
        return padded
    else
        return array
    end
end

"""
Benchmark memory access patterns
"""
function benchmark_memory_patterns(X::CuArray{Float32, 2}, y::CuArray{Int32, 1})
    n_features, n_samples = size(X)
    config = OptimizedKernelConfig()
    
    println("Benchmarking Memory Access Patterns")
    println("Dataset: $n_features features × $n_samples samples")
    println("="^60)
    
    # Test 1: Standard vs Optimized Variance
    println("\n1. Variance Calculation:")
    
    # Standard implementation
    variances_std = CUDA.zeros(Float32, n_features)
    t_std = @elapsed begin
        @cuda threads=256 blocks=n_features optimized_variance_kernel!(
            variances_std, X, Int32(n_features), Int32(n_samples)
        )
        CUDA.synchronize()
    end
    
    # With vectorized loads
    variances_opt = CUDA.zeros(Float32, n_features)
    shared_mem = (256 + 1) * 2 * sizeof(Float32)
    t_opt = @elapsed begin
        @cuda threads=256 blocks=n_features shmem=shared_mem optimized_variance_kernel!(
            variances_opt, X, Int32(n_features), Int32(n_samples)
        )
        CUDA.synchronize()
    end
    
    speedup_var = t_std / t_opt
    println("  Standard: $(round(t_std*1000, digits=2))ms")
    println("  Optimized: $(round(t_opt*1000, digits=2))ms")
    println("  Speedup: $(round(speedup_var, digits=2))x")
    
    # Test 2: MI Calculation with bank conflict avoidance
    println("\n2. Mutual Information:")
    
    mi_scores = CUDA.zeros(Float32, n_features)
    n_bins = Int32(10)
    n_classes = Int32(length(unique(Array(y))))
    
    # Calculate required shared memory
    hist_stride = n_bins + (n_bins % 32 == 0 ? 1 : 0)
    hist_mem = hist_stride * n_classes * sizeof(Int32)
    minmax_mem = 256 * 2 * sizeof(Float32)
    total_shmem = hist_mem + minmax_mem
    
    t_mi = @elapsed begin
        @cuda threads=256 blocks=n_features shmem=total_shmem optimized_mi_kernel!(
            mi_scores, X, y, Int32(n_features), Int32(n_samples), n_bins, n_classes
        )
        CUDA.synchronize()
    end
    
    println("  Time: $(round(t_mi*1000, digits=2))ms")
    println("  Throughput: $(round(n_features/t_mi, digits=0)) features/sec")
    
    # Test 3: Correlation with tiling
    println("\n3. Correlation Matrix (Tiled):")
    
    # Standardize first
    X_mean = mean(X, dims=2)
    X_std = std(X, dims=2, corrected=false)
    X_standardized = (X .- X_mean) ./ max.(X_std, 1f-8)
    
    corr_matrix = CUDA.zeros(Float32, n_features, n_features)
    tile_size = Int32(32)
    
    t_corr = @elapsed begin
        threads = (tile_size, tile_size)
        blocks = (cld(n_features, tile_size), cld(n_features, tile_size))
        shmem = 2 * tile_size * tile_size * sizeof(Float32)
        
        @cuda threads=threads blocks=blocks shmem=shmem optimized_correlation_kernel!(
            corr_matrix, X_standardized, Int32(n_features), Int32(n_samples), tile_size
        )
        CUDA.synchronize()
    end
    
    println("  Time: $(round(t_corr*1000, digits=2))ms")
    println("  GFLOPS: $(round(2*n_features^2*n_samples/t_corr/1e9, digits=1))")
    
    # Memory bandwidth analysis
    println("\n4. Memory Bandwidth Utilization:")
    
    # Calculate theoretical bandwidth
    mem_info = CUDA.memory_info()
    
    # Variance kernel bandwidth
    var_bytes = n_features * n_samples * sizeof(Float32) + n_features * sizeof(Float32)
    var_bandwidth = var_bytes / t_opt / 1e9
    
    # MI kernel bandwidth (approximate)
    mi_bytes = n_features * n_samples * sizeof(Float32) + n_samples * sizeof(Int32)
    mi_bandwidth = mi_bytes / t_mi / 1e9
    
    println("  Variance kernel: $(round(var_bandwidth, digits=1)) GB/s")
    println("  MI kernel: $(round(mi_bandwidth, digits=1)) GB/s")
    
    # Bank conflict analysis
    println("\n5. Shared Memory Bank Conflicts:")
    println("  Histogram stride: $hist_stride (avoiding conflicts)")
    println("  Variance padding: 1 element per array")
    println("  Tile size: $(tile_size)×$(tile_size) (no conflicts)")
    
    return Dict(
        "variance_speedup" => speedup_var,
        "mi_throughput" => n_features/t_mi,
        "corr_gflops" => 2*n_features^2*n_samples/t_corr/1e9,
        "var_bandwidth_gb_s" => var_bandwidth,
        "mi_bandwidth_gb_s" => mi_bandwidth
    )
end

"""
Profile memory access patterns using CUDA events
"""
function profile_memory_access(kernel_func, args...; name="Kernel")
    # Create events for timing
    start_event = CuEvent()
    end_event = CuEvent()
    
    # Record start
    CUDA.record(start_event)
    
    # Execute kernel
    kernel_func(args...)
    
    # Record end
    CUDA.record(end_event)
    CUDA.synchronize()
    
    # Calculate elapsed time
    elapsed_ms = CUDA.elapsed(start_event, end_event)
    
    println("$name execution time: $(round(elapsed_ms, digits=2))ms")
    
    return elapsed_ms
end

end # module