# GPU Programming Guide

```@meta
CurrentModule = HSOF
```

## Introduction

This guide covers GPU programming patterns and best practices for extending HSOF with custom CUDA kernels.

## CUDA.jl Basics

### Kernel Definition

```julia
# Basic CUDA kernel
function vector_add_kernel(a, b, c)
    # Get thread index
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    # Bounds check
    if i <= length(c)
        c[i] = a[i] + b[i]
    end
    
    return nothing  # Kernels must return nothing
end

# Launch kernel
a = CUDA.rand(1000)
b = CUDA.rand(1000)
c = CUDA.zeros(1000)

@cuda threads=256 blocks=4 vector_add_kernel(a, b, c)
synchronize()  # Wait for completion
```

### Thread Indexing

```julia
# 1D indexing
idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

# 2D indexing
row = threadIdx().y + (blockIdx().y - 1) * blockDim().y
col = threadIdx().x + (blockIdx().x - 1) * blockDim().x

# 3D indexing
x = threadIdx().x + (blockIdx().x - 1) * blockDim().x
y = threadIdx().y + (blockIdx().y - 1) * blockDim().y
z = threadIdx().z + (blockIdx().z - 1) * blockDim().z
```

## Memory Patterns

### Coalesced Access

```julia
# Good: Adjacent threads access adjacent memory
function coalesced_kernel(data)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if tid <= length(data)
        # Thread 0 → data[1], Thread 1 → data[2], etc.
        data[tid] = sqrt(data[tid])
    end
    return
end

# Bad: Strided access pattern
function strided_kernel(data, stride)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    idx = 1 + (tid - 1) * stride
    if idx <= length(data)
        # Thread 0 → data[1], Thread 1 → data[33], etc. (stride=32)
        data[idx] = sqrt(data[idx])
    end
    return
end
```

### Shared Memory

```julia
function reduction_kernel(input, output)
    # Declare shared memory
    shared = @cuDynamicSharedMem(Float32, blockDim().x)
    
    tid = threadIdx().x
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    # Load to shared memory
    shared[tid] = i <= length(input) ? input[i] : 0.0f0
    sync_threads()  # Ensure all threads have loaded
    
    # Reduction in shared memory
    stride = blockDim().x ÷ 2
    while stride > 0
        if tid <= stride
            shared[tid] += shared[tid + stride]
        end
        sync_threads()
        stride ÷= 2
    end
    
    # Write result
    if tid == 1
        output[blockIdx().x] = shared[1]
    end
    return
end

# Launch with shared memory
threads = 256
shmem_size = threads * sizeof(Float32)
@cuda threads=threads blocks=10 shmem=shmem_size reduction_kernel(input, output)
```

## Advanced Patterns

### Warp-Level Operations

```julia
function warp_reduce_kernel(data)
    tid = threadIdx().x
    lane = tid & 31  # Lane within warp (0-31)
    warp_id = tid >> 5  # Warp ID
    
    val = data[tid]
    
    # Warp-level reduction (no sync needed within warp)
    val += shfl_down_sync(0xffffffff, val, 16)
    val += shfl_down_sync(0xffffffff, val, 8)
    val += shfl_down_sync(0xffffffff, val, 4)
    val += shfl_down_sync(0xffffffff, val, 2)
    val += shfl_down_sync(0xffffffff, val, 1)
    
    # Lane 0 has the sum for the warp
    if lane == 0
        warp_sums[warp_id] = val
    end
    return
end
```

### Matrix Operations

```julia
# Tiled matrix multiplication
function matmul_tiled_kernel(A, B, C, M, N, K)
    # Shared memory tiles
    tile_size = 32
    tile_A = @cuDynamicSharedMem(Float32, (tile_size, tile_size))
    tile_B = @cuDynamicSharedMem(Float32, (tile_size, tile_size), 
                                 sizeof(Float32) * tile_size * tile_size)
    
    row = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    col = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    sum = 0.0f0
    
    # Loop over tiles
    for tile_idx in 0:tile_size:K-1
        # Load tiles cooperatively
        if row <= M && tile_idx + threadIdx().x <= K
            tile_A[threadIdx().y, threadIdx().x] = 
                A[row, tile_idx + threadIdx().x]
        else
            tile_A[threadIdx().y, threadIdx().x] = 0.0f0
        end
        
        if col <= N && tile_idx + threadIdx().y <= K
            tile_B[threadIdx().y, threadIdx().x] = 
                B[tile_idx + threadIdx().y, col]
        else
            tile_B[threadIdx().y, threadIdx().x] = 0.0f0
        end
        
        sync_threads()
        
        # Compute partial product
        for k in 1:tile_size
            sum += tile_A[threadIdx().y, k] * tile_B[k, threadIdx().x]
        end
        
        sync_threads()
    end
    
    # Write result
    if row <= M && col <= N
        C[row, col] = sum
    end
    return
end
```

## Feature Selection Kernels

### Statistical Filter Kernel

```julia
function variance_filter_kernel(features, variances, n_samples)
    feature_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if feature_idx <= size(features, 2)
        # Calculate mean
        mean = 0.0f0
        for i in 1:n_samples
            mean += features[i, feature_idx]
        end
        mean /= n_samples
        
        # Calculate variance
        var = 0.0f0
        for i in 1:n_samples
            diff = features[i, feature_idx] - mean
            var += diff * diff
        end
        var /= n_samples
        
        variances[feature_idx] = var
    end
    return
end
```

### Mutual Information Kernel

```julia
function mutual_info_kernel(X, y, scores, n_bins)
    feature_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if feature_idx <= size(X, 2)
        # Histogram for joint distribution
        hist_xy = @cuDynamicSharedMem(Float32, (n_bins, n_bins))
        
        # Initialize histogram
        for i in 1:n_bins, j in 1:n_bins
            hist_xy[i, j] = 0.0f0
        end
        
        # Fill histogram
        for sample in 1:size(X, 1)
            x_bin = discretize(X[sample, feature_idx], n_bins)
            y_bin = discretize(y[sample], n_bins)
            hist_xy[x_bin, y_bin] += 1.0f0
        end
        
        # Calculate mutual information
        mi = calculate_mi_from_histogram(hist_xy, size(X, 1))
        scores[feature_idx] = mi
    end
    return
end
```

## Optimization Techniques

### Launch Configuration

```julia
# Calculate optimal launch parameters
function optimal_launch(problem_size, kernel_func)
    # Get device properties
    device = CUDA.device()
    max_threads = attribute(device, 
                           CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    
    # Common thread counts
    thread_options = [32, 64, 128, 256, 512, min(1024, max_threads)]
    
    best_config = (threads=256, blocks=cld(problem_size, 256))
    best_time = Inf
    
    # Benchmark different configurations
    for threads in thread_options
        blocks = cld(problem_size, threads)
        
        time = CUDA.@elapsed begin
            @cuda threads=threads blocks=blocks kernel_func(args...)
            synchronize()
        end
        
        if time < best_time
            best_time = time
            best_config = (threads=threads, blocks=blocks)
        end
    end
    
    return best_config
end
```

### Memory Bandwidth Optimization

```julia
# Vectorized memory access
function vectorized_copy_kernel(dst, src)
    # Use float4 for 4x bandwidth
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    # Cast to float4 pointers
    dst4 = reinterpret(Float4, dst)
    src4 = reinterpret(Float4, src)
    
    if tid <= length(dst4)
        dst4[tid] = src4[tid]
    end
    return
end
```

## Multi-GPU Patterns

### Distributed Reduction

```julia
function distributed_reduction(data::Vector{CuArray})
    # Each GPU reduces its portion
    partial_sums = Vector{Float32}(undef, length(data))
    
    @sync for (i, gpu_data) in enumerate(data)
        @async begin
            device!(i-1)
            partial_sums[i] = reduce(+, gpu_data)
        end
    end
    
    # Final reduction on CPU
    return sum(partial_sums)
end
```

### Pipeline Pattern

```julia
function pipeline_process(stages, data)
    n_gpus = length(devices())
    streams = [CuStream() for _ in 1:n_gpus]
    
    # Process in pipeline
    for (i, batch) in enumerate(batches)
        gpu_idx = (i - 1) % n_gpus
        stream = streams[gpu_idx + 1]
        
        device!(gpu_idx)
        
        # Stage 1 on GPU
        @async begin
            CUDA.stream!(stream) do
                stage1_result = stages[1](batch)
                
                # Transfer to next GPU if needed
                if n_gpus > 1
                    next_gpu = (gpu_idx + 1) % n_gpus
                    stage2_input = transfer_to_gpu(stage1_result, next_gpu)
                    
                    # Stage 2 on next GPU
                    device!(next_gpu)
                    stage2_result = stages[2](stage2_input)
                end
            end
        end
    end
end
```

## Debugging GPU Code

### Error Checking

```julia
# Wrap kernel launches
function safe_kernel_launch(kernel, args...; threads, blocks)
    try
        @cuda threads=threads blocks=blocks kernel(args...)
        synchronize()
    catch e
        if isa(e, CUDAError)
            # Get more information
            @error "CUDA error" exception=e code=e.code
            
            # Check for common issues
            if e.code == CUDA.cudaErrorInvalidConfiguration
                @error "Invalid launch configuration" threads blocks
            elseif e.code == CUDA.cudaErrorOutOfMemory
                @error "Out of GPU memory"
                CUDA.memory_status()
            end
        end
        rethrow(e)
    end
end
```

### Debugging Kernels

```julia
# Add debug prints (slow!)
function debug_kernel(data)
    tid = threadIdx().x
    
    if tid == 1 && blockIdx().x == 1
        @cuprintf("First thread: data[1] = %f\n", data[1])
    end
    
    # Process
    if tid <= length(data)
        data[tid] = data[tid] * 2
    end
    
    sync_threads()
    
    if tid == 1 && blockIdx().x == 1
        @cuprintf("After processing: data[1] = %f\n", data[1])
    end
    
    return
end
```

## Performance Profiling

### NVTX Markers

```julia
# Mark regions for profiling
CUDA.@profile "Feature_Selection" begin
    CUDA.@profile "Stage1_Filtering" begin
        filtering_results = run_filtering(data)
    end
    
    CUDA.@profile "Stage2_MCTS" begin
        mcts_results = run_mcts(filtering_results)
    end
end

# Run with: nsys profile julia script.jl
```

### Benchmarking

```julia
# Benchmark kernel performance
function benchmark_kernel(kernel, args...; samples=100)
    # Warmup
    @cuda kernel(args...)
    synchronize()
    
    times = Float64[]
    
    for _ in 1:samples
        time = CUDA.@elapsed begin
            @cuda kernel(args...)
            synchronize()
        end
        push!(times, time)
    end
    
    return (
        min = minimum(times),
        median = median(times),
        mean = mean(times),
        max = maximum(times)
    )
end
```

## Best Practices

### Do's

1. **Always check bounds** in kernels
2. **Use appropriate types** (Float32 for performance)
3. **Minimize divergent branching** within warps
4. **Reuse shared memory** when possible
5. **Profile before optimizing**

### Don'ts

1. **Don't allocate** in kernels
2. **Don't use recursive** functions
3. **Don't ignore** memory access patterns
4. **Don't synchronize** unnecessarily
5. **Don't use** double precision without need

## Next Steps

- [Custom Kernels Tutorial](@ref): Implement feature selection kernels
- [Performance Benchmarks](@ref): See optimization results
- [API Reference](@ref): Kernel API documentation