# GPU Architecture Design

```@meta
CurrentModule = HSOF
```

## Overview

HSOF is optimized for dual NVIDIA RTX 4090 GPUs without NVLink, requiring careful design to maximize performance while minimizing PCIe bottlenecks. The architecture emphasizes:

- Efficient workload distribution
- Minimal inter-GPU communication
- Optimized memory access patterns
- Concurrent kernel execution

## Hardware Configuration

### Target System

- **GPUs**: 2x NVIDIA RTX 4090 (24GB VRAM each)
- **Compute Capability**: 8.9 (Ada Lovelace)
- **PCIe**: Gen 4 x16 (theoretical 31.5 GB/s bidirectional)
- **No NVLink**: Direct GPU-to-GPU communication via PCIe

### Memory Hierarchy

```
┌─────────────────┐     ┌─────────────────┐
│   GPU 0 (24GB)  │     │   GPU 1 (24GB)  │
│  ┌───────────┐  │     │  ┌───────────┐  │
│  │ L2 Cache  │  │     │  │ L2 Cache  │  │
│  │   98MB    │  │     │  │   98MB    │  │
│  └───────────┘  │     │  └───────────┘  │
│  ┌───────────┐  │     │  ┌───────────┐  │
│  │ Shared Mem│  │     │  │ Shared Mem│  │
│  │  128KB/SM │  │     │  │  128KB/SM │  │
│  └───────────┘  │     │  └───────────┘  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
              PCIe Gen 4 x16
                     │
              ┌──────┴──────┐
              │  Host RAM   │
              │   (64GB)    │
              └─────────────┘
```

## GPU Manager Architecture

### Device Management

```julia
struct GPUDevice
    index::Int
    cuda_device::CUDA.CuDevice
    name::String
    compute_capability::Tuple{Int,Int}
    total_memory::Int
    uuid::String
end

struct GPUManagerState
    devices::Vector{GPUDevice}
    current_device::Int
    peer_access_matrix::Matrix{Bool}
    workload_distribution::Vector{Float64}
    single_gpu_mode::Bool
end
```

### Workload Distribution

```julia
function distribute_workload(total_work::Int)
    # Equal distribution for homogeneous GPUs
    work_per_gpu = [total_work ÷ 2, total_work - (total_work ÷ 2)]
    
    # Adjust for memory availability
    for i in 1:2
        available_mem = get_available_memory(i-1)
        if available_mem < required_memory(work_per_gpu[i])
            # Rebalance workload
            adjust_workload!(work_per_gpu, i)
        end
    end
    
    return work_per_gpu
end
```

## Memory Management

### Pooled Allocation Strategy

```julia
mutable struct MemoryPool
    device_id::Int
    limit_gb::Float64
    allocated::Int
    peak_allocated::Int
    allocations::Dict{Ptr{Nothing}, Int}
end
```

### Memory Patterns

1. **Persistent Buffers**: Reused across iterations
2. **Temporary Buffers**: Allocated per-stage
3. **Pinned Memory**: For efficient PCIe transfers
4. **Unified Memory**: For overflow handling

### Allocation Best Practices

```julia
# Good: Reuse buffers
buffer = get_buffer_from_pool(size)
process_data!(buffer, data)
return_buffer_to_pool(buffer)

# Bad: Allocate every iteration
for i in 1:n_iterations
    buffer = CUDA.zeros(size)  # Allocation overhead
    process_data!(buffer, data)
end
```

## Stream Management

### Concurrent Execution

```julia
# Create multiple streams per GPU
streams_gpu0 = [CuStream() for _ in 1:4]
streams_gpu1 = [CuStream() for _ in 1:4]

# Launch concurrent kernels
for (i, batch) in enumerate(batches)
    stream = streams_gpu0[i % 4 + 1]
    @cuda stream=stream process_batch(batch)
end
```

### Stream Synchronization

```
GPU 0 Stream 0: |--Kernel A--|--Copy--|
GPU 0 Stream 1:     |--Kernel B--|--Copy--|
GPU 1 Stream 0: |--Kernel C--|--Copy--|
GPU 1 Stream 1:     |--Kernel D--|--Copy--|
                ↓
           Synchronization Point
                ↓
         Aggregate Results
```

## Kernel Design Principles

### Coalesced Memory Access

```julia
# Good: Coalesced access
function coalesced_kernel(data)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if tid <= length(data)
        data[tid] = process(data[tid])
    end
end

# Bad: Strided access
function strided_kernel(data, stride)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    idx = tid * stride  # Non-coalesced
    if idx <= length(data)
        data[idx] = process(data[idx])
    end
end
```

### Shared Memory Usage

```julia
function reduction_kernel(input, output)
    shared = @cuDynamicSharedMem(Float32, blockDim().x)
    tid = threadIdx().x
    
    # Load to shared memory
    shared[tid] = tid <= length(input) ? input[tid] : 0.0f0
    sync_threads()
    
    # Reduction in shared memory
    stride = blockDim().x ÷ 2
    while stride > 0
        if tid <= stride
            shared[tid] += shared[tid + stride]
        end
        sync_threads()
        stride ÷= 2
    end
    
    if tid == 1
        output[blockIdx().x] = shared[1]
    end
end
```

### Occupancy Optimization

```julia
# Calculate optimal launch configuration
function optimal_launch_config(kernel_func, shared_mem_per_block)
    max_threads = attribute(device(), 
                           CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    
    # Get occupancy information
    blocks_per_sm = active_blocks(kernel_func, max_threads, shared_mem_per_block)
    
    # Optimize for occupancy
    threads = 256  # Good default for most kernels
    blocks = cld(problem_size, threads)
    
    return threads, blocks
end
```

## Multi-GPU Coordination

### Data Partitioning

```julia
# Feature-wise partitioning for independence
features_gpu0 = features[:, 1:n_features÷2]
features_gpu1 = features[:, n_features÷2+1:end]

# Process independently
@sync begin
    @async begin
        device!(0)
        results_gpu0 = process_features(features_gpu0)
    end
    @async begin
        device!(1)
        results_gpu1 = process_features(features_gpu1)
    end
end
```

### Communication Patterns

1. **Minimize Transfers**: Process data where it resides
2. **Batch Communications**: Aggregate small transfers
3. **Overlap Computation**: Hide transfer latency
4. **Direct Peer Access**: When available

```julia
# Efficient GPU-to-GPU transfer
function transfer_between_gpus(src_array, dst_device)
    if peer_access_available(current_device(), dst_device)
        # Direct P2P copy
        device!(dst_device)
        dst_array = similar(src_array)
        copyto!(dst_array, src_array)
    else
        # Through host memory
        host_array = Array(src_array)
        device!(dst_device)
        dst_array = CuArray(host_array)
    end
    return dst_array
end
```

## Performance Monitoring

### Key Metrics

```julia
struct GPUMetrics
    kernel_time_ms::Float64
    memory_bandwidth_gb_s::Float64
    occupancy::Float64
    sm_efficiency::Float64
    memory_usage_gb::Float64
end

function collect_metrics(kernel_func, args...)
    # Time kernel execution
    time = CUDA.@elapsed begin
        @cuda kernel_func(args...)
        synchronize()
    end
    
    # Calculate bandwidth
    bytes_transferred = calculate_bytes(args...)
    bandwidth = bytes_transferred / time / 1e9
    
    # Get occupancy
    occupancy = calculate_occupancy(kernel_func)
    
    return GPUMetrics(time*1000, bandwidth, occupancy, 0.0, 0.0)
end
```

### Profiling Integration

```julia
# NVTX markers for profiling
CUDA.@profile "Stage1_Filtering" begin
    filter_results = gpu_statistical_filter(data)
end

CUDA.@profile "Stage2_MCTS" begin
    mcts_results = gpu_mcts_search(filtered_data)
end
```

## Error Handling

### Memory Errors

```julia
function safe_allocate(T, dims...)
    try
        return CUDA.zeros(T, dims...)
    catch e
        if isa(e, CUDAOutOfMemoryError)
            # Try garbage collection
            GC.gc()
            CUDA.reclaim()
            
            # Retry allocation
            try
                return CUDA.zeros(T, dims...)
            catch
                # Fall back to smaller allocation
                return allocate_chunked(T, dims...)
            end
        else
            rethrow(e)
        end
    end
end
```

### Device Errors

```julia
function with_gpu_error_handling(f, args...)
    try
        result = f(args...)
        synchronize()  # Catch async errors
        return result
    catch e
        if isa(e, CUDAError)
            @error "GPU error occurred" exception=e
            # Reset device state
            device_reset!()
            # Retry or fallback
            return cpu_fallback(f, args...)
        else
            rethrow(e)
        end
    end
end
```

## Best Practices

### Do's

1. ✓ Profile before optimizing
2. ✓ Minimize host-device transfers
3. ✓ Use appropriate block sizes (multiples of 32)
4. ✓ Coalesce memory accesses
5. ✓ Reuse allocated memory
6. ✓ Overlap computation with transfers

### Don'ts

1. ✗ Allocate in kernels
2. ✗ Use excessive shared memory
3. ✗ Ignore occupancy
4. ✗ Transfer data unnecessarily
5. ✗ Use atomic operations excessively
6. ✗ Launch kernels with few threads

## Next Steps

- [Algorithm Design](@ref): Algorithm implementation details
- [GPU Programming Tutorial](@ref): Hands-on GPU programming guide
- [Performance Benchmarks](@ref): Performance analysis and results