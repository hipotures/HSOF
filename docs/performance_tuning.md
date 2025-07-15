# HSOF Multi-GPU Performance Tuning Guide

## Overview

This guide provides hardware-specific performance tuning recommendations for the HSOF multi-GPU system. The configurations are optimized for dual RTX 4090 setups but include guidance for other GPU configurations.

## Hardware-Specific Configurations

### NVIDIA RTX 4090

**Specifications**:
- CUDA Cores: 16,384
- Memory: 24 GB GDDR6X
- Memory Bandwidth: 1,008 GB/s
- TDP: 450W (can be limited)
- PCIe: Gen 4 x16

**Optimal Configuration**:
```julia
config = DistributedMCTSConfig(
    # Tree distribution
    num_gpus = 2,
    total_trees = 100,
    
    # Memory settings (24GB per GPU)
    dataset_memory_limit = 8 * 1024^3,    # 8 GB for dataset
    tree_memory_limit = 150 * 1024^2,     # 150 MB per tree
    buffer_memory_limit = 2 * 1024^3,     # 2 GB for buffers
    
    # Compute settings
    batch_size = 512,                      # Optimal for 16K cores
    max_iterations = 100000,
    
    # Communication
    sync_interval = 1000,                  # Balance compute/comm
    top_k_candidates = 10,
    
    # Performance
    exploration_range = (0.5f0, 2.0f0),
    subsample_ratio = 0.8f0
)
```

**Power and Thermal Management**:
```bash
# Set power limit to 350W for efficiency
sudo nvidia-smi -pl 350

# Set fan curve for optimal cooling
sudo nvidia-settings -a "[gpu:0]/GPUFanControlState=1"
sudo nvidia-settings -a "[fan:0]/GPUTargetFanSpeed=75"
```

### NVIDIA RTX 3090

**Specifications**:
- CUDA Cores: 10,496
- Memory: 24 GB GDDR6X
- Memory Bandwidth: 936 GB/s
- TDP: 350W

**Optimal Configuration**:
```julia
config = DistributedMCTSConfig(
    batch_size = 384,                      # Smaller than 4090
    sync_interval = 1200,                  # Slightly less frequent
    # Memory limits same as 4090 (same 24GB)
)
```

### NVIDIA A100

**Specifications**:
- CUDA Cores: 6,912 (but Tensor Cores: 432)
- Memory: 40/80 GB HBM2e
- Memory Bandwidth: 1,555/2,039 GB/s
- NVLink: 600 GB/s

**Optimal Configuration**:
```julia
config = DistributedMCTSConfig(
    # Leverage higher memory
    dataset_memory_limit = 16 * 1024^3,    # 16 GB
    tree_memory_limit = 300 * 1024^2,      # 300 MB per tree
    
    # Leverage NVLink
    sync_interval = 500,                    # More frequent syncs OK
    top_k_candidates = 20,                  # More data transfer
    
    # Use Tensor Cores if applicable
    batch_size = 768                        # Multiple of 8 for Tensor Cores
)
```

## Multi-GPU Topology Optimization

### With NVLink

**Configuration**:
```julia
# NVLink allows frequent communication
config = DistributedMCTSConfig(
    sync_interval = 500,         # 2x more frequent
    top_k_candidates = 20,       # 2x more candidates
    enable_peer_access = true
)

# Memory settings
config.dataset_replication = :partial  # Can share some data
config.communication_backend = :nccl   # Use NCCL for NVLink
```

### PCIe Only

**Configuration**:
```julia
# Minimize PCIe traffic
config = DistributedMCTSConfig(
    sync_interval = 2000,        # Less frequent
    top_k_candidates = 5,        # Fewer candidates
    compression_enabled = true   # Compress transfers
)

# Memory settings
config.dataset_replication = :full  # Full replication
config.communication_backend = :cuda_aware_mpi
```

### Mixed GPU Types

**Configuration**:
```julia
# Balance for weakest GPU
weaker_gpu_memory = min(gpu0_memory, gpu1_memory)
weaker_gpu_compute = min(gpu0_cores, gpu1_cores)

config = DistributedMCTSConfig(
    # Memory based on smallest GPU
    dataset_memory_limit = weaker_gpu_memory * 0.3,
    
    # Compute based on weakest GPU
    batch_size = optimal_batch_size(weaker_gpu_compute),
    
    # Adjust tree distribution
    gpu0_tree_range = 1:40,      # Fewer trees on weaker GPU
    gpu1_tree_range = 41:100     # More trees on stronger GPU
)
```

## Performance Optimization Strategies

### 1. Memory Optimization

**Coalesced Access**:
```julia
# Bad: Random access
for i in indices
    sum += data[random_index[i]]
end

# Good: Coalesced access
coalesced_read!(output, data, sorted_indices)
```

**Data Layout**:
```julia
# Optimize for access pattern
data_tiled = optimize_data_layout(data, 
    access_pattern = :tiled,
    tile_size = 32  # Warp size
)
```

**Memory Pooling**:
```julia
# Create pools for different allocation sizes
small_pool = create_memory_pool(Float32, 
    block_size = 1024,
    num_blocks = 1000
)
large_pool = create_memory_pool(Float32,
    block_size = 1024 * 1024,
    num_blocks = 10
)
```

### 2. Compute Optimization

**Batch Size Tuning**:
```julia
function optimal_batch_size(gpu_cores::Int)
    # Rule of thumb: 32-64 threads per SM
    sms = gpu_cores ÷ 128  # Approximate
    return sms * 32 * 4    # 4 blocks per SM
end

# RTX 4090: 512
# RTX 3090: 384
# A100: 768
```

**Kernel Launch Configuration**:
```julia
# Optimal thread block size
threads = 256  # Good for most GPUs

# Grid size based on work
blocks = cld(work_size, threads)

# Limit blocks for occupancy
max_blocks_per_sm = 16
total_sms = gpu_cores ÷ 128
max_blocks = total_sms * max_blocks_per_sm
blocks = min(blocks, max_blocks)
```

### 3. Communication Optimization

**Batching Transfers**:
```julia
# Bad: Many small transfers
for candidate in candidates
    transfer_candidate(candidate)
end

# Good: Batch transfer
batch = collect_candidates(candidates)
transfer_batch(batch)
```

**Compression**:
```julia
# Enable compression for large transfers
if length(data) > compression_threshold
    compressed = compress_candidates(data)
    transfer_data(compressed)
else
    transfer_data(data)
end
```

**Overlap Compute and Communication**:
```julia
# Use streams for overlap
compute_stream = CuStream()
transfer_stream = CuStream()

# Start compute on stream 1
@cuda stream=compute_stream compute_kernel(...)

# Start transfer on stream 2
@cuda stream=transfer_stream transfer_kernel(...)

# Synchronize when needed
synchronize(compute_stream)
synchronize(transfer_stream)
```

### 4. Load Balancing

**Static Distribution**:
```julia
# Based on GPU capabilities
gpu0_capability = gpu_cores[0] * gpu_memory[0]
gpu1_capability = gpu_cores[1] * gpu_memory[1]
total_capability = gpu0_capability + gpu1_capability

gpu0_trees = round(Int, 100 * gpu0_capability / total_capability)
gpu1_trees = 100 - gpu0_trees
```

**Dynamic Rebalancing**:
```julia
# Monitor and adjust
if abs(gpu0_load - gpu1_load) > 0.1  # 10% imbalance
    trees_to_move = calculate_rebalance_count(gpu0_load, gpu1_load)
    migrate_trees(source_gpu, target_gpu, trees_to_move)
end
```

## Benchmarking and Validation

### Performance Benchmarks

```julia
# 1. Scaling Efficiency Test
function benchmark_scaling()
    configs = [
        (gpus=1, trees=50),
        (gpus=2, trees=100)
    ]
    
    for config in configs
        result = run_benchmark(config)
        println("GPUs: $(config.gpus), Time: $(result.time)")
    end
    
    efficiency = calculate_efficiency(results)
    @assert efficiency > 0.85 "Scaling efficiency below target"
end

# 2. Memory Bandwidth Test
function benchmark_memory()
    sizes = [1_000_000, 10_000_000, 100_000_000]
    
    for size in sizes
        bandwidth = measure_bandwidth(size)
        theoretical = 1008.0  # GB/s for RTX 4090
        utilization = bandwidth / theoretical
        
        println("Size: $size, Bandwidth: $bandwidth GB/s")
        println("Utilization: $(utilization * 100)%")
    end
end

# 3. Communication Overhead Test
function benchmark_communication()
    data_sizes = [1024, 1024^2, 10*1024^2]  # 1KB, 1MB, 10MB
    
    for size in data_sizes
        time = measure_transfer_time(size)
        bandwidth = size / time / 1024^3  # GB/s
        
        println("Size: $(size/1024^2) MB, Time: $time ms")
        println("Effective bandwidth: $bandwidth GB/s")
    end
end
```

### Profiling Tools

**NVIDIA Nsight Compute**:
```bash
# Profile kernel performance
ncu --target-processes all \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
             dram__throughput.avg.pct_of_peak_sustained_elapsed \
    julia distributed_mcts.jl
```

**NVIDIA Nsight Systems**:
```bash
# System-wide profiling
nsys profile --stats=true \
    --trace=cuda,nvtx,osrt \
    --output=profile_report \
    julia distributed_mcts.jl
```

**Julia Profiling**:
```julia
using Profile
using ProfileView

# Profile code
@profile run_distributed_mcts!(engine, iterations=1000)

# Visualize
ProfileView.view()

# GPU-specific profiling
using CUDA
CUDA.@profile run_gpu_kernel(...)
```

## Troubleshooting Performance Issues

### Common Bottlenecks

1. **Memory Bandwidth Saturation**
   - Symptom: <70% GPU utilization with high memory controller load
   - Solution: Optimize data layout, use shared memory

2. **PCIe Bandwidth Limitation**
   - Symptom: Long synchronization times, low GPU utilization during sync
   - Solution: Reduce sync frequency, compress data, use NVLink

3. **Kernel Launch Overhead**
   - Symptom: Many small kernels, low GPU utilization
   - Solution: Batch operations, use persistent kernels

4. **Load Imbalance**
   - Symptom: One GPU at 95%, other at 60%
   - Solution: Enable dynamic rebalancing, adjust tree distribution

### Performance Checklist

Before deployment:
- [ ] Verify GPU cooling is adequate
- [ ] Set appropriate power limits
- [ ] Check PCIe link width (x16)
- [ ] Enable GPU persistence mode
- [ ] Disable GPU boost if consistent performance needed

During operation:
- [ ] Monitor GPU temperature (<83°C)
- [ ] Check for thermal throttling
- [ ] Verify memory isn't fragmented
- [ ] Monitor PCIe bandwidth usage
- [ ] Check for CPU bottlenecks

## Best Practices Summary

1. **Memory**: Keep 10-15% free for dynamic allocations
2. **Compute**: Use batch sizes that are multiples of 32 (warp size)
3. **Communication**: Sync every 1000-2000 iterations without NVLink
4. **Power**: Run at 80-90% TDP for best efficiency
5. **Temperature**: Keep below 80°C for consistent performance
6. **Monitoring**: Check metrics every 5 seconds, export every 5 minutes

## References

- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA GPU Architecture Whitepapers](https://www.nvidia.com/en-us/data-center/resources/whitepapers/)
- [Julia GPU Programming](https://cuda.juliagpu.org/stable/)