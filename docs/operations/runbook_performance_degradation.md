# Performance Degradation Runbook

## Overview
This runbook addresses performance degradation issues in the HSOF multi-GPU system.

## Detection

### Key Performance Indicators (KPIs)
- **Scaling Efficiency**: Should be >85% (target: 90%)
- **GPU Utilization**: Should be 80-95%
- **Tree Evaluations/sec**: Should meet minimum threshold
- **Synchronization Latency**: Should be <20ms (NVLink) or <50ms (PCIe)
- **Memory Bandwidth**: Should be >70% of theoretical maximum

### Symptoms
- Scaling efficiency drops below 85%
- Increased iteration times
- GPU utilization below 80%
- High synchronization overhead
- Unbalanced GPU workloads

### Monitoring Alerts
- `performance_scaling_degraded`: Efficiency < 85%
- `gpu_X_low_utilization`: GPU util < 80%
- `sync_latency_high`: Sync time > threshold
- `workload_imbalance`: GPU load difference > 10%

## Diagnosis

### 1. Check Current Performance Metrics
```julia
# Get performance summary
using GPU.PerformanceMonitoring

perf_summary = get_performance_summary(engine.perf_monitor)
println("Current Performance Metrics:")
for (gpu_id, metrics) in perf_summary
    println("  $gpu_id: $(metrics)")
end

# Check scaling efficiency
efficiency = calculate_scaling_efficiency(scaling_benchmark, config)
println("Scaling Efficiency: $(efficiency * 100)%")
```

### 2. Identify Bottlenecks
```julia
# Run bottleneck analysis
bottlenecks = identify_bottlenecks(scaling_benchmark, results)
println("Identified Bottlenecks:")
for b in bottlenecks
    println("  $(b.component): $(b.impact * 100)% impact")
end
```

### 3. Profile GPU Kernels
```bash
# Use NVIDIA Nsight Compute
ncu --target-processes all -o profile_output julia distributed_mcts.jl

# Analyze with Nsight Systems
nsys profile --stats=true julia distributed_mcts.jl
```

## Common Issues and Solutions

### Issue 1: Poor Scaling Efficiency

**Diagnosis**:
```julia
# Compare single vs multi-GPU performance
single_gpu_time = run_single_gpu_baseline(benchmark, config)
multi_gpu_time = run_multi_gpu_benchmark(benchmark, config).total_time
speedup = single_gpu_time / multi_gpu_time
efficiency = speedup / num_gpus
```

**Solutions**:

1. **Reduce Synchronization Frequency**:
```julia
# Increase sync interval
config = DistributedMCTSConfig(
    sync_interval = 2000,  # Was 1000
    top_k_candidates = 5   # Was 10
)
```

2. **Optimize Communication**:
```julia
# Enable compression
engine.transfer_manager.compression_enabled = true
engine.transfer_manager.compression_threshold = 1024  # bytes

# Use larger batches
config.batch_size = 512  # Was 256
```

3. **Check PCIe Bandwidth**:
```bash
# Test PCIe bandwidth
nvidia-smi nvlink -s
bandwidthTest --device=all
```

### Issue 2: Workload Imbalance

**Diagnosis**:
```julia
# Check per-GPU metrics
for (gpu_id, load) in get_gpu_loads(engine)
    println("GPU $gpu_id: $load% load")
end
```

**Solutions**:

1. **Enable Dynamic Rebalancing**:
```julia
# Activate rebalancing
enable_auto_rebalancing!(engine.rebalancing_manager)
set_rebalancing_threshold!(engine.rebalancing_manager, 0.1)  # 10% imbalance
```

2. **Manual Tree Redistribution**:
```julia
# Move trees from overloaded to underloaded GPU
migration_plan = create_migration_plan(engine.rebalancing_manager)
execute_migration!(engine.rebalancing_manager, migration_plan)
```

### Issue 3: Memory Bandwidth Saturation

**Diagnosis**:
```julia
# Check bandwidth utilization
bandwidth_stats = analyze_bandwidth_utilization(data_size, elapsed_time)
println("Bandwidth: $(bandwidth_stats["achieved_bandwidth_gb_s"]) GB/s")
println("Utilization: $(bandwidth_stats["utilization_percent"])%")
```

**Solutions**:

1. **Optimize Memory Access Patterns**:
```julia
# Enable memory optimization
data_optimized = optimize_data_layout(data, access_pattern=:tiled)

# Use coalesced access
coalesced_read!(dst, src, indices)
```

2. **Reduce Memory Transfers**:
```julia
# Keep data on GPU longer
config.dataset_memory_limit = Int(floor(available_memory * 0.5))  # Was 0.4
```

### Issue 4: High Synchronization Overhead

**Diagnosis**:
```julia
# Measure sync time
sync_start = time()
synchronize_trees!(engine)
sync_time = (time() - sync_start) * 1000  # ms
```

**Solutions**:

1. **Reduce Sync Frequency**:
```julia
# Only sync when necessary
if should_synchronize(engine, iteration)
    synchronize_trees!(engine)
end
```

2. **Optimize Sync Data**:
```julia
# Only transfer essential data
config.top_k_candidates = 3  # Reduce from 10
```

## Performance Optimization Checklist

### Before Starting Work
- [ ] Record baseline performance metrics
- [ ] Check GPU temperatures and power settings
- [ ] Verify no other processes using GPUs
- [ ] Ensure datasets are cached in GPU memory

### During Operation
- [ ] Monitor GPU utilization (target: 85-95%)
- [ ] Check memory usage (keep 10% free)
- [ ] Watch for thermal throttling
- [ ] Track synchronization frequency

### Optimization Steps
1. **Memory Optimization**
   - [ ] Enable memory pooling
   - [ ] Use optimal data layouts
   - [ ] Implement prefetching

2. **Computation Optimization**
   - [ ] Increase batch sizes
   - [ ] Use tensor cores if available
   - [ ] Optimize kernel launch parameters

3. **Communication Optimization**
   - [ ] Enable peer-to-peer access
   - [ ] Use compression for large transfers
   - [ ] Batch small transfers

4. **Load Balancing**
   - [ ] Enable dynamic rebalancing
   - [ ] Adjust tree distribution
   - [ ] Monitor per-GPU metrics

## Emergency Performance Recovery

### Quick Wins (5 minutes)
```julia
# 1. Reduce sync frequency
engine.config.sync_interval = 5000

# 2. Disable non-essential features
engine.config.enable_subsampling = false

# 3. Reduce batch size for lower latency
engine.config.batch_size = 128
```

### Medium-term Fixes (30 minutes)
```julia
# 1. Restart with optimized configuration
config = load_config("configs/performance_optimized.toml")

# 2. Clear GPU memory and caches
for gpu_id in 0:num_gpus-1
    device!(gpu_id)
    CUDA.reclaim()
end

# 3. Rebalance workload
rebalance_all_trees!(engine)
```

### Long-term Solutions (2+ hours)
1. Profile and optimize hot code paths
2. Redesign data structures for better locality
3. Implement custom CUDA kernels for bottlenecks
4. Upgrade hardware (NVLink, faster GPUs)

## Performance Tuning by Hardware

### RTX 4090 Specific
```julia
# Optimal settings for RTX 4090
config = DeploymentConfig(
    # Memory: 24GB per GPU
    dataset_memory_limit = 8 * 1024^3,  # 8GB
    tree_memory_limit = 150 * 1024^2,   # 150MB per tree
    
    # Compute: 16384 CUDA cores
    batch_size = 512,
    block_size = 256,
    
    # Bandwidth: 1008 GB/s
    prefetch_distance = 16
)
```

### Multi-GPU Topology
```julia
# With NVLink
if topology.nvlink_available
    config.sync_interval = 500  # More frequent syncs OK
    config.top_k_candidates = 20  # Can transfer more
end

# Without NVLink (PCIe only)
if !topology.nvlink_available
    config.sync_interval = 2000  # Less frequent syncs
    config.top_k_candidates = 5   # Minimize transfers
end
```

## Monitoring Commands

### Real-time GPU Monitoring
```bash
# Watch GPU stats
watch -n 1 nvidia-smi

# Detailed monitoring
nvidia-smi dmon -s pucvmet -d 1

# Power and temperature
nvidia-smi -q -d POWER,TEMPERATURE -l 1
```

### Application Profiling
```julia
# Enable detailed profiling
ENV["JULIA_CUDA_PROFILING"] = "1"

# Run with profiling
@profile include("distributed_mcts.jl")
Profile.print()
```

## Related Documents
- [GPU Failure Runbook](runbook_gpu_failures.md)
- [Memory Issues Runbook](runbook_memory_issues.md)
- [Performance Tuning Guide](../performance_tuning.md)