# MCTS GPU Integration Architecture

## Overview

The MCTS GPU Integration module provides a distributed implementation of Monte Carlo Tree Search across multiple GPUs, specifically designed for the HSOF (Hybrid Stage-Optimized Feature) project. This integration enables efficient feature selection using 100 distributed MCTS trees split across dual RTX 4090 GPUs.

## Architecture

### Core Components

1. **DistributedMCTSEngine**
   - Central coordinator for multi-GPU MCTS execution
   - Manages tree distribution, synchronization, and result aggregation
   - Integrates with existing Stage 2 MCTS components

2. **DistributedMCTSConfig**
   - Configuration for distributed execution
   - Defines tree distribution (GPU0: trees 1-50, GPU1: trees 51-100)
   - Sets metamodel split (training on GPU0, inference on GPU1)
   - Controls synchronization intervals and diversity parameters

3. **Work Distribution**
   - Trees 1-50 assigned to GPU 0
   - Trees 51-100 assigned to GPU 1
   - Dynamic load balancing via WorkDistributor
   - GPU affinity management for optimal performance

4. **Communication Layer**
   - PCIeTransferManager for efficient GPU-to-GPU communication
   - Candidate sharing with compression support
   - Peer access enablement when available
   - Fallback to CPU-mediated transfers

5. **Synchronization**
   - Periodic synchronization every N iterations (configurable)
   - Top-K candidate sharing between GPUs
   - Feature importance aggregation
   - Barrier synchronization for coordinated phases

## Key Features

### 1. Tree Distribution
```julia
config = DistributedMCTSConfig(
    num_gpus = 2,
    total_trees = 100,
    sync_interval = 1000,
    top_k_candidates = 10
)
# Results in:
# GPU 0: trees 1-50
# GPU 1: trees 51-100
```

### 2. Diversity Mechanisms
- **Exploration Variation**: Different exploration constants per tree (0.5-2.0 range)
- **Subsampling**: Each tree can use different data subsets (80% default)
- **Feature Masking**: Probabilistic feature masking based on importance

### 3. Metamodel Integration
```julia
metamodel_interface = (
    train = (data, gpu) -> train_on_gpu(data, gpu),
    transfer = (src, dst) -> transfer_model(src, dst),
    inference = (data) -> run_inference(data)
)
```
- Training occurs on GPU 0 with collected feature data
- Model transferred to GPU 1 for inference
- Results used to update tree policies

### 4. Result Aggregation
- Per-tree results collected with feature rankings
- Consensus features determined across all trees
- Confidence-weighted aggregation
- Support for caching and incremental updates

## Usage Example

```julia
using GPU.MCTSGPUIntegration

# Create distributed engine
engine = create_distributed_engine(
    num_gpus = 2,
    total_trees = 100,
    sync_interval = 1000,
    enable_exploration_variation = true,
    enable_subsampling = true
)

# Initialize with problem dimensions
initialize_distributed_mcts!(
    engine,
    num_features = 10000,
    num_samples = 50000,
    metamodel_interface = my_metamodel
)

# Run distributed MCTS
run_distributed_mcts!(engine, iterations = 100000)

# Get results
ensemble_result = get_distributed_results(engine)
top_features = ensemble_result.feature_rankings[1:100]
```

## Performance Considerations

### Memory Layout
- Optimized for coalesced memory access
- Feature masks stored as bit arrays for efficiency
- Shared memory utilization in kernels

### Communication Optimization
- Batch candidate transfers to minimize overhead
- Compression for large candidate sets
- Peer-to-peer transfers when available

### Synchronization Strategy
- Asynchronous tree expansion between sync points
- Minimal blocking during synchronization
- Parallel evaluation batch processing

## Integration with Stage 2

### Feature Masking
The integration supports Stage 2's feature masking mechanism:
```julia
# Feature masks updated based on importance
update_feature_importance!(engine, top_candidates)
```

### Diversity Parameters
Compatible with Stage 2 diversity mechanisms:
- Per-tree exploration constants
- Data subsampling ratios
- Feature mask variations

### Metamodel Split
Designed for Stage 2's metamodel architecture:
- Training data collection from GPU 0 trees
- Model training on GPU 0
- Transfer to GPU 1 for inference
- Policy updates based on predictions

## Monitoring and Debugging

### Performance Monitoring
```julia
# Real-time monitoring
start_monitoring!(engine.perf_monitor)

# Get performance report
report = get_performance_summary(engine.perf_monitor)
```

### Tree Statistics
```julia
# Get tree statistics
stats = get_tree_summary(engine)

# View synchronization state
sync_state = get_sync_state(engine.sync_manager)
```

### Transfer Statistics
```julia
# Check communication performance
transfer_stats = get_transfer_stats(engine.transfer_manager)
```

## Configuration Options

### Key Parameters
- `num_gpus`: Number of GPUs to use (1 or 2)
- `total_trees`: Total MCTS trees (default: 100)
- `sync_interval`: Iterations between synchronizations (default: 1000)
- `top_k_candidates`: Features to share between GPUs (default: 10)
- `enable_subsampling`: Enable data subsampling (default: true)
- `subsample_ratio`: Fraction of data per tree (default: 0.8)
- `enable_exploration_variation`: Vary exploration across trees (default: true)
- `exploration_range`: Min/max exploration constants (default: 0.5-2.0)

### Advanced Options
- `batch_size`: Evaluation batch size (default: 256)
- `max_iterations`: Maximum MCTS iterations (default: 10000)
- `metamodel_training_gpu`: GPU for model training (default: 0)
- `metamodel_inference_gpu`: GPU for model inference (default: 1)

## Troubleshooting

### Common Issues

1. **Insufficient GPU Memory**
   - Reduce `batch_size`
   - Lower number of trees per GPU
   - Enable memory pooling

2. **Synchronization Timeouts**
   - Increase sync interval
   - Check GPU health status
   - Verify peer access configuration

3. **Poor Scaling Efficiency**
   - Check PCIe bandwidth utilization
   - Optimize sync_interval
   - Reduce top_k_candidates if bandwidth-limited

### Debug Mode
```julia
# Enable debug logging
ENV["JULIA_DEBUG"] = "MCTSGPUIntegration"

# Check GPU status
for (gpu_id, engine) in engine.gpu_engines
    @info "GPU $gpu_id status" active_trees=length(tree_range)
end
```

## Future Enhancements

1. **Dynamic Tree Redistribution**
   - Runtime load balancing based on tree complexity
   - Automatic migration of slow trees

2. **Multi-Model Support**
   - Multiple metamodels with different architectures
   - Ensemble metamodel predictions

3. **Fault Tolerance**
   - Automatic recovery from GPU failures
   - Checkpoint/restart capability

4. **Extended Scaling**
   - Support for 4+ GPUs
   - Hierarchical synchronization for large clusters

## References

- HSOF Stage 2 MCTS Implementation
- CUDA.jl Documentation
- Multi-GPU Programming Best Practices