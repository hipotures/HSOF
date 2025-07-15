# HSOF Performance Benchmark Framework

## Overview

This comprehensive benchmarking framework measures and tracks the performance of the Hybrid Stage-Optimized Feature (HSOF) selection pipeline across multiple dimensions:

- **GPU Performance**: Utilization, memory usage, kernel launches
- **Memory Profiling**: Host/device allocations, leak detection
- **Latency Analysis**: Inference timing, throughput, tail latencies
- **Regression Detection**: Automated performance regression identification
- **Pipeline Benchmarks**: End-to-end performance validation

## Directory Structure

```
test/benchmarks/
├── gpu/                    # GPU profiling tools
│   └── gpu_profiler.jl
├── memory/                 # Memory profiling tools
│   └── memory_profiler.jl
├── latency/               # Latency measurement tools
│   └── latency_profiler.jl
├── reports/               # Report generation
│   └── report_generator.jl
├── regression_detector.jl  # Performance regression detection
├── benchmark_runner.jl     # Main benchmark orchestrator
└── results/               # Benchmark results storage
```

## Quick Start

### Basic Usage

```bash
# Run all benchmarks
julia test/benchmarks/benchmark_runner.jl --gpu

# Run specific stages
julia test/benchmarks/benchmark_runner.jl --stages gpu,memory --gpu

# Compare with baseline
julia test/benchmarks/benchmark_runner.jl --baseline results/baseline.json

# Save new baseline
julia test/benchmarks/benchmark_runner.jl --save-baseline
```

### Command Line Options

- `--name`: Benchmark run name (default: timestamped)
- `--stages`: Comma-separated stages to run (gpu,memory,latency,pipeline)
- `--dataset-sizes`: Dataset dimensions as "samples:features" pairs
- `--gpu`: Enable GPU benchmarks
- `--output-dir`: Results directory (default: test/benchmarks/results)
- `--baseline`: Path to baseline for regression detection
- `--formats`: Report formats (text,markdown,html)
- `--save-baseline`: Save results as new baseline

## Benchmark Components

### 1. GPU Profiling

Measures GPU performance using CUDA events and NVML:

```julia
using .GPUProfiler

# Profile GPU operation
result = profile_gpu_operation("matrix_multiply", () -> begin
    A = CUDA.rand(1000, 1000)
    B = CUDA.rand(1000, 1000)
    C = A * B
    CUDA.synchronize()
end)

# Analyze results
analysis = analyze_profile(result)
println("Avg GPU Utilization: $(analysis["avg_utilization"])%")
println("Peak Memory: $(analysis["memory_peak_mb"]) MB")
```

**Metrics Collected:**
- Execution time (ms)
- GPU utilization (%)
- Memory usage (MB)
- Temperature (°C)
- Power consumption (W)
- Kernel launch count

### 2. Memory Profiling

Tracks host and device memory allocations:

```julia
using .MemoryProfiler

# Profile memory usage
profile = profile_memory("feature_selection", () -> begin
    X = randn(10000, 5000)
    filter = VarianceFilter()
    selected = fit_transform(filter, X, y)
    return X[:, selected]
end)

# Generate summary
summary = memory_summary(profile)
detect_memory_leaks(profile)
```

**Capabilities:**
- Host/device memory tracking
- Allocation pattern analysis
- Memory leak detection
- GC event monitoring
- Peak memory identification

### 3. Latency Profiling

Measures operation latencies with statistical analysis:

```julia
using .LatencyProfiler

# Benchmark metamodel inference
results = benchmark_metamodel_inference(
    metamodel_predict,
    input_sizes=[100, 500, 1000, 5000],
    batch_sizes=[1, 10, 100]
)

# Generate latency report
latency_report(results["input_1000"], target_latency_us=1000.0)
```

**Features:**
- Percentile analysis (P50, P90, P95, P99)
- Outlier detection
- Throughput calculation
- Jitter analysis
- Batch size scaling

### 4. Regression Detection

Automatically identifies performance regressions:

```julia
using .RegressionDetector

# Create baseline
baseline = create_baseline(current_metrics)
save_baseline(baseline, "baseline.json")

# Later: Compare with baseline
report = compare_with_baseline("baseline.json", new_metrics)
generate_regression_report(report)
```

**Thresholds:**
- Minor: 5% regression
- Moderate: 10% regression  
- Severe: 20% regression

## Pipeline Benchmarks

### Stage 1: Statistical Filtering

**Performance Targets:**
- Runtime: <30s for 5000 features
- Memory: <2GB host memory
- Feature reduction: 10:1 ratio

### Stage 2: MCTS Feature Selection

**Performance Targets:**
- GPU utilization: >80%
- Runtime: <120s for 500 features
- Memory: <4GB GPU memory
- Metamodel inference: <1ms per combination

### Stage 3: Ensemble Optimization

**Performance Targets:**
- Runtime: <20s for 50 features
- Accuracy: >0.95 correlation with full model
- Memory: <1GB host memory
- Feature stability: >0.8 across runs

## Output Reports

### Text Report

Basic performance summary with key metrics:

```
HSOF Performance Benchmark
==========================
Generated: 2024-01-15 10:30:00

GPU Performance:
  matmul_10000x5000:
    Execution Time: 125.3 ms
    Avg Utilization: 87.2%
    Peak Memory: 1024.5 MB

Memory Usage:
  pipeline_10000x5000:
    Host Peak: 512.3 MB
    Device Peak: 2048.7 MB
```

### Markdown Report

Formatted tables and sections for documentation:

```markdown
# HSOF Performance Benchmark

## GPU Performance

| Operation | Time (ms) | Utilization (%) | Memory (MB) |
|-----------|-----------|-----------------|-------------|
| matmul    | 125.3     | 87.2           | 1024.5      |
```

### HTML Report

Interactive report with charts (requires Chart.js):

- Performance visualizations
- Latency distribution graphs
- Memory usage over time
- Regression highlights

## Best Practices

### 1. Baseline Management

```bash
# Create initial baseline
julia benchmark_runner.jl --save-baseline --name initial_baseline

# Regular regression checks
julia benchmark_runner.jl --baseline results/baseline_initial.json
```

### 2. Dataset Size Selection

Choose sizes that represent your use cases:

```bash
# Small, medium, large datasets
--dataset-sizes "100:50,1000:500,10000:5000"
```

### 3. Multi-Run Averaging

For stable results, run multiple times:

```julia
# In custom script
results = []
for i in 1:5
    push!(results, run_benchmarks(config))
end
average_results = aggregate_results(results)
```

## Troubleshooting

### CUDA Not Available

```
Error: CUDA not functional
Solution: 
- Check CUDA installation
- Verify GPU drivers
- Set CUDA_VISIBLE_DEVICES
```

### Memory Profiling Issues

```
Error: Cannot read /proc/meminfo
Solution:
- Linux-specific feature
- Falls back to Julia GC stats
```

### High Latency Variance

```
Warning: High coefficient of variation
Solution:
- Increase warmup iterations
- Check for background processes
- Use CPU affinity
```

## Performance Optimization Tips

1. **GPU Utilization**
   - Batch operations when possible
   - Use appropriate data types (Float32 vs Float64)
   - Minimize CPU-GPU transfers

2. **Memory Usage**
   - Preallocate arrays
   - Use views instead of copies
   - Clear intermediate results

3. **Latency Reduction**
   - Profile hot paths
   - Cache metamodel predictions
   - Use batch inference

## CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
- name: Run Performance Benchmarks
  run: |
    julia test/benchmarks/benchmark_runner.jl \
      --stages pipeline \
      --baseline ${{ github.event.before }}.json \
      --save-baseline
      
- name: Check Regressions
  run: |
    if grep -q "SEVERE REGRESSIONS" results/latest.txt; then
      exit 1
    fi
```

## Contributing

When adding new benchmarks:

1. Create module in appropriate directory
2. Export measurement functions
3. Add to benchmark_runner.jl
4. Update performance targets
5. Document metrics collected

## Future Enhancements

- [ ] GPU memory bandwidth profiling
- [ ] Network I/O benchmarking (for distributed)
- [ ] Energy efficiency metrics
- [ ] Comparative analysis with baseline methods
- [ ] Automated performance tuning recommendations