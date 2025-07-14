# Quick Start Guide

```@meta
CurrentModule = HSOF
```

## Overview

This guide will help you run your first feature selection task with HSOF in under 5 minutes.

## Basic Usage

### 1. Load HSOF

```julia
using HSOF
using DataFrames
using CSV

# Initialize HSOF with default configuration
HSOF.initialize()
```

### 2. Load Your Data

```julia
# From CSV file
data = CSV.read("your_data.csv", DataFrame)
X = Matrix(data[:, 1:end-1])  # Features
y = Vector(data[:, end])       # Target

# Or generate sample data
X, y = HSOF.generate_sample_data(
    n_samples=10000,
    n_features=5000,
    n_informative=50
)
```

### 3. Run Feature Selection

```julia
# Run with default settings
results = HSOF.select_features(X, y)

# Get selected feature indices
selected_indices = results.selected_indices
println("Selected $(length(selected_indices)) features: ", selected_indices[1:10], "...")

# Get feature importance scores
importance_scores = results.feature_scores
```

### 4. Evaluate Results

```julia
# View selection summary
HSOF.print_summary(results)

# Output:
# ═══════════════════════════════════════════════════
# HSOF Feature Selection Results
# ═══════════════════════════════════════════════════
# Initial features: 5000
# Selected features: 15
# 
# Stage 1 (Filtering): 5000 → 500 (10.0%)
#   Time: 2.34s
#   Method: Statistical filtering
# 
# Stage 2 (MCTS): 500 → 50 (10.0%)
#   Time: 45.67s
#   Method: Monte Carlo Tree Search
# 
# Stage 3 (Ensemble): 50 → 15 (30.0%)
#   Time: 8.91s
#   Method: XGBoost + Random Forest
# 
# Total time: 56.92s
# GPU memory peak: 4.2 GB
# ═══════════════════════════════════════════════════
```

## Advanced Usage

### Custom Configuration

```julia
# Load custom configuration
config = HSOF.load_config("configs/my_config.toml")

# Or create programmatically
config = HSOF.Config(
    # GPU settings
    gpu = HSOF.GPUConfig(
        device_ids = [0, 1],
        memory_limit_gb = 20.0,
        stream_count = 4
    ),
    
    # Algorithm settings
    algorithms = HSOF.AlgorithmConfig(
        filtering = HSOF.FilteringConfig(
            variance_threshold = 0.01,
            correlation_threshold = 0.95
        ),
        mcts = HSOF.MCTSConfig(
            n_iterations = 10000,
            exploration_weight = 1.414
        ),
        ensemble = HSOF.EnsembleConfig(
            models = ["xgboost", "random_forest"],
            cv_folds = 5
        )
    )
)

# Run with custom config
results = HSOF.select_features(X, y, config)
```

### Stage-by-Stage Execution

```julia
# Run stages individually for more control
pipeline = HSOF.Pipeline(config)

# Stage 1: Statistical Filtering
stage1_indices = pipeline.run_stage1(X, y)
X_filtered = X[:, stage1_indices]

# Stage 2: MCTS Search
stage2_indices = pipeline.run_stage2(X_filtered, y)
X_selected = X_filtered[:, stage2_indices]

# Stage 3: Ensemble Refinement
final_indices = pipeline.run_stage3(X_selected, y)

# Map back to original indices
original_indices = stage1_indices[stage2_indices[final_indices]]
```

### Batch Processing

```julia
# Process multiple datasets
datasets = ["data1.csv", "data2.csv", "data3.csv"]

results_dict = Dict{String, HSOF.FeatureSelectionResult}()

for dataset in datasets
    @info "Processing $dataset"
    
    # Load data
    data = CSV.read(dataset, DataFrame)
    X = Matrix(data[:, 1:end-1])
    y = Vector(data[:, end])
    
    # Run selection
    results_dict[dataset] = HSOF.select_features(X, y)
end

# Compare results
HSOF.compare_results(results_dict)
```

## Real-World Example

### Genomic Data Analysis

```julia
# Load genomic dataset (high dimensional)
using HDF5
data = h5open("genomic_data.h5", "r") do file
    X = read(file, "expressions")  # 10000 samples × 20000 genes
    y = read(file, "phenotypes")
    return X, y
end

# Configure for high-dimensional data
config = HSOF.Config()
config.algorithms.filtering.variance_threshold = 0.001
config.algorithms.mcts.n_iterations = 50000
config.gpu.memory_limit_gb = 22.0  # Conservative for 24GB GPUs

# Run selection
@time results = HSOF.select_features(X, y, config)

# Analyze selected genes
gene_names = ["BRCA1", "TP53", "EGFR", ...]  # Your gene names
selected_genes = gene_names[results.selected_indices]

println("Top selected genes:")
for (gene, score) in zip(selected_genes[1:10], results.feature_scores[1:10])
    println("  $gene: $(round(score, digits=3))")
end
```

### Financial Data Analysis

```julia
# Load financial features
data = CSV.read("financial_features.csv", DataFrame)

# Handle missing values
X = Matrix(coalesce.(data[:, 1:end-1], 0.0))
y = Vector(data.target)

# Configure for financial data
config = HSOF.Config()
config.algorithms.filtering.handle_missing = true
config.algorithms.ensemble.models = ["xgboost", "lightgbm"]  # Good for financial data

# Run selection with cross-validation
results = HSOF.select_features_cv(X, y, config, n_folds=10)

# Get stability scores (how often each feature was selected)
stability_scores = results.stability_scores
stable_features = findall(stability_scores .> 0.8)

println("Most stable features: ", feature_names[stable_features])
```

## Visualization

### Feature Importance Plot

```julia
using Plots

# Plot feature importance
bar(results.feature_scores[1:20],
    title="Top 20 Feature Importance Scores",
    xlabel="Feature Index",
    ylabel="Importance Score",
    legend=false)
```

### Pipeline Progress

```julia
# Monitor pipeline progress
HSOF.select_features(X, y) do progress
    println("Stage: $(progress.stage), Progress: $(progress.percent)%")
end
```

## Performance Tips

### 1. Memory Management

```julia
# For large datasets, use chunked processing
results = HSOF.select_features_chunked(
    X, y,
    chunk_size = 1000,  # Process 1000 samples at a time
    config = config
)
```

### 2. GPU Utilization

```julia
# Monitor GPU usage during execution
HSOF.with_gpu_monitoring() do
    results = HSOF.select_features(X, y)
end

# Output GPU statistics
# GPU 0: Avg utilization: 85%, Peak memory: 18.5 GB
# GPU 1: Avg utilization: 82%, Peak memory: 17.2 GB
```

### 3. Caching Results

```julia
# Cache intermediate results
HSOF.enable_caching("./cache")
results = HSOF.select_features(X, y)  # Results cached

# Subsequent runs with same data are faster
results = HSOF.select_features(X, y)  # Loaded from cache
```

## Common Issues

### Out of Memory

```julia
# If you get CUDAOutOfMemoryError:
config.gpu.memory_limit_gb = 18.0  # Reduce from 22.0
config.algorithms.mcts.batch_size = 16  # Reduce from 32
```

### Slow Performance

```julia
# Speed up at the cost of quality
config.algorithms.filtering.variance_threshold = 0.1  # More aggressive
config.algorithms.mcts.n_iterations = 1000  # Fewer iterations
config.algorithms.ensemble.cv_folds = 3  # Fewer CV folds
```

### Single GPU

```julia
# HSOF automatically detects and adapts
# But you can force single GPU mode:
config.gpu.device_ids = [0]
config.pipeline.single_gpu_mode = true
```

## Next Steps

- [Tutorials](@ref): In-depth tutorials for specific use cases
- [API Reference](@ref): Complete API documentation
- [Configuration Guide](@ref): Detailed configuration options
- [Performance Tuning](@ref): Optimize for your hardware