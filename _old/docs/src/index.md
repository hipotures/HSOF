# HSOF - Hybrid Selection of Features

```@meta
CurrentModule = HSOF
```

## Overview

HSOF (Hybrid Selection of Features) is a high-performance, GPU-accelerated feature selection system designed for large-scale machine learning applications. The system leverages dual NVIDIA RTX 4090 GPUs to efficiently process datasets with millions of samples and thousands of features.

## Key Features

- **GPU-Accelerated Processing**: Optimized for dual RTX 4090 GPUs without NVLink
- **Three-Stage Feature Selection Pipeline**: Progressive reduction from 5000→500→50→10-20 features
- **Advanced Algorithms**: Monte Carlo Tree Search (MCTS) with neural network metamodel
- **High Performance**: Designed for datasets with millions of samples
- **Flexible Configuration**: TOML-based configuration with environment-specific overrides
- **Comprehensive Testing**: Kernel benchmarking and performance validation

## Architecture

HSOF implements a sophisticated three-stage feature selection pipeline:

1. **Stage 1: Initial Filtering (5000 → 500 features)**
   - GPU-accelerated statistical filtering
   - Mutual information and correlation analysis
   - Parallel processing across feature subsets

2. **Stage 2: MCTS Selection (500 → 50 features)**
   - Monte Carlo Tree Search with UCB1 exploration
   - Neural network metamodel for evaluation
   - Distributed search across GPUs

3. **Stage 3: Final Refinement (50 → 10-20 features)**
   - Ensemble methods (XGBoost, Random Forest)
   - Cross-validation and stability analysis
   - Feature importance aggregation

## Quick Example

```julia
using HSOF

# Load configuration
config = HSOF.load_config("configs/config.toml")

# Initialize GPU devices
HSOF.initialize_devices(config)

# Load your data
X, y = load_data("path/to/data.csv")

# Run feature selection
selected_features = HSOF.select_features(X, y, config)

# Get detailed results
results = HSOF.get_selection_results()
```

## Documentation Structure

- **[Getting Started](@ref)**: Installation, requirements, and quick start guide
- **[Architecture](@ref)**: Detailed system design and algorithm descriptions
- **[API Reference](@ref)**: Complete API documentation for all modules
- **[Tutorials](@ref)**: Step-by-step guides and examples
- **[Benchmarks](@ref)**: Performance metrics and comparisons

## System Requirements

- **GPUs**: 2x NVIDIA RTX 4090 (24GB VRAM each) or equivalent
- **CUDA**: Version 11.8 or higher
- **Julia**: Version 1.9 or higher
- **RAM**: 64GB recommended
- **OS**: Linux (Ubuntu 20.04+ recommended)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please see our [Contributing Guide](https://github.com/your-org/HSOF/blob/main/CONTRIBUTING.md) for details.

## Index

```@index
```