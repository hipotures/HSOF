# HSOF - Hybrid Selection of Features

A high-performance GPU-accelerated feature selection system leveraging dual RTX 4090 GPUs for massive dataset processing.

## Overview

HSOF implements a three-stage hybrid feature selection pipeline:

1. **Stage 1 - Fast Filtering**: GPU-accelerated univariate filtering (5000→500 features in <30s)
2. **Stage 2 - GPU-MCTS Selection**: Monte Carlo Tree Search with neural network metamodel (500→50 features)
3. **Stage 3 - Precise Evaluation**: Full model evaluation with XGBoost/RandomForest (50→10-20 features)

## Key Features

- **Dual GPU Processing**: Optimized for 2x RTX 4090 without NVLink
- **GPU-Native MCTS**: Persistent CUDA kernels with lock-free tree operations
- **Neural Metamodel**: 1000x speedup over actual model training
- **Real-time Dashboard**: Rich console UI with GPU monitoring
- **SQLite Integration**: Seamless database connectivity for large datasets

## System Requirements

- 2x NVIDIA RTX 4090 GPUs (24GB VRAM each)
- CUDA 11.8+ with compute capability 8.9
- Julia 1.9+ with CUDA.jl support
- 64GB+ system RAM recommended
- Ubuntu 22.04+ or compatible Linux distribution

## Quick Start

```bash
# Clone the repository
git clone https://github.com/hipotures/HSOF.git
cd HSOF

# Initialize Julia environment
julia --project=.
julia> using Pkg; Pkg.instantiate()

# Run GPU validation
julia> include("scripts/validate_environment.jl")

# Run example
julia> include("examples/basic_feature_selection.jl")
```

## Project Structure

```
HSOF/
├── src/
│   ├── stages/          # Three-stage pipeline implementation
│   ├── gpu/            # CUDA kernels and GPU management
│   ├── metamodel/      # Neural network metamodel
│   ├── ui/             # Rich console dashboard
│   └── database/       # SQLite integration layer
├── test/               # Comprehensive test suite
├── docs/               # Documentation
├── benchmarks/         # Performance benchmarks
└── configs/            # Configuration files
```

## Performance Targets

- Stage 1: Process 1M samples × 5000 features → 500 features in <30 seconds
- Stage 2: MCTS ensemble with 100+ trees achieving >80% GPU utilization
- Stage 3: Full cross-validation on 50 features in <5 minutes
- Metamodel: >0.9 correlation with actual model scores at 1000x speedup

## Development

This project uses Task Master for project management. Run `task-master list` to see all tasks.

## License

[License information to be added]

## Contributors

- Project initialized with Task Master AI