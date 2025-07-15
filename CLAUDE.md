# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HSOF (Hybrid Search for Optimal Features) is a GPU-accelerated multi-stage feature selection platform that combines:
- **Stage 1**: Fast univariate filtering (5000→500 features in <30s)
- **Stage 2**: GPU-MCTS with metamodel evaluation (500→50 features)
- **Stage 3**: Precise evaluation with real models (50→10-20 features)

The system is optimized for dual RTX 4090 GPUs (without NVLink) and uses Julia with CUDA.jl for GPU computation.

## Essential Build & Test Commands

```bash
# Environment setup
julia --project=.
julia> using Pkg; Pkg.instantiate()

# Validate GPU environment
julia> include("scripts/validate_environment.jl")

# Run tests
julia> Pkg.test()                        # Run all tests
julia> Pkg.test(test_args=["unit"])     # Run only unit tests
julia> Pkg.test(test_args=["gpu"])      # Run only GPU tests

# Run specific test file
julia> include("test/ui/test_console_dashboard.jl")
julia> include("test/ui/test_realtime_update.jl")

# Run benchmarks
julia> include("benchmarks/run_benchmarks.jl")

# Check linting (Julia doesn't have standard linting, but you can check formatting)
julia> using JuliaFormatter
julia> format("src", verbose=true)
```

## High-Level Architecture

### Three-Stage Pipeline Architecture

1. **Stage 1 - Fast Filtering** (`src/stages/stage1_filter/`)
   - GPU-accelerated mutual information calculation
   - Correlation matrix computation using custom CUDA kernels
   - Variance thresholding for feature reduction
   - Achieves 5000→500 feature reduction in <30 seconds

2. **Stage 2 - GPU-MCTS** (`src/gpu/mcts/`)
   - Persistent CUDA kernels for tree operations
   - Lock-free parallel tree exploration
   - Neural metamodel for fast evaluation without model training
   - Ensemble of 100+ trees distributed across GPUs

3. **Stage 3 - Precise Evaluation** (`src/stage3_evaluation/`)
   - Real model training (XGBoost/RandomForest)
   - Full cross-validation on reduced feature set
   - Feature interaction analysis

### Key Architectural Components

#### GPU Architecture (`src/gpu/`)
- **Dual GPU Strategy**: Independent tree forests per GPU with minimal PCIe communication
- **Memory Layout**: Structure of Arrays (SoA) for coalesced memory access
- **Persistent Kernels**: Continuous GPU execution without CPU intervention
- **Synchronization**: Atomic operations and warp-level primitives

#### Metamodel System (`src/metamodel/`)
- Neural network that predicts model performance without actual training
- Architecture: Binary input → Dense layers → Multi-Head Attention → Score prediction
- 1000x faster than actual model training
- Online learning with experience replay buffer

#### Console Dashboard (`src/ui/`)
- Rich.jl-based terminal UI with real-time updates
- 2x3 grid layout: GPU status, progress, metrics, analysis, logs
- 100ms refresh rate with double buffering
- GPU monitoring integration

#### Database Integration (`src/database/`)
- SQLite connection with lazy loading
- Metadata-driven configuration from existing feature stores
- Minimal write frequency to avoid GPU stalls
- Checkpoint system for long-running searches

## Task Master Integration

This project uses Task Master for project management. Essential commands:

```bash
# View current tasks
task-master list
task-master next
task-master show <id>

# Update task progress
task-master update-subtask --id=<id> --prompt="implementation notes"
task-master set-status --id=<id> --status=done

# After completing a subtask, check and commit changes
git status
git add -A
git commit -m "feat: complete subtask <id>: <description>"
git push
```

## Performance Considerations

### GPU Memory Management
- Each RTX 4090 has 24GB VRAM
- Dataset duplicated on both GPUs for independent processing
- Memory pools configured in `configs/gpu_config.toml`
- Target: <8GB VRAM usage per GPU for 5000 features

### Optimization Targets
- Stage 2 GPU utilization: >80% sustained
- Memory bandwidth utilization: >80% efficiency
- PCIe transfer: >8GB/s between GPUs
- Metamodel accuracy: >0.9 correlation with true scores

### Critical Performance Paths
1. **Stage 1→2 Transition**: Minimize data movement, keep on GPU
2. **MCTS Tree Operations**: Lock-free atomics, warp-level reductions
3. **Metamodel Inference**: Batch 1000+ evaluations, use FP16 when possible
4. **Database I/O**: Async reads, batched writes every 1000 iterations

## Console Dashboard Details

The primary interface is a Rich terminal dashboard with these panels:
- **GPU Status** (2x): Utilization, memory, temperature, power
- **Progress**: Current stage, best scores, convergence tracking
- **Metrics**: Nodes/sec, bandwidth, cache hits
- **Analysis**: Feature importance, correlations
- **Log**: System events, warnings, checkpoints

Keyboard shortcuts: Q(uit), P(ause), S(ave), E(xport), C(onfig), H(elp)

## Development Workflow

1. **Check current task**: `task-master next`
2. **Review requirements**: `task-master show <id>`
3. **Implement feature**: Follow the architecture patterns above
4. **Test implementation**: Run relevant tests from test suite
5. **Update progress**: `task-master update-subtask --id=<id> --prompt="notes"`
6. **Complete task**: `task-master set-status --id=<id> --status=done`

## PRD Implementation Notes

Key requirements from `prd/gpu-mcts-prd.md`:
- Hybrid approach optimized for different scales (Stage 1: coarse, Stage 2: exploration, Stage 3: precise)
- Zero CPU-GPU transfer during MCTS exploration phase
- Metamodel must achieve >0.9 correlation with actual model scores
- Support existing SQLite feature databases with minimal modifications
- Real-time Rich console dashboard for monitoring