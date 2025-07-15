# HSOF Integration Tests

## Overview

This directory contains comprehensive integration tests for the Hybrid Stage-Optimized Feature (HSOF) selection pipeline. The tests validate the complete end-to-end functionality across all three stages with reference datasets.

## Test Structure

```
test/integration/
├── data/                    # Test datasets and loaders
│   ├── dataset_loaders.jl  # Dataset loading utilities
│   ├── titanic.csv         # Titanic dataset (auto-downloaded)
│   ├── synthetic_dataset.csv # Generated synthetic data
│   └── datasets_info.txt   # Dataset metadata
├── fixtures/               # Expected outputs and configurations
│   └── expected_outputs.jl # Expected results for each stage
├── benchmarks/             # Performance benchmark results
├── pipeline_runner.jl      # Main pipeline test runner
├── result_validation.jl    # Result validation framework
├── test_integration_suite.jl # Main test suite
└── prepare_datasets.jl     # Dataset preparation script
```

## Running the Tests

### 1. Prepare Datasets

First, download and prepare all test datasets:

```bash
cd test/integration
julia prepare_datasets.jl
```

This will:
- Download the Titanic dataset (891×12)
- Load MNIST features subset (10K×784)
- Generate synthetic dataset (10K×5000)

### 2. Run Full Test Suite

```bash
julia test_integration_suite.jl
```

Or from the project root:

```bash
julia test/integration/test_integration_suite.jl
```

### 3. Run Specific Tests

```julia
using Test
include("test/integration/pipeline_runner.jl")
using .PipelineRunner

# Run single dataset test
datasets = load_all_reference_datasets()
result = run_pipeline_test(datasets["titanic"])

# Run with custom configuration
result = run_pipeline_test(datasets["synthetic"], verbose=true, validate_gpu=false)
```

## Test Coverage

### Datasets

1. **Titanic** (Small dataset)
   - 891 samples × 12 features
   - Tests basic functionality
   - Expected: 12 → 12 → 8 → 5 features
   - Runtime: <1 minute

2. **MNIST** (Medium dataset)
   - 10,000 samples × 784 features
   - Tests dimensionality reduction
   - Expected: 784 → 200 → 50 → 20 features
   - Runtime: <5 minutes

3. **Synthetic** (Large dataset)
   - 10,000 samples × 5,000 features
   - Tests scalability and GPU acceleration
   - Expected: 5000 → 500 → 50 → 15 features
   - Runtime: <10 minutes

### Test Categories

1. **End-to-End Pipeline Tests**
   - Complete pipeline execution
   - Feature reduction validation
   - Quality score progression

2. **Performance Constraint Tests**
   - Runtime limits per stage
   - Memory usage limits
   - GPU utilization (when available)

3. **Quality Validation Tests**
   - Feature relevance scores
   - Redundancy minimization
   - Coverage metrics
   - Comparison with baseline methods

4. **Stability Tests**
   - Multiple run consistency
   - Feature selection stability
   - Result reproducibility

5. **Parametrized Tests**
   - Different dataset sizes
   - Various feature counts
   - Scalability validation

## Expected Outputs

### Stage 1: Statistical Filtering
- **Titanic**: All 12 features (too small to filter)
- **MNIST**: Top 200 most variable pixels
- **Synthetic**: Top 500 features by correlation

### Stage 2: MCTS Feature Selection
- **Titanic**: 8 most important features
- **MNIST**: 50 discriminative features
- **Synthetic**: 50 features via GPU-accelerated MCTS

### Stage 3: Ensemble Optimization
- **Titanic**: Final 5 features
- **MNIST**: Final 20 features
- **Synthetic**: Final 15 features

## Performance Targets

| Dataset   | Stage 1 | Stage 2 | Stage 3 | Total  |
|-----------|---------|---------|---------|--------|
| Titanic   | <5s     | <10s    | <5s     | <30s   |
| MNIST     | <15s    | <60s    | <10s    | <2min  |
| Synthetic | <30s    | <120s   | <20s    | <3min  |

## GPU Requirements

- GPU tests are automatically skipped if CUDA is not available
- Synthetic dataset tests benefit most from GPU acceleration
- Multi-GPU tests require 2+ GPUs with >8GB memory each

## Validation Criteria

Tests pass when:
1. Feature counts decrease monotonically through stages
2. Quality scores remain above thresholds (0.6-0.9 depending on dataset)
3. Runtime stays within limits
4. Memory usage stays within limits
5. Final features are better than or competitive with baseline methods

## Troubleshooting

### Dataset Download Issues
- Check internet connection
- Manually download from URLs in `dataset_loaders.jl`
- Place files in `test/integration/data/`

### Memory Issues
- Reduce dataset sizes in test configuration
- Disable GPU tests if insufficient VRAM
- Use subset of MNIST data

### GPU Issues
- Ensure CUDA.jl is properly installed
- Check GPU drivers are up to date
- Set `validate_gpu=false` to skip GPU validation

## Output Files

- `test_summary_YYYYMMDD_HHMMSS.txt` - Test execution summary
- `synthetic_dataset_info.txt` - Synthetic dataset details
- Individual dataset CSVs for inspection