# HSOF - Hybrid Search for Optimal Features

A simplified 3-stage feature selection system implemented in Julia.

## Overview

HSOF implements a 3-stage feature selection pipeline:

1. **Stage 1**: Fast filtering (5000→500 features) - correlation, variance, mutual information
2. **Stage 2**: MCTS selection (500→50 features) - Monte Carlo Tree Search  
3. **Stage 3**: Final evaluation (50→15 features) - XGBoost, RandomForest, LightGBM

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd HSOF

# Install dependencies
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

### Usage

```bash
# Run complete pipeline
julia --project=. src/hsof.jl config/titanic_simple.yaml

# Run tests
julia --project=. test/test_stage1.jl
julia --project=. test/test_stage2.jl
julia --project=. test/test_stage3.jl
```

## Configuration

Create a YAML configuration file:

```yaml
name: titanic
description: 'Titanic survival prediction'
database:
  path: /path/to/dataset.sqlite
tables:
  train_features: train_features
target_column: Survived
id_columns:
  - PassengerId
problem_type: binary_classification
stage1_features: 20
stage2_features: 8
final_features: 4
```

## Project Structure

```
HSOF/
├── src/
│   ├── hsof.jl                    # Main entry point
│   ├── config_loader.jl           # YAML configuration loading
│   ├── data_loader.jl             # SQLite data loading
│   ├── stage1_filter.jl           # Fast filtering stage
│   ├── stage2_mcts.jl             # MCTS selection stage
│   └── stage3_evaluation.jl       # Final evaluation stage
├── test/
│   ├── test_stage1.jl
│   ├── test_stage2.jl
│   └── test_stage3.jl
├── config/
│   └── titanic_simple.yaml        # Example configuration
└── Project.toml                   # Dependencies
```

## Expected Output

```
Starting HSOF pipeline...
Config: config/titanic_simple.yaml
Loading dataset: titanic
Data loaded: 891 samples × 142 features

=== Stage 1: Fast Filtering ===
Input: 142 features → Target: 20 features
Selected 20 features
Top 5 features: [Pclass, Sex_encoded, Age_filled, Fare_log, Family_size]

=== Stage 2: MCTS Selection ===
Input: 20 features → Target: 8 features
MCTS iteration: 100/1000, best score: 0.7234
Selected 8 features
Best score: 0.7654

=== Stage 3: Final Evaluation ===
Input: 8 features → Target: 4 features
Final selection: 4 features
Best model: XGBoost
Best score: 0.8234
Selected features: [Pclass, Sex_encoded, Fare_log, Family_size]

=== HSOF COMPLETED ===
Original features: 142
Final features: 4
Reduction: 97.2%
Results saved: titanic_hsof_results.json
```

## Dependencies

- Julia 1.9+
- YAML.jl - Configuration loading
- SQLite.jl - Database connection
- DataFrames.jl - Data manipulation
- MLJ.jl - Machine learning models
- XGBoost.jl, DecisionTree.jl, LightGBM.jl - Model implementations
- Statistics.jl, StatsBase.jl - Statistical functions

## Advanced Features

For GPU acceleration and advanced features, see the complete implementation in `_old/` directory.