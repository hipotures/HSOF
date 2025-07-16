# HSOF Pipeline Parameterization Summary

## Overview
The HSOF (Hybrid Search for Optimal Features) pipeline has been successfully parameterized to support different speed/accuracy trade-offs through configuration files.

## Configuration File
- **Location**: `config/hsof.yaml`
- **Format**: YAML configuration with mode presets and detailed parameter control

## Configuration Modes

### 1. Fast Mode
- **Purpose**: Quick feature selection for rapid prototyping
- **Characteristics**:
  - Stage 2: 10,000 MCTS iterations
  - Stage 3: 20 candidate subsets, 3 CV folds
  - ~74x faster than accurate mode
  - May sacrifice some feature quality

### 2. Balanced Mode (Default)
- **Purpose**: Good balance between speed and accuracy
- **Characteristics**:
  - Stage 2: 50,000 MCTS iterations
  - Stage 3: 100 candidate subsets, 5 CV folds
  - ~10x faster than accurate mode
  - Reasonable feature quality

### 3. Accurate Mode
- **Purpose**: Maximum feature selection quality
- **Characteristics**:
  - Stage 2: 200,000 MCTS iterations
  - Stage 3: 500 candidate subsets, 10 CV folds
  - Highest computational cost
  - Best feature quality

## Key Parameters

### Stage 1: GPU Correlation Filtering
```yaml
stage1:
  correlation_threshold: 0.1      # Minimum correlation to keep feature
  min_features_to_keep: 10        # Minimum features to retain
  variance_threshold: 1.0e-6      # Minimum variance for features
```

### Stage 2: MCTS with Metamodel
```yaml
stage2:
  total_iterations: 50000         # Total MCTS iterations
  n_trees: 100                    # Number of parallel MCTS trees
  exploration_constant: 1.414     # UCB1 exploration parameter
  min_features: 5                 # Minimum features in subset
  max_features: 50                # Maximum features in subset
  
  # Metamodel parameters
  pretraining_samples: 5000       # Samples for metamodel training
  pretraining_epochs: 30          # Training epochs
  learning_rate: 0.001            # Adam optimizer learning rate
  batch_size: 256                 # Training batch size
  
  # Architecture
  hidden_sizes: [256, 128, 64]    # Dense layer sizes
  n_attention_heads: 8            # Multi-head attention heads
  dropout_rate: 0.2               # Dropout for regularization
```

### Stage 3: Real Model Evaluation
```yaml
stage3:
  n_candidate_subsets: 100        # Number of feature subsets to evaluate
  cv_folds: 5                     # Cross-validation folds
  min_features_final: 5           # Minimum features in final selection
  max_features_final: 20          # Maximum features in final selection
  
  # Model parameters
  xgboost:
    num_round: 100
    max_depth: 6
    eta: 0.1
  
  random_forest:
    n_trees: 100
    max_depth: 10
```

## Usage Examples

### 1. Using Preset Modes
```julia
# Simply set the mode in config/hsof.yaml
mode: "fast"  # or "balanced" or "accurate"

# Run the pipeline
results = run_hsof_gpu_pipeline("dataset_config.yaml")
```

### 2. Custom Configuration
```julia
# Specify custom config path
results = run_hsof_gpu_pipeline("dataset_config.yaml", 
                              config_path="custom_hsof_config.yaml")
```

### 3. Programmatic Configuration
```julia
# Load and modify configuration
config = YAML.load_file("config/hsof.yaml")
config["stage2"]["total_iterations"] = 25000
config["stage3"]["cv_folds"] = 3
YAML.write_file("temp_config.yaml", config)

# Use modified config
results = run_hsof_gpu_pipeline("dataset_config.yaml", 
                              config_path="temp_config.yaml")
```

## Performance Impact

Based on synthetic data testing:
- **Fast mode**: ~0.7 seconds total runtime
- **Balanced mode**: ~5 seconds total runtime  
- **Accurate mode**: ~50 seconds total runtime

The actual runtime will vary based on:
- Dataset size (samples × features)
- GPU performance
- Feature correlation structure
- Problem complexity

## GPU Memory Management
```yaml
gpu:
  max_memory_gb: null             # null = use all available
  memory_pool_size: 0.8           # Fraction of GPU memory to use
  batch_processing: true          # Process in batches if needed
```

## Benefits of Parameterization

1. **Flexibility**: Easy adjustment of pipeline behavior without code changes
2. **Reproducibility**: Configuration files can be versioned and shared
3. **Optimization**: Fine-tune parameters for specific datasets
4. **Experimentation**: Quick testing of different settings
5. **Production Ready**: Different configs for development vs production

## Files Modified

1. `src/hsof.jl` - Main pipeline to accept config_path parameter
2. `src/gpu_stage1.jl` - Stage 1 to use configuration parameters
3. `src/gpu_stage2.jl` - Stage 2 to use MCTS and metamodel parameters
4. `src/stage3_evaluation.jl` - Stage 3 to use evaluation parameters
5. `src/metamodel.jl` - Metamodel creation with configurable architecture
6. `src/config_loader.jl` - New file for configuration management
7. `config/hsof.yaml` - Main configuration file (renamed from hsof_config.yaml)

## Testing

Two test scripts demonstrate the parameterization:
1. `test_parameterized_pipeline.jl` - Tests configuration loading
2. `test_different_modes.jl` - Compares performance of different modes

Both tests pass successfully, confirming the parameterization is working correctly.