# Hyperparameter Optimization System Documentation

## Overview

The Hyperparameter Optimization System provides automated tuning for learning rate, batch size, replay buffer sampling, and network architecture parameters using Bayesian optimization with parallel evaluation, early stopping based on correlation metrics, and comprehensive experiment tracking.

## Key Features

- **Bayesian Optimization Framework**: Gaussian Process surrogate model with Expected Improvement acquisition function
- **Comprehensive Search Space**: Learning rate, batch size, dropout, architecture, and training parameters
- **Parallel Evaluation**: Support for multi-GPU parallel hyperparameter trials
- **Early Stopping**: Intelligent termination based on correlation thresholds and improvement stagnation
- **Experiment Tracking**: Local JSON logging with optional MLflow integration
- **Automated Architecture Search**: Layer size and attention head optimization

## Architecture

### Core Components

1. **HyperparameterSpace**: Defines search bounds and discrete options for all parameters
2. **BayesianOptimizer**: Main optimization coordinator with Gaussian Process surrogate
3. **GaussianProcess**: Surrogate model for predicting hyperparameter performance
4. **TrialResult**: Comprehensive trial tracking with metrics and metadata
5. **Experiment Tracking**: JSON-based logging with MLflow integration support

### Optimization Flow

```
Initialize Search Space → Random Exploration → Bayesian Optimization → 
Early Stopping Check → Best Configuration Selection → Experiment Logging
```

## Configuration

### HyperparameterSpace Parameters

```julia
struct HyperparameterSpace
    # Learning parameters
    learning_rate_bounds::Tuple{Float64, Float64}      # (1e-5, 1e-2)
    batch_size_options::Vector{Int}                    # [16, 32, 64, 128, 256]
    dropout_rate_bounds::Tuple{Float64, Float64}       # (0.0, 0.5)
    
    # Architecture parameters
    hidden_dim_bounds::Tuple{Int, Int}                 # (64, 512)
    attention_heads_options::Vector{Int}               # [4, 8, 12, 16]
    num_layers_bounds::Tuple{Int, Int}                 # (2, 6)
    
    # Training parameters
    replay_buffer_size_bounds::Tuple{Int, Int}         # (500, 2000)
    update_frequency_bounds::Tuple{Int, Int}           # (50, 200)
    warmup_steps_bounds::Tuple{Int, Int}               # (100, 1000)
    
    # Optimization parameters
    optimizer_type_options::Vector{String}            # ["Adam", "AdamW", "RMSprop"]
    scheduler_type_options::Vector{String}            # ["Cosine", "StepLR", "Plateau"]
    gradient_clip_bounds::Tuple{Float64, Float64}     # (0.1, 5.0)
end
```

### Default Configuration

```julia
space = create_hyperparameter_space(
    learning_rate_bounds = (1e-5, 1e-2),
    batch_size_options = [16, 32, 64, 128, 256],
    dropout_rate_bounds = (0.0, 0.5),
    hidden_dim_bounds = (64, 512),
    attention_heads_options = [4, 8, 12, 16],
    num_layers_bounds = (2, 6),
    optimizer_type_options = ["Adam", "AdamW", "RMSprop"],
    scheduler_type_options = ["Cosine", "StepLR", "Plateau"]
)
```

## Usage

### Basic Hyperparameter Optimization

```julia
using HyperparameterOptimization

# Define search space
space = create_hyperparameter_space()

# Create optimizer
optimizer = BayesianOptimizer(
    space,
    early_stopping_patience = 10,
    early_stopping_threshold = 0.01,
    min_correlation_threshold = 0.9,
    max_parallel_trials = 2,
    experiment_name = "metamodel_hyperopt",
    save_directory = "experiments"
)

# Define evaluation function
function evaluate_hyperparameters(config::HyperparameterConfig)
    # Train metamodel with given hyperparameters
    model = create_and_train_metamodel(config)
    
    # Evaluate correlation score
    correlation = evaluate_correlation(model, validation_data)
    
    return TrialResult(
        config, 
        correlation,
        training_time_seconds = training_time,
        memory_usage_mb = peak_memory,
        converged = training_converged,
        metadata = Dict("validation_size" => length(validation_data))
    )
end

# Run optimization
best_result = optimize_hyperparameters!(
    optimizer,
    evaluate_hyperparameters,
    n_initial_points = 10,
    n_optimization_iterations = 50,
    acquisition_samples = 1000
)

println("Best configuration: $(best_result.config.config_id)")
println("Best correlation: $(best_result.correlation_score)")
```

### Parallel Evaluation

```julia
# For parallel evaluation across multiple GPUs
optimizer = BayesianOptimizer(
    space,
    max_parallel_trials = 4,  # Use 4 parallel evaluations
    experiment_name = "parallel_hyperopt"
)

# Evaluation function with GPU assignment
function parallel_evaluate_hyperparameters(config::HyperparameterConfig)
    # Assign GPU based on worker ID or configuration
    gpu_id = assign_gpu_for_trial(config)
    
    # Train on assigned GPU
    model = create_and_train_metamodel(config, gpu_id = gpu_id)
    correlation = evaluate_correlation(model, validation_data)
    
    return TrialResult(config, correlation, metadata = Dict("gpu_id" => gpu_id))
end

best_result = optimize_hyperparameters!(
    optimizer,
    parallel_evaluate_hyperparameters,
    n_initial_points = 20,
    n_optimization_iterations = 100
)
```

### Custom Search Space

```julia
# Define custom search space for specific requirements
custom_space = create_hyperparameter_space(
    learning_rate_bounds = (1e-4, 1e-1),          # Wider learning rate range
    batch_size_options = [32, 64, 128],           # Fewer batch size options
    hidden_dim_bounds = (128, 256),               # Smaller networks
    attention_heads_options = [8],                # Fixed attention heads
    optimizer_type_options = ["Adam", "AdamW"]    # Only Adam variants
)

optimizer = BayesianOptimizer(custom_space)
```

## Bayesian Optimization Details

### Gaussian Process Surrogate Model

The system uses a Gaussian Process with RBF (Radial Basis Function) kernel to model the relationship between hyperparameters and correlation scores:

```julia
K(x₁, x₂) = σ² * exp(-0.5 * Σᵢ((x₁ᵢ - x₂ᵢ) / lᵢ)²)
```

Where:
- `σ²` is the signal variance
- `lᵢ` are the length scales for each dimension
- The kernel captures similarity between hyperparameter configurations

### Expected Improvement Acquisition Function

The Expected Improvement (EI) acquisition function balances exploration and exploitation:

```julia
EI(x) = (μ(x) - f⁺ - ξ) * Φ(z) + σ(x) * φ(z)
```

Where:
- `μ(x)` and `σ(x)` are the GP predictive mean and standard deviation
- `f⁺` is the best observed value
- `ξ` is the exploration parameter
- `Φ` and `φ` are the standard normal CDF and PDF

### Feature Encoding

Hyperparameters are encoded as normalized features for the Gaussian Process:

- **Continuous parameters**: Normalized to [0, 1] using bounds
- **Log-scale parameters**: Learning rate uses log-uniform distribution
- **Discrete parameters**: Encoded as categorical indices normalized by range
- **Total dimensions**: 12 features representing all hyperparameter types

## Early Stopping

### Convergence Criteria

The optimization stops early when:

1. **Correlation Threshold**: Best result exceeds `min_correlation_threshold` (default: 0.9)
2. **Improvement Stagnation**: No significant improvement over `early_stopping_patience` trials
3. **Improvement Threshold**: Recent improvement below `early_stopping_threshold` (default: 0.01)

### Configuration

```julia
optimizer = BayesianOptimizer(
    space,
    early_stopping_patience = 15,        # Stop after 15 trials without improvement
    early_stopping_threshold = 0.005,    # Minimum improvement threshold
    min_correlation_threshold = 0.95     # Target correlation for early termination
)
```

## Experiment Tracking

### Local JSON Logging

All trials are automatically logged to JSON files:

```julia
# Individual trial files
"trial_<config_id>.json" = {
    "config_id": "abc12345",
    "timestamp": "2024-01-15T10:30:00",
    "hyperparameters": {...},
    "results": {
        "correlation_score": 0.92,
        "training_time_seconds": 120.5,
        "memory_usage_mb": 2048.0,
        "converged": true
    },
    "metadata": {...}
}

# Complete experiment summary
"experiment_<name>.json" = {
    "experiment_name": "hyperopt_experiment",
    "search_space": {...},
    "trials": [...],
    "best_result": {...},
    "optimization_statistics": {...}
}
```

### MLflow Integration

Optional MLflow integration for enterprise experiment tracking:

```julia
# Requires MLflow package: Pkg.add("MLFlow")
optimizer = BayesianOptimizer(space, experiment_name = "mlflow_experiment")

# Automatically logs to MLflow if available:
# - Hyperparameters as parameters
# - Metrics (correlation, training time, memory usage)
# - Convergence and early stopping flags
```

### Experiment Reports

Generate comprehensive optimization reports:

```julia
report = generate_optimization_report(optimizer)
println(report)

# Output includes:
# - Summary statistics (total trials, best correlation, convergence rate)
# - Best configuration details
# - Top 5 configurations comparison
# - Search space coverage analysis
# - Performance trends and insights
```

## Performance Analysis

### Optimization Statistics

The system tracks comprehensive performance metrics:

```julia
# Access optimization statistics
stats = optimizer.best_result
println("Best correlation: $(stats.correlation_score)")
println("Training time: $(stats.training_time_seconds)s")
println("Memory usage: $(stats.memory_usage_mb)MB")
println("Converged: $(stats.converged)")

# Overall optimization metrics
total_trials = length(optimizer.trials)
total_time = sum(trial.training_time_seconds for trial in optimizer.trials)
convergence_rate = sum(trial.converged for trial in optimizer.trials) / total_trials

println("Total optimization time: $(total_time/3600) hours")
println("Convergence rate: $(convergence_rate*100)%")
```

### Search Space Coverage

Monitor exploration of the hyperparameter space:

```julia
# Analyze parameter coverage
learning_rates = [trial.config.learning_rate for trial in optimizer.trials]
batch_sizes = unique([trial.config.batch_size for trial in optimizer.trials])
optimizers = unique([trial.config.optimizer_type for trial in optimizer.trials])

println("Learning rate range: $(extrema(learning_rates))")
println("Batch sizes explored: $(batch_sizes)")
println("Optimizers tested: $(optimizers)")
```

## Integration with Metamodel Pipeline

### Metamodel Training Integration

```julia
function evaluate_metamodel_hyperparameters(config::HyperparameterConfig)
    # Create metamodel with specified architecture
    model_config = create_metamodel_config(
        hidden_dims = fill(config.hidden_dim, config.num_layers),
        attention_heads = config.attention_heads,
        dropout_rate = config.dropout_rate
    )
    
    model = create_metamodel(model_config)
    
    # Configure training with hyperparameters
    training_config = create_training_config(
        learning_rate = config.learning_rate,
        batch_size = config.batch_size,
        optimizer_type = config.optimizer_type,
        scheduler_type = config.scheduler_type,
        gradient_clip = config.gradient_clip
    )
    
    # Configure replay buffer
    buffer_config = create_replay_config(
        buffer_size = config.replay_buffer_size,
        update_frequency = config.update_frequency
    )
    
    # Train metamodel
    training_results = train_metamodel(
        model, training_config, buffer_config,
        warmup_steps = config.warmup_steps
    )
    
    # Evaluate correlation with ground truth
    correlation = evaluate_metamodel_correlation(model, validation_set)
    
    return TrialResult(
        config,
        correlation,
        training_time_seconds = training_results.total_time,
        memory_usage_mb = training_results.peak_memory,
        converged = training_results.converged,
        final_loss = training_results.final_loss,
        metadata = Dict(
            "validation_samples" => length(validation_set),
            "training_iterations" => training_results.iterations
        )
    )
end
```

### MCTS Integration

```julia
# Optimize hyperparameters for MCTS performance
function evaluate_mcts_performance(config::HyperparameterConfig)
    # Train metamodel with hyperparameters
    model = train_metamodel_with_config(config)
    
    # Integrate with MCTS
    mcts_config = create_mcts_config(
        metamodel = model,
        exploration_weight = 1.0,
        max_iterations = 1000
    )
    
    # Evaluate MCTS performance on benchmark problems
    performance_scores = []
    for problem in benchmark_problems
        mcts_result = run_mcts(problem, mcts_config)
        push!(performance_scores, mcts_result.solution_quality)
    end
    
    # Use average performance as optimization target
    avg_performance = mean(performance_scores)
    
    return TrialResult(
        config,
        avg_performance,
        metadata = Dict("benchmark_problems" => length(benchmark_problems))
    )
end
```

## Best Practices

### Optimization Strategy

1. **Start with Wide Search**: Use broad parameter bounds initially
2. **Random Exploration**: Allocate 10-20% of budget to random exploration
3. **Early Stopping**: Set reasonable correlation thresholds (0.9-0.95)
4. **Parallel Evaluation**: Use multiple GPUs for faster optimization
5. **Incremental Refinement**: Narrow search space based on initial results

### Parameter Selection Guidelines

- **Learning Rate**: Use log-uniform distribution (1e-5 to 1e-2)
- **Batch Size**: Test powers of 2 for memory efficiency
- **Architecture**: Start with moderate sizes (64-512 hidden dimensions)
- **Attention Heads**: Use multiples of 4 for tensor core efficiency
- **Dropout**: Conservative range (0.0-0.3) for stability

### Resource Management

```julia
# Monitor resource usage during optimization
function resource_aware_evaluation(config::HyperparameterConfig)
    # Check GPU memory before training
    available_memory = CUDA.available_memory()
    estimated_usage = estimate_memory_usage(config)
    
    if estimated_usage > available_memory
        # Skip configurations that won't fit in memory
        return TrialResult(
            config, 0.0,
            converged = false,
            metadata = Dict("memory_insufficient" => true)
        )
    end
    
    # Proceed with training
    return train_and_evaluate(config)
end
```

## Troubleshooting

### Common Issues

#### 1. Memory Constraints
```julia
# Reduce batch size for large architectures
space = create_hyperparameter_space(
    batch_size_options = [16, 32, 64],  # Smaller batch sizes
    hidden_dim_bounds = (64, 256)      # Smaller networks
)
```

#### 2. Slow Convergence
```julia
# Increase exploration budget
optimizer = BayesianOptimizer(
    space,
    early_stopping_patience = 20,      # More patience
    min_correlation_threshold = 0.85   # Lower threshold
)
```

#### 3. Poor Initial Performance
```julia
# Use more random exploration
best_result = optimize_hyperparameters!(
    optimizer,
    evaluation_function,
    n_initial_points = 20,             # More random points
    n_optimization_iterations = 30    # Fewer optimization steps
)
```

### Performance Monitoring

```julia
# Monitor optimization progress
function monitor_optimization_progress(optimizer::BayesianOptimizer)
    best_scores = []
    for i in 1:length(optimizer.trials)
        current_best = maximum([trial.correlation_score for trial in optimizer.trials[1:i]])
        push!(best_scores, current_best)
    end
    
    # Plot improvement over time
    plot_optimization_progress(best_scores)
end
```

## Future Enhancements

### Planned Features

1. **Multi-Objective Optimization**: Optimize for correlation AND speed
2. **Transfer Learning**: Use previous experiments to warm-start optimization
3. **Ensemble Methods**: Combine multiple optimization strategies
4. **Dynamic Search Space**: Adapt bounds based on early results
5. **Hierarchical Optimization**: Optimize architecture first, then training parameters

### Research Directions

1. **Advanced Acquisition Functions**: UCB, Thompson Sampling, Information Gain
2. **Multi-Fidelity Optimization**: Use cheaper approximations for screening
3. **Contextual Bandits**: Adapt to different problem types
4. **Neural Architecture Search**: Automated architecture design
5. **Hyperparameter Sensitivity Analysis**: Understand parameter importance

## Conclusion

The Hyperparameter Optimization System provides a comprehensive, scalable solution for automated hyperparameter tuning in the metamodel training pipeline. With Bayesian optimization, intelligent early stopping, and comprehensive experiment tracking, it enables efficient discovery of optimal configurations while minimizing computational overhead.

Key achievements:
- ✅ Bayesian optimization framework with Gaussian Process surrogate model
- ✅ Comprehensive hyperparameter search space covering all key parameters  
- ✅ Expected Improvement acquisition function for balanced exploration
- ✅ Intelligent early stopping based on correlation metrics
- ✅ Parallel evaluation support for multi-GPU optimization
- ✅ Comprehensive experiment tracking with JSON and MLflow integration
- ✅ Automated architecture search for neural network parameters
- ✅ Resource-aware evaluation with memory and time constraints
- ✅ Integration-ready for metamodel training and MCTS pipelines