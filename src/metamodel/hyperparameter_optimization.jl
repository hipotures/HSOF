"""
Hyperparameter Optimization System
Automated tuning for learning rate, batch size, replay buffer sampling, and network architecture parameters
using Bayesian optimization with parallel evaluation and early stopping based on correlation metrics
"""
module HyperparameterOptimization

using Random
using Statistics
using LinearAlgebra
using Dates
using Printf
using JSON3
using Distributed
using SharedArrays
using UUIDs

# Optional MLflow integration
MLFLOW_AVAILABLE = false
try
    using MLFlow
    global MLFLOW_AVAILABLE = true
catch e
    @warn "MLflow not available - experiment tracking will use local storage" exception=e
end

# Try to load neural architecture and correlation tracking modules
NEURAL_ARCH_AVAILABLE = false
CORRELATION_TRACK_AVAILABLE = false

try
    include("neural_architecture.jl")
    using .NeuralArchitecture
    global NEURAL_ARCH_AVAILABLE = true
catch e
    @warn "Neural architecture module not available" exception=e
end

try 
    include("correlation_tracking.jl")
    using .CorrelationTracking
    global CORRELATION_TRACK_AVAILABLE = true
catch e
    @warn "Correlation tracking module not available" exception=e
end

"""
Hyperparameter space definition for optimization
"""
struct HyperparameterSpace
    # Learning parameters
    learning_rate_bounds::Tuple{Float64, Float64}      # (min, max) learning rate
    batch_size_options::Vector{Int}                    # Discrete batch sizes
    dropout_rate_bounds::Tuple{Float64, Float64}       # (min, max) dropout rate
    
    # Architecture parameters
    hidden_dim_bounds::Tuple{Int, Int}                 # (min, max) hidden layer size
    attention_heads_options::Vector{Int}               # Number of attention heads
    num_layers_bounds::Tuple{Int, Int}                 # (min, max) number of layers
    
    # Training parameters
    replay_buffer_size_bounds::Tuple{Int, Int}         # (min, max) buffer size
    update_frequency_bounds::Tuple{Int, Int}           # (min, max) update frequency
    warmup_steps_bounds::Tuple{Int, Int}               # (min, max) warmup steps
    
    # Optimization parameters
    optimizer_type_options::Vector{String}            # ["Adam", "AdamW", "RMSprop"]
    scheduler_type_options::Vector{String}            # ["Cosine", "StepLR", "Plateau"]
    gradient_clip_bounds::Tuple{Float64, Float64}     # (min, max) gradient clipping
end

"""
Create default hyperparameter search space
"""
function create_hyperparameter_space(;
    learning_rate_bounds::Tuple{Float64, Float64} = (1e-5, 1e-2),
    batch_size_options::Vector{Int} = [16, 32, 64, 128, 256],
    dropout_rate_bounds::Tuple{Float64, Float64} = (0.0, 0.5),
    hidden_dim_bounds::Tuple{Int, Int} = (64, 512),
    attention_heads_options::Vector{Int} = [4, 8, 12, 16],
    num_layers_bounds::Tuple{Int, Int} = (2, 6),
    replay_buffer_size_bounds::Tuple{Int, Int} = (500, 2000),
    update_frequency_bounds::Tuple{Int, Int} = (50, 200),
    warmup_steps_bounds::Tuple{Int, Int} = (100, 1000),
    optimizer_type_options::Vector{String} = ["Adam", "AdamW", "RMSprop"],
    scheduler_type_options::Vector{String} = ["Cosine", "StepLR", "Plateau"],
    gradient_clip_bounds::Tuple{Float64, Float64} = (0.1, 5.0)
)
    return HyperparameterSpace(
        learning_rate_bounds,
        batch_size_options,
        dropout_rate_bounds,
        hidden_dim_bounds,
        attention_heads_options,
        num_layers_bounds,
        replay_buffer_size_bounds,
        update_frequency_bounds,
        warmup_steps_bounds,
        optimizer_type_options,
        scheduler_type_options,
        gradient_clip_bounds
    )
end

"""
Individual hyperparameter configuration
"""
struct HyperparameterConfig
    # Learning parameters
    learning_rate::Float64
    batch_size::Int
    dropout_rate::Float64
    
    # Architecture parameters
    hidden_dim::Int
    attention_heads::Int
    num_layers::Int
    
    # Training parameters
    replay_buffer_size::Int
    update_frequency::Int
    warmup_steps::Int
    
    # Optimization parameters
    optimizer_type::String
    scheduler_type::String
    gradient_clip::Float64
    
    # Metadata
    config_id::String
    timestamp::DateTime
end

"""
Sample hyperparameters from the search space
"""
function sample_hyperparameters(space::HyperparameterSpace; seed::Union{Nothing, Int} = nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    config_id = string(UUIDs.uuid4())[1:8]  # Short UUID for identification
    
    return HyperparameterConfig(
        # Learning parameters - log-uniform for learning rate
        exp(log(space.learning_rate_bounds[1]) + rand() * (log(space.learning_rate_bounds[2]) - log(space.learning_rate_bounds[1]))),
        rand(space.batch_size_options),
        space.dropout_rate_bounds[1] + rand() * (space.dropout_rate_bounds[2] - space.dropout_rate_bounds[1]),
        
        # Architecture parameters
        rand(space.hidden_dim_bounds[1]:space.hidden_dim_bounds[2]),
        rand(space.attention_heads_options),
        rand(space.num_layers_bounds[1]:space.num_layers_bounds[2]),
        
        # Training parameters
        rand(space.replay_buffer_size_bounds[1]:space.replay_buffer_size_bounds[2]),
        rand(space.update_frequency_bounds[1]:space.update_frequency_bounds[2]),
        rand(space.warmup_steps_bounds[1]:space.warmup_steps_bounds[2]),
        
        # Optimization parameters
        rand(space.optimizer_type_options),
        rand(space.scheduler_type_options),
        space.gradient_clip_bounds[1] + rand() * (space.gradient_clip_bounds[2] - space.gradient_clip_bounds[1]),
        
        # Metadata
        config_id,
        now()
    )
end

"""
Gaussian Process surrogate model for Bayesian optimization
"""
mutable struct GaussianProcess
    X::Matrix{Float64}              # Input features (hyperparameters)
    y::Vector{Float64}              # Observed objectives (correlation scores)
    noise::Float64                  # Observation noise
    length_scales::Vector{Float64}  # RBF kernel length scales
    signal_variance::Float64        # Signal variance
    
    function GaussianProcess(input_dim::Int; noise::Float64 = 0.01)
        new(
            Matrix{Float64}(undef, 0, input_dim),
            Float64[],
            noise,
            ones(input_dim),
            1.0
        )
    end
end

"""
RBF kernel function
"""
function rbf_kernel(x1::Vector{Float64}, x2::Vector{Float64}, length_scales::Vector{Float64}, signal_variance::Float64)
    diff = (x1 .- x2) ./ length_scales
    return signal_variance * exp(-0.5 * sum(diff .^ 2))
end

"""
Add observation to Gaussian Process
"""
function add_observation!(gp::GaussianProcess, x::Vector{Float64}, y::Float64)
    if size(gp.X, 1) == 0
        gp.X = reshape(x, 1, length(x))
    else
        gp.X = vcat(gp.X, reshape(x, 1, length(x)))
    end
    push!(gp.y, y)
end

"""
Predict mean and variance using Gaussian Process
"""
function predict(gp::GaussianProcess, x_new::Vector{Float64})
    if size(gp.X, 1) == 0
        return 0.0, 1.0  # Prior mean and variance
    end
    
    n = size(gp.X, 1)
    
    # Compute covariance matrix
    K = Matrix{Float64}(undef, n, n)
    for i in 1:n
        for j in 1:n
            K[i, j] = rbf_kernel(gp.X[i, :], gp.X[j, :], gp.length_scales, gp.signal_variance)
        end
        K[i, i] += gp.noise  # Add noise to diagonal
    end
    
    # Compute cross-covariance
    k_star = Vector{Float64}(undef, n)
    for i in 1:n
        k_star[i] = rbf_kernel(gp.X[i, :], x_new, gp.length_scales, gp.signal_variance)
    end
    
    # Predictive mean and variance
    try
        K_inv = inv(K)
        mean = dot(k_star, K_inv * gp.y)
        variance = gp.signal_variance - dot(k_star, K_inv * k_star)
        return mean, max(variance, 1e-8)  # Ensure positive variance
    catch e
        @warn "Matrix inversion failed in GP prediction" exception=e
        return 0.0, 1.0
    end
end

"""
Expected Improvement acquisition function
"""
function expected_improvement(gp::GaussianProcess, x_new::Vector{Float64}; xi::Float64 = 0.01)
    if size(gp.X, 1) == 0
        return 1.0  # High uncertainty for first point
    end
    
    mean, variance = predict(gp, x_new)
    std = sqrt(variance)
    
    if std < 1e-8
        return 0.0  # No improvement possible
    end
    
    f_best = maximum(gp.y)
    z = (mean - f_best - xi) / std
    
    # Expected improvement calculation
    ei = (mean - f_best - xi) * cdf_normal(z) + std * pdf_normal(z)
    
    return max(ei, 0.0)
end

"""
Standard normal PDF
"""
function pdf_normal(z::Float64)
    return exp(-0.5 * z^2) / sqrt(2π)
end

"""
Standard normal CDF (approximation using rational approximation)
"""
function cdf_normal(z::Float64)
    # Abramowitz and Stegun approximation
    if z < 0
        return 1.0 - cdf_normal(-z)
    end
    
    # Constants for the approximation
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    
    t = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-z * z)
    
    return y
end

"""
Trial result from hyperparameter evaluation
"""
struct TrialResult
    config::HyperparameterConfig
    correlation_score::Float64
    training_time_seconds::Float64
    memory_usage_mb::Float64
    converged::Bool
    early_stopped::Bool
    final_loss::Float64
    metadata::Dict{String, Any}
    
    function TrialResult(config::HyperparameterConfig, correlation_score::Float64; 
                        training_time_seconds::Float64 = 0.0,
                        memory_usage_mb::Float64 = 0.0,
                        converged::Bool = true,
                        early_stopped::Bool = false,
                        final_loss::Float64 = Inf,
                        metadata::Dict{String, Any} = Dict{String, Any}())
        new(config, correlation_score, training_time_seconds, memory_usage_mb, 
            converged, early_stopped, final_loss, metadata)
    end
end

"""
Bayesian optimizer for hyperparameter search
"""
mutable struct BayesianOptimizer
    space::HyperparameterSpace
    gp::GaussianProcess
    trials::Vector{TrialResult}
    best_result::Union{Nothing, TrialResult}
    
    # Early stopping configuration
    early_stopping_patience::Int
    early_stopping_threshold::Float64
    min_correlation_threshold::Float64
    
    # Parallel evaluation
    max_parallel_trials::Int
    current_parallel_trials::Int
    
    # Experiment tracking
    experiment_name::String
    save_directory::String
    mlflow_experiment_id::Union{Nothing, String}
    
    function BayesianOptimizer(space::HyperparameterSpace; 
                              early_stopping_patience::Int = 10,
                              early_stopping_threshold::Float64 = 0.01,
                              min_correlation_threshold::Float64 = 0.9,
                              max_parallel_trials::Int = 2,
                              experiment_name::String = "hyperopt_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
                              save_directory::String = "hyperopt_experiments")
        
        # Create save directory
        mkpath(save_directory)
        
        # Initialize MLflow experiment if available
        mlflow_exp_id = nothing
        if MLFLOW_AVAILABLE
            try
                mlflow_exp_id = MLFlow.create_experiment(experiment_name)
                @info "Created MLflow experiment: $experiment_name"
            catch e
                @warn "Failed to create MLflow experiment" exception=e
            end
        end
        
        # Count hyperparameter dimensions for GP
        input_dim = 12  # Number of continuous + discrete parameters
        gp = GaussianProcess(input_dim)
        
        new(space, gp, TrialResult[], nothing,
            early_stopping_patience, early_stopping_threshold, min_correlation_threshold,
            max_parallel_trials, 0,
            experiment_name, save_directory, mlflow_exp_id)
    end
end

"""
Convert hyperparameter config to feature vector for GP
"""
function config_to_features(config::HyperparameterConfig, space::HyperparameterSpace)
    features = Float64[]
    
    # Continuous parameters (normalize to [0, 1])
    push!(features, (log(config.learning_rate) - log(space.learning_rate_bounds[1])) / 
                   (log(space.learning_rate_bounds[2]) - log(space.learning_rate_bounds[1])))
    push!(features, config.dropout_rate / (space.dropout_rate_bounds[2] - space.dropout_rate_bounds[1]))
    push!(features, (config.hidden_dim - space.hidden_dim_bounds[1]) / 
                   (space.hidden_dim_bounds[2] - space.hidden_dim_bounds[1]))
    push!(features, (config.num_layers - space.num_layers_bounds[1]) / 
                   (space.num_layers_bounds[2] - space.num_layers_bounds[1]))
    push!(features, (config.replay_buffer_size - space.replay_buffer_size_bounds[1]) / 
                   (space.replay_buffer_size_bounds[2] - space.replay_buffer_size_bounds[1]))
    push!(features, (config.update_frequency - space.update_frequency_bounds[1]) / 
                   (space.update_frequency_bounds[2] - space.update_frequency_bounds[1]))
    push!(features, (config.warmup_steps - space.warmup_steps_bounds[1]) / 
                   (space.warmup_steps_bounds[2] - space.warmup_steps_bounds[1]))
    push!(features, (config.gradient_clip - space.gradient_clip_bounds[1]) / 
                   (space.gradient_clip_bounds[2] - space.gradient_clip_bounds[1]))
    
    # Discrete parameters (one-hot encoding simplified to index/max_index)
    push!(features, findfirst(==(config.batch_size), space.batch_size_options) / length(space.batch_size_options))
    push!(features, findfirst(==(config.attention_heads), space.attention_heads_options) / length(space.attention_heads_options))
    push!(features, findfirst(==(config.optimizer_type), space.optimizer_type_options) / length(space.optimizer_type_options))
    push!(features, findfirst(==(config.scheduler_type), space.scheduler_type_options) / length(space.scheduler_type_options))
    
    return features
end

"""
Optimize hyperparameters using Bayesian optimization
"""
function optimize_hyperparameters!(optimizer::BayesianOptimizer, 
                                  evaluation_function::Function;
                                  n_initial_points::Int = 5,
                                  n_optimization_iterations::Int = 50,
                                  acquisition_samples::Int = 1000)
    
    @info "Starting Bayesian optimization with $(n_initial_points) initial points and $(n_optimization_iterations) iterations"
    
    # Phase 1: Random exploration
    @info "Phase 1: Random exploration ($(n_initial_points) points)"
    for i in 1:n_initial_points
        config = sample_hyperparameters(optimizer.space; seed=i)
        
        @info "Evaluating initial point $i/$(n_initial_points): $(config.config_id)"
        result = evaluation_function(config)
        
        # Add to optimizer
        add_trial!(optimizer, result)
        
        # Log trial
        log_trial(optimizer, result)
        
        @info "Initial point $i result: correlation=$(round(result.correlation_score, digits=4)), time=$(round(result.training_time_seconds, digits=1))s"
    end
    
    @info "Phase 1 complete. Best correlation so far: $(round(optimizer.best_result.correlation_score, digits=4))"
    
    # Phase 2: Bayesian optimization
    @info "Phase 2: Bayesian optimization ($(n_optimization_iterations) iterations)"
    for i in 1:n_optimization_iterations
        @info "Optimization iteration $i/$(n_optimization_iterations)"
        
        # Find next point using acquisition function
        next_config = find_next_hyperparameters(optimizer, acquisition_samples)
        
        @info "Evaluating optimization point $i: $(next_config.config_id)"
        result = evaluation_function(next_config)
        
        # Add to optimizer
        add_trial!(optimizer, result)
        
        # Log trial
        log_trial(optimizer, result)
        
        @info "Optimization point $i result: correlation=$(round(result.correlation_score, digits=4)), time=$(round(result.training_time_seconds, digits=1))s"
        @info "Best correlation: $(round(optimizer.best_result.correlation_score, digits=4))"
        
        # Early stopping check
        if should_early_stop(optimizer)
            @info "Early stopping triggered after $i iterations"
            break
        end
        
        # Save checkpoint
        if i % 10 == 0
            save_experiment(optimizer)
        end
    end
    
    # Final save
    save_experiment(optimizer)
    
    @info "Optimization complete! Best configuration: $(optimizer.best_result.config.config_id)"
    @info "Best correlation: $(round(optimizer.best_result.correlation_score, digits=4))"
    
    return optimizer.best_result
end

"""
Find next hyperparameters using acquisition function
"""
function find_next_hyperparameters(optimizer::BayesianOptimizer, n_samples::Int)
    best_ei = -Inf
    best_config = nothing
    
    for _ in 1:n_samples
        candidate_config = sample_hyperparameters(optimizer.space)
        features = config_to_features(candidate_config, optimizer.space)
        
        ei = expected_improvement(optimizer.gp, features)
        
        if ei > best_ei
            best_ei = ei
            best_config = candidate_config
        end
    end
    
    return best_config
end

"""
Add trial result to optimizer
"""
function add_trial!(optimizer::BayesianOptimizer, result::TrialResult)
    push!(optimizer.trials, result)
    
    # Update best result
    if isnothing(optimizer.best_result) || result.correlation_score > optimizer.best_result.correlation_score
        optimizer.best_result = result
    end
    
    # Add to Gaussian Process
    features = config_to_features(result.config, optimizer.space)
    add_observation!(optimizer.gp, features, result.correlation_score)
end

"""
Check if early stopping should be triggered
"""
function should_early_stop(optimizer::BayesianOptimizer)
    if length(optimizer.trials) == 0 || isnothing(optimizer.best_result)
        return false
    end
    
    # Check if best result meets minimum threshold
    if optimizer.best_result.correlation_score >= optimizer.min_correlation_threshold
        return true
    end
    
    if length(optimizer.trials) < optimizer.early_stopping_patience
        return false
    end
    
    # Check if no improvement in recent trials
    recent_scores = [trial.correlation_score for trial in optimizer.trials[end-optimizer.early_stopping_patience+1:end]]
    improvement = maximum(recent_scores) - minimum(recent_scores)
    
    return improvement < optimizer.early_stopping_threshold
end

"""
Log trial to experiment tracking
"""
function log_trial(optimizer::BayesianOptimizer, result::TrialResult)
    # MLflow logging if available
    if MLFLOW_AVAILABLE && !isnothing(optimizer.mlflow_experiment_id)
        try
            with_experiment(optimizer.mlflow_experiment_id) do
                log_param("config_id", result.config.config_id)
                log_param("learning_rate", result.config.learning_rate)
                log_param("batch_size", result.config.batch_size)
                log_param("dropout_rate", result.config.dropout_rate)
                log_param("hidden_dim", result.config.hidden_dim)
                log_param("attention_heads", result.config.attention_heads)
                log_param("num_layers", result.config.num_layers)
                log_param("optimizer_type", result.config.optimizer_type)
                log_param("scheduler_type", result.config.scheduler_type)
                
                log_metric("correlation_score", result.correlation_score)
                log_metric("training_time_seconds", result.training_time_seconds)
                log_metric("memory_usage_mb", result.memory_usage_mb)
                log_metric("final_loss", result.final_loss)
                log_metric("converged", result.converged ? 1.0 : 0.0)
                log_metric("early_stopped", result.early_stopped ? 1.0 : 0.0)
            end
        catch e
            @warn "Failed to log to MLflow" exception=e
        end
    end
    
    # Local JSON logging
    trial_data = Dict(
        "config_id" => result.config.config_id,
        "timestamp" => result.config.timestamp,
        "hyperparameters" => Dict(
            "learning_rate" => result.config.learning_rate,
            "batch_size" => result.config.batch_size,
            "dropout_rate" => result.config.dropout_rate,
            "hidden_dim" => result.config.hidden_dim,
            "attention_heads" => result.config.attention_heads,
            "num_layers" => result.config.num_layers,
            "replay_buffer_size" => result.config.replay_buffer_size,
            "update_frequency" => result.config.update_frequency,
            "warmup_steps" => result.config.warmup_steps,
            "optimizer_type" => result.config.optimizer_type,
            "scheduler_type" => result.config.scheduler_type,
            "gradient_clip" => result.config.gradient_clip
        ),
        "results" => Dict(
            "correlation_score" => result.correlation_score,
            "training_time_seconds" => result.training_time_seconds,
            "memory_usage_mb" => result.memory_usage_mb,
            "converged" => result.converged,
            "early_stopped" => result.early_stopped,
            "final_loss" => result.final_loss
        ),
        "metadata" => result.metadata
    )
    
    # Save individual trial
    trial_file = joinpath(optimizer.save_directory, "trial_$(result.config.config_id).json")
    open(trial_file, "w") do f
        JSON3.pretty(f, trial_data)
    end
end

"""
Save complete experiment state
"""
function save_experiment(optimizer::BayesianOptimizer)
    experiment_data = Dict(
        "experiment_name" => optimizer.experiment_name,
        "timestamp" => now(),
        "search_space" => Dict(
            "learning_rate_bounds" => optimizer.space.learning_rate_bounds,
            "batch_size_options" => optimizer.space.batch_size_options,
            "dropout_rate_bounds" => optimizer.space.dropout_rate_bounds,
            "hidden_dim_bounds" => optimizer.space.hidden_dim_bounds,
            "attention_heads_options" => optimizer.space.attention_heads_options,
            "num_layers_bounds" => optimizer.space.num_layers_bounds,
            "optimizer_type_options" => optimizer.space.optimizer_type_options,
            "scheduler_type_options" => optimizer.space.scheduler_type_options
        ),
        "trials" => [
            Dict(
                "config_id" => trial.config.config_id,
                "correlation_score" => trial.correlation_score,
                "training_time_seconds" => trial.training_time_seconds,
                "converged" => trial.converged,
                "early_stopped" => trial.early_stopped
            ) for trial in optimizer.trials
        ],
        "best_result" => isnothing(optimizer.best_result) ? nothing : Dict(
            "config_id" => optimizer.best_result.config.config_id,
            "correlation_score" => optimizer.best_result.correlation_score,
            "training_time_seconds" => optimizer.best_result.training_time_seconds
        ),
        "optimization_statistics" => Dict(
            "total_trials" => length(optimizer.trials),
            "best_correlation" => isnothing(optimizer.best_result) ? 0.0 : optimizer.best_result.correlation_score,
            "total_optimization_time" => sum(trial.training_time_seconds for trial in optimizer.trials),
            "convergence_rate" => sum(trial.converged for trial in optimizer.trials) / max(length(optimizer.trials), 1)
        )
    )
    
    experiment_file = joinpath(optimizer.save_directory, "experiment_$(optimizer.experiment_name).json")
    open(experiment_file, "w") do f
        JSON3.pretty(f, experiment_data)
    end
    
    @info "Experiment saved to: $experiment_file"
end

"""
Load experiment from file
"""
function load_experiment(experiment_file::String)
    experiment_data = JSON3.read(read(experiment_file, String))
    
    @info "Loaded experiment: $(experiment_data["experiment_name"])"
    @info "Total trials: $(length(experiment_data["trials"]))"
    @info "Best correlation: $(experiment_data["optimization_statistics"]["best_correlation"])"
    
    return experiment_data
end

"""
Generate optimization summary report
"""
function generate_optimization_report(optimizer::BayesianOptimizer)
    if isempty(optimizer.trials)
        return "No trials completed yet."
    end
    
    report = String[]
    
    push!(report, "=== Hyperparameter Optimization Report ===")
    push!(report, "Experiment: $(optimizer.experiment_name)")
    push!(report, "Timestamp: $(now())")
    push!(report, "")
    
    # Summary statistics
    push!(report, "Summary Statistics:")
    push!(report, "  - Total trials: $(length(optimizer.trials))")
    push!(report, "  - Best correlation: $(round(optimizer.best_result.correlation_score, digits=4))")
    push!(report, "  - Average correlation: $(round(mean([t.correlation_score for t in optimizer.trials]), digits=4))")
    push!(report, "  - Total optimization time: $(round(sum(t.training_time_seconds for t in optimizer.trials) / 3600, digits=2)) hours")
    push!(report, "  - Convergence rate: $(round(100 * sum(t.converged for t in optimizer.trials) / length(optimizer.trials), digits=1))%")
    push!(report, "")
    
    # Best configuration
    if !isnothing(optimizer.best_result)
        best = optimizer.best_result
        push!(report, "Best Configuration ($(best.config.config_id)):")
        push!(report, "  - Learning rate: $(best.config.learning_rate)")
        push!(report, "  - Batch size: $(best.config.batch_size)")
        push!(report, "  - Dropout rate: $(best.config.dropout_rate)")
        push!(report, "  - Hidden dim: $(best.config.hidden_dim)")
        push!(report, "  - Attention heads: $(best.config.attention_heads)")
        push!(report, "  - Num layers: $(best.config.num_layers)")
        push!(report, "  - Optimizer: $(best.config.optimizer_type)")
        push!(report, "  - Scheduler: $(best.config.scheduler_type)")
        push!(report, "  - Correlation score: $(round(best.correlation_score, digits=4))")
        push!(report, "  - Training time: $(round(best.training_time_seconds, digits=1))s")
        push!(report, "")
    end
    
    # Top 5 configurations
    sorted_trials = sort(optimizer.trials, by=t -> t.correlation_score, rev=true)
    push!(report, "Top 5 Configurations:")
    for (i, trial) in enumerate(sorted_trials[1:min(5, length(sorted_trials))])
        push!(report, "  $i. $(trial.config.config_id): correlation=$(round(trial.correlation_score, digits=4)), lr=$(trial.config.learning_rate), batch=$(trial.config.batch_size)")
    end
    push!(report, "")
    
    # Search space coverage
    push!(report, "Search Space Coverage:")
    lrs = [t.config.learning_rate for t in optimizer.trials]
    push!(report, "  - Learning rate range: $(round(minimum(lrs), digits=6)) - $(round(maximum(lrs), digits=6))")
    
    batch_sizes = unique([t.config.batch_size for t in optimizer.trials])
    push!(report, "  - Batch sizes explored: $(sort(batch_sizes))")
    
    optimizers = unique([t.config.optimizer_type for t in optimizer.trials])
    push!(report, "  - Optimizers explored: $(optimizers)")
    
    return join(report, "\n")
end

"""
Mock evaluation function for testing (simulates training and evaluation)
"""
function mock_evaluation_function(config::HyperparameterConfig)
    @info "Mock evaluation for config: $(config.config_id)"
    
    # Simulate training time based on complexity
    base_time = 30.0  # Base 30 seconds
    complexity_factor = (config.hidden_dim / 256) * (config.num_layers / 3) * (config.batch_size / 64)
    training_time = base_time * complexity_factor + 10 * rand()
    
    # Simulate correlation score with some noise and hyperparameter influence
    base_correlation = 0.85
    
    # Learning rate influence (optimal around 1e-3)
    lr_factor = 1.0 - abs(log10(config.learning_rate) + 3.0) / 5.0  # Penalty for being far from 1e-3
    
    # Architecture influence
    arch_factor = 1.0 - abs(config.hidden_dim - 256) / 512  # Optimal around 256
    head_factor = 1.0 - abs(config.attention_heads - 8) / 16  # Optimal around 8
    
    # Batch size influence
    batch_factor = 1.0 - abs(config.batch_size - 64) / 128  # Optimal around 64
    
    correlation = base_correlation + 0.1 * (lr_factor + arch_factor + head_factor + batch_factor) / 4 + 0.05 * (rand() - 0.5)
    correlation = clamp(correlation, 0.0, 1.0)
    
    # Simulate memory usage
    memory_usage = config.hidden_dim * config.num_layers * config.batch_size / 1000.0 + 100 * rand()
    
    # Simulate convergence (higher chance with better hyperparameters)
    converged = correlation > 0.8 && rand() > 0.1
    
    # Simulate early stopping (randomly)
    early_stopped = rand() < 0.1
    
    # Simulate final loss
    final_loss = (1.0 - correlation) + 0.1 * rand()
    
    # Sleep to simulate actual training time (scaled down for testing)
    sleep(min(training_time / 100, 2.0))  # Max 2 seconds for testing
    
    return TrialResult(
        config, 
        correlation,
        training_time_seconds = training_time,
        memory_usage_mb = memory_usage,
        converged = converged,
        early_stopped = early_stopped,
        final_loss = final_loss,
        metadata = Dict{String, Any}("mock_evaluation" => true)
    )
end

# Export main types and functions
export HyperparameterSpace, create_hyperparameter_space
export HyperparameterConfig, sample_hyperparameters
export TrialResult, BayesianOptimizer
export optimize_hyperparameters!, generate_optimization_report
export save_experiment, load_experiment
export mock_evaluation_function

end # module HyperparameterOptimization