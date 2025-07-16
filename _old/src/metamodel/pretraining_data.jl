module PretrainingData

using Random
using Statistics
using DataFrames
using MLJ
using MLJXGBoostInterface
using MLJDecisionTreeInterface
using Distributed
using HDF5
using ProgressMeter
using Printf
using StatsBase: sample  # For sampling functions

"""
Configuration for pre-training data generation
"""
struct PretrainingConfig
    n_samples::Int                  # Number of training samples to generate
    n_features::Int                 # Total number of features in dataset
    min_features::Int               # Minimum features per combination
    max_features::Int               # Maximum features per combination
    n_cv_folds::Int                 # Number of cross-validation folds
    n_parallel_workers::Int         # Number of parallel workers
    use_xgboost::Bool              # Whether to use XGBoost
    use_random_forest::Bool        # Whether to use RandomForest
    augmentation_factor::Float64   # Data augmentation multiplier
    output_path::String            # Path to save HDF5 file
end

"""
Create default pre-training configuration
"""
function create_pretraining_config(;
    n_samples::Int = 10000,
    n_features::Int = 500,
    min_features::Int = 10,
    max_features::Int = 100,
    n_cv_folds::Int = 5,
    n_parallel_workers::Int = Sys.CPU_THREADS,
    use_xgboost::Bool = true,
    use_random_forest::Bool = true,
    augmentation_factor::Float64 = 1.5,
    output_path::String = "metamodel_pretraining_data.h5"
)
    return PretrainingConfig(
        n_samples,
        n_features,
        min_features,
        max_features,
        n_cv_folds,
        n_parallel_workers,
        use_xgboost,
        use_random_forest,
        augmentation_factor,
        output_path
    )
end

"""
Structure to hold a feature combination and its scores
"""
struct FeatureCombination
    indices::Vector{Int32}           # Selected feature indices
    xgboost_score::Float32          # XGBoost CV score
    rf_score::Float32               # RandomForest CV score
    avg_score::Float32              # Average of both scores
    n_features::Int32               # Number of features selected
    generation_time::Float32        # Time to generate this sample
end

"""
Generate diverse feature combinations using stratified sampling
"""
function generate_diverse_combinations(
    config::PretrainingConfig,
    n_combinations::Int
)
    combinations = Vector{Vector{Int32}}()
    
    # Stratify by number of features
    n_features_range = config.min_features:config.max_features
    combinations_per_size = div(n_combinations, length(n_features_range))
    
    for n_feats in n_features_range
        for _ in 1:combinations_per_size
            # Generate random combination
            indices = sample(1:config.n_features, n_feats, replace=false)
            push!(combinations, Int32.(sort(indices)))
        end
    end
    
    # Add remaining combinations randomly
    while length(combinations) < n_combinations
        n_feats = rand(n_features_range)
        indices = sample(1:config.n_features, n_feats, replace=false)
        push!(combinations, Int32.(sort(indices)))
    end
    
    return combinations
end

"""
Generate synthetic dataset for testing
"""
function generate_synthetic_data(n_samples::Int, n_features::Int)
    # Create synthetic classification dataset
    # Features follow different distributions
    X = zeros(Float32, n_samples, n_features)
    
    # Create different feature types
    n_informative = div(n_features, 3)
    n_redundant = div(n_features, 3)
    n_noise = n_features - n_informative - n_redundant
    
    # Informative features
    for i in 1:n_informative
        if i <= div(n_informative, 2)
            # Linear features
            X[:, i] = randn(n_samples)
        else
            # Non-linear features
            X[:, i] = sin.(randn(n_samples) * 2) .+ 0.1 * randn(n_samples)
        end
    end
    
    # Redundant features (combinations of informative)
    for i in 1:n_redundant
        idx1 = rand(1:n_informative)
        idx2 = rand(1:n_informative)
        X[:, n_informative + i] = 0.7 * X[:, idx1] + 0.3 * X[:, idx2] + 0.1 * randn(n_samples)
    end
    
    # Noise features
    for i in 1:n_noise
        X[:, n_informative + n_redundant + i] = randn(n_samples)
    end
    
    # Generate target based on informative features
    y = zeros(Int, n_samples)
    
    # Create non-linear decision boundary
    for i in 1:n_samples
        score = sum(X[i, 1:5])  # Linear contribution
        score += sum(sin.(X[i, 6:10] * π))  # Non-linear contribution
        score += 0.5 * sum(X[i, 1:3] .* X[i, 4:6])  # Interaction terms
        
        # Add noise and threshold
        score += 0.3 * randn()
        y[i] = score > 0 ? 1 : 0
    end
    
    return X, y
end

"""
Train lightweight model and get CV score
"""
function train_and_evaluate(
    X::Matrix{Float32},
    y::Vector{Int},
    feature_indices::Vector{Int32},
    model_type::Symbol,
    n_folds::Int
)
    # Select features
    X_selected = X[:, feature_indices]
    
    # Create model
    if model_type == :xgboost
        model = XGBoostClassifier(
            max_depth=3,
            n_rounds=50,
            eta=0.3,
            subsample=0.8,
            colsample_bytree=0.8
        )
    elseif model_type == :random_forest
        model = RandomForestClassifier(
            n_trees=50,
            max_depth=5,
            min_samples_split=5,
            n_subfeatures=round(Int, sqrt(length(feature_indices)))
        )
    else
        error("Unknown model type: $model_type")
    end
    
    # Perform cross-validation
    cv_scores = Float32[]
    fold_size = div(size(X, 1), n_folds)
    
    for fold in 1:n_folds
        # Create train/test split
        test_start = (fold - 1) * fold_size + 1
        test_end = fold * fold_size
        test_idx = test_start:test_end
        train_idx = setdiff(1:size(X, 1), test_idx)
        
        # Train model
        mach = machine(model, X_selected[train_idx, :], y[train_idx])
        fit!(mach, verbosity=0)
        
        # Predict and evaluate
        y_pred = predict_mode(mach, X_selected[test_idx, :])
        accuracy = mean(y_pred .== y[test_idx])
        push!(cv_scores, accuracy)
    end
    
    return mean(cv_scores)
end

"""
Generate single training sample
"""
function generate_single_sample(
    X::Matrix{Float32},
    y::Vector{Int},
    feature_indices::Vector{Int32},
    config::PretrainingConfig
)
    start_time = time()
    
    # Train XGBoost
    xgb_score = config.use_xgboost ? 
        train_and_evaluate(X, y, feature_indices, :xgboost, config.n_cv_folds) : 
        Float32(0.5)
    
    # Train RandomForest
    rf_score = config.use_random_forest ? 
        train_and_evaluate(X, y, feature_indices, :random_forest, config.n_cv_folds) : 
        Float32(0.5)
    
    # Average score
    avg_score = (xgb_score + rf_score) / 2
    
    generation_time = Float32(time() - start_time)
    
    return FeatureCombination(
        feature_indices,
        xgb_score,
        rf_score,
        avg_score,
        Int32(length(feature_indices)),
        generation_time
    )
end

"""
Parallel data generation worker
"""
function generation_worker(
    worker_id::Int,
    combinations::Vector{Vector{Int32}},
    X::Matrix{Float32},
    y::Vector{Int},
    config::PretrainingConfig
)
    results = FeatureCombination[]
    
    for combo in combinations
        try
            result = generate_single_sample(X, y, combo, config)
            push!(results, result)
        catch e
            @warn "Worker $worker_id failed on combination: $e"
        end
    end
    
    return results
end

"""
Apply data augmentation to feature combinations
"""
function augment_combinations(
    combinations::Vector{Vector{Int32}},
    config::PretrainingConfig
)
    augmented = copy(combinations)
    n_augment = round(Int, length(combinations) * (config.augmentation_factor - 1))
    
    for _ in 1:n_augment
        # Select random combination
        base_combo = rand(combinations)
        n_base = length(base_combo)
        
        # Augmentation strategies
        strategy = rand(1:3)
        
        if strategy == 1 && n_base > config.min_features
            # Remove random features
            n_remove = rand(1:min(5, n_base - config.min_features))
            new_combo = setdiff(base_combo, sample(base_combo, n_remove, replace=false))
        elseif strategy == 2 && n_base < config.max_features
            # Add random features
            available = setdiff(1:config.n_features, base_combo)
            n_add = rand(1:min(5, config.max_features - n_base, length(available)))
            new_combo = vcat(base_combo, sample(available, n_add, replace=false))
        else
            # Swap features
            available = setdiff(1:config.n_features, base_combo)
            if !isempty(available)
                n_swap = rand(1:min(3, n_base, length(available)))
                to_remove = sample(base_combo, n_swap, replace=false)
                to_add = sample(available, n_swap, replace=false)
                new_combo = vcat(setdiff(base_combo, to_remove), to_add)
            else
                continue
            end
        end
        
        push!(augmented, Int32.(sort(new_combo)))
    end
    
    return augmented
end

"""
Save results to HDF5 file
"""
function save_to_hdf5(
    results::Vector{FeatureCombination},
    config::PretrainingConfig
)
    h5open(config.output_path, "w") do file
        # Save configuration
        attrs(file)["n_samples"] = length(results)
        attrs(file)["n_features"] = config.n_features
        attrs(file)["min_features"] = config.min_features
        attrs(file)["max_features"] = config.max_features
        attrs(file)["n_cv_folds"] = config.n_cv_folds
        
        # Create datasets
        n_results = length(results)
        max_features = maximum(r.n_features for r in results)
        
        # Feature indices (padded with -1)
        indices_data = fill(Int32(-1), n_results, max_features)
        for (i, r) in enumerate(results)
            indices_data[i, 1:length(r.indices)] = r.indices
        end
        
        write(file, "feature_indices", indices_data)
        write(file, "xgboost_scores", Float32[r.xgboost_score for r in results])
        write(file, "rf_scores", Float32[r.rf_score for r in results])
        write(file, "avg_scores", Float32[r.avg_score for r in results])
        write(file, "n_features", Int32[r.n_features for r in results])
        write(file, "generation_times", Float32[r.generation_time for r in results])
        
        # Save statistics
        write(file, "score_mean", Float32(mean(r.avg_score for r in results)))
        write(file, "score_std", Float32(std(r.avg_score for r in results)))
    end
end

"""
Main function to generate pre-training data
"""
function generate_pretraining_data(
    config::PretrainingConfig = create_pretraining_config();
    X::Union{Matrix{Float32}, Nothing} = nothing,
    y::Union{Vector{Int}, Nothing} = nothing
)
    println("Generating pre-training data for metamodel...")
    println("Configuration:")
    println("  - Samples: $(config.n_samples)")
    println("  - Features: $(config.n_features)")
    println("  - Feature range: [$(config.min_features), $(config.max_features)]")
    println("  - CV folds: $(config.n_cv_folds)")
    println("  - Workers: $(config.n_parallel_workers)")
    
    # Generate synthetic data if not provided
    if isnothing(X) || isnothing(y)
        println("\nGenerating synthetic dataset...")
        X, y = generate_synthetic_data(10000, config.n_features)
    end
    
    # Generate diverse combinations
    println("\nGenerating feature combinations...")
    combinations = generate_diverse_combinations(config, config.n_samples)
    
    # Apply augmentation
    println("Applying data augmentation...")
    combinations = augment_combinations(combinations, config)
    
    # Remove duplicates
    unique!(combinations)
    
    # Limit to requested number
    if length(combinations) > config.n_samples
        combinations = combinations[1:config.n_samples]
    end
    
    println("Generated $(length(combinations)) unique combinations")
    
    # Split work among workers
    if config.n_parallel_workers > 1
        println("\nStarting parallel generation with $(config.n_parallel_workers) workers...")
        
        # Add workers if needed
        if nworkers() < config.n_parallel_workers
            addprocs(config.n_parallel_workers - nworkers())
        end
        
        # Distribute work
        chunk_size = div(length(combinations), config.n_parallel_workers)
        chunks = [combinations[(i-1)*chunk_size+1 : min(i*chunk_size, end)] 
                 for i in 1:config.n_parallel_workers]
        
        # Parallel execution
        futures = [@spawnat :any generation_worker(i, chunks[i], X, y, config) 
                  for i in 1:length(chunks)]
        
        # Collect results with progress bar
        all_results = FeatureCombination[]
        p = Progress(length(futures), desc="Processing chunks: ")
        
        for f in futures
            chunk_results = fetch(f)
            append!(all_results, chunk_results)
            next!(p)
        end
    else
        # Sequential execution
        println("\nGenerating samples sequentially...")
        all_results = FeatureCombination[]
        p = Progress(length(combinations), desc="Generating samples: ")
        
        for combo in combinations
            result = generate_single_sample(X, y, combo, config)
            push!(all_results, result)
            next!(p)
        end
    end
    
    # Save results
    println("\nSaving results to $(config.output_path)...")
    save_to_hdf5(all_results, config)
    
    # Print summary statistics
    println("\nGeneration complete!")
    println("Summary statistics:")
    println("  - Total samples: $(length(all_results))")
    println("  - Average score: $(round(mean(r.avg_score for r in all_results), digits=3))")
    println("  - Score std: $(round(std(r.avg_score for r in all_results), digits=3))")
    println("  - Average features: $(round(mean(r.n_features for r in all_results), digits=1))")
    println("  - Total time: $(round(sum(r.generation_time for r in all_results) / 60, digits=1)) minutes")
    
    return all_results
end

"""
Load pre-training data from HDF5 file
"""
function load_pretraining_data(filepath::String)
    results = FeatureCombination[]
    
    h5open(filepath, "r") do file
        indices_data = read(file, "feature_indices")
        xgb_scores = read(file, "xgboost_scores")
        rf_scores = read(file, "rf_scores")
        avg_scores = read(file, "avg_scores")
        n_features = read(file, "n_features")
        gen_times = read(file, "generation_times")
        
        for i in 1:size(indices_data, 1)
            # Extract non-padded indices
            indices = indices_data[i, :]
            valid_indices = indices[indices .!= -1]
            
            push!(results, FeatureCombination(
                valid_indices,
                xgb_scores[i],
                rf_scores[i],
                avg_scores[i],
                n_features[i],
                gen_times[i]
            ))
        end
    end
    
    return results
end

# Export functions
export PretrainingConfig, create_pretraining_config
export FeatureCombination
export generate_pretraining_data, load_pretraining_data
export generate_synthetic_data
export generate_diverse_combinations, augment_combinations

end # module PretrainingData