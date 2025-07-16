module CrossValidation

using Random
using Statistics
using StatsBase
using DataFrames
using MLJ
using MLJBase

export StratifiedKFold, QuantileStratifiedKFold, CVIterator
export create_folds, validate_folds, get_fold_metrics
export cross_validate_model, aggregate_cv_results
export SMOTESampler, apply_smote

"""
Stratified K-Fold cross-validation splitter for classification
"""
struct StratifiedKFold
    n_folds::Int
    shuffle::Bool
    random_state::Union{Nothing, Int}
    
    function StratifiedKFold(; n_folds::Int=5, shuffle::Bool=true, random_state::Union{Nothing, Int}=nothing)
        n_folds >= 2 || error("n_folds must be >= 2")
        new(n_folds, shuffle, random_state)
    end
end

"""
Quantile-based stratified K-Fold for regression tasks
"""
struct QuantileStratifiedKFold
    n_folds::Int
    n_quantiles::Int
    shuffle::Bool
    random_state::Union{Nothing, Int}
    
    function QuantileStratifiedKFold(; n_folds::Int=5, n_quantiles::Int=10, 
                                    shuffle::Bool=true, random_state::Union{Nothing, Int}=nothing)
        n_folds >= 2 || error("n_folds must be >= 2")
        n_quantiles >= n_folds || error("n_quantiles must be >= n_folds")
        new(n_folds, n_quantiles, shuffle, random_state)
    end
end

"""
CV Iterator structure holding fold indices
"""
struct CVIterator
    train_indices::Vector{Vector{Int}}
    test_indices::Vector{Vector{Int}}
    n_folds::Int
    
    function CVIterator(train_indices::Vector{Vector{Int}}, test_indices::Vector{Vector{Int}})
        n_folds = length(train_indices)
        length(test_indices) == n_folds || error("Mismatch in train/test fold counts")
        new(train_indices, test_indices, n_folds)
    end
end

"""
Create stratified folds for classification tasks
"""
function create_folds(cv::StratifiedKFold, y::AbstractVector)
    n_samples = length(y)
    
    # Set random seed if provided
    rng = isnothing(cv.random_state) ? Random.GLOBAL_RNG : MersenneTwister(cv.random_state)
    
    # Get unique classes and their indices
    classes = unique(y)
    class_indices = Dict(c => findall(y .== c) for c in classes)
    
    # Shuffle indices within each class if requested
    if cv.shuffle
        for c in classes
            shuffle!(rng, class_indices[c])
        end
    end
    
    # Initialize fold assignments
    fold_assignments = zeros(Int, n_samples)
    
    # Distribute each class across folds
    for (class, indices) in class_indices
        n_class_samples = length(indices)
        
        # Calculate samples per fold for this class
        base_fold_size = n_class_samples ÷ cv.n_folds
        n_larger_folds = n_class_samples % cv.n_folds
        
        # Assign samples to folds
        current_idx = 1
        for fold in 1:cv.n_folds
            fold_size = base_fold_size + (fold <= n_larger_folds ? 1 : 0)
            fold_end = current_idx + fold_size - 1
            
            for idx in current_idx:fold_end
                if idx <= n_class_samples
                    fold_assignments[indices[idx]] = fold
                end
            end
            
            current_idx = fold_end + 1
        end
    end
    
    # Create train/test indices for each fold
    train_indices = Vector{Vector{Int}}()
    test_indices = Vector{Vector{Int}}()
    
    for fold in 1:cv.n_folds
        test_idx = findall(fold_assignments .== fold)
        train_idx = findall(fold_assignments .!= fold)
        push!(test_indices, test_idx)
        push!(train_indices, train_idx)
    end
    
    return CVIterator(train_indices, test_indices)
end

"""
Create quantile-based stratified folds for regression tasks
"""
function create_folds(cv::QuantileStratifiedKFold, y::AbstractVector)
    n_samples = length(y)
    
    # Set random seed if provided
    rng = isnothing(cv.random_state) ? Random.GLOBAL_RNG : MersenneTwister(cv.random_state)
    
    # Create quantile bins
    quantile_edges = quantile(y, range(0, 1, length=cv.n_quantiles+1))
    
    # Assign samples to quantile bins
    quantile_labels = zeros(Int, n_samples)
    for i in 1:n_samples
        for q in 1:cv.n_quantiles
            if y[i] >= quantile_edges[q] && y[i] <= quantile_edges[q+1]
                quantile_labels[i] = q
                break
            end
        end
    end
    
    # Handle edge case where highest value might not be assigned
    quantile_labels[quantile_labels .== 0] .= cv.n_quantiles
    
    # Use stratified k-fold on quantile labels
    stratified_cv = StratifiedKFold(n_folds=cv.n_folds, shuffle=cv.shuffle, 
                                   random_state=cv.random_state)
    return create_folds(stratified_cv, quantile_labels)
end

"""
Validate folds to ensure minimum samples per class
"""
function validate_folds(cv_iter::CVIterator, y::AbstractVector; 
                       min_samples_per_class::Int=1, verbose::Bool=false)
    classes = unique(y)
    is_valid = true
    validation_results = Dict{Int, Dict{Any, Int}}()
    
    for (fold_idx, (train_idx, test_idx)) in enumerate(zip(cv_iter.train_indices, cv_iter.test_indices))
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # Check class distribution in train and test sets
        train_counts = countmap(y_train)
        test_counts = countmap(y_test)
        
        validation_results[fold_idx] = Dict(
            "train" => train_counts,
            "test" => test_counts
        )
        
        # Validate minimum samples
        for class in classes
            train_class_count = get(train_counts, class, 0)
            test_class_count = get(test_counts, class, 0)
            
            if train_class_count < min_samples_per_class || test_class_count < min_samples_per_class
                is_valid = false
                if verbose
                    println("Warning: Fold $fold_idx has insufficient samples for class $class")
                    println("  Train: $train_class_count, Test: $test_class_count")
                end
            end
        end
    end
    
    if verbose && is_valid
        println("All folds validated successfully!")
    end
    
    return is_valid, validation_results
end

"""
Base implementation for SMOTE sampling (simplified version)
"""
struct SMOTESampler
    sampling_ratio::Float64
    k_neighbors::Int
    random_state::Union{Nothing, Int}
    
    function SMOTESampler(; sampling_ratio::Float64=1.0, k_neighbors::Int=5, 
                         random_state::Union{Nothing, Int}=nothing)
        sampling_ratio > 0 || error("sampling_ratio must be > 0")
        k_neighbors > 0 || error("k_neighbors must be > 0")
        new(sampling_ratio, k_neighbors, random_state)
    end
end

"""
Apply SMOTE to balance dataset (simplified implementation)
Note: This is a basic implementation. For production, use Imbalance.jl or similar
"""
function apply_smote(sampler::SMOTESampler, X::AbstractMatrix, y::AbstractVector)
    # Get class counts
    class_counts = countmap(y)
    minority_class = argmin(class_counts)
    majority_class = argmax(class_counts)
    
    # Calculate number of synthetic samples needed
    n_minority = class_counts[minority_class]
    n_majority = class_counts[majority_class]
    n_synthetic = round(Int, (n_majority - n_minority) * sampler.sampling_ratio)
    
    if n_synthetic == 0
        return X, y
    end
    
    # Get minority class samples
    minority_indices = findall(y .== minority_class)
    X_minority = X[minority_indices, :]
    
    # Generate synthetic samples (simplified - just add noise to existing samples)
    rng = isnothing(sampler.random_state) ? Random.GLOBAL_RNG : MersenneTwister(sampler.random_state)
    
    synthetic_X = zeros(n_synthetic, size(X, 2))
    for i in 1:n_synthetic
        # Select random minority sample
        idx = rand(rng, 1:n_minority)
        base_sample = X_minority[idx, :]
        
        # Add small random noise (simplified SMOTE)
        noise = randn(rng, size(X, 2)) * 0.1
        synthetic_X[i, :] = base_sample + noise
    end
    
    # Combine original and synthetic data
    X_balanced = vcat(X, synthetic_X)
    y_balanced = vcat(y, fill(minority_class, n_synthetic))
    
    # Shuffle the combined dataset
    shuffle_indices = shuffle(rng, 1:length(y_balanced))
    
    return X_balanced[shuffle_indices, :], y_balanced[shuffle_indices]
end

"""
Calculate metrics for a single fold
"""
function get_fold_metrics(y_true::AbstractVector, y_pred::AbstractVector, 
                         y_proba::Union{Nothing, AbstractMatrix}=nothing;
                         task_type::Symbol=:classification)
    metrics = Dict{Symbol, Float64}()
    
    if task_type == :classification
        # Basic classification metrics
        metrics[:accuracy] = mean(y_true .== y_pred)
        
        # Binary classification metrics
        unique_classes = unique(y_true)
        if length(unique_classes) == 2
            pos_class = maximum(unique_classes)
            y_true_binary = y_true .== pos_class
            y_pred_binary = y_pred .== pos_class
            
            # Confusion matrix elements
            tp = sum(y_true_binary .& y_pred_binary)
            tn = sum(.!y_true_binary .& .!y_pred_binary)
            fp = sum(.!y_true_binary .& y_pred_binary)
            fn = sum(y_true_binary .& .!y_pred_binary)
            
            # Calculate metrics
            metrics[:precision] = tp / (tp + fp + eps())
            metrics[:recall] = tp / (tp + fn + eps())
            metrics[:f1_score] = 2 * metrics[:precision] * metrics[:recall] / 
                                (metrics[:precision] + metrics[:recall] + eps())
            metrics[:specificity] = tn / (tn + fp + eps())
            
            # AUC if probabilities available
            if !isnothing(y_proba) && size(y_proba, 2) >= 2
                # Simple AUC calculation (for more robust, use MLJ metrics)
                pos_proba = y_proba[:, 2]
                metrics[:auc] = calculate_simple_auc(y_true_binary, pos_proba)
            end
        end
        
        # Multi-class metrics
        if length(unique_classes) > 2
            # Macro-averaged F1
            f1_scores = Float64[]
            for class in unique_classes
                y_true_class = y_true .== class
                y_pred_class = y_pred .== class
                
                tp = sum(y_true_class .& y_pred_class)
                fp = sum(.!y_true_class .& y_pred_class)
                fn = sum(y_true_class .& .!y_pred_class)
                
                precision = tp / (tp + fp + eps())
                recall = tp / (tp + fn + eps())
                f1 = 2 * precision * recall / (precision + recall + eps())
                push!(f1_scores, f1)
            end
            metrics[:macro_f1] = mean(f1_scores)
        end
        
    else  # regression
        residuals = y_true .- y_pred
        metrics[:mse] = mean(residuals.^2)
        metrics[:rmse] = sqrt(metrics[:mse])
        metrics[:mae] = mean(abs.(residuals))
        metrics[:r2] = 1 - sum(residuals.^2) / sum((y_true .- mean(y_true)).^2)
        
        # MAPE for non-zero values
        non_zero_mask = y_true .!= 0
        if any(non_zero_mask)
            metrics[:mape] = mean(abs.(residuals[non_zero_mask] ./ y_true[non_zero_mask])) * 100
        end
    end
    
    return metrics
end

"""
Simple AUC calculation (for binary classification)
"""
function calculate_simple_auc(y_true::AbstractVector{Bool}, scores::AbstractVector{Float64})
    # Sort by scores
    sorted_indices = sortperm(scores, rev=true)
    y_sorted = y_true[sorted_indices]
    
    # Calculate AUC using trapezoidal rule
    n_pos = sum(y_sorted)
    n_neg = length(y_sorted) - n_pos
    
    if n_pos == 0 || n_neg == 0
        return 0.5  # No discrimination possible
    end
    
    # Count true positives and false positives at each threshold
    tp = 0
    fp = 0
    auc = 0.0
    prev_fp = 0
    prev_tp = 0
    
    for i in 1:length(y_sorted)
        if y_sorted[i]
            tp += 1
        else
            fp += 1
        end
        
        # Add area under curve segment
        auc += (fp - prev_fp) * (tp + prev_tp) / 2
        
        prev_fp = fp
        prev_tp = tp
    end
    
    # Normalize
    auc = auc / (n_pos * n_neg)
    
    return auc
end

"""
Perform cross-validation for a model
"""
function cross_validate_model(model, X::AbstractMatrix, y::AbstractVector, cv::Union{StratifiedKFold, QuantileStratifiedKFold};
                             task_type::Symbol=:classification, 
                             return_predictions::Bool=false,
                             apply_smote::Bool=false,
                             smote_params::Union{Nothing, SMOTESampler}=nothing,
                             verbose::Bool=true)
    # Create folds
    cv_iter = create_folds(cv, y)
    
    # Validate folds
    is_valid, _ = validate_folds(cv_iter, y, verbose=verbose)
    if !is_valid && verbose
        @warn "Some folds have very few samples per class. Results may be unreliable."
    end
    
    # Storage for results
    fold_metrics = Vector{Dict{Symbol, Float64}}()
    fold_predictions = return_predictions ? Vector{Tuple{Vector{Int}, AbstractVector}}() : nothing
    
    # Iterate through folds
    for (fold_idx, (train_idx, test_idx)) in enumerate(zip(cv_iter.train_indices, cv_iter.test_indices))
        if verbose
            println("Processing fold $fold_idx/$(cv.n_folds)...")
        end
        
        # Split data
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Apply SMOTE if requested (only for classification)
        if apply_smote && task_type == :classification
            sampler = isnothing(smote_params) ? SMOTESampler() : smote_params
            X_train, y_train = apply_smote(sampler, X_train, y_train)
        end
        
        # Train model (assuming model has fit! and predict methods)
        # This is a simplified version - actual implementation would use MLJ properly
        X_train_table = MLJ.table(X_train)
        X_test_table = MLJ.table(X_test)
        
        mach = machine(model, X_train_table, y_train)
        fit!(mach, verbosity=0)
        
        # Make predictions
        y_pred = predict(mach, X_test_table)
        
        # Get probabilities for classification
        y_proba = nothing
        if task_type == :classification
            y_proba_raw = predict_mode(mach, X_test_table)
            # Convert to matrix format (simplified)
            if !isnothing(y_proba_raw)
                classes = levels(y_pred)
                y_proba = zeros(length(y_pred), length(classes))
                # Note: Actual implementation would properly extract probabilities
            end
        end
        
        # Calculate metrics
        metrics = get_fold_metrics(y_test, y_pred, y_proba, task_type=task_type)
        push!(fold_metrics, metrics)
        
        # Store predictions if requested
        if return_predictions
            push!(fold_predictions, (test_idx, y_pred))
        end
    end
    
    # Aggregate results
    aggregated_metrics = aggregate_cv_results(fold_metrics)
    
    return Dict(
        "fold_metrics" => fold_metrics,
        "aggregated_metrics" => aggregated_metrics,
        "predictions" => fold_predictions,
        "cv_iterator" => cv_iter
    )
end

"""
Aggregate CV results across folds
"""
function aggregate_cv_results(fold_metrics::Vector{Dict{Symbol, Float64}})
    if isempty(fold_metrics)
        return Dict{Symbol, Any}()
    end
    
    # Get all metric names
    metric_names = keys(fold_metrics[1])
    
    aggregated = Dict{Symbol, Any}()
    
    for metric in metric_names
        values = [fold[metric] for fold in fold_metrics if haskey(fold, metric)]
        
        if !isempty(values)
            aggregated[Symbol(string(metric) * "_mean")] = mean(values)
            aggregated[Symbol(string(metric) * "_std")] = std(values)
            aggregated[Symbol(string(metric) * "_min")] = minimum(values)
            aggregated[Symbol(string(metric) * "_max")] = maximum(values)
        end
    end
    
    # Add fold count
    aggregated[:n_folds] = length(fold_metrics)
    
    return aggregated
end

# Utility function for reproducible CV splits
function get_reproducible_cv(cv_type::Symbol, n_folds::Int; seed::Int=42, kwargs...)
    if cv_type == :stratified
        return StratifiedKFold(n_folds=n_folds, random_state=seed; kwargs...)
    elseif cv_type == :quantile_stratified
        return QuantileStratifiedKFold(n_folds=n_folds, random_state=seed; kwargs...)
    else
        error("Unknown CV type: $cv_type")
    end
end

end # module