module UnifiedPrediction

using MLJ
using MLJModelInterface
using DataFrames
using Statistics
using StatsBase

include("mlj_infrastructure.jl")
using .MLJInfrastructure

export UnifiedPredictor, PredictionResult
export predict_unified, predict_proba_unified, predict_with_confidence
export ensemble_predict, get_prediction_metrics

"""
Structure to hold prediction results with metadata
"""
struct PredictionResult
    predictions::AbstractVector
    probabilities::Union{Nothing, AbstractMatrix}  # For classification
    confidence_scores::Union{Nothing, AbstractVector}
    feature_importance::Union{Nothing, AbstractVector}
    model_type::Symbol
    task_type::Symbol
    metadata::Dict{Symbol, Any}
end

"""
Unified predictor handling both classification and regression
"""
struct UnifiedPredictor
    wrapper::ModelWrapper
    machine::Union{Nothing, Machine}
    task_type::Symbol
    n_features::Int
    feature_names::Union{Nothing, Vector{String}}
    
    function UnifiedPredictor(wrapper::ModelWrapper, machine::Union{Nothing, Machine}=nothing; 
                             n_features::Int=0, feature_names::Union{Nothing, Vector{String}}=nothing)
        new(wrapper, machine, wrapper.config.task_type, n_features, feature_names)
    end
end

"""
Make unified predictions handling both classification and regression
"""
function predict_unified(predictor::UnifiedPredictor, X::AbstractMatrix; 
                        return_proba::Bool=false, return_confidence::Bool=true)
    if isnothing(predictor.machine)
        error("Model must be fitted before making predictions")
    end
    
    # Convert to MLJ format
    X_table = MLJ.table(X)
    
    # Get base predictions
    predictions = predict(predictor.machine, X_table)
    
    # Initialize result components
    probabilities = nothing
    confidence_scores = nothing
    feature_importance = nothing
    
    # Handle task-specific predictions
    if predictor.task_type == :classification
        if return_proba
            # Get probability predictions for classification
            prob_preds = predict_mode(predictor.machine, X_table)
            probabilities = pdf.(prob_preds, levels(predictions))
        end
        
        if return_confidence
            # Calculate confidence as max probability
            if isnothing(probabilities)
                prob_preds = predict_mode(predictor.machine, X_table)
                probabilities = pdf.(prob_preds, levels(predictions))
            end
            confidence_scores = vec(maximum(probabilities, dims=2))
        end
    else  # regression
        if return_confidence
            # For regression, calculate prediction intervals
            confidence_scores = calculate_regression_confidence(predictor, X, predictions)
        end
    end
    
    # Try to get feature importance if available
    feature_importance = get_feature_importance(predictor)
    
    # Create metadata
    metadata = Dict{Symbol, Any}(
        :n_samples => size(X, 1),
        :n_features => size(X, 2),
        :prediction_time => time()
    )
    
    return PredictionResult(
        predictions,
        probabilities,
        confidence_scores,
        feature_importance,
        predictor.wrapper.config.model_type,
        predictor.task_type,
        metadata
    )
end

"""
Get probability predictions for classification
"""
function predict_proba_unified(predictor::UnifiedPredictor, X::AbstractMatrix)
    if predictor.task_type != :classification
        error("Probability predictions only available for classification tasks")
    end
    
    result = predict_unified(predictor, X, return_proba=true)
    return result.probabilities
end

"""
Make predictions with confidence intervals
"""
function predict_with_confidence(predictor::UnifiedPredictor, X::AbstractMatrix; 
                                confidence_level::Float64=0.95)
    result = predict_unified(predictor, X, return_confidence=true)
    
    if predictor.task_type == :regression
        # Calculate confidence intervals for regression
        z_score = quantile(Normal(), (1 + confidence_level) / 2)
        std_errors = result.confidence_scores
        
        lower_bounds = result.predictions .- z_score .* std_errors
        upper_bounds = result.predictions .+ z_score .* std_errors
        
        return result.predictions, lower_bounds, upper_bounds
    else
        # For classification, return predictions with confidence scores
        return result.predictions, result.confidence_scores
    end
end

"""
Ensemble prediction combining multiple models
"""
function ensemble_predict(predictors::Vector{UnifiedPredictor}, X::AbstractMatrix; 
                         voting::Symbol=:soft, weights::Union{Nothing, Vector{Float64}}=nothing)
    n_models = length(predictors)
    n_samples = size(X, 1)
    
    # Validate inputs
    if !isnothing(weights) && length(weights) != n_models
        error("Number of weights must match number of models")
    end
    
    # Check all models have same task type
    task_types = unique([p.task_type for p in predictors])
    if length(task_types) > 1
        error("All models must have the same task type")
    end
    task_type = task_types[1]
    
    # Collect predictions from all models
    all_predictions = []
    all_probabilities = []
    
    for predictor in predictors
        result = predict_unified(predictor, X, return_proba=(task_type == :classification))
        push!(all_predictions, result.predictions)
        if task_type == :classification && !isnothing(result.probabilities)
            push!(all_probabilities, result.probabilities)
        end
    end
    
    # Combine predictions based on task type and voting method
    if task_type == :classification
        if voting == :hard
            # Hard voting - majority vote
            ensemble_preds = hard_voting(all_predictions, weights)
        else  # soft voting
            # Soft voting - average probabilities
            ensemble_preds = soft_voting(all_probabilities, weights)
        end
    else  # regression
        # Average predictions (weighted if provided)
        if isnothing(weights)
            ensemble_preds = mean(all_predictions)
        else
            ensemble_preds = weighted_mean(all_predictions, weights)
        end
    end
    
    return ensemble_preds
end

"""
Calculate regression confidence based on prediction variance
"""
function calculate_regression_confidence(predictor::UnifiedPredictor, X::AbstractMatrix, predictions::AbstractVector)
    # Simple approach: use residual standard error from training
    # In practice, this would be calculated during training
    # For now, return a placeholder
    n_samples = length(predictions)
    
    # Estimate based on prediction variance (simplified)
    pred_std = std(predictions)
    confidence_scores = fill(pred_std * 0.1, n_samples)  # Simplified confidence
    
    return confidence_scores
end

"""
Extract feature importance from fitted model
"""
function get_feature_importance(predictor::UnifiedPredictor)
    model_type = predictor.wrapper.config.model_type
    
    try
        if model_type == :xgboost
            # XGBoost feature importance
            if !isnothing(predictor.machine) && fitted(predictor.machine)
                # Note: Actual implementation would extract from XGBoost model
                return nothing  # Placeholder
            end
        elseif model_type == :random_forest
            # Random Forest feature importance
            if !isnothing(predictor.machine) && fitted(predictor.machine)
                # Note: Actual implementation would extract from RF model
                return nothing  # Placeholder
            end
        elseif model_type == :lightgbm
            # LightGBM feature importance
            if !isnothing(predictor.machine) && fitted(predictor.machine)
                # Note: Actual implementation would extract from LightGBM model
                return nothing  # Placeholder
            end
        end
    catch e
        @warn "Could not extract feature importance: $e"
    end
    
    return nothing
end

"""
Hard voting for classification ensemble
"""
function hard_voting(predictions::Vector, weights::Union{Nothing, Vector{Float64}})
    n_samples = length(predictions[1])
    n_models = length(predictions)
    
    ensemble_preds = []
    
    for i in 1:n_samples
        # Get predictions from all models for this sample
        sample_preds = [predictions[j][i] for j in 1:n_models]
        
        # Count votes (weighted if provided)
        if isnothing(weights)
            # Simple majority vote
            vote_counts = countmap(sample_preds)
            pred = argmax(vote_counts)
        else
            # Weighted voting
            vote_weights = Dict{Any, Float64}()
            for (j, pred) in enumerate(sample_preds)
                vote_weights[pred] = get(vote_weights, pred, 0.0) + weights[j]
            end
            pred = argmax(vote_weights)
        end
        
        push!(ensemble_preds, pred)
    end
    
    return ensemble_preds
end

"""
Soft voting for classification ensemble
"""
function soft_voting(probabilities::Vector, weights::Union{Nothing, Vector{Float64}})
    n_models = length(probabilities)
    
    if isnothing(weights)
        weights = fill(1.0 / n_models, n_models)
    else
        # Normalize weights
        weights = weights ./ sum(weights)
    end
    
    # Weighted average of probabilities
    ensemble_probs = sum(w .* p for (w, p) in zip(weights, probabilities))
    
    # Get class with highest average probability
    ensemble_preds = [argmax(row) for row in eachrow(ensemble_probs)]
    
    return ensemble_preds
end

"""
Weighted mean for regression ensemble
"""
function weighted_mean(predictions::Vector, weights::Vector{Float64})
    # Normalize weights
    weights = weights ./ sum(weights)
    
    # Weighted average
    ensemble_preds = sum(w .* p for (w, p) in zip(weights, predictions))
    
    return ensemble_preds
end

"""
Get prediction metrics for evaluation
"""
function get_prediction_metrics(result::PredictionResult, y_true::AbstractVector)
    metrics = Dict{Symbol, Float64}()
    
    if result.task_type == :classification
        # Classification metrics
        metrics[:accuracy] = mean(result.predictions .== y_true)
        
        # Additional metrics if binary classification
        unique_classes = unique(y_true)
        if length(unique_classes) == 2
            # Binary classification metrics
            pos_class = maximum(unique_classes)
            y_pred_binary = result.predictions .== pos_class
            y_true_binary = y_true .== pos_class
            
            # Confusion matrix elements
            tp = sum(y_pred_binary .& y_true_binary)
            tn = sum(.!y_pred_binary .& .!y_true_binary)
            fp = sum(y_pred_binary .& .!y_true_binary)
            fn = sum(.!y_pred_binary .& y_true_binary)
            
            # Metrics
            metrics[:precision] = tp / (tp + fp + eps())
            metrics[:recall] = tp / (tp + fn + eps())
            metrics[:f1_score] = 2 * metrics[:precision] * metrics[:recall] / 
                                (metrics[:precision] + metrics[:recall] + eps())
        end
        
        # Average confidence
        if !isnothing(result.confidence_scores)
            metrics[:avg_confidence] = mean(result.confidence_scores)
        end
    else  # regression
        # Regression metrics
        residuals = result.predictions .- y_true
        metrics[:mse] = mean(residuals.^2)
        metrics[:rmse] = sqrt(metrics[:mse])
        metrics[:mae] = mean(abs.(residuals))
        metrics[:r2] = 1 - sum(residuals.^2) / sum((y_true .- mean(y_true)).^2)
        
        # Relative errors
        non_zero_mask = y_true .!= 0
        if any(non_zero_mask)
            metrics[:mape] = mean(abs.(residuals[non_zero_mask] ./ y_true[non_zero_mask])) * 100
        end
    end
    
    return metrics
end

# Utility function to ensure consistent output format
function format_predictions(predictions::AbstractVector, task_type::Symbol)
    if task_type == :classification
        # Ensure categorical output
        return categorical(predictions)
    else
        # Ensure numeric output
        return Float64.(predictions)
    end
end

end # module