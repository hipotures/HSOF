module FeatureImportance

using Statistics
using Random
using LinearAlgebra
using DataFrames
using CSV
using MLJ
using MLJBase
using ProgressMeter
using Distributed
using Base.Threads

export SHAPCalculator, PermutationImportance
export calculate_shap_values, calculate_permutation_importance
export aggregate_importance_cv, get_feature_interactions
export ImportanceResult, InteractionMatrix
export combine_importance_methods, export_importance_plot
export accuracy, mse, rmse

"""
Result structure for feature importance calculations
"""
struct ImportanceResult
    feature_indices::Vector{Int}
    feature_names::Union{Nothing, Vector{String}}
    importance_values::Vector{Float64}
    confidence_intervals::Union{Nothing, Matrix{Float64}}  # [lower, upper] for each feature
    method::Symbol  # :shap, :permutation, :combined
    metadata::Dict{Symbol, Any}
end

"""
Interaction matrix for feature pairs
"""
struct InteractionMatrix
    feature_indices::Vector{Int}
    interaction_values::Matrix{Float64}
    feature_names::Union{Nothing, Vector{String}}
end

"""
SHAP value calculator for tree-based models
"""
mutable struct SHAPCalculator
    model_type::Symbol
    n_samples::Union{Nothing, Int}  # For sampling large datasets
    baseline_type::Symbol  # :mean, :median, :zeros
    random_state::Union{Nothing, Int}
    
    function SHAPCalculator(; model_type::Symbol=:auto,
                           n_samples::Union{Nothing, Int}=nothing,
                           baseline_type::Symbol=:mean,
                           random_state::Union{Nothing, Int}=nothing)
        new(model_type, n_samples, baseline_type, random_state)
    end
end

"""
Permutation importance calculator
"""
mutable struct PermutationImportance
    n_repeats::Int
    scoring_function::Function
    random_state::Union{Nothing, Int}
    n_jobs::Int
    
    function PermutationImportance(; n_repeats::Int=10,
                                  scoring_function::Function=accuracy,
                                  random_state::Union{Nothing, Int}=nothing,
                                  n_jobs::Int=1)
        new(n_repeats, scoring_function, random_state, n_jobs)
    end
end

"""
Calculate SHAP values for tree-based models
"""
function calculate_shap_values(calc::SHAPCalculator, model, machine, 
                             X::AbstractMatrix, y::AbstractVector;
                             feature_names::Union{Nothing, Vector{String}}=nothing)
    
    # Detect model type if auto
    if calc.model_type == :auto
        calc.model_type = detect_model_type(model)
    end
    
    # Sample data if needed
    X_sample, sample_indices = sample_data(X, calc.n_samples, calc.random_state)
    
    # Calculate baseline
    baseline = calculate_baseline(X_sample, y[sample_indices], calc.baseline_type)
    
    # Calculate SHAP values based on model type
    if calc.model_type == :xgboost
        shap_values = calculate_xgboost_shap(model, machine, X_sample, baseline)
    elseif calc.model_type == :random_forest
        shap_values = calculate_rf_shap(model, machine, X_sample, baseline)
    elseif calc.model_type == :lightgbm
        shap_values = calculate_lgbm_shap(model, machine, X_sample, baseline)
    else
        error("Unsupported model type: $(calc.model_type)")
    end
    
    # Average absolute SHAP values across samples
    feature_importance = vec(mean(abs.(shap_values), dims=1))
    
    # Calculate confidence intervals using bootstrap
    ci_lower, ci_upper = bootstrap_confidence_intervals(shap_values)
    
    return ImportanceResult(
        collect(1:size(X, 2)),
        feature_names,
        feature_importance,
        hcat(ci_lower, ci_upper),
        :shap,
        Dict(:n_samples => size(X_sample, 1),
             :baseline => baseline,
             :model_type => calc.model_type)
    )
end

"""
TreeSHAP implementation for XGBoost
"""
function calculate_xgboost_shap(model, machine, X::AbstractMatrix, baseline::Float64)
    n_samples, n_features = size(X)
    shap_values = zeros(n_samples, n_features)
    
    # For XGBoost, we need to extract tree structure
    # This is a simplified implementation - full TreeSHAP would require tree traversal
    
    # Get predictions for baseline
    baseline_pred = baseline
    
    # Calculate SHAP values using interventional approach
    for i in 1:n_samples
        x_sample = X[i, :]
        
        # Full prediction
        full_pred = predict_single(machine, reshape(x_sample, 1, :))
        
        # Feature contributions
        for j in 1:n_features
            # Create modified sample with feature j set to baseline
            x_modified = copy(x_sample)
            x_modified[j] = mean(X[:, j])  # Use mean as baseline for feature
            
            # Prediction without feature j
            pred_without_j = predict_single(machine, reshape(x_modified, 1, :))
            
            # SHAP value is the difference
            shap_values[i, j] = full_pred - pred_without_j
        end
    end
    
    return shap_values
end

"""
TreeSHAP implementation for Random Forest
"""
function calculate_rf_shap(model, machine, X::AbstractMatrix, baseline::Float64)
    n_samples, n_features = size(X)
    shap_values = zeros(n_samples, n_features)
    
    # For Random Forest, average SHAP values across trees
    # This is a simplified implementation
    
    for i in 1:n_samples
        x_sample = X[i, :]
        
        # Get prediction
        full_pred = predict_single(machine, reshape(x_sample, 1, :))
        
        # Calculate feature contributions
        for j in 1:n_features
            # Permute feature j across training samples
            x_permuted = copy(x_sample)
            x_permuted[j] = X[rand(1:n_samples), j]
            
            # Prediction with permuted feature
            pred_permuted = predict_single(machine, reshape(x_permuted, 1, :))
            
            # SHAP approximation
            shap_values[i, j] = full_pred - pred_permuted
        end
    end
    
    return shap_values
end

"""
TreeSHAP implementation for LightGBM
"""
function calculate_lgbm_shap(model, machine, X::AbstractMatrix, baseline::Float64)
    # Similar to XGBoost implementation
    # LightGBM has similar tree structure
    return calculate_xgboost_shap(model, machine, X, baseline)
end

"""
Calculate permutation importance
"""
function calculate_permutation_importance(calc::PermutationImportance, 
                                        model, machine,
                                        X::AbstractMatrix, y::AbstractVector;
                                        feature_names::Union{Nothing, Vector{String}}=nothing)
    
    n_features = size(X, 2)
    n_samples = size(X, 1)
    
    # Calculate baseline score
    y_pred = predict(machine, MLJ.table(X))
    baseline_score = calc.scoring_function(y, y_pred)
    
    # Storage for importance scores
    importance_scores = zeros(n_features, calc.n_repeats)
    
    # Progress bar
    p = Progress(n_features * calc.n_repeats, desc="Calculating permutation importance: ")
    
    # RNG for reproducibility
    rng = isnothing(calc.random_state) ? Random.GLOBAL_RNG : MersenneTwister(calc.random_state)
    
    # Calculate importance for each feature
    if calc.n_jobs == 1
        # Sequential execution
        for j in 1:n_features
            for r in 1:calc.n_repeats
                # Permute feature j
                X_permuted = copy(X)
                X_permuted[:, j] = X_permuted[randperm(rng, n_samples), j]
                
                # Calculate score with permuted feature
                y_pred_permuted = predict(machine, MLJ.table(X_permuted))
                permuted_score = calc.scoring_function(y, y_pred_permuted)
                
                # Importance is the decrease in score
                importance_scores[j, r] = baseline_score - permuted_score
                
                ProgressMeter.next!(p)
            end
        end
    else
        # Parallel execution
        importance_scores = parallel_permutation_importance(
            calc, machine, X, y, baseline_score, p
        )
    end
    
    ProgressMeter.finish!(p)
    
    # Calculate mean and confidence intervals
    mean_importance = vec(mean(importance_scores, dims=2))
    std_importance = vec(std(importance_scores, dims=2))
    
    # 95% confidence intervals
    ci_lower = mean_importance .- 1.96 * std_importance / sqrt(calc.n_repeats)
    ci_upper = mean_importance .+ 1.96 * std_importance / sqrt(calc.n_repeats)
    
    return ImportanceResult(
        collect(1:n_features),
        feature_names,
        mean_importance,
        hcat(ci_lower, ci_upper),
        :permutation,
        Dict(:n_repeats => calc.n_repeats,
             :baseline_score => baseline_score,
             :scoring_function => nameof(calc.scoring_function))
    )
end

"""
Parallel computation of permutation importance
"""
function parallel_permutation_importance(calc::PermutationImportance,
                                       machine, X::AbstractMatrix, y::AbstractVector,
                                       baseline_score::Float64, progress_bar)
    n_features = size(X, 2)
    n_samples = size(X, 1)
    
    # Create chunks for parallel processing
    chunk_size = ceil(Int, n_features / calc.n_jobs)
    chunks = [(i:min(i+chunk_size-1, n_features)) for i in 1:chunk_size:n_features]
    
    # Process chunks in parallel
    results = Vector{Matrix{Float64}}(undef, length(chunks))
    
    @threads for (chunk_idx, feature_range) in collect(enumerate(chunks))
        chunk_scores = zeros(length(feature_range), calc.n_repeats)
        
        for (local_idx, j) in enumerate(feature_range)
            for r in 1:calc.n_repeats
                # Permute feature j
                X_permuted = copy(X)
                X_permuted[:, j] = X_permuted[randperm(n_samples), j]
                
                # Calculate score
                y_pred_permuted = predict(machine, MLJ.table(X_permuted))
                permuted_score = calc.scoring_function(y, y_pred_permuted)
                
                chunk_scores[local_idx, r] = baseline_score - permuted_score
                
                # Update progress
                lock(progress_bar.lock) do
                    ProgressMeter.next!(progress_bar)
                end
            end
        end
        
        results[chunk_idx] = chunk_scores
    end
    
    # Combine results
    return vcat(results...)
end

"""
Calculate feature interactions using SHAP
"""
function get_feature_interactions(shap_calc::SHAPCalculator, model, machine,
                                X::AbstractMatrix, y::AbstractVector;
                                feature_names::Union{Nothing, Vector{String}}=nothing,
                                top_k::Int=10)
    
    n_features = size(X, 2)
    
    # Sample data if needed
    X_sample, sample_indices = sample_data(X, shap_calc.n_samples, shap_calc.random_state)
    
    # Calculate pairwise interactions
    interaction_matrix = zeros(n_features, n_features)
    
    # For each pair of features
    for i in 1:n_features
        for j in (i+1):n_features
            # Calculate interaction strength
            interaction = calculate_pairwise_interaction(
                model, machine, X_sample, i, j
            )
            interaction_matrix[i, j] = interaction
            interaction_matrix[j, i] = interaction  # Symmetric
        end
    end
    
    # Find top interactions
    interaction_pairs = []
    for i in 1:n_features
        for j in (i+1):n_features
            push!(interaction_pairs, (i, j, interaction_matrix[i, j]))
        end
    end
    
    # Sort by interaction strength
    sort!(interaction_pairs, by=x->abs(x[3]), rev=true)
    
    # Return top k interactions
    top_interactions = first(interaction_pairs, min(top_k, length(interaction_pairs)))
    
    return InteractionMatrix(
        collect(1:n_features),
        interaction_matrix,
        feature_names
    ), top_interactions
end

"""
Calculate pairwise interaction between two features
"""
function calculate_pairwise_interaction(model, machine, X::AbstractMatrix, 
                                      feature_i::Int, feature_j::Int)
    n_samples = size(X, 1)
    
    # Calculate main effects and interaction effect
    main_effect_i = 0.0
    main_effect_j = 0.0
    interaction_effect = 0.0
    
    for k in 1:min(n_samples, 100)  # Sample for efficiency
        x_sample = X[k, :]
        
        # Original prediction
        pred_original = predict_single(machine, reshape(x_sample, 1, :))
        
        # Permute feature i
        x_perm_i = copy(x_sample)
        x_perm_i[feature_i] = X[rand(1:n_samples), feature_i]
        pred_perm_i = predict_single(machine, reshape(x_perm_i, 1, :))
        
        # Permute feature j
        x_perm_j = copy(x_sample)
        x_perm_j[feature_j] = X[rand(1:n_samples), feature_j]
        pred_perm_j = predict_single(machine, reshape(x_perm_j, 1, :))
        
        # Permute both
        x_perm_both = copy(x_sample)
        x_perm_both[feature_i] = X[rand(1:n_samples), feature_i]
        x_perm_both[feature_j] = X[rand(1:n_samples), feature_j]
        pred_perm_both = predict_single(machine, reshape(x_perm_both, 1, :))
        
        # Calculate effects
        main_effect_i += abs(pred_original - pred_perm_i)
        main_effect_j += abs(pred_original - pred_perm_j)
        
        # Interaction is the deviation from additive effects
        expected_additive = pred_original - (pred_original - pred_perm_i) - (pred_original - pred_perm_j)
        interaction_effect += abs(pred_perm_both - expected_additive)
    end
    
    # Normalize
    n_used = min(n_samples, 100)
    interaction_strength = interaction_effect / n_used
    
    return interaction_strength
end

"""
Aggregate importance scores across CV folds
"""
function aggregate_importance_cv(importance_results::Vector{ImportanceResult};
                               aggregation_method::Symbol=:mean)
    
    if isempty(importance_results)
        error("No importance results to aggregate")
    end
    
    n_features = length(importance_results[1].importance_values)
    n_folds = length(importance_results)
    
    # Collect importance values
    importance_matrix = zeros(n_features, n_folds)
    for (i, result) in enumerate(importance_results)
        importance_matrix[:, i] = result.importance_values
    end
    
    # Aggregate
    if aggregation_method == :mean
        aggregated_importance = vec(mean(importance_matrix, dims=2))
        importance_std = vec(std(importance_matrix, dims=2))
    elseif aggregation_method == :median
        aggregated_importance = vec(median(importance_matrix, dims=2))
        # Use MAD for robust estimate
        mad_values = vec(median(abs.(importance_matrix .- aggregated_importance), dims=2))
        importance_std = 1.4826 * mad_values  # MAD to std conversion
    else
        error("Unknown aggregation method: $aggregation_method")
    end
    
    # Calculate confidence intervals
    ci_lower = aggregated_importance .- 1.96 * importance_std / sqrt(n_folds)
    ci_upper = aggregated_importance .+ 1.96 * importance_std / sqrt(n_folds)
    
    # Create aggregated result
    return ImportanceResult(
        importance_results[1].feature_indices,
        importance_results[1].feature_names,
        aggregated_importance,
        hcat(ci_lower, ci_upper),
        :combined,
        Dict(:n_folds => n_folds,
             :aggregation_method => aggregation_method,
             :std_deviation => importance_std)
    )
end

"""
Helper: Detect model type
"""
function detect_model_type(model)
    model_name = string(typeof(model))
    
    if occursin("XGBoost", model_name)
        return :xgboost
    elseif occursin("RandomForest", model_name) || occursin("DecisionTree", model_name)
        return :random_forest
    elseif occursin("LIGHTGBM", model_name) || occursin("LightGBM", model_name)
        return :lightgbm
    else
        error("Cannot auto-detect model type for: $model_name")
    end
end

"""
Helper: Sample data for large datasets
"""
function sample_data(X::AbstractMatrix, n_samples::Union{Nothing, Int}, 
                    random_state::Union{Nothing, Int})
    n_total = size(X, 1)
    
    if isnothing(n_samples) || n_samples >= n_total
        return X, collect(1:n_total)
    end
    
    rng = isnothing(random_state) ? Random.GLOBAL_RNG : MersenneTwister(random_state)
    sample_indices = randperm(rng, n_total)[1:n_samples]
    
    return X[sample_indices, :], sample_indices
end

"""
Helper: Calculate baseline value
"""
function calculate_baseline(X::AbstractMatrix, y::AbstractVector, baseline_type::Symbol)
    if baseline_type == :mean
        return mean(y)
    elseif baseline_type == :median
        return median(y)
    elseif baseline_type == :zeros
        return 0.0
    else
        error("Unknown baseline type: $baseline_type")
    end
end

"""
Helper: Single prediction extraction
"""
function predict_single(machine, X::AbstractMatrix)
    pred = predict(machine, MLJ.table(X))
    return first(pred)
end

"""
Helper: Bootstrap confidence intervals for SHAP values
"""
function bootstrap_confidence_intervals(shap_values::Matrix{Float64}, 
                                      n_bootstrap::Int=100, alpha::Float64=0.05)
    n_samples, n_features = size(shap_values)
    
    ci_lower = zeros(n_features)
    ci_upper = zeros(n_features)
    
    for j in 1:n_features
        # Bootstrap samples
        bootstrap_means = zeros(n_bootstrap)
        
        for b in 1:n_bootstrap
            # Sample with replacement
            indices = rand(1:n_samples, n_samples)
            bootstrap_means[b] = mean(abs.(shap_values[indices, j]))
        end
        
        # Calculate percentiles
        sorted_means = sort(bootstrap_means)
        ci_lower[j] = sorted_means[Int(floor(alpha/2 * n_bootstrap))]
        ci_upper[j] = sorted_means[Int(ceil((1-alpha/2) * n_bootstrap))]
    end
    
    return ci_lower, ci_upper
end

"""
Visualization export for importance rankings
"""
function export_importance_plot(result::ImportanceResult, filename::String;
                              top_k::Int=20, plot_ci::Bool=true)
    # Create DataFrame for easy plotting
    df = DataFrame(
        feature = isnothing(result.feature_names) ? 
                 string.("Feature ", result.feature_indices) : 
                 result.feature_names,
        importance = result.importance_values
    )
    
    if !isnothing(result.confidence_intervals) && plot_ci
        df.ci_lower = result.confidence_intervals[:, 1]
        df.ci_upper = result.confidence_intervals[:, 2]
    end
    
    # Sort by importance
    sort!(df, :importance, rev=true)
    
    # Take top k
    df_top = first(df, min(top_k, nrow(df)))
    
    # Export to CSV for plotting
    CSV.write(filename, df_top)
    
    return df_top
end

"""
Combined importance from multiple methods
"""
function combine_importance_methods(shap_result::ImportanceResult, 
                                  perm_result::ImportanceResult;
                                  weights::Tuple{Float64, Float64}=(0.5, 0.5))
    
    @assert length(shap_result.importance_values) == length(perm_result.importance_values)
    @assert sum(weights) ≈ 1.0
    
    # Normalize importance values to [0, 1]
    shap_norm = shap_result.importance_values / maximum(shap_result.importance_values)
    perm_norm = perm_result.importance_values / maximum(perm_result.importance_values)
    
    # Weighted combination
    combined_importance = weights[1] * shap_norm + weights[2] * perm_norm
    
    # Combine confidence intervals if available
    combined_ci = nothing
    if !isnothing(shap_result.confidence_intervals) && !isnothing(perm_result.confidence_intervals)
        # Normalize CIs
        shap_ci_norm = shap_result.confidence_intervals / maximum(shap_result.importance_values)
        perm_ci_norm = perm_result.confidence_intervals / maximum(perm_result.importance_values)
        
        # Weighted combination
        combined_ci = weights[1] * shap_ci_norm + weights[2] * perm_ci_norm
    end
    
    return ImportanceResult(
        shap_result.feature_indices,
        shap_result.feature_names,
        combined_importance,
        combined_ci,
        :combined,
        Dict(:shap_weight => weights[1],
             :permutation_weight => weights[2],
             :methods => [:shap, :permutation])
    )
end

# Scoring functions for permutation importance
accuracy(y_true, y_pred) = mean(y_true .== y_pred)
mse(y_true, y_pred) = mean((y_true .- y_pred).^2)
rmse(y_true, y_pred) = sqrt(mse(y_true, y_pred))

end # module