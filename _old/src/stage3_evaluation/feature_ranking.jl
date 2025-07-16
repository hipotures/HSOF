module FeatureRanking

using Statistics
using LinearAlgebra
using DataFrames
using StatsBase
using Printf
using Random
using Dates
using CSV

# Include dependencies (commented out for testing without full dependencies)
# include("feature_importance.jl")
# include("feature_interactions.jl")
# include("statistical_testing.jl")

# using .FeatureImportance
# using .FeatureInteractions
# using .StatisticalTesting

export FeatureRanker, RankingConfig, RankingResult
export FeatureSelectionResult, SelectionConfig, ConstraintConfig
export rank_features, select_features
export pareto_optimal_selection, incremental_forward_selection
export ensemble_feature_voting, analyze_feature_stability
export generate_selection_report, export_selection_results

"""
Configuration for feature ranking methods and weights
"""
struct RankingConfig
    use_shap::Bool
    use_permutation::Bool
    use_interactions::Bool
    use_model_specific::Bool
    shap_weight::Float64
    permutation_weight::Float64
    interaction_weight::Float64
    model_specific_weight::Float64
    normalize_scores::Bool
    aggregation_method::Symbol  # :weighted_mean, :rank_aggregation, :borda_count
    
    function RankingConfig(;
        use_shap::Bool = true,
        use_permutation::Bool = true,
        use_interactions::Bool = true,
        use_model_specific::Bool = true,
        shap_weight::Float64 = 0.35,
        permutation_weight::Float64 = 0.25,
        interaction_weight::Float64 = 0.20,
        model_specific_weight::Float64 = 0.20,
        normalize_scores::Bool = true,
        aggregation_method::Symbol = :weighted_mean
    )
        # Validate weights sum to 1.0
        total_weight = 0.0
        if use_shap
            total_weight += shap_weight
        end
        if use_permutation
            total_weight += permutation_weight
        end
        if use_interactions
            total_weight += interaction_weight
        end
        if use_model_specific
            total_weight += model_specific_weight
        end
        
        if abs(total_weight - 1.0) > 1e-6
            # Normalize weights
            if use_shap
                shap_weight /= total_weight
            end
            if use_permutation
                permutation_weight /= total_weight
            end
            if use_interactions
                interaction_weight /= total_weight
            end
            if use_model_specific
                model_specific_weight /= total_weight
            end
        end
        
        new(use_shap, use_permutation, use_interactions, use_model_specific,
            shap_weight, permutation_weight, interaction_weight, model_specific_weight,
            normalize_scores, aggregation_method)
    end
end

"""
Configuration for feature selection constraints
"""
struct ConstraintConfig
    must_include::Vector{Int}      # Feature indices that must be included
    must_exclude::Vector{Int}      # Feature indices that must be excluded
    min_features::Int              # Minimum number of features to select
    max_features::Int              # Maximum number of features to select
    correlation_threshold::Float64  # Max correlation between selected features
    
    function ConstraintConfig(;
        must_include::Vector{Int} = Int[],
        must_exclude::Vector{Int} = Int[],
        min_features::Int = 10,
        max_features::Int = 20,
        correlation_threshold::Float64 = 0.9
    )
        @assert min_features <= max_features "min_features must be <= max_features"
        @assert correlation_threshold > 0 && correlation_threshold <= 1 "correlation_threshold must be in (0, 1]"
        @assert isempty(intersect(must_include, must_exclude)) "Features cannot be in both must_include and must_exclude"
        
        new(must_include, must_exclude, min_features, max_features, correlation_threshold)
    end
end

"""
Configuration for feature selection methods
"""
struct SelectionConfig
    method::Symbol                 # :pareto, :forward, :backward, :recursive
    ranking_config::RankingConfig
    constraint_config::ConstraintConfig
    cv_folds::Int                 # Number of CV folds for stability analysis
    stability_threshold::Float64   # Minimum stability score
    early_stopping_rounds::Int    # For incremental selection
    performance_metric::Symbol    # :accuracy, :auc, :rmse, etc.
    
    function SelectionConfig(;
        method::Symbol = :pareto,
        ranking_config::RankingConfig = RankingConfig(),
        constraint_config::ConstraintConfig = ConstraintConfig(),
        cv_folds::Int = 5,
        stability_threshold::Float64 = 0.7,
        early_stopping_rounds::Int = 3,
        performance_metric::Symbol = :accuracy
    )
        @assert method in [:pareto, :forward, :backward, :recursive] "Unknown selection method: $method"
        @assert cv_folds >= 2 "cv_folds must be >= 2"
        @assert stability_threshold >= 0 && stability_threshold <= 1 "stability_threshold must be in [0, 1]"
        
        new(method, ranking_config, constraint_config, cv_folds, 
            stability_threshold, early_stopping_rounds, performance_metric)
    end
end

"""
Result of feature ranking
"""
struct RankingResult
    feature_scores::Dict{Int, Float64}      # Feature index => aggregated score
    method_scores::Dict{Symbol, Vector{Float64}}  # Method => scores for all features
    feature_ranks::Vector{Int}              # Sorted feature indices (best to worst)
    method_contributions::Dict{Symbol, Float64}  # Method => average contribution
    metadata::Dict{Symbol, Any}
end

"""
Result of feature selection
"""
struct FeatureSelectionResult
    selected_features::Vector{Int}          # Selected feature indices
    ranking_result::RankingResult           # Full ranking information
    selection_scores::Vector{Float64}       # Scores for selected features
    performance_trajectory::Vector{Float64} # Performance at each selection step
    pareto_front::Union{Nothing, Vector{Tuple{Int, Float64, Float64}}}  # For Pareto selection
    stability_scores::Union{Nothing, Dict{Int, Float64}}  # Feature => stability
    selection_report::String
    metadata::Dict{Symbol, Any}
end

"""
Main feature ranker that combines multiple importance methods
"""
struct FeatureRanker
    config::RankingConfig
    n_features::Int
    feature_names::Union{Nothing, Vector{String}}
    
    function FeatureRanker(n_features::Int, config::RankingConfig = RankingConfig();
                          feature_names::Union{Nothing, Vector{String}} = nothing)
        if !isnothing(feature_names)
            @assert length(feature_names) == n_features "feature_names length must match n_features"
        end
        new(config, n_features, feature_names)
    end
end

"""
Rank features using multiple importance methods
"""
function rank_features(ranker::FeatureRanker,
                      importance_results::Dict{Symbol, Any};
                      interaction_matrix::Union{Nothing, Matrix{Float64}} = nothing)
    
    config = ranker.config
    n_features = ranker.n_features
    
    # Initialize score storage
    method_scores = Dict{Symbol, Vector{Float64}}()
    
    # Extract SHAP values if available
    if config.use_shap && haskey(importance_results, :shap)
        shap_result = importance_results[:shap]
        shap_scores = abs.(shap_result.mean_abs_shap)
        method_scores[:shap] = shap_scores
    end
    
    # Extract permutation importance if available
    if config.use_permutation && haskey(importance_results, :permutation)
        perm_result = importance_results[:permutation]
        perm_scores = perm_result.importance_mean
        method_scores[:permutation] = perm_scores
    end
    
    # Extract model-specific importance if available
    if config.use_model_specific && haskey(importance_results, :model_specific)
        model_scores = importance_results[:model_specific]
        method_scores[:model_specific] = model_scores
    end
    
    # Calculate interaction scores if provided
    if config.use_interactions && !isnothing(interaction_matrix)
        interaction_scores = calculate_interaction_importance(interaction_matrix)
        method_scores[:interactions] = interaction_scores
    end
    
    # Normalize scores if requested
    if config.normalize_scores
        for (method, scores) in method_scores
            method_scores[method] = normalize_scores(scores)
        end
    end
    
    # Aggregate scores based on method
    if config.aggregation_method == :weighted_mean
        feature_scores = aggregate_weighted_mean(method_scores, config)
    elseif config.aggregation_method == :rank_aggregation
        feature_scores = aggregate_by_ranks(method_scores, config)
    elseif config.aggregation_method == :borda_count
        feature_scores = aggregate_borda_count(method_scores, config)
    else
        error("Unknown aggregation method: $(config.aggregation_method)")
    end
    
    # Sort features by score
    feature_ranks = sortperm(collect(values(feature_scores)), rev=true)
    
    # Calculate method contributions
    method_contributions = calculate_method_contributions(method_scores, feature_scores, config)
    
    return RankingResult(
        feature_scores,
        method_scores,
        feature_ranks,
        method_contributions,
        Dict(:timestamp => now(),
             :n_methods => length(method_scores),
             :aggregation_method => config.aggregation_method)
    )
end

"""
Select features using specified method
"""
function select_features(ranker::FeatureRanker,
                        ranking_result::RankingResult,
                        X::Matrix{Float64},
                        y::Vector;
                        models::Vector = [],
                        config::SelectionConfig = SelectionConfig())
    
    if config.method == :pareto
        return pareto_optimal_selection(ranker, ranking_result, X, y, models, config)
    elseif config.method == :forward
        return incremental_forward_selection(ranker, ranking_result, X, y, models, config)
    elseif config.method == :backward
        error("Backward elimination not yet implemented")
    elseif config.method == :recursive
        error("Recursive feature elimination not yet implemented")
    else
        error("Unknown selection method: $(config.method)")
    end
end

"""
Pareto-optimal feature selection balancing performance and feature count
"""
function pareto_optimal_selection(ranker::FeatureRanker,
                                 ranking_result::RankingResult,
                                 X::Matrix{Float64},
                                 y::Vector,
                                 models::Vector,
                                 config::SelectionConfig)
    
    constraints = config.constraint_config
    n_features = size(X, 2)
    
    # Initialize with must-include features
    selected_features = copy(constraints.must_include)
    available_features = setdiff(1:n_features, constraints.must_exclude)
    available_features = setdiff(available_features, selected_features)
    
    # Sort available features by rank
    ranked_available = filter(f -> f in available_features, ranking_result.feature_ranks)
    
    # Evaluate different feature subset sizes
    pareto_points = Vector{Tuple{Int, Float64, Float64}}()
    performance_trajectory = Float64[]
    
    # Start from minimum features
    current_size = max(length(selected_features), constraints.min_features)
    
    while current_size <= min(constraints.max_features, length(available_features) + length(selected_features))
        # Add top-ranked features to reach current size
        n_to_add = current_size - length(selected_features)
        if n_to_add > 0
            candidates = ranked_available[1:min(n_to_add, length(ranked_available))]
            
            # Check correlation constraints
            candidates = filter_by_correlation(candidates, selected_features, X, 
                                             constraints.correlation_threshold)
            
            selected_features = vcat(selected_features, candidates)
        end
        
        # Evaluate performance
        X_subset = X[:, selected_features]
        performance = evaluate_feature_subset(X_subset, y, models, config)
        
        push!(performance_trajectory, performance)
        push!(pareto_points, (length(selected_features), performance, 
                             mean([ranking_result.feature_scores[f] for f in selected_features])))
        
        current_size += 1
    end
    
    # Find Pareto-optimal point
    optimal_idx = find_pareto_optimal(pareto_points)
    optimal_size = pareto_points[optimal_idx][1]
    
    # Get final selected features
    final_features = ranking_result.feature_ranks[1:optimal_size]
    final_features = filter(f -> f in available_features || f in constraints.must_include, 
                           final_features)
    
    # Analyze stability if requested
    stability_scores = nothing
    if config.cv_folds > 1
        stability_scores = analyze_feature_stability(
            X[:, final_features], y, models, config.cv_folds
        )
    end
    
    # Generate report
    report = generate_selection_report(
        ranker, ranking_result, final_features, pareto_points,
        performance_trajectory, stability_scores, config
    )
    
    return FeatureSelectionResult(
        final_features,
        ranking_result,
        [ranking_result.feature_scores[f] for f in final_features],
        performance_trajectory,
        pareto_points,
        stability_scores,
        report,
        Dict(:selection_method => :pareto,
             :optimal_size => optimal_size,
             :performance => pareto_points[optimal_idx][2])
    )
end

"""
Incremental forward selection with early stopping
"""
function incremental_forward_selection(ranker::FeatureRanker,
                                     ranking_result::RankingResult,
                                     X::Matrix{Float64},
                                     y::Vector,
                                     models::Vector,
                                     config::SelectionConfig)
    
    constraints = config.constraint_config
    
    # Initialize with must-include features
    selected_features = copy(constraints.must_include)
    available_features = setdiff(1:size(X, 2), constraints.must_exclude)
    available_features = setdiff(available_features, selected_features)
    
    # Sort available features by rank
    ranked_available = filter(f -> f in available_features, ranking_result.feature_ranks)
    
    # Track performance
    performance_trajectory = Float64[]
    best_performance = -Inf
    rounds_without_improvement = 0
    
    # Evaluate initial performance if we have must-include features
    if !isempty(selected_features)
        X_subset = X[:, selected_features]
        current_performance = evaluate_feature_subset(X_subset, y, models, config)
        push!(performance_trajectory, current_performance)
        best_performance = current_performance
    end
    
    # Forward selection loop
    for feature in ranked_available
        # Check if we've reached max features
        if length(selected_features) >= constraints.max_features
            break
        end
        
        # Check correlation constraint
        if !check_correlation_constraint(feature, selected_features, X, 
                                        constraints.correlation_threshold)
            continue
        end
        
        # Try adding this feature
        candidate_features = vcat(selected_features, feature)
        X_subset = X[:, candidate_features]
        candidate_performance = evaluate_feature_subset(X_subset, y, models, config)
        
        push!(performance_trajectory, candidate_performance)
        
        # Check if performance improved or we haven't reached min features yet
        if candidate_performance > best_performance || length(selected_features) < constraints.min_features
            selected_features = candidate_features
            if candidate_performance > best_performance
                best_performance = candidate_performance
                rounds_without_improvement = 0
            else
                rounds_without_improvement += 1
            end
        else
            rounds_without_improvement += 1
            
            # Early stopping check (only after minimum features reached)
            if length(selected_features) >= constraints.min_features && 
               rounds_without_improvement >= config.early_stopping_rounds
                break
            end
        end
    end
    
    # Calculate selection scores
    selection_scores = [ranking_result.feature_scores[f] for f in selected_features]
    
    # Generate report
    report = generate_selection_report(
        ranker, ranking_result, selected_features, nothing,
        performance_trajectory, nothing, config
    )
    
    return FeatureSelectionResult(
        selected_features,
        ranking_result,
        selection_scores,
        performance_trajectory,
        nothing,  # No Pareto front for forward selection
        nothing,  # No stability analysis by default
        report,
        Dict(:selection_method => :forward,
             :final_performance => best_performance,
             :stopped_early => rounds_without_improvement >= config.early_stopping_rounds)
    )
end

"""
Ensemble voting across different models for feature selection
"""
function ensemble_feature_voting(feature_rankings::Vector{RankingResult};
                               voting_method::Symbol = :borda,
                               top_k::Int = 20)
    
    n_rankers = length(feature_rankings)
    @assert n_rankers > 0 "Need at least one ranking result"
    
    # Collect all feature indices
    all_features = Set{Int}()
    for ranking in feature_rankings
        union!(all_features, keys(ranking.feature_scores))
    end
    
    # Initialize vote counts
    feature_votes = Dict{Int, Float64}(f => 0.0 for f in all_features)
    
    if voting_method == :borda
        # Borda count: higher rank gets more points
        for ranking in feature_rankings
            ranks = ranking.feature_ranks
            for (idx, feature) in enumerate(ranks)
                # Points = n_features - rank + 1
                points = length(ranks) - idx + 1
                feature_votes[feature] += points / n_rankers
            end
        end
    elseif voting_method == :average_rank
        # Average rank across all rankings
        feature_ranks = Dict{Int, Vector{Int}}(f => Int[] for f in all_features)
        
        for ranking in feature_rankings
            ranks = ranking.feature_ranks
            for (idx, feature) in enumerate(ranks)
                push!(feature_ranks[feature], idx)
            end
            # Add worst rank for missing features
            for feature in all_features
                if !(feature in ranks)
                    push!(feature_ranks[feature], length(ranks) + 1)
                end
            end
        end
        
        # Convert average ranks to scores (lower rank = higher score)
        for (feature, ranks) in feature_ranks
            avg_rank = mean(ranks)
            feature_votes[feature] = 1.0 / avg_rank
        end
    elseif voting_method == :majority
        # Count how many times each feature appears in top-k
        for ranking in feature_rankings
            top_features = ranking.feature_ranks[1:min(top_k, length(ranking.feature_ranks))]
            for feature in top_features
                feature_votes[feature] += 1.0 / n_rankers
            end
        end
    else
        error("Unknown voting method: $voting_method")
    end
    
    # Sort by votes
    sorted_features = sort(collect(keys(feature_votes)), 
                          by=f -> feature_votes[f], rev=true)
    
    return sorted_features[1:min(top_k, length(sorted_features))], feature_votes
end

"""
Analyze feature stability across CV folds
"""
function analyze_feature_stability(X::Matrix{Float64}, y::Vector,
                                 models::Vector, n_folds::Int)
    
    n_samples, n_features = size(X)
    
    # Track feature importance across folds
    fold_importances = Vector{Dict{Int, Float64}}()
    
    # Create CV folds
    fold_size = div(n_samples, n_folds)
    indices = randperm(n_samples)
    
    for fold in 1:n_folds
        # Create train/val split
        val_start = (fold - 1) * fold_size + 1
        val_end = fold == n_folds ? n_samples : fold * fold_size
        val_indices = indices[val_start:val_end]
        train_indices = setdiff(indices, val_indices)
        
        X_train = X[train_indices, :]
        y_train = y[train_indices]
        
        # Calculate importance for this fold
        fold_importance = Dict{Int, Float64}()
        
        # Average importance across models
        for model in models
            # This is a placeholder - actual implementation would calculate
            # feature importance for each model
            model_importance = calculate_model_importance(model, X_train, y_train)
            for (feat_idx, importance) in model_importance
                fold_importance[feat_idx] = get(fold_importance, feat_idx, 0.0) + 
                                           importance / length(models)
            end
        end
        
        push!(fold_importances, fold_importance)
    end
    
    # Calculate stability scores
    stability_scores = Dict{Int, Float64}()
    
    for feature in 1:n_features
        # Get importance values across folds
        importance_values = Float64[]
        for fold_imp in fold_importances
            push!(importance_values, get(fold_imp, feature, 0.0))
        end
        
        # Calculate stability as 1 - coefficient of variation
        if mean(importance_values) > 0
            cv = std(importance_values) / mean(importance_values)
            stability_scores[feature] = 1.0 / (1.0 + cv)
        else
            stability_scores[feature] = 0.0
        end
    end
    
    return stability_scores
end

# Helper functions

"""
Calculate feature importance from interaction matrix
"""
function calculate_interaction_importance(interaction_matrix::Matrix{Float64})
    n_features = size(interaction_matrix, 1)
    interaction_scores = zeros(n_features)
    
    for i in 1:n_features
        # Sum of all interactions involving feature i
        interaction_scores[i] = sum(abs.(interaction_matrix[i, :])) + 
                               sum(abs.(interaction_matrix[:, i])) - 
                               abs(interaction_matrix[i, i])
    end
    
    return interaction_scores
end

"""
Normalize scores to [0, 1] range
"""
function normalize_scores(scores::Vector{Float64})
    min_score = minimum(scores)
    max_score = maximum(scores)
    
    if max_score == min_score
        return ones(length(scores))
    end
    
    return (scores .- min_score) ./ (max_score - min_score)
end

"""
Aggregate scores using weighted mean
"""
function aggregate_weighted_mean(method_scores::Dict{Symbol, Vector{Float64}},
                               config::RankingConfig)
    
    n_features = length(first(values(method_scores)))
    aggregated = Dict{Int, Float64}(i => 0.0 for i in 1:n_features)
    
    # Weight mapping
    weight_map = Dict(
        :shap => config.shap_weight,
        :permutation => config.permutation_weight,
        :interactions => config.interaction_weight,
        :model_specific => config.model_specific_weight
    )
    
    for (method, scores) in method_scores
        weight = get(weight_map, method, 0.0)
        for (idx, score) in enumerate(scores)
            aggregated[idx] += weight * score
        end
    end
    
    return aggregated
end

"""
Aggregate scores using rank aggregation
"""
function aggregate_by_ranks(method_scores::Dict{Symbol, Vector{Float64}},
                          config::RankingConfig)
    
    n_features = length(first(values(method_scores)))
    rank_sum = Dict{Int, Float64}(i => 0.0 for i in 1:n_features)
    
    # Weight mapping
    weight_map = Dict(
        :shap => config.shap_weight,
        :permutation => config.permutation_weight,
        :interactions => config.interaction_weight,
        :model_specific => config.model_specific_weight
    )
    
    for (method, scores) in method_scores
        weight = get(weight_map, method, 0.0)
        ranks = ordinalrank(scores, rev=true)
        
        for (idx, rank) in enumerate(ranks)
            rank_sum[idx] += weight * (n_features - rank + 1)
        end
    end
    
    return rank_sum
end

"""
Aggregate scores using Borda count
"""
function aggregate_borda_count(method_scores::Dict{Symbol, Vector{Float64}},
                             config::RankingConfig)
    
    n_features = length(first(values(method_scores)))
    borda_scores = Dict{Int, Float64}(i => 0.0 for i in 1:n_features)
    
    for (method, scores) in method_scores
        sorted_indices = sortperm(scores, rev=true)
        
        for (rank, idx) in enumerate(sorted_indices)
            # Borda score: n_features - rank
            borda_scores[idx] += n_features - rank + 1
        end
    end
    
    # Normalize by number of methods
    for idx in keys(borda_scores)
        borda_scores[idx] /= length(method_scores)
    end
    
    return borda_scores
end

"""
Calculate method contributions to final ranking
"""
function calculate_method_contributions(method_scores::Dict{Symbol, Vector{Float64}},
                                      feature_scores::Dict{Int, Float64},
                                      config::RankingConfig)
    
    contributions = Dict{Symbol, Float64}()
    
    # Calculate correlation between each method's scores and final scores
    final_scores = [feature_scores[i] for i in sort(collect(keys(feature_scores)))]
    
    for (method, scores) in method_scores
        if length(scores) == length(final_scores)
            correlation = cor(scores, final_scores)
            contributions[method] = abs(correlation)
        end
    end
    
    return contributions
end

"""
Filter features by correlation constraint
"""
function filter_by_correlation(candidates::Vector{Int}, 
                             selected::Vector{Int},
                             X::Matrix{Float64},
                             threshold::Float64)
    
    if isempty(selected)
        return candidates
    end
    
    filtered = Int[]
    
    for candidate in candidates
        max_corr = 0.0
        for selected_feat in selected
            corr = abs(cor(X[:, candidate], X[:, selected_feat]))
            max_corr = max(max_corr, corr)
        end
        
        if max_corr < threshold
            push!(filtered, candidate)
        end
    end
    
    return filtered
end

"""
Check correlation constraint for a single feature
"""
function check_correlation_constraint(feature::Int,
                                    selected::Vector{Int},
                                    X::Matrix{Float64},
                                    threshold::Float64)
    
    if isempty(selected)
        return true
    end
    
    for selected_feat in selected
        if abs(cor(X[:, feature], X[:, selected_feat])) >= threshold
            return false
        end
    end
    
    return true
end

"""
Evaluate performance of a feature subset
"""
function evaluate_feature_subset(X::Matrix{Float64}, y::Vector,
                               models::Vector, config::SelectionConfig)
    
    # Placeholder implementation
    # In practice, this would:
    # 1. Train models on X with cross-validation
    # 2. Evaluate using the specified metric
    # 3. Return average performance across models
    
    # For now, return a simple proxy based on feature matrix properties
    n_samples, n_features = size(X)
    
    # Simple heuristic: balance between number of features and data variance
    feature_variance = mean(var(X, dims=1))
    feature_penalty = exp(-n_features / 20)  # Penalty for too many features
    
    return feature_variance * feature_penalty
end

"""
Find Pareto-optimal point balancing performance and complexity
"""
function find_pareto_optimal(points::Vector{Tuple{Int, Float64, Float64}})
    # Find the knee point in the Pareto front
    # Using simple heuristic: maximize performance/n_features ratio
    
    best_idx = 1
    best_ratio = 0.0
    
    for (idx, (n_features, performance, _)) in enumerate(points)
        ratio = performance / sqrt(n_features)  # Penalize complexity
        if ratio > best_ratio
            best_ratio = ratio
            best_idx = idx
        end
    end
    
    return best_idx
end

"""
Calculate model-specific feature importance
"""
function calculate_model_importance(model, X::Matrix{Float64}, y::Vector)
    # Placeholder implementation
    # Would extract feature importance from trained model
    n_features = size(X, 2)
    
    # Return random importance scores for now
    importance = Dict{Int, Float64}()
    for i in 1:n_features
        importance[i] = rand()
    end
    
    return importance
end

"""
Generate comprehensive feature selection report
"""
function generate_selection_report(ranker::FeatureRanker,
                                 ranking_result::RankingResult,
                                 selected_features::Vector{Int},
                                 pareto_points::Union{Nothing, Vector},
                                 performance_trajectory::Vector{Float64},
                                 stability_scores::Union{Nothing, Dict{Int, Float64}},
                                 config::SelectionConfig)
    
    report_lines = String[]
    
    push!(report_lines, "="^80)
    push!(report_lines, "FEATURE SELECTION REPORT")
    push!(report_lines, "="^80)
    push!(report_lines, "Generated: $(now())")
    push!(report_lines, "")
    
    # Summary
    push!(report_lines, "SUMMARY")
    push!(report_lines, "-"^40)
    push!(report_lines, "Total features evaluated: $(ranker.n_features)")
    push!(report_lines, "Features selected: $(length(selected_features))")
    push!(report_lines, "Selection method: $(config.method)")
    push!(report_lines, "Performance metric: $(config.performance_metric)")
    push!(report_lines, "")
    
    # Selected features
    push!(report_lines, "SELECTED FEATURES")
    push!(report_lines, "-"^40)
    
    for (rank, feat_idx) in enumerate(selected_features)
        score = ranking_result.feature_scores[feat_idx]
        feat_name = isnothing(ranker.feature_names) ? "Feature_$feat_idx" : 
                                                      ranker.feature_names[feat_idx]
        
        line = @sprintf("%2d. %s (Score: %.4f", rank, feat_name, score)
        
        if !isnothing(stability_scores) && haskey(stability_scores, feat_idx)
            line *= @sprintf(", Stability: %.3f", stability_scores[feat_idx])
        end
        
        line *= ")"
        push!(report_lines, line)
    end
    push!(report_lines, "")
    
    # Method contributions
    push!(report_lines, "METHOD CONTRIBUTIONS")
    push!(report_lines, "-"^40)
    
    for (method, contribution) in ranking_result.method_contributions
        push!(report_lines, @sprintf("%s: %.1f%%", 
                                   string(method), contribution * 100))
    end
    push!(report_lines, "")
    
    # Performance trajectory
    if !isempty(performance_trajectory)
        push!(report_lines, "PERFORMANCE TRAJECTORY")
        push!(report_lines, "-"^40)
        
        for (i, perf) in enumerate(performance_trajectory)
            push!(report_lines, @sprintf("Step %2d: %.4f", i, perf))
        end
        push!(report_lines, "")
    end
    
    # Pareto analysis (if applicable)
    if !isnothing(pareto_points)
        push!(report_lines, "PARETO ANALYSIS")
        push!(report_lines, "-"^40)
        push!(report_lines, "Features | Performance | Avg Score")
        push!(report_lines, "-"^35)
        
        for (n_feat, perf, avg_score) in pareto_points
            push!(report_lines, @sprintf("%8d | %11.4f | %9.4f", 
                                       n_feat, perf, avg_score))
        end
        push!(report_lines, "")
    end
    
    # Constraints applied
    constraints = config.constraint_config
    if !isempty(constraints.must_include) || !isempty(constraints.must_exclude)
        push!(report_lines, "CONSTRAINTS APPLIED")
        push!(report_lines, "-"^40)
        
        if !isempty(constraints.must_include)
            push!(report_lines, "Must include: $(constraints.must_include)")
        end
        if !isempty(constraints.must_exclude)
            push!(report_lines, "Must exclude: $(constraints.must_exclude)")
        end
        push!(report_lines, "Correlation threshold: $(constraints.correlation_threshold)")
        push!(report_lines, "")
    end
    
    push!(report_lines, "="^80)
    
    return join(report_lines, "\n")
end

"""
Export selection results to file
"""
function export_selection_results(result::FeatureSelectionResult, 
                                filepath::String;
                                format::Symbol = :csv)
    
    if format == :csv
        # Create DataFrame with results
        df = DataFrame(
            feature_index = result.selected_features,
            score = result.selection_scores,
            rank = 1:length(result.selected_features)
        )
        
        # Add stability scores if available
        if !isnothing(result.stability_scores)
            df.stability = [get(result.stability_scores, idx, missing) 
                          for idx in result.selected_features]
        end
        
        CSV.write(filepath, df)
    elseif format == :json
        # Export as JSON (simplified without JSON package)
        error("JSON export not implemented in this version")
    else
        error("Unknown export format: $format")
    end
end

end # module