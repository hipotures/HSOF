# Example usage of Feature Ranking and Selection Module

using Random
using Statistics
using LinearAlgebra
using DataFrames
using CSV

# Include the modules
include("../src/stage3_evaluation/feature_ranking.jl")

using .FeatureRanking

# Set random seed
Random.seed!(123)

"""
Example 1: Basic feature ranking with multiple importance methods
"""
function basic_ranking_example()
    println("=== Basic Feature Ranking Example ===\n")
    
    # Simulate data
    n_samples, n_features = 500, 50
    X = randn(n_samples, n_features)
    
    # Create some informative features
    informative_features = [1, 5, 10, 15, 20, 25, 30]
    y = sum(X[:, f] for f in informative_features) + 0.5 * randn(n_samples)
    
    # Feature names
    feature_names = ["Feature_$i" for i in 1:n_features]
    
    # Create ranker with default configuration
    ranker = FeatureRanker(n_features, feature_names=feature_names)
    
    # Simulate importance results from different methods
    # In practice, these would come from actual SHAP/permutation calculations
    importance_results = Dict{Symbol, Any}(
        :shap => (
            mean_abs_shap = create_mock_shap_values(n_features, informative_features),
        ),
        :permutation => (
            importance_mean = create_mock_permutation_importance(n_features, informative_features),
        ),
        :model_specific => create_mock_model_importance(n_features, informative_features)
    )
    
    # Create interaction matrix
    interaction_matrix = create_mock_interaction_matrix(n_features, informative_features)
    
    # Rank features
    ranking_result = rank_features(ranker, importance_results, 
                                 interaction_matrix=interaction_matrix)
    
    # Display results
    println("Top 10 Features by Aggregated Score:")
    println("-" * 40)
    
    for i in 1:10
        feat_idx = ranking_result.feature_ranks[i]
        score = ranking_result.feature_scores[feat_idx]
        println("$i. $(feature_names[feat_idx]) - Score: $(round(score, digits=4))")
    end
    
    println("\nMethod Contributions:")
    for (method, contribution) in ranking_result.method_contributions
        println("  $method: $(round(contribution * 100, digits=1))%")
    end
    
    return ranking_result
end

"""
Example 2: Feature ranking with custom configuration
"""
function custom_ranking_example()
    println("\n\n=== Custom Ranking Configuration Example ===\n")
    
    n_samples, n_features = 300, 30
    X = randn(n_samples, n_features)
    y = rand(Bool, n_samples)
    
    # Custom ranking configuration
    ranking_config = RankingConfig(
        use_shap = true,
        use_permutation = true,
        use_interactions = false,  # Disable interaction analysis
        use_model_specific = true,
        shap_weight = 0.5,         # Give more weight to SHAP
        permutation_weight = 0.3,
        model_specific_weight = 0.2,
        normalize_scores = true,
        aggregation_method = :rank_aggregation  # Use rank-based aggregation
    )
    
    ranker = FeatureRanker(n_features, ranking_config)
    
    # Create importance results
    importance_results = Dict{Symbol, Any}(
        :shap => (mean_abs_shap = rand(n_features) .+ 0.1,),
        :permutation => (importance_mean = rand(n_features) .+ 0.05,),
        :model_specific => rand(n_features) .+ 0.02
    )
    
    ranking_result = rank_features(ranker, importance_results)
    
    println("Ranking Method: $(ranking_config.aggregation_method)")
    println("Top 5 Features:")
    for i in 1:5
        feat_idx = ranking_result.feature_ranks[i]
        score = ranking_result.feature_scores[feat_idx]
        println("  $i. Feature_$feat_idx - Score: $(round(score, digits=4))")
    end
    
    # Compare with Borda count
    ranking_config_borda = RankingConfig(
        aggregation_method = :borda_count,
        use_interactions = false
    )
    ranker_borda = FeatureRanker(n_features, ranking_config_borda)
    ranking_result_borda = rank_features(ranker_borda, importance_results)
    
    println("\nComparison with Borda Count:")
    println("Rank | Rank Aggregation | Borda Count")
    println("-" * 40)
    for i in 1:5
        feat_rank = ranking_result.feature_ranks[i]
        feat_borda = ranking_result_borda.feature_ranks[i]
        println("  $i  |    Feature_$feat_rank    |  Feature_$feat_borda")
    end
    
    return ranking_result, ranking_result_borda
end

"""
Example 3: Pareto-optimal feature selection
"""
function pareto_selection_example()
    println("\n\n=== Pareto-Optimal Feature Selection Example ===\n")
    
    # Create synthetic dataset with known important features
    n_samples, n_features = 1000, 40
    X, y, true_important = create_synthetic_classification_data(n_samples, n_features)
    
    println("True important features: $true_important")
    
    # Create ranker
    ranker = FeatureRanker(n_features)
    
    # Create mock importance results
    importance_results = create_importance_results_for_known_features(
        n_features, true_important
    )
    
    # Rank features
    ranking_result = rank_features(ranker, importance_results)
    
    # Configure Pareto selection
    selection_config = SelectionConfig(
        method = :pareto,
        constraint_config = ConstraintConfig(
            min_features = 8,
            max_features = 15,
            correlation_threshold = 0.85
        ),
        cv_folds = 5,
        stability_threshold = 0.7
    )
    
    # Perform selection
    selection_result = select_features(ranker, ranking_result, X, y,
                                     models=[], config=selection_config)
    
    println("\nPareto-Optimal Selection Results:")
    println("-" * 40)
    println("Selected $(length(selection_result.selected_features)) features")
    println("Selected features: $(selection_result.selected_features)")
    
    # Check how many true features were selected
    true_selected = intersect(selection_result.selected_features, true_important)
    println("True features recovered: $(length(true_selected))/$(length(true_important))")
    
    # Display Pareto front
    if !isnothing(selection_result.pareto_front)
        println("\nPareto Front Analysis:")
        println("Features | Performance | Avg Score")
        println("-" * 35)
        for (n_feat, perf, avg_score) in selection_result.pareto_front
            println("   $n_feat    |   $(round(perf, digits=3))    | $(round(avg_score, digits=3))")
        end
    end
    
    return selection_result
end

"""
Example 4: Forward selection with constraints
"""
function forward_selection_example()
    println("\n\n=== Forward Selection with Constraints Example ===\n")
    
    n_samples, n_features = 500, 35
    X = randn(n_samples, n_features)
    
    # Create target with specific feature dependencies
    important_features = [2, 5, 8, 12, 18, 22]
    y = 2 * X[:, 2] + 1.5 * X[:, 5] - X[:, 8] + 
        0.8 * X[:, 12] + 0.5 * X[:, 18] + 0.3 * X[:, 22] + 
        0.2 * randn(n_samples)
    
    # Domain knowledge: must include features 2 and 5
    # Cannot use features 15, 20, 25 (e.g., they're expensive to compute)
    
    # Create ranker
    feature_names = ["Var_$i" for i in 1:n_features]
    ranker = FeatureRanker(n_features, feature_names=feature_names)
    
    # Create importance results
    importance_results = create_importance_results_for_known_features(
        n_features, important_features
    )
    
    ranking_result = rank_features(ranker, importance_results)
    
    # Configure forward selection with constraints
    selection_config = SelectionConfig(
        method = :forward,
        constraint_config = ConstraintConfig(
            must_include = [2, 5],      # Domain knowledge
            must_exclude = [15, 20, 25], # Expensive features
            min_features = 8,
            max_features = 12,
            correlation_threshold = 0.9
        ),
        early_stopping_rounds = 3,
        performance_metric = :rmse
    )
    
    selection_result = select_features(ranker, ranking_result, X, y,
                                     models=[], config=selection_config)
    
    println("Forward Selection Results:")
    println("-" * 40)
    println("Selected $(length(selection_result.selected_features)) features")
    println("\nSelected features with scores:")
    
    for (idx, feat) in enumerate(selection_result.selected_features)
        score = selection_result.selection_scores[idx]
        constraint_info = ""
        if feat in [2, 5]
            constraint_info = " [MUST INCLUDE]"
        end
        println("  $(feature_names[feat]): $(round(score, digits=4))$constraint_info")
    end
    
    # Verify constraints
    println("\nConstraint Verification:")
    println("  Must-include satisfied: $(all(f in selection_result.selected_features for f in [2, 5]))")
    println("  Must-exclude satisfied: $(all(f ∉ selection_result.selected_features for f in [15, 20, 25]))")
    
    # Show performance trajectory
    println("\nPerformance Trajectory:")
    for (i, perf) in enumerate(selection_result.performance_trajectory)
        println("  Step $i: $(round(perf, digits=4))")
    end
    
    stopped_early = get(selection_result.metadata, :stopped_early, false)
    println("\nEarly stopping triggered: $stopped_early")
    
    return selection_result
end

"""
Example 5: Ensemble voting across multiple models
"""
function ensemble_voting_example()
    println("\n\n=== Ensemble Feature Voting Example ===\n")
    
    n_features = 25
    
    # Simulate rankings from different models
    # (In practice, these would come from actual model training)
    model_names = ["XGBoost", "RandomForest", "LightGBM", "SVM"]
    ranking_results = RankingResult[]
    
    println("Individual Model Rankings (Top 5):")
    println("-" * 40)
    
    for model in model_names
        # Create model-specific importance scores
        if model == "XGBoost"
            # XGBoost might favor different features
            scores = Dict(i => rand() * (i in [1, 3, 5, 7, 9] ? 2.0 : 1.0) 
                         for i in 1:n_features)
        elseif model == "RandomForest"
            # RF might favor other features
            scores = Dict(i => rand() * (i in [2, 4, 6, 8, 10] ? 2.0 : 1.0) 
                         for i in 1:n_features)
        else
            scores = Dict(i => rand() for i in 1:n_features)
        end
        
        feature_ranks = sortperm(collect(values(scores)), rev=true)
        
        ranking_result = RankingResult(
            scores,
            Dict{Symbol, Vector{Float64}}(),
            feature_ranks,
            Dict{Symbol, Float64}(),
            Dict(:model => model)
        )
        
        push!(ranking_results, ranking_result)
        
        # Display top 5 for this model
        print("$model: ")
        top_5 = feature_ranks[1:5]
        println(join(["F$f" for f in top_5], ", "))
    end
    
    # Perform ensemble voting
    println("\n\nEnsemble Voting Results:")
    println("-" * 40)
    
    # Method 1: Borda count
    selected_borda, votes_borda = ensemble_feature_voting(
        ranking_results,
        voting_method = :borda,
        top_k = 10
    )
    
    println("\nBorda Count (Top 10):")
    for (i, feat) in enumerate(selected_borda)
        println("  $i. Feature_$feat - Borda Score: $(round(votes_borda[feat], digits=2))")
    end
    
    # Method 2: Average rank
    selected_avg, votes_avg = ensemble_feature_voting(
        ranking_results,
        voting_method = :average_rank,
        top_k = 10
    )
    
    println("\nAverage Rank (Top 10):")
    for (i, feat) in enumerate(selected_avg)
        println("  $i. Feature_$feat - Avg Rank Score: $(round(votes_avg[feat], digits=4))")
    end
    
    # Method 3: Majority voting
    selected_majority, votes_majority = ensemble_feature_voting(
        ranking_results,
        voting_method = :majority,
        top_k = 10
    )
    
    println("\nMajority Voting (Top 10):")
    for (i, feat) in enumerate(selected_majority)
        percentage = votes_majority[feat] * 100
        println("  $i. Feature_$feat - In top 10 for $(round(percentage, digits=0))% of models")
    end
    
    # Find consensus features (appear in all three methods)
    consensus = intersect(selected_borda, selected_avg, selected_majority)
    println("\nConsensus Features (in all three methods): $consensus")
    
    return selected_borda, selected_avg, selected_majority
end

"""
Example 6: Complete feature selection pipeline with stability analysis
"""
function complete_pipeline_example()
    println("\n\n=== Complete Feature Selection Pipeline Example ===\n")
    
    # Create a more realistic dataset
    n_samples, n_features = 800, 60
    X, y, feature_groups = create_grouped_features_dataset(n_samples, n_features)
    
    println("Dataset Information:")
    println("  Samples: $n_samples")
    println("  Features: $n_features")
    println("  Feature groups: $(length(feature_groups)) groups")
    
    # Step 1: Create feature ranker with names
    feature_names = create_feature_names_for_groups(feature_groups)
    
    ranking_config = RankingConfig(
        use_shap = true,
        use_permutation = true,
        use_interactions = true,
        use_model_specific = true,
        normalize_scores = true,
        aggregation_method = :weighted_mean
    )
    
    ranker = FeatureRanker(n_features, ranking_config, feature_names=feature_names)
    
    # Step 2: Generate importance results
    importance_results = generate_comprehensive_importance_results(X, y, feature_groups)
    interaction_matrix = generate_interaction_matrix(X, y, feature_groups)
    
    # Step 3: Rank features
    println("\nStep 1: Ranking Features...")
    ranking_result = rank_features(ranker, importance_results, 
                                 interaction_matrix=interaction_matrix)
    
    println("Top 10 features by initial ranking:")
    for i in 1:10
        feat_idx = ranking_result.feature_ranks[i]
        println("  $i. $(feature_names[feat_idx])")
    end
    
    # Step 4: Perform Pareto-optimal selection
    println("\nStep 2: Pareto-Optimal Selection...")
    
    selection_config = SelectionConfig(
        method = :pareto,
        ranking_config = ranking_config,
        constraint_config = ConstraintConfig(
            min_features = 10,
            max_features = 20,
            correlation_threshold = 0.85
        ),
        cv_folds = 5,
        stability_threshold = 0.7,
        performance_metric = :accuracy
    )
    
    pareto_result = select_features(ranker, ranking_result, X, y,
                                  models=[], config=selection_config)
    
    println("Pareto-optimal selection: $(length(pareto_result.selected_features)) features")
    
    # Step 5: Refine with forward selection
    println("\nStep 3: Refining with Forward Selection...")
    
    forward_config = SelectionConfig(
        method = :forward,
        ranking_config = ranking_config,
        constraint_config = ConstraintConfig(
            must_include = pareto_result.selected_features[1:3],  # Keep top 3
            min_features = 12,
            max_features = 18,
            correlation_threshold = 0.9
        ),
        early_stopping_rounds = 3
    )
    
    final_result = select_features(ranker, ranking_result, X, y,
                                 models=[], config=forward_config)
    
    # Step 6: Generate comprehensive report
    println("\n" * "="*60)
    println("FINAL FEATURE SELECTION SUMMARY")
    println("="*60)
    
    println("\nSelected Features: $(length(final_result.selected_features))")
    println("\nFeature Details:")
    println("-"*40)
    
    for (idx, feat) in enumerate(final_result.selected_features)
        score = final_result.selection_scores[idx]
        group = find_feature_group(feat, feature_groups)
        println("$(lpad(idx, 2)). $(rpad(feature_names[feat], 20)) Score: $(round(score, digits=3)) [Group $group]")
    end
    
    # Export results
    output_file = "feature_selection_results.csv"
    export_selection_results(final_result, output_file, format=:csv)
    println("\nResults exported to: $output_file")
    
    # Clean up
    rm(output_file, force=true)
    
    return final_result
end

# Helper functions for examples

function create_mock_shap_values(n_features::Int, informative_features::Vector{Int})
    shap_values = 0.1 * rand(n_features)
    for feat in informative_features
        shap_values[feat] = 0.5 + 0.4 * rand()
    end
    return shap_values
end

function create_mock_permutation_importance(n_features::Int, informative_features::Vector{Int})
    perm_importance = 0.05 * rand(n_features)
    for feat in informative_features
        perm_importance[feat] = 0.3 + 0.3 * rand()
    end
    return perm_importance
end

function create_mock_model_importance(n_features::Int, informative_features::Vector{Int})
    model_importance = 0.02 * rand(n_features)
    for feat in informative_features
        model_importance[feat] = 0.2 + 0.2 * rand()
    end
    return model_importance
end

function create_mock_interaction_matrix(n_features::Int, informative_features::Vector{Int})
    matrix = 0.01 * rand(n_features, n_features)
    
    # Add stronger interactions between informative features
    for i in informative_features
        for j in informative_features
            if i != j
                matrix[i, j] = 0.1 + 0.2 * rand()
            end
        end
    end
    
    # Make symmetric
    return 0.5 * (matrix + matrix')
end

function create_synthetic_classification_data(n_samples::Int, n_features::Int)
    X = randn(n_samples, n_features)
    
    # Select random important features
    n_important = 8
    true_important = sort(randperm(n_features)[1:n_important])
    
    # Create target based on important features
    y_continuous = sum(randn() * X[:, f] for f in true_important)
    y = y_continuous .> median(y_continuous)
    
    return X, y, true_important
end

function create_importance_results_for_known_features(n_features::Int, 
                                                    important_features::Vector{Int})
    return Dict{Symbol, Any}(
        :shap => (
            mean_abs_shap = create_mock_shap_values(n_features, important_features),
        ),
        :permutation => (
            importance_mean = create_mock_permutation_importance(n_features, important_features),
        ),
        :model_specific => create_mock_model_importance(n_features, important_features)
    )
end

function create_grouped_features_dataset(n_samples::Int, n_features::Int)
    # Create feature groups (e.g., different measurement types)
    n_groups = 6
    features_per_group = div(n_features, n_groups)
    
    feature_groups = Dict{Int, Vector{Int}}()
    for g in 1:n_groups
        start_idx = (g-1) * features_per_group + 1
        end_idx = g == n_groups ? n_features : g * features_per_group
        feature_groups[g] = collect(start_idx:end_idx)
    end
    
    # Generate data
    X = randn(n_samples, n_features)
    
    # Add correlation within groups
    for (group, features) in feature_groups
        if length(features) > 1
            base = randn(n_samples)
            for feat in features
                X[:, feat] = 0.7 * base + 0.3 * randn(n_samples)
            end
        end
    end
    
    # Select important features from different groups
    important_features = Int[]
    for g in 1:n_groups
        group_features = feature_groups[g]
        n_select = g <= 3 ? 2 : 1  # More from first 3 groups
        selected = randperm(length(group_features))[1:n_select]
        append!(important_features, group_features[selected])
    end
    
    # Create target
    y = sum(randn() * X[:, f] for f in important_features)
    y = y .> median(y)  # Binary classification
    
    return X, y, feature_groups
end

function create_feature_names_for_groups(feature_groups::Dict{Int, Vector{Int}})
    group_names = ["Demographic", "Clinical", "Laboratory", "Imaging", "Genetic", "Behavioral"]
    
    feature_names = String[]
    for feat_idx in 1:maximum(vcat(values(feature_groups)...))
        group = findfirst(g -> feat_idx in feature_groups[g], keys(feature_groups))
        if !isnothing(group)
            group_name = group <= length(group_names) ? group_names[group] : "Other"
            feat_num = findfirst(==(feat_idx), feature_groups[group])
            push!(feature_names, "$(group_name)_$feat_num")
        else
            push!(feature_names, "Feature_$feat_idx")
        end
    end
    
    return feature_names
end

function find_feature_group(feature_idx::Int, feature_groups::Dict{Int, Vector{Int}})
    for (group, features) in feature_groups
        if feature_idx in features
            return group
        end
    end
    return nothing
end

function generate_comprehensive_importance_results(X::Matrix{Float64}, y::Vector,
                                                 feature_groups::Dict{Int, Vector{Int}})
    n_features = size(X, 2)
    
    # Identify truly important features based on correlation with target
    correlations = [abs(cor(X[:, i], Float64.(y))) for i in 1:n_features]
    important_threshold = quantile(correlations, 0.7)
    important_features = findall(c -> c >= important_threshold, correlations)
    
    return create_importance_results_for_known_features(n_features, important_features)
end

function generate_interaction_matrix(X::Matrix{Float64}, y::Vector,
                                   feature_groups::Dict{Int, Vector{Int}})
    n_features = size(X, 2)
    matrix = zeros(n_features, n_features)
    
    # Add within-group interactions
    for (group, features) in feature_groups
        for i in features
            for j in features
                if i != j
                    # Simulate interaction based on correlation
                    matrix[i, j] = abs(cor(X[:, i], X[:, j])) * 0.3
                end
            end
        end
    end
    
    # Add some cross-group interactions
    for i in 1:n_features
        for j in (i+1):n_features
            if matrix[i, j] == 0  # Not within same group
                # Small chance of cross-group interaction
                if rand() < 0.1
                    matrix[i, j] = matrix[j, i] = 0.1 + 0.1 * rand()
                end
            end
        end
    end
    
    return matrix
end

# Run all examples
if abspath(PROGRAM_FILE) == @__FILE__
    println("Feature Ranking and Selection Examples")
    println("="^60)
    
    # Run examples
    ranking_result = basic_ranking_example()
    custom_rank, borda_rank = custom_ranking_example()
    pareto_result = pareto_selection_example()
    forward_result = forward_selection_example()
    borda_selected, avg_selected, majority_selected = ensemble_voting_example()
    final_result = complete_pipeline_example()
    
    println("\n" * "="^60)
    println("All feature ranking examples completed!")
    
    # Summary
    println("\nKey Takeaways:")
    println("1. Multiple ranking methods can be combined for robust feature selection")
    println("2. Pareto-optimal selection balances performance and complexity")
    println("3. Forward selection allows incremental feature addition with early stopping")
    println("4. Constraints ensure domain knowledge is incorporated")
    println("5. Ensemble voting across models improves selection robustness")
    println("6. Feature stability analysis helps identify reliable features")
end