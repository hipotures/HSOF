# Example usage of Cross-Validation system for Stage 3

using Random
using Statistics
using DataFrames

# Include the modules
include("../src/stage3_evaluation/cross_validation.jl")
include("../src/stage3_evaluation/mlj_infrastructure.jl")
include("../src/stage3_evaluation/unified_prediction.jl")

using .CrossValidation
using .MLJInfrastructure
using .UnifiedPrediction

# Set random seed
Random.seed!(123)

"""
Example 1: Basic stratified k-fold cross-validation
"""
function basic_cv_example()
    println("=== Basic Stratified K-Fold Example ===\n")
    
    # Generate synthetic classification data
    n_samples = 200
    n_features = 10
    X = randn(n_samples, n_features)
    
    # Create target with 3 classes
    y = zeros(Int, n_samples)
    y[1:80] .= 1
    y[81:140] .= 2
    y[141:200] .= 3
    
    # Add some noise
    shuffle_idx = shuffle(1:n_samples)
    y = y[shuffle_idx]
    X = X[shuffle_idx, :]
    
    # Create stratified k-fold
    cv = StratifiedKFold(n_folds=5, shuffle=true, random_state=42)
    cv_iter = create_folds(cv, y)
    
    println("Created $(cv_iter.n_folds) stratified folds")
    
    # Examine fold distribution
    println("\nFold sizes and class distribution:")
    for i in 1:cv_iter.n_folds
        y_train = y[cv_iter.train_indices[i]]
        y_test = y[cv_iter.test_indices[i]]
        
        train_dist = [sum(y_train .== c) for c in 1:3]
        test_dist = [sum(y_test .== c) for c in 1:3]
        
        println("Fold $i:")
        println("  Train: $(length(y_train)) samples, distribution: $train_dist")
        println("  Test:  $(length(y_test)) samples, distribution: $test_dist")
    end
    
    # Validate folds
    is_valid, validation_results = validate_folds(cv_iter, y, min_samples_per_class=5)
    println("\nFold validation: $(is_valid ? "PASSED" : "FAILED")")
    
    return cv_iter, X, y
end

"""
Example 2: Cross-validation for regression with quantile stratification
"""
function regression_cv_example()
    println("\n\n=== Regression with Quantile Stratification Example ===\n")
    
    # Generate regression data
    n_samples = 300
    n_features = 8
    X = randn(n_samples, n_features)
    
    # Non-linear target with heteroscedastic noise
    y = 2 * X[:, 1].^2 + 0.5 * X[:, 2] .* X[:, 3] - X[:, 4] + 
        0.1 * randn(n_samples) .* (1 .+ abs.(X[:, 1]))
    
    # Create quantile-stratified CV
    cv = QuantileStratifiedKFold(n_folds=5, n_quantiles=10, random_state=42)
    cv_iter = create_folds(cv, y)
    
    println("Created $(cv_iter.n_folds) quantile-stratified folds")
    
    # Examine target distribution in folds
    println("\nTarget value statistics per fold:")
    for i in 1:cv_iter.n_folds
        y_test = y[cv_iter.test_indices[i]]
        println("Fold $i: mean=$(round(mean(y_test), digits=2)), " *
                "std=$(round(std(y_test), digits=2)), " *
                "range=[$(round(minimum(y_test), digits=2)), $(round(maximum(y_test), digits=2))]")
    end
    
    # Run cross-validation with a model
    model = create_model(:random_forest, :regression, n_estimators=50)
    
    println("\nRunning cross-validation...")
    results = cross_validate_model(model.model, X, y, cv, 
                                 task_type=:regression, 
                                 verbose=false)
    
    # Display results
    agg_metrics = results["aggregated_metrics"]
    println("\nCross-validation results:")
    println("  RMSE: $(round(agg_metrics[:rmse_mean], digits=3)) ± $(round(agg_metrics[:rmse_std], digits=3))")
    println("  MAE:  $(round(agg_metrics[:mae_mean], digits=3)) ± $(round(agg_metrics[:mae_std], digits=3))")
    println("  R²:   $(round(agg_metrics[:r2_mean], digits=3)) ± $(round(agg_metrics[:r2_std], digits=3))")
    
    return results
end

"""
Example 3: Handling imbalanced data with SMOTE
"""
function imbalanced_cv_example()
    println("\n\n=== Imbalanced Data with SMOTE Example ===\n")
    
    # Create highly imbalanced dataset
    n_majority = 900
    n_minority = 100
    n_features = 15
    
    X_maj = randn(n_majority, n_features)
    X_min = randn(n_minority, n_features) .+ 2  # Shifted distribution
    
    X = vcat(X_maj, X_min)
    y = vcat(zeros(Int, n_majority), ones(Int, n_minority))
    
    # Shuffle
    shuffle_idx = shuffle(1:length(y))
    X = X[shuffle_idx, :]
    y = y[shuffle_idx]
    
    println("Original class distribution:")
    println("  Class 0: $(sum(y .== 0)) samples")
    println("  Class 1: $(sum(y .== 1)) samples")
    println("  Imbalance ratio: $(round(sum(y .== 0) / sum(y .== 1), digits=2)):1")
    
    # Create model
    model = create_model(:lightgbm, :classification, n_estimators=50)
    
    # CV without SMOTE
    cv = StratifiedKFold(n_folds=5, random_state=42)
    
    println("\nCross-validation WITHOUT SMOTE:")
    results_no_smote = cross_validate_model(model.model, X, y, cv,
                                          task_type=:classification,
                                          apply_smote=false,
                                          verbose=false)
    
    metrics_no_smote = results_no_smote["aggregated_metrics"]
    println("  F1 Score: $(round(metrics_no_smote[:f1_score_mean], digits=3))")
    println("  Recall:   $(round(metrics_no_smote[:recall_mean], digits=3))")
    
    # CV with SMOTE
    println("\nCross-validation WITH SMOTE:")
    smote_sampler = SMOTESampler(sampling_ratio=0.5, k_neighbors=5, random_state=42)
    results_smote = cross_validate_model(model.model, X, y, cv,
                                       task_type=:classification,
                                       apply_smote=true,
                                       smote_params=smote_sampler,
                                       verbose=false)
    
    metrics_smote = results_smote["aggregated_metrics"]
    println("  F1 Score: $(round(metrics_smote[:f1_score_mean], digits=3))")
    println("  Recall:   $(round(metrics_smote[:recall_mean], digits=3))")
    
    return results_no_smote, results_smote
end

"""
Example 4: Model comparison using cross-validation
"""
function model_comparison_cv_example()
    println("\n\n=== Model Comparison with CV Example ===\n")
    
    # Generate data with feature interactions
    n_samples = 500
    n_features = 20
    X = randn(n_samples, n_features)
    
    # Target depends on specific features and interactions
    y = Int.((
        2 * X[:, 1] .+ 
        1.5 * X[:, 2].^2 .+ 
        X[:, 3] .* X[:, 4] .- 
        0.5 * X[:, 5] .+ 
        0.3 * randn(n_samples)
    ) .> 0)
    
    # Models to compare
    models = Dict(
        "XGBoost" => create_model(:xgboost, :classification, 
                                 max_depth=4, n_estimators=100),
        "Random Forest" => create_model(:random_forest, :classification, 
                                       n_estimators=100),
        "LightGBM" => create_model(:lightgbm, :classification, 
                                  num_leaves=31, n_estimators=100)
    )
    
    # Cross-validation setup
    cv = StratifiedKFold(n_folds=5, random_state=42)
    
    # Compare models
    model_results = Dict{String, Dict}()
    
    println("Comparing models with 5-fold CV:")
    for (name, model_wrapper) in models
        println("\nEvaluating $name...")
        
        results = cross_validate_model(model_wrapper.model, X, y, cv,
                                     task_type=:classification,
                                     verbose=false)
        
        model_results[name] = results
        
        # Display metrics
        metrics = results["aggregated_metrics"]
        println("  Accuracy: $(round(metrics[:accuracy_mean], digits=3)) ± $(round(metrics[:accuracy_std], digits=3))")
        println("  F1 Score: $(round(metrics[:f1_score_mean], digits=3)) ± $(round(metrics[:f1_score_std], digits=3))")
    end
    
    # Find best model
    best_model = argmax(Dict(
        name => results["aggregated_metrics"][:f1_score_mean] 
        for (name, results) in model_results
    ))
    
    println("\nBest model: $best_model")
    
    return model_results
end

"""
Example 5: Feature selection stability using CV
"""
function feature_selection_cv_example()
    println("\n\n=== Feature Selection Stability Example ===\n")
    
    # Generate data with relevant and irrelevant features
    n_samples = 400
    n_relevant = 5
    n_irrelevant = 15
    n_features = n_relevant + n_irrelevant
    
    X = randn(n_samples, n_features)
    
    # Only first n_relevant features matter
    y = Int.(
        X[:, 1] .+ 0.8 * X[:, 2] .- 0.6 * X[:, 3] .+ 
        0.4 * X[:, 4] .- 0.2 * X[:, 5] .+ 0.3 * randn(n_samples)
    ) .> 0
    
    # Create CV splitter with return_predictions
    cv = StratifiedKFold(n_folds=5, random_state=42)
    
    # Track feature importance across folds
    feature_importance_folds = []
    
    println("Running CV to assess feature importance stability...")
    
    # Manual CV loop for feature importance extraction
    cv_iter = create_folds(cv, y)
    
    for (fold_idx, (train_idx, test_idx)) in enumerate(zip(cv_iter.train_indices, cv_iter.test_indices))
        # Train model
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        
        model = create_model(:xgboost, :classification, n_estimators=50)
        fitted_model, machine = fit_model!(model, X_train, y_train, verbosity=0)
        
        # Simulate feature importance (in practice, extract from model)
        # For now, use coefficient-like values based on correlation
        importance = abs.([cor(X_train[:, i], y_train) for i in 1:n_features])
        push!(feature_importance_folds, importance)
    end
    
    # Analyze stability
    importance_matrix = hcat(feature_importance_folds...)
    mean_importance = mean(importance_matrix, dims=2)[:, 1]
    std_importance = std(importance_matrix, dims=2)[:, 1]
    
    # Rank features
    feature_ranks = sortperm(mean_importance, rev=true)
    
    println("\nTop 10 features by average importance:")
    for i in 1:10
        feat_idx = feature_ranks[i]
        println("  Feature $feat_idx: $(round(mean_importance[feat_idx], digits=3)) " *
                "± $(round(std_importance[feat_idx], digits=3))")
    end
    
    # Calculate stability ratio (std/mean)
    stability_ratios = std_importance ./ (mean_importance .+ 1e-10)
    println("\nFeature selection stability (lower is better):")
    println("  Mean stability ratio: $(round(mean(stability_ratios), digits=3))")
    println("  Most stable features: $(feature_ranks[sortperm(stability_ratios[feature_ranks])[1:5]])")
    
    return importance_matrix, feature_ranks
end

"""
Example 6: Nested cross-validation for hyperparameter tuning
"""
function nested_cv_example()
    println("\n\n=== Nested Cross-Validation Example ===\n")
    
    # Generate data
    n_samples = 300
    n_features = 10
    X = randn(n_samples, n_features)
    y = Int.(X[:, 1] .+ 0.5 * X[:, 2] .> 0.2 * randn(n_samples))
    
    # Outer CV for model evaluation
    outer_cv = StratifiedKFold(n_folds=5, random_state=42)
    
    # Inner CV for hyperparameter tuning
    inner_cv = StratifiedKFold(n_folds=3, random_state=123)
    
    # Hyperparameter grid
    param_grid = [
        Dict(:max_depth => 3, :learning_rate => 0.1),
        Dict(:max_depth => 5, :learning_rate => 0.1),
        Dict(:max_depth => 3, :learning_rate => 0.3),
        Dict(:max_depth => 5, :learning_rate => 0.3)
    ]
    
    println("Running nested CV with:")
    println("  Outer folds: $(outer_cv.n_folds) (model evaluation)")
    println("  Inner folds: $(inner_cv.n_folds) (hyperparameter tuning)")
    println("  Parameter combinations: $(length(param_grid))")
    
    # Simplified nested CV (in practice, use more sophisticated approach)
    outer_results = []
    
    outer_iter = create_folds(outer_cv, y)
    
    for (outer_fold, (train_idx, test_idx)) in enumerate(zip(outer_iter.train_indices, outer_iter.test_indices))
        println("\nOuter fold $outer_fold:")
        
        X_train_outer = X[train_idx, :]
        y_train_outer = y[train_idx]
        X_test_outer = X[test_idx, :]
        y_test_outer = y[test_idx]
        
        # Find best parameters using inner CV
        best_score = -Inf
        best_params = nothing
        
        for params in param_grid
            # Inner CV
            model = create_model(:xgboost, :classification; 
                               n_estimators=50, params...)
            
            inner_results = cross_validate_model(model.model, X_train_outer, y_train_outer, inner_cv,
                                               task_type=:classification,
                                               verbose=false)
            
            score = inner_results["aggregated_metrics"][:accuracy_mean]
            
            if score > best_score
                best_score = score
                best_params = params
            end
        end
        
        println("  Best params: $best_params")
        println("  Inner CV score: $(round(best_score, digits=3))")
        
        # Train on full outer training set with best params
        best_model = create_model(:xgboost, :classification; 
                                n_estimators=50, best_params...)
        fitted_model, machine = fit_model!(best_model, X_train_outer, y_train_outer, verbosity=0)
        
        # Evaluate on outer test set
        predictor = UnifiedPredictor(fitted_model, machine)
        result = predict_unified(predictor, X_test_outer)
        
        outer_score = mean(result.predictions .== y_test_outer)
        push!(outer_results, outer_score)
        println("  Outer test score: $(round(outer_score, digits=3))")
    end
    
    println("\nFinal nested CV result: $(round(mean(outer_results), digits=3)) ± $(round(std(outer_results), digits=3))")
    
    return outer_results
end

# Run all examples
if abspath(PROGRAM_FILE) == @__FILE__
    println("Cross-Validation System Examples")
    println("=" ^ 50)
    
    # Run examples
    cv_iter1, X1, y1 = basic_cv_example()
    results2 = regression_cv_example()
    results3_no_smote, results3_smote = imbalanced_cv_example()
    model_results4 = model_comparison_cv_example()
    importance5, ranks5 = feature_selection_cv_example()
    nested_results6 = nested_cv_example()
    
    println("\n" * "=" ^ 50)
    println("All cross-validation examples completed successfully!")
end