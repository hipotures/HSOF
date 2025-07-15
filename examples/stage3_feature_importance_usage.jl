# Example usage of Feature Importance analysis for Stage 3

using Random
using Statistics
using DataFrames
using CSV

# Include the modules
include("../src/stage3_evaluation/feature_importance.jl")
include("../src/stage3_evaluation/mlj_infrastructure.jl")
include("../src/stage3_evaluation/cross_validation.jl")
include("../src/stage3_evaluation/unified_prediction.jl")

using .FeatureImportance
using .MLJInfrastructure
using .CrossValidation
using .UnifiedPrediction

# Set random seed
Random.seed!(123)

"""
Example 1: Basic SHAP values calculation
"""
function basic_shap_example()
    println("=== Basic SHAP Values Example ===\n")
    
    # Generate synthetic data with known feature importance
    n_samples = 500
    n_features = 10
    X = randn(n_samples, n_features)
    
    # Create target where features 1-3 are important, others are noise
    y = Int.((
        2.0 * X[:, 1] +              # Strong positive effect
        -1.5 * X[:, 2] +             # Strong negative effect
        1.0 * X[:, 3] +              # Moderate effect
        0.1 * X[:, 4] +              # Weak effect
        0.2 * randn(n_samples)       # Noise
    ) .> 0)
    
    println("Data shape: $(size(X))")
    println("Class distribution: $(countmap(y))")
    
    # Feature names
    feature_names = ["Strong_Pos", "Strong_Neg", "Moderate", "Weak", 
                    "Noise1", "Noise2", "Noise3", "Noise4", "Noise5", "Noise10"]
    
    # Train XGBoost model
    println("\nTraining XGBoost model...")
    model = create_model(:xgboost, :classification, 
                        n_estimators=100, 
                        max_depth=4,
                        learning_rate=0.1)
    fitted_model, machine = fit_model!(model, X, y, verbosity=0)
    
    # Calculate SHAP values
    println("\nCalculating SHAP values...")
    shap_calc = SHAPCalculator(
        model_type=:xgboost,
        n_samples=200,  # Use subset for efficiency
        baseline_type=:mean,
        random_state=42
    )
    
    shap_result = calculate_shap_values(shap_calc, fitted_model, machine, X, y,
                                      feature_names=feature_names)
    
    # Display results
    println("\n--- SHAP Feature Importance ---")
    importance_df = DataFrame(
        Feature = shap_result.feature_names,
        Importance = round.(shap_result.importance_values, digits=4),
        CI_Lower = round.(shap_result.confidence_intervals[:, 1], digits=4),
        CI_Upper = round.(shap_result.confidence_intervals[:, 2], digits=4)
    )
    sort!(importance_df, :Importance, rev=true)
    
    println(first(importance_df, 10))
    
    # Verify that important features are detected
    top_3_features = importance_df.Feature[1:3]
    println("\nTop 3 features: $top_3_features")
    println("Expected: Strong_Pos, Strong_Neg, Moderate (in any order)")
    
    return shap_result
end

"""
Example 2: Permutation importance analysis
"""
function permutation_importance_example()
    println("\n\n=== Permutation Importance Example ===\n")
    
    # Generate regression data
    n_samples = 400
    n_features = 8
    X = randn(n_samples, n_features)
    
    # Non-linear relationships
    y = (
        X[:, 1].^2 +                    # Quadratic effect
        2 * sin.(X[:, 2]) +             # Non-linear
        X[:, 3] .* X[:, 4] +            # Interaction
        0.5 * X[:, 5] +                 # Linear
        0.3 * randn(n_samples)          # Noise
    )
    
    feature_names = ["Quadratic", "Sine", "Inter1", "Inter2", 
                    "Linear", "Noise1", "Noise2", "Noise3"]
    
    # Train Random Forest for regression
    println("Training Random Forest model...")
    model = create_model(:random_forest, :regression,
                        n_estimators=100,
                        max_depth=10)
    fitted_model, machine = fit_model!(model, X, y, verbosity=0)
    
    # Calculate permutation importance
    println("\nCalculating permutation importance...")
    perm_calc = PermutationImportance(
        n_repeats=20,
        scoring_function=rmse,  # Use RMSE for regression
        random_state=42,
        n_jobs=min(4, Threads.nthreads())
    )
    
    perm_result = calculate_permutation_importance(perm_calc, fitted_model, machine, X, y,
                                                 feature_names=feature_names)
    
    # Display results
    println("\n--- Permutation Feature Importance ---")
    importance_df = DataFrame(
        Feature = perm_result.feature_names,
        Importance = round.(perm_result.importance_values, digits=4),
        CI_Lower = round.(perm_result.confidence_intervals[:, 1], digits=4),
        CI_Upper = round.(perm_result.confidence_intervals[:, 2], digits=4)
    )
    sort!(importance_df, :Importance, rev=true)
    
    println(importance_df)
    
    # Note: For permutation importance, higher values = more important
    println("\nBaseline RMSE: $(round(perm_result.metadata[:baseline_score], digits=4))")
    
    return perm_result
end

"""
Example 3: Feature interactions analysis
"""
function feature_interactions_example()
    println("\n\n=== Feature Interactions Example ===\n")
    
    # Create data with strong interactions
    n_samples = 600
    n_features = 6
    X = randn(n_samples, n_features)
    
    # Target with explicit interactions
    y = Int.((
        X[:, 1] +                           # Main effect 1
        X[:, 2] +                           # Main effect 2
        3 * X[:, 1] .* X[:, 2] +           # Strong interaction 1-2
        1.5 * X[:, 3] .* X[:, 4] +         # Moderate interaction 3-4
        0.5 * X[:, 5] +                    # Weak main effect
        0.3 * randn(n_samples)
    ) .> 0)
    
    feature_names = ["Main1", "Main2", "Inter3", "Inter4", "Weak", "Noise"]
    
    println("Data with interactions:")
    println("- Strong interaction: Main1 × Main2")
    println("- Moderate interaction: Inter3 × Inter4")
    
    # Train model
    model = create_model(:xgboost, :classification,
                        n_estimators=100,
                        max_depth=5)
    fitted_model, machine = fit_model!(model, X, y, verbosity=0)
    
    # Calculate interactions
    println("\nCalculating feature interactions...")
    shap_calc = SHAPCalculator(model_type=:xgboost, n_samples=200, random_state=42)
    
    interaction_matrix, top_interactions = get_feature_interactions(
        shap_calc, fitted_model, machine, X, y,
        feature_names=feature_names,
        top_k=10
    )
    
    # Display interaction matrix
    println("\n--- Feature Interaction Matrix ---")
    println("(Showing interaction strengths)")
    
    # Create readable matrix
    interaction_df = DataFrame(interaction_matrix.interaction_values, :auto)
    rename!(interaction_df, [Symbol(name) for name in feature_names])
    insertcols!(interaction_df, 1, :Feature => feature_names)
    
    # Round values for display
    for col in names(interaction_df)[2:end]
        interaction_df[!, col] = round.(interaction_df[!, col], digits=3)
    end
    
    println(interaction_df)
    
    # Display top interactions
    println("\n--- Top Feature Interactions ---")
    for (i, (feat1_idx, feat2_idx, strength)) in enumerate(top_interactions)
        feat1 = feature_names[feat1_idx]
        feat2 = feature_names[feat2_idx]
        println("$i. $feat1 × $feat2: $(round(strength, digits=4))")
    end
    
    return interaction_matrix, top_interactions
end

"""
Example 4: Cross-validation aggregation
"""
function cv_importance_aggregation_example()
    println("\n\n=== CV Importance Aggregation Example ===\n")
    
    # Generate data
    n_samples = 300
    n_features = 8
    X = randn(n_samples, n_features)
    
    # Create target
    y = Int.((
        1.5 * X[:, 1] + 
        1.0 * X[:, 2] - 
        0.8 * X[:, 3] + 
        0.3 * randn(n_samples)
    ) .> 0)
    
    feature_names = ["Feat$i" for i in 1:n_features]
    
    # Set up cross-validation
    cv = StratifiedKFold(n_folds=5, shuffle=true, random_state=42)
    cv_iter = create_folds(cv, y)
    
    # Calculate importance for each fold
    println("Calculating importance across CV folds...")
    fold_importance_results = ImportanceResult[]
    
    for (fold_idx, (train_idx, test_idx)) in enumerate(zip(cv_iter.train_indices, cv_iter.test_indices))
        println("\nFold $fold_idx/$(cv.n_folds)")
        
        # Split data
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model = create_model(:lightgbm, :classification,
                           n_estimators=50,
                           num_leaves=20)
        fitted_model, machine = fit_model!(model, X_train, y_train, verbosity=0)
        
        # Calculate SHAP for this fold
        shap_calc = SHAPCalculator(model_type=:lightgbm, n_samples=100)
        shap_result = calculate_shap_values(shap_calc, fitted_model, machine, 
                                          X_train, y_train, feature_names=feature_names)
        
        push!(fold_importance_results, shap_result)
    end
    
    # Aggregate across folds
    println("\nAggregating importance across folds...")
    
    # Mean aggregation
    agg_mean = aggregate_importance_cv(fold_importance_results, aggregation_method=:mean)
    
    # Median aggregation (more robust)
    agg_median = aggregate_importance_cv(fold_importance_results, aggregation_method=:median)
    
    # Compare results
    println("\n--- Aggregated Feature Importance ---")
    comparison_df = DataFrame(
        Feature = feature_names,
        Mean_Importance = round.(agg_mean.importance_values, digits=4),
        Mean_CI_Width = round.(agg_mean.confidence_intervals[:, 2] - 
                              agg_mean.confidence_intervals[:, 1], digits=4),
        Median_Importance = round.(agg_median.importance_values, digits=4),
        Std_Dev = round.(agg_mean.metadata[:std_deviation], digits=4)
    )
    sort!(comparison_df, :Mean_Importance, rev=true)
    
    println(comparison_df)
    
    println("\nNote: Smaller CI width indicates more stable importance across folds")
    
    return agg_mean, agg_median
end

"""
Example 5: Combined SHAP and permutation importance
"""
function combined_importance_example()
    println("\n\n=== Combined Importance Methods Example ===\n")
    
    # Generate data
    n_samples = 400
    n_features = 10
    X = randn(n_samples, n_features)
    
    # Complex target with various effects
    y = Int.((
        2 * X[:, 1] +                    # Strong linear
        X[:, 2].^2 +                     # Quadratic
        sign.(X[:, 3]) .* sqrt.(abs.(X[:, 3])) +  # Non-linear
        0.5 * X[:, 4] +                  # Weak linear
        0.3 * randn(n_samples)
    ) .> 0.5)
    
    feature_names = ["Strong_Linear", "Quadratic", "NonLinear", "Weak_Linear",
                    "Noise1", "Noise2", "Noise3", "Noise4", "Noise5", "Noise10"]
    
    # Train model
    println("Training XGBoost model...")
    model = create_model(:xgboost, :classification,
                        n_estimators=100,
                        max_depth=4)
    fitted_model, machine = fit_model!(model, X, y, verbosity=0)
    
    # Calculate SHAP importance
    println("\nCalculating SHAP values...")
    shap_calc = SHAPCalculator(model_type=:xgboost, n_samples=200, random_state=42)
    shap_result = calculate_shap_values(shap_calc, fitted_model, machine, X, y,
                                      feature_names=feature_names)
    
    # Calculate permutation importance
    println("Calculating permutation importance...")
    perm_calc = PermutationImportance(n_repeats=10, random_state=42)
    perm_result = calculate_permutation_importance(perm_calc, fitted_model, machine, X, y,
                                                 feature_names=feature_names)
    
    # Combine methods with different weights
    println("\nCombining importance methods...")
    
    # Equal weights
    combined_equal = combine_importance_methods(shap_result, perm_result, 
                                              weights=(0.5, 0.5))
    
    # SHAP-heavy (more trust in model interpretation)
    combined_shap_heavy = combine_importance_methods(shap_result, perm_result,
                                                   weights=(0.7, 0.3))
    
    # Permutation-heavy (more trust in empirical measurement)
    combined_perm_heavy = combine_importance_methods(shap_result, perm_result,
                                                   weights=(0.3, 0.7))
    
    # Compare all methods
    println("\n--- Importance Method Comparison ---")
    comparison_df = DataFrame(
        Feature = feature_names,
        SHAP = round.(shap_result.importance_values / 
                     maximum(shap_result.importance_values), digits=3),
        Permutation = round.(perm_result.importance_values / 
                           maximum(perm_result.importance_values), digits=3),
        Combined_Equal = round.(combined_equal.importance_values, digits=3),
        Combined_SHAP = round.(combined_shap_heavy.importance_values, digits=3),
        Combined_Perm = round.(combined_perm_heavy.importance_values, digits=3)
    )
    sort!(comparison_df, :Combined_Equal, rev=true)
    
    println(comparison_df)
    
    # Export top features
    println("\nExporting importance rankings...")
    export_importance_plot(combined_equal, "feature_importance_combined.csv", top_k=10)
    println("Saved to: feature_importance_combined.csv")
    
    return combined_equal
end

"""
Example 6: Large dataset with sampling
"""
function large_dataset_example()
    println("\n\n=== Large Dataset with Sampling Example ===\n")
    
    # Generate large dataset
    n_samples = 10000
    n_features = 50
    
    println("Generating large dataset: $n_samples × $n_features")
    X = randn(n_samples, n_features)
    
    # Only first 5 features matter
    y = Int.((
        X[:, 1] + 0.8 * X[:, 2] - 0.6 * X[:, 3] + 
        0.4 * X[:, 4] - 0.2 * X[:, 5] + 
        0.1 * randn(n_samples)
    ) .> 0)
    
    # Train model
    println("\nTraining model on full dataset...")
    model = create_model(:lightgbm, :classification,
                        n_estimators=100,
                        num_leaves=31,
                        learning_rate=0.1)
    
    @time fitted_model, machine = fit_model!(model, X, y, verbosity=0)
    
    # Calculate SHAP with sampling
    println("\nCalculating SHAP values with sampling...")
    
    # Small sample
    shap_calc_small = SHAPCalculator(
        model_type=:lightgbm,
        n_samples=500,  # Only use 500 samples
        random_state=42
    )
    
    @time shap_small = calculate_shap_values(shap_calc_small, fitted_model, machine, X, y)
    
    # Medium sample
    shap_calc_medium = SHAPCalculator(
        model_type=:lightgbm,
        n_samples=2000,
        random_state=42
    )
    
    @time shap_medium = calculate_shap_values(shap_calc_medium, fitted_model, machine, X, y)
    
    # Compare sampling results
    println("\n--- Effect of Sampling on Feature Importance ---")
    
    # Get top 10 features from each
    top_10_small = sortperm(shap_small.importance_values, rev=true)[1:10]
    top_10_medium = sortperm(shap_medium.importance_values, rev=true)[1:10]
    
    println("Top 10 features (500 samples): $top_10_small")
    println("Top 10 features (2000 samples): $top_10_medium")
    
    # Check overlap
    overlap = length(intersect(top_10_small, top_10_medium))
    println("Overlap in top 10: $overlap/10")
    
    # Check if true important features are captured
    true_important = [1, 2, 3, 4, 5]
    captured_small = length(intersect(top_10_small, true_important))
    captured_medium = length(intersect(top_10_medium, true_important))
    
    println("\nTrue important features captured:")
    println("  500 samples: $captured_small/5")
    println("  2000 samples: $captured_medium/5")
    
    return shap_small, shap_medium
end

# Run all examples
if abspath(PROGRAM_FILE) == @__FILE__
    println("Feature Importance Analysis Examples")
    println("=" ^ 50)
    
    # Run examples
    shap_result = basic_shap_example()
    perm_result = permutation_importance_example()
    interaction_matrix, top_interactions = feature_interactions_example()
    agg_mean, agg_median = cv_importance_aggregation_example()
    combined_result = combined_importance_example()
    
    # Large dataset example (optional - takes longer)
    println("\n" * "=" * 50)
    println("Run large dataset example? (y/n): ", )
    if readline() == "y"
        shap_small, shap_medium = large_dataset_example()
    end
    
    println("\n" * "=" ^ 50)
    println("All feature importance examples completed!")
    
    # Summary
    println("\nKey takeaways:")
    println("1. SHAP values provide model-specific feature importance")
    println("2. Permutation importance is model-agnostic and robust")
    println("3. Feature interactions can reveal complex relationships")
    println("4. CV aggregation provides stable importance estimates")
    println("5. Combining methods leverages strengths of each approach")
    println("6. Sampling enables analysis of large datasets efficiently")
end