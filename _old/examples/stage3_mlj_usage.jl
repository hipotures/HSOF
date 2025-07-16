# Example usage of Stage 3 MLJ evaluation infrastructure

using Random
using DataFrames
using Statistics

# Include the modules
include("../src/stage3_evaluation/mlj_infrastructure.jl")
include("../src/stage3_evaluation/unified_prediction.jl")

using .MLJInfrastructure
using .UnifiedPrediction

# Set random seed
Random.seed!(123)

"""
Example 1: Basic model creation and training
"""
function basic_model_example()
    println("=== Basic Model Example ===\n")
    
    # Generate synthetic data
    n_samples = 1000
    n_features = 20
    X = randn(n_samples, n_features)
    
    # Create target variable (binary classification)
    # True relationship: y depends on first 3 features
    y = Int.((X[:, 1] .+ 0.5 * X[:, 2] .- 0.3 * X[:, 3] .+ 0.2 * randn(n_samples)) .> 0)
    
    # Split data (simple train/test split)
    train_idx = 1:800
    test_idx = 801:1000
    
    X_train, X_test = X[train_idx, :], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Create XGBoost model with custom parameters
    println("Creating XGBoost classifier...")
    xgb_model = create_model(:xgboost, :classification,
        max_depth = 5,
        learning_rate = 0.1,
        n_estimators = 100,
        subsample = 0.8,
        colsample_bytree = 0.8
    )
    
    # Fit the model
    println("Training model...")
    xgb_fitted, xgb_machine = fit_model!(xgb_model, X_train, y_train, verbosity=1)
    
    # Create unified predictor
    predictor = UnifiedPredictor(xgb_fitted, xgb_machine, n_features=n_features)
    
    # Make predictions
    println("\nMaking predictions...")
    result = predict_unified(predictor, X_test, return_proba=true)
    
    # Calculate metrics
    metrics = get_prediction_metrics(result, y_test)
    
    println("\nTest Set Performance:")
    println("  Accuracy: $(round(metrics[:accuracy], digits=3))")
    println("  Precision: $(round(metrics[:precision], digits=3))")
    println("  Recall: $(round(metrics[:recall], digits=3))")
    println("  F1-Score: $(round(metrics[:f1_score], digits=3))")
    println("  Avg Confidence: $(round(metrics[:avg_confidence], digits=3))")
    
    return predictor, result
end

"""
Example 2: Model comparison
"""
function model_comparison_example()
    println("\n\n=== Model Comparison Example ===\n")
    
    # Generate regression data
    n_samples = 500
    n_features = 15
    X = randn(n_samples, n_features)
    
    # True relationship: non-linear with interactions
    y = (
        2 * X[:, 1] .+ 
        X[:, 2].^2 .+ 
        0.5 * X[:, 3] .* X[:, 4] .- 
        X[:, 5] .+ 
        0.3 * randn(n_samples)
    )
    
    # Train different models
    models = Dict(
        "XGBoost" => create_model(:xgboost, :regression, n_estimators=50),
        "Random Forest" => create_model(:random_forest, :regression, n_estimators=50),
        "LightGBM" => create_model(:lightgbm, :regression, n_estimators=50)
    )
    
    results = Dict{String, Dict{Symbol, Float64}}()
    
    for (name, model) in models
        println("Training $name...")
        
        # Fit model
        fitted_model, machine = fit_model!(model, X, y, verbosity=0)
        
        # Create predictor
        predictor = UnifiedPredictor(fitted_model, machine)
        
        # Make predictions (on same data for simplicity)
        result = predict_unified(predictor, X)
        
        # Get metrics
        metrics = get_prediction_metrics(result, y)
        results[name] = metrics
        
        println("  RMSE: $(round(metrics[:rmse], digits=3))")
        println("  R²: $(round(metrics[:r2], digits=3))")
    end
    
    # Find best model
    best_model = argmin(Dict(k => v[:rmse] for (k, v) in results))
    println("\nBest model: $best_model (RMSE: $(round(results[best_model][:rmse], digits=3)))")
    
    return results
end

"""
Example 3: Ensemble learning
"""
function ensemble_example()
    println("\n\n=== Ensemble Learning Example ===\n")
    
    # Generate challenging classification data
    n_samples = 600
    n_features = 10
    X = randn(n_samples, n_features)
    
    # Create two clusters with some overlap
    cluster1 = 1:300
    cluster2 = 301:600
    X[cluster1, 1:2] .+= 1.5
    X[cluster2, 1:2] .-= 1.5
    
    y = vcat(zeros(Int, 300), ones(Int, 300))
    # Add some label noise
    noise_idx = rand(1:n_samples, 30)
    y[noise_idx] = 1 .- y[noise_idx]
    
    # Create diverse models
    println("Creating ensemble models...")
    model_configs = [
        create_model(:xgboost, :classification, max_depth=3, learning_rate=0.3),
        create_model(:xgboost, :classification, max_depth=5, learning_rate=0.1),
        create_model(:random_forest, :classification, n_estimators=100),
        create_model(:lightgbm, :classification, num_leaves=15),
        create_model(:lightgbm, :classification, num_leaves=31, learning_rate=0.05)
    ]
    
    # Train all models
    predictors = UnifiedPredictor[]
    for (i, model) in enumerate(model_configs)
        println("  Training model $i...")
        fitted_model, machine = fit_model!(model, X, y, verbosity=0)
        push!(predictors, UnifiedPredictor(fitted_model, machine))
    end
    
    # Individual model predictions
    println("\nIndividual model performance:")
    for (i, predictor) in enumerate(predictors)
        result = predict_unified(predictor, X)
        accuracy = mean(result.predictions .== y)
        println("  Model $i accuracy: $(round(accuracy, digits=3))")
    end
    
    # Ensemble predictions
    println("\nEnsemble performance:")
    
    # Hard voting
    hard_preds = ensemble_predict(predictors, X, voting=:hard)
    hard_accuracy = mean(hard_preds .== y)
    println("  Hard voting accuracy: $(round(hard_accuracy, digits=3))")
    
    # Soft voting
    soft_preds = ensemble_predict(predictors, X, voting=:soft)
    soft_accuracy = mean(soft_preds .== y)
    println("  Soft voting accuracy: $(round(soft_accuracy, digits=3))")
    
    # Weighted voting (give more weight to better models)
    weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    weighted_preds = ensemble_predict(predictors, X, voting=:soft, weights=weights)
    weighted_accuracy = mean(weighted_preds .== y)
    println("  Weighted voting accuracy: $(round(weighted_accuracy, digits=3))")
    
    return predictors
end

"""
Example 4: Model persistence
"""
function model_persistence_example()
    println("\n\n=== Model Persistence Example ===\n")
    
    # Create and train a model
    X = randn(200, 8)
    y = Int.(sum(X[:, 1:3], dims=2)[:, 1] .> 0)
    
    model = create_model(:lightgbm, :classification, n_estimators=30)
    fitted_model, machine = fit_model!(model, X, y, verbosity=0)
    
    # Save model
    model_path = tempname() * "_model.jld2"
    println("Saving model to: $model_path")
    save_model(fitted_model, machine, model_path)
    
    # Load model
    println("Loading model...")
    loaded_model, loaded_machine = load_model(model_path)
    
    # Verify loaded model works
    predictor = UnifiedPredictor(loaded_model, loaded_machine)
    result = predict_unified(predictor, X[1:10, :])
    
    println("Loaded model predictions: ", result.predictions)
    println("Model successfully saved and loaded!")
    
    # Cleanup
    rm(model_path)
    
    return loaded_model
end

"""
Example 5: Feature importance analysis (placeholder)
"""
function feature_importance_example()
    println("\n\n=== Feature Importance Example ===\n")
    
    # Generate data with known important features
    n_samples = 500
    n_features = 20
    X = randn(n_samples, n_features)
    
    # Only first 5 features are important
    y = (
        2.0 * X[:, 1] .+
        1.5 * X[:, 2] .-
        1.0 * X[:, 3] .+
        0.5 * X[:, 4] .+
        0.2 * X[:, 5] .+
        0.1 * randn(n_samples)
    )
    
    # Train model
    model = create_model(:xgboost, :regression, n_estimators=100)
    fitted_model, machine = fit_model!(model, X, y, verbosity=0)
    
    println("Model trained. Feature importance extraction would be implemented here.")
    println("Important features: 1-5 (by design)")
    println("Model R²: ", round(1 - sum((predict_model(fitted_model, machine, X) .- y).^2) / sum((y .- mean(y)).^2), digits=3))
    
    return fitted_model
end

# Run all examples
if abspath(PROGRAM_FILE) == @__FILE__
    println("Stage 3 MLJ Infrastructure Examples")
    println("=" ^ 50)
    
    # Run examples
    predictor1, result1 = basic_model_example()
    results2 = model_comparison_example()
    predictors3 = ensemble_example()
    model4 = model_persistence_example()
    model5 = feature_importance_example()
    
    println("\n" * "=" ^ 50)
    println("All examples completed successfully!")
end