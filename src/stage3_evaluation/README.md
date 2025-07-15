# Stage 3 Precise Evaluation Module

This module implements the MLJ.jl-based model evaluation infrastructure for the final stage of feature selection, reducing features from 50 to 10-20 optimal features using real ML models with cross-validation.

## Overview

Stage 3 provides a unified interface for training and evaluating multiple ML models:
- **XGBoost**: Gradient boosting with tree-based learners
- **Random Forest**: Ensemble of decision trees with bagging
- **LightGBM**: Fast gradient boosting implementation

## Components

### 1. MLJ Infrastructure (`mlj_infrastructure.jl`)

Core infrastructure for model management:

```julia
# Create models with default configuration
xgb_model = create_model(:xgboost, :classification)
rf_model = create_model(:random_forest, :regression)
lgbm_model = create_model(:lightgbm, :classification)

# Create with custom parameters
model = create_model(:xgboost, :classification,
    max_depth = 5,
    learning_rate = 0.1,
    n_estimators = 100
)

# Fit model
fitted_model, machine = fit_model!(model, X, y)

# Make predictions
predictions = predict_model(fitted_model, machine, X_test)
```

### 2. Unified Prediction Interface (`unified_prediction.jl`)

Standardized prediction interface for all models:

```julia
# Create unified predictor
predictor = UnifiedPredictor(fitted_model, machine)

# Get predictions with metadata
result = predict_unified(predictor, X_test, return_proba=true)

# Access results
predictions = result.predictions
probabilities = result.probabilities  # For classification
confidence = result.confidence_scores

# Get metrics
metrics = get_prediction_metrics(result, y_true)
```

### 3. Cross-Validation System (`cross_validation.jl`)

Comprehensive cross-validation framework:

```julia
# Stratified K-Fold for classification
cv = StratifiedKFold(n_folds=5, shuffle=true, random_state=42)

# Quantile-based stratification for regression
cv_reg = QuantileStratifiedKFold(n_folds=5, n_quantiles=10)

# Run cross-validation
results = cross_validate_model(model, X, y, cv, 
                             task_type=:classification,
                             apply_smote=true)  # For imbalanced data

# Access results
fold_metrics = results["fold_metrics"]
aggregated = results["aggregated_metrics"]
```

### 4. Parallel Training System (`parallel_training.jl`)

High-performance parallel model training for evaluating many feature combinations:

```julia
# Create work items for different feature combinations
feature_combinations = [[1,2,3], [4,5,6], [7,8,9,10]]
model_specs = [
    (:xgboost, Dict(:max_depth => 5)),
    (:random_forest, Dict(:n_estimators => 100))
]

work_items = create_work_queue(feature_combinations, model_specs)

# Create parallel trainer
trainer = ParallelTrainer(work_items, 
                         n_threads=Threads.nthreads(),
                         memory_limit_mb=4096,
                         show_progress=true)

# Run parallel training
results = train_models_parallel(trainer, X, y, 
                              model_factory, cv_function)

# Aggregate results by model type
aggregated = collect_results(results)
```

Key features:
- **Thread Pool Management**: Efficient distribution across available CPU cores
- **Work Prioritization**: Process smaller feature sets first for quick feedback
- **Progress Monitoring**: Real-time ETA calculation and progress bars
- **Memory Efficiency**: Reusable feature buffers to minimize allocations
- **Graceful Interruption**: Save partial results on Ctrl+C
- **Load Balancing**: Dynamic work redistribution based on execution times

### 5. Model Configuration

Each model type has specific parameters:

#### XGBoost
- `max_depth`: Maximum tree depth (default: 6)
- `learning_rate`: Step size shrinkage (default: 0.3)
- `n_estimators`: Number of boosting rounds (default: 100)
- `subsample`: Subsample ratio of training instances
- `colsample_bytree`: Subsample ratio of columns

#### Random Forest
- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth (default: -1, unlimited)
- `min_samples_split`: Minimum samples to split node
- `max_features`: Number of features to consider

#### LightGBM
- `n_estimators`: Number of boosting iterations (default: 100)
- `learning_rate`: Boosting learning rate (default: 0.1)
- `num_leaves`: Number of leaves (default: 31)
- `min_child_samples`: Minimum samples in leaf

## Key Features

### 1. Model Factory Pattern
```julia
# Get default configuration
config = get_default_config(:xgboost, :classification)

# Update configuration
update_config!(config, max_depth=10, learning_rate=0.05)

# Validate configuration
validate_config(config)
```

### 2. Ensemble Predictions
```julia
# Create ensemble of models
predictors = [predictor1, predictor2, predictor3]

# Hard voting (majority vote)
hard_preds = ensemble_predict(predictors, X, voting=:hard)

# Soft voting (average probabilities)
soft_preds = ensemble_predict(predictors, X, voting=:soft)

# Weighted voting
weights = [0.5, 0.3, 0.2]
weighted_preds = ensemble_predict(predictors, X, voting=:soft, weights=weights)
```

### 3. Model Persistence
```julia
# Save fitted model
save_model(fitted_model, machine, "model.jld2")

# Load model
loaded_model, loaded_machine = load_model("model.jld2")
```

### 4. Prediction with Confidence
```julia
# Get predictions with confidence intervals
preds, lower, upper = predict_with_confidence(predictor, X, confidence_level=0.95)
```

## Usage Example

```julia
using .MLJInfrastructure
using .UnifiedPrediction

# Load your data
X_train, y_train = load_features_from_stage2()
X_test, y_test = load_test_data()

# Create and train multiple models
models = [
    create_model(:xgboost, :classification, max_depth=5),
    create_model(:random_forest, :classification, n_estimators=200),
    create_model(:lightgbm, :classification, num_leaves=50)
]

# Train all models
predictors = []
for model in models
    fitted, machine = fit_model!(model, X_train, y_train)
    push!(predictors, UnifiedPredictor(fitted, machine))
end

# Ensemble prediction
ensemble_preds = ensemble_predict(predictors, X_test, voting=:soft)

# Evaluate
accuracy = mean(ensemble_preds .== y_test)
println("Ensemble accuracy: $accuracy")
```

## Integration with Feature Selection Pipeline

Stage 3 is designed to work with the 50 features selected by Stage 2:

1. **Input**: 50 high-quality features from Stage 2 MCTS
2. **Process**: Train multiple models with cross-validation
3. **Analysis**: Extract feature importance and interactions
4. **Output**: Final 10-20 most important features

## Performance Considerations

- **Parallel Training**: Models can be trained in parallel using Julia's threading
- **Memory Usage**: LightGBM and XGBoost are memory-efficient for large datasets
- **Speed**: LightGBM is fastest, followed by XGBoost, then Random Forest
- **Accuracy**: Ensemble typically outperforms individual models

## Testing

Run the test suite:
```julia
include("test/stage3_evaluation/test_mlj_infrastructure.jl")
```

## Dependencies

- MLJ.jl: Machine learning framework
- MLJXGBoostInterface.jl: XGBoost integration
- MLJDecisionTreeInterface.jl: Random Forest implementation
- MLJLIGHTGBMInterface.jl: LightGBM integration
- MLJModelInterface.jl: Model interface definitions

### 5. Feature Importance Analysis (`feature_importance.jl`)

Comprehensive feature importance analysis using multiple methods:

```julia
# SHAP values calculation
shap_calc = SHAPCalculator(model_type=:xgboost, n_samples=500)
shap_result = calculate_shap_values(shap_calc, fitted_model, machine, X, y,
                                  feature_names=feature_names)

# Permutation importance
perm_calc = PermutationImportance(n_repeats=10, n_jobs=4)
perm_result = calculate_permutation_importance(perm_calc, fitted_model, machine, X, y)

# Combine multiple methods
combined_result = combine_importance_methods(shap_result, perm_result, 
                                           weights=(0.6, 0.4))

# Export results
export_importance_plot(combined_result, "importance_ranking.csv", top_k=20)
```

### 6. Feature Interactions Analysis (`feature_interactions.jl`)

Advanced pairwise feature interaction detection:

```julia
# Create interaction calculator
calc = InteractionCalculator(
    method=:all,  # Use all methods
    n_samples=300,
    categorical_features=[5, 6],
    n_jobs=4
)

# Calculate all pairwise interactions
interaction_matrix = calculate_all_interactions(
    calc, fitted_model, machine, X, y,
    feature_names=feature_names,
    threshold=0.05
)

# Get top interactions
top_interactions = get_significant_interactions(interaction_matrix, top_k=10)

# Export interaction heatmap
export_interaction_heatmap(interaction_matrix, "interactions.csv")
```

Key interaction detection methods:
- **H-statistic**: Measures variance explained by interactions
- **Mutual Information**: Captures linear and non-linear dependencies
- **Partial Dependence**: Visualizes interaction effects on predictions
- **Performance Degradation**: Reveals synergistic feature pairs

### 7. Final Feature Ranking and Selection (`feature_ranking.jl`)

Comprehensive feature ranking and selection system that combines multiple importance methods:

```julia
# Create feature ranker
ranking_config = RankingConfig(
    use_shap = true,
    use_permutation = true,
    use_interactions = true,
    shap_weight = 0.35,
    permutation_weight = 0.25,
    interaction_weight = 0.20,
    aggregation_method = :weighted_mean
)

ranker = FeatureRanker(n_features, ranking_config, feature_names=feature_names)

# Rank features using multiple importance methods
ranking_result = rank_features(ranker, importance_results, 
                             interaction_matrix=interaction_matrix)

# Pareto-optimal selection
selection_config = SelectionConfig(
    method = :pareto,
    constraint_config = ConstraintConfig(
        must_include = [1, 2],      # Domain knowledge
        must_exclude = [10, 15],    # Expensive features
        min_features = 10,
        max_features = 20,
        correlation_threshold = 0.85
    )
)

selection_result = select_features(ranker, ranking_result, X, y, 
                                 models=models, config=selection_config)

# Forward selection with early stopping
forward_config = SelectionConfig(
    method = :forward,
    early_stopping_rounds = 3,
    performance_metric = :accuracy
)

# Ensemble voting across multiple models
selected_features, votes = ensemble_feature_voting(
    [ranking1, ranking2, ranking3],
    voting_method = :borda,
    top_k = 15
)
```

Key features:
- **Multiple Aggregation Methods**: Weighted mean, rank aggregation, Borda count
- **Selection Algorithms**: Pareto-optimal, forward selection, ensemble voting
- **Constraint Handling**: Must-include/exclude features, correlation threshold
- **Stability Analysis**: Feature consistency across CV folds
- **Comprehensive Reporting**: Detailed selection reports with visualizations

## Future Enhancements

1. **Hyperparameter Optimization**: Automated tuning with MLJ's tuning interface
2. **Custom Metrics**: Support for domain-specific evaluation metrics
3. **GPU Acceleration**: XGBoost and LightGBM GPU support
4. **Distributed Training**: Multi-node training for large datasets
5. **Deep Learning Integration**: Support for neural network models