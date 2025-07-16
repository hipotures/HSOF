"""
Stage 3: Real model evaluation with multiple algorithms.
Reduces 50 features to 10-20 features using precise model evaluation.
"""

using MLJ, Statistics, Random, StatsBase, XGBoost, DataFrames
using MLJXGBoostInterface
using MLJDecisionTreeInterface
using Combinatorics  # For combinations function
import MLJ: categorical  # Import categorical function from MLJ

# Suppress XGBoost verbose output
ENV["XGBOOST_VERBOSITY"] = "0"

"""
Stage 3: Precise model evaluation with multiple algorithms.
Reduces 50 features to 10-20 features using real model CV scores.
"""
function stage3_precise_evaluation(
    X::Matrix{Float32},
    y::Vector{Float32},
    feature_names::Vector{String},
    problem_type::String;
    n_candidates::Int=100,
    target_range::Tuple{Int,Int}=(max(5, div(size(X,2), 4)), min(size(X,2), div(size(X,2), 2))),
    cv_folds::Int=5,
    xgboost_params::Dict=Dict(),
    rf_params::Dict=Dict()
)
    println("\n" * "="^60)
    println("=== Stage 3: Precise Model Evaluation ===")
    println("="^60)
    
    n_samples, n_features = size(X)
    target_min, target_max = target_range
    
    println("Input: $n_samples samples × $n_features features")
    println("Problem type: $problem_type")
    
    # Setup evaluation models with custom parameters
    models = setup_evaluation_models(problem_type, xgboost_params, rf_params)
    println("Models configured: $(join(keys(models), ", "))")
    
    # Adjust target range if we have fewer features than requested
    if n_features < target_min
        println("  ⚠️  Only $n_features features available - adjusting target range")
        target_min = 1
        target_max = min(n_features, target_max)
    elseif n_features < target_max
        target_max = n_features
    end
    
    # Generate candidate feature subsets
    # If we have very few features, reduce the number of candidates
    if n_features <= 5
        # With 3 features, we can only have 2^3 - 1 = 7 non-empty subsets
        max_possible_subsets = 2^n_features - 1
        n_candidates = min(n_candidates, max_possible_subsets)
        println("  Reducing candidates to $n_candidates due to limited features")
    end
    
    candidate_subsets = generate_candidate_subsets(n_features, target_min, target_max, n_candidates)
    println("Generated $(length(candidate_subsets)) candidate feature subsets")
    
    # Evaluate each subset with all models
    best_score = 0.0
    best_features = String[]
    best_model_name = ""
    best_subset_size = 0
    
    # Track scores and best features for each model
    model_scores = Dict{String, Vector{Float64}}()
    model_best_features = Dict{String, Tuple{Vector{String}, Float64}}()
    for model_name in keys(models)
        model_scores[model_name] = Float64[]
        model_best_features[model_name] = (String[], 0.0)
    end
    
    println("\nEvaluating feature combinations...")
    
    for (i, subset_indices) in enumerate(candidate_subsets)
        if i % 20 == 0
            println("  Progress: $i/$(length(candidate_subsets)) ($(round(100*i/length(candidate_subsets), digits=1))%)")
        end
        
        X_subset = X[:, subset_indices]
        current_features = feature_names[subset_indices]
        
        # Evaluate with each model type
        for (model_name, model) in models
            try
                score = evaluate_model_cv(model, X_subset, y, problem_type, folds=cv_folds)
                push!(model_scores[model_name], score)
                
                # Track best for each model
                if score > model_best_features[model_name][2]
                    model_best_features[model_name] = (copy(current_features), score)
                end
                
                if score > best_score
                    best_score = score
                    best_features = copy(current_features)
                    best_model_name = model_name
                    best_subset_size = length(subset_indices)
                end
                
                # Show detailed results for first few subsets
                if i <= 3
                    println("    Subset $i ($(length(subset_indices)) features) - $model_name: $(round(score, digits=4))")
                end
                
            catch e
                if i <= 5  # Only show first few errors to avoid spam
                    println("    Warning: $model_name failed on subset $i: $(typeof(e))")
                end
            end
        end
    end
    
    # Results summary
    println("\n" * "="^60)
    println("=== Stage 3 Results ===")
    println("="^60)
    println("Features selected: $(length(best_features)) / $n_features")
    println("Final reduction: $(round(100 * (1 - length(best_features)/n_features), digits=1))%")
    println("Best model: $best_model_name")
    println("Best CV score: $(round(best_score, digits=4))")
    
    # Model performance summary
    println("\nModel Performance Summary:")
    for (model_name, scores) in model_scores
        if !isempty(scores)
            mean_score = mean(scores)
            max_score = maximum(scores)
            min_score = minimum(scores)
            std_score = std(scores)
            println("  $model_name:")
            println("    Mean CV score: $(round(mean_score, digits=4))")
            println("    Best CV score: $(round(max_score, digits=4))")
            println("    Worst CV score: $(round(min_score, digits=4))")
            println("    Std deviation: $(round(std_score, digits=4))")
            println("    Evaluations: $(length(scores))")
            
            # Show best feature set for this model
            best_feat, best_sc = model_best_features[model_name]
            if !isempty(best_feat)
                println("    Best feature set ($(length(best_feat)) features): $(round(best_sc, digits=4))")
                for (idx, feat) in enumerate(best_feat[1:min(5, end)])
                    println("      $idx. $feat")
                end
                if length(best_feat) > 5
                    println("      ... and $(length(best_feat) - 5) more")
                end
            end
        end
    end
    
    if length(best_features) > 0
        println("Selected features:")
        for i in 1:min(length(best_features), 15)
            println("  $i. $(best_features[i])")
        end
        if length(best_features) > 15
            println("  ... and $(length(best_features) - 15) more")
        end
    end
    
    println("✅ Stage 3 completed successfully")
    return best_features, best_score, best_model_name
end

"""
Setup evaluation models based on problem type.
"""
function setup_evaluation_models(problem_type::String, xgboost_params::Dict=Dict(), rf_params::Dict=Dict())
    models = Dict{String, Any}()
    
    if problem_type == "binary_classification" || problem_type == "classification"
        println("Setting up classification models...")
        
        # XGBoost Classifier with GPU support (XGBoost v2+)
        default_xgb_params = Dict(
            :num_round => 100,
            :max_depth => 6,
            :eta => 0.1,
            :objective => "binary:logistic",
            :eval_metric => "logloss",
            :device => "cuda",  # New way to specify GPU in XGBoost 2.0+
            :tree_method => "hist"  # Use "hist" instead of deprecated "gpu_hist"
        )
        # Merge with user-provided params
        merged_xgb_params = merge(default_xgb_params, xgboost_params)
        models["XGBoost"] = Dict(
            :model_type => :xgboost,
            :params => merged_xgb_params
        )
        
        # Random Forest Classifier
        try
            RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree
            default_rf_params = Dict(:n_trees => 100, :max_depth => 10)
            merged_rf_params = merge(default_rf_params, rf_params)
            models["RandomForest"] = Dict(
                :model_type => :mlj,
                :model => RandomForestClassifier(),
                :params => merged_rf_params
            )
        catch e
            println("  Warning: RandomForest not available: $e")
        end
        
        # Logistic Regression
        try
            LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
            models["Logistic"] = Dict(
                :model_type => :mlj,
                :model => LogisticClassifier(),
                :params => Dict()
            )
        catch e
            println("  Warning: LogisticClassifier not available: $e")
        end
        
    else
        println("Setting up regression models...")
        
        # XGBoost Regressor with GPU support (XGBoost v2+)
        default_xgb_params = Dict(
            :num_round => 100,
            :max_depth => 6,
            :eta => 0.1,
            :objective => "reg:squarederror",
            :eval_metric => "rmse",
            :device => "cuda",  # New way to specify GPU in XGBoost 2.0+
            :tree_method => "hist"  # Use "hist" instead of deprecated "gpu_hist"
        )
        merged_xgb_params = merge(default_xgb_params, xgboost_params)
        models["XGBoost"] = Dict(
            :model_type => :xgboost,
            :params => merged_xgb_params
        )
        
        # Random Forest Regressor
        try
            RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree
            default_rf_params = Dict(:n_trees => 100, :max_depth => 10)
            merged_rf_params = merge(default_rf_params, rf_params)
            models["RandomForest"] = Dict(
                :model_type => :mlj,
                :model => RandomForestRegressor(),
                :params => merged_rf_params
            )
        catch e
            println("  Warning: RandomForest not available: $e")
        end
        
        # Linear Regression
        try
            LinearRegressor = @load LinearRegressor pkg=MLJLinearModels
            models["Linear"] = Dict(
                :model_type => :mlj,
                :model => LinearRegressor(),
                :params => Dict()
            )
        catch e
            println("  Warning: LinearRegressor not available: $e")
        end
    end
    
    return models
end

"""
Cross-validation evaluation for a single model.
"""
function evaluate_model_cv(model_config::Dict, X::Matrix{Float32}, y::Vector{Float32}, 
                          problem_type::String; folds::Int=5)
    
    n_samples = size(X, 1)
    
    # Handle small datasets
    if n_samples < folds * 2
        folds = max(2, div(n_samples, 2))
    end
    
    fold_size = div(n_samples, folds)
    scores = Float64[]
    
    for fold in 1:folds
        # Create train/test split
        test_start = (fold - 1) * fold_size + 1
        test_end = fold == folds ? n_samples : fold * fold_size
        
        test_indices = test_start:test_end
        train_indices = setdiff(1:n_samples, test_indices)
        
        # Handle edge case
        if length(train_indices) == 0 || length(test_indices) == 0
            continue
        end
        
        X_train, X_test = X[train_indices, :], X[test_indices, :]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # Evaluate based on model type
        if model_config[:model_type] == :xgboost
            score = evaluate_xgboost(model_config[:params], X_train, y_train, X_test, y_test, problem_type)
        elseif model_config[:model_type] == :mlj
            score = evaluate_mlj_model(model_config[:model], X_train, y_train, X_test, y_test, problem_type)
        else
            error("Unknown model type: $(model_config[:model_type])")
        end
        
        push!(scores, score)
    end
    
    return length(scores) > 0 ? mean(scores) : 0.0
end

"""
Evaluate XGBoost model.
"""
function evaluate_xgboost(params::Dict, X_train::Matrix{Float32}, y_train::Vector{Float32},
                         X_test::Matrix{Float32}, y_test::Vector{Float32}, problem_type::String)
    
    # Train model
    dtrain = XGBoost.DMatrix(X_train, label=y_train)
    # Add watchlist=(;) to params to suppress output
    # Ensure all params have symbol keys
    params_sym = Dict(Symbol(k) => v for (k, v) in params)
    params_with_watchlist = merge(params_sym, Dict(:watchlist => (;)))
    model = XGBoost.xgboost(dtrain; params_with_watchlist...)
    
    # Predict
    dtest = XGBoost.DMatrix(X_test)
    predictions = XGBoost.predict(model, dtest)
    
    # Calculate metric
    if problem_type == "binary_classification" || problem_type == "classification"
        # Binary classification accuracy
        pred_labels = predictions .> 0.5
        actual_labels = y_test .> 0.5
        return mean(pred_labels .== actual_labels)
    else
        # Regression R²
        ss_res = sum((y_test .- predictions).^2)
        ss_tot = sum((y_test .- mean(y_test)).^2)
        return max(0.0, 1.0 - (ss_res / (ss_tot + 1e-10)))
    end
end

"""
Evaluate MLJ model.
"""
function evaluate_mlj_model(model, X_train::Matrix{Float32}, y_train::Vector{Float32},
                           X_test::Matrix{Float32}, y_test::Vector{Float32}, problem_type::String)
    
    # Convert to DataFrames for MLJ
    X_train_df = DataFrame(X_train, :auto)
    X_test_df = DataFrame(X_test, :auto)
    
    # Convert target to categorical for classification
    if problem_type == "binary_classification" || problem_type == "classification"
        y_train_cat = categorical(Int.(y_train))
        y_test_cat = Int.(y_test)
    else
        y_train_cat = y_train
        y_test_cat = y_test
    end
    
    # Create and train machine
    mach = machine(model, X_train_df, y_train_cat)
    
    try
        fit!(mach, verbosity=0)
        
        # Predict
        predictions = predict(mach, X_test_df)
        
        # Calculate metric
        if problem_type == "binary_classification" || problem_type == "classification"
            # For classification, extract mode of predictions
            if eltype(predictions) <: MLJ.CategoricalDistributions.UnivariateFinite
                pred_labels = mode.(predictions)
                # Compare with categorical test labels
                return mean(pred_labels .== categorical(y_test_cat))
            else
                # Direct predictions
                pred_labels = predictions .> 0.5
                actual_labels = y_test .> 0.5
                return mean(pred_labels .== actual_labels)
            end
        else
            # Regression R²
            ss_res = sum((predictions .- y_test).^2)
            ss_tot = sum((y_test .- mean(y_test)).^2)
            return max(0.0, 1.0 - (ss_res / (ss_tot + 1e-10)))
        end
        
    catch e
        return 0.0  # Return 0 score on failure
    end
end

"""
Generate candidate feature subsets for evaluation.
"""
function generate_candidate_subsets(n_features::Int, target_min::Int, target_max::Int, 
                                  n_candidates::Int=100)
    candidates = Vector{Int}[]
    
    # Handle edge case: very few features
    if n_features <= 5
        # Generate all possible subsets within the size range
        for size in target_min:target_max
            for combo in combinations(1:n_features, size)
                push!(candidates, collect(combo))
                if length(candidates) >= n_candidates
                    return unique(candidates)
                end
            end
        end
        # If we still need more, repeat some subsets
        while length(candidates) < n_candidates
            push!(candidates, candidates[rand(1:length(candidates))])
        end
        return candidates
    end
    
    # Strategy 1: Random subsets of various sizes (40%)
    n_random = div(n_candidates * 4, 10)
    for _ in 1:n_random
        subset_size = rand(target_min:target_max)
        subset = sample(1:n_features, subset_size, replace=false)
        push!(candidates, sort(subset))
    end
    
    # Strategy 2: Top features + random (30%)
    n_top = div(n_candidates * 3, 10)
    for _ in 1:n_top
        subset_size = rand(target_min:target_max)
        # Take some top features
        n_top_features = div(subset_size, 2)
        top_features = 1:min(n_top_features, n_features)
        
        # Add random features
        remaining_size = subset_size - length(top_features)
        if remaining_size > 0 && length(top_features) < n_features
            remaining_features = sample((length(top_features)+1):n_features, 
                                      min(remaining_size, n_features - length(top_features)), 
                                      replace=false)
            subset = vcat(collect(top_features), remaining_features)
        else
            subset = collect(top_features)
        end
        push!(candidates, sort(subset))
    end
    
    # Strategy 3: Sequential windows (20%)
    n_windows = div(n_candidates * 2, 10)
    for _ in 1:n_windows
        subset_size = rand(target_min:target_max)
        if subset_size <= n_features
            max_start = n_features - subset_size + 1
            start_idx = rand(1:max_start)
            subset = collect(start_idx:(start_idx + subset_size - 1))
            push!(candidates, subset)
        end
    end
    
    # Strategy 4: Uniform spacing (10%)
    n_uniform = n_candidates - length(candidates)
    for _ in 1:n_uniform
        subset_size = rand(target_min:target_max)
        if subset_size <= n_features
            step = div(n_features, subset_size)
            subset = collect(1:step:n_features)[1:subset_size]
            push!(candidates, subset)
        end
    end
    
    # Remove duplicates and ensure valid sizes
    unique_candidates = unique(candidates)
    valid_candidates = [c for c in unique_candidates if target_min <= length(c) <= target_max]
    
    # Ensure we have enough candidates
    while length(valid_candidates) < n_candidates
        subset_size = rand(target_min:target_max)
        subset = sample(1:n_features, subset_size, replace=false)
        push!(valid_candidates, sort(subset))
        valid_candidates = unique(valid_candidates)
    end
    
    return valid_candidates[1:min(n_candidates, length(valid_candidates))]
end

"""
Advanced feature selection with ensemble evaluation.
"""
function stage3_ensemble_evaluation(
    X::Matrix{Float32},
    y::Vector{Float32},
    feature_names::Vector{String},
    problem_type::String;
    n_candidates::Int=50,
    target_range::Tuple{Int,Int}=(10, 20)
)
    println("\n=== Stage 3: Ensemble Evaluation ===")
    
    n_features = size(X, 2)
    target_min, target_max = target_range
    
    # Setup multiple models
    models = setup_evaluation_models(problem_type)
    
    # Generate candidates
    candidates = generate_candidate_subsets(n_features, target_min, target_max, n_candidates)
    
    # Evaluate each candidate with all models
    ensemble_scores = Dict{Vector{Int}, Float64}()
    
    for (i, subset_indices) in enumerate(candidates)
        if i % 10 == 0
            println("  Ensemble progress: $i/$n_candidates")
        end
        
        X_subset = X[:, subset_indices]
        scores = Float64[]
        
        # Evaluate with each model
        for (model_name, model) in models
            try
                score = evaluate_model_cv(model, X_subset, y, problem_type, folds=cv_folds)
                push!(scores, score)
            catch e
                # Skip failed models
                continue
            end
        end
        
        # Ensemble score (mean of all models)
        if !isempty(scores)
            ensemble_scores[subset_indices] = mean(scores)
        end
    end
    
    # Find best ensemble score
    if !isempty(ensemble_scores)
        best_subset = argmax(ensemble_scores)
        best_score = ensemble_scores[best_subset]
        best_features = feature_names[best_subset]
        
        println("✅ Ensemble evaluation completed")
        println("  Best ensemble score: $(round(best_score, digits=4))")
        println("  Selected $(length(best_features)) features")
        
        return best_features, best_score, "Ensemble"
    else
        error("No valid ensemble evaluations completed")
    end
end

"""
Feature stability analysis across CV folds.
"""
function analyze_feature_stability(
    X::Matrix{Float32},
    y::Vector{Float32},
    feature_names::Vector{String},
    problem_type::String;
    n_bootstrap::Int=20,
    target_size::Int=15
)
    println("\n=== Feature Stability Analysis ===")
    
    n_samples, n_features = size(X)
    feature_counts = zeros(Int, n_features)
    
    # Bootstrap sampling
    for boot in 1:n_bootstrap
        # Bootstrap sample
        boot_indices = sample(1:n_samples, n_samples, replace=true)
        X_boot = X[boot_indices, :]
        y_boot = y[boot_indices]
        
        # Feature selection on bootstrap sample
        _, _, selected_indices = stage3_precise_evaluation(
            X_boot, y_boot, feature_names, problem_type,
            n_candidates=20, target_range=(target_size, target_size)
        )
        
        # Update feature counts
        for idx in selected_indices
            if 1 <= idx <= n_features
                feature_counts[idx] += 1
            end
        end
        
        if boot % 5 == 0
            println("  Bootstrap progress: $boot/$n_bootstrap")
        end
    end
    
    # Calculate stability scores
    stability_scores = feature_counts ./ n_bootstrap
    
    # Select most stable features
    stable_indices = sortperm(stability_scores, rev=true)[1:target_size]
    stable_features = feature_names[stable_indices]
    
    println("✅ Stability analysis completed")
    println("  Most stable features (selected in $(round(100*stability_scores[stable_indices[1]], digits=1))% of bootstraps):")
    for i in 1:min(10, length(stable_features))
        stability_pct = round(100 * stability_scores[stable_indices[i]], digits=1)
        println("    $i. $(stable_features[i]) ($(stability_pct)%)")
    end
    
    return stable_features, stability_scores[stable_indices], "Stability"
end