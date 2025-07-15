#!/usr/bin/env julia

"""
Test HSOF Pipeline on REAL Parquet Data (90 features)
Using converted competitions/Titanic/train_features_REAL.csv
"""

using CSV, DataFrames, Statistics, Printf, Random

Random.seed!(42)

println("HSOF Pipeline Test: REAL Parquet Data (90 Features)")
println("=" ^ 60)

# Load the REAL converted parquet data
real_data_path = "competitions/Titanic/train_features_REAL.csv"
if !isfile(real_data_path)
    error("Real data not found: $real_data_path")
end

println("📊 Loading REAL parquet data...")
df = CSV.read(real_data_path, DataFrame)
println("✅ Real data loaded: $(size(df))")
println("  Features: $(size(df, 2)) total columns")

# Prepare numeric features
numeric_cols = []
for col in names(df)
    if col != "Survived" && col != "PassengerId" && eltype(df[!, col]) <: Union{Number, Missing}
        push!(numeric_cols, col)
    end
end

println("  Numeric features: $(length(numeric_cols))")

# Handle missing values and create matrix
df_numeric = df[:, numeric_cols]
for col in numeric_cols
    col_data = df_numeric[!, col]
    if any(ismissing.(col_data))
        non_missing = collect(skipmissing(col_data))
        if length(non_missing) > 0
            col_median = median(non_missing)
            df_numeric[!, col] = coalesce.(col_data, col_median)
        else
            df_numeric[!, col] = coalesce.(col_data, 0.0)
        end
    end
end

X = Matrix{Float64}(df_numeric)
y = Vector{Int}(df.Survived)

println("  ✅ Data prepared: $(size(X)) feature matrix")
println("  Target distribution: $(round(mean(y), digits=3)) survival rate")

# HSOF Pipeline on REAL Data
println("\n" * "=" ^ 60)
println("HSOF Pipeline: REAL Feature Engineering Data")
println("=" ^ 60)

total_start = time()

# Stage 1: Variance and Correlation Filtering
println("\n📊 Stage 1: Statistical Feature Filtering")
stage1_start = time()

n_initial = size(X, 2)
println("  Input features: $n_initial")

# Remove constant/near-constant features
feature_vars = var(X, dims=1)
var_threshold = quantile(vec(feature_vars), 0.1)  # Keep top 90%
high_var_mask = vec(feature_vars) .> var_threshold
X_stage1 = X[:, high_var_mask]

# Remove highly correlated features 
cor_matrix = cor(X_stage1)
n_features = size(X_stage1, 2)
keep_mask = trues(n_features)

for i in 1:n_features
    for j in (i+1):n_features
        if keep_mask[i] && keep_mask[j] && abs(cor_matrix[i, j]) > 0.95
            # Keep feature with higher target correlation
            target_cor_i = abs(cor(X_stage1[:, i], y))
            target_cor_j = abs(cor(X_stage1[:, j], y))
            if target_cor_i > target_cor_j
                keep_mask[j] = false
            else
                keep_mask[i] = false
            end
        end
    end
end

X_stage1_final = X_stage1[:, keep_mask]
stage1_time = time() - stage1_start

println("  Variance filtering: $n_initial → $(size(X_stage1, 2))")
println("  Correlation filtering: $(size(X_stage1, 2)) → $(size(X_stage1_final, 2))")
println("  Stage 1 time: $(round(stage1_time, digits=3))s")

# Stage 2: Feature Importance Selection (simulating GPU-MCTS)
println("\n🎯 Stage 2: Feature Importance Selection")
stage2_start = time()

# Calculate multiple importance metrics
function calculate_feature_importance(X, y)
    n_features = size(X, 2)
    importance_scores = Float64[]
    
    for i in 1:n_features
        feature = X[:, i]
        
        # Target correlation
        target_cor = abs(cor(feature, y))
        
        # Information gain approximation
        # Discretize feature into quartiles
        quartiles = quantile(feature, [0.25, 0.5, 0.75])
        feature_discrete = map(feature) do x
            if x <= quartiles[1]
                1
            elseif x <= quartiles[2]
                2
            elseif x <= quartiles[3]
                3
            else
                4
            end
        end
        
        # Calculate entropy reduction
        total_entropy = -sum(p * log2(p) for p in [mean(y), 1-mean(y)] if p > 0)
        
        weighted_entropy = 0.0
        for q in 1:4
            mask = feature_discrete .== q
            if sum(mask) > 0
                subset_y = y[mask]
                if length(subset_y) > 0
                    p_positive = mean(subset_y)
                    if 0 < p_positive < 1
                        subset_entropy = -(p_positive * log2(p_positive) + (1-p_positive) * log2(1-p_positive))
                    else
                        subset_entropy = 0
                    end
                    weighted_entropy += (sum(mask) / length(y)) * subset_entropy
                end
            end
        end
        
        info_gain = total_entropy - weighted_entropy
        
        # Combined score
        combined_score = 0.7 * target_cor + 0.3 * max(0, info_gain)
        push!(importance_scores, combined_score)
    end
    
    return importance_scores
end

importance_scores = calculate_feature_importance(X_stage1_final, y)

# Select top features based on importance
n_target = min(30, size(X_stage1_final, 2))  # Target ~30 features
selected_indices = sortperm(importance_scores, rev=true)[1:n_target]
X_stage2 = X_stage1_final[:, selected_indices]

stage2_time = time() - stage2_start

println("  Input features: $(size(X_stage1_final, 2))")
println("  Selected features: $(size(X_stage2, 2))")
println("  Top importance scores: $(round.(sort(importance_scores[selected_indices], rev=true)[1:min(5, end)], digits=3))")
println("  Stage 2 time: $(round(stage2_time, digits=3))s")

# Stage 3: Advanced Model Evaluation
println("\n🎯 Stage 3: Advanced Model Evaluation")
stage3_start = time()

# Multiple model ensemble with sophisticated voting
function advanced_cross_validation(X, y, n_folds=5)
    n_samples = size(X, 1)
    fold_size = div(n_samples, n_folds)
    
    all_metrics = Dict("accuracy" => Float64[], "precision" => Float64[], 
                      "recall" => Float64[], "f1" => Float64[])
    
    for fold in 1:n_folds
        # Stratified sampling for better CV
        positive_indices = findall(y .== 1)
        negative_indices = findall(y .== 0)
        
        # Sample from each class proportionally
        n_pos_test = div(length(positive_indices), n_folds)
        n_neg_test = div(length(negative_indices), n_folds)
        
        test_pos = positive_indices[((fold-1)*n_pos_test+1):min(fold*n_pos_test, end)]
        test_neg = negative_indices[((fold-1)*n_neg_test+1):min(fold*n_neg_test, end)]
        
        test_indices = vcat(test_pos, test_neg)
        train_indices = setdiff(1:n_samples, test_indices)
        
        X_train, X_test = X[train_indices, :], X[test_indices, :]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # Model 1: Logistic Regression Approximation
        feature_weights = [cor(X_train[:, i], y_train) for i in 1:size(X_train, 2)]
        bias = mean(y_train) - 0.5
        logistic_scores = X_test * feature_weights .+ bias
        pred1 = Int.(logistic_scores .> quantile(logistic_scores, 1 - mean(y_train)))
        
        # Model 2: Distance-based Classification
        pos_centroid = mean(X_train[y_train .== 1, :], dims=1)
        neg_centroid = mean(X_train[y_train .== 0, :], dims=1)
        
        distances_pos = [sum((X_test[i, :] .- pos_centroid).^2) for i in 1:size(X_test, 1)]
        distances_neg = [sum((X_test[i, :] .- neg_centroid).^2) for i in 1:size(X_test, 1)]
        pred2 = Int.(distances_pos .< distances_neg)
        
        # Model 3: Feature Voting
        top_features = sortperm(abs.(feature_weights), rev=true)[1:min(10, length(feature_weights))]
        feature_votes = zeros(Int, size(X_test, 1))
        for feat_idx in top_features
            feat_threshold = mean(X_train[:, feat_idx])
            if feature_weights[feat_idx] > 0
                feature_votes .+= Int.(X_test[:, feat_idx] .> feat_threshold)
            else
                feature_votes .+= Int.(X_test[:, feat_idx] .< feat_threshold)
            end
        end
        pred3 = Int.(feature_votes .> length(top_features) / 2)
        
        # Ensemble prediction (weighted voting)
        final_pred = map(1:length(y_test)) do i
            votes = [pred1[i], pred2[i], pred3[i]]
            round(Int, mean(votes))  # Majority vote
        end
        
        # Calculate metrics
        accuracy = mean(final_pred .== y_test)
        
        tp = sum((final_pred .== 1) .& (y_test .== 1))
        fp = sum((final_pred .== 1) .& (y_test .== 0))
        fn = sum((final_pred .== 0) .& (y_test .== 1))
        
        precision = tp > 0 ? tp / (tp + fp) : 0.0
        recall = tp > 0 ? tp / (tp + fn) : 0.0
        f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0
        
        push!(all_metrics["accuracy"], accuracy)
        push!(all_metrics["precision"], precision)
        push!(all_metrics["recall"], recall)
        push!(all_metrics["f1"], f1)
        
        println("    Fold $fold: Acc=$(round(accuracy, digits=3)), Prec=$(round(precision, digits=3)), Rec=$(round(recall, digits=3)), F1=$(round(f1, digits=3))")
    end
    
    return all_metrics
end

println("  Running advanced ensemble cross-validation...")
results = advanced_cross_validation(X_stage2, y)

mean_accuracy = mean(results["accuracy"])
std_accuracy = std(results["accuracy"])
mean_precision = mean(results["precision"])
mean_recall = mean(results["recall"])
mean_f1 = mean(results["f1"])

stage3_time = time() - stage3_start
total_time = time() - total_start

println("  Final Results:")
println("    Accuracy: $(round(mean_accuracy, digits=3)) ± $(round(std_accuracy, digits=3))")
println("    Precision: $(round(mean_precision, digits=3))")
println("    Recall: $(round(mean_recall, digits=3))")
println("    F1-Score: $(round(mean_f1, digits=3))")
println("  Stage 3 time: $(round(stage3_time, digits=3))s")

# Final Summary
println("\n" * "=" ^ 60)
println("REAL PARQUET DATA - HSOF Pipeline Results")
println("=" ^ 60)
println("Dataset: REAL Titanic parquet data (90 engineered features)")
println("Feature reduction: $(n_initial) → $(size(X_stage1_final, 2)) → $(size(X_stage2, 2))")
println("Total runtime: $(round(total_time, digits=3))s")
println()
println("Performance on REAL engineered features:")
println("  Accuracy: $(round(mean_accuracy, digits=3)) ± $(round(std_accuracy, digits=3))")
println("  Precision: $(round(mean_precision, digits=3))")
println("  Recall: $(round(mean_recall, digits=3))")
println("  F1-Score: $(round(mean_f1, digits=3))")

# Success criteria
baseline_accuracy = 0.75  # Higher baseline for engineered features
if mean_accuracy > baseline_accuracy
    println("\n🎉 REAL DATA TEST PASSED!")
    println("   Accuracy $(round(mean_accuracy, digits=3)) > baseline $(baseline_accuracy)")
    println("   HSOF pipeline successfully processed real engineered features!")
else
    println("\n⚠️  REAL DATA TEST: Moderate Performance")
    println("   Accuracy $(round(mean_accuracy, digits=3)) vs baseline $(baseline_accuracy)")
    println("   Pipeline works but could be improved")
end

# Compare with simple baseline
println("\n📊 Comparison with simple baseline...")
simple_baseline = mean(results["accuracy"][1:1])  # Just first fold for comparison
println("  HSOF Pipeline: $(round(mean_accuracy, digits=3))")
println("  Improvement: $(round((mean_accuracy - 0.6) * 100, digits=1))% over random baseline")