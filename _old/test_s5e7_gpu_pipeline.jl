#!/usr/bin/env julia

"""
HSOF GPU Pipeline Test for Playground Series S5E7 Dataset
Real GPU-accelerated feature selection on Kaggle competition data
"""

using CSV, DataFrames, Statistics, Printf, Random, CUDA

Random.seed!(42)

println("HSOF GPU Pipeline Test: Playground Series S5E7")
println("=" ^ 60)

# Check GPU
if CUDA.functional()
    println("✅ CUDA functional: $(length(CUDA.devices())) GPU")
    println("   GPU memory: $(CUDA.memory_status())")
    gpu_available = true
else
    println("❌ CUDA not functional")
    gpu_available = false
end

# Load S5E7 data
data_path = "competitions/playground-series-s5e7/export/playground_s5e7_train_features.csv"
if !isfile(data_path)
    error("S5E7 data not found: $data_path - run conversion script first")
end

println("\n📊 Loading S5E7 Competition Data...")
df = CSV.read(data_path, DataFrame)
println("✅ S5E7 data loaded: $(size(df))")
println("  Total columns: $(size(df, 2))")

# Identify target column
global target_col = nothing
possible_targets = ["Personality", "target", "label", "y", "class", "Target", "Label"]
for col in possible_targets
    if col in names(df)
        global target_col = col
        println("  Found target column: $col")
        break
    end
end

if target_col === nothing
    # Look for categorical column (string or numeric with few unique values)
    for col in names(df)
        unique_vals = unique(skipmissing(df[!, col]))
        if length(unique_vals) <= 10  # Likely categorical
            global target_col = col
            println("  Detected likely target column: $col")
            break
        end
    end
end

if target_col === nothing
    error("No target column found in S5E7 data. Columns: $(names(df))")
end

println("  Target column: $target_col")

# Process features
feature_cols = [col for col in names(df) if col != target_col && col != "id" && eltype(df[!, col]) <: Union{Number, Missing}]
println("  Numeric feature columns: $(length(feature_cols))")

if length(feature_cols) == 0
    error("No numeric feature columns found!")
end

# Clean data
df_clean = copy(df[:, vcat([target_col], feature_cols)])
for col in feature_cols
    col_data = df_clean[!, col]
    if any(ismissing.(col_data))
        non_missing = collect(skipmissing(col_data))
        if length(non_missing) > 0
            df_clean[!, col] = coalesce.(col_data, median(non_missing))
        else
            df_clean[!, col] = coalesce.(col_data, 0.0)
        end
    end
end

# Extract features and target
X = Matrix{Float64}(df_clean[:, feature_cols])
y_raw = df_clean[!, target_col]

# Convert target to binary if needed
unique_targets = unique(skipmissing(y_raw))
if length(unique_targets) == 2
    # Binary classification
    target_map = Dict(unique_targets[1] => 0, unique_targets[2] => 1)
    y = [target_map[val] for val in y_raw]
    task_type = "binary_classification"
elseif length(unique_targets) <= 10
    # Multi-class classification  
    target_map = Dict(val => i-1 for (i, val) in enumerate(unique_targets))
    y = [target_map[val] for val in y_raw]
    task_type = "multiclass_classification"
else
    # Regression - convert to binary based on median
    median_val = median(skipmissing(y_raw))
    y = [val > median_val ? 1 : 0 for val in y_raw]
    task_type = "regression_to_binary"
end

println("  Data prepared: $(size(X)) feature matrix")
println("  Task type: $task_type")
println("  Target distribution: $(round(mean(y), digits=3))")
println("  Unique targets: $(length(unique_targets))")

# Initialize global variables
global importance_scores, n_select, selected_indices, X_selected, gpu_used

# GPU-Accelerated Feature Selection
if gpu_available
    println("\n🚀 GPU-Accelerated Feature Selection...")
    
    try
        # Simple CUDA kernel for feature correlation (exact copy from working Titanic)
        function gpu_correlation_kernel(X_gpu, y_gpu, correlations)
            idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            if idx <= size(X_gpu, 2)
                # Calculate correlation for feature idx
                n = size(X_gpu, 1)
                
                # Mean calculation
                x_mean = 0.0f0
                y_mean = 0.0f0
                for i in 1:n
                    x_mean += X_gpu[i, idx]
                    y_mean += y_gpu[i]
                end
                x_mean /= n
                y_mean /= n
                
                # Correlation calculation
                numerator = 0.0f0
                x_var = 0.0f0
                y_var = 0.0f0
                
                for i in 1:n
                    x_diff = X_gpu[i, idx] - x_mean
                    y_diff = y_gpu[i] - y_mean
                    numerator += x_diff * y_diff
                    x_var += x_diff * x_diff
                    y_var += y_diff * y_diff
                end
                
                correlation = numerator / sqrt(x_var * y_var)
                correlations[idx] = abs(correlation)
            end
            return nothing
        end
        
        # Transfer data to GPU
        println("  Transferring data to GPU...")
        X_gpu = CuArray{Float32}(X)
        y_gpu = CuArray{Float32}(y)
        correlations_gpu = CUDA.zeros(Float32, size(X, 2))
        
        println("  GPU arrays created: X=$(size(X_gpu)), y=$(size(y_gpu))")
        
        # Launch GPU kernel
        n_features = size(X, 2)
        n_samples = size(X, 1)
        threads_per_block = 256
        blocks = div(n_features + threads_per_block - 1, threads_per_block)
        
        println("  Launching GPU kernel: $blocks blocks, $threads_per_block threads")
        println("  Processing $n_features features on $n_samples samples")
        
        # Time the GPU execution
        gpu_start = time()
        CUDA.@cuda threads=threads_per_block blocks=blocks gpu_correlation_kernel(
            X_gpu, y_gpu, correlations_gpu
        )
        CUDA.synchronize()
        gpu_time = time() - gpu_start
        
        # Get results from GPU
        global importance_scores = Array(correlations_gpu)
        
        println("  ✅ GPU kernel completed in $(round(gpu_time, digits=3))s")
        println("  GPU correlations (top 5): $(round.(sort(importance_scores, rev=true)[1:5], digits=3))")
        
        # Feature selection based on GPU scores
        global n_select = min(50, div(n_features, 2))  # Select top 50 or half of features
        global selected_indices = sortperm(importance_scores, rev=true)[1:n_select]
        global X_selected = X[:, selected_indices]
        
        println("  GPU selected $n_select features from $n_features")
        
        global gpu_used = true
        
    catch e
        println("  ❌ GPU feature selection failed: $e")
        error("GPU kernel must work - no fallback allowed!")
    end
else
    error("CUDA not available - GPU required for this test!")
end

# Advanced Cross-Validation with Ensemble
println("\n🎯 Advanced Ensemble Cross-Validation...")

function ensemble_cross_validation(X, y, n_folds=5)
    n_samples = size(X, 1)
    
    # Stratified sampling for better CV
    unique_classes = unique(y)
    stratified_indices = []
    
    for class in unique_classes
        class_indices = findall(y .== class)
        shuffle!(class_indices)
        push!(stratified_indices, class_indices)
    end
    
    all_metrics = Dict("accuracy" => Float64[], "precision" => Float64[], 
                      "recall" => Float64[], "f1" => Float64[])
    
    for fold in 1:n_folds
        # Stratified train/test split
        test_indices = Int[]
        for class_indices in stratified_indices
            fold_size = div(length(class_indices), n_folds)
            start_idx = (fold - 1) * fold_size + 1
            end_idx = min(fold * fold_size, length(class_indices))
            if fold == n_folds
                end_idx = length(class_indices)  # Include remaining samples in last fold
            end
            append!(test_indices, class_indices[start_idx:end_idx])
        end
        
        train_indices = setdiff(1:n_samples, test_indices)
        
        X_train, X_test = X[train_indices, :], X[test_indices, :]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # Ensemble of 3 models
        predictions_ensemble = []
        
        # Model 1: Correlation-weighted linear model
        feature_weights = [cor(X_train[:, i], y_train) for i in 1:size(X_train, 2)]
        bias = mean(y_train) - 0.5
        scores1 = X_test * feature_weights .+ bias
        
        # Dynamic threshold based on class balance
        threshold1 = quantile(scores1, 1 - mean(y_train))
        pred1 = Int.(scores1 .> threshold1)
        push!(predictions_ensemble, pred1)
        
        # Model 2: Distance-based classification
        class_centroids = []
        for class in unique_classes
            class_mask = y_train .== class
            if sum(class_mask) > 0
                centroid = mean(X_train[class_mask, :], dims=1)
                push!(class_centroids, (class, centroid))
            end
        end
        
        pred2 = map(1:length(y_test)) do i
            test_point = X_test[i, :]'
            distances = [sum((test_point .- centroid).^2) for (class, centroid) in class_centroids]
            min_idx = argmin(distances)
            class_centroids[min_idx][1]
        end
        push!(predictions_ensemble, pred2)
        
        # Model 3: Variance-weighted voting
        feature_variances = [var(X_train[:, i]) for i in 1:size(X_train, 2)]
        variance_weights = feature_variances ./ sum(feature_variances)
        
        scores3 = sum(X_test .* variance_weights', dims=2)
        threshold3 = median(scores3)
        pred3 = Int.(scores3 .> threshold3)
        push!(predictions_ensemble, pred3)
        
        # Ensemble prediction (majority voting)
        final_pred = map(1:length(y_test)) do i
            votes = [pred[i] for pred in predictions_ensemble]
            round(Int, mean(votes))
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

cv_start = time()
results = ensemble_cross_validation(X_selected, y)
cv_time = time() - cv_start

mean_accuracy = mean(results["accuracy"])
std_accuracy = std(results["accuracy"])
mean_precision = mean(results["precision"])
mean_recall = mean(results["recall"])
mean_f1 = mean(results["f1"])

# Final Results
println("\n" * "=" ^ 60)
println("S5E7 HSOF GPU Pipeline Results")
println("=" ^ 60)
println("Competition: Playground Series S5E7")
println("Dataset: $(size(df)) → $(size(X_selected)) features selected")
println("Task: $task_type")
println("GPU acceleration: $(gpu_used ? "✅ Used" : "❌ CPU fallback")")
println("Cross-validation time: $(round(cv_time, digits=3))s")
println()
println("Performance Metrics:")
println("  Accuracy: $(round(mean_accuracy, digits=3)) ± $(round(std_accuracy, digits=3))")
println("  Precision: $(round(mean_precision, digits=3))")
println("  Recall: $(round(mean_recall, digits=3))")
println("  F1-Score: $(round(mean_f1, digits=3))")

# Performance assessment
baseline_accuracy = 0.70  # Competition baseline
if mean_accuracy > baseline_accuracy
    println("\n🎉 S5E7 HSOF PIPELINE TEST PASSED!")
    println("   Accuracy $(round(mean_accuracy, digits=3)) > baseline $(baseline_accuracy)")
    if gpu_used
        println("   ✅ GPU acceleration successful")
    end
    println("   ✅ Ready for Kaggle competition!")
    exit(0)
else
    println("\n⚠️  S5E7 HSOF PIPELINE: Moderate Performance")
    println("   Accuracy $(round(mean_accuracy, digits=3)) vs baseline $(baseline_accuracy)")
    println("   Pipeline works but could be improved")
    exit(1)
end