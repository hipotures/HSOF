#!/usr/bin/env julia

"""
MINIMAL Real HSOF Test - bez problematycznych modułów
Używa tylko Database i UI, omija GPU.jl
"""

using CSV, DataFrames, Statistics, Printf, Random, CUDA

Random.seed!(42)

println("MINIMAL HSOF Test - Prawdziwe dane, omijając problematyczne GPU.jl")
println("=" ^ 70)

# Test CUDA
if CUDA.functional()
    println("✅ CUDA functional: $(length(CUDA.devices())) GPU")
    gpu_available = true
else
    println("❌ CUDA not functional")
    gpu_available = false
end

# Load real parquet data (converted)
println("\n📊 Loading REAL parquet data...")
df = CSV.read("competitions/Titanic/train_features_REAL.csv", DataFrame)
println("✅ Real data loaded: $(size(df))")

# Process data
numeric_cols = [col for col in names(df) if col != "Survived" && col != "PassengerId" && eltype(df[!, col]) <: Union{Number, Missing}]
println("  Numeric features: $(length(numeric_cols))")

# Handle missing values
df_clean = copy(df[:, vcat(["Survived"], numeric_cols)])
for col in numeric_cols
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

X = Matrix{Float64}(df_clean[:, numeric_cols])
y = Vector{Int}(df_clean.Survived)
println("  Data prepared: $(size(X)) matrix, target rate: $(round(mean(y), digits=3))")

# Try loading individual HSOF components (NOT full HSOF)
println("\n🔧 Testing HSOF Components...")

# Test Database module
try
    println("  Testing Database module...")
    push!(LOAD_PATH, "src")
    include("src/database/Database.jl")
    using .Database
    println("  ✅ Database module loaded successfully")
catch e
    println("  ❌ Database module failed: $e")
end

# Test UI module  
try
    println("  Testing UI module...")
    include("src/ui/UI.jl")
    using .UI
    println("  ✅ UI module loaded successfully")
catch e
    println("  ❌ UI module failed: $e")
end

# Manual GPU kernels (simplified)
if gpu_available
    println("\n🎯 Manual GPU Kernel Test...")
    try
        # Simple CUDA kernel for feature correlation
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
        X_gpu = CuArray{Float32}(X)
        y_gpu = CuArray{Float32}(y)
        correlations_gpu = CUDA.zeros(Float32, size(X, 2))
        
        # Launch kernel
        threads_per_block = 256
        blocks = div(size(X, 2) + threads_per_block - 1, threads_per_block)
        
        CUDA.@cuda threads=threads_per_block blocks=blocks gpu_correlation_kernel(X_gpu, y_gpu, correlations_gpu)
        CUDA.synchronize()
        
        # Get results
        correlations_cpu = Array(correlations_gpu)
        
        println("  ✅ GPU kernel executed successfully!")
        println("  GPU correlations (top 5): $(round.(sort(correlations_cpu, rev=true)[1:5], digits=3))")
        
        # Select top features using GPU results
        global n_select = min(20, length(correlations_cpu))
        global selected_indices = sortperm(correlations_cpu, rev=true)[1:n_select]
        global X_selected = X[:, selected_indices]
        
        println("  GPU selected $(length(selected_indices)) top features")
        
    catch e
        println("  ❌ GPU kernel failed: $e")
        println("  Falling back to CPU...")
        
        # CPU fallback
        global correlations_cpu = [abs(cor(X[:, i], y)) for i in 1:size(X, 2)]
        global n_select = min(20, size(X, 2))
        global selected_indices = sortperm(correlations_cpu, rev=true)[1:n_select]
        global X_selected = X[:, selected_indices]
        global gpu_available = false
    end
else
    println("\n💻 CPU Feature Selection...")
    global correlations_cpu = [abs(cor(X[:, i], y)) for i in 1:size(X, 2)]
    global n_select = min(20, size(X, 2))
    global selected_indices = sortperm(correlations_cpu, rev=true)[1:n_select]
    global X_selected = X[:, selected_indices]
    println("  CPU selected $(length(selected_indices)) features")
end

# Cross-validation test
println("\n🎯 Cross-Validation Test...")
function cross_validate(X, y, n_folds=5)
    accuracies = Float64[]
    
    for fold in 1:n_folds
        n_samples = size(X, 1)
        test_size = div(n_samples, n_folds)
        test_start = (fold - 1) * test_size + 1
        test_end = min(fold * test_size, n_samples)
        
        test_indices = test_start:test_end
        train_indices = setdiff(1:n_samples, test_indices)
        
        X_train, X_test = X[train_indices, :], X[test_indices, :]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # Weighted prediction
        weights = [cor(X_train[:, i], y_train) for i in 1:size(X_train, 2)]
        scores = X_test * weights
        predictions = Int.(scores .> median(scores))
        
        accuracy = mean(predictions .== y_test)
        push!(accuracies, accuracy)
        println("  Fold $fold: $(round(accuracy, digits=3))")
    end
    
    return mean(accuracies), std(accuracies)
end

final_accuracy, final_std = cross_validate(X_selected, y)

# Results
println("\n" * "=" ^ 70)
println("MINIMAL HSOF Results Summary")
println("=" ^ 70)
println("Real parquet data: $(size(df)) → $(size(X_selected)) features selected")
println("GPU acceleration: $(gpu_available ? "✅ Used" : "❌ Fallback to CPU")")
println("Cross-validation accuracy: $(round(final_accuracy, digits=3)) ± $(round(final_std, digits=3))")

if final_accuracy > 0.70
    println("\n🎉 MINIMAL HSOF TEST PASSED!")
    println("✅ Prawdziwe dane działają z uproszczonym pipeline")
    if gpu_available
        println("✅ GPU kernels działają")
    end
    exit(0)
else
    println("\n⚠️ MINIMAL HSOF TEST: Moderate results")
    println("Pipeline działa ale wyniki można poprawić")
    exit(1)
end