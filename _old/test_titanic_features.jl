#!/usr/bin/env julia

"""
Simple Feature Selection Test on Titanic Dataset
Shows best features without dashboard clearing screen
"""

using CUDA, CSV, DataFrames, Statistics, Printf

println("=== TITANIC FEATURE SELECTION TEST ===")
println("Loading Titanic dataset...")

# Load Titanic data
df = CSV.read("competitions/Titanic/export/titanic_train_features.parquet", DataFrame)
println("✅ Loaded Titanic: $(size(df))")

# Check columns
println("Columns: $(names(df))")

# Prepare data - find target and features
target_col = "Survived"
feature_cols = [col for col in names(df) if col != target_col && col != "PassengerId" && eltype(df[!, col]) <: Union{Number, Missing}]

println("Target: $target_col")
println("Features found: $(length(feature_cols))")
for (i, col) in enumerate(feature_cols)
    println("  $i. $col")
end

# Clean data
df_clean = copy(df[:, feature_cols])
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

X = Matrix{Float32}(df_clean)
y = Float32.(df[!, target_col])

println("\nData prepared: $(size(X, 1)) samples, $(size(X, 2)) features")

# Simple GPU correlation calculation
if CUDA.functional()
    println("✅ GPU available: $(CUDA.name(CUDA.device()))")
    
    # Transfer to GPU
    X_gpu = CuArray(X)
    y_gpu = CuArray(y)
    
    # GPU kernel for correlation
    function correlation_kernel(X_gpu, y_gpu, correlations)
        idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        
        if idx <= size(X_gpu, 2)
            n = size(X_gpu, 1)
            
            sum_x = Float32(0.0)
            sum_y = Float32(0.0)
            sum_xx = Float32(0.0)
            sum_yy = Float32(0.0)
            sum_xy = Float32(0.0)
            
            for i in 1:n
                x = X_gpu[i, idx]
                y_val = y_gpu[i]
                sum_x += x
                sum_y += y_val
                sum_xx += x * x
                sum_yy += y_val * y_val
                sum_xy += x * y_val
            end
            
            mean_x = sum_x / n
            mean_y = sum_y / n
            
            cov_xy = sum_xy / n - mean_x * mean_y
            var_x = sum_xx / n - mean_x * mean_x
            var_y = sum_yy / n - mean_y * mean_y
            
            correlations[idx] = abs(cov_xy / sqrt(var_x * var_y + Float32(1e-8)))
        end
        
        return nothing
    end
    
    # Calculate correlations
    correlations_gpu = CUDA.zeros(Float32, length(feature_cols))
    threads = 256
    blocks = cld(length(feature_cols), threads)
    
    println("Calculating feature correlations on GPU...")
    CUDA.@cuda threads=threads blocks=blocks correlation_kernel(X_gpu, y_gpu, correlations_gpu)
    CUDA.synchronize()
    
    correlations = Array(correlations_gpu)
    
    # Sort features by correlation
    feature_ranking = sortperm(correlations, rev=true)
    
    println("\n🏆 BEST FEATURES (ranked by correlation with target):")
    println("=" * 50)
    
    for (rank, idx) in enumerate(feature_ranking)
        correlation = correlations[idx]
        feature_name = feature_cols[idx]
        println("$(rank). $(feature_name): $(round(correlation, digits=4))")
    end
    
    # Save results to CSV
    results_df = DataFrame(
        rank = 1:length(feature_cols),
        feature_name = feature_cols[feature_ranking],
        correlation = correlations[feature_ranking]
    )
    
    CSV.write("titanic_feature_ranking.csv", results_df)
    println("\n💾 Results saved to: titanic_feature_ranking.csv")
    
    # Show top 10
    println("\n🥇 TOP 10 FEATURES:")
    for i in 1:min(10, length(feature_cols))
        idx = feature_ranking[i]
        println("  $(i). $(feature_cols[idx]) (corr: $(round(correlations[idx], digits=4)))")
    end
    
else
    println("❌ GPU not available")
end

println("\n✅ Feature selection completed!")