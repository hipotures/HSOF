#!/usr/bin/env julia

using CUDA, CSV, DataFrames, Statistics, Random

# Load data
df = CSV.read("competitions/Titanic/train_features_REAL.csv", DataFrame)
numeric_cols = [col for col in names(df) if col != "Survived" && col != "PassengerId" && eltype(df[!, col]) <: Union{Number, Missing}]

# Clean data
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

println("Data loaded: $(size(X)) matrix")
println("Target rate: $(round(mean(y), digits=3))")

# CPU baseline correlations
println("\nCPU correlations (first 5):")
cpu_correlations = [abs(cor(X[:, i], y)) for i in 1:min(5, size(X, 2))]
println(cpu_correlations)

# Test czy mój kernel faktycznie się wywołuje
println("\nTesting GPU kernel...")

function debug_gpu_correlation_kernel(X_gpu, y_gpu, correlations, debug_flag)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    # Debug: set flag that kernel was called
    if idx == 1
        debug_flag[1] = 42.0f0  # Magic number to prove kernel ran
    end
    
    if idx <= size(X_gpu, 2)
        n = size(X_gpu, 1)
        
        # Simplified correlation (just for testing)
        sum_xy = 0.0f0
        sum_x = 0.0f0
        sum_y = 0.0f0
        
        for i in 1:n
            x_val = X_gpu[i, idx]
            y_val = y_gpu[i]
            sum_xy += x_val * y_val
            sum_x += x_val  
            sum_y += y_val
        end
        
        # Very simple correlation approximation
        mean_x = sum_x / n
        mean_y = sum_y / n
        correlation_approx = abs(sum_xy / n - mean_x * mean_y)
        
        correlations[idx] = correlation_approx
    end
    return nothing
end

try
    # Transfer to GPU
    X_gpu = CuArray{Float32}(X[:, 1:5])  # Test tylko pierwsze 5 features
    y_gpu = CuArray{Float32}(y)
    correlations_gpu = CUDA.zeros(Float32, 5)
    debug_flag = CUDA.zeros(Float32, 1)
    
    println("  Data transferred to GPU")
    println("  X_gpu size: $(size(X_gpu))")
    println("  y_gpu size: $(size(y_gpu))")
    
    # Launch kernel
    threads_per_block = 256
    blocks = 1
    
    println("  Launching kernel: $blocks blocks, $threads_per_block threads")
    
    CUDA.@cuda threads=threads_per_block blocks=blocks debug_gpu_correlation_kernel(X_gpu, y_gpu, correlations_gpu, debug_flag)
    CUDA.synchronize()
    
    println("  Kernel completed")
    
    # Check results
    correlations_result = Array(correlations_gpu)
    debug_result = Array(debug_flag)
    
    println("  Debug flag: $(debug_result[1]) (should be 42 if kernel ran)")
    println("  GPU correlations: $correlations_result")
    println("  CPU correlations: $cpu_correlations")
    
    if debug_result[1] == 42.0
        println("✅ GPU kernel definitely executed!")
        
        # Compare with CPU
        if any(correlations_result .> 0)
            println("✅ GPU computed some non-zero correlations")
        else
            println("❌ GPU returned all zeros")
        end
    else
        println("❌ GPU kernel did NOT execute (debug flag not set)")
    end
    
catch e
    println("❌ GPU kernel test failed: $e")
    println("Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end