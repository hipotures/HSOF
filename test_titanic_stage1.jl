#!/usr/bin/env julia

"""
Test Stage 1 GPU filtering with Titanic dataset
"""

push!(LOAD_PATH, "src/")
using CUDA
using DataFrames
using CSV
using Statistics
using Random

include("src/gpu_stage1.jl")

function test_titanic_stage1()
    println("="^80)
    println("TESTING STAGE 1 WITH TITANIC-LIKE DATA")
    println("="^80)
    
    # Generate Titanic-like synthetic data
    Random.seed!(42)
    n_samples = 891  # Like Titanic
    
    # Create realistic features
    df = DataFrame(
        Pclass = rand([1, 2, 3], n_samples),
        Sex = rand([0, 1], n_samples),  # 0=female, 1=male
        Age = 0.5 .+ 80 .* rand(n_samples),
        SibSp = rand(0:5, n_samples),
        Parch = rand(0:4, n_samples),
        Fare = 5.0 .+ 500.0 .* rand(n_samples).^2,  # Skewed distribution
        Embarked_C = rand([0, 1], n_samples),
        Embarked_Q = rand([0, 1], n_samples),
        Embarked_S = rand([0, 1], n_samples),
        # Add some low-variance features
        Constant1 = fill(1.0, n_samples),
        Constant2 = fill(2.0, n_samples),
        LowVar1 = 5.0 .+ 0.001 .* randn(n_samples),
        LowVar2 = 10.0 .+ 0.0001 .* randn(n_samples),
        # Add noise features
        Noise1 = randn(n_samples),
        Noise2 = randn(n_samples),
        Noise3 = randn(n_samples)
    )
    
    # Create target (Survived) with realistic correlations
    # Survival correlates with: Sex (female+), Pclass (-), Age (children+), Fare (+)
    survival_score = (
        -0.5 * df.Pclass +
        -0.8 * df.Sex +  # Males have lower survival
        0.3 * (df.Age .< 16) +  # Children have higher survival
        0.2 * log.(df.Fare .+ 1) +
        0.5 * randn(n_samples)
    )
    df.Survived = Float32.(survival_score .> median(survival_score))
    
    # Convert to matrix format
    feature_cols = setdiff(names(df), ["Survived"])
    X = Matrix{Float32}(df[:, feature_cols])
    y = Vector{Float32}(df.Survived)
    feature_names = feature_cols
    
    println("Dataset created:")
    println("  Samples: $(size(X, 1))")
    println("  Features: $(size(X, 2))")
    println("  Target distribution: $(round(mean(y), digits=3)) positive")
    
    # Show feature variances
    println("\nFeature variances:")
    variances = vec(var(X, dims=1))
    for (i, (name, var_val)) in enumerate(zip(feature_names, variances))
        println("  $i. $name: $(round(var_val, digits=6))")
    end
    
    # Test Stage 1 with different thresholds
    println("\n" * "="^60)
    println("TESTING DIFFERENT VARIANCE THRESHOLDS")
    println("="^60)
    
    for variance_threshold in [1e-6, 0.001, 0.01, 0.1]
        println("\nVariance threshold: $variance_threshold")
        
        # Count features that would pass
        features_passing = sum(variances .> variance_threshold)
        println("  Features passing variance filter: $features_passing / $(length(variances))")
        
        if features_passing > 0
            passing_names = feature_names[variances .> variance_threshold]
            println("  Passing features: $(join(passing_names[1:min(5, end)], ", "))$(length(passing_names) > 5 ? "..." : "")")
        end
    end
    
    # Run actual Stage 1
    println("\n" * "="^60)
    println("RUNNING GPU STAGE 1")
    println("="^60)
    
    X_filtered, features_selected, indices = gpu_stage1_filter(
        X, y, feature_names,
        correlation_threshold=0.1,
        min_features_to_keep=5,
        variance_threshold=0.01
    )
    
    println("\nFinal selected features:")
    for (i, feat) in enumerate(features_selected[1:min(10, end)])
        idx = findfirst(==(feat), feature_names)
        if idx !== nothing
            corr = abs(cor(X[:, idx], y))
            println("  $i. $feat (correlation: $(round(corr, digits=3)))")
        end
    end
end

# Run test
if abspath(PROGRAM_FILE) == @__FILE__
    test_titanic_stage1()
end