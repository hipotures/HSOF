#!/usr/bin/env julia

# Test metamodel debugging with a small dataset
using Pkg
Pkg.activate(".")

using Random, Statistics, StatsBase
Random.seed!(42)

# Include necessary files
include("src/config_loader.jl")
include("src/data_loader.jl")
include("src/metamodel.jl")
include("src/gpu_stage1.jl")

# Create synthetic data for testing
println("Creating synthetic test data...")
n_samples = 1000
n_features = 50

X = randn(Float32, n_samples, n_features)
y = Float32.((X[:, 1] .+ 2*X[:, 2] .+ randn(n_samples)*0.1) .> 0)
feature_names = ["feature_$i" for i in 1:n_features]

println("Dataset: $n_samples samples × $n_features features")
println("Target distribution: $(round(100*mean(y), digits=1))% positive")

# Create metamodel
println("\n=== Creating Metamodel ===")
model = create_metamodel(n_features)

# Test pretrain with debug output
println("\n=== Testing Metamodel Pre-training ===")
pretrain_metamodel!(
    model, X, y,
    n_samples=500,      # Small for quick testing
    epochs=20,          # Fewer epochs
    batch_size=64,
    learning_rate=0.001f0,
    min_features=5,
    max_features=15
)

# Test validation
println("\n=== Testing Metamodel Validation ===")
correlation, mae = validate_metamodel_accuracy(
    model, X, y,
    n_test=20,         # Small for quick testing
    min_features=5,
    max_features=15
)

println("\n=== Summary ===")
println("Final correlation: $correlation")
println("Mean absolute error: $mae")

if correlation < 0.3
    println("\n⚠️  Very low correlation detected!")
    println("Running attention analysis...")
    analyze_metamodel_attention(model, X, 2)
end