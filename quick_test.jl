"""
Quick test script for GPU HSOF pipeline.
Tests data loading and Stage 1 for selected datasets.
"""

using Pkg
Pkg.activate(".")

println("="^80)
println("QUICK GPU HSOF PIPELINE TEST")
println("="^80)

# Test GPU
using CUDA
if !CUDA.functional()
    error("CUDA not functional - GPU required for HSOF")
end
println("✅ GPU available: $(CUDA.name(CUDA.device()))")

# Load packages
using Flux, JSON3, YAML, SQLite, DataFrames, Statistics, Random
using MLJ, XGBoost, StatsBase

# Include pipeline
include("src/hsof.jl")

# Test configurations
test_configs = [
    "/home/xai/.mdm/config/datasets/titanic.yaml",
    "/home/xai/.mdm/config/datasets/playground_s4e11.yaml",
    "/home/xai/.mdm/config/datasets/s4e12.yaml"
]

for config_path in test_configs
    config_name = basename(config_path)
    println("\n" * "="^60)
    println("Testing: $config_name")
    println("="^60)

    try
        # Load config
        config = load_config(config_path)
        println("Dataset: $(config.name)")
        
        # Load data
        X, y, feature_names = load_dataset(config)
        println("✅ Data loaded: $(size(X, 1)) samples × $(size(X, 2)) features")
        
        # Test Stage 1
        println("\nTesting Stage 1...")
        X1, features1, indices1 = gpu_stage1_filter(X, y, feature_names)
        println("✅ Stage 1 complete: $(size(X, 2)) → $(size(X1, 2)) features")
        
        # Test metamodel if features are suitable
        if size(X1, 2) <= 64
            println("\nTesting metamodel...")
            metamodel = create_metamodel(size(X1, 2))
            println("✅ Metamodel created")
            
            # Quick metamodel test
            println("Testing metamodel evaluation...")
            test_masks = CUDA.rand(Float32, size(X1, 2), 10)
            scores = evaluate_metamodel_batch(metamodel, test_masks)
            println("✅ Metamodel evaluation: $(length(scores)) scores computed")
        end
        
        println("\n✅ Test passed for $config_name!")
        
    catch e
        println("\n❌ Test failed for $config_name: $e")
    end
    
    # Memory cleanup
    CUDA.reclaim()
end

# Cleanup
CUDA.reclaim()
println("\nTest complete.")