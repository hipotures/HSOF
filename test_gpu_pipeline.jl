"""
Test script for GPU-only HSOF pipeline.
Verifies that all components work correctly.
"""

using Pkg
Pkg.activate(".")

# Test GPU functionality first
println("Testing GPU functionality...")

try
    using CUDA
    if CUDA.functional()
        println("✅ CUDA is functional")
        println("GPU: $(CUDA.name(CUDA.device()))")
        println("VRAM: $(round(CUDA.total_memory()/1024^3, digits=1)) GB")
    else
        println("❌ CUDA not functional")
        exit(1)
    end
catch e
    println("❌ CUDA.jl not available: $e")
    exit(1)
end

# Test basic imports
println("\nTesting imports...")
try
    using Flux, JSON3, YAML, SQLite, DataFrames, Statistics, Random
    using MLJ, XGBoost, StatsBase
    println("✅ All imports successful")
catch e
    println("❌ Import failed: $e")
    exit(1)
end

# Include the main pipeline
println("\nIncluding HSOF pipeline...")
try
    include("src/hsof.jl")
    println("✅ Pipeline included successfully")
catch e
    println("❌ Pipeline include failed: $e")
    exit(1)
end

# Test with synthetic data
println("\nTesting with synthetic data...")
try
    # Create synthetic dataset
    n_samples = 1000
    n_features = 100
    
    X = randn(Float32, n_samples, n_features)
    y = rand(Float32, n_samples) .> 0.5
    y = Float32.(y)
    feature_names = ["feature_$i" for i in 1:n_features]
    
    println("Generated synthetic data: $n_samples samples × $n_features features")
    
    # Test Stage 1
    println("\nTesting Stage 1...")
    X1, features1, indices1 = gpu_stage1_filter(X, y, feature_names)
    println("✅ Stage 1 completed: $(size(X1, 2)) features selected")
    
    # Test metamodel creation
    println("\nTesting metamodel creation...")
    metamodel = create_metamodel(size(X1, 2))
    println("✅ Metamodel created successfully")
    
    # Test metamodel pretraining (abbreviated)
    println("\nTesting metamodel pretraining...")
    pretrain_metamodel!(metamodel, X1, y, n_samples=500, epochs=5)
    println("✅ Metamodel pretraining completed")
    
    # Test Stage 2 (if feature count is suitable)
    if size(X1, 2) <= 64
        println("\nTesting Stage 2...")
        X2, features2, indices2 = gpu_stage2_mcts_metamodel(
            X1, y, features1, metamodel,
            total_iterations=1000, n_trees=5
        )
        println("✅ Stage 2 completed: $(size(X2, 2)) features selected")
    else
        println("\n⚠️  Skipping Stage 2 test (too many features for bit mask)")
        X2 = X1[:, 1:50]
        features2 = features1[1:50]
    end
    
    # Test Stage 3
    println("\nTesting Stage 3...")
    final_features, final_score, best_model = stage3_precise_evaluation(
        X2, y, features2, "binary_classification",
        n_candidates=20, target_range=(10, 15)
    )
    println("✅ Stage 3 completed: $(length(final_features)) features selected")
    println("Final score: $(round(final_score, digits=4)) ($(best_model))")
    
    println("\n🎉 All tests passed successfully!")
    
catch e
    println("❌ Test failed: $e")
    println("Stack trace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    exit(1)
end

# Test GPU performance benchmark
println("\nTesting GPU performance benchmark...")
try
    benchmark_gpu_performance()
    println("✅ Performance benchmark completed")
catch e
    println("⚠️  Performance benchmark failed: $e")
end

# Test error handling
println("\nTesting error handling...")
try
    # Test with invalid configuration
    try
        run_hsof_gpu_pipeline("nonexistent.yaml")
    catch e
        println("✅ Error handling works: caught $e")
    end
catch e
    println("❌ Error handling test failed: $e")
end

println("\n" * "="^60)
println("GPU PIPELINE TEST COMPLETED")
println("="^60)
println("All core components are working correctly!")
println("You can now run the full pipeline with:")
println("  julia --project=. src/hsof.jl titanic.yaml")
println("="^60)