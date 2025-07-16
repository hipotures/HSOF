#!/usr/bin/env julia

println("=== HSOF Testing Suite ===")

# Test individual stages
println("\n1. Testing Stage 1...")
try
    include("test/test_stage1.jl")
    println("✓ Stage 1 tests passed")
catch e
    println("✗ Stage 1 tests failed: $e")
end

println("\n2. Testing Stage 2...")
try
    include("test/test_stage2.jl")
    println("✓ Stage 2 tests passed")
catch e
    println("✗ Stage 2 tests failed: $e")
end

println("\n3. Testing Stage 3...")
try
    include("test/test_stage3.jl")
    println("✓ Stage 3 tests passed")
catch e
    println("✗ Stage 3 tests failed: $e")
end

println("\n4. Testing complete pipeline...")
try
    include("src/hsof.jl")
    if isfile("config/titanic_simple.yaml") && isfile("/home/xai/.mdm/datasets/titanic/titanic.sqlite")
        results = run_hsof("config/titanic_simple.yaml")
        println("✓ Complete pipeline test passed")
        println("  - Dataset: $(results["dataset"])")
        println("  - Original features: $(results["original_features"])")
        println("  - Final features: $(length(results["final_features"]))")
        println("  - Reduction: $(results["reduction_ratio"])%")
        println("  - Final score: $(round(results["final_score"], digits=4))")
    else
        println("⚠ Skipping pipeline test (missing configuration or database)")
    end
catch e
    println("✗ Complete pipeline test failed: $e")
end

println("\n=== Testing Complete ===")