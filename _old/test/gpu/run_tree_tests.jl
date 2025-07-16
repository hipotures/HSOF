#!/usr/bin/env julia

using Test
using CUDA

println("🌳 Running MCTS Tree Operations Test Suite")
println("="^60)

# Check CUDA availability
if !CUDA.functional()
    @error "CUDA is not functional. Cannot run GPU tests."
    exit(1)
end

# Display GPU info
device = CUDA.device()
println("GPU Device: $(CUDA.name(device))")
println("Compute Capability: $(CUDA.capability(device))")
println("Available Memory: $(round(CUDA.available_memory() / 1024^3, digits=2)) GB")
println("="^60)

# Test configuration
const VERBOSE = get(ENV, "VERBOSE", "false") == "true"
const STRESS_TESTS = get(ENV, "STRESS_TESTS", "true") == "true"
const MEMORY_TESTS = get(ENV, "MEMORY_TESTS", "true") == "true"

# Run tests with timing
function run_test_file(filename, description)
    println("\n📋 $description")
    println("-"^40)
    
    start_time = time()
    try
        include(filename)
        elapsed = time() - start_time
        println("✅ Completed in $(round(elapsed, digits=2)) seconds")
    catch e
        elapsed = time() - start_time
        println("❌ Failed after $(round(elapsed, digits=2)) seconds")
        rethrow(e)
    end
end

# Main test execution
total_start = time()

try
    # Core tree operations tests
    run_test_file("test_tree_operations.jl", "Core Tree Operations Tests")
    
    # Stress tests (optional)
    if STRESS_TESTS
        run_test_file("test_tree_stress.jl", "Tree Stress Tests")
    else
        println("\n⚠️  Skipping stress tests (set STRESS_TESTS=true to enable)")
    end
    
    # Memory leak tests (optional)
    if MEMORY_TESTS
        run_test_file("test_memory_leaks.jl", "Memory Leak Detection Tests")
    else
        println("\n⚠️  Skipping memory tests (set MEMORY_TESTS=true to enable)")
    end
    
    # Summary
    total_elapsed = time() - total_start
    println("\n" * "="^60)
    println("🎉 All tests completed successfully!")
    println("Total time: $(round(total_elapsed, digits=2)) seconds")
    println("="^60)
    
catch e
    total_elapsed = time() - total_start
    println("\n" * "="^60)
    println("💥 Test suite failed!")
    println("Total time: $(round(total_elapsed, digits=2)) seconds")
    println("="^60)
    rethrow(e)
end