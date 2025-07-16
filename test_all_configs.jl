"""
Comprehensive test script for GPU HSOF pipeline.
Tests all available dataset configurations from ~/.mdm/config/datasets/
"""

using Pkg
Pkg.activate(".")

println("="^80)
println("COMPREHENSIVE GPU HSOF PIPELINE TEST")
println("="^80)
println("Testing all configurations from ~/.mdm/config/datasets/")
println("="^80)

# Test results storage
test_results = Dict()
start_time = time()

# Dataset configurations to test
config_dir = "/home/xai/.mdm/config/datasets"
config_files = [
    "titanic.yaml",
    "playground_s4e11.yaml", 
    "playground_s4e2.yaml",
    "playground_s4e4.yaml",
    "playground_s5e7.yaml",
    "s4e12.yaml",
    "s5e6.yaml"
]

# Test GPU setup first
gpu_available = false
println("\n🔧 TESTING GPU SETUP...")
try
    using CUDA
    if CUDA.functional()
        println("✅ CUDA is functional")
        println("   GPU: $(CUDA.name(CUDA.device()))")
        println("   VRAM: $(round(CUDA.total_memory()/1024^3, digits=1)) GB")
        println("   Available: $(round(CUDA.available_memory()/1024^3, digits=1)) GB")
        global gpu_available = true
    else
        println("❌ CUDA not functional - will test with synthetic data only")
        global gpu_available = false
    end
catch e
    println("❌ CUDA.jl not available: $e")
    println("   Will test with synthetic data only")
    global gpu_available = false
end

# Test imports
println("\n📦 TESTING IMPORTS...")
try
    using Flux, JSON3, YAML, SQLite, DataFrames, Statistics, Random
    using MLJ, XGBoost, StatsBase
    println("✅ All imports successful")
catch e
    println("❌ Import failed: $e")
    println("   Please install missing packages")
    exit(1)
end

# Include pipeline
println("\n📋 LOADING PIPELINE...")
try
    include("src/hsof.jl")
    println("✅ Pipeline loaded successfully")
catch e
    println("❌ Pipeline load failed: $e")
    exit(1)
end

# Test each configuration
println("\n🔬 TESTING CONFIGURATIONS...")
println("="^80)

for config_file in config_files
    config_path = joinpath(config_dir, config_file)
    
    if !isfile(config_path)
        println("⚠️  Config file not found: $config_path")
        test_results[config_file] = "file_not_found"
        continue
    end
    
    println("\n🔍 Testing: $config_file")
    println("-"^60)
    
    try
        # Load and analyze configuration
        config = load_config(config_path)
        println("Dataset: $(config.name)")
        println("Problem type: $(config.problem_type)")
        println("Database: $(config.db_path)")
        println("Target: $(config.target)")
        
        # Check if database exists
        if !isfile(config.db_path)
            println("❌ Database file not found: $(config.db_path)")
            test_results[config_file] = "database_not_found"
            continue
        end
        
        # Test data loading
        println("📥 Testing data loading...")
        X, y, feature_names = load_dataset(config)
        n_samples, n_features = size(X)
        
        println("✅ Data loaded successfully:")
        println("   Samples: $n_samples")
        println("   Features: $n_features")
        println("   Memory: $(round(sizeof(X)/1024^2, digits=2)) MB")
        
        # Check if dataset is suitable for GPU
        if gpu_available
            validate_gpu_memory(X, y)
            println("✅ GPU memory validation passed")
        end
        
        # Test Stage 1 only (fastest test)
        println("🚀 Testing Stage 1 (GPU correlation)...")
        stage1_start = time()
        
        if n_features > 10000
            println("⚠️  Large dataset ($n_features features) - using subset for test")
            X_subset = X[:, 1:min(1000, n_features)]
            feature_subset = feature_names[1:min(1000, n_features)]
        else
            X_subset = X
            feature_subset = feature_names
        end
        
        if gpu_available
            X1, features1, indices1 = gpu_stage1_filter(X_subset, y, feature_subset)
            stage1_time = time() - stage1_start
            
            println("✅ Stage 1 completed successfully:")
            println("   Time: $(round(stage1_time, digits=2)) seconds")
            println("   Features: $(size(X_subset, 2)) → $(size(X1, 2))")
            println("   Throughput: $(round(size(X_subset, 2)/stage1_time, digits=0)) features/second")
            
            test_results[config_file] = Dict(
                "status" => "success",
                "samples" => n_samples,
                "features" => n_features,
                "stage1_time" => stage1_time,
                "stage1_output" => size(X1, 2),
                "problem_type" => config.problem_type
            )
        else
            println("⚠️  GPU not available - skipping Stage 1 test")
            test_results[config_file] = Dict(
                "status" => "gpu_not_available",
                "samples" => n_samples,
                "features" => n_features,
                "problem_type" => config.problem_type
            )
        end
        
        # Memory cleanup
        if gpu_available
            CUDA.reclaim()
        end
        
        println("✅ Test completed for $config_file")
        
    catch e
        println("❌ Test failed for $config_file: $e")
        test_results[config_file] = Dict(
            "status" => "failed",
            "error" => string(e)
        )
        
        # Show stack trace for debugging
        if gpu_available
            try
                CUDA.reclaim()
            catch
                # Ignore cleanup errors
            end
        end
    end
end

# Summary report
total_time = time() - start_time
println("\n" * "="^80)
println("TEST SUMMARY REPORT")
println("="^80)
println("Total test time: $(round(total_time, digits=2)) seconds")
println("GPU available: $(gpu_available ? "✅ Yes" : "❌ No")")
println()

# Count results by status
successful_tests = 0
failed_tests = 0
skipped_tests = 0

for (config_file, result) in test_results
    if isa(result, Dict) && haskey(result, "status")
        if result["status"] == "success"
            successful_tests += 1
        elseif result["status"] == "failed"
            failed_tests += 1
        else
            skipped_tests += 1
        end
    else
        skipped_tests += 1
    end
end

println("📊 RESULTS SUMMARY:")
println("   ✅ Successful: $successful_tests")
println("   ❌ Failed: $failed_tests")
println("   ⚠️  Skipped: $skipped_tests")
println("   📋 Total: $(length(test_results))")
println()

# Detailed results
println("📝 DETAILED RESULTS:")
println("-"^80)

for (config_file, result) in sort(collect(test_results))
    if isa(result, Dict) && haskey(result, "status")
        status = result["status"]
        if status == "success"
            println("✅ $config_file")
            println("   Problem: $(result["problem_type"])")
            println("   Data: $(result["samples"]) samples × $(result["features"]) features")
            if haskey(result, "stage1_time")
                println("   Stage 1: $(round(result["stage1_time"], digits=2))s → $(result["stage1_output"]) features")
            end
        elseif status == "failed"
            println("❌ $config_file")
            println("   Error: $(result["error"])")
        elseif status == "gpu_not_available"
            println("⚠️  $config_file (GPU not available)")
            println("   Data: $(result["samples"]) samples × $(result["features"]) features")
        else
            println("⚠️  $config_file ($status)")
        end
    else
        println("⚠️  $config_file ($result)")
    end
    println()
end

# Performance analysis
if successful_tests > 0
    println("📈 PERFORMANCE ANALYSIS:")
    println("-"^80)
    
    stage1_times = []
    throughputs = []
    
    for (config_file, result) in test_results
        if isa(result, Dict) && haskey(result, "stage1_time")
            push!(stage1_times, result["stage1_time"])
            throughput = result["features"] / result["stage1_time"]
            push!(throughputs, throughput)
        end
    end
    
    if !isempty(stage1_times)
        println("Stage 1 Performance:")
        println("   Average time: $(round(sum(stage1_times)/length(stage1_times), digits=2)) seconds")
        println("   Min time: $(round(minimum(stage1_times), digits=2)) seconds")
        println("   Max time: $(round(maximum(stage1_times), digits=2)) seconds")
        println("   Average throughput: $(round(sum(throughputs)/length(throughputs), digits=0)) features/second")
    end
    println()
end

# Recommendations
println("💡 RECOMMENDATIONS:")
println("-"^80)

if !gpu_available
    println("⚠️  Install CUDA drivers and restart Julia for GPU acceleration")
end

if failed_tests > 0
    println("⚠️  Check database file paths and permissions for failed tests")
end

if successful_tests > 0
    println("✅ $(successful_tests) configurations tested successfully")
    println("   Ready for full pipeline execution with:")
    println("   julia --project=. src/hsof.jl <config_file>")
end

println()
println("="^80)
println("COMPREHENSIVE TEST COMPLETED")
println("="^80)

# Export results to JSON
using JSON3, Dates
results_file = "test_results_$(round(Int, time())).json"
JSON3.write(results_file, Dict(
    "test_summary" => Dict(
        "total_time_seconds" => round(total_time, digits=2),
        "gpu_available" => gpu_available,
        "successful_tests" => successful_tests,
        "failed_tests" => failed_tests,
        "skipped_tests" => skipped_tests,
        "total_tests" => length(test_results)
    ),
    "test_results" => test_results,
    "timestamp" => string(now())
))

println("📊 Results exported to: $results_file")