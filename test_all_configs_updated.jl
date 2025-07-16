#!/usr/bin/env julia

"""
Comprehensive test script for GPU HSOF pipeline.
Tests all available dataset configurations from ~/.mdm/config/datasets/
Includes all improvements and metrics.
"""

using Pkg
Pkg.activate(".")
using Dates

println("="^80)
println("COMPREHENSIVE GPU HSOF PIPELINE TEST - UPDATED")
println("="^80)
println("Testing all configurations from ~/.mdm/config/datasets/")
println("="^80)

# Suppress XGBoost logging
ENV["XGBOOST_VERBOSITY"] = "0"

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
        println("❌ CUDA not functional")
        global gpu_available = false
    end
catch e
    println("❌ CUDA.jl not available: $e")
    global gpu_available = false
end

if !gpu_available
    println("\n⚠️  GPU not available. Cannot run GPU-only HSOF pipeline.")
    println("Exiting...")
    exit(1)
end

# Load pipeline
println("\n📦 LOADING HSOF PIPELINE...")
try
    include("src/hsof.jl")
    println("✅ Pipeline loaded successfully")
catch e
    println("❌ Failed to load pipeline: $e")
    exit(1)
end

# Load configuration
config_path = "config/hsof.yaml"
hsof_config = load_hsof_config(config_path)
println("\n📋 CONFIGURATION:")
println("   Mode: $(hsof_config["mode"])")
println("   Stage 1 - Correlation threshold: $(hsof_config["stage1"]["correlation_threshold"])")
println("   Stage 1 - Variance threshold: $(hsof_config["stage1"]["variance_threshold"])")
println("   Stage 2 - MCTS iterations: $(hsof_config["stage2"]["total_iterations"])")
println("   Stage 3 - CV folds: $(hsof_config["stage3"]["cv_folds"])")

# Test each configuration
global successful_tests = 0
global failed_tests = 0

for config_file in config_files
    yaml_path = joinpath(config_dir, config_file)
    
    if !isfile(yaml_path)
        println("\n⚠️  Skipping $config_file - file not found")
        continue
    end
    
    println("\n" * "="^80)
    println("📊 TESTING: $config_file")
    println("="^80)
    
    test_start = time()
    
    try
        # Run pipeline with config path parameter
        results = run_hsof_gpu_pipeline(yaml_path, config_path=config_path)
        
        test_time = time() - test_start
        global successful_tests += 1
        
        # Store results
        test_results[config_file] = Dict(
            "status" => "success",
            "time" => test_time,
            "results" => results,
            "stages" => Dict(
                "stage1_features" => results["pipeline_stages"]["stage1"]["output_features"],
                "stage2_features" => results["pipeline_stages"]["stage2"]["output_features"],
                "final_features" => results["final_results"]["feature_count"],
                "reduction_percent" => results["final_results"]["reduction_percent"],
                "best_model" => results["final_results"]["best_model"],
                "cv_score" => results["final_results"]["final_cv_score"]
            )
        )
        
        println("\n✅ Test completed for $config_file in $(round(test_time, digits=2)) seconds")
        
    catch e
        global failed_tests += 1
        test_results[config_file] = Dict(
            "status" => "failed",
            "error" => string(e),
            "time" => time() - test_start
        )
        
        println("\n❌ Test failed for $config_file: $e")
        
        # Print stack trace for debugging
        if haskey(ENV, "DEBUG") && ENV["DEBUG"] == "1"
            println("\nStack trace:")
            showerror(stdout, e, catch_backtrace())
        end
    end
    
    # Memory cleanup after each test
    if gpu_available
        CUDA.reclaim()
        println("   GPU memory reclaimed")
    end
    
    # Small delay between tests
    sleep(0.5)
end

# Summary
total_time = time() - start_time
println("\n" * "="^80)
println("📊 TEST SUMMARY")
println("="^80)
println("Total tests: $(successful_tests + failed_tests)")
println("✅ Successful: $successful_tests")
println("❌ Failed: $failed_tests")
println("⏱️  Total time: $(round(total_time, digits=2)) seconds")

# Detailed results table
if successful_tests > 0
    println("\n📈 PERFORMANCE METRICS:")
    println("┌─────────────────────┬────────┬─────────┬─────────┬─────────┬──────────┬─────────┬──────────┐")
    println("│ Dataset             │ Time(s)│ Stage 1 │ Stage 2 │  Final  │ Reduction│ Model   │ CV Score │")
    println("├─────────────────────┼────────┼─────────┼─────────┼─────────┼──────────┼─────────┼──────────┤")
    
    for (config, result) in sort(collect(test_results), by=x->x[1])
        if result["status"] == "success"
            stages = result["stages"]
            dataset_name = replace(config, ".yaml" => "")
            dataset_name = length(dataset_name) > 18 ? dataset_name[1:15] * "..." : dataset_name
            
            println("│ $(rpad(dataset_name, 19)) │ " *
                    "$(lpad(round(result["time"], digits=1), 6)) │ " *
                    "$(lpad(stages["stage1_features"], 7)) │ " *
                    "$(lpad(stages["stage2_features"], 7)) │ " *
                    "$(lpad(stages["final_features"], 7)) │ " *
                    "$(lpad(round(stages["reduction_percent"], digits=1), 8))% │ " *
                    "$(rpad(stages["best_model"][1:min(7, end)], 7)) │ " *
                    "$(lpad(round(stages["cv_score"], digits=3), 8)) │")
        end
    end
    println("└─────────────────────┴────────┴─────────┴─────────┴─────────┴──────────┴─────────┴──────────┘")
end

# Failed tests details
if failed_tests > 0
    println("\n⚠️  FAILED TESTS:")
    for (config, result) in test_results
        if result["status"] == "failed"
            println("   - $config: $(result["error"])")
        end
    end
end

# GPU memory status
if gpu_available
    println("\n💾 FINAL GPU MEMORY STATUS:")
    println("   Available: $(round(CUDA.available_memory()/1024^3, digits=2)) GB")
    println("   Used: $(round((CUDA.total_memory() - CUDA.available_memory())/1024^3, digits=2)) GB")
end

println("\n✅ All tests completed!")

# Save results to JSON
using JSON3
results_file = "hsof_test_results_$(Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")).json"
open(results_file, "w") do f
    JSON3.write(f, test_results)
end
println("\n📄 Results saved to: $results_file")