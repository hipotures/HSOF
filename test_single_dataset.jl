#!/usr/bin/env julia

"""
Test single dataset with full metrics
"""

using Pkg
Pkg.activate(".")

# Suppress XGBoost logging
ENV["XGBOOST_VERBOSITY"] = "0"

println("="^80)
println("TESTING SINGLE DATASET WITH FULL METRICS")
println("="^80)

# Load pipeline
include("src/hsof.jl")

# Test with titanic config
yaml_path = "/home/xai/.mdm/config/datasets/titanic.yaml"
config_path = "config/hsof.yaml"

println("\nRunning HSOF pipeline on titanic.yaml...")
println("Configuration: $config_path")

try
    results = run_hsof_gpu_pipeline(yaml_path, config_path=config_path)
    
    println("\n" * "="^80)
    println("FINAL RESULTS SUMMARY")
    println("="^80)
    
    # Extract key metrics
    stages = results["pipeline_stages"]
    final = results["final_results"]
    perf = results["performance"]
    
    println("\nFeature Reduction:")
    println("  Original: $(results["dataset_info"]["original_features"]) features")
    println("  Stage 1: $(stages["stage1"]["output_features"]) features ($(round(stages["stage1"]["reduction_percent"], digits=1))% reduction)")
    println("  Stage 2: $(stages["stage2"]["output_features"]) features ($(round(stages["stage2"]["reduction_percent"], digits=1))% reduction)")
    println("  Final: $(final["feature_count"]) features ($(round(final["reduction_percent"], digits=1))% total reduction)")
    
    println("\nSelected Features:")
    for (i, feat) in enumerate(final["selected_features"])
        println("  $i. $feat")
    end
    
    println("\nModel Performance:")
    println("  Best Model: $(final["best_model"])")
    println("  CV Score: $(round(final["final_cv_score"], digits=4))")
    
    println("\nMetamodel Performance:")
    println("  Correlation: $(round(stages["stage2"]["metamodel_correlation"], digits=4))")
    println("  MAE: $(round(stages["stage2"]["metamodel_mae"], digits=4))")
    
    println("\nTiming Breakdown:")
    println("  Stage 1: $(stages["stage1"]["time_seconds"])s")
    println("  Stage 2: $(stages["stage2"]["time_seconds"])s")
    println("  Stage 3: $(stages["stage3"]["time_seconds"])s")
    println("  Total: $(perf["total_time_seconds"])s")
    
    println("\n✅ Test completed successfully!")
    
catch e
    println("\n❌ Test failed: $e")
    println("\nStack trace:")
    showerror(stdout, e, catch_backtrace())
end