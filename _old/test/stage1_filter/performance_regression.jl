#!/usr/bin/env julia

"""
Automated Performance Regression Detection System
Monitors Stage 1 Fast Filtering performance and alerts on regressions
"""

using CUDA
using Statistics
using DataFrames
using CSV
using Dates
using JSON

# Include modules
include("../../src/stage1_filter/feature_ranking.jl")
include("../../src/stage1_filter/fused_pipeline.jl")

using .FeatureRanking
using .FusedPipeline

# Configuration
const REGRESSION_TOLERANCE = 0.05  # 5% tolerance
const BASELINE_FILE = "performance_baseline.json"
const RESULTS_DIR = "regression_test_results"

"""
Performance baseline structure
"""
struct PerformanceBaseline
    date::DateTime
    gpu_name::String
    julia_version::String
    cuda_version::String
    benchmarks::Dict{String, Float64}
end

"""
Run standardized performance tests
"""
function run_performance_tests()
    if !CUDA.functional()
        error("CUDA not functional - cannot run performance tests")
    end
    
    println("Running performance regression tests...")
    println("GPU: $(CUDA.name(CUDA.device()))")
    println("Date: $(Dates.now())")
    
    results = Dict{String, Float64}()
    
    # Test 1: Small dataset (1K × 1K)
    println("\n--- Test 1: Small Dataset (1K × 1K) ---")
    X_small = CUDA.randn(Float32, 1000, 1000)
    y_small = CuArray(Int32.(rand(1:2, 1000)))
    
    # Warm up
    _ = select_features(X_small, y_small, RankingConfig(n_features_to_select = 100))
    CUDA.synchronize()
    
    times_small = Float64[]
    for i in 1:5
        t = @elapsed begin
            _ = select_features(X_small, y_small, RankingConfig(n_features_to_select = 100))
            CUDA.synchronize()
        end
        push!(times_small, t)
    end
    results["small_dataset_ms"] = median(times_small) * 1000
    
    # Test 2: Medium dataset (5K × 5K)
    println("\n--- Test 2: Medium Dataset (5K × 5K) ---")
    X_medium = CUDA.randn(Float32, 5000, 5000)
    y_medium = CuArray(Int32.(rand(1:2, 5000)))
    
    times_medium = Float64[]
    for i in 1:3
        t = @elapsed begin
            _ = select_features(X_medium, y_medium, RankingConfig(n_features_to_select = 500))
            CUDA.synchronize()
        end
        push!(times_medium, t)
    end
    results["medium_dataset_ms"] = median(times_medium) * 1000
    
    # Test 3: Large features (10K × 1K)
    println("\n--- Test 3: Large Features (10K × 1K) ---")
    if CUDA.available_memory() > 1 * 1024^3
        X_large_feat = CUDA.randn(Float32, 10000, 1000)
        y_large_feat = CuArray(Int32.(rand(1:3, 1000)))
        
        times_large_feat = Float64[]
        for i in 1:3
            t = @elapsed begin
                _ = select_features(X_large_feat, y_large_feat, 
                    RankingConfig(n_features_to_select = 500))
                CUDA.synchronize()
            end
            push!(times_large_feat, t)
        end
        results["large_features_ms"] = median(times_large_feat) * 1000
    end
    
    # Test 4: Fused pipeline
    println("\n--- Test 4: Fused Pipeline (5K × 5K) ---")
    fused_config = FusedPipelineConfig(n_features_to_select = Int32(500))
    selected_fused = CUDA.fill(Int32(-1), 500)
    
    times_fused = Float64[]
    for i in 1:3
        t = @elapsed begin
            fused_feature_selection_pipeline!(selected_fused, X_medium, y_medium, fused_config)
            CUDA.synchronize()
        end
        push!(times_fused, t)
    end
    results["fused_pipeline_ms"] = median(times_fused) * 1000
    
    # Test 5: Memory bandwidth
    println("\n--- Test 5: Memory Bandwidth ---")
    n_elements = 5000 * 5000
    element_size = sizeof(Float32)
    data_size_gb = n_elements * element_size / 1024^3
    
    # Time memory transfer
    X_mem = CUDA.randn(Float32, 5000, 5000)
    t_transfer = @elapsed begin
        Y_mem = copy(X_mem)
        CUDA.synchronize()
    end
    
    bandwidth_gb_s = data_size_gb / t_transfer
    results["memory_bandwidth_gb_s"] = bandwidth_gb_s
    
    # Cleanup
    CUDA.reclaim()
    
    return results
end

"""
Load performance baseline
"""
function load_baseline(filename::String)
    if !isfile(filename)
        return nothing
    end
    
    data = JSON.parsefile(filename)
    return PerformanceBaseline(
        DateTime(data["date"]),
        data["gpu_name"],
        data["julia_version"],
        data["cuda_version"],
        Dict{String, Float64}(data["benchmarks"])
    )
end

"""
Save performance baseline
"""
function save_baseline(results::Dict{String, Float64}, filename::String)
    baseline = Dict(
        "date" => string(Dates.now()),
        "gpu_name" => CUDA.name(CUDA.device()),
        "julia_version" => string(VERSION),
        "cuda_version" => string(CUDA.version()),
        "benchmarks" => results
    )
    
    open(filename, "w") do f
        JSON.print(f, baseline, 4)
    end
end

"""
Check for performance regression
"""
function check_regression(current::Dict{String, Float64}, baseline::PerformanceBaseline)
    regressions = String[]
    improvements = String[]
    
    println("\n" * "="^60)
    println("PERFORMANCE REGRESSION ANALYSIS")
    println("="^60)
    println("Baseline from: $(baseline.date)")
    println("Tolerance: ±$(REGRESSION_TOLERANCE * 100)%")
    println("-"^60)
    
    for (test_name, current_value) in current
        if haskey(baseline.benchmarks, test_name)
            baseline_value = baseline.benchmarks[test_name]
            
            # Calculate relative change
            if endswith(test_name, "_ms")
                # For time metrics, higher is worse
                change = (current_value - baseline_value) / baseline_value
            else
                # For throughput metrics, lower is worse
                change = (baseline_value - current_value) / baseline_value
            end
            
            status = if change > REGRESSION_TOLERANCE
                push!(regressions, test_name)
                "REGRESSION"
            elseif change < -REGRESSION_TOLERANCE
                push!(improvements, test_name)
                "IMPROVED"
            else
                "OK"
            end
            
            println("$test_name:")
            println("  Baseline: $(round(baseline_value, digits=2))")
            println("  Current:  $(round(current_value, digits=2))")
            println("  Change:   $(round(change * 100, digits=1))%")
            println("  Status:   $status")
            println()
        else
            println("$test_name: NEW TEST (no baseline)")
        end
    end
    
    println("-"^60)
    println("Summary:")
    println("  Regressions: $(length(regressions))")
    println("  Improvements: $(length(improvements))")
    println("  Total tests: $(length(current))")
    
    return regressions, improvements
end

"""
Generate regression report
"""
function generate_report(results::Dict{String, Float64}, 
                        regressions::Vector{String},
                        improvements::Vector{String})
    
    mkpath(RESULTS_DIR)
    report_file = joinpath(RESULTS_DIR, "regression_report_$(Dates.now()).md")
    
    open(report_file, "w") do f
        write(f, """
        # Performance Regression Test Report
        
        Date: $(Dates.now())
        GPU: $(CUDA.name(CUDA.device()))
        Julia: $(VERSION)
        CUDA: $(CUDA.version())
        
        ## Summary
        
        - **Regressions Found**: $(length(regressions))
        - **Improvements Found**: $(length(improvements))
        - **Tolerance**: ±$(REGRESSION_TOLERANCE * 100)%
        
        ## Results
        
        | Test | Time/Value | Unit |
        |------|------------|------|
        """)
        
        for (test, value) in results
            unit = endswith(test, "_ms") ? "ms" : 
                   endswith(test, "_gb_s") ? "GB/s" : ""
            write(f, "| $test | $(round(value, digits=2)) | $unit |\n")
        end
        
        if !isempty(regressions)
            write(f, """
            
            ## ⚠️ Performance Regressions
            
            The following tests showed performance degradation:
            
            """)
            for reg in regressions
                write(f, "- **$reg**: Performance decreased beyond tolerance\n")
            end
        end
        
        if !isempty(improvements)
            write(f, """
            
            ## ✅ Performance Improvements
            
            The following tests showed performance improvement:
            
            """)
            for imp in improvements
                write(f, "- **$imp**: Performance improved significantly\n")
            end
        end
        
        write(f, """
        
        ## Recommendations
        
        """)
        
        if !isempty(regressions)
            write(f, """
            1. Investigate recent changes that may have caused regressions
            2. Profile the affected operations to identify bottlenecks
            3. Consider reverting recent optimizations if necessary
            """)
        else
            write(f, """
            Performance is within acceptable bounds. Continue monitoring.
            """)
        end
    end
    
    println("\nReport saved to: $report_file")
    return report_file
end

"""
Main regression test runner
"""
function main()
    println("\n" * "="^80)
    println("AUTOMATED PERFORMANCE REGRESSION DETECTION")
    println("="^80)
    
    # Run performance tests
    results = run_performance_tests()
    
    # Print current results
    println("\n--- Current Performance Results ---")
    for (test, value) in results
        println("$test: $(round(value, digits=2))")
    end
    
    # Load baseline
    baseline = load_baseline(BASELINE_FILE)
    
    if baseline === nothing
        # No baseline exists, create one
        println("\nNo baseline found. Creating baseline...")
        save_baseline(results, BASELINE_FILE)
        println("Baseline saved to: $BASELINE_FILE")
        return 0
    end
    
    # Check for regressions
    regressions, improvements = check_regression(results, baseline)
    
    # Generate report
    report_file = generate_report(results, regressions, improvements)
    
    # Update baseline if requested
    if "--update-baseline" in ARGS
        println("\nUpdating baseline...")
        save_baseline(results, BASELINE_FILE)
        println("Baseline updated.")
    end
    
    # Return exit code based on regressions
    if !isempty(regressions)
        println("\n❌ PERFORMANCE REGRESSIONS DETECTED!")
        return 1
    else
        println("\n✅ NO PERFORMANCE REGRESSIONS DETECTED")
        return 0
    end
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    exit_code = main()
    exit(exit_code)
end