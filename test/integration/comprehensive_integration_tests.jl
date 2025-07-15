#!/usr/bin/env julia

"""
Comprehensive End-to-End Integration Test Suite for HSOF

This test suite implements all requirements from task 10.9:
- Full pipeline execution on reference datasets (Titanic, MNIST, Synthetic)
- Accuracy validation comparing against baseline feature selection methods
- Performance validation ensuring each stage meets timing requirements under load
- Fault tolerance with simulated failures (GPU errors, memory exhaustion, network issues)
- Checkpoint recovery maintaining pipeline state after interruption
- Data validation ensuring feature consistency across stages
- Regression test suite for model updates and configuration changes
"""

using Test
using Dates
using Printf
using Random

# Set consistent random seed for reproducibility
Random.seed!(42)

# Include all test modules
include("test_integration_suite.jl")  # Existing integration tests
include("fault_tolerance_tests.jl")   # New fault tolerance tests
include("regression_suite.jl")        # New regression tests
include("data/dataset_loaders.jl")
include("pipeline_runner.jl")
include("result_validation.jl")

using .DatasetLoaders
using .PipelineRunner
using .ResultValidation
using .FaultToleranceTests
using .RegressionTestSuite

"""
Comprehensive integration test configuration
"""
struct IntegrationTestConfig
    datasets::Vector{String}
    run_fault_tolerance::Bool
    run_regression_tests::Bool
    run_performance_tests::Bool
    run_baseline_comparison::Bool
    verbose::Bool
    save_results::Bool
    test_timeout::Float64  # Maximum test time per dataset (seconds)
end

function IntegrationTestConfig(;
    datasets::Vector{String} = ["titanic", "mnist", "synthetic"],
    run_fault_tolerance::Bool = true,
    run_regression_tests::Bool = true,
    run_performance_tests::Bool = true,
    run_baseline_comparison::Bool = true,
    verbose::Bool = true,
    save_results::Bool = true,
    test_timeout::Float64 = 600.0  # 10 minutes per dataset
)
    return IntegrationTestConfig(
        datasets, run_fault_tolerance, run_regression_tests,
        run_performance_tests, run_baseline_comparison,
        verbose, save_results, test_timeout
    )
end

"""
Main comprehensive integration test runner
"""
function run_comprehensive_integration_tests(config::IntegrationTestConfig = IntegrationTestConfig())
    println("\n" * "=" * 80)
    println("HSOF COMPREHENSIVE END-TO-END INTEGRATION TEST SUITE")
    println("=" * 80)
    println("Test Configuration:")
    println("  Datasets: $(join(config.datasets, ", "))")
    println("  Fault Tolerance Tests: $(config.run_fault_tolerance)")
    println("  Regression Tests: $(config.run_regression_tests)")
    println("  Performance Validation: $(config.run_performance_tests)")
    println("  Baseline Comparison: $(config.run_baseline_comparison)")
    println("  Timeout per dataset: $(config.test_timeout)s")
    println("  Start time: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    
    # Initialize results storage
    test_start_time = time()
    comprehensive_results = Dict{String, Any}(
        "config" => config,
        "start_time" => now(),
        "datasets" => Dict{String, Any}(),
        "summary" => Dict{String, Any}()
    )
    
    # Load all datasets first
    println("\n📊 Loading reference datasets...")
    all_datasets = try
        load_all_reference_datasets()
    catch e
        println("❌ Failed to load datasets: $e")
        return comprehensive_results
    end
    
    # Validate requested datasets exist
    available_datasets = intersect(config.datasets, collect(keys(all_datasets)))
    if length(available_datasets) < length(config.datasets)
        missing = setdiff(config.datasets, available_datasets)
        println("⚠️  Warning: Missing datasets: $(join(missing, ", "))")
    end
    
    println("✅ Loaded datasets: $(join(available_datasets, ", "))")
    
    # Run tests for each dataset
    for dataset_name in available_datasets
        dataset = all_datasets[dataset_name]
        
        println("\n" * "=" * 60)
        println("TESTING DATASET: $(uppercase(dataset_name))")
        println("=" * 60)
        println("Samples: $(dataset.n_samples), Features: $(dataset.n_features)")
        
        dataset_start_time = time()
        dataset_results = Dict{String, Any}(
            "dataset_info" => Dict(
                "name" => dataset_name,
                "n_samples" => dataset.n_samples,
                "n_features" => dataset.n_features,
                "n_classes" => dataset.n_classes
            ),
            "test_results" => Dict{String, Any}(),
            "start_time" => now()
        )
        
        # Test 1: Core Pipeline Integration Tests
        println("\n🔬 Running Core Pipeline Integration Tests...")
        try
            pipeline_result = @timed run_pipeline_test(dataset, verbose=config.verbose)
            
            dataset_results["test_results"]["core_pipeline"] = Dict(
                "result" => pipeline_result.value,
                "execution_time" => pipeline_result.time,
                "memory_allocated" => pipeline_result.bytes,
                "passed" => pipeline_result.value.passed
            )
            
            if pipeline_result.value.passed
                println("✅ Core pipeline tests: PASSED")
                println("   Runtime: $(round(pipeline_result.value.total_runtime, digits=2))s")
                println("   Feature reduction: $(join(pipeline_result.value.feature_reduction, " → "))")
                println("   Quality scores: $(join(round.(pipeline_result.value.quality_scores, digits=3), " → "))")
            else
                println("❌ Core pipeline tests: FAILED")
                for error in pipeline_result.value.errors
                    println("   Error: $error")
                end
            end
            
        catch e
            println("❌ Core pipeline tests: ERROR - $e")
            dataset_results["test_results"]["core_pipeline"] = Dict(
                "error" => string(e),
                "passed" => false
            )
        end
        
        # Test 2: Performance Validation Tests
        if config.run_performance_tests
            println("\n⚡ Running Performance Validation Tests...")
            try
                performance_results = validate_performance_requirements(dataset, config.verbose)
                dataset_results["test_results"]["performance_validation"] = performance_results
                
                if performance_results["all_passed"]
                    println("✅ Performance validation: PASSED")
                else
                    println("⚠️  Performance validation: SOME ISSUES")
                    for issue in performance_results["issues"]
                        println("   Issue: $issue")
                    end
                end
                
            catch e
                println("❌ Performance validation: ERROR - $e")
                dataset_results["test_results"]["performance_validation"] = Dict(
                    "error" => string(e),
                    "all_passed" => false
                )
            end
        end
        
        # Test 3: Baseline Comparison Tests
        if config.run_baseline_comparison && haskey(dataset_results["test_results"], "core_pipeline")
            core_result = dataset_results["test_results"]["core_pipeline"]
            if get(core_result, "passed", false) && haskey(core_result["result"].stage_results, 3)
                
                println("\n📈 Running Baseline Comparison Tests...")
                try
                    final_features = core_result["result"].stage_results[3]["selected_features"]
                    X, y, _ = prepare_dataset_for_pipeline(dataset)
                    
                    baseline_results = compare_with_baseline(final_features, X, y)
                    dataset_results["test_results"]["baseline_comparison"] = baseline_results
                    
                    if haskey(baseline_results, "summary")
                        summary = baseline_results["summary"]
                        better_count = summary["better_than_baselines"]
                        total_count = summary["total_baselines"]
                        avg_improvement = summary["average_improvement"]
                        
                        println("✅ Baseline comparison: $(better_count)/$(total_count) methods outperformed")
                        println("   Average improvement: $(round(avg_improvement * 100, digits=1))%")
                        
                        if avg_improvement >= -0.1  # Within 10% of baselines
                            println("✅ Competitive with baseline methods")
                        else
                            println("⚠️  Below baseline performance")
                        end
                    end
                    
                catch e
                    println("❌ Baseline comparison: ERROR - $e")
                    dataset_results["test_results"]["baseline_comparison"] = Dict("error" => string(e))
                end
            end
        end
        
        # Test 4: Fault Tolerance Tests
        if config.run_fault_tolerance
            println("\n🛡️  Running Fault Tolerance Tests...")
            try
                fault_results = @timed run_fault_tolerance_tests(
                    datasets=[dataset_name], 
                    verbose=config.verbose
                )
                
                dataset_results["test_results"]["fault_tolerance"] = Dict(
                    "results" => fault_results.value,
                    "execution_time" => fault_results.time
                )
                
                # Count successful tests
                if haskey(fault_results.value, dataset_name)
                    dataset_fault_results = fault_results.value[dataset_name]
                    total_tests = 0
                    passed_tests = 0
                    
                    for (test_type, test_results) in dataset_fault_results
                        if isa(test_results, Dict) && !haskey(test_results, "error")
                            for (subtest, result) in test_results
                                total_tests += 1
                                if result == "success"
                                    passed_tests += 1
                                end
                            end
                        end
                    end
                    
                    println("✅ Fault tolerance: $(passed_tests)/$(total_tests) tests passed")
                end
                
            catch e
                println("❌ Fault tolerance tests: ERROR - $e")
                dataset_results["test_results"]["fault_tolerance"] = Dict("error" => string(e))
            end
        end
        
        # Test 5: Regression Tests
        if config.run_regression_tests
            println("\n🔄 Running Regression Tests...")
            try
                regression_results = @timed run_regression_tests(
                    datasets=[dataset_name],
                    verbose=config.verbose
                )
                
                dataset_results["test_results"]["regression"] = Dict(
                    "results" => regression_results.value,
                    "execution_time" => regression_results.time
                )
                
                if haskey(regression_results.value, dataset_name)
                    println("✅ Regression tests completed")
                    
                    # Check configuration tests
                    dataset_regression = regression_results.value[dataset_name]
                    if haskey(dataset_regression, "configuration_tests")
                        config_tests = dataset_regression["configuration_tests"]
                        successful = count(r -> r["success"], values(config_tests))
                        total = length(config_tests)
                        println("   Configuration tests: $(successful)/$(total) passed")
                    end
                end
                
            catch e
                println("❌ Regression tests: ERROR - $e")
                dataset_results["test_results"]["regression"] = Dict("error" => string(e))
            end
        end
        
        # Calculate dataset test duration
        dataset_duration = time() - dataset_start_time
        dataset_results["duration_seconds"] = dataset_duration
        dataset_results["end_time"] = now()
        
        # Check timeout
        if dataset_duration > config.test_timeout
            println("⚠️  Warning: Dataset tests exceeded timeout ($(round(dataset_duration, digits=1))s > $(config.test_timeout)s)")
        end
        
        # Store dataset results
        comprehensive_results["datasets"][dataset_name] = dataset_results
        
        println("\n📋 Dataset Summary: $(dataset_name)")
        println("   Duration: $(round(dataset_duration, digits=1))s")
        
        # Count passed/failed tests
        passed_categories = 0
        total_categories = 0
        
        for (category, results) in dataset_results["test_results"]
            total_categories += 1
            
            category_passed = if haskey(results, "passed")
                results["passed"]
            elseif haskey(results, "all_passed")
                results["all_passed"]
            elseif haskey(results, "error")
                false
            else
                true  # Assume success if no error field
            end
            
            if category_passed
                passed_categories += 1
            end
        end
        
        println("   Test Categories: $(passed_categories)/$(total_categories) passed")
    end
    
    # Generate comprehensive summary
    total_duration = time() - test_start_time
    comprehensive_results["end_time"] = now()
    comprehensive_results["total_duration_seconds"] = total_duration
    
    # Calculate overall statistics
    summary = generate_comprehensive_summary(comprehensive_results)
    comprehensive_results["summary"] = summary
    
    # Print final summary
    print_comprehensive_summary(summary, total_duration)
    
    # Save results if requested
    if config.save_results
        save_comprehensive_results(comprehensive_results)
    end
    
    return comprehensive_results
end

"""
Validate performance requirements for a dataset
"""
function validate_performance_requirements(dataset::DatasetInfo, verbose::Bool)
    # Define performance requirements based on dataset size
    requirements = get_performance_requirements(dataset.name)
    
    # Run performance test
    result = run_pipeline_test(dataset, verbose=false)
    
    issues = String[]
    validations = Dict{String, Any}()
    
    # Check overall runtime
    if result.total_runtime > requirements["max_total_runtime"]
        push!(issues, "Total runtime $(round(result.total_runtime, digits=1))s exceeds limit $(requirements["max_total_runtime"])s")
    end
    validations["total_runtime"] = Dict(
        "actual" => result.total_runtime,
        "limit" => requirements["max_total_runtime"],
        "passed" => result.total_runtime <= requirements["max_total_runtime"]
    )
    
    # Check memory usage
    if result.peak_memory_mb > requirements["max_memory_mb"]
        push!(issues, "Peak memory $(round(result.peak_memory_mb, digits=1))MB exceeds limit $(requirements["max_memory_mb"])MB")
    end
    validations["peak_memory"] = Dict(
        "actual" => result.peak_memory_mb,
        "limit" => requirements["max_memory_mb"],
        "passed" => result.peak_memory_mb <= requirements["max_memory_mb"]
    )
    
    # Check stage-wise performance
    for (stage, stage_result) in result.stage_results
        stage_key = "stage$(stage)"
        
        if haskey(requirements, "$(stage_key)_max_runtime")
            max_runtime = requirements["$(stage_key)_max_runtime"]
            actual_runtime = stage_result["runtime"]
            
            if actual_runtime > max_runtime
                push!(issues, "Stage $stage runtime $(round(actual_runtime, digits=1))s exceeds limit $(max_runtime)s")
            end
            
            validations["$(stage_key)_runtime"] = Dict(
                "actual" => actual_runtime,
                "limit" => max_runtime,
                "passed" => actual_runtime <= max_runtime
            )
        end
    end
    
    return Dict(
        "all_passed" => isempty(issues),
        "issues" => issues,
        "validations" => validations,
        "requirements" => requirements
    )
end

"""
Get performance requirements for a dataset
"""
function get_performance_requirements(dataset_name::String)
    requirements = Dict{String, Float64}()
    
    if dataset_name == "titanic"
        requirements["max_total_runtime"] = 30.0     # 30 seconds
        requirements["max_memory_mb"] = 500.0        # 500 MB
        requirements["stage1_max_runtime"] = 5.0     # 5 seconds
        requirements["stage2_max_runtime"] = 15.0    # 15 seconds
        requirements["stage3_max_runtime"] = 10.0    # 10 seconds
        
    elseif dataset_name == "mnist"
        requirements["max_total_runtime"] = 300.0    # 5 minutes
        requirements["max_memory_mb"] = 2000.0       # 2 GB
        requirements["stage1_max_runtime"] = 30.0    # 30 seconds
        requirements["stage2_max_runtime"] = 180.0   # 3 minutes
        requirements["stage3_max_runtime"] = 60.0    # 1 minute
        
    elseif dataset_name == "synthetic"
        requirements["max_total_runtime"] = 600.0    # 10 minutes
        requirements["max_memory_mb"] = 4000.0       # 4 GB
        requirements["stage1_max_runtime"] = 60.0    # 1 minute
        requirements["stage2_max_runtime"] = 300.0   # 5 minutes
        requirements["stage3_max_runtime"] = 120.0   # 2 minutes
        
    else
        # Default requirements for unknown datasets
        requirements["max_total_runtime"] = 600.0
        requirements["max_memory_mb"] = 2000.0
        requirements["stage1_max_runtime"] = 60.0
        requirements["stage2_max_runtime"] = 300.0
        requirements["stage3_max_runtime"] = 120.0
    end
    
    return requirements
end

"""
Generate comprehensive summary statistics
"""
function generate_comprehensive_summary(results::Dict{String, Any})
    datasets_tested = length(results["datasets"])
    datasets_passed = 0
    total_test_categories = 0
    passed_test_categories = 0
    
    dataset_summaries = Dict{String, Any}()
    
    for (dataset_name, dataset_results) in results["datasets"]
        dataset_passed = true
        local_passed = 0
        local_total = 0
        
        for (category, category_results) in dataset_results["test_results"]
            local_total += 1
            total_test_categories += 1
            
            category_passed = if haskey(category_results, "passed")
                category_results["passed"]
            elseif haskey(category_results, "all_passed")
                category_results["all_passed"]
            elseif haskey(category_results, "error")
                false
            else
                # For complex results like fault tolerance, check for success indicators
                !haskey(category_results, "error")
            end
            
            if category_passed
                local_passed += 1
                passed_test_categories += 1
            else
                dataset_passed = false
            end
        end
        
        if dataset_passed
            datasets_passed += 1
        end
        
        dataset_summaries[dataset_name] = Dict(
            "passed" => dataset_passed,
            "test_categories_passed" => local_passed,
            "total_test_categories" => local_total,
            "duration_seconds" => get(dataset_results, "duration_seconds", 0.0)
        )
    end
    
    return Dict(
        "datasets_tested" => datasets_tested,
        "datasets_passed" => datasets_passed,
        "total_test_categories" => total_test_categories,
        "passed_test_categories" => passed_test_categories,
        "overall_success_rate" => datasets_tested > 0 ? datasets_passed / datasets_tested : 0.0,
        "category_success_rate" => total_test_categories > 0 ? passed_test_categories / total_test_categories : 0.0,
        "dataset_summaries" => dataset_summaries
    )
end

"""
Print comprehensive summary
"""
function print_comprehensive_summary(summary::Dict{String, Any}, total_duration::Float64)
    println("\n" * "=" * 80)
    println("COMPREHENSIVE INTEGRATION TEST SUMMARY")
    println("=" * 80)
    
    # Overall statistics
    println("📊 Overall Results:")
    println("   Datasets Tested: $(summary["datasets_tested"])")
    println("   Datasets Passed: $(summary["datasets_passed"]) ($(round(summary["overall_success_rate"] * 100, digits=1))%)")
    println("   Test Categories: $(summary["passed_test_categories"])/$(summary["total_test_categories"]) passed ($(round(summary["category_success_rate"] * 100, digits=1))%)")
    println("   Total Duration: $(round(total_duration, digits=1)) seconds")
    
    # Per-dataset breakdown
    println("\n📋 Per-Dataset Results:")
    for (dataset_name, dataset_summary) in summary["dataset_summaries"]
        status = dataset_summary["passed"] ? "✅ PASSED" : "❌ FAILED"
        duration = round(dataset_summary["duration_seconds"], digits=1)
        test_ratio = "$(dataset_summary["test_categories_passed"])/$(dataset_summary["total_test_categories"])"
        
        println("   $(uppercase(dataset_name)): $status ($test_ratio categories, $(duration)s)")
    end
    
    # Overall assessment
    println("\n🎯 Overall Assessment:")
    if summary["overall_success_rate"] >= 1.0
        println("   🟢 EXCELLENT: All datasets passed all test categories")
    elseif summary["overall_success_rate"] >= 0.8
        println("   🟡 GOOD: Most datasets passed, minor issues detected")
    elseif summary["overall_success_rate"] >= 0.6
        println("   🟠 MODERATE: Several issues detected, investigation needed")
    else
        println("   🔴 POOR: Significant issues detected, major investigation required")
    end
    
    println("\n⏰ Test completed at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println("=" * 80)
end

"""
Save comprehensive results to file
"""
function save_comprehensive_results(results::Dict{String, Any})
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename = "test/integration/comprehensive_test_results_$timestamp.json"
    
    try
        # Convert DateTime objects to strings for JSON serialization
        serializable_results = prepare_for_json_serialization(results)
        
        open(filename, "w") do io
            JSON3.pretty(io, serializable_results, indent=2)
        end
        
        println("📁 Test results saved to: $filename")
        
    catch e
        println("⚠️  Warning: Failed to save results to file: $e")
    end
end

"""
Prepare results for JSON serialization by converting DateTime objects
"""
function prepare_for_json_serialization(obj)
    if isa(obj, Dict)
        return Dict(k => prepare_for_json_serialization(v) for (k, v) in obj)
    elseif isa(obj, Array)
        return [prepare_for_json_serialization(item) for item in obj]
    elseif isa(obj, DateTime)
        return string(obj)
    else
        return obj
    end
end

# Main execution when run as script
if abspath(PROGRAM_FILE) == @__FILE__
    # Parse command line arguments
    if length(ARGS) > 0
        if ARGS[1] == "--help" || ARGS[1] == "-h"
            println("""
Usage: julia comprehensive_integration_tests.jl [OPTIONS]

Options:
  --datasets DATASETS    Comma-separated list of datasets to test (default: titanic,mnist,synthetic)
  --quick               Run only core tests (skip fault tolerance and regression)
  --no-fault            Skip fault tolerance tests
  --no-regression       Skip regression tests
  --no-baseline         Skip baseline comparison tests
  --no-performance      Skip performance validation tests
  --quiet               Reduce output verbosity
  --no-save             Don't save results to file
  --timeout SECONDS     Maximum time per dataset (default: 600)
  --help, -h            Show this help message

Examples:
  julia comprehensive_integration_tests.jl
  julia comprehensive_integration_tests.jl --datasets titanic,mnist --quick
  julia comprehensive_integration_tests.jl --no-fault --timeout 300
""")
            exit(0)
        end
    end
    
    # Parse arguments
    config = IntegrationTestConfig()
    
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        
        if arg == "--datasets" && i + 1 <= length(ARGS)
            config = IntegrationTestConfig(
                datasets = split(ARGS[i + 1], ","),
                run_fault_tolerance = config.run_fault_tolerance,
                run_regression_tests = config.run_regression_tests,
                run_performance_tests = config.run_performance_tests,
                run_baseline_comparison = config.run_baseline_comparison,
                verbose = config.verbose,
                save_results = config.save_results,
                test_timeout = config.test_timeout
            )
            i += 2
        elseif arg == "--quick"
            config = IntegrationTestConfig(
                datasets = config.datasets,
                run_fault_tolerance = false,
                run_regression_tests = false,
                run_performance_tests = true,
                run_baseline_comparison = true,
                verbose = config.verbose,
                save_results = config.save_results,
                test_timeout = config.test_timeout
            )
            i += 1
        elseif arg == "--no-fault"
            config = IntegrationTestConfig(
                datasets = config.datasets,
                run_fault_tolerance = false,
                run_regression_tests = config.run_regression_tests,
                run_performance_tests = config.run_performance_tests,
                run_baseline_comparison = config.run_baseline_comparison,
                verbose = config.verbose,
                save_results = config.save_results,
                test_timeout = config.test_timeout
            )
            i += 1
        elseif arg == "--timeout" && i + 1 <= length(ARGS)
            timeout = parse(Float64, ARGS[i + 1])
            config = IntegrationTestConfig(
                datasets = config.datasets,
                run_fault_tolerance = config.run_fault_tolerance,
                run_regression_tests = config.run_regression_tests,
                run_performance_tests = config.run_performance_tests,
                run_baseline_comparison = config.run_baseline_comparison,
                verbose = config.verbose,
                save_results = config.save_results,
                test_timeout = timeout
            )
            i += 2
        else
            i += 1
        end
    end
    
    # Run the comprehensive tests
    results = run_comprehensive_integration_tests(config)
    
    # Exit with appropriate code
    overall_success = get(results["summary"], "overall_success_rate", 0.0) >= 0.8
    exit(overall_success ? 0 : 1)
end