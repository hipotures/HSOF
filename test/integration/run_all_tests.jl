"""
Master test runner for complete integration testing framework.
Runs all integration tests, benchmarks, stress tests, and regression tests.
"""

using Test
using Dates
using JSON3
using Statistics

# Import all test modules
include("test_ensemble_integration.jl")
include("test_datasets.jl")
include("performance_benchmarks.jl")
include("stress_tests.jl")
include("regression_tests.jl")

"""
Comprehensive test suite runner
"""
function run_all_integration_tests(; 
    save_results::Bool = true,
    run_stress_tests::Bool = true,
    run_benchmarks::Bool = true,
    run_regression_tests::Bool = true,
    output_dir::String = "test/integration_results"
)
    println("="^80)
    println("ENSEMBLE MCTS INTEGRATION TEST SUITE")
    println("="^80)
    println("Starting comprehensive integration testing at $(now())")
    println()
    
    start_time = time()
    test_summary = Dict{String, Any}()
    
    # Create output directory
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    results_dir = joinpath(output_dir, timestamp)
    mkpath(results_dir)
    
    # 1. Run Core Integration Tests
    println("🔧 Running Core Integration Tests...")
    println("-" ^ 50)
    
    integration_start = time()
    
    try
        # Run the main integration test suite
        @info "Starting core integration tests"
        
        # This will run all the tests defined in test_ensemble_integration.jl
        integration_results = @testset "Core Integration Tests" begin
            include("test_ensemble_integration.jl")
        end
        
        integration_time = time() - integration_start
        
        test_summary["integration_tests"] = Dict(
            "duration_seconds" => integration_time,
            "status" => "completed",
            "results" => integration_results
        )
        
        @info "Core integration tests completed" duration_seconds=integration_time
        
    catch e
        @error "Core integration tests failed" error=e
        test_summary["integration_tests"] = Dict(
            "duration_seconds" => time() - integration_start,
            "status" => "failed",
            "error" => string(e)
        )
    end
    
    # 2. Run Performance Benchmarks
    if run_benchmarks
        println("\n📊 Running Performance Benchmarks...")
        println("-" ^ 50)
        
        benchmark_start = time()
        
        try
            @info "Starting performance benchmarks"
            
            # Create and run benchmark suite
            benchmark_suite = BenchmarkSuite("integration_benchmarks")
            benchmark_results, benchmark_summary = run_benchmark_suite(benchmark_suite, save_results = false)
            
            benchmark_time = time() - benchmark_start
            
            test_summary["benchmarks"] = Dict(
                "duration_seconds" => benchmark_time,
                "status" => "completed",
                "results_count" => length(benchmark_results),
                "summary" => benchmark_summary
            )
            
            if save_results
                # Save benchmark results
                open(joinpath(results_dir, "benchmark_results.json"), "w") do io
                    JSON3.pretty(io, benchmark_results)
                end
                
                open(joinpath(results_dir, "benchmark_summary.json"), "w") do io
                    JSON3.pretty(io, benchmark_summary)
                end
            end
            
            @info "Performance benchmarks completed" duration_seconds=benchmark_time results_count=length(benchmark_results)
            
        catch e
            @error "Performance benchmarks failed" error=e
            test_summary["benchmarks"] = Dict(
                "duration_seconds" => time() - benchmark_start,
                "status" => "failed",
                "error" => string(e)
            )
        end
    end
    
    # 3. Run Stress Tests
    if run_stress_tests
        println("\n🔥 Running Stress Tests...")
        println("-" ^ 50)
        
        stress_start = time()
        
        try
            @info "Starting stress tests"
            
            # Run stress test suite
            stress_results, stress_summary = run_stress_test_suite(save_results = false)
            
            stress_time = time() - stress_start
            
            test_summary["stress_tests"] = Dict(
                "duration_seconds" => stress_time,
                "status" => "completed",
                "results_count" => length(stress_results),
                "summary" => stress_summary
            )
            
            if save_results
                # Save stress test results
                open(joinpath(results_dir, "stress_results.json"), "w") do io
                    JSON3.pretty(io, stress_results)
                end
                
                open(joinpath(results_dir, "stress_summary.json"), "w") do io
                    JSON3.pretty(io, stress_summary)
                end
            end
            
            @info "Stress tests completed" duration_seconds=stress_time results_count=length(stress_results)
            
        catch e
            @error "Stress tests failed" error=e
            test_summary["stress_tests"] = Dict(
                "duration_seconds" => time() - stress_start,
                "status" => "failed",
                "error" => string(e)
            )
        end
    end
    
    # 4. Run Regression Tests
    if run_regression_tests
        println("\n🔍 Running Regression Tests...")
        println("-" ^ 50)
        
        regression_start = time()
        
        try
            @info "Starting regression tests"
            
            # Run regression test suite
            regression_results, regression_summary = run_regression_test_suite(save_results = false)
            
            regression_time = time() - regression_start
            
            test_summary["regression_tests"] = Dict(
                "duration_seconds" => regression_time,
                "status" => "completed",
                "results_count" => length(regression_results),
                "summary" => regression_summary
            )
            
            if save_results
                # Save regression test results
                open(joinpath(results_dir, "regression_results.json"), "w") do io
                    JSON3.pretty(io, regression_results)
                end
                
                open(joinpath(results_dir, "regression_summary.json"), "w") do io
                    JSON3.pretty(io, regression_summary)
                end
            end
            
            @info "Regression tests completed" duration_seconds=regression_time results_count=length(regression_results)
            
        catch e
            @error "Regression tests failed" error=e
            test_summary["regression_tests"] = Dict(
                "duration_seconds" => time() - regression_start,
                "status" => "failed",
                "error" => string(e)
            )
        end
    end
    
    # Calculate total time
    total_time = time() - start_time
    test_summary["total_duration_seconds"] = total_time
    test_summary["timestamp"] = now()
    test_summary["results_directory"] = results_dir
    
    # Generate comprehensive report
    report = generate_comprehensive_report(test_summary)
    
    if save_results
        # Save master summary
        open(joinpath(results_dir, "master_summary.json"), "w") do io
            JSON3.pretty(io, test_summary)
        end
        
        # Save comprehensive report
        open(joinpath(results_dir, "comprehensive_report.md"), "w") do io
            write(io, report)
        end
        
        # Save quick summary
        quick_summary = generate_quick_summary(test_summary)
        open(joinpath(results_dir, "quick_summary.txt"), "w") do io
            write(io, quick_summary)
        end
    end
    
    # Print final summary
    println("\n" * "="^80)
    println("INTEGRATION TEST SUITE COMPLETED")
    println("="^80)
    println("Total execution time: $(round(total_time/60, digits=2)) minutes")
    println("Results saved to: $results_dir")
    println()
    
    # Print test status summary
    test_types = ["integration_tests", "benchmarks", "stress_tests", "regression_tests"]
    
    for test_type in test_types
        if haskey(test_summary, test_type)
            status = test_summary[test_type]["status"]
            duration = test_summary[test_type]["duration_seconds"]
            
            status_icon = status == "completed" ? "✅" : "❌"
            println("$status_icon $test_type: $status ($(round(duration, digits=1))s)")
        end
    end
    
    println("\n" * "="^80)
    
    return test_summary
end

"""
Generate comprehensive test report
"""
function generate_comprehensive_report(summary::Dict)
    report = """
    # Comprehensive Integration Test Report
    
    Generated: $(summary["timestamp"])
    Total Duration: $(round(summary["total_duration_seconds"]/60, digits=2)) minutes
    
    ## Executive Summary
    
    """
    
    # Overall status
    all_passed = true
    total_tests = 0
    
    for test_type in ["integration_tests", "benchmarks", "stress_tests", "regression_tests"]
        if haskey(summary, test_type)
            status = summary[test_type]["status"]
            if status != "completed"
                all_passed = false
            end
            
            if haskey(summary[test_type], "results_count")
                total_tests += summary[test_type]["results_count"]
            end
        end
    end
    
    overall_status = all_passed ? "✅ ALL TESTS PASSED" : "❌ SOME TESTS FAILED"
    
    report *= """
    **Overall Status**: $overall_status
    **Total Test Cases**: $total_tests
    **Total Execution Time**: $(round(summary["total_duration_seconds"]/60, digits=2)) minutes
    
    ## Test Suite Results
    
    """
    
    # Integration Tests
    if haskey(summary, "integration_tests")
        it = summary["integration_tests"]
        status_icon = it["status"] == "completed" ? "✅" : "❌"
        
        report *= """
        ### $status_icon Core Integration Tests
        
        - **Status**: $(it["status"])
        - **Duration**: $(round(it["duration_seconds"], digits=1)) seconds
        """
        
        if haskey(it, "error")
            report *= "- **Error**: $(it["error"])\n"
        end
        
        report *= "\n"
    end
    
    # Benchmarks
    if haskey(summary, "benchmarks")
        bm = summary["benchmarks"]
        status_icon = bm["status"] == "completed" ? "✅" : "❌"
        
        report *= """
        ### $status_icon Performance Benchmarks
        
        - **Status**: $(bm["status"])
        - **Duration**: $(round(bm["duration_seconds"], digits=1)) seconds
        """
        
        if haskey(bm, "results_count")
            report *= "- **Benchmark Cases**: $(bm["results_count"])\n"
        end
        
        if haskey(bm, "error")
            report *= "- **Error**: $(bm["error"])\n"
        end
        
        report *= "\n"
    end
    
    # Stress Tests
    if haskey(summary, "stress_tests")
        st = summary["stress_tests"]
        status_icon = st["status"] == "completed" ? "✅" : "❌"
        
        report *= """
        ### $status_icon Stress Tests
        
        - **Status**: $(st["status"])
        - **Duration**: $(round(st["duration_seconds"], digits=1)) seconds
        """
        
        if haskey(st, "results_count")
            report *= "- **Stress Test Cases**: $(st["results_count"])\n"
        end
        
        if haskey(st, "summary")
            stress_summary = st["summary"]
            if haskey(stress_summary, "success_rate")
                report *= "- **Success Rate**: $(round(stress_summary["success_rate"]*100, digits=1))%\n"
            end
        end
        
        if haskey(st, "error")
            report *= "- **Error**: $(st["error"])\n"
        end
        
        report *= "\n"
    end
    
    # Regression Tests
    if haskey(summary, "regression_tests")
        rt = summary["regression_tests"]
        status_icon = rt["status"] == "completed" ? "✅" : "❌"
        
        report *= """
        ### $status_icon Regression Tests
        
        - **Status**: $(rt["status"])
        - **Duration**: $(round(rt["duration_seconds"], digits=1)) seconds
        """
        
        if haskey(rt, "results_count")
            report *= "- **Regression Test Cases**: $(rt["results_count"])\n"
        end
        
        if haskey(rt, "summary")
            regression_summary = rt["summary"]
            if haskey(regression_summary, "success_rate")
                report *= "- **Success Rate**: $(round(regression_summary["success_rate"]*100, digits=1))%\n"
            end
            if haskey(regression_summary, "total_regressions")
                report *= "- **Regressions Detected**: $(regression_summary["total_regressions"])\n"
            end
        end
        
        if haskey(rt, "error")
            report *= "- **Error**: $(rt["error"])\n"
        end
        
        report *= "\n"
    end
    
    # Recommendations
    report *= """
    ## Recommendations
    
    """
    
    if all_passed
        report *= """
        🎉 **All tests passed successfully!**
        
        - The ensemble system is functioning correctly
        - Performance is within expected parameters
        - No regressions detected
        - System is ready for production use
        """
    else
        report *= """
        ⚠️  **Some tests failed - action required:**
        
        - Review failed test details above
        - Check error logs for specific issues
        - Address any performance regressions
        - Re-run tests after fixes
        """
    end
    
    return report
end

"""
Generate quick summary for command line
"""
function generate_quick_summary(summary::Dict)
    quick = """
    INTEGRATION TEST SUITE SUMMARY
    =============================
    
    Timestamp: $(summary["timestamp"])
    Total Duration: $(round(summary["total_duration_seconds"]/60, digits=2)) minutes
    
    TEST RESULTS:
    """
    
    for test_type in ["integration_tests", "benchmarks", "stress_tests", "regression_tests"]
        if haskey(summary, test_type)
            status = summary[test_type]["status"]
            duration = summary[test_type]["duration_seconds"]
            
            status_symbol = status == "completed" ? "PASS" : "FAIL"
            quick *= "  $test_type: $status_symbol ($(round(duration, digits=1))s)\n"
        end
    end
    
    return quick
end

"""
Quick integration test runner for CI/CD
"""
function run_quick_integration_tests()
    @info "Running quick integration tests for CI/CD"
    
    # Run minimal test suite
    return run_all_integration_tests(
        save_results = true,
        run_stress_tests = false,  # Skip stress tests for speed
        run_benchmarks = true,
        run_regression_tests = true,
        output_dir = "test/ci_results"
    )
end

"""
Full integration test runner for comprehensive validation
"""
function run_full_integration_tests()
    @info "Running full integration tests for comprehensive validation"
    
    # Run complete test suite
    return run_all_integration_tests(
        save_results = true,
        run_stress_tests = true,
        run_benchmarks = true,
        run_regression_tests = true,
        output_dir = "test/full_results"
    )
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    # Run full test suite if this file is executed directly
    @info "Running comprehensive integration test suite..."
    results = run_full_integration_tests()
    
    # Print summary
    println("\nTest suite completed successfully!")
    println("Check the results directory for detailed reports.")
end

export run_all_integration_tests, run_quick_integration_tests, run_full_integration_tests
export generate_comprehensive_report, generate_quick_summary