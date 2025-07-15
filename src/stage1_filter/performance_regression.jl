module PerformanceRegression

using CUDA
using Statistics
using JSON
using Dates
using Printf

# Include profiling modules
include("cuda_timing.jl")
include("memory_profiling.jl")
include("kernel_metrics.jl")
include("bottleneck_analyzer.jl")

using .CUDATiming
using .MemoryProfiling
using .KernelMetrics
using .BottleneckAnalyzer

export PerformanceBaseline, RegressionTest, RegressionReport
export create_baseline, run_regression_test, compare_performance
export save_baseline, load_baseline, generate_regression_report

"""
Performance metric for regression testing
"""
struct PerformanceMetric
    name::String
    value::Float64
    unit::String
    tolerance::Float64  # Acceptable variance percentage
end

"""
Performance baseline for a specific operation
"""
struct PerformanceBaseline
    operation::String
    timestamp::DateTime
    gpu_model::String
    cuda_version::String
    metrics::Vector{PerformanceMetric}
    metadata::Dict{String, Any}
end

"""
Regression test configuration
"""
struct RegressionTest
    name::String
    setup_fn::Function
    benchmark_fn::Function
    teardown_fn::Function
    metrics_extractors::Dict{String, Function}
    tolerances::Dict{String, Float64}
    warmup_runs::Int
    test_runs::Int
end

"""
Regression test result
"""
struct RegressionResult
    test_name::String
    baseline::Union{PerformanceBaseline, Nothing}
    current::PerformanceBaseline
    passed::Bool
    regressions::Vector{Tuple{String, Float64, Float64}}  # (metric, baseline, current)
    improvements::Vector{Tuple{String, Float64, Float64}}
end

"""
Complete regression report
"""
struct RegressionReport
    timestamp::DateTime
    results::Vector{RegressionResult}
    summary::Dict{String, Any}
end

"""
Create a performance baseline
"""
function create_baseline(
    operation::String,
    benchmark_fn::Function;
    warmup_runs::Int = 5,
    test_runs::Int = 20,
    metadata::Dict{String, Any} = Dict()
)
    # Warmup
    for _ in 1:warmup_runs
        benchmark_fn()
        CUDA.synchronize()
    end
    
    # Collect metrics
    timer = create_timer()
    mem_tracker = create_memory_tracker()
    
    elapsed_times = Float64[]
    
    for _ in 1:test_runs
        reset_timer!(timer)
        reset_tracker!(mem_tracker)
        
        start_time = time()
        result = benchmark_fn(timer, mem_tracker)
        CUDA.synchronize()
        end_time = time()
        
        push!(elapsed_times, (end_time - start_time) * 1000)  # Convert to ms
    end
    
    # Extract metrics
    timing_results = get_timing_results(timer)
    memory_stats = get_bandwidth_stats(mem_tracker)
    
    metrics = PerformanceMetric[]
    
    # Overall execution time
    push!(metrics, PerformanceMetric(
        "total_time",
        mean(elapsed_times),
        "ms",
        5.0  # 5% tolerance
    ))
    
    # Timing metrics
    for result in timing_results
        push!(metrics, PerformanceMetric(
            "kernel_$(result.name)",
            result.mean_ms,
            "ms",
            10.0  # 10% tolerance for kernels
        ))
    end
    
    # Memory bandwidth metrics
    for (direction, stat) in memory_stats
        push!(metrics, PerformanceMetric(
            "bandwidth_$(direction)",
            stat.bandwidth_gbps,
            "GB/s",
            15.0  # 15% tolerance for bandwidth
        ))
    end
    
    # GPU info
    device = CUDA.device()
    gpu_model = CUDA.name(device)
    cuda_version = string(CUDA.version())
    
    return PerformanceBaseline(
        operation,
        now(),
        gpu_model,
        cuda_version,
        metrics,
        metadata
    )
end

"""
Run a regression test
"""
function run_regression_test(
    test::RegressionTest,
    baseline::Union{PerformanceBaseline, Nothing} = nothing
)
    # Setup
    test_data = test.setup_fn()
    
    try
        # Create current baseline
        current = create_baseline(
            test.name,
            (timer, tracker) -> test.benchmark_fn(test_data, timer, tracker),
            warmup_runs = test.warmup_runs,
            test_runs = test.test_runs
        )
        
        # Compare if baseline exists
        if !isnothing(baseline)
            passed, regressions, improvements = compare_performance(baseline, current)
            
            return RegressionResult(
                test.name,
                baseline,
                current,
                passed,
                regressions,
                improvements
            )
        else
            # No baseline to compare against
            return RegressionResult(
                test.name,
                nothing,
                current,
                true,
                [],
                []
            )
        end
    finally
        # Teardown
        test.teardown_fn(test_data)
    end
end

"""
Compare performance against baseline
"""
function compare_performance(
    baseline::PerformanceBaseline,
    current::PerformanceBaseline
)
    regressions = Tuple{String, Float64, Float64}[]
    improvements = Tuple{String, Float64, Float64}[]
    passed = true
    
    # Create metric lookup
    baseline_metrics = Dict(m.name => m for m in baseline.metrics)
    
    for current_metric in current.metrics
        if haskey(baseline_metrics, current_metric.name)
            baseline_metric = baseline_metrics[current_metric.name]
            
            # Calculate percentage change
            if baseline_metric.value > 0
                if current_metric.unit in ["ms", "s"]  # Lower is better
                    pct_change = ((current_metric.value - baseline_metric.value) / baseline_metric.value) * 100
                else  # Higher is better (bandwidth, FLOPS)
                    pct_change = ((baseline_metric.value - current_metric.value) / baseline_metric.value) * 100
                end
                
                if pct_change > baseline_metric.tolerance
                    # Regression detected
                    push!(regressions, (current_metric.name, baseline_metric.value, current_metric.value))
                    passed = false
                elseif pct_change < -baseline_metric.tolerance
                    # Improvement detected
                    push!(improvements, (current_metric.name, baseline_metric.value, current_metric.value))
                end
            end
        end
    end
    
    return passed, regressions, improvements
end

"""
Save baseline to JSON file
"""
function save_baseline(baseline::PerformanceBaseline, filepath::String)
    data = Dict(
        "operation" => baseline.operation,
        "timestamp" => string(baseline.timestamp),
        "gpu_model" => baseline.gpu_model,
        "cuda_version" => baseline.cuda_version,
        "metrics" => [
            Dict(
                "name" => m.name,
                "value" => m.value,
                "unit" => m.unit,
                "tolerance" => m.tolerance
            ) for m in baseline.metrics
        ],
        "metadata" => baseline.metadata
    )
    
    open(filepath, "w") do io
        JSON.print(io, data, 2)
    end
end

"""
Load baseline from JSON file
"""
function load_baseline(filepath::String)::PerformanceBaseline
    data = JSON.parsefile(filepath)
    
    metrics = [
        PerformanceMetric(
            m["name"],
            m["value"],
            m["unit"],
            m["tolerance"]
        ) for m in data["metrics"]
    ]
    
    return PerformanceBaseline(
        data["operation"],
        DateTime(data["timestamp"]),
        data["gpu_model"],
        data["cuda_version"],
        metrics,
        data["metadata"]
    )
end

"""
Generate regression test report
"""
function generate_regression_report(results::Vector{RegressionResult})
    total_tests = length(results)
    passed_tests = count(r -> r.passed, results)
    total_regressions = sum(length(r.regressions) for r in results)
    total_improvements = sum(length(r.improvements) for r in results)
    
    summary = Dict(
        "total_tests" => total_tests,
        "passed" => passed_tests,
        "failed" => total_tests - passed_tests,
        "total_regressions" => total_regressions,
        "total_improvements" => total_improvements,
        "pass_rate" => passed_tests / total_tests * 100
    )
    
    return RegressionReport(
        now(),
        results,
        summary
    )
end

"""
Pretty print regression result
"""
function Base.show(io::IO, result::RegressionResult)
    println(io, "Test: $(result.test_name)")
    println(io, "Status: $(result.passed ? "PASSED" : "FAILED")")
    
    if !isnothing(result.baseline)
        println(io, "Baseline from: $(result.baseline.timestamp)")
    else
        println(io, "No baseline available (recording current performance)")
    end
    
    if !isempty(result.regressions)
        println(io, "\nRegressions detected:")
        for (metric, baseline, current) in result.regressions
            pct_change = abs((current - baseline) / baseline * 100)
            println(io, "  - $metric: $baseline → $current ($(round(pct_change, digits=1))% worse)")
        end
    end
    
    if !isempty(result.improvements)
        println(io, "\nImprovements detected:")
        for (metric, baseline, current) in result.improvements
            pct_change = abs((current - baseline) / baseline * 100)
            println(io, "  - $metric: $baseline → $current ($(round(pct_change, digits=1))% better)")
        end
    end
end

"""
Pretty print regression report
"""
function Base.show(io::IO, report::RegressionReport)
    println(io, "Performance Regression Report")
    println(io, "Generated: $(report.timestamp)")
    println(io, "="^60)
    
    println(io, "\nSummary:")
    println(io, "  Total tests: $(report.summary["total_tests"])")
    println(io, "  Passed: $(report.summary["passed"])")
    println(io, "  Failed: $(report.summary["failed"])")
    println(io, "  Pass rate: $(round(report.summary["pass_rate"], digits=1))%")
    println(io, "  Total regressions: $(report.summary["total_regressions"])")
    println(io, "  Total improvements: $(report.summary["total_improvements"])")
    
    if report.summary["failed"] > 0
        println(io, "\nFailed Tests:")
        for result in report.results
            if !result.passed
                println(io, "\n$(result.test_name):")
                for (metric, baseline, current) in result.regressions
                    println(io, "  - $metric regression: $baseline → $current")
                end
            end
        end
    end
    
    println(io, "\nDetailed Results:")
    for result in report.results
        println(io, "\n" * "="^40)
        show(io, result)
    end
end

"""
Define standard regression tests for Stage 1
"""
function create_stage1_regression_tests()
    tests = RegressionTest[]
    
    # Variance calculation test
    push!(tests, RegressionTest(
        "variance_calculation",
        # Setup
        () -> CUDA.randn(Float32, 5000, 100000),
        # Benchmark
        (data, timer, tracker) -> begin
            @cuda_time timer "variance_kernel" begin
                variances = CUDA.zeros(Float32, size(data, 1))
                # Would call actual variance calculation here
                CUDA.@sync variances .= vec(var(data, dims=2))
            end
            return variances
        end,
        # Teardown
        (data) -> nothing,
        # Metrics extractors
        Dict{String, Function}(),
        # Tolerances
        Dict("variance_kernel" => 10.0),
        5,   # warmup runs
        20   # test runs
    ))
    
    # Correlation matrix test
    push!(tests, RegressionTest(
        "correlation_matrix",
        # Setup
        () -> CUDA.randn(Float32, 1000, 50000),
        # Benchmark
        (data, timer, tracker) -> begin
            n_features = size(data, 1)
            corr_matrix = CUDA.zeros(Float32, n_features, n_features)
            
            @cuda_time timer "correlation_kernel" begin
                # Would call actual correlation calculation here
                CUDA.@sync corr_matrix .= cor(data')
            end
            
            return corr_matrix
        end,
        # Teardown
        (data) -> nothing,
        Dict{String, Function}(),
        Dict("correlation_kernel" => 15.0),
        5,
        20
    ))
    
    # Batch processing test
    push!(tests, RegressionTest(
        "batch_processing",
        # Setup
        () -> (
            data = CUDA.randn(Float32, 500, 1000000),
            batch_size = 100000
        ),
        # Benchmark
        (test_data, timer, tracker) -> begin
            data, batch_size = test_data
            n_batches = cld(size(data, 2), batch_size)
            
            @cuda_time timer "batch_processing" begin
                for i in 1:n_batches
                    start_idx = (i-1) * batch_size + 1
                    end_idx = min(i * batch_size, size(data, 2))
                    batch = @view data[:, start_idx:end_idx]
                    
                    # Process batch
                    CUDA.@sync sum(batch, dims=2)
                end
            end
        end,
        # Teardown
        (data) -> nothing,
        Dict{String, Function}(),
        Dict("batch_processing" => 10.0),
        3,
        10
    ))
    
    return tests
end

export create_stage1_regression_tests

"""
Run all regression tests and generate report
"""
function run_all_regression_tests(
    tests::Vector{RegressionTest},
    baseline_dir::String = ".taskmaster/benchmarks/baselines"
)
    results = RegressionResult[]
    
    mkpath(baseline_dir)
    
    for test in tests
        println("Running regression test: $(test.name)")
        
        # Try to load baseline
        baseline_file = joinpath(baseline_dir, "$(test.name).json")
        baseline = nothing
        
        if isfile(baseline_file)
            try
                baseline = load_baseline(baseline_file)
                println("  Loaded baseline from $(baseline.timestamp)")
            catch e
                @warn "Failed to load baseline" exception=e
            end
        else
            println("  No baseline found, will create new baseline")
        end
        
        # Run test
        result = run_regression_test(test, baseline)
        push!(results, result)
        
        # Save new baseline if test passed or no baseline exists
        if result.passed || isnothing(baseline)
            save_baseline(result.current, baseline_file)
            println("  Saved baseline to $baseline_file")
        end
        
        # Print result
        println("  Result: $(result.passed ? "PASSED" : "FAILED")")
        if !result.passed
            println("  Regressions: $(length(result.regressions))")
        end
    end
    
    # Generate report
    report = generate_regression_report(results)
    
    # Save report
    report_file = joinpath(
        dirname(baseline_dir),
        "regression_report_$(Dates.format(now(), "yyyymmdd_HHMMSS")).txt"
    )
    
    open(report_file, "w") do io
        show(io, report)
    end
    
    println("\nRegression report saved to: $report_file")
    
    return report
end

export run_all_regression_tests

end # module