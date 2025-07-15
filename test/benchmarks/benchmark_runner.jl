module BenchmarkRunner

using ArgParse
using Dates
using JSON

# Include all benchmark modules
include("gpu/gpu_profiler.jl")
include("memory/memory_profiler.jl")
include("latency/latency_profiler.jl")
include("regression_detector.jl")
include("reports/report_generator.jl")

# Include HSOF modules
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
include("../../src/core/models.jl")
include("../../src/stage1/statistical_filter.jl")
include("../../src/stage2/mcts_feature_selection.jl")
include("../../src/stage3/ensemble_optimizer.jl")
include("../../src/gpu/GPU.jl")

using .GPUProfiler
using .MemoryProfiler
using .LatencyProfiler
using .RegressionDetector
using .ReportGenerator
using .Models
using .StatisticalFilter
using .MCTSFeatureSelection
using .EnsembleOptimizer
using .GPU

export run_benchmarks, BenchmarkConfig

"""
Benchmark configuration
"""
struct BenchmarkConfig
    name::String
    stages::Vector{Symbol}  # [:gpu, :memory, :latency, :pipeline]
    dataset_sizes::Vector{Tuple{Int, Int}}  # [(samples, features)]
    gpu_enabled::Bool
    output_dir::String
    baseline_path::Union{Nothing, String}
    report_formats::Vector{Symbol}  # [:text, :markdown, :html]
end

"""
Parse command line arguments
"""
function parse_commandline()
    s = ArgParseSettings()
    
    @add_arg_table s begin
        "--name"
            help = "Benchmark run name"
            default = "hsof_benchmark_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
        "--stages"
            help = "Benchmark stages to run (gpu,memory,latency,pipeline)"
            default = "gpu,memory,latency"
        "--dataset-sizes"
            help = "Dataset sizes as 'samples:features' pairs (e.g., '1000:500,10000:5000')"
            default = "1000:500,5000:1000,10000:5000"
        "--gpu"
            help = "Enable GPU benchmarks"
            action = :store_true
        "--output-dir"
            help = "Output directory for results"
            default = "test/benchmarks/results"
        "--baseline"
            help = "Path to baseline file for regression detection"
            default = nothing
        "--formats"
            help = "Report formats (text,markdown,html)"
            default = "text,markdown"
        "--save-baseline"
            help = "Save results as new baseline"
            action = :store_true
    end
    
    return parse_args(s)
end

"""
Parse dataset sizes from string
"""
function parse_dataset_sizes(sizes_str::String)::Vector{Tuple{Int, Int}}
    sizes = Tuple{Int, Int}[]
    
    for size_pair in split(sizes_str, ",")
        parts = split(size_pair, ":")
        if length(parts) == 2
            samples = parse(Int, parts[1])
            features = parse(Int, parts[2])
            push!(sizes, (samples, features))
        end
    end
    
    return sizes
end

"""
Run GPU benchmarks
"""
function run_gpu_benchmarks(config::BenchmarkConfig)
    println("\n=== Running GPU Benchmarks ===")
    
    if !CUDA.functional()
        println("CUDA not available - skipping GPU benchmarks")
        return Dict{String, Any}()
    end
    
    results = Dict{String, Any}()
    
    # Benchmark different operations
    operations = Dict{String, Function}()
    
    # Matrix multiplication benchmark
    for (samples, features) in config.dataset_sizes
        name = "matmul_$(samples)x$(features)"
        operations[name] = () -> begin
            A = CUDA.rand(Float32, samples, features)
            B = CUDA.rand(Float32, features, 100)
            C = A * B
            CUDA.synchronize()
        end
    end
    
    # Feature selection benchmark
    operations["feature_selection"] = () -> begin
        X = CUDA.rand(Float32, 1000, 500)
        y = CUDA.rand(Int32, 1000)
        
        # Simulate feature scoring
        scores = vec(sum(X, dims=1))
        sorted_indices = sortperm(scores, rev=true)
        top_features = sorted_indices[1:50]
        
        CUDA.synchronize()
    end
    
    # Profile each operation
    for (name, op) in operations
        println("  Profiling: $name")
        profile = GPUProfiler.profile_gpu_operation(name, op)
        results[name] = GPUProfiler.analyze_profile(profile)
    end
    
    # Generate comparison
    GPUProfiler.performance_summary(results)
    
    return results
end

"""
Run memory benchmarks
"""
function run_memory_benchmarks(config::BenchmarkConfig)
    println("\n=== Running Memory Benchmarks ===")
    
    results = Dict{String, Any}()
    
    # Benchmark different stages
    for (samples, features) in config.dataset_sizes
        name = "pipeline_$(samples)x$(features)"
        println("  Profiling: $name")
        
        profile = MemoryProfiler.profile_memory(name, () -> begin
            # Stage 1: Statistical filtering
            X = randn(Float64, samples, features)
            y = rand(1:2, samples)
            
            filter = VarianceFilter(threshold=0.01)
            selected = fit_transform(filter, X, y)
            X_filtered = X[:, selected]
            
            # Stage 2: MCTS (simulated)
            n_select = min(50, length(selected) ÷ 2)
            mcts_selected = randperm(size(X_filtered, 2))[1:n_select]
            X_mcts = X_filtered[:, mcts_selected]
            
            # Stage 3: Ensemble (simulated)
            final_selected = randperm(size(X_mcts, 2))[1:min(15, size(X_mcts, 2))]
            X_final = X_mcts[:, final_selected]
            
            X_final
        end)
        
        results[name] = MemoryProfiler.memory_summary(profile, verbose=false)
        
        # Check for memory leaks
        MemoryProfiler.detect_memory_leaks(profile)
    end
    
    # Compare profiles
    println("\nMemory Usage Comparison:")
    for (name, summary) in results
        println("  $name:")
        println("    Host Peak: $(round(summary["host_peak_mb"], digits=1)) MB")
        println("    Device Peak: $(round(summary["device_peak_mb"], digits=1)) MB")
        println("    Allocations: $(summary["allocation_count"])")
    end
    
    return results
end

"""
Run latency benchmarks
"""
function run_latency_benchmarks(config::BenchmarkConfig)
    println("\n=== Running Latency Benchmarks ===")
    
    results = Dict{String, Any}()
    
    # Mock metamodel prediction function
    metamodel_predict = function(features)
        # Simulate neural network inference
        n_features = length(features)
        hidden_size = 128
        
        # Layer 1
        W1 = randn(Float32, hidden_size, n_features)
        b1 = randn(Float32, hidden_size)
        h1 = tanh.(W1 * features .+ b1)
        
        # Layer 2
        W2 = randn(Float32, 1, hidden_size)
        b2 = randn(Float32, 1)
        output = W2 * h1 .+ b2
        
        return output[1]
    end
    
    # Benchmark metamodel inference
    input_sizes = [f for (_, f) in config.dataset_sizes]
    benchmark_results = LatencyProfiler.benchmark_metamodel_inference(
        metamodel_predict,
        input_sizes=input_sizes,
        batch_sizes=[1, 10, 100],
        use_gpu=config.gpu_enabled
    )
    
    # Generate reports
    for (config_name, profiles) in benchmark_results
        LatencyProfiler.latency_report(profiles, target_latency_us=1000.0)
        results[config_name] = Dict(
            bs => Dict(
                "mean_us" => p.mean_us,
                "percentiles" => p.percentiles,
                "throughput_ops_per_sec" => p.throughput_ops_per_sec
            ) for (bs, p) in profiles
        )
    end
    
    return results
end

"""
Run complete pipeline benchmark
"""
function run_pipeline_benchmark(config::BenchmarkConfig)
    println("\n=== Running Pipeline Benchmark ===")
    
    results = Dict{String, Any}()
    
    for (samples, features) in config.dataset_sizes
        name = "full_pipeline_$(samples)x$(features)"
        println("  Running: $name")
        
        start_time = time()
        
        # Generate test data
        X = randn(Float64, samples, features)
        y = rand(1:2, samples)
        
        # Stage 1: Statistical filtering
        stage1_start = time()
        filter = VarianceFilter(threshold=0.01)
        selected1 = fit_transform(filter, X, y)
        X_stage1 = X[:, selected1]
        stage1_time = time() - stage1_start
        
        # Stage 2: MCTS (simplified)
        stage2_start = time()
        n_select2 = min(50, size(X_stage1, 2) ÷ 2)
        selector = MCTSFeatureSelector(
            n_iterations=100,
            exploration_constant=1.414,
            n_trees=10,
            use_gpu=config.gpu_enabled
        )
        selected2, scores = select_features(selector, X_stage1, y, n_select2)
        X_stage2 = X_stage1[:, selected2]
        stage2_time = time() - stage2_start
        
        # Stage 3: Ensemble optimization
        stage3_start = time()
        n_final = min(15, size(X_stage2, 2) ÷ 2)
        optimizer = EnsembleFeatureOptimizer(
            base_models=[
                RandomForestImportance(n_estimators=50),
                MutualInformationImportance()
            ],
            ensemble_size=5
        )
        final_features, _ = optimize_features(optimizer, X_stage2, y, n_final)
        stage3_time = time() - stage3_start
        
        total_time = time() - start_time
        
        # Store results
        results[name] = Dict(
            "total_time_s" => total_time,
            "stage1_time_s" => stage1_time,
            "stage2_time_s" => stage2_time,
            "stage3_time_s" => stage3_time,
            "feature_reduction" => [features, length(selected1), length(selected2), n_final],
            "reduction_ratio" => n_final / features
        )
        
        println("    Feature reduction: $(features) → $(length(selected1)) → $(length(selected2)) → $n_final")
        println("    Total time: $(round(total_time, digits=2))s")
    end
    
    return results
end

"""
Run all benchmarks
"""
function run_benchmarks(config::BenchmarkConfig)
    println("\nHSOF Performance Benchmark Suite")
    println("================================")
    println("Name: $(config.name)")
    println("Stages: $(join(config.stages, ", "))")
    println("GPU Enabled: $(config.gpu_enabled)")
    println()
    
    # Create output directory
    mkpath(config.output_dir)
    
    # Collect all results
    all_results = Dict{String, Any}()
    
    # Run selected benchmarks
    if :gpu in config.stages && config.gpu_enabled
        all_results["gpu"] = run_gpu_benchmarks(config)
    end
    
    if :memory in config.stages
        all_results["memory"] = run_memory_benchmarks(config)
    end
    
    if :latency in config.stages
        all_results["latency"] = run_latency_benchmarks(config)
    end
    
    if :pipeline in config.stages
        all_results["pipeline"] = run_pipeline_benchmark(config)
    end
    
    # Create performance metrics for regression detection
    metrics = create_performance_metrics(all_results)
    
    # Regression detection if baseline provided
    regression_report = nothing
    if !isnothing(config.baseline_path) && isfile(config.baseline_path)
        println("\n=== Regression Detection ===")
        baseline = RegressionDetector.load_baseline(config.baseline_path)
        regression_report = RegressionDetector.compare_with_baseline(
            config.baseline_path,
            metrics
        )
        RegressionDetector.generate_regression_report(regression_report)
    end
    
    # Generate benchmark report
    report = ReportGenerator.generate_report(
        config.name,
        gpu_profiles=get(all_results, "gpu", Dict()),
        memory_profiles=get(all_results, "memory", Dict()),
        latency_profiles=get(all_results, "latency", Dict()),
        regression_results=regression_report,
        configuration=Dict(
            "dataset_sizes" => config.dataset_sizes,
            "gpu_enabled" => config.gpu_enabled,
            "julia_version" => string(VERSION)
        )
    )
    
    # Save reports in different formats
    for format in config.report_formats
        filename = joinpath(config.output_dir, "$(config.name).$(format)")
        ReportGenerator.save_report(report, filename, format=format)
        println("\nReport saved: $filename")
    end
    
    # Save raw results as JSON
    results_file = joinpath(config.output_dir, "$(config.name)_results.json")
    open(results_file, "w") do f
        JSON.print(f, all_results, 2)
    end
    
    return report
end

"""
Create performance metrics from results
"""
function create_performance_metrics(results::Dict{String, Any})
    metrics = Dict{String, PerformanceMetric}()
    
    # GPU metrics
    if haskey(results, "gpu")
        for (op, profile) in results["gpu"]
            if isa(profile, Dict)
                metrics["gpu_$(op)_time_ms"] = PerformanceMetric(
                    "gpu_$(op)_time_ms",
                    get(profile, "execution_time_ms", 0.0),
                    "ms",
                    false  # Lower is better
                )
                metrics["gpu_$(op)_utilization"] = PerformanceMetric(
                    "gpu_$(op)_utilization",
                    get(profile, "avg_utilization", 0.0),
                    "%",
                    true  # Higher is better
                )
            end
        end
    end
    
    # Memory metrics
    if haskey(results, "memory")
        for (op, profile) in results["memory"]
            if isa(profile, Dict)
                metrics["memory_$(op)_peak_mb"] = PerformanceMetric(
                    "memory_$(op)_peak_mb",
                    get(profile, "host_peak_mb", 0.0),
                    "MB",
                    false  # Lower is better
                )
            end
        end
    end
    
    # Latency metrics
    if haskey(results, "latency")
        for (config_name, profiles) in results["latency"]
            for (batch_size, profile) in profiles
                if isa(profile, Dict)
                    metrics["latency_$(config_name)_b$(batch_size)_p99"] = PerformanceMetric(
                        "latency_$(config_name)_b$(batch_size)_p99",
                        get(get(profile, "percentiles", Dict()), 99, 0.0),
                        "μs",
                        false  # Lower is better
                    )
                end
            end
        end
    end
    
    # Pipeline metrics
    if haskey(results, "pipeline")
        for (name, profile) in results["pipeline"]
            if isa(profile, Dict)
                metrics["pipeline_$(name)_time"] = PerformanceMetric(
                    "pipeline_$(name)_time",
                    get(profile, "total_time_s", 0.0),
                    "s",
                    false  # Lower is better
                )
            end
        end
    end
    
    return metrics
end

"""
Main function
"""
function main()
    args = parse_commandline()
    
    # Create configuration
    config = BenchmarkConfig(
        args["name"],
        Symbol.(split(args["stages"], ",")),
        parse_dataset_sizes(args["dataset-sizes"]),
        args["gpu"],
        args["output-dir"],
        args["baseline"],
        Symbol.(split(args["formats"], ","))
    )
    
    # Run benchmarks
    report = run_benchmarks(config)
    
    # Save as baseline if requested
    if args["save-baseline"]
        baseline_file = joinpath(config.output_dir, "baseline_$(config.name).json")
        metrics = create_performance_metrics(report.gpu_profiles)
        baseline = RegressionDetector.create_baseline(metrics)
        RegressionDetector.save_baseline(baseline, baseline_file)
        println("\nBaseline saved: $baseline_file")
    end
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module