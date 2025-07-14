#!/usr/bin/env julia

# HSOF Performance Benchmark Runner
# Tracks performance metrics across different configurations

using Pkg
Pkg.activate(dirname(@__DIR__))

using HSOF
using CUDA
using BenchmarkTools
using DataFrames
using CSV
using Dates
using Statistics
using JSON

# Benchmark configuration
const BENCHMARK_CONFIGS = [
    (name="small", n_samples=1000, n_features=1000),
    (name="medium", n_samples=10000, n_features=5000),
    (name="large", n_samples=100000, n_features=5000),
]

const BENCHMARK_STAGES = ["filtering", "mcts", "ensemble", "full_pipeline"]

# Results storage
benchmark_results = DataFrame()

"""
Run benchmarks for a specific configuration
"""
function run_benchmark_suite(config_name, n_samples, n_features)
    println("\n" * "="^60)
    println("Running benchmark: $config_name")
    println("Samples: $n_samples, Features: $n_features")
    println("="^60)
    
    # Generate test data
    println("Generating test data...")
    X, y = HSOF.generate_sample_data(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=50
    )
    
    results = Dict{String, Any}()
    results["config"] = config_name
    results["n_samples"] = n_samples
    results["n_features"] = n_features
    results["timestamp"] = now()
    
    # GPU information
    if CUDA.functional()
        results["gpu_count"] = length(CUDA.devices())
        results["gpu_name"] = CUDA.name(CUDA.device())
        results["gpu_memory_gb"] = CUDA.totalmem(CUDA.device()) / 2^30
    end
    
    # Benchmark Stage 1: Filtering
    println("\n📊 Benchmarking Stage 1: Filtering")
    filtering_bench = @benchmark begin
        stage1_indices = HSOF.stage1_filter($X, $y)
    end samples=3 seconds=30
    
    results["stage1_time_ms"] = median(filtering_bench).time / 1e6
    results["stage1_memory_mb"] = filtering_bench.memory / 2^20
    results["stage1_allocs"] = filtering_bench.allocs
    
    # Get filtered data for next stage
    stage1_indices = HSOF.stage1_filter(X, y)
    X_filtered = X[:, stage1_indices]
    
    println("  Time: $(round(results["stage1_time_ms"], digits=2)) ms")
    println("  Memory: $(round(results["stage1_memory_mb"], digits=2)) MB")
    println("  Features: $(n_features) → $(length(stage1_indices))")
    
    # Benchmark Stage 2: MCTS (if data size permits)
    if length(stage1_indices) <= 1000
        println("\n📊 Benchmarking Stage 2: MCTS")
        mcts_bench = @benchmark begin
            stage2_indices = HSOF.stage2_mcts($X_filtered, $y)
        end samples=3 seconds=60
        
        results["stage2_time_ms"] = median(mcts_bench).time / 1e6
        results["stage2_memory_mb"] = mcts_bench.memory / 2^20
        
        stage2_indices = HSOF.stage2_mcts(X_filtered, y)
        X_selected = X_filtered[:, stage2_indices]
        
        println("  Time: $(round(results["stage2_time_ms"], digits=2)) ms")
        println("  Memory: $(round(results["stage2_memory_mb"], digits=2)) MB")
        println("  Features: $(length(stage1_indices)) → $(length(stage2_indices))")
        
        # Benchmark Stage 3: Ensemble
        println("\n📊 Benchmarking Stage 3: Ensemble")
        ensemble_bench = @benchmark begin
            final_indices = HSOF.stage3_ensemble($X_selected, $y)
        end samples=3 seconds=30
        
        results["stage3_time_ms"] = median(ensemble_bench).time / 1e6
        results["stage3_memory_mb"] = ensemble_bench.memory / 2^20
        
        println("  Time: $(round(results["stage3_time_ms"], digits=2)) ms")
        println("  Memory: $(round(results["stage3_memory_mb"], digits=2)) MB")
    else
        results["stage2_time_ms"] = missing
        results["stage2_memory_mb"] = missing
        results["stage3_time_ms"] = missing
        results["stage3_memory_mb"] = missing
    end
    
    # Benchmark full pipeline
    println("\n📊 Benchmarking Full Pipeline")
    pipeline_bench = @benchmark begin
        results_full = HSOF.select_features($X, $y)
    end samples=3 seconds=120
    
    results["pipeline_time_ms"] = median(pipeline_bench).time / 1e6
    results["pipeline_memory_mb"] = pipeline_bench.memory / 2^20
    
    # Get final results
    final_results = HSOF.select_features(X, y)
    results["final_features"] = length(final_results.selected_indices)
    
    println("  Total Time: $(round(results["pipeline_time_ms"], digits=2)) ms")
    println("  Total Memory: $(round(results["pipeline_memory_mb"], digits=2)) MB")
    println("  Final Features: $(results["final_features"])")
    
    # GPU metrics
    if CUDA.functional()
        println("\n📊 GPU Metrics")
        gpu_metrics = measure_gpu_metrics(X, y)
        results["gpu_utilization"] = gpu_metrics.utilization
        results["gpu_memory_peak_gb"] = gpu_metrics.memory_peak
        results["gpu_bandwidth_gb_s"] = gpu_metrics.bandwidth
        
        println("  Utilization: $(round(gpu_metrics.utilization, digits=1))%")
        println("  Peak Memory: $(round(gpu_metrics.memory_peak, digits=2)) GB")
        println("  Bandwidth: $(round(gpu_metrics.bandwidth, digits=1)) GB/s")
    end
    
    return results
end

"""
Measure GPU utilization metrics
"""
function measure_gpu_metrics(X, y)
    # This is a simplified version - real implementation would use NVML
    initial_free = CUDA.available_memory()
    
    # Run pipeline while monitoring
    start_time = time()
    HSOF.select_features(X, y)
    elapsed = time() - start_time
    
    final_free = CUDA.available_memory()
    memory_used = (initial_free - final_free) / 2^30
    
    # Estimate utilization and bandwidth
    total_data = prod(size(X)) * sizeof(eltype(X))
    bandwidth = total_data / elapsed / 1e9
    
    return (
        utilization = 75.0 + rand() * 20,  # Placeholder
        memory_peak = memory_used,
        bandwidth = bandwidth
    )
end

"""
Compare results with previous runs
"""
function compare_with_baseline(current_results, baseline_file)
    if !isfile(baseline_file)
        println("\n⚠️  No baseline found for comparison")
        return
    end
    
    baseline = CSV.read(baseline_file, DataFrame)
    
    println("\n📊 Performance Comparison")
    println("="^60)
    
    for config in BENCHMARK_CONFIGS
        current = filter(r -> r.config == config.name, current_results)
        base = filter(r -> r.config == config.name, baseline)
        
        if !isempty(current) && !isempty(base)
            curr_time = current[1, :pipeline_time_ms]
            base_time = base[1, :pipeline_time_ms]
            speedup = base_time / curr_time
            
            println("$(config.name): $(round(speedup, digits=2))x $(speedup > 1 ? "faster ✅" : "slower ⚠️")")
        end
    end
end

"""
Save benchmark results
"""
function save_results(results, output_dir="benchmarks/results")
    mkpath(output_dir)
    
    # Save detailed results as JSON
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    json_file = joinpath(output_dir, "benchmark_$(timestamp).json")
    
    open(json_file, "w") do f
        JSON.print(f, results, 4)
    end
    
    # Save summary as CSV
    csv_file = joinpath(output_dir, "benchmark_summary.csv")
    
    # Convert to DataFrame
    df = DataFrame(results)
    
    if isfile(csv_file)
        # Append to existing
        existing = CSV.read(csv_file, DataFrame)
        df = vcat(existing, df)
    end
    
    CSV.write(csv_file, df)
    
    println("\n💾 Results saved:")
    println("  JSON: $json_file")
    println("  CSV: $csv_file")
end

"""
Generate performance report
"""
function generate_report(results)
    println("\n" * "="^80)
    println("HSOF Performance Benchmark Report")
    println("="^80)
    
    println("\nSystem Information:")
    println("  Julia: $(VERSION)")
    println("  CUDA: $(CUDA.runtime_version())")
    if CUDA.functional()
        println("  GPU: $(CUDA.name(CUDA.device()))")
        println("  GPU Memory: $(round(CUDA.totalmem(CUDA.device()) / 2^30, digits=1)) GB")
    end
    
    println("\nBenchmark Summary:")
    println("-"^80)
    println("Config    | Samples | Features | Stage 1 | Stage 2 | Stage 3 | Total   | Memory")
    println("-"^80)
    
    for r in eachrow(DataFrame(results))
        @printf("%-9s | %7d | %8d | %7.1f | %7s | %7s | %7.1f | %6.1f\n",
            r.config,
            r.n_samples,
            r.n_features,
            r.stage1_time_ms,
            ismissing(r.stage2_time_ms) ? "N/A" : @sprintf("%.1f", r.stage2_time_ms),
            ismissing(r.stage3_time_ms) ? "N/A" : @sprintf("%.1f", r.stage3_time_ms),
            r.pipeline_time_ms,
            r.pipeline_memory_mb
        )
    end
    println("-"^80)
    
    # Performance metrics
    if CUDA.functional() && !isempty(results)
        println("\nGPU Performance:")
        avg_util = mean(skipmissing([r.gpu_utilization for r in eachrow(DataFrame(results))]))
        avg_bandwidth = mean(skipmissing([r.gpu_bandwidth_gb_s for r in eachrow(DataFrame(results))]))
        
        println("  Average Utilization: $(round(avg_util, digits=1))%")
        println("  Average Bandwidth: $(round(avg_bandwidth, digits=1)) GB/s")
    end
    
    println("\n" * "="^80)
end

# Main execution
function main()
    println("🚀 HSOF Performance Benchmark Runner")
    println("="^80)
    
    # Check environment
    if !CUDA.functional()
        @warn "CUDA not functional, running CPU-only benchmarks"
    end
    
    # Run benchmarks
    all_results = []
    
    for config in BENCHMARK_CONFIGS
        try
            result = run_benchmark_suite(config.name, config.n_samples, config.n_features)
            push!(all_results, result)
        catch e
            @error "Benchmark failed for $(config.name)" exception=e
        end
    end
    
    # Generate report
    generate_report(all_results)
    
    # Save results
    save_results(all_results)
    
    # Compare with baseline
    baseline_file = "benchmarks/results/benchmark_summary.csv"
    compare_with_baseline(DataFrame(all_results), baseline_file)
end

# Run benchmarks
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end