#!/usr/bin/env julia

# Scaling Efficiency Validation Demo
# Demonstrates comprehensive scaling efficiency testing for multi-GPU HSOF system

using CUDA
using Printf
using Statistics
using Dates

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

# Load GPU modules
include("../src/gpu/GPU.jl")
using .GPU

# Import scaling efficiency from GPU module
using .GPU.ScalingEfficiency

"""
Display progress bar for benchmark execution
"""
function progress_bar(current::Int, total::Int, width::Int = 40)
    percentage = current / total
    filled = Int(round(percentage * width))
    bar = "█" ^ filled * "░" ^ (width - filled)
    print("\r[$bar] $(round(percentage * 100, digits=1))% ($current/$total)")
    if current == total
        println()
    end
end

"""
Run comprehensive scaling efficiency demo
"""
function demo_scaling_efficiency()
    println("GPU Scaling Efficiency Validation Demo")
    println("=" ^ 60)
    
    # Check GPU availability
    if !CUDA.functional()
        println("⚠ CUDA not functional - demo requires GPU support")
        return
    end
    
    num_gpus = length(CUDA.devices())
    println("GPUs detected: $num_gpus")
    
    if num_gpus < 2
        println("⚠ Demo is optimized for multi-GPU systems (2+ GPUs)")
        println("  Running in single-GPU mode with simulated multi-GPU")
    end
    
    # Display GPU information
    println("\nGPU Configuration:")
    for i in 0:num_gpus-1
        device!(i)
        dev = device()
        println("  GPU $i: $(CUDA.name(dev))")
        println("    Memory: $(round(CUDA.totalmem(dev) / 1024^3, digits=2)) GB")
        println("    Compute capability: $(CUDA.capability(dev))")
    end
    
    # Create benchmark with progress callback
    num_benchmark_gpus = min(num_gpus, 2)  # Use at most 2 GPUs for demo
    benchmark = create_benchmark(num_gpus = num_benchmark_gpus)
    
    # Set up progress tracking
    benchmark.progress_callback = (tree, total, gpu) -> begin
        if gpu == "baseline"
            print("  Single GPU: ")
        else
            print("  Multi-GPU:  ")
        end
        progress_bar(tree, total)
    end
    
    # Demo 1: Quick Efficiency Test
    println("\n1. Quick Efficiency Test")
    println("-" * 40)
    
    quick_config = BenchmarkConfig(
        num_trees = 20,
        num_features = 500,
        num_samples = 5000,
        iterations_per_tree = 50,
        sync_interval = 10
    )
    
    println("Configuration:")
    println("  Trees: $(quick_config.num_trees)")
    println("  Features: $(quick_config.num_features)")
    println("  Samples: $(quick_config.num_samples)")
    println("  Sync interval: $(quick_config.sync_interval)")
    
    println("\nRunning baseline...")
    baseline_time = run_single_gpu_baseline(benchmark, quick_config)
    
    println("\nRunning multi-GPU...")
    result = run_multi_gpu_benchmark(benchmark, quick_config)
    
    println("\nResults:")
    println("  Single GPU time: $(round(baseline_time, digits=2))s")
    println("  Multi-GPU time: $(round(result.metrics.multi_gpu_time, digits=2))s")
    println("  Speedup: $(round(result.metrics.speedup, digits=2))x")
    println("  Efficiency: $(round(result.metrics.efficiency * 100, digits=1))%")
    
    if result.metrics.efficiency >= 0.85
        println("  ✓ Meets 85% efficiency target!")
    else
        println("  ✗ Below 85% efficiency target")
    end
    
    # Demo 2: Strong Scaling Test
    println("\n2. Strong Scaling Test (Fixed Problem Size)")
    println("-" * 40)
    
    strong_configs = [
        BenchmarkConfig(num_trees=10, num_features=1000, num_samples=10000),
        BenchmarkConfig(num_trees=20, num_features=1000, num_samples=10000),
        BenchmarkConfig(num_trees=40, num_features=1000, num_samples=10000),
    ]
    
    println("Testing with fixed dataset (1000 features, 10000 samples)")
    println("Varying number of trees: 10, 20, 40")
    
    strong_results = []
    for config in strong_configs
        println("\n  Trees: $(config.num_trees)")
        baseline = run_single_gpu_baseline(benchmark, config)
        result = run_multi_gpu_benchmark(benchmark, config)
        
        efficiency = result.metrics.efficiency * 100
        push!(strong_results, (config.num_trees, efficiency))
        
        println("    Efficiency: $(round(efficiency, digits=1))%")
    end
    
    # Demo 3: Weak Scaling Test
    println("\n3. Weak Scaling Test (Scaled Problem Size)")
    println("-" * 40)
    
    weak_configs = [
        BenchmarkConfig(num_trees=10, num_features=500, num_samples=5000),
        BenchmarkConfig(num_trees=20, num_features=1000, num_samples=10000),
        BenchmarkConfig(num_trees=40, num_features=2000, num_samples=20000),
    ]
    
    println("Scaling problem size with workload")
    
    weak_results = []
    for config in weak_configs
        println("\n  Trees: $(config.num_trees), Features: $(config.num_features), Samples: $(config.num_samples)")
        baseline = run_single_gpu_baseline(benchmark, config)
        result = run_multi_gpu_benchmark(benchmark, config)
        
        # For weak scaling, we look at how well time is maintained
        time_ratio = result.metrics.multi_gpu_time / baseline
        weak_efficiency = result.metrics.weak_scaling_efficiency * 100
        push!(weak_results, (config.num_trees, weak_efficiency))
        
        println("    Weak scaling efficiency: $(round(weak_efficiency, digits=1))%")
    end
    
    # Demo 4: Communication Overhead Analysis
    println("\n4. Communication Overhead Analysis")
    println("-" * 40)
    
    sync_configs = [
        BenchmarkConfig(num_trees=20, num_features=1000, num_samples=10000, sync_interval=5),
        BenchmarkConfig(num_trees=20, num_features=1000, num_samples=10000, sync_interval=20),
        BenchmarkConfig(num_trees=20, num_features=1000, num_samples=10000, sync_interval=50),
    ]
    
    println("Testing impact of synchronization frequency")
    
    comm_results = []
    for config in sync_configs
        println("\n  Sync interval: $(config.sync_interval) trees")
        run_single_gpu_baseline(benchmark, config)
        result = run_multi_gpu_benchmark(benchmark, config)
        
        comm_overhead = result.metrics.communication_time / result.metrics.multi_gpu_time * 100
        push!(comm_results, (config.sync_interval, comm_overhead))
        
        println("    Communication overhead: $(round(comm_overhead, digits=1))%")
        println("    Efficiency: $(round(result.metrics.efficiency * 100, digits=1))%")
    end
    
    # Demo 5: Bottleneck Analysis
    println("\n5. Bottleneck Identification")
    println("-" * 40)
    
    bottlenecks = identify_bottlenecks(benchmark)
    
    if !isempty(bottlenecks)
        println("\nPrimary bottlenecks across all tests:")
        
        # Count bottleneck types
        bottleneck_counts = Dict{Symbol, Int}()
        for (type, severity) in bottlenecks
            bottleneck_counts[type] = get(bottleneck_counts, type, 0) + 1
        end
        
        for (type, count) in sort(collect(bottleneck_counts), by=x->x[2], rev=true)
            percentage = count / length(bottlenecks) * 100
            println("  $type: $count occurrences ($(round(percentage, digits=1))%)")
        end
        
        # Show most severe bottlenecks
        println("\nMost severe bottlenecks:")
        for (i, (type, severity)) in enumerate(bottlenecks[1:min(3, end)])
            println("  $i. $type - $(round(severity * 100, digits=1))% of execution time")
        end
    end
    
    # Demo 6: Comprehensive Report
    println("\n6. Generating Comprehensive Report")
    println("-" * 40)
    
    report = generate_efficiency_report(benchmark)
    
    # Save report to file
    report_path = joinpath(@__DIR__, "scaling_efficiency_report_$(Dates.format(now(), "yyyymmdd_HHMMSS")).txt")
    open(report_path, "w") do f
        write(f, report)
    end
    
    println("Report saved to: $report_path")
    
    # Display summary from report
    println("\nReport Summary:")
    
    # Extract efficiency summary
    lines = split(report, '\n')
    in_summary = false
    for line in lines
        if occursin("Efficiency Summary:", line)
            in_summary = true
        elseif occursin("Detailed Results:", line)
            break
        elseif in_summary && !isempty(strip(line))
            println(line)
        end
    end
    
    # Demo 7: Optimization Recommendations
    println("\n7. Optimization Opportunities")
    println("-" * 40)
    
    # Calculate average efficiency across all tests
    all_efficiencies = [r.metrics.efficiency * 100 for r in benchmark.results]
    avg_efficiency = mean(all_efficiencies)
    
    println("Overall average efficiency: $(round(avg_efficiency, digits=1))%")
    
    if avg_efficiency < 85.0
        println("\nOptimization recommendations to reach 85% target:")
        
        # Analyze bottlenecks for recommendations
        primary_bottleneck = !isempty(bottlenecks) ? bottlenecks[1][1] : :unknown
        
        if primary_bottleneck == :communication
            println("  • Reduce synchronization frequency")
            println("  • Compress transferred data")
            println("  • Use larger batches for PCIe transfers")
        elseif primary_bottleneck == :synchronization
            println("  • Implement asynchronous operations")
            println("  • Overlap computation with communication")
            println("  • Use double buffering for data transfers")
        elseif primary_bottleneck == :computation
            println("  • Optimize kernel launch parameters")
            println("  • Improve memory coalescing")
            println("  • Use shared memory for frequently accessed data")
        end
    else
        println("\n✓ System achieves target efficiency!")
        println("  Consider these optimizations for even better performance:")
        println("  • Fine-tune sync intervals based on workload")
        println("  • Implement adaptive load balancing")
        println("  • Use CUDA graphs for kernel launch optimization")
    end
    
    # Demo 8: Interactive Exploration
    println("\n8. Configuration Recommendations")
    println("-" * 40)
    
    println("\nBased on testing, optimal configuration:")
    
    # Find best performing configuration
    best_result = nothing
    best_efficiency = 0.0
    
    for result in benchmark.results
        if result.metrics.efficiency > best_efficiency
            best_efficiency = result.metrics.efficiency
            best_result = result
        end
    end
    
    if !isnothing(best_result)
        config = best_result.config
        println("  Trees: $(config.num_trees)")
        println("  Features: $(config.num_features)")
        println("  Samples: $(config.num_samples)")
        println("  Sync interval: $(config.sync_interval)")
        println("  Achieved efficiency: $(round(best_efficiency * 100, digits=1))%")
    end
    
    println("\n" * "=" ^ 60)
    println("Demo completed!")
end

"""
Run focused efficiency test with specific configuration
"""
function test_specific_configuration(
    num_trees::Int = 100,
    num_features::Int = 1000,
    num_samples::Int = 10000
)
    println("\nTesting specific configuration:")
    println("  Trees: $num_trees")
    println("  Features: $num_features")
    println("  Samples: $num_samples")
    
    benchmark = create_benchmark()
    config = BenchmarkConfig(
        num_trees = num_trees,
        num_features = num_features,
        num_samples = num_samples
    )
    
    baseline = run_single_gpu_baseline(benchmark, config)
    result = run_multi_gpu_benchmark(benchmark, config)
    
    println("\nResults:")
    println("  Speedup: $(round(result.metrics.speedup, digits=2))x")
    println("  Efficiency: $(round(result.metrics.efficiency * 100, digits=1))%")
    
    if result.metrics.efficiency >= 0.85
        println("  ✓ Meets target!")
    else
        println("  ✗ Below target - needs optimization")
    end
    
    return result.metrics.efficiency * 100
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    demo_scaling_efficiency()
    
    # Uncomment to test specific configurations
    # test_specific_configuration(50, 2000, 50000)
end