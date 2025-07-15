using Test
using CUDA
using Statistics

# Include required modules
include("../../src/gpu/GPU.jl")
using .GPU

# Import from GPU module
using .GPU.ScalingEfficiency

@testset "Scaling Efficiency Tests" begin
    
    @testset "Benchmark Configuration" begin
        # Test default configuration
        config = BenchmarkConfig()
        @test config.num_trees == 100
        @test config.num_features == 1000
        @test config.num_samples == 10000
        @test config.iterations_per_tree == 1000
        @test config.batch_size == 256
        @test config.sync_interval == 100
        
        # Test custom configuration
        config2 = BenchmarkConfig(
            num_trees = 50,
            num_features = 500,
            num_samples = 5000,
            iterations_per_tree = 500,
            batch_size = 128,
            sync_interval = 50
        )
        @test config2.num_trees == 50
        @test config2.num_features == 500
        @test config2.num_samples == 5000
        @test config2.iterations_per_tree == 500
        @test config2.batch_size == 128
        @test config2.sync_interval == 50
    end
    
    @testset "Benchmark Creation" begin
        benchmark = create_benchmark()
        @test isa(benchmark, ScalingBenchmark)
        @test benchmark.num_gpus == 2
        @test isempty(benchmark.configs)
        @test isempty(benchmark.results)
        @test !benchmark.enable_profiling
        
        # Test with profiling enabled
        benchmark2 = create_benchmark(num_gpus = 4, enable_profiling = true)
        @test benchmark2.num_gpus == 4
        @test benchmark2.enable_profiling
    end
    
    @testset "Scaling Metrics" begin
        metrics = ScalingMetrics()
        @test metrics.single_gpu_time == 0.0
        @test metrics.multi_gpu_time == 0.0
        @test metrics.speedup == 0.0
        @test metrics.efficiency == 0.0
        @test isempty(metrics.gpu_utilization)
        @test metrics.bottleneck_type == :none
        @test metrics.bottleneck_severity == 0.0
    end
    
    if CUDA.functional() && length(CUDA.devices()) >= 1
        @testset "Single GPU Baseline" begin
            benchmark = create_benchmark()
            config = BenchmarkConfig(
                num_trees = 10,  # Small for testing
                num_features = 100,
                num_samples = 1000,
                iterations_per_tree = 10
            )
            
            # Run baseline
            baseline_time = run_single_gpu_baseline(benchmark, config)
            
            @test baseline_time > 0.0
            @test haskey(benchmark.baseline_results, config)
            @test benchmark.baseline_results[config] == baseline_time
        end
        
        @testset "Multi-GPU Benchmark" begin
            num_gpus = min(length(CUDA.devices()), 2)
            benchmark = create_benchmark(num_gpus = num_gpus)
            
            config = BenchmarkConfig(
                num_trees = 10,
                num_features = 100,
                num_samples = 1000,
                iterations_per_tree = 10,
                sync_interval = 5
            )
            
            # Run baseline first
            baseline_time = run_single_gpu_baseline(benchmark, config)
            
            # Run multi-GPU benchmark
            result = run_multi_gpu_benchmark(benchmark, config)
            
            @test isa(result, BenchmarkResult)
            @test result.config == config
            @test result.metrics.multi_gpu_time > 0.0
            @test length(result.gpu_info) == num_gpus
            
            # Check metrics calculation
            @test result.metrics.speedup > 0.0
            @test result.metrics.efficiency > 0.0
            @test result.metrics.efficiency <= 1.0  # Can't exceed 100%
            
            # Check timing breakdown
            @test result.metrics.computation_time > 0.0
            @test result.metrics.communication_time >= 0.0
            @test result.metrics.synchronization_time >= 0.0
            
            # Check utilization metrics
            @test length(result.metrics.gpu_utilization) == num_gpus
            @test all(0.0 <= u <= 1.0 for u in result.metrics.gpu_utilization)
            
            # Check bottleneck analysis
            @test result.metrics.bottleneck_type in [:computation, :communication, :synchronization, :memory]
            @test 0.0 <= result.metrics.bottleneck_severity <= 1.0
        end
        
        @testset "Efficiency Calculation" begin
            metrics = ScalingMetrics()
            metrics.single_gpu_time = 10.0
            metrics.multi_gpu_time = 6.0
            
            # Calculate efficiency for 2 GPUs
            ScalingEfficiency.calculate_efficiency_metrics!(metrics, 2)
            
            @test metrics.speedup ≈ 10.0 / 6.0
            @test metrics.efficiency ≈ metrics.speedup / 2.0
            @test metrics.strong_scaling_efficiency == metrics.efficiency
            
            # Test perfect scaling
            metrics2 = ScalingMetrics()
            metrics2.single_gpu_time = 10.0
            metrics2.multi_gpu_time = 5.0  # Perfect 2x speedup
            metrics2.communication_time = 0.0
            metrics2.synchronization_time = 0.0
            
            ScalingEfficiency.calculate_efficiency_metrics!(metrics2, 2)
            @test metrics2.speedup ≈ 2.0
            @test metrics2.efficiency ≈ 1.0  # 100% efficiency
        end
        
        @testset "Bottleneck Identification" begin
            metrics = ScalingMetrics()
            metrics.multi_gpu_time = 10.0
            metrics.computation_time = 7.0
            metrics.communication_time = 2.0
            metrics.synchronization_time = 0.5
            metrics.memory_transfer_time = 0.5
            
            ScalingEfficiency.identify_bottleneck!(metrics)
            
            @test metrics.bottleneck_type == :computation
            @test metrics.bottleneck_severity ≈ 0.7
            
            # Test communication bottleneck
            metrics2 = ScalingMetrics()
            metrics2.multi_gpu_time = 10.0
            metrics2.computation_time = 3.0
            metrics2.communication_time = 5.0
            metrics2.synchronization_time = 1.0
            metrics2.memory_transfer_time = 1.0
            
            ScalingEfficiency.identify_bottleneck!(metrics2)
            
            @test metrics2.bottleneck_type == :communication
            @test metrics2.bottleneck_severity ≈ 0.5
        end
        
        @testset "Scaling Experiments" begin
            # Test with very small workload for quick testing
            benchmark = create_benchmark()
            
            # Override configs with tiny workload
            benchmark.configs = [
                BenchmarkConfig(
                    num_trees = 4,
                    num_features = 50,
                    num_samples = 100,
                    iterations_per_tree = 5
                ),
                BenchmarkConfig(
                    num_trees = 4,
                    num_features = 100,
                    num_samples = 200,
                    iterations_per_tree = 5
                )
            ]
            
            # Run mini experiments
            for config in benchmark.configs
                run_single_gpu_baseline(benchmark, config)
                run_multi_gpu_benchmark(benchmark, config)
            end
            
            @test length(benchmark.results) == 2
            
            # Check efficiency calculation
            for config in benchmark.configs
                efficiency = calculate_scaling_efficiency(benchmark, config)
                @test efficiency > 0.0
                @test efficiency <= 100.0
            end
            
            # Test bottleneck identification
            bottlenecks = identify_bottlenecks(benchmark)
            @test length(bottlenecks) == 2
            @test all(b[2] >= 0.0 && b[2] <= 1.0 for b in bottlenecks)
        end
        
        @testset "Efficiency Report Generation" begin
            benchmark = create_benchmark()
            
            # Add some test results
            config = BenchmarkConfig(num_trees = 4, num_features = 50, num_samples = 100)
            run_single_gpu_baseline(benchmark, config)
            run_multi_gpu_benchmark(benchmark, config)
            
            # Generate report
            report = generate_efficiency_report(benchmark)
            
            @test isa(report, String)
            @test !isempty(report)
            @test occursin("Scaling Efficiency Validation Report", report)
            @test occursin("Efficiency Summary:", report)
            @test occursin("Target: 85%", report)
            @test occursin("Detailed Results:", report)
            @test occursin("Bottleneck Analysis:", report)
            @test occursin("Optimization Recommendations:", report)
        end
        
        @testset "Target Efficiency Validation" begin
            # Test scenario meeting 85% target
            benchmark = create_benchmark()
            
            # Create a result that meets target
            config = BenchmarkConfig()
            metrics = ScalingMetrics()
            metrics.single_gpu_time = 10.0
            metrics.multi_gpu_time = 5.88  # ~85% efficiency for 2 GPUs
            ScalingEfficiency.calculate_efficiency_metrics!(metrics, 2)
            
            result = BenchmarkResult(
                config,
                metrics,
                now(),
                [],
                Dict{String, Vector{Float64}}(),
                String[]
            )
            
            push!(benchmark.results, result)
            benchmark.baseline_results[config] = metrics.single_gpu_time
            
            efficiency = calculate_scaling_efficiency(benchmark, config)
            @test efficiency >= 85.0
            
            # Test scenario not meeting target
            metrics2 = ScalingMetrics()
            metrics2.single_gpu_time = 10.0
            metrics2.multi_gpu_time = 7.0  # ~71% efficiency for 2 GPUs
            ScalingEfficiency.calculate_efficiency_metrics!(metrics2, 2)
            
            @test metrics2.efficiency < 0.85
        end
        
        @testset "Progress Callback" begin
            benchmark = create_benchmark()
            
            # Track progress
            progress_log = []
            benchmark.progress_callback = (tree, total, gpu) -> begin
                push!(progress_log, (tree, total, gpu))
            end
            
            config = BenchmarkConfig(num_trees = 4, num_features = 50, num_samples = 100)
            run_single_gpu_baseline(benchmark, config)
            
            @test !isempty(progress_log)
            @test progress_log[end][1] == 4  # Last tree
            @test progress_log[end][2] == 4  # Total trees
            @test progress_log[end][3] == "baseline"
        end
        
    else
        @testset "CPU-only Fallback" begin
            @test_skip "CUDA not functional or insufficient GPUs - skipping GPU tests"
            
            # Test that module loads without GPU
            benchmark = create_benchmark()
            @test isa(benchmark, ScalingBenchmark)
            
            # Test configuration creation
            config = BenchmarkConfig()
            @test isa(config, BenchmarkConfig)
            
            # Test metrics structure
            metrics = ScalingMetrics()
            @test isa(metrics, ScalingMetrics)
        end
    end
    
    @testset "Edge Cases" begin
        benchmark = create_benchmark()
        
        # Test with no results
        bottlenecks = identify_bottlenecks(benchmark)
        @test isempty(bottlenecks)
        
        # Test efficiency calculation with missing baseline
        config = BenchmarkConfig()
        efficiency = calculate_scaling_efficiency(benchmark, config)
        @test efficiency == 0.0
        
        # Test report with no results
        report = generate_efficiency_report(benchmark)
        @test occursin("Scaling Efficiency Validation Report", report)
    end
    
end

# Print summary
println("\nScaling Efficiency Test Summary:")
println("==================================")
if CUDA.functional()
    num_gpus = length(CUDA.devices())
    println("✓ CUDA functional - Scaling efficiency tests executed")
    println("  GPUs detected: $num_gpus")
    
    if num_gpus >= 2
        println("  Multi-GPU scaling tests completed")
        println("  Efficiency validation performed")
    else
        println("  Single GPU only - limited multi-GPU testing")
    end
    
    println("  Bottleneck analysis validated")
    println("  Report generation tested")
else
    println("⚠ CUDA not functional - CPU simulation tests only")
end
println("\nAll scaling efficiency tests completed!")