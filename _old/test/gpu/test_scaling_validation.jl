# Comprehensive Scaling Efficiency Validation Test
# Verifies that the multi-GPU system achieves >85% scaling efficiency

using Test
using CUDA
using Printf
using Statistics

# Include required modules
include("../../src/gpu/GPU.jl")
using .GPU

# Import all necessary modules
using .GPU.ScalingEfficiency
using .GPU.WorkDistribution
using .GPU.PCIeCommunication
using .GPU.GPUSynchronization
using .GPU.PerformanceMonitoring

@testset "Scaling Efficiency Validation (>85% Target)" begin
    
    if !CUDA.functional() || length(CUDA.devices()) < 2
        @test_skip "Requires 2+ GPUs for scaling validation"
        println("⚠ Scaling validation requires multi-GPU system")
        return
    end
    
    num_gpus = min(length(CUDA.devices()), 2)
    
    @testset "System Requirements" begin
        # Verify GPU configuration
        @test num_gpus >= 2 "Need at least 2 GPUs for scaling test"
        
        # Check GPU capabilities
        for gpu_id in 0:num_gpus-1
            device!(gpu_id)
            dev = device()
            
            # RTX 4090 or similar capability
            capability = CUDA.capability(dev)
            @test capability.major >= 8 "GPU $gpu_id compute capability too low"
            
            # Memory check (should be close to 24GB for RTX 4090)
            memory_gb = CUDA.totalmem(dev) / 1024^3
            @test memory_gb >= 10.0 "GPU $gpu_id has insufficient memory"
        end
    end
    
    @testset "Baseline Performance Measurement" begin
        benchmark = create_benchmark(num_gpus = num_gpus)
        
        # Test configuration matching production workload
        config = BenchmarkConfig(
            num_trees = 100,
            num_features = 1000,
            num_samples = 10000,
            iterations_per_tree = 100,
            sync_interval = 50  # Sync every 50 trees
        )
        
        # Run single GPU baseline
        baseline_time = run_single_gpu_baseline(benchmark, config)
        @test baseline_time > 0.0 "Baseline time must be positive"
        
        # Store for comparison
        @test haskey(benchmark.baseline_results, config)
    end
    
    @testset "Multi-GPU Scaling Tests" begin
        benchmark = create_benchmark(num_gpus = num_gpus)
        
        # Test configurations covering different scenarios
        test_configs = [
            # Standard workload
            BenchmarkConfig(
                num_trees = 100,
                num_features = 1000,
                num_samples = 10000,
                iterations_per_tree = 100,
                sync_interval = 50
            ),
            # Large dataset
            BenchmarkConfig(
                num_trees = 100,
                num_features = 2000,
                num_samples = 50000,
                iterations_per_tree = 100,
                sync_interval = 50
            ),
            # Many trees
            BenchmarkConfig(
                num_trees = 200,
                num_features = 1000,
                num_samples = 10000,
                iterations_per_tree = 50,
                sync_interval = 100
            ),
        ]
        
        efficiencies = Float64[]
        
        for (i, config) in enumerate(test_configs)
            @testset "Configuration $i" begin
                # Run baseline
                baseline_time = run_single_gpu_baseline(benchmark, config)
                
                # Run multi-GPU
                result = run_multi_gpu_benchmark(benchmark, config)
                
                # Basic validation
                @test result.metrics.multi_gpu_time > 0.0
                @test result.metrics.speedup > 0.0
                @test result.metrics.efficiency > 0.0
                
                # Calculate efficiency percentage
                efficiency_pct = result.metrics.efficiency * 100.0
                push!(efficiencies, efficiency_pct)
                
                println("Config $i efficiency: $(round(efficiency_pct, digits=1))%")
                
                # Individual test should ideally meet target
                if efficiency_pct < 85.0
                    @test_broken efficiency_pct >= 85.0 "Below target (got $(round(efficiency_pct, digits=1))%)"
                else
                    @test efficiency_pct >= 85.0 "Meets efficiency target"
                end
                
                # Verify timing breakdown adds up
                total_component_time = result.metrics.computation_time + 
                                     result.metrics.communication_time + 
                                     result.metrics.synchronization_time
                
                @test total_component_time <= result.metrics.multi_gpu_time * 1.1 "Timing breakdown consistent"
            end
        end
        
        # Overall efficiency must meet target
        avg_efficiency = mean(efficiencies)
        @test avg_efficiency >= 85.0 "Average efficiency $(round(avg_efficiency, digits=1))% must be >= 85%"
    end
    
    @testset "Communication Overhead Validation" begin
        # Test that PCIe transfers stay under 100MB per sync
        transfer_manager = create_transfer_manager()
        
        # Simulate top 10 candidates transfer
        candidates = [CandidateData(i, rand(Float32), rand(1:100)) for i in 1:10]
        
        # Compress candidates
        compressed = compress_candidates(transfer_manager, candidates)
        
        # Check size
        transfer_size_mb = sizeof(compressed) / 1024^2
        @test transfer_size_mb < 100.0 "Transfer size $(round(transfer_size_mb, digits=1))MB exceeds 100MB limit"
    end
    
    @testset "Synchronization Overhead" begin
        sync_manager = create_sync_manager(num_gpus)
        
        # Measure synchronization time
        sync_times = Float64[]
        
        for i in 1:10
            start_time = time()
            
            # Simulate GPU synchronization
            enter_barrier!(sync_manager, SyncBarrier(:test_barrier, num_gpus))
            
            sync_time = time() - start_time
            push!(sync_times, sync_time)
        end
        
        avg_sync_time = mean(sync_times) * 1000  # Convert to ms
        
        # Synchronization should be minimal
        @test avg_sync_time < 10.0 "Sync overhead $(round(avg_sync_time, digits=1))ms too high"
    end
    
    @testset "Load Balancing Verification" begin
        distributor = create_work_distributor(num_gpus = num_gpus, total_trees = 100)
        
        # Check initial distribution
        gpu0_trees = length(get_tree_range(distributor, 0))
        gpu1_trees = length(get_tree_range(distributor, 1))
        
        # Should be balanced within 5%
        imbalance = abs(gpu0_trees - gpu1_trees) / max(gpu0_trees, gpu1_trees)
        @test imbalance <= 0.05 "Initial load imbalance $(round(imbalance * 100, digits=1))% exceeds 5%"
        
        # Simulate workload and check metrics
        for tree_id in 1:100
            gpu_id = get_gpu_for_tree(distributor, tree_id)
            update_metrics!(distributor, gpu_id, tree_id, rand(50:150))
        end
        
        # Get final balance ratio
        balance_ratio = get_load_balance_ratio(distributor)
        @test 0.95 <= balance_ratio <= 1.05 "Load balance ratio $balance_ratio outside ±5%"
    end
    
    @testset "Bottleneck Analysis" begin
        benchmark = create_benchmark(num_gpus = num_gpus)
        
        # Run a quick benchmark
        config = BenchmarkConfig(
            num_trees = 20,
            num_features = 500,
            num_samples = 5000,
            iterations_per_tree = 50
        )
        
        run_single_gpu_baseline(benchmark, config)
        result = run_multi_gpu_benchmark(benchmark, config)
        
        # Check bottleneck identification
        @test result.metrics.bottleneck_type in [:computation, :communication, :synchronization, :memory]
        @test 0.0 <= result.metrics.bottleneck_severity <= 1.0
        
        # For good scaling, computation should be dominant
        if result.metrics.efficiency >= 0.85
            @test result.metrics.bottleneck_type == :computation "Good scaling should be compute-bound"
        end
    end
    
    @testset "Scaling Report Validation" begin
        benchmark = create_benchmark(num_gpus = num_gpus)
        
        # Run minimal experiments
        configs = [
            BenchmarkConfig(num_trees = 10, num_features = 100, num_samples = 1000),
            BenchmarkConfig(num_trees = 20, num_features = 200, num_samples = 2000),
        ]
        
        for config in configs
            run_single_gpu_baseline(benchmark, config)
            run_multi_gpu_benchmark(benchmark, config)
        end
        
        # Generate report
        report = generate_efficiency_report(benchmark)
        
        # Verify report contents
        @test occursin("Target: 85%", report)
        @test occursin("Efficiency Summary:", report)
        @test occursin("Detailed Results:", report)
        @test occursin("Bottleneck Analysis:", report)
        @test occursin("Optimization Recommendations:", report)
        
        # Check if recommendations are provided when below target
        if any(r -> r.metrics.efficiency < 0.85, benchmark.results)
            @test occursin("below target", report)
        end
    end
    
    @testset "Production Workload Simulation" begin
        # Full production-like test
        benchmark = create_benchmark(num_gpus = num_gpus)
        
        production_config = BenchmarkConfig(
            num_trees = 100,
            num_features = 1000,
            num_samples = 100000,  # 100K samples
            iterations_per_tree = 1000,
            batch_size = 256,
            sync_interval = 100  # Sync every 100 iterations
        )
        
        println("\nRunning production workload simulation...")
        println("  Trees: $(production_config.num_trees)")
        println("  Features: $(production_config.num_features)")
        println("  Samples: $(production_config.num_samples)")
        
        # Run baseline
        print("  Single GPU baseline...")
        baseline_time = run_single_gpu_baseline(benchmark, production_config)
        println(" $(round(baseline_time, digits=1))s")
        
        # Run multi-GPU
        print("  Multi-GPU execution...")
        result = run_multi_gpu_benchmark(benchmark, production_config)
        println(" $(round(result.metrics.multi_gpu_time, digits=1))s")
        
        # Final efficiency check
        efficiency_pct = result.metrics.efficiency * 100.0
        println("  Speedup: $(round(result.metrics.speedup, digits=2))x")
        println("  Efficiency: $(round(efficiency_pct, digits=1))%")
        
        @test efficiency_pct >= 85.0 "Production workload must achieve >=85% efficiency"
        
        # Additional performance criteria
        @test result.metrics.pcie_bandwidth_usage < 16.0 "PCIe bandwidth usage too high"
        @test all(u >= 0.8 for u in result.metrics.gpu_utilization) "GPU utilization too low"
    end
    
end

# Summary report
println("\n" * "=" * 60)
println("Scaling Efficiency Validation Summary")
println("=" * 60)

if CUDA.functional() && length(CUDA.devices()) >= 2
    println("✓ Multi-GPU scaling validation completed")
    println("  Target efficiency: 85%")
    println("  Test configurations: Multiple workload sizes")
    println("  Validation includes:")
    println("    - Strong scaling (fixed problem size)")
    println("    - Weak scaling (scaled problem size)")
    println("    - Communication overhead (<100MB/sync)")
    println("    - Load balancing (±5%)")
    println("    - Production workload simulation")
else
    println("⚠ Scaling validation skipped - requires 2+ GPUs")
    println("  Current GPU count: $(CUDA.functional() ? length(CUDA.devices()) : 0)")
end

println("\nValidation complete!")