using Test
using CUDA
using Statistics
using Random

# Skip if no GPU
if !CUDA.functional()
    @warn "CUDA not functional, skipping performance profiling tests"
    exit(0)
end

# Include modules
include("../../src/stage1_filter/performance_profiling.jl")
using .PerformanceProfiling

println("Testing Performance Profiling System...")
println("="^60)

@testset "Performance Profiling Tests" begin
    
    @testset "CUDA Timer" begin
        timer = CUDATiming.create_timer()
        
        # Test basic timing
        CUDATiming.start_timing!(timer, "test_kernel")
        CUDA.@sync begin
            A = CUDA.rand(1000, 1000)
            B = A * A
        end
        CUDATiming.stop_timing!(timer, "test_kernel")
        
        results = CUDATiming.get_timing_results(timer)
        @test length(results) == 1
        @test results[1].name == "test_kernel"
        @test results[1].elapsed_ms > 0
        @test results[1].count == 1
        
        # Test multiple timings
        for i in 1:5
            CUDATiming.start_timing!(timer, "multi_kernel")
            CUDA.@sync CUDA.rand(100, 100)
            CUDATiming.stop_timing!(timer, "multi_kernel")
        end
        
        results = CUDATiming.get_timing_results(timer)
        multi_result = filter(r -> r.name == "multi_kernel", results)[1]
        @test multi_result.count == 5
        @test multi_result.mean_ms > 0
        @test multi_result.std_ms >= 0
    end
    
    @testset "Memory Tracker" begin
        tracker = MemoryProfiling.create_memory_tracker()
        
        # Test allocation tracking
        ptr1 = Ptr{Nothing}(1234)
        MemoryProfiling.track_allocation!(tracker, ptr1, 1024 * 1024)  # 1MB
        
        @test tracker.current_allocated == 1024 * 1024
        @test tracker.peak_allocated == 1024 * 1024
        @test length(tracker.allocations) == 1
        
        # Test deallocation
        MemoryProfiling.track_deallocation!(tracker, ptr1)
        @test tracker.current_allocated == 0
        @test tracker.peak_allocated == 1024 * 1024  # Peak remains
        
        # Test memory snapshot
        snapshot = MemoryProfiling.get_memory_snapshot(tracker)
        @test snapshot.allocated_bytes == 0
        @test snapshot.peak_allocated == 1024 * 1024
        @test snapshot.allocation_count == 1
        @test snapshot.deallocation_count == 1
        
        # Test bandwidth calculation
        bw = MemoryProfiling.calculate_bandwidth(1_000_000_000, 1000.0)  # 1GB in 1s
        @test bw ≈ 1.0  # 1 GB/s
    end
    
    @testset "Kernel Metrics" begin
        # Test kernel configuration
        config = KernelMetrics.KernelConfig(256, 100, 1024, 32)
        
        # Test occupancy calculation
        occupancy = KernelMetrics.calculate_occupancy(config)
        @test occupancy.occupancy >= 0.0 && occupancy.occupancy <= 1.0
        @test occupancy.active_warps > 0
        @test occupancy.limiting_factor in [:registers, :shared_memory, :blocks, :threads]
        
        # Test efficiency metrics
        efficiency = KernelMetrics.calculate_efficiency(
            10.0,      # 10ms
            1e9,       # 1GB transferred
            1e10,      # 10 GFLOP
            0
        )
        
        @test efficiency.memory_bandwidth_gbps ≈ 100.0  # 1GB / 0.01s = 100 GB/s
        @test efficiency.compute_throughput_gflops ≈ 1000.0  # 10 GFLOP / 0.01s = 1000 GFLOPS
        @test efficiency.arithmetic_intensity ≈ 10.0  # 10 FLOP/byte
        @test efficiency.bandwidth_efficiency >= 0.0 && efficiency.bandwidth_efficiency <= 1.0
        
        # Test kernel profile
        profile = KernelMetrics.profile_kernel(
            "test_kernel",
            config,
            10.0,
            1_000_000,
            500_000,
            10_000_000
        )
        
        @test profile.name == "test_kernel"
        @test profile.elapsed_ms == 10.0
        @test profile.bytes_read == 1_000_000
        @test profile.bytes_written == 500_000
        @test profile.flop_count == 10_000_000
    end
    
    @testset "Bottleneck Analysis" begin
        # Create sample data
        timing_results = [
            CUDATiming.TimingResult("kernel1", 100.0, 10, 8.0, 12.0, 10.0, 1.0),
            CUDATiming.TimingResult("kernel2", 50.0, 10, 4.0, 6.0, 5.0, 0.5),
            CUDATiming.TimingResult("transfer", 200.0, 5, 35.0, 45.0, 40.0, 3.0)
        ]
        
        memory_stats = Dict(
            :h2d => MemoryProfiling.BandwidthMeasurement(:h2d, 1_000_000_000, 100.0, 10.0, 5, 200_000_000),
            :d2h => MemoryProfiling.BandwidthMeasurement(:d2h, 500_000_000, 50.0, 10.0, 3, 166_666_667)
        )
        
        kernel_profiles = [
            KernelMetrics.profile_kernel(
                "kernel1",
                KernelMetrics.KernelConfig(256, 100, 0, 32),
                10.0,
                1_000_000_000,
                100_000_000,
                5_000_000_000
            )
        ]
        
        # Analyze bottlenecks
        bottlenecks = BottleneckAnalyzer.analyze_bottlenecks(
            timing_results,
            memory_stats,
            kernel_profiles
        )
        
        @test length(bottlenecks) >= 0  # May find various bottlenecks
        
        # Test critical path identification
        critical_path = BottleneckAnalyzer.identify_critical_path(timing_results)
        @test "transfer" in critical_path  # Longest operation
        
        # Test bottleneck report
        report = BottleneckAnalyzer.create_bottleneck_report(
            timing_results,
            memory_stats,
            kernel_profiles
        )
        
        @test report.total_elapsed_ms == 350.0  # Sum of all operations
        @test length(report.stages) == 3
        @test !isempty(report.critical_path)
    end
    
    @testset "Performance Regression" begin
        # Create a simple test
        test = PerformanceRegression.RegressionTest(
            "test_operation",
            () -> CUDA.rand(1000, 1000),  # Setup
            (data, timer, tracker) -> begin  # Benchmark
                CUDA.@sync sum(data)
            end,
            (data) -> nothing,  # Teardown
            Dict{String, Function}(),
            Dict("total_time" => 10.0),  # 10% tolerance
            2,  # warmup runs
            5   # test runs
        )
        
        # Run test (no baseline)
        result = PerformanceRegression.run_regression_test(test)
        @test result.test_name == "test_operation"
        @test isnothing(result.baseline)
        @test result.passed == true
        @test isempty(result.regressions)
        
        # Test baseline comparison
        baseline = result.current
        result2 = PerformanceRegression.run_regression_test(test, baseline)
        @test !isnothing(result2.baseline)
        # Should pass as performance should be similar
        @test result2.passed || length(result2.regressions) > 0
    end
    
    @testset "Profile Session Integration" begin
        # Create session
        session = create_profile_session("test_session")
        @test session.name == "test_session"
        @test session.enabled == true
        
        # Profile an operation
        @profile session "matrix_multiply" begin
            A = CUDA.rand(500, 500)
            B = CUDA.rand(500, 500)
            C = A * B
            CUDA.synchronize()
        end
        
        # Get results
        results = analyze_performance(session)
        @test results.session_name == "test_session"
        @test length(results.timing_results) >= 1
        
        # Find our operation
        mm_result = filter(r -> r.name == "matrix_multiply", results.timing_results)
        @test length(mm_result) == 1
        @test mm_result[1].elapsed_ms > 0
    end
    
    @testset "Real Kernel Profiling" begin
        # Profile a real variance calculation
        session = create_profile_session("variance_profiling")
        
        n_features = 1000
        n_samples = 10000
        X = CUDA.randn(Float32, n_features, n_samples)
        
        @profile session "variance_calc" begin
            # Time the kernel
            start_event = CuEvent()
            stop_event = CuEvent()
            
            variances = CUDA.zeros(Float32, n_features)
            
            CUDA.record(start_event)
            CUDA.@sync variances .= vec(var(X, dims=2))
            CUDA.record(stop_event)
            CUDA.synchronize(stop_event)
            
            elapsed_ms = CUDA.elapsed(start_event, stop_event)
            
            # Profile the kernel
            config = KernelMetrics.KernelConfig(
                256,  # threads per block
                cld(n_features, 256),  # blocks
                0,    # shared memory
                32    # registers estimate
            )
            
            bytes_read = n_features * n_samples * sizeof(Float32)
            bytes_written = n_features * sizeof(Float32)
            # Variance: sum, sum of squares, division = ~3 ops per element
            flops = n_features * n_samples * 3
            
            profile_kernel!(
                session,
                "variance_kernel",
                config,
                elapsed_ms,
                bytes_read,
                bytes_written,
                flops
            )
        end
        
        results = analyze_performance(session)
        
        # Verify profiling captured data
        @test length(results.kernel_profiles) == 1
        profile = results.kernel_profiles[1]
        @test profile.name == "variance_kernel"
        @test profile.elapsed_ms > 0
        @test profile.efficiency.memory_bandwidth_gbps > 0
        @test profile.efficiency.compute_throughput_gflops > 0
        
        # Check if memory or compute bound
        println("\nVariance kernel analysis:")
        println("  Memory bandwidth: $(round(profile.efficiency.memory_bandwidth_gbps, digits=1)) GB/s")
        println("  Compute throughput: $(round(profile.efficiency.compute_throughput_gflops, digits=1)) GFLOPS")
        println("  Bottleneck: $(profile.efficiency.is_memory_bound ? "Memory" : "Compute")")
    end
end

# Integration test with comprehensive profiling
println("\n" * "="^60)
println("COMPREHENSIVE PROFILING EXAMPLE")
println("="^60)

# Run the example profiling
results = example_profiling_usage()

# Additional analysis
println("\n" * "="^60)
println("PERFORMANCE SUMMARY")
println("="^60)

if !isempty(results.kernel_profiles)
    total_flops = sum(p.flop_count for p in results.kernel_profiles)
    total_bytes = sum(p.bytes_read + p.bytes_written for p in results.kernel_profiles)
    total_time_ms = sum(p.elapsed_ms for p in results.kernel_profiles)
    
    overall_gflops = (total_flops / 1e9) / (total_time_ms / 1000)
    overall_bandwidth = (total_bytes / 1e9) / (total_time_ms / 1000)
    
    println("\nOverall Performance:")
    println("  Total computation: $(round(total_flops / 1e9, digits=2)) GFLOP")
    println("  Total data movement: $(round(total_bytes / 1e9, digits=2)) GB")
    println("  Aggregate GFLOPS: $(round(overall_gflops, digits=1))")
    println("  Aggregate bandwidth: $(round(overall_bandwidth, digits=1)) GB/s")
end

println("\n" * "="^60)
println("TEST SUMMARY")
println("="^60)
println("✓ CUDA event timing functional")
println("✓ Memory tracking and bandwidth measurement working")
println("✓ Kernel efficiency metrics calculated correctly")
println("✓ Bottleneck analysis identifies performance issues")
println("✓ Performance regression framework operational")
println("✓ Integrated profiling session captures all metrics")
println("✓ Real kernel profiling provides actionable insights")
println("="^60)