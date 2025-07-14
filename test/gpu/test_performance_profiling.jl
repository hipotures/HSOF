using Test
using CUDA
using JSON3

# Include modules
include("../../src/gpu/mcts_gpu.jl")

using .MCTSGPU
using .MCTSGPU.PerformanceProfiling: start_timing!, end_timing!, calculate_occupancy, update_metrics!, check_regression!

@testset "Performance Profiling Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping performance profiling tests"
        return
    end
    
    @testset "Basic Profiling" begin
        engine = MCTSGPUEngine(
            max_iterations = 100,
            block_size = 128,
            grid_size = 32
        )
        
        # Check profiler exists
        @test !isnothing(engine.profiler)
        @test !isnothing(engine.monitor)
        @test !isnothing(engine.regression_detector)
        
        # Check device properties
        @test engine.profiler.num_sms > 0
        @test engine.profiler.max_threads_per_sm > 0
    end
    
    @testset "Timing Operations" begin
        engine = MCTSGPUEngine()
        
        # Test timing
        start_timing!(engine.profiler, "test_operation")
        
        # Simulate some work
        function dummy_kernel()
            x = threadIdx().x
            y = x * 2
            return nothing
        end
        
        CUDA.@sync @cuda threads=256 blocks=1 dummy_kernel()
        
        duration = end_timing!(engine.profiler, "test_operation")
        
        @test duration > 0
        @test length(engine.profiler.kernel_durations) == 1
    end
    
    @testset "Occupancy Calculation" begin
        engine = MCTSGPUEngine()
        
        # Test occupancy calculation
        occupancy = calculate_occupancy(
            engine.profiler,
            Int32(256),  # block size
            Int32(32),   # registers per thread
            Int32(0)     # shared memory
        )
        
        @test 0.0f0 <= occupancy <= 1.0f0
        @test length(engine.profiler.kernel_occupancy) == 1
    end
    
    @testset "Realtime Monitoring" begin
        engine = MCTSGPUEngine()
        
        # Update metrics
        update_metrics!(
            engine.monitor,
            75.0f0,   # GPU utilization
            150.0f0,  # Bandwidth GB/s
            1000.0f0  # Throughput
        )
        
        @test engine.monitor.gpu_utilization[1] ≈ 75.0f0
        @test engine.monitor.memory_bandwidth[1] ≈ 150.0f0
        @test engine.monitor.kernel_throughput[1] ≈ 1000.0f0
        
        # Check aggregates
        @test engine.monitor.avg_gpu_util[] ≈ 75.0f0
        @test engine.monitor.avg_bandwidth[] ≈ 150.0f0
        @test engine.monitor.peak_bandwidth[] ≈ 150.0f0
    end
    
    @testset "Regression Detection" begin
        engine = MCTSGPUEngine()
        
        # Set baseline
        regression1 = check_regression!(
            engine.regression_detector,
            "test_metric",
            100.0f0
        )
        @test regression1 == false
        
        # Check no regression
        regression2 = check_regression!(
            engine.regression_detector,
            "test_metric",
            95.0f0
        )
        @test regression2 == false
        
        # Check regression detected
        regression3 = check_regression!(
            engine.regression_detector,
            "test_metric",
            80.0f0  # 20% drop
        )
        @test regression3 == true
        @test length(engine.regression_detector.alerts) == 1
    end
    
    @testset "Performance Report Generation" begin
        engine = MCTSGPUEngine()
        
        # Initialize and add some data
        initialize!(engine)
        
        # Add timing data
        start_timing!(engine.profiler, "test_kernel")
        sleep(0.01)
        end_timing!(engine.profiler, "test_kernel")
        
        # Add metrics
        update_metrics!(engine.monitor, 80.0f0, 200.0f0, 500.0f0)
        
        # Generate report
        report = get_performance_report(engine)
        
        @test haskey(report, "kernel_stats")
        @test haskey(report, "occupancy_stats")
        @test haskey(report, "system_metrics")
        @test haskey(report, "device_info")
        @test haskey(report, "mcts_metrics")
        
        # Check device info
        @test haskey(report["device_info"], "name")
        @test haskey(report["device_info"], "compute_capability")
        @test report["device_info"]["num_sms"] > 0
    end
    
    @testset "JSON Export" begin
        engine = MCTSGPUEngine()
        
        # Create test directory
        test_dir = "test_performance_reports"
        mkpath(test_dir)
        
        # Export metrics
        filename = joinpath(test_dir, "test_report.json")
        exported_file = export_performance_metrics(engine, filename)
        
        @test isfile(exported_file)
        
        # Read and verify JSON
        json_content = read(exported_file, String)
        parsed = JSON3.read(json_content)
        
        @test haskey(parsed, "device_info")
        @test haskey(parsed, "mcts_metrics")
        
        # Cleanup
        rm(test_dir, recursive=true)
    end
    
    @testset "Integration with MCTS" begin
        engine = MCTSGPUEngine(
            max_iterations = 10,
            block_size = 64
        )
        
        # Initialize and run briefly
        initialize!(engine, [1, 2, 3])
        start!(engine)
        
        # Let it run for a moment
        sleep(0.1)
        
        # Stop and check metrics
        stop!(engine)
        
        report = get_performance_report(engine)
        
        # Should have collected some metrics
        @test !isempty(engine.profiler.kernel_durations)
        @test engine.stats.nodes_allocated > 0
        
        # GPU utilization should be tracked
        @test haskey(report["system_metrics"], "avg_gpu_utilization")
    end
end

println("\n✅ Performance profiling tests passed!")