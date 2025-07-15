"""
Test suite for Performance Profiling Framework
Testing comprehensive profiling system, CUDA event timers, and performance analysis
"""

using Test
using CUDA
using Statistics
using Random

# Include required modules
include("../../src/metamodel/performance_profiling.jl")
include("../../src/metamodel/neural_architecture.jl")

using .PerformanceProfiling
using .NeuralArchitecture

@testset "Performance Profiling Tests" begin
    
    Random.seed!(42)  # For reproducible tests
    
    @testset "Configuration Tests" begin
        config = create_profiling_config()
        
        @test config.enable_cuda_events == true
        @test config.enable_nsight_markers == true
        @test config.enable_memory_profiling == true
        @test config.profile_inference_ops == true
        @test config.profile_training_ops == true
        @test config.sampling_frequency == 10.0f0
        @test config.warmup_iterations == 50
        @test config.measurement_iterations == 100
        @test config.export_format == "json"
        @test config.inference_latency_target_ms == 1.0f0
        @test config.training_throughput_target == 1000.0f0
        
        println("✓ Configuration tests passed")
    end
    
    @testset "CUDA Event Timer Tests" begin
        if CUDA.functional()
            # Test basic timer functionality
            timer = CUDAEventTimer()
            @test timer.is_recording == false
            
            # Test start/stop timing
            start_timer!(timer)
            @test timer.is_recording == true
            
            # Simulate some work
            x = CUDA.randn(Float32, 1000, 1000)
            y = x * x
            CUDA.synchronize()
            
            duration = stop_timer!(timer)
            @test timer.is_recording == false
            @test duration >= 0.0f0
            
            println("✓ CUDA event timer tests passed - duration: $(round(duration, digits=3)) ms")
        else
            println("⚠ Skipping CUDA timer tests - CUDA not functional")
        end
    end
    
    @testset "Memory Profiler Tests" begin
        memory_profiler = MemoryProfiler()
        
        @test memory_profiler.initial_gpu_memory > 0
        @test memory_profiler.initial_cpu_memory > 0
        @test isempty(memory_profiler.gpu_allocations)
        @test isempty(memory_profiler.cpu_allocations)
        
        # Take memory snapshots
        for i in 1:5
            take_memory_snapshot!(memory_profiler)
            
            # Allocate some memory to change usage
            if CUDA.functional()
                temp_array = CUDA.randn(Float32, 100, 100)
            end
        end
        
        @test length(memory_profiler.gpu_allocations) == 5
        @test length(memory_profiler.cpu_allocations) == 5
        @test length(memory_profiler.timestamps) == 5
        
        println("✓ Memory profiler tests passed")
    end
    
    @testset "Nsight Marker Tests" begin
        # Test marker creation
        marker_blue = NsightMarker("test_operation", :blue)
        @test marker_blue.name == "test_operation"
        @test marker_blue.color == 0x0000FF
        
        marker_red = NsightMarker("training_op", :red)
        @test marker_red.color == 0xFF0000
        
        # Test marker push/pop (should not crash)
        push_nsight_marker(marker_blue)
        pop_nsight_marker()
        
        println("✓ Nsight marker tests passed")
    end
    
    @testset "Performance Profiler Initialization Tests" begin
        config = create_profiling_config(
            export_path = "test_profiling_results",
            warmup_iterations = 5,
            measurement_iterations = 10
        )
        
        profiler = initialize_profiler(config)
        
        @test profiler.config === config
        @test length(profiler.cuda_timers) > 0
        @test length(profiler.nsight_markers) > 0
        @test !isnothing(profiler.memory_profiler)
        @test isempty(profiler.results)
        @test profiler.total_measurements == 0
        @test profiler.failed_measurements == 0
        
        println("✓ Profiler initialization tests passed")
    end
    
    @testset "Operation Profiling Tests" begin
        config = create_profiling_config(
            export_path = "test_profiling_results",
            warmup_iterations = 2,
            measurement_iterations = 5
        )
        
        profiler = initialize_profiler(config)
        
        # Test simple operation profiling
        result, perf_result = profile_operation!(profiler, "test_operation", () -> begin
            sleep(0.001)  # 1ms sleep
            return 42
        end, category="general")
        
        @test result == 42
        @test perf_result.operation_name == "test_operation"
        @test perf_result.duration_ms > 0.0f0
        @test profiler.total_measurements == 1
        @test profiler.failed_measurements == 0
        @test haskey(profiler.current_measurements, "test_operation")
        
        # Test operation that throws error
        result_error, perf_result_error = profile_operation!(profiler, "error_operation", () -> begin
            error("Test error")
        end)
        
        @test result_error === nothing
        @test profiler.failed_measurements == 1
        
        println("✓ Operation profiling tests passed")
    end
    
    if CUDA.functional()
        @testset "Model Inference Benchmarking Tests" begin
            config = create_profiling_config(
                warmup_iterations = 3,
                measurement_iterations = 5
            )
            
            profiler = initialize_profiler(config)
            
            # Create test model
            model_config = create_metamodel_config(input_dim = 100, hidden_dims = [64, 32, 16])
            model = create_metamodel(model_config) |> gpu
            
            # Test small batch sizes
            batch_sizes = [1, 8, 16]
            results = benchmark_inference(profiler, model, batch_sizes)
            
            @test length(results) == length(batch_sizes)
            
            for batch_size in batch_sizes
                @test haskey(results, batch_size)
                batch_result = results[batch_size]
                
                @test haskey(batch_result, "mean_duration_ms")
                @test haskey(batch_result, "throughput_samples_per_sec")
                @test haskey(batch_result, "per_sample_latency_ms")
                
                @test batch_result["mean_duration_ms"] > 0.0f0
                @test batch_result["throughput_samples_per_sec"] > 0.0f0
                @test batch_result["per_sample_latency_ms"] > 0.0f0
            end
            
            # Verify throughput scaling with batch size
            throughput_1 = results[1]["throughput_samples_per_sec"]
            throughput_16 = results[16]["throughput_samples_per_sec"]
            @test throughput_16 > throughput_1  # Larger batches should have higher throughput
            
            println("✓ Inference benchmarking tests passed")
        end
        
        @testset "Memory Bandwidth Benchmarking Tests" begin
            config = create_profiling_config(
                warmup_iterations = 2,
                measurement_iterations = 3
            )
            
            profiler = initialize_profiler(config)
            
            # Test small data sizes for quick testing
            bandwidth_results = benchmark_memory_bandwidth(profiler)
            
            @test length(bandwidth_results) > 0
            
            for (size_mb, results) in bandwidth_results
                @test haskey(results, "h2d_bandwidth_mbps")
                @test haskey(results, "d2h_bandwidth_mbps")
                @test haskey(results, "d2d_bandwidth_mbps")
                
                @test results["h2d_bandwidth_mbps"] > 0.0f0
                @test results["d2h_bandwidth_mbps"] > 0.0f0
                @test results["d2d_bandwidth_mbps"] > 0.0f0
                
                # D2D should typically be faster than H2D/D2H
                @test results["d2d_bandwidth_mbps"] >= results["h2d_bandwidth_mbps"]
            end
            
            println("✓ Memory bandwidth benchmarking tests passed")
        end
        
        @testset "Performance Report Generation Tests" begin
            config = create_profiling_config(
                export_path = "test_profiling_results",
                export_format = "json"
            )
            
            profiler = initialize_profiler(config)
            
            # Generate some test measurements
            for i in 1:10
                profile_operation!(profiler, "test_op_$i", () -> begin
                    x = CUDA.randn(Float32, 100, 100)
                    y = x * x
                    return sum(y)
                end)
            end
            
            # Generate report
            report = generate_performance_report(profiler)
            
            @test haskey(report, "profiling_config")
            @test haskey(report, "overall_statistics")
            @test haskey(report, "operation_statistics")
            @test haskey(report, "gpu_info")
            @test haskey(report, "timestamp")
            
            # Check overall statistics
            overall_stats = report["overall_statistics"]
            @test overall_stats["total_measurements"] == 10
            @test overall_stats["success_rate"] == 1.0f0
            @test overall_stats["total_runtime_seconds"] > 0.0
            
            # Check operation statistics
            op_stats = report["operation_statistics"]
            @test length(op_stats) == 10  # 10 different operations
            
            for (op_name, stats) in op_stats
                @test haskey(stats, "mean_ms")
                @test haskey(stats, "std_ms")
                @test haskey(stats, "min_ms")
                @test haskey(stats, "max_ms")
                @test haskey(stats, "p95_ms")
                @test stats["count"] == 1.0f0  # Each operation run once
            end
            
            # Test report export
            export_path = export_performance_report(profiler, "test_report")
            @test isfile(export_path)
            
            println("✓ Performance report generation tests passed")
        end
        
        @testset "Regression Testing Tests" begin
            config = create_profiling_config(
                regression_testing = true,
                warmup_iterations = 2,
                measurement_iterations = 3
            )
            
            profiler = initialize_profiler(config)
            
            # Run baseline measurement
            profile_operation!(profiler, "regression_test", () -> begin
                sleep(0.001)  # 1ms baseline
                return 1
            end)
            
            baseline_duration = profiler.baseline_results["regression_test"]
            @test baseline_duration > 0.0f0
            
            # Run measurement that should trigger regression warning
            # (We can't easily test the actual warning, but we can test the logic)
            profile_operation!(profiler, "regression_test", () -> begin
                sleep(0.002)  # 2ms - should be flagged as regression
                return 2
            end)
            
            # Check that baseline is still recorded
            @test haskey(profiler.baseline_results, "regression_test")
            
            println("✓ Regression testing tests passed")
        end
        
        @testset "Real-time Dashboard Data Tests" begin
            config = create_profiling_config()
            profiler = initialize_profiler(config)
            
            # Generate some measurements
            for i in 1:5
                profile_operation!(profiler, "dashboard_test", () -> begin
                    x = CUDA.randn(Float32, 50, 50)
                    return sum(x)
                end)
            end
            
            dashboard_data = get_realtime_dashboard_data(profiler)
            
            @test haskey(dashboard_data, "runtime_seconds")
            @test haskey(dashboard_data, "total_measurements")
            @test haskey(dashboard_data, "measurements_per_second")
            @test haskey(dashboard_data, "success_rate")
            @test haskey(dashboard_data, "current_measurements")
            @test haskey(dashboard_data, "gpu_memory_available_gb")
            @test haskey(dashboard_data, "gpu_memory_total_gb")
            
            @test dashboard_data["total_measurements"] == 5
            @test dashboard_data["success_rate"] == 1.0f0
            @test dashboard_data["gpu_memory_available_gb"] > 0.0
            @test dashboard_data["gpu_memory_total_gb"] > 0.0
            
            println("✓ Real-time dashboard data tests passed")
        end
    else
        println("⚠ Skipping GPU-dependent tests - CUDA not functional")
    end
    
    @testset "Utility Function Tests" begin
        # Test GPU utilization estimation
        low_util = PerformanceProfiling.estimate_gpu_utilization(0.05f0)  # 0.05ms
        high_util = PerformanceProfiling.estimate_gpu_utilization(1.0f0)   # 1.0ms
        
        @test low_util < high_util
        @test 0.0f0 <= low_util <= 100.0f0
        @test 0.0f0 <= high_util <= 100.0f0
        
        # Test memory bandwidth estimation
        bandwidth_zero = PerformanceProfiling.estimate_memory_bandwidth(0, 1.0f0)
        @test bandwidth_zero == 0.0f0
        
        bandwidth_normal = PerformanceProfiling.estimate_memory_bandwidth(1024*1024*1024, 1000.0f0)  # 1GB in 1 second
        @test bandwidth_normal ≈ 1.0f0  # Should be ~1 GB/s
        
        println("✓ Utility function tests passed")
    end
    
    @testset "Cleanup Tests" begin
        config = create_profiling_config()
        profiler = initialize_profiler(config)
        
        # Add some measurements
        for i in 1:3
            profile_operation!(profiler, "cleanup_test", () -> sleep(0.001))
        end
        
        # Test cleanup (should not crash)
        cleanup_profiler!(profiler)
        
        println("✓ Cleanup tests passed")
    end
end

println("Performance Profiling tests completed successfully!")