using Test
using CUDA
using Dates
using Statistics

# Include required modules
include("../../src/gpu/GPU.jl")
using .GPU

# Import from GPU module
using .GPU.PerformanceMonitoring
using .GPU.PerformanceMonitoring: LOG_NONE, LOG_ERROR, LOG_WARN, LOG_INFO, LOG_DEBUG, LOG_TRACE

@testset "Performance Monitoring Tests" begin
    
    @testset "Monitor Creation" begin
        # Test with default parameters
        monitor = create_performance_monitor()
        @test isa(monitor, PerformanceMonitor)
        @test monitor.update_interval == 1.0
        @test monitor.anomaly_detection_enabled
        @test monitor.log_level == LOG_INFO
        @test !monitor.monitoring_active[]
        
        # Test with custom parameters
        monitor2 = create_performance_monitor(
            num_gpus = 2,
            update_interval = 0.5,
            anomaly_detection = false,
            log_level = LOG_DEBUG,
            utilization_threshold = 90.0f0,
            temperature_threshold = 80.0f0
        )
        @test length(monitor2.gpu_metrics) == 2
        @test monitor2.update_interval == 0.5
        @test !monitor2.anomaly_detection_enabled
        @test monitor2.log_level == LOG_DEBUG
        @test monitor2.utilization_threshold == 90.0f0
        
        # Check initialization
        for gpu_id in 0:1
            @test haskey(monitor2.gpu_metrics, gpu_id)
            @test haskey(monitor2.memory_metrics, gpu_id)
            @test haskey(monitor2.historical_metrics, gpu_id)
            @test isempty(monitor2.historical_metrics[gpu_id])
        end
    end
    
    @testset "Kernel Profiling" begin
        monitor = create_performance_monitor(num_gpus=1)
        
        if CUDA.functional()
            # Record kernel execution
            kernel_name = "test_kernel"
            gpu_id = 0
            
            # Start timing
            start_event = record_kernel_start!(monitor, gpu_id, kernel_name)
            
            # Simulate kernel work
            CUDA.@sync begin
                test_data = CUDA.zeros(Float32, 1000)
                test_data .= 1.0f0
            end
            
            # End timing
            record_kernel_end!(monitor, gpu_id, kernel_name)
            
            # Check profile
            stats = get_kernel_stats(monitor, gpu_id, kernel_name)
            @test !isnothing(stats)
            @test stats["name"] == kernel_name
            @test stats["gpu_id"] == gpu_id
            @test stats["call_count"] == 1
            @test stats["total_time_ms"] > 0
            @test stats["avg_time_ms"] > 0
            
            # Record multiple executions
            for i in 1:5
                record_kernel_start!(monitor, gpu_id, kernel_name)
                CUDA.@sync CUDA.zeros(Float32, 100)
                record_kernel_end!(monitor, gpu_id, kernel_name)
            end
            
            stats2 = get_kernel_stats(monitor, gpu_id, kernel_name)
            @test stats2["call_count"] == 6
            @test stats2["avg_time_ms"] == stats2["total_time_ms"] / 6
            @test stats2["min_time_ms"] <= stats2["avg_time_ms"]
            @test stats2["max_time_ms"] >= stats2["avg_time_ms"]
        else
            # Test without CUDA
            kernel_name = "test_kernel"
            gpu_id = 0
            
            record_kernel_start!(monitor, gpu_id, kernel_name)
            record_kernel_end!(monitor, gpu_id, kernel_name)
            
            stats = get_kernel_stats(monitor, gpu_id, kernel_name)
            @test isnothing(stats)
        end
    end
    
    @testset "Memory Transfer Tracking" begin
        monitor = create_performance_monitor(num_gpus=1)
        gpu_id = 0
        
        # Record H2D transfer
        PerformanceMonitoring.record_memory_transfer!(
            monitor, gpu_id, :h2d, 100_000_000, 10.0
        )
        
        mem_metrics = monitor.memory_metrics[gpu_id]
        @test mem_metrics.h2d_transfers == 1
        @test mem_metrics.h2d_bytes == 100_000_000
        @test mem_metrics.peak_bandwidth > 0
        
        # Record D2H transfer
        PerformanceMonitoring.record_memory_transfer!(
            monitor, gpu_id, :d2h, 50_000_000, 5.0
        )
        
        @test mem_metrics.d2h_transfers == 1
        @test mem_metrics.d2h_bytes == 50_000_000
        
        # Record D2D transfer
        PerformanceMonitoring.record_memory_transfer!(
            monitor, gpu_id, :d2d, 200_000_000, 2.0
        )
        
        @test mem_metrics.d2d_transfers == 1
        @test mem_metrics.d2d_bytes == 200_000_000
        @test mem_metrics.peak_bandwidth >= 100.0  # 200MB in 2ms = 100GB/s
    end
    
    @testset "GPU Metrics Update" begin
        monitor = create_performance_monitor(num_gpus=1)
        gpu_id = 0
        
        # Update metrics
        update_gpu_metrics!(monitor, gpu_id)
        
        metrics = monitor.gpu_metrics[gpu_id]
        @test metrics.gpu_id == gpu_id
        @test metrics.timestamp <= now()
        
        if CUDA.functional()
            @test metrics.total_memory > 0
            @test metrics.free_memory >= 0
            @test metrics.used_memory >= 0
            @test 0 <= metrics.memory_utilization <= 100
            
            # Check simulated values are in range
            @test 50 <= metrics.gpu_utilization <= 90
            @test 60 <= metrics.temperature <= 80
            @test 150 <= metrics.power_usage <= 250
        end
        
        # Check historical data
        @test length(monitor.historical_metrics[gpu_id]) == 1
        
        # Update again
        update_gpu_metrics!(monitor, gpu_id)
        @test length(monitor.historical_metrics[gpu_id]) == 2
    end
    
    @testset "Anomaly Detection" begin
        monitor = create_performance_monitor(
            num_gpus = 1,
            utilization_threshold = 80.0f0,
            memory_threshold = 80.0f0,
            temperature_threshold = 75.0f0
        )
        gpu_id = 0
        
        # Set high utilization
        metrics = monitor.gpu_metrics[gpu_id]
        metrics.gpu_utilization = 85.0f0
        metrics.memory_utilization = 85.0f0
        metrics.temperature = 80.0f0
        
        anomalies = detect_anomalies(monitor, gpu_id)
        @test length(anomalies) == 3
        
        # Check anomaly types
        anomaly_types = Set([a.anomaly_type for a in anomalies])
        @test "high_gpu_utilization" in anomaly_types
        @test "high_memory_usage" in anomaly_types
        @test "high_temperature" in anomaly_types
        
        # Test performance drop detection
        # Add historical data
        for i in 1:15
            hist_metrics = GPUMetrics(gpu_id)
            hist_metrics.gpu_utilization = 80.0f0
            push!(monitor.historical_metrics[gpu_id], hist_metrics)
        end
        
        # Current low utilization
        metrics.gpu_utilization = 30.0f0
        anomalies2 = detect_anomalies(monitor, gpu_id)
        
        perf_drop = findfirst(a -> a.anomaly_type == "performance_drop", anomalies2)
        @test !isnothing(perf_drop)
    end
    
    @testset "Performance Summary" begin
        monitor = create_performance_monitor(num_gpus=2)
        
        # Add some data
        for gpu_id in keys(monitor.gpu_metrics)
            # Update metrics
            update_gpu_metrics!(monitor, gpu_id)
            
            # Add kernel profiles (only for GPU 0 or if no CUDA)
            if gpu_id == 0 || !CUDA.functional()
                for i in 1:3
                    kernel_name = "kernel_$i"
                    PerformanceMonitoring.record_kernel_start!(monitor, gpu_id, kernel_name)
                    PerformanceMonitoring.record_kernel_end!(monitor, gpu_id, kernel_name)
                end
            end
            
            # Add memory transfers
            PerformanceMonitoring.record_memory_transfer!(
                monitor, gpu_id, :h2d, 50_000_000, 5.0
            )
        end
        
        # Get summary
        summary = get_performance_summary(monitor)
        
        @test haskey(summary, "monitoring_duration")
        @test haskey(summary, "total_kernels_profiled")
        @test haskey(summary, "anomalies_detected")
        @test haskey(summary, "gpu_summaries")
        
        # Check GPU summaries
        @test haskey(summary["gpu_summaries"], 0)
        @test haskey(summary["gpu_summaries"], 1)
        
        gpu0_summary = summary["gpu_summaries"][0]
        @test haskey(gpu0_summary, "current")
        @test haskey(gpu0_summary, "memory_transfers")
        
        # Check memory transfer summary
        mem_summary = gpu0_summary["memory_transfers"]
        @test mem_summary["h2d_count"] == 1
        @test mem_summary["total_gb"] ≈ 0.05  # 50MB
    end
    
    @testset "Logging System" begin
        # Create temporary log file
        temp_file = tempname()
        
        monitor = create_performance_monitor(
            num_gpus = 1,
            log_level = LOG_DEBUG,
            log_file = temp_file
        )
        
        # Test different log levels
        set_log_level!(monitor, LOG_WARN)
        @test monitor.log_level == LOG_WARN
        
        # Log some performance data
        test_data = Dict{String, Any}(
            "gpu_id" => 0,
            "kernel" => "test",
            "time_ms" => 5.5
        )
        log_performance_data(monitor, test_data)
        
        # Stop monitoring to close file
        stop_monitoring!(monitor)
        
        # Check log file exists and has content
        @test isfile(temp_file)
        @test filesize(temp_file) > 0
        
        # Cleanup
        rm(temp_file)
    end
    
    @testset "Monitoring Integration" begin
        monitor = create_performance_monitor(
            num_gpus = 1,
            update_interval = 0.1,
            anomaly_detection = true
        )
        
        # Start monitoring
        start_monitoring!(monitor)
        @test monitor.monitoring_active[]
        @test haskey(monitor.monitor_tasks, 0)
        
        # Let it run briefly
        sleep(0.3)
        
        # Check that metrics have been updated
        @test length(monitor.historical_metrics[0]) > 0
        
        # Stop monitoring
        stop_monitoring!(monitor)
        @test !monitor.monitoring_active[]
        @test isempty(monitor.monitor_tasks)
    end
    
    @testset "Metrics Reset and Export" begin
        monitor = create_performance_monitor(num_gpus=1)
        
        # Add some data
        update_gpu_metrics!(monitor, 0)
        PerformanceMonitoring.record_kernel_start!(monitor, 0, "test")
        PerformanceMonitoring.record_kernel_end!(monitor, 0, "test")
        monitor.anomalies_detected[] = 5
        
        # Reset metrics
        reset_metrics!(monitor)
        
        @test monitor.total_kernels_profiled[] == 0
        @test monitor.anomalies_detected[] == 0
        @test isempty(monitor.kernel_profiles)
        @test isempty(monitor.historical_metrics[0])
        
        # Test export
        temp_file = tempname() * ".json"
        
        # Add some data again
        update_gpu_metrics!(monitor, 0)
        export_metrics(monitor, temp_file)
        
        @test isfile(temp_file)
        @test filesize(temp_file) > 0
        
        # Cleanup
        rm(temp_file)
    end
    
    @testset "Multi-GPU Support" begin
        num_gpus = CUDA.functional() ? min(2, length(CUDA.devices())) : 2
        monitor = create_performance_monitor(num_gpus=num_gpus)
        
        # Update metrics for all GPUs
        for gpu_id in 0:(num_gpus-1)
            update_gpu_metrics!(monitor, gpu_id)
        end
        
        # Get all metrics
        all_metrics = get_all_metrics(monitor)
        @test length(all_metrics) == num_gpus
        
        for gpu_id in 0:(num_gpus-1)
            @test haskey(all_metrics, gpu_id)
            @test all_metrics[gpu_id].gpu_id == gpu_id
        end
    end
    
end

# Print summary
println("\nPerformance Monitoring Test Summary:")
println("====================================")
if CUDA.functional()
    println("✓ CUDA functional - Full performance monitoring tests executed")
    println("  Kernel profiling validated")
    println("  Memory transfer tracking tested")
    println("  Anomaly detection operational")
    println("  Real-time monitoring verified")
else
    println("⚠ CUDA not functional - Basic performance monitoring tests only")
end
println("\nAll performance monitoring tests completed!")