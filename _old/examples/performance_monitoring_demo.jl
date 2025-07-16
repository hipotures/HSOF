#!/usr/bin/env julia

# Performance Monitoring Demo
# Demonstrates real-time GPU performance tracking and anomaly detection

using CUDA
using Printf
using Random
using Dates
using Statistics

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

# Load GPU modules
include("../src/gpu/GPU.jl")
using .GPU

# Import from GPU module
using .GPU.PerformanceMonitoring
using .GPU.PerformanceMonitoring: LOG_NONE, LOG_ERROR, LOG_WARN, LOG_INFO, LOG_DEBUG, LOG_TRACE

"""
Simulate intensive GPU workload
"""
function gpu_workload(size::Int, iterations::Int)
    if !CUDA.functional()
        println("  Simulating GPU workload on CPU...")
        data = rand(Float32, size, size)
        for i in 1:iterations
            data = data * 0.999f0 .+ 0.001f0
        end
        return sum(data)
    end
    
    # GPU workload
    data = CUDA.rand(Float32, size, size)
    
    for i in 1:iterations
        CUDA.@sync begin
            data = data * 0.999f0 .+ 0.001f0
        end
    end
    
    return sum(data)
end

"""
Simulate memory-intensive operations
"""
function memory_stress_test(monitor::PerformanceMonitor, gpu_id::Int, size_mb::Int)
    if !CUDA.functional()
        println("  Simulating memory transfers...")
        return
    end
    
    prev_device = CUDA.device()
    try
        CUDA.device!(gpu_id)
        
        # H2D transfer
        start_time = time()
        host_data = rand(Float32, size_mb * 1024 * 256)  # size_mb * 1MB
        device_data = CuArray(host_data)
        h2d_time = (time() - start_time) * 1000  # ms
        
        PerformanceMonitoring.record_memory_transfer!(
            monitor, gpu_id, :h2d, sizeof(host_data), h2d_time
        )
        
        # D2H transfer
        start_time = time()
        result = Array(device_data)
        d2h_time = (time() - start_time) * 1000  # ms
        
        PerformanceMonitoring.record_memory_transfer!(
            monitor, gpu_id, :d2h, sizeof(result), d2h_time
        )
        
        # Cleanup
        CUDA.unsafe_free!(device_data)
        
    finally
        CUDA.device!(prev_device)
    end
end

"""
Run kernel benchmarks
"""
function benchmark_kernels(monitor::PerformanceMonitor, gpu_id::Int)
    kernel_names = ["matmul", "reduction", "elementwise", "convolution"]
    sizes = [256, 512, 1024]
    
    for kernel_name in kernel_names
        for size in sizes
            # Record kernel execution
            record_kernel_start!(monitor, gpu_id, kernel_name)
            
            # Simulate different kernel workloads
            if kernel_name == "matmul"
                gpu_workload(size, 5)
            elseif kernel_name == "reduction"
                gpu_workload(size ÷ 2, 10)
            elseif kernel_name == "elementwise"
                gpu_workload(size * 2, 2)
            else  # convolution
                gpu_workload(size ÷ 4, 20)
            end
            
            record_kernel_end!(monitor, gpu_id, kernel_name)
        end
    end
end

"""
Demo performance monitoring system
"""
function demo_performance_monitoring()
    println("GPU Performance Monitoring Demo")
    println("=" ^ 60)
    
    # Create performance monitor
    num_gpus = CUDA.functional() ? min(2, length(CUDA.devices())) : 1
    monitor = create_performance_monitor(
        num_gpus = num_gpus,
        update_interval = 0.5,
        anomaly_detection = true,
        log_level = LOG_INFO,
        utilization_threshold = 75.0f0,
        memory_threshold = 80.0f0,
        temperature_threshold = 80.0f0
    )
    
    println("\nPerformance Monitor Configuration:")
    println("  Number of GPUs: $num_gpus")
    println("  Update interval: $(monitor.update_interval)s")
    println("  Anomaly detection: $(monitor.anomaly_detection_enabled)")
    println("  Utilization threshold: $(monitor.utilization_threshold)%")
    
    # Demo 1: Basic Monitoring
    println("\n1. Starting Real-Time Monitoring:")
    println("-" ^ 40)
    
    start_monitoring!(monitor)
    println("✓ Performance monitoring started")
    
    # Let monitoring establish baseline
    sleep(2)
    
    # Demo 2: Kernel Profiling
    println("\n2. Kernel Execution Profiling:")
    println("-" ^ 40)
    
    println("Running kernel benchmarks...")
    for gpu_id in 0:(num_gpus-1)
        println("\nGPU $gpu_id:")
        benchmark_kernels(monitor, gpu_id)
        
        # Show kernel stats
        for kernel in ["matmul", "reduction", "elementwise", "convolution"]
            stats = get_kernel_stats(monitor, gpu_id, kernel)
            if !isnothing(stats)
                println("  $kernel: $(stats["call_count"]) calls, " *
                       "avg=$(round(stats["avg_time_ms"], digits=2))ms, " *
                       "total=$(round(stats["total_time_ms"], digits=2))ms")
            end
        end
    end
    
    # Demo 3: Memory Transfer Monitoring
    println("\n3. Memory Bandwidth Testing:")
    println("-" ^ 40)
    
    for gpu_id in 0:(num_gpus-1)
        println("\nGPU $gpu_id memory transfers:")
        for size_mb in [10, 50, 100]
            print("  Testing $(size_mb)MB transfer... ")
            memory_stress_test(monitor, gpu_id, size_mb)
            
            mem_metrics = monitor.memory_metrics[gpu_id]
            println("Peak: $(round(mem_metrics.peak_bandwidth, digits=2))GB/s")
        end
    end
    
    # Demo 4: Live Metrics Display
    println("\n4. Live Performance Metrics:")
    println("-" ^ 40)
    
    println("\nMonitoring for 10 seconds...")
    start_time = time()
    
    while time() - start_time < 10
        # Simulate varying workloads
        workload_type = rand(1:3)
        
        for gpu_id in 0:(num_gpus-1)
            if workload_type == 1
                # Light workload
                record_kernel_start!(monitor, gpu_id, "light_kernel")
                gpu_workload(256, 2)
                record_kernel_end!(monitor, gpu_id, "light_kernel")
            elseif workload_type == 2
                # Heavy workload
                record_kernel_start!(monitor, gpu_id, "heavy_kernel")
                gpu_workload(1024, 10)
                record_kernel_end!(monitor, gpu_id, "heavy_kernel")
            else
                # Memory intensive
                memory_stress_test(monitor, gpu_id, 25)
            end
        end
        
        # Display current metrics
        print("\r")
        for gpu_id in 0:(num_gpus-1)
            metrics = get_gpu_metrics(monitor, gpu_id)
            if !isnothing(metrics)
                print("GPU$gpu_id: $(round(metrics.gpu_utilization, digits=1))% util, " *
                      "$(round(metrics.memory_utilization, digits=1))% mem, " *
                      "$(round(metrics.temperature, digits=1))°C  ")
            end
        end
        
        sleep(1)
    end
    println()
    
    # Demo 5: Performance Summary
    println("\n5. Performance Summary Report:")
    println("-" ^ 40)
    
    summary = get_performance_summary(monitor)
    
    println("\nOverall Statistics:")
    println("  Monitoring duration: $(round(summary["monitoring_duration"], digits=1))s")
    println("  Total kernels profiled: $(summary["total_kernels_profiled"])")
    println("  Anomalies detected: $(summary["anomalies_detected"])")
    
    for (gpu_id, gpu_summary) in summary["gpu_summaries"]
        println("\nGPU $gpu_id Summary:")
        
        # Current metrics
        current = gpu_summary["current"]
        println("  Current state:")
        println("    GPU utilization: $(round(current["gpu_utilization"], digits=1))%")
        println("    Memory utilization: $(round(current["memory_utilization"], digits=1))%")
        println("    Temperature: $(round(current["temperature"], digits=1))°C")
        println("    Power: $(round(current["power_usage"], digits=1))W")
        
        # Average metrics
        if haskey(gpu_summary, "average")
            avg = gpu_summary["average"]
            println("  Average values:")
            println("    GPU utilization: $(round(avg["gpu_utilization"], digits=1))%")
            println("    Memory utilization: $(round(avg["memory_utilization"], digits=1))%")
            println("    Temperature: $(round(avg["temperature"], digits=1))°C")
        end
        
        # Memory transfers
        if haskey(gpu_summary, "memory_transfers")
            mem = gpu_summary["memory_transfers"]
            println("  Memory transfers:")
            println("    H2D: $(mem["h2d_count"]) transfers")
            println("    D2H: $(mem["d2h_count"]) transfers")
            println("    Total: $(round(mem["total_gb"], digits=2))GB")
            println("    Peak bandwidth: $(round(mem["peak_bandwidth_gb_s"], digits=2))GB/s")
        end
        
        # Top kernels
        if haskey(gpu_summary, "top_kernels")
            println("  Top kernels by time:")
            for (i, kernel) in enumerate(gpu_summary["top_kernels"][1:min(3, end)])
                println("    $i. $(kernel["name"]): " *
                       "$(round(kernel["total_time_ms"], digits=2))ms total, " *
                       "$(kernel["call_count"]) calls")
            end
        end
    end
    
    # Demo 6: Anomaly Detection
    println("\n6. Performance Anomaly Detection:")
    println("-" ^ 40)
    
    # Force some anomalies for demonstration
    if num_gpus > 0
        metrics = monitor.gpu_metrics[0]
        
        # High utilization
        metrics.gpu_utilization = 95.0f0
        metrics.memory_utilization = 85.0f0
        metrics.temperature = 82.0f0
        
        anomalies = detect_anomalies(monitor, 0)
        
        if !isempty(anomalies)
            println("\nDetected anomalies:")
            for anomaly in anomalies
                severity_symbol = Dict(
                    "low" => "ℹ",
                    "medium" => "⚠",
                    "high" => "🚨"
                )[anomaly.severity]
                
                println("  $severity_symbol $(anomaly.anomaly_type) on GPU $(anomaly.gpu_id): " *
                       "$(anomaly.description)")
                println("    Value: $(round(anomaly.metric_value, digits=2)), " *
                       "Threshold: $(round(anomaly.threshold, digits=2))")
            end
        else
            println("  No anomalies detected")
        end
    end
    
    # Demo 7: Export Metrics
    println("\n7. Exporting Performance Data:")
    println("-" ^ 40)
    
    export_file = joinpath(@__DIR__, "performance_report.json")
    export_metrics(monitor, export_file)
    println("✓ Performance data exported to: $export_file")
    
    # Stop monitoring
    stop_monitoring!(monitor)
    println("\n✓ Performance monitoring stopped")
    
    # Demo 8: Best Practices
    println("\n8. Performance Monitoring Best Practices:")
    println("-" ^ 40)
    println("• Set appropriate update intervals based on workload")
    println("• Configure anomaly thresholds for your hardware")
    println("• Use kernel profiling to identify bottlenecks")
    println("• Monitor memory bandwidth for optimization opportunities")
    println("• Export metrics regularly for historical analysis")
    println("• Integrate with existing logging infrastructure")
    println("• Use performance data to guide optimization efforts")
    
    # Cleanup
    if isfile(export_file)
        rm(export_file)
    end
    
    println("\n" * "=" ^ 60)
    println("Demo completed!")
end

# Utility function to display real-time dashboard
function display_dashboard(monitor::PerformanceMonitor, duration::Int = 30)
    println("\nReal-Time Performance Dashboard")
    println("=" * 40)
    println("Press Ctrl+C to stop\n")
    
    start_time = time()
    
    try
        while time() - start_time < duration
            # Clear previous lines (simple approach)
            print("\033[$(length(monitor.gpu_metrics) + 4)A")
            
            # Header
            println("Time: $(Dates.format(now(), "HH:MM:SS"))  " *
                   "Uptime: $(round(time() - start_time, digits=1))s")
            println("-" * 40)
            
            # GPU metrics
            for (gpu_id, metrics) in sort(collect(monitor.gpu_metrics), by=x->x[1])
                util_bar = "█" ^ Int(round(metrics.gpu_utilization / 5))
                mem_bar = "█" ^ Int(round(metrics.memory_utilization / 5))
                
                println("GPU $gpu_id: $(rpad(util_bar, 20)) $(round(metrics.gpu_utilization, digits=1))% util")
                println("  Mem: $(rpad(mem_bar, 20)) $(round(metrics.memory_utilization, digits=1))% " *
                       "($(round(metrics.used_memory / 1e9, digits=1))GB)")
                println("  Temp: $(round(metrics.temperature, digits=1))°C  " *
                       "Power: $(round(metrics.power_usage, digits=0))W")
                println()
            end
            
            sleep(1)
        end
    catch e
        if !isa(e, InterruptException)
            rethrow(e)
        end
    end
    
    println("\nDashboard stopped.")
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    demo_performance_monitoring()
end