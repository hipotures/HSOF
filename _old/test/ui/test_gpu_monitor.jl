using Test
using Dates
using CUDA
using Statistics

# Include the modules
include("../../src/ui/gpu_monitor.jl")
using .GPUMonitor

@testset "GPU Monitor Tests" begin
    
    @testset "Configuration Tests" begin
        # Test default configuration
        config = GPUMonitorConfig()
        @test config.poll_interval_ms == 100
        @test config.history_size == 600
        @test config.history_duration_sec == 60
        @test config.enable_caching == true
        @test config.cache_duration_ms == 50
        @test config.simulate_metrics == true
        
        # Test custom configuration
        custom_config = GPUMonitorConfig(
            poll_interval_ms = 200,
            history_size = 300,
            history_duration_sec = 30,
            enable_caching = false,
            cache_duration_ms = 0,
            simulate_metrics = false
        )
        @test custom_config.poll_interval_ms == 200
        @test custom_config.history_size == 300
        @test custom_config.history_duration_sec == 30
        @test custom_config.enable_caching == false
        @test custom_config.cache_duration_ms == 0
        @test custom_config.simulate_metrics == false
        
        # Test invalid configurations
        @test_throws AssertionError GPUMonitorConfig(poll_interval_ms = 0)
        @test_throws AssertionError GPUMonitorConfig(history_size = 0)
        @test_throws AssertionError GPUMonitorConfig(history_duration_sec = 0)
        @test_throws AssertionError GPUMonitorConfig(cache_duration_ms = -1)
    end
    
    @testset "GPU Monitor Creation" begin
        monitor = create_gpu_monitor()
        @test isa(monitor, GPUMonitorState)
        @test monitor.is_monitoring == false
        @test isnothing(monitor.monitor_task)
        
        # Check device detection
        if CUDA.functional()
            @test length(monitor.devices) > 0
            @test monitor.selected_device >= 0
        else
            @test length(monitor.devices) == 0
            @test monitor.selected_device == -1
        end
    end
    
    @testset "Metrics Collection" begin
        config = GPUMonitorConfig(simulate_metrics = true)
        monitor = create_gpu_monitor(config)
        
        if !isempty(monitor.devices)
            device_id = monitor.devices[1]
            
            # Collect metrics
            metrics = GPUMonitor.collect_gpu_metrics!(monitor, device_id)
            @test isa(metrics, GPUMetrics)
            @test metrics.gpu_id == device_id
            
            if CUDA.functional()
                @test metrics.is_available == true
                @test metrics.memory_total > 0
                @test metrics.memory_used >= 0
                @test metrics.memory_used <= metrics.memory_total
                @test 0 <= metrics.memory_percent <= 100
                
                # Check simulated metrics
                @test 0 <= metrics.utilization <= 100
                @test 30 <= metrics.temperature <= 95
                @test 50 <= metrics.power_draw <= 450
                @test 1200 <= metrics.clock_speed <= 2500
                @test 0 <= metrics.fan_speed <= 100
            end
            
            # Check that metrics were stored
            @test haskey(monitor.current_metrics, device_id)
            @test length(monitor.metrics_history[device_id]) > 0
        end
    end
    
    @testset "Caching Tests" begin
        config = GPUMonitorConfig(
            enable_caching = true,
            cache_duration_ms = 100
        )
        monitor = create_gpu_monitor(config)
        
        if !isempty(monitor.devices)
            device_id = monitor.devices[1]
            
            # First collection
            metrics1 = GPUMonitor.collect_gpu_metrics!(monitor, device_id)
            time1 = metrics1.timestamp
            
            # Immediate second collection should return cached
            sleep(0.01)  # 10ms
            metrics2 = GPUMonitor.collect_gpu_metrics!(monitor, device_id)
            @test metrics2.timestamp == time1  # Same timestamp means cached
            
            # Wait for cache expiry
            sleep(0.15)  # 150ms > 100ms cache duration
            metrics3 = GPUMonitor.collect_gpu_metrics!(monitor, device_id)
            @test metrics3.timestamp > time1  # New timestamp means fresh data
        end
    end
    
    @testset "History Management" begin
        config = GPUMonitorConfig(
            history_size = 10,
            history_duration_sec = 1
        )
        monitor = create_gpu_monitor(config)
        
        if !isempty(monitor.devices)
            device_id = monitor.devices[1]
            
            # Fill history beyond size limit
            for i in 1:15
                GPUMonitor.collect_gpu_metrics!(monitor, device_id)
                sleep(0.01)
            end
            
            history = get_historical_metrics(monitor, device_id)
            @test length(history) <= 10  # Should be trimmed to history_size
            
            # Test time-based trimming
            sleep(1.1)  # Wait for history duration to expire
            GPUMonitor.collect_gpu_metrics!(monitor, device_id)
            
            history = get_historical_metrics(monitor, device_id)
            # All old entries should be removed
            @test all(m -> m.timestamp > now() - Second(2), history)
        end
    end
    
    @testset "GPU Selection" begin
        monitor = create_gpu_monitor()
        
        if length(monitor.devices) >= 2
            # Test selecting valid GPU
            @test select_gpu!(monitor, monitor.devices[2]) == true
            @test monitor.selected_device == monitor.devices[2]
            
            # Test selecting invalid GPU
            @test select_gpu!(monitor, 999) == false
            @test monitor.selected_device == monitor.devices[2]  # Unchanged
        end
    end
    
    @testset "Available GPUs" begin
        monitor = create_gpu_monitor()
        gpu_list = get_available_gpus(monitor)
        
        @test isa(gpu_list, Vector)
        @test length(gpu_list) == length(monitor.devices)
        
        for gpu_info in gpu_list
            @test haskey(gpu_info, :id)
            @test haskey(gpu_info, :name)
            @test isa(gpu_info.id, Int)
            @test isa(gpu_info.name, String)
        end
    end
    
    @testset "Background Monitoring" begin
        monitor = create_gpu_monitor()
        
        # Test starting monitoring
        start_monitoring!(monitor)
        @test monitor.is_monitoring == true
        @test !isnothing(monitor.monitor_task)
        @test istaskstarted(monitor.monitor_task)
        
        # Let it run for a bit
        sleep(0.3)
        
        # Test stopping monitoring
        stop_monitoring!(monitor)
        @test monitor.is_monitoring == false
        @test !isnothing(monitor.monitor_task) && istaskdone(monitor.monitor_task)
        
        # Test double start/stop
        @test_logs (:warn, "Monitoring is not active") stop_monitoring!(monitor)
        
        start_monitoring!(monitor)
        @test_logs (:warn, "Monitoring is already active") start_monitoring!(monitor)
        stop_monitoring!(monitor)
    end
    
    @testset "Metrics Reset" begin
        monitor = create_gpu_monitor()
        
        # Add some data
        update_metrics!(monitor)
        
        # Reset
        reset_metrics!(monitor)
        
        @test isempty(monitor.current_metrics)
        for history in values(monitor.metrics_history)
            @test isempty(history)
        end
        for count in values(monitor.error_count)
            @test count == 0
        end
    end
    
    @testset "Error Handling" begin
        # Create monitor with no GPUs to test error paths
        config = GPUMonitorConfig(simulate_metrics = false)
        monitor = GPUMonitorState(config)
        
        # Force an invalid device
        push!(monitor.devices, 999)
        monitor.selected_device = 999
        monitor.metrics_history[999] = GPUMetrics[]
        monitor.error_count[999] = 0
        
        # Try to collect metrics for invalid device
        metrics = GPUMonitor.collect_gpu_metrics!(monitor, 999)
        @test !metrics.is_available
        @test !isnothing(metrics.error_message)
        @test monitor.error_count[999] > 0
    end
    
    @testset "Simulated Metrics Behavior" begin
        config = GPUMonitorConfig(simulate_metrics = true)
        monitor = create_gpu_monitor(config)
        
        if !isempty(monitor.devices)
            device_id = monitor.devices[1]
            
            # Collect multiple samples
            samples = GPUMetrics[]
            for i in 1:10
                metrics = GPUMonitor.collect_gpu_metrics!(monitor, device_id)
                push!(samples, metrics)
                sleep(0.05)
            end
            
            # Check that values vary but stay in reasonable ranges
            utilizations = [s.utilization for s in samples]
            temperatures = [s.temperature for s in samples]
            
            @test minimum(utilizations) >= 0
            @test maximum(utilizations) <= 100
            @test std(utilizations) > 0  # Should have some variation
            
            @test minimum(temperatures) >= 30
            @test maximum(temperatures) <= 95
            @test std(temperatures) > 0  # Should have some variation
            
            # Check correlations (temperature should correlate with utilization)
            if length(samples) > 2
                utils = [s.utilization for s in samples]
                temps = [s.temperature for s in samples]
                # Very rough correlation check
                high_util_samples = filter(s -> s.utilization > 70, samples)
                low_util_samples = filter(s -> s.utilization < 50, samples)
                
                if !isempty(high_util_samples) && !isempty(low_util_samples)
                    avg_high_temp = Statistics.mean(s.temperature for s in high_util_samples)
                    avg_low_temp = Statistics.mean(s.temperature for s in low_util_samples)
                    @test avg_high_temp > avg_low_temp  # Higher util should mean higher temp
                end
            end
        end
    end
end

println("All GPU monitor tests passed! ✓")