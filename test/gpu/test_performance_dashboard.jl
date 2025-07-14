using Test
using CUDA
using Dates
using JSON3

# Include the performance dashboard module
include("../../src/gpu/performance_dashboard.jl")

using .PerformanceDashboard

# Create temporary directory for exports
test_export_dir = mktempdir()

@testset "Performance Dashboard Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU performance dashboard tests"
        return
    end
    
    @testset "DashboardConfig Creation" begin
        config = DashboardConfig(
            Int32(100),     # update_interval_ms
            Int32(1000),    # history_size
            true,           # enable_gpu_metrics
            true,           # enable_kernel_metrics
            true,           # enable_tree_metrics
            true,           # enable_alerts
            Int32(60),      # export_interval_s
            test_export_dir # export_path
        )
        
        @test config.update_interval_ms == 100
        @test config.history_size == 1000
        @test config.enable_gpu_metrics == true
        @test config.enable_kernel_metrics == true
        @test config.enable_tree_metrics == true
        @test config.enable_alerts == true
        @test config.export_interval_s == 60
        @test config.export_path == test_export_dir
    end
    
    @testset "AlertThresholds Creation" begin
        thresholds = AlertThresholds(
            20.0,    # gpu_utilization_low
            95.0,    # gpu_utilization_high
            90.0,    # memory_usage_high
            100.0,   # kernel_time_high
            85.0,    # temperature_high
            300.0,   # power_usage_high
            10.0     # bandwidth_low
        )
        
        @test thresholds.gpu_utilization_low == 20.0
        @test thresholds.gpu_utilization_high == 95.0
        @test thresholds.memory_usage_high == 90.0
        @test thresholds.kernel_time_high == 100.0
        @test thresholds.temperature_high == 85.0
        @test thresholds.power_usage_high == 300.0
        @test thresholds.bandwidth_low == 10.0
    end
    
    @testset "MetricRingBuffer" begin
        buffer = MetricRingBuffer(10)
        
        @test buffer.capacity == 10
        @test buffer.head == 1
        @test buffer.size == 0
        
        # Add samples
        for i in 1:5
            sample = MetricSample(now(), METRIC_GPU_UTILIZATION, Float64(i * 10), "%")
            add_sample!(buffer, sample)
        end
        
        @test buffer.size == 5
        
        # Get recent samples
        recent = get_recent_samples(buffer, 3)
        @test length(recent) == 3
        @test recent[1].value == 30.0  # Most recent first after reverse
        @test recent[2].value == 40.0
        @test recent[3].value == 50.0
        
        # Test overflow
        for i in 6:15
            sample = MetricSample(now(), METRIC_GPU_UTILIZATION, Float64(i * 10), "%")
            add_sample!(buffer, sample)
        end
        
        @test buffer.size == 10  # Capped at capacity
    end
    
    @testset "Dashboard Creation" begin
        config = DashboardConfig(
            Int32(100), Int32(100), true, true, true, true, 
            Int32(0), test_export_dir  # Disable auto-export for test
        )
        
        thresholds = AlertThresholds(
            20.0, 95.0, 90.0, 100.0, 85.0, 300.0, 10.0
        )
        
        dashboard = Dashboard(config, thresholds)
        
        @test dashboard.config == config
        @test dashboard.thresholds == thresholds
        @test length(dashboard.metric_buffers) == length(instances(MetricType))
        @test isempty(dashboard.current_metrics)
        @test isempty(dashboard.alerts)
        @test dashboard.total_samples == 0
        @test dashboard.total_alerts == 0
    end
    
    @testset "Metric Collection" begin
        config = DashboardConfig(
            Int32(10), Int32(100), true, true, true, false,
            Int32(0), test_export_dir
        )
        
        thresholds = AlertThresholds(
            20.0, 95.0, 90.0, 100.0, 85.0, 300.0, 10.0
        )
        
        dashboard = Dashboard(config, thresholds)
        
        # Add manual metric
        add_metric!(dashboard, METRIC_GPU_UTILIZATION, 75.0, "%")
        
        @test dashboard.total_samples == 1
        @test dashboard.current_metrics[METRIC_GPU_UTILIZATION] == 75.0
        
        # Collect GPU metrics
        collect_gpu_metrics!(dashboard)
        
        @test dashboard.total_samples > 1
        @test haskey(dashboard.current_metrics, METRIC_MEMORY_USAGE)
        @test haskey(dashboard.current_metrics, METRIC_TEMPERATURE)
        @test haskey(dashboard.current_metrics, METRIC_POWER_USAGE)
        
        # Collect kernel metrics
        collect_kernel_metrics!(dashboard, "test_kernel", 25.5)
        
        @test haskey(dashboard.current_metrics, METRIC_KERNEL_TIME)
        @test dashboard.current_metrics[METRIC_KERNEL_TIME] == 25.5
    end
    
    @testset "Alert System" begin
        config = DashboardConfig(
            Int32(10), Int32(100), true, true, true, true,
            Int32(0), test_export_dir
        )
        
        thresholds = AlertThresholds(
            20.0, 90.0, 80.0, 50.0, 80.0, 250.0, 10.0
        )
        
        dashboard = Dashboard(config, thresholds)
        
        # Trigger low GPU utilization alert
        add_metric!(dashboard, METRIC_GPU_UTILIZATION, 15.0, "%")
        @test dashboard.total_alerts == 1
        @test !isempty(dashboard.alerts)
        @test dashboard.alerts[1].severity == ALERT_WARNING
        
        # Trigger high memory usage alert
        add_metric!(dashboard, METRIC_MEMORY_USAGE, 85.0, "%")
        @test dashboard.total_alerts == 2
        
        # Trigger high kernel time alert
        add_metric!(dashboard, METRIC_KERNEL_TIME, 75.0, "ms")
        @test dashboard.total_alerts == 3
        
        # Get recent alerts
        recent = get_recent_alerts(dashboard, 2)
        @test length(recent) == 2
    end
    
    @testset "Metric Statistics" begin
        config = DashboardConfig(
            Int32(10), Int32(100), false, false, false, false,
            Int32(0), test_export_dir
        )
        
        thresholds = AlertThresholds(
            20.0, 95.0, 90.0, 100.0, 85.0, 300.0, 10.0
        )
        
        dashboard = Dashboard(config, thresholds)
        
        # Add multiple samples
        for i in 1:10
            add_metric!(dashboard, METRIC_GPU_UTILIZATION, Float64(70 + i), "%")
        end
        
        stats = get_metric_stats(dashboard, METRIC_GPU_UTILIZATION, 5)
        
        @test stats["count"] == 5
        @test stats["current"] == 80.0
        @test stats["mean"] ≈ 78.0
        @test stats["min"] == 76.0
        @test stats["max"] == 80.0
        @test stats["unit"] == "%"
    end
    
    @testset "Dashboard Update" begin
        config = DashboardConfig(
            Int32(50), Int32(100), true, false, false, false,
            Int32(0), test_export_dir
        )
        
        thresholds = AlertThresholds(
            20.0, 95.0, 90.0, 100.0, 85.0, 300.0, 10.0
        )
        
        dashboard = Dashboard(config, thresholds)
        
        # Initial update
        update!(dashboard)
        initial_samples = dashboard.total_samples
        
        # Wait less than update interval
        sleep(0.02)
        update!(dashboard)
        @test dashboard.total_samples == initial_samples  # No update
        
        # Wait more than update interval
        sleep(0.06)
        update!(dashboard)
        @test dashboard.total_samples > initial_samples  # Updated
    end
    
    @testset "Metric Export" begin
        config = DashboardConfig(
            Int32(10), Int32(100), true, true, true, true,
            Int32(0), test_export_dir
        )
        
        thresholds = AlertThresholds(
            20.0, 95.0, 90.0, 100.0, 85.0, 300.0, 10.0
        )
        
        dashboard = Dashboard(config, thresholds)
        
        # Add some data
        for i in 1:5
            add_metric!(dashboard, METRIC_GPU_UTILIZATION, Float64(70 + i), "%")
            add_metric!(dashboard, METRIC_MEMORY_USAGE, Float64(50 + i), "%")
        end
        
        # Export metrics
        export_metrics(dashboard)
        
        @test dashboard.export_counter == 1
        
        # Check export file exists
        files = readdir(test_export_dir)
        @test length(files) == 1
        @test endswith(files[1], ".json")
        
        # Read and verify export
        export_file = joinpath(test_export_dir, files[1])
        export_data = JSON3.read(read(export_file, String))
        
        @test haskey(export_data, "timestamp")
        @test export_data["export_id"] == 1
        @test export_data["total_samples"] == 10
        @test haskey(export_data, "metrics")
        @test haskey(export_data["metrics"], string(METRIC_GPU_UTILIZATION))
    end
    
    @testset "Summary Formatting" begin
        config = DashboardConfig(
            Int32(10), Int32(100), true, false, false, false,
            Int32(0), test_export_dir
        )
        
        thresholds = AlertThresholds(
            20.0, 95.0, 90.0, 100.0, 85.0, 300.0, 10.0
        )
        
        dashboard = Dashboard(config, thresholds)
        
        # Add some metrics
        add_metric!(dashboard, METRIC_GPU_UTILIZATION, 75.0, "%")
        add_metric!(dashboard, METRIC_MEMORY_USAGE, 60.0, "%")
        
        summary = format_summary(dashboard)
        
        @test contains(summary, "Performance Dashboard Summary")
        @test contains(summary, "Total Samples: 2")
        @test contains(summary, "METRIC_GPU_UTILIZATION")
        @test contains(summary, "75.00 %")
    end
    
    @testset "Kernel Timing Macro" begin
        config = DashboardConfig(
            Int32(10), Int32(100), false, true, false, false,
            Int32(0), test_export_dir
        )
        
        thresholds = AlertThresholds(
            20.0, 95.0, 90.0, 100.0, 85.0, 300.0, 10.0
        )
        
        dashboard = Dashboard(config, thresholds)
        
        # Use the measurement macro
        elapsed = @measure_kernel dashboard "test_computation" begin
            # Simulate some work
            A = CUDA.rand(100, 100)
            B = CUDA.rand(100, 100)
            C = A * B
            CUDA.synchronize()
        end
        
        @test elapsed > 0.0
        @test haskey(dashboard.current_metrics, METRIC_KERNEL_TIME)
    end
    
    @testset "Kernel Callback" begin
        config = DashboardConfig(
            Int32(10), Int32(100), false, true, false, false,
            Int32(0), test_export_dir
        )
        
        thresholds = AlertThresholds(
            20.0, 95.0, 90.0, 100.0, 85.0, 300.0, 10.0
        )
        
        dashboard = Dashboard(config, thresholds)
        
        # Create callback
        callback = create_kernel_callback(dashboard)
        
        # Call it
        callback("my_kernel", 15.5)
        
        @test dashboard.current_metrics[METRIC_KERNEL_TIME] == 15.5
        @test dashboard.total_samples == 1
    end
end

# Cleanup
rm(test_export_dir, recursive=true)

println("\n✅ Performance dashboard tests completed!")