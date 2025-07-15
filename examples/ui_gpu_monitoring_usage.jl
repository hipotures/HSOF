# Example usage of GPU Monitoring with Console Dashboard Integration

using Dates
using Printf

# Include the modules
include("../src/ui/gpu_monitor.jl")
using .GPUMonitor

include("../src/ui/gpu_dashboard_integration.jl")
using .GPUDashboardIntegration
using .GPUDashboardIntegration.ConsoleDashboard

"""
Simple GPU monitoring demo
"""
function simple_gpu_monitoring_demo()
    println("Simple GPU Monitoring Demo")
    println("========================")
    
    # Create GPU monitor with default config
    monitor = create_gpu_monitor()
    
    # Show available GPUs
    gpus = get_available_gpus(monitor)
    println("\nAvailable GPUs:")
    for gpu in gpus
        println("  GPU $(gpu.id): $(gpu.name)")
    end
    
    if isempty(gpus)
        println("  No GPUs detected!")
        return
    end
    
    println("\nCollecting GPU metrics...")
    
    # Collect and display metrics for 10 iterations
    for i in 1:10
        println("\n--- Iteration $i ---")
        
        # Update metrics for all GPUs
        update_metrics!(monitor)
        
        # Display metrics for each GPU
        for gpu in gpus
            metrics = get_current_metrics(monitor, gpu.id)
            if !isnothing(metrics)
                println(metrics)
            end
        end
        
        sleep(1.0)  # 1 second between updates
    end
    
    # Show history summary
    println("\n\nHistory Summary:")
    for gpu in gpus
        history = get_historical_metrics(monitor, gpu.id)
        if !isempty(history)
            valid_metrics = filter(m -> m.is_available, history)
            if !isempty(valid_metrics)
                avg_util = mean(m.utilization for m in valid_metrics)
                avg_temp = mean(m.temperature for m in valid_metrics)
                avg_power = mean(m.power_draw for m in valid_metrics)
                
                println("\nGPU $(gpu.id) Averages:")
                println("  Utilization: $(round(avg_util, digits=1))%")
                println("  Temperature: $(round(avg_temp, digits=1))°C")
                println("  Power Draw: $(round(avg_power, digits=1))W")
            end
        end
    end
end

"""
Advanced monitoring demo with configuration
"""
function advanced_gpu_monitoring_demo()
    println("Advanced GPU Monitoring Demo")
    println("==========================")
    
    # Create custom configuration
    config = GPUMonitorConfig(
        poll_interval_ms = 50,      # Fast polling
        history_size = 1200,        # 60 seconds at 50ms
        history_duration_sec = 60,
        enable_caching = true,
        cache_duration_ms = 25,     # Cache for half the poll interval
        simulate_metrics = true     # Use simulation
    )
    
    monitor = create_gpu_monitor(config)
    
    # Start background monitoring
    println("Starting background monitoring...")
    start_monitoring!(monitor)
    
    # Monitor for 15 seconds
    start_time = now()
    duration = Second(15)
    
    println("\nMonitoring for 15 seconds...")
    println("Press Ctrl+C to stop early")
    
    try
        while now() - start_time < duration
            # Clear screen (simple version)
            print("\033[2J\033[H")
            
            println("GPU Monitoring Dashboard")
            println("=======================")
            println("Time: $(Dates.format(now(), "HH:MM:SS"))")
            println("Elapsed: $(round((now() - start_time).value / 1000, digits=1))s")
            println()
            
            # Display current metrics
            for device_id in monitor.devices
                metrics = get_current_metrics(monitor, device_id)
                if !isnothing(metrics) && metrics.is_available
                    println("GPU $device_id:")
                    println("  Utilization: $(create_bar(metrics.utilization, 100)) $(round(metrics.utilization, digits=1))%")
                    println("  Memory:      $(create_bar(metrics.memory_percent, 100)) $(round(metrics.memory_used, digits=1))/$(round(metrics.memory_total, digits=1)) GB")
                    println("  Temperature: $(create_bar(metrics.temperature, 100, 30, 95)) $(round(metrics.temperature, digits=0))°C")
                    println("  Power:       $(create_bar(metrics.power_draw, 450, 50)) $(round(metrics.power_draw, digits=0))W")
                    println("  Clock:       $(round(metrics.clock_speed, digits=0)) MHz")
                    println("  Fan:         $(create_bar(metrics.fan_speed, 100)) $(round(metrics.fan_speed, digits=0))%")
                    
                    # Show sparkline for utilization
                    history = get_historical_metrics(monitor, device_id)
                    if length(history) > 20
                        sparkline = create_sparkline([h.utilization for h in history[end-19:end]])
                        println("  Trend:       $sparkline")
                    end
                    println()
                end
            end
            
            sleep(0.5)  # Update display every 500ms
        end
    catch e
        if isa(e, InterruptException)
            println("\n\nMonitoring interrupted by user")
        else
            rethrow(e)
        end
    finally
        # Stop monitoring
        stop_monitoring!(monitor)
    end
    
    println("\nMonitoring stopped.")
end

"""
Dashboard integration demo
"""
function dashboard_integration_demo()
    println("GPU Dashboard Integration Demo")
    println("============================")
    
    # Create GPU monitor
    gpu_monitor = create_gpu_monitor()
    
    # Create dashboard with GPU integration
    dashboard = create_integrated_dashboard(gpu_monitor)
    
    println("\nRunning integrated dashboard for 20 seconds...")
    println("This will show real GPU metrics in the console dashboard")
    
    # Create a live dashboard
    include("../src/ui/realtime_update.jl")
    using .RealtimeUpdate
    
    # Create live dashboard with fast updates
    update_config = DashboardUpdateConfig(
        update_interval_ms = 100,
        enable_double_buffer = true,
        enable_delta_updates = true
    )
    live_dashboard = LiveDashboard(dashboard, update_config)
    
    # Start both monitoring systems
    start_monitoring!(gpu_monitor)
    start_live_dashboard!(live_dashboard)
    
    start_time = now()
    duration = Second(20)
    
    try
        while now() - start_time < duration
            # Update GPU panels with real metrics
            update_gpu_panels!(dashboard, gpu_monitor)
            
            # Also update some dummy data for other panels
            updates = Dict{Symbol, PanelContent}(
                :progress => ProgressPanelContent(
                    "GPU Monitoring Demo",
                    (now() - start_time).value / duration.value * 100,
                    50.0, 100, 200, 0.85, 0.90, 
                    Int(round((start_time + duration - now()).value / 1000))
                ),
                :metrics => MetricsPanelContent(
                    1234.5, 8.2, 45.0, 16.0, 85.0, 5, 16
                )
            )
            
            update_live_data!(live_dashboard, updates)
            
            sleep(0.1)
        end
    catch e
        if isa(e, InterruptException)
            println("\n\nDemo interrupted")
        else
            rethrow(e)
        end
    finally
        stop_live_dashboard!(live_dashboard)
        stop_monitoring!(gpu_monitor)
    end
    
    # Show final statistics
    stats = get_update_stats(live_dashboard)
    println("\n\nDashboard Performance:")
    println("  Average FPS: $(round(stats.fps, digits=1))")
    println("  Dropped frames: $(stats.dropped_frames)")
end

# Helper functions for visualization

"""
Create a simple text progress bar
"""
function create_bar(value::Float64, max_value::Float64, min_value::Float64 = 0.0, width::Int = 20)
    percentage = (value - min_value) / (max_value - min_value)
    percentage = clamp(percentage, 0.0, 1.0)
    
    filled = round(Int, percentage * width)
    empty = width - filled
    
    bar = "█"^filled * "░"^empty
    return "[$bar]"
end

"""
Create a sparkline from data points
"""
function create_sparkline(data::Vector{Float64})
    if isempty(data)
        return ""
    end
    
    # Normalize to 0-7 for sparkline characters
    min_val = minimum(data)
    max_val = maximum(data)
    range = max_val - min_val
    
    if range == 0
        return "─"^length(data)
    end
    
    sparkchars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    
    sparkline = ""
    for val in data
        normalized = (val - min_val) / range
        idx = round(Int, normalized * 7) + 1
        idx = clamp(idx, 1, 8)
        sparkline *= sparkchars[idx]
    end
    
    return sparkline
end

# Main menu
function main()
    println("GPU Monitoring Examples")
    println("======================")
    println("1. Simple GPU Monitoring")
    println("2. Advanced Monitoring with Background Updates")
    println("3. Dashboard Integration Demo")
    println("\nSelect demo (1-3): ")
    
    choice = readline()
    
    if choice == "1"
        simple_gpu_monitoring_demo()
    elseif choice == "2"
        advanced_gpu_monitoring_demo()
    elseif choice == "3"
        dashboard_integration_demo()
    else
        println("Running simple demo...")
        simple_gpu_monitoring_demo()
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end