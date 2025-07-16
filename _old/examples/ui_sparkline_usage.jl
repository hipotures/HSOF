# Example usage of Sparkline Graph Components

using Dates
using Printf

# Include the module
include("../src/ui/sparkline_graph.jl")
using .SparklineGraph

"""
Basic sparkline demo showing different types
"""
function basic_sparkline_demo()
    println("Basic Sparkline Demo")
    println("===================")
    
    # Sample data
    data = [30.0, 35.0, 32.0, 38.0, 45.0, 42.0, 48.0, 50.0, 47.0, 55.0,
            53.0, 58.0, 60.0, 57.0, 62.0, 65.0, 63.0, 68.0, 70.0, 72.0]
    
    # Basic sparkline
    println("\n1. Basic Sparkline:")
    sparkline = create_sparkline(data)
    println("   Data: $(join(round.(data, digits=1), ", "))")
    println("   Sparkline: $sparkline")
    
    # Sparkline with custom width
    println("\n2. Custom Width Sparkline (10 chars):")
    config = SparklineConfig(width = 10)
    sparkline = create_sparkline(data, config)
    println("   Sparkline: $sparkline")
    
    # Sparkline with boundaries
    println("\n3. Sparkline with Boundaries:")
    config = SparklineConfig(show_boundaries = true)
    sparkline = create_sparkline(data, config)
    println("   Sparkline: $sparkline")
    
    # Smoothed sparkline
    println("\n4. Smoothed Sparkline:")
    config = SparklineConfig(smooth = true, smooth_window = 5)
    sparkline = create_sparkline(data, config)
    println("   Smoothed: $sparkline")
    println("   Original: $(create_sparkline(data))")
    
    # Bar sparkline
    println("\n5. Bar-style Sparkline:")
    bar_data = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 80.0, 60.0, 40.0, 20.0]
    bar_sparkline = create_bar_sparkline(bar_data, SparklineConfig())
    println("   Data: $(join(bar_data, ", "))")
    println("   Bars: $bar_sparkline")
end

"""
Colored sparkline demo with gradients
"""
function colored_sparkline_demo()
    println("\n\nColored Sparkline Demo")
    println("=====================")
    
    # Temperature data (simulating daily temperatures)
    temps = [15.0, 16.0, 18.0, 22.0, 25.0, 28.0, 30.0, 32.0, 31.0, 29.0,
             26.0, 23.0, 20.0, 18.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0]
    
    # Gradient sparkline (cold=blue, hot=red)
    println("\n1. Temperature Gradient:")
    config = SparklineConfig(
        gradient = true,
        min_color = "blue",
        max_color = "red"
    )
    sparkline = create_colored_sparkline(temps, config)
    println("   Temps: $(join(round.(temps, digits=1), ", "))")
    println("   Graph: $sparkline")
    
    # Performance metrics with color coding
    println("\n2. Performance Metrics:")
    perf_data = [85.0, 88.0, 92.0, 95.0, 98.0, 97.0, 99.0, 100.0, 98.0, 95.0]
    config = SparklineConfig(
        gradient = true,
        min_color = "red",
        max_color = "green"
    )
    sparkline = create_colored_sparkline(perf_data, config)
    println("   Performance: $(join(perf_data, ", "))")
    println("   Graph: $sparkline")
end

"""
Specialized sparklines demo
"""
function specialized_sparklines_demo()
    println("\n\nSpecialized Sparklines Demo")
    println("==========================")
    
    # Score sparkline with trend
    println("\n1. Score Sparkline with Trend:")
    scores = [0.65, 0.67, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80]
    sparkline = create_score_sparkline(scores)
    println("   Scores: $(join(round.(scores, digits=2), ", "))")
    println("   Trend: $sparkline")
    
    # GPU utilization with threshold
    println("\n2. GPU Utilization Sparkline:")
    gpu_utils = [60.0, 65.0, 70.0, 75.0, 82.0, 88.0, 92.0, 95.0, 90.0, 85.0]
    sparkline = create_gpu_sparkline(gpu_utils, 80.0)  # 80% threshold
    println("   GPU %: $(join(gpu_utils, ", "))")
    println("   Graph: $sparkline (threshold: 80%)")
    
    # Performance vs target
    println("\n3. Performance vs Target:")
    perf_values = [95.0, 98.0, 102.0, 105.0, 108.0, 110.0, 107.0, 112.0, 115.0, 118.0]
    sparkline = create_performance_sparkline(perf_values, 100.0)  # Target: 100
    println("   Values: $(join(perf_values, ", "))")
    println("   Graph: $sparkline (target: 100)")
end

"""
Real-time monitoring demo with circular buffer
"""
function realtime_monitoring_demo()
    println("\n\nReal-time Monitoring Demo")
    println("========================")
    println("Simulating 30 seconds of monitoring (press Ctrl+C to stop)")
    
    # Create renderer with auto-scaling
    config = SparklineConfig(
        width = 30,
        smooth = true,
        smooth_window = 3,
        gradient = true,
        min_color = "green",
        max_color = "red"
    )
    
    renderer = SparklineRenderer(config, 100)  # Keep 100 points
    
    # Simulate real-time data
    start_time = now()
    base_value = 50.0
    
    try
        while (now() - start_time).value < 30000  # 30 seconds
            # Generate simulated value
            time_offset = (now() - start_time).value / 1000.0
            value = base_value + 
                    20.0 * sin(time_offset / 5.0) +  # Slow wave
                    10.0 * sin(time_offset / 1.0) +  # Fast wave
                    5.0 * randn()                     # Noise
            
            # Push to renderer
            push_data!(renderer, value)
            
            # Clear line and print
            print("\r")
            print("Value: $(round(value, digits=1)) | ")
            
            # Get and display sparkline
            sparkline = SparklineGraph.render(renderer)
            print("Last 30s: $sparkline")
            
            # Get stats
            stats = SparklineGraph.get_stats(renderer)
            print(" | Min: $(round(stats.min, digits=1))")
            print(" Max: $(round(stats.max, digits=1))")
            print(" Avg: $(round(stats.mean, digits=1))")
            
            flush(stdout)
            sleep(0.1)
        end
    catch e
        if isa(e, InterruptException)
            println("\n\nMonitoring stopped by user")
        else
            rethrow(e)
        end
    end
    
    println("\n")
end

"""
Advanced circular buffer demo
"""
function circular_buffer_demo()
    println("\n\nCircular Buffer Demo")
    println("===================")
    
    # Create a small buffer for demonstration
    buffer = CircularBuffer{Float64}(10)
    
    println("Buffer capacity: $(buffer.capacity)")
    println("\nPushing 15 values (buffer will wrap around):")
    
    for i in 1:15
        value = Float64(i * 10)
        push_data!(buffer, value)
        data, _ = get_data(buffer)
        println("  Push $value -> Buffer: [$(join(data, ", "))]")
    end
    
    println("\nGetting recent 5 values:")
    recent, _ = get_recent_data(buffer, 5)
    println("  Recent 5: [$(join(recent, ", "))]")
    
    println("\nClearing buffer:")
    clear_buffer!(buffer)
    data, _ = get_data(buffer)
    println("  Buffer after clear: [$(join(data, ", "))] (size: $(buffer.size))")
end

"""
Data smoothing comparison demo
"""
function smoothing_demo()
    println("\n\nData Smoothing Demo")
    println("==================")
    
    # Create noisy data
    t = 0:0.5:10
    clean_data = [50.0 + 20.0 * sin(x) for x in t]
    noisy_data = clean_data + 5.0 * randn(length(clean_data))
    
    println("Comparing smoothing methods on noisy sine wave:")
    
    # Original noisy data
    config_raw = SparklineConfig(width = 40, smooth = false)
    sparkline_raw = create_sparkline(noisy_data, config_raw)
    println("\n1. Raw noisy data:")
    println("   $sparkline_raw")
    
    # Moving average smoothing
    config_ma = SparklineConfig(width = 40, smooth = true, smooth_window = 3)
    sparkline_ma = create_sparkline(noisy_data, config_ma)
    println("\n2. Moving average (window=3):")
    println("   $sparkline_ma")
    
    config_ma5 = SparklineConfig(width = 40, smooth = true, smooth_window = 5)
    sparkline_ma5 = create_sparkline(noisy_data, config_ma5)
    println("\n3. Moving average (window=5):")
    println("   $sparkline_ma5")
    
    # Exponential smoothing
    exp_smoothed = exponential_smoothing(noisy_data, 0.3)
    sparkline_exp = create_sparkline(exp_smoothed, config_raw)
    println("\n4. Exponential smoothing (α=0.3):")
    println("   $sparkline_exp")
    
    # Clean data for comparison
    sparkline_clean = create_sparkline(clean_data, config_raw)
    println("\n5. Original clean signal:")
    println("   $sparkline_clean")
end

# Main menu
function main()
    println("Sparkline Graph Examples")
    println("=======================")
    println("1. Basic Sparklines")
    println("2. Colored Sparklines")
    println("3. Specialized Sparklines")
    println("4. Real-time Monitoring")
    println("5. Circular Buffer Demo")
    println("6. Data Smoothing Demo")
    println("7. Run All Demos")
    println("\nSelect demo (1-7): ")
    
    choice = readline()
    
    if choice == "1"
        basic_sparkline_demo()
    elseif choice == "2"
        colored_sparkline_demo()
    elseif choice == "3"
        specialized_sparklines_demo()
    elseif choice == "4"
        realtime_monitoring_demo()
    elseif choice == "5"
        circular_buffer_demo()
    elseif choice == "6"
        smoothing_demo()
    elseif choice == "7"
        basic_sparkline_demo()
        colored_sparkline_demo()
        specialized_sparklines_demo()
        circular_buffer_demo()
        smoothing_demo()
        println("\nSkipping real-time demo in batch mode...")
    else
        println("Invalid choice. Running basic demo...")
        basic_sparkline_demo()
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end