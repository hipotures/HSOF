#!/usr/bin/env julia

# GPU Status Panel Demo
# This example demonstrates the GPU status visualization capabilities

# Add the parent directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

# Include required modules
include("../src/ui/color_theme.jl")
include("../src/ui/gpu_monitor.jl")
include("../src/ui/sparkline_graph.jl")
include("../src/ui/terminal_compat.jl")
include("../src/ui/gpu_status_panel.jl")

using .ColorTheme
using .GPUMonitor
using .TerminalCompat
using .GPUStatusPanel
using Printf

function demo_gpu_status_panel()
    println("\n=== GPU Status Panel Demo ===\n")
    
    # Create theme and panel
    theme = create_theme()
    
    # Demo 1: Single GPU Display
    println("1. Single GPU Status Display:")
    println("─" ^ 60)
    
    state = create_gpu_panel([0], theme=theme)
    update_gpu_panel!(state)
    
    # Render at different sizes
    content = render_gpu_status(state, 60, 10)
    println(join(content.lines, "\n"))
    
    println("\n2. Detailed View with History:")
    println("─" ^ 80)
    
    # Enable all features
    state.show_details = true
    state.show_history = true
    
    # Simulate some GPU activity to build history
    monitor = state.monitor
    for i in 1:30
        GPUMonitor.update_metrics!(monitor)
        update_gpu_panel!(state)
        sleep(0.05)  # Small delay to generate varied data
    end
    
    # Render with history
    content = render_gpu_status(state, 80, 20)
    println(join(content.lines, "\n"))
    
    # Demo 3: Multi-GPU Comparison (if available)
    println("\n3. Multi-GPU Comparison View:")
    println("─" ^ 80)
    
    # Try to create multi-GPU view
    multi_state = create_gpu_panel([0, 1], theme=theme)
    
    if length(multi_state.gpu_indices) > 1
        update_gpu_panel!(multi_state)
        content = render_gpu_status(multi_state, 80, 10)
        println(join(content.lines, "\n"))
    else
        println("Only one GPU available. Multi-GPU comparison requires multiple GPUs.")
    end
    
    # Demo 4: Component Demonstrations
    println("\n4. Individual Component Examples:")
    println("─" ^ 80)
    
    # Utilization Bar
    println("\nUtilization Bars at Different Levels:")
    for util in [0.0, 25.0, 50.0, 75.0, 100.0]
        bar = render_utilization_bar(util, 50, theme)
        println(bar)
    end
    
    # Memory Chart
    println("\nMemory Usage Display:")
    for (used, total) in [(2048.0, 8192.0), (4096.0, 8192.0), (7168.0, 8192.0)]
        chart = render_memory_chart(used, total, 60, theme)
        println(chart)
    end
    
    # Temperature Gauge
    println("\nTemperature Gauges:")
    for temp in [40.0, 60.0, 80.0, 85.0]
        throttle = check_thermal_throttling(temp)
        gauge = render_temperature_gauge(temp, 60, theme, throttle)
        println(gauge)
    end
    
    # Power Display
    println("\nPower Consumption:")
    for power in [100.0, 200.0, 300.0, 350.0]
        display = render_power_display(power, 350.0, 60, theme)
        println(display)
    end
    
    # Clock Speeds
    println("\nClock Speed Display:")
    for (gpu_clock, mem_clock) in [(1500.0, 8000.0), (2000.0, 10000.0), (2500.0, 11000.0)]
        clocks = render_clock_speeds(gpu_clock, mem_clock, 60, theme)
        println(clocks)
    end
    
    # Demo 5: Activity Indicators
    println("\n5. Activity Indicator Animation:")
    println("─" ^ 40)
    
    print("GPU Activity: ")
    for i in 1:20
        symbol = get_activity_symbol(i)
        print("\r", " " ^ 15, "\rGPU Activity: ", symbol, " Processing...")
        sleep(0.1)
    end
    println("\n")
    
    # Demo 6: Stats Summary
    println("6. Historical Statistics:")
    println("─" ^ 60)
    
    # Get stats from first GPU
    if haskey(state.stats_history, 0)
        stats = state.stats_history[0]
        if stats.sample_count > 0
            lines = format_gpu_stats(stats, 60, theme)
            for line in lines
                println(line)
            end
        end
    end
    
    # Demo 7: Thermal Throttling Warning
    println("\n7. Thermal Throttling Detection:")
    println("─" ^ 60)
    
    # Create a state with thermal warning
    warn_state = create_gpu_panel([0], theme=theme)
    warn_state.thermal_warnings[0] = true
    
    # Get current metrics and render header
    metrics = GPUMonitor.get_current_metrics(warn_state.monitor, 0)
    if !isnothing(metrics)
        # Simulate high temperature
        metrics.temperature = 85.0
        header = GPUStatusPanel.render_gpu_header(warn_state, 0, metrics, 60)
        println(header)
        
        temp_gauge = render_temperature_gauge(85.0, 60, theme, true)
        println("  ", temp_gauge)
    end
    
    println("\n=== End of Demo ===\n")
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    try
        demo_gpu_status_panel()
    catch e
        println("Error running demo: ", e)
        println("Note: This demo requires a CUDA-capable GPU or will use simulated data.")
    end
end