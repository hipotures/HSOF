using Test
using Dates

# Include required modules
include("../../src/ui/color_theme.jl")
include("../../src/ui/gpu_monitor.jl")
include("../../src/ui/sparkline_graph.jl")
include("../../src/ui/terminal_compat.jl")
include("../../src/ui/gpu_status_panel.jl")

using .ColorTheme
using .GPUMonitor
using .SparklineGraph
using .TerminalCompat
using .GPUStatusPanel

@testset "GPU Status Panel Tests" begin
    
    @testset "Panel State Creation" begin
        # Test with auto-detection
        state = create_gpu_panel()
        @test isa(state, GPUStatusPanel.GPUPanelState)
        @test !isempty(state.gpu_indices)
        @test isa(state.monitor, GPUMonitor.GPUMonitorState)
        @test state.show_details == true
        @test state.show_history == true
        @test state.activity_phase == 1
        
        # Test with specific GPUs
        state2 = create_gpu_panel([1, 2])
        @test state2.gpu_indices == [1, 2]
        @test length(state2.stats_history) == 2
        @test length(state2.thermal_warnings) == 2
    end
    
    @testset "Stats History" begin
        history = GPUStatusPanel.GPUStatsHistory()
        @test history.min_utilization == 100.0
        @test history.max_utilization == 0.0
        @test history.sample_count == 0
        
        # Create mock monitor
        monitor = GPUMonitor.create_gpu_monitor([1])
        
        # Update history
        GPUStatusPanel.update_stats_history!(history, monitor, 1)
        @test history.sample_count == 1
        
        # Min/max should be updated
        metrics = GPUMonitor.get_current_metrics(monitor, 1)
        if !isnothing(metrics)
            @test history.min_utilization <= metrics.utilization
            @test history.max_utilization >= metrics.utilization
        end
    end
    
    @testset "Activity Symbol" begin
        # Test activity symbol rotation
        symbols = [GPUStatusPanel.get_activity_symbol(i) for i in 1:10]
        @test length(unique(symbols)) == length(GPUStatusPanel.ACTIVITY_SYMBOLS)
        @test all(s -> s in GPUStatusPanel.ACTIVITY_SYMBOLS, symbols)
        
        # Test wrapping
        symbol1 = GPUStatusPanel.get_activity_symbol(1)
        symbol_wrap = GPUStatusPanel.get_activity_symbol(length(GPUStatusPanel.ACTIVITY_SYMBOLS) + 1)
        @test symbol1 == symbol_wrap
    end
    
    @testset "Thermal Throttling Detection" begin
        @test GPUStatusPanel.check_thermal_throttling(82.0) == false
        @test GPUStatusPanel.check_thermal_throttling(83.0) == true
        @test GPUStatusPanel.check_thermal_throttling(84.0) == true
        @test GPUStatusPanel.check_thermal_throttling(90.0) == true
    end
    
    @testset "Utilization Bar Rendering" begin
        theme = create_theme()
        
        # Test various utilization levels
        bar = render_utilization_bar(0.0, 40, theme)
        @test occursin("0%", bar)
        @test occursin("░", bar)
        
        bar = render_utilization_bar(50.0, 40, theme)
        @test occursin("50%", bar)
        @test occursin("█", bar)
        @test occursin("░", bar)
        
        bar = render_utilization_bar(100.0, 40, theme)
        @test occursin("100%", bar)
        @test occursin("█", bar)
        
        # Test narrow width
        bar = render_utilization_bar(75.0, 20, theme)
        @test occursin("75%", bar)
    end
    
    @testset "Memory Chart Rendering" begin
        theme = create_theme()
        
        # Test memory display
        chart = render_memory_chart(4096.0, 8192.0, 40, theme)
        @test occursin("4.0/8.0 GB", chart)
        @test occursin("50%", chart)
        
        # Test full memory
        chart = render_memory_chart(8192.0, 8192.0, 40, theme)
        @test occursin("100%", chart)
        
        # Test narrow width
        chart = render_memory_chart(2048.0, 8192.0, 25, theme)
        @test occursin("2.0/8.0 GB", chart)
    end
    
    @testset "Temperature Gauge Rendering" begin
        theme = create_theme()
        
        # Test normal temperature
        gauge = render_temperature_gauge(60.0, 40, theme, false)
        @test occursin("60°C", gauge)
        @test !occursin("⚠", gauge)
        
        # Test high temperature
        gauge = render_temperature_gauge(86.0, 40, theme, false)
        @test occursin("86°C", gauge)
        @test occursin("!", gauge)
        
        # Test thermal throttling
        gauge = render_temperature_gauge(85.0, 40, theme, true)
        @test occursin("85°C", gauge)
        @test occursin("⚠", gauge)
    end
    
    @testset "Power Display Rendering" begin
        theme = create_theme()
        
        # Test normal power
        display = render_power_display(150.0, 300.0, 40, theme)
        @test occursin("150W/300W", display)
        @test occursin("50%", display)
        
        # Test high power
        display = render_power_display(290.0, 300.0, 40, theme)
        @test occursin("290W/300W", display)
        @test occursin("97%", display)
    end
    
    @testset "Clock Speed Rendering" begin
        theme = create_theme()
        
        # Test clock display
        display = render_clock_speeds(1800.0, 10000.0, 50, theme)
        @test occursin("1.80 GHz", display)
        @test occursin("10.00 GHz", display)
        
        # Test boost clocks
        display = render_clock_speeds(2500.0, 11000.0, 50, theme)
        @test occursin("2.50 GHz", display)
        @test occursin("▮", display)  # Visual indicator
    end
    
    @testset "Stats Formatting" begin
        theme = create_theme()
        history = GPUStatusPanel.GPUStatsHistory()
        
        # Test empty stats
        lines = format_gpu_stats(history, 50, theme)
        @test lines[1] == "No historical data available"
        
        # Test with data
        history.sample_count = 10
        history.min_utilization = 20.0
        history.avg_utilization = 50.0
        history.max_utilization = 80.0
        history.min_temperature = 40.0
        history.avg_temperature = 60.0
        history.max_temperature = 75.0
        
        lines = format_gpu_stats(history, 50, theme)
        @test length(lines) >= 4
        @test occursin("Historical Statistics", lines[1])
        @test occursin("20%", lines[2])  # Min util
        @test occursin("80%", lines[2])  # Max util
        @test occursin("40°C", lines[4]) # Min temp
        @test occursin("75°C", lines[4]) # Max temp
    end
    
    @testset "Panel Update" begin
        state = create_gpu_panel([1])
        initial_phase = state.activity_phase
        
        # Update panel
        update_gpu_panel!(state)
        
        # Activity phase should advance
        @test state.activity_phase == initial_phase + 1
        
        # Last update time should be recent
        @test (now() - state.last_update).value < 1000  # Less than 1 second
        
        # Stats should be updated
        history = state.stats_history[1]
        @test history.sample_count >= 1
    end
    
    @testset "Single GPU Rendering" begin
        state = create_gpu_panel([1])
        content = render_gpu_status(state, 60, 10)
        
        @test isa(content, GPUPanelContent)
        @test content.width == 60
        @test content.height == 10
        @test !isempty(content.lines)
        
        # Check for expected content
        all_text = join(content.lines, "\n")
        @test occursin("GPU", all_text)
        @test occursin("Utilization", all_text)
        @test occursin("Memory", all_text)
        @test occursin("Temperature", all_text)
    end
    
    @testset "Multi-GPU Comparison" begin
        # Create state with multiple GPUs
        gpu_count = GPUMonitor.get_gpu_count()
        if gpu_count > 1
            state = create_gpu_panel([1, 2])
            content = render_gpu_status(state, 80, 8)
            
            @test !isempty(content.lines)
            all_text = join(content.lines, "\n")
            @test occursin("GPU 1", all_text)
            @test occursin("GPU 2", all_text)
            @test occursin("│", all_text)  # Separator
        else
            @test true  # Skip if only one GPU
        end
    end
    
    @testset "Mini Bar Rendering" begin
        theme = create_theme()
        
        # Test mini bar for comparison view
        bar = GPUStatusPanel.render_mini_bar(0.0, 10, theme)
        @test length(bar) > 0
        @test occursin("▱", bar)
        
        bar = GPUStatusPanel.render_mini_bar(50.0, 10, theme)
        @test occursin("▰", bar)
        @test occursin("▱", bar)
        
        bar = GPUStatusPanel.render_mini_bar(100.0, 10, theme)
        @test occursin("▰", bar)
    end
    
    @testset "GPU Header Rendering" begin
        state = create_gpu_panel([1])
        monitor = state.monitor
        metrics = GPUMonitor.get_current_metrics(monitor, 1)
        
        if !isnothing(metrics)
            header = GPUStatusPanel.render_gpu_header(state, 1, metrics, 60)
            @test occursin("GPU 1", header)
            @test occursin(GPUStatusPanel.ACTIVITY_SYMBOLS[1], header)
            
            # Test thermal warning
            state.thermal_warnings[1] = true
            header = GPUStatusPanel.render_gpu_header(state, 1, metrics, 60)
            @test occursin("THERMAL THROTTLE", header)
        end
    end
    
    @testset "Edge Cases" begin
        theme = create_theme()
        
        # Test very narrow widths
        bar = render_utilization_bar(50.0, 15, theme)
        @test length(bar) <= 15
        
        mem = render_memory_chart(4096.0, 8192.0, 20, theme)
        @test length(mem) <= 20
        
        # Test zero values
        bar = render_utilization_bar(0.0, 30, theme)
        @test occursin("0%", bar)
        
        power = render_power_display(0.0, 300.0, 30, theme)
        @test occursin("0W", power)
    end
end

println("All GPU status panel tests passed! ✓")