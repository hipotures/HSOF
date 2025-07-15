module GPUStatusPanel

using Dates
using Printf
using Statistics
using ..GPUMonitor
using ..ColorTheme
using ..SparklineGraph
using ..TerminalCompat

export GPUPanelState, GPUPanelContent
export create_gpu_panel, update_gpu_panel!
export render_gpu_status, render_utilization_bar
export render_memory_chart, render_temperature_gauge
export render_power_display, render_clock_speeds
export render_gpu_comparison, get_activity_symbol
export format_gpu_stats, check_thermal_throttling

# Historical statistics for a GPU
mutable struct GPUStatsHistory
    min_utilization::Float64
    max_utilization::Float64
    avg_utilization::Float64
    min_memory::Float64
    max_memory::Float64
    avg_memory::Float64
    min_temperature::Float64
    max_temperature::Float64
    avg_temperature::Float64
    sample_count::Int
    
    function GPUStatsHistory()
        new(100.0, 0.0, 0.0,    # Utilization
            100.0, 0.0, 0.0,    # Memory
            100.0, 0.0, 0.0,    # Temperature
            0)
    end
end

# GPU panel state
mutable struct GPUPanelState
    gpu_indices::Vector{Int}  # Which GPUs to display
    monitor::GPUMonitor.GPUMonitorState
    theme::ThemeConfig
    show_details::Bool
    show_history::Bool
    activity_phase::Int  # For animated activity indicator
    last_update::DateTime
    stats_history::Dict{Int, GPUStatsHistory}  # Per-GPU historical stats
    thermal_warnings::Dict{Int, Bool}  # Thermal throttling state per GPU
end

# Panel content for rendering
struct GPUPanelContent
    lines::Vector{String}
    width::Int
    height::Int
end

# Activity indicator symbols
const ACTIVITY_SYMBOLS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# Create GPU panel state
function create_gpu_panel(gpu_indices::Vector{Int} = Int[];
                         monitor::Union{GPUMonitor.GPUMonitorState, Nothing} = nothing,
                         theme::ThemeConfig = create_theme())
    # Create or use provided monitor
    if isnothing(monitor)
        config = GPUMonitor.GPUMonitorConfig(poll_interval_ms=1000)
        monitor = GPUMonitor.create_gpu_monitor(config)
        GPUMonitor.start_monitoring!(monitor)
    end
    
    # If no GPU indices specified, use all available GPUs
    if isempty(gpu_indices)
        gpu_indices = monitor.devices
    end
    
    # Initialize stats history
    stats_history = Dict{Int, GPUStatsHistory}()
    thermal_warnings = Dict{Int, Bool}()
    for idx in gpu_indices
        stats_history[idx] = GPUStatsHistory()
        thermal_warnings[idx] = false
    end
    
    GPUPanelState(
        gpu_indices,
        monitor,
        theme,
        true,   # show_details
        true,   # show_history
        1,      # activity_phase
        now(),
        stats_history,
        thermal_warnings
    )
end

# Update GPU panel state
function update_gpu_panel!(state::GPUPanelState)
    # Update monitoring data
    GPUMonitor.update_metrics!(state.monitor)
    
    # Update activity indicator
    state.activity_phase = (state.activity_phase % length(ACTIVITY_SYMBOLS)) + 1
    
    # Update historical statistics
    for idx in state.gpu_indices
        update_stats_history!(state.stats_history[idx], state.monitor, idx)
        
        # Check thermal throttling
        metrics = GPUMonitor.get_current_metrics(state.monitor, idx)
        if !isnothing(metrics)
            state.thermal_warnings[idx] = check_thermal_throttling(metrics.temperature)
        end
    end
    
    state.last_update = now()
end

# Update historical statistics
function update_stats_history!(history::GPUStatsHistory, monitor::GPUMonitor.GPUMonitorState, gpu_idx::Int)
    metrics = GPUMonitor.get_current_metrics(monitor, gpu_idx)
    if isnothing(metrics)
        return
    end
    
    # Update utilization stats
    history.min_utilization = min(history.min_utilization, metrics.utilization)
    history.max_utilization = max(history.max_utilization, metrics.utilization)
    
    # Update memory stats
    memory_percent = metrics.memory_used / metrics.memory_total * 100
    history.min_memory = min(history.min_memory, memory_percent)
    history.max_memory = max(history.max_memory, memory_percent)
    
    # Update temperature stats
    history.min_temperature = min(history.min_temperature, metrics.temperature)
    history.max_temperature = max(history.max_temperature, metrics.temperature)
    
    # Update averages (simple moving average)
    history.sample_count += 1
    α = 1.0 / history.sample_count  # Weight for new sample
    
    history.avg_utilization = (1 - α) * history.avg_utilization + α * metrics.utilization
    history.avg_memory = (1 - α) * history.avg_memory + α * memory_percent
    history.avg_temperature = (1 - α) * history.avg_temperature + α * metrics.temperature
end

# Render GPU status panel
function render_gpu_status(state::GPUPanelState, width::Int, height::Int)::GPUPanelContent
    lines = String[]
    
    if length(state.gpu_indices) == 1
        # Single GPU view
        render_single_gpu!(lines, state, state.gpu_indices[1], width, height)
    else
        # Multi-GPU comparison view
        render_gpu_comparison!(lines, state, width, height)
    end
    
    GPUPanelContent(lines, width, height)
end

# Render single GPU view
function render_single_gpu!(lines::Vector{String}, state::GPUPanelState, gpu_idx::Int, width::Int, height::Int)
    metrics = GPUMonitor.get_current_metrics(state.monitor, gpu_idx)
    if isnothing(metrics)
        push!(lines, "GPU $gpu_idx: Not Available")
        return
    end
    
    # Header with activity indicator
    header = render_gpu_header(state, gpu_idx, metrics, width)
    push!(lines, header)
    
    # Utilization bar
    if height > 2
        util_bar = render_utilization_bar(metrics.utilization, width-2, state.theme)
        push!(lines, "  " * util_bar)
    end
    
    # Memory usage
    if height > 3
        mem_display = render_memory_chart(metrics.memory_used, metrics.memory_total, 
                                         width-2, state.theme)
        push!(lines, "  " * mem_display)
    end
    
    # Temperature gauge
    if height > 4
        temp_gauge = render_temperature_gauge(metrics.temperature, width-2, state.theme,
                                            state.thermal_warnings[gpu_idx])
        push!(lines, "  " * temp_gauge)
    end
    
    # Power and clocks
    if height > 5 && state.show_details
        power_line = render_power_display(metrics.power_draw, metrics.power_limit, 
                                        width-2, state.theme)
        push!(lines, "  " * power_line)
    end
    
    if height > 6 && state.show_details
        clock_line = render_clock_speeds(metrics.clock_graphics, metrics.clock_memory,
                                       width-2, state.theme)
        push!(lines, "  " * clock_line)
    end
    
    # Sparkline history
    if height > 8 && state.show_history
        push!(lines, "")  # Blank line
        history = GPUMonitor.get_metric_history(state.monitor, gpu_idx, :utilization)
        if !isempty(history)
            spark_line = SparklineGraph.render_sparkline(
                last(history, min(60, length(history))),
                width = width - 4,
                height = 1,
                theme = state.theme,
                show_axes = false
            )
            push!(lines, "  " * spark_line)
        end
    end
    
    # Historical stats
    if height > 10 && state.show_history
        stats_lines = format_gpu_stats(state.stats_history[gpu_idx], width-2, state.theme)
        for line in stats_lines
            push!(lines, "  " * line)
        end
    end
end

# Render GPU header with name and activity
function render_gpu_header(state::GPUPanelState, gpu_idx::Int, metrics::GPUMonitor.GPUMetrics, width::Int)
    # GPU name and index
    name = "GPU $gpu_idx: $(metrics.name)"
    
    # Activity indicator
    activity = get_activity_symbol(state.activity_phase)
    activity_color = metrics.utilization > 50 ? 
                    get_status_color(state.theme, :normal) :
                    get_status_color(state.theme, :dim)
    activity_str = apply_theme_color(activity, activity_color)
    
    # Status indicator
    status = if state.thermal_warnings[gpu_idx]
        apply_theme_color(" [THERMAL THROTTLE]", get_status_color(state.theme, :critical))
    elseif metrics.utilization > 90
        apply_theme_color(" [HIGH LOAD]", get_status_color(state.theme, :warning))
    else
        ""
    end
    
    # Combine with proper spacing
    header_text = name * status
    padding = width - length(name) - length(status) - 2
    
    return activity_str * " " * header_text * (" " ^ max(0, padding))
end

# Render utilization bar graph
function render_utilization_bar(utilization::Float64, width::Int, theme::ThemeConfig)
    label = @sprintf("Utilization: %3.0f%%", utilization)
    bar_width = width - length(label) - 2
    
    if bar_width < 10
        return label
    end
    
    # Create bar
    filled = round(Int, utilization / 100 * bar_width)
    bar_chars = "█" ^ filled * "░" ^ (bar_width - filled)
    
    # Apply color based on utilization
    color = get_color(theme, :gpu_utilization, utilization)
    colored_bar = apply_theme_color(bar_chars[1:filled], color) * 
                  apply_theme_color(bar_chars[filled+1:end], get_status_color(theme, :dim))
    
    return @sprintf("%s [%s]", label, colored_bar)
end

# Render memory usage chart (simplified pie chart using text)
function render_memory_chart(used::Float64, total::Float64, width::Int, theme::ThemeConfig)
    percent = used / total * 100
    used_gb = used / 1024
    total_gb = total / 1024
    
    label = @sprintf("Memory: %.1f/%.1f GB (%.0f%%)", used_gb, total_gb, percent)
    
    if width - length(label) < 12
        return label
    end
    
    # Create simple text-based "pie chart" using blocks
    chart_size = 10
    filled_blocks = round(Int, percent / 100 * chart_size)
    
    # Use different block characters for visual effect
    blocks = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    chart = ""
    
    for i in 1:chart_size
        if i <= filled_blocks
            block_idx = min(8, round(Int, (i / filled_blocks) * 8))
            block_idx = max(1, block_idx)
            color = get_color(theme, :memory_usage, percent)
            chart *= apply_theme_color(blocks[block_idx], color)
        else
            chart *= apply_theme_color("░", get_status_color(theme, :dim))
        end
    end
    
    return @sprintf("%s [%s]", label, chart)
end

# Render temperature gauge
function render_temperature_gauge(temperature::Float64, width::Int, theme::ThemeConfig, 
                                throttling::Bool = false)
    label = @sprintf("Temperature: %2.0f°C", temperature)
    gauge_width = width - length(label) - 4
    
    if gauge_width < 10
        return label
    end
    
    # Temperature scale (0-100°C)
    temp_normalized = clamp(temperature / 100, 0, 1)
    filled = round(Int, temp_normalized * gauge_width)
    
    # Create gauge with gradient effect
    gauge = ""
    for i in 1:gauge_width
        if i <= filled
            # Use gradient from cold to hot
            t = i / gauge_width
            if t < 0.6
                char_color = get_status_color(theme, :normal)
            elseif t < 0.8
                char_color = get_status_color(theme, :warning)
            else
                char_color = get_status_color(theme, :critical)
            end
            gauge *= apply_theme_color("▓", char_color)
        else
            gauge *= apply_theme_color("░", get_status_color(theme, :dim))
        end
    end
    
    # Add warning indicator
    warning = if throttling
        apply_theme_color(" ⚠", get_status_color(theme, :critical))
    elseif temperature > 85
        apply_theme_color(" !", get_status_color(theme, :warning))
    else
        ""
    end
    
    return @sprintf("%s [%s]%s", label, gauge, warning)
end

# Render power consumption display
function render_power_display(power_draw::Float64, power_limit::Float64, width::Int, theme::ThemeConfig)
    percent = power_draw / power_limit * 100
    label = @sprintf("Power: %.0fW/%.0fW (%.0f%%)", power_draw, power_limit, percent)
    
    bar_width = width - length(label) - 2
    if bar_width < 5
        return label
    end
    
    # Simple bar for power
    filled = round(Int, percent / 100 * bar_width)
    bar = "▪" ^ filled * "·" ^ (bar_width - filled)
    
    # Color based on power usage
    color = if percent > 95
        get_status_color(theme, :critical)
    elseif percent > 80
        get_status_color(theme, :warning)
    else
        get_status_color(theme, :normal)
    end
    
    colored_bar = apply_theme_color(bar[1:filled], color) * 
                  apply_theme_color(bar[filled+1:end], get_status_color(theme, :dim))
    
    return @sprintf("%s [%s]", label, colored_bar)
end

# Render clock speeds
function render_clock_speeds(graphics_clock::Float64, memory_clock::Float64, width::Int, theme::ThemeConfig)
    gpu_ghz = graphics_clock / 1000
    mem_ghz = memory_clock / 1000
    
    label = @sprintf("Clocks: GPU %.2f GHz, Mem %.2f GHz", gpu_ghz, mem_ghz)
    
    # Add visual indicators for boost status
    remaining = width - length(label)
    if remaining > 10
        # Show relative clock speeds as mini bars
        gpu_bar = "▮" ^ min(5, round(Int, gpu_ghz / 3.0 * 5))  # Assume max ~3GHz
        mem_bar = "▮" ^ min(5, round(Int, mem_ghz / 12.0 * 5))  # Assume max ~12GHz
        
        gpu_color = graphics_clock > 2000 ? get_status_color(theme, :accent) : get_status_color(theme, :normal)
        mem_color = memory_clock > 10000 ? get_status_color(theme, :accent) : get_status_color(theme, :normal)
        
        bars = " " * apply_theme_color(gpu_bar, gpu_color) * " " * apply_theme_color(mem_bar, mem_color)
        return label * bars
    end
    
    return label
end

# Get activity symbol
function get_activity_symbol(phase::Int)
    return ACTIVITY_SYMBOLS[((phase - 1) % length(ACTIVITY_SYMBOLS)) + 1]
end

# Check thermal throttling
function check_thermal_throttling(temperature::Float64)::Bool
    return temperature >= 83.0  # NVIDIA GPUs typically throttle at 83°C
end

# Format GPU statistics
function format_gpu_stats(stats::GPUStatsHistory, width::Int, theme::ThemeConfig)::Vector{String}
    lines = String[]
    
    if stats.sample_count == 0
        push!(lines, "No historical data available")
        return lines
    end
    
    # Header
    header_color = get_status_color(theme, :accent)
    push!(lines, apply_theme_color("Historical Statistics:", header_color))
    
    # Utilization stats
    util_line = @sprintf("  Util: Min %.0f%% | Avg %.1f%% | Max %.0f%%",
                        stats.min_utilization, stats.avg_utilization, stats.max_utilization)
    push!(lines, util_line)
    
    # Memory stats
    mem_line = @sprintf("  Mem:  Min %.0f%% | Avg %.1f%% | Max %.0f%%",
                       stats.min_memory, stats.avg_memory, stats.max_memory)
    push!(lines, mem_line)
    
    # Temperature stats with color coding
    temp_line = @sprintf("  Temp: Min %.0f°C | Avg %.1f°C | Max %.0f°C",
                        stats.min_temperature, stats.avg_temperature, stats.max_temperature)
    
    # Color the max temperature if it's high
    if stats.max_temperature > 80
        temp_parts = split(temp_line, "|")
        if length(temp_parts) >= 3
            max_part = temp_parts[3]
            color = stats.max_temperature > 85 ? get_status_color(theme, :critical) : get_status_color(theme, :warning)
            temp_parts[3] = apply_theme_color(max_part, color)
            temp_line = join(temp_parts, "|")
        end
    end
    
    push!(lines, temp_line)
    
    return lines
end

# Render GPU comparison view for multiple GPUs
function render_gpu_comparison!(lines::Vector{String}, state::GPUPanelState, width::Int, height::Int)
    gpu_count = length(state.gpu_indices)
    if gpu_count == 0
        push!(lines, "No GPUs available")
        return
    end
    
    # Calculate width per GPU
    gpu_width = div(width - (gpu_count - 1), gpu_count)  # Account for separators
    
    # Header
    header_line = ""
    for (i, gpu_idx) in enumerate(state.gpu_indices)
        metrics = GPUMonitor.get_current_metrics(state.monitor, gpu_idx)
        if !isnothing(metrics)
            gpu_header = @sprintf("GPU %d: %s", gpu_idx, 
                                 metrics.name[1:min(end, gpu_width-8)])
            
            # Pad to gpu_width
            gpu_header = rpad(gpu_header, gpu_width)
            
            # Add activity indicator
            activity = get_activity_symbol(state.activity_phase + i)
            activity_color = metrics.utilization > 50 ? 
                           get_status_color(state.theme, :normal) :
                           get_status_color(state.theme, :dim)
            gpu_header = apply_theme_color(activity, activity_color) * " " * gpu_header[2:end]
            
            header_line *= gpu_header
            if i < gpu_count
                header_line *= "│"
            end
        end
    end
    push!(lines, header_line)
    
    # Separator
    sep_line = "─" ^ gpu_width
    for i in 2:gpu_count
        sep_line *= "┼" * ("─" ^ gpu_width)
    end
    push!(lines, apply_theme_color(sep_line, get_status_color(state.theme, :dim)))
    
    # Metrics rows
    if height > 3
        # Utilization row
        util_line = ""
        for (i, gpu_idx) in enumerate(state.gpu_indices)
            metrics = GPUMonitor.get_current_metrics(state.monitor, gpu_idx)
            if !isnothing(metrics)
                util_str = @sprintf("Util: %3.0f%%", metrics.utilization)
                util_bar = render_mini_bar(metrics.utilization, gpu_width - length(util_str) - 2, state.theme)
                gpu_util = util_str * " " * util_bar
                util_line *= rpad(gpu_util, gpu_width)
            else
                util_line *= rpad("N/A", gpu_width)
            end
            if i < gpu_count
                util_line *= "│"
            end
        end
        push!(lines, util_line)
    end
    
    if height > 4
        # Memory row
        mem_line = ""
        for (i, gpu_idx) in enumerate(state.gpu_indices)
            metrics = GPUMonitor.get_current_metrics(state.monitor, gpu_idx)
            if !isnothing(metrics)
                mem_pct = metrics.memory_used / metrics.memory_total * 100
                mem_str = @sprintf("Mem: %3.0f%%", mem_pct)
                mem_bar = render_mini_bar(mem_pct, gpu_width - length(mem_str) - 2, state.theme)
                gpu_mem = mem_str * " " * mem_bar
                mem_line *= rpad(gpu_mem, gpu_width)
            else
                mem_line *= rpad("N/A", gpu_width)
            end
            if i < gpu_count
                mem_line *= "│"
            end
        end
        push!(lines, mem_line)
    end
    
    if height > 5
        # Temperature row
        temp_line = ""
        for (i, gpu_idx) in enumerate(state.gpu_indices)
            metrics = GPUMonitor.get_current_metrics(state.monitor, gpu_idx)
            if !isnothing(metrics)
                temp_str = @sprintf("Temp: %2.0f°C", metrics.temperature)
                warning = state.thermal_warnings[gpu_idx] ? " ⚠" : ""
                color = get_color(state.theme, :gpu_temp, metrics.temperature)
                temp_colored = apply_theme_color(temp_str * warning, color)
                temp_line *= rpad(temp_colored, gpu_width)
            else
                temp_line *= rpad("N/A", gpu_width)
            end
            if i < gpu_count
                temp_line *= "│"
            end
        end
        push!(lines, temp_line)
    end
    
    if height > 6
        # Power row
        power_line = ""
        for (i, gpu_idx) in enumerate(state.gpu_indices)
            metrics = GPUMonitor.get_current_metrics(state.monitor, gpu_idx)
            if !isnothing(metrics)
                power_str = @sprintf("Power: %.0fW", metrics.power_draw)
                power_line *= rpad(power_str, gpu_width)
            else
                power_line *= rpad("N/A", gpu_width)
            end
            if i < gpu_count
                power_line *= "│"
            end
        end
        push!(lines, power_line)
    end
    
    # Sparklines for each GPU if space allows
    if height > 8 && state.show_history
        push!(lines, "")  # Blank line
        spark_line = ""
        for (i, gpu_idx) in enumerate(state.gpu_indices)
            history = GPUMonitor.get_metric_history(state.monitor, gpu_idx, :utilization)
            if !isempty(history)
                spark = SparklineGraph.render_sparkline(
                    last(history, min(30, length(history))),
                    width = gpu_width - 2,
                    height = 1,
                    theme = state.theme,
                    show_axes = false
                )
                spark_line *= " " * spark * " "
            else
                spark_line *= rpad("", gpu_width)
            end
            if i < gpu_count
                spark_line *= "│"
            end
        end
        push!(lines, spark_line)
    end
end

# Render mini bar for comparison view
function render_mini_bar(value::Float64, width::Int, theme::ThemeConfig)
    if width < 3
        return ""
    end
    
    filled = round(Int, value / 100 * width)
    bar = "▰" ^ filled * "▱" ^ (width - filled)
    
    # Determine color
    color = if value > 90
        get_status_color(theme, :critical)
    elseif value > 70
        get_status_color(theme, :warning)
    else
        get_status_color(theme, :normal)
    end
    
    return apply_theme_color(bar[1:filled], color) * 
           apply_theme_color(bar[filled+1:end], get_status_color(theme, :dim))
end

end # module