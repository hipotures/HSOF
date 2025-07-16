module ConsoleDashboard

using Term
using Term.Layout
using Term.Panels
using Term.Progress
using Term.Renderables
using Term.Measures
using Term.Boxes
using Term.Style
using Term.Colors
using Dates
using Printf

export DashboardLayout, DashboardConfig, PanelContent
export create_dashboard, update_dashboard!, update_panel!, render_dashboard
export GPUPanelContent, ProgressPanelContent, MetricsPanelContent
export AnalysisPanelContent, LogPanelContent
export check_terminal_resize, handle_resize!

"""
Configuration for the dashboard layout and appearance
"""
struct DashboardConfig
    refresh_rate_ms::Int        # Refresh rate in milliseconds (default 100ms)
    color_scheme::Symbol        # :default, :dark, :light, :high_contrast
    border_style::Symbol        # :single, :double, :rounded, :heavy
    show_timestamps::Bool       # Show timestamps in panels
    responsive::Bool            # Enable responsive sizing
    min_width::Int             # Minimum terminal width required
    min_height::Int            # Minimum terminal height required
    
    function DashboardConfig(;
        refresh_rate_ms::Int = 100,
        color_scheme::Symbol = :default,
        border_style::Symbol = :rounded,
        show_timestamps::Bool = true,
        responsive::Bool = true,
        min_width::Int = 80,
        min_height::Int = 24
    )
        @assert refresh_rate_ms > 0 "Refresh rate must be positive"
        @assert color_scheme in [:default, :dark, :light, :high_contrast]
        @assert border_style in [:single, :double, :rounded, :heavy]
        @assert min_width >= 60 "Minimum width must be at least 60"
        @assert min_height >= 20 "Minimum height must be at least 20"
        
        new(refresh_rate_ms, color_scheme, border_style, show_timestamps,
            responsive, min_width, min_height)
    end
end

"""
Abstract type for panel content
"""
abstract type PanelContent end

"""
GPU panel content
"""
struct GPUPanelContent <: PanelContent
    gpu_id::Int
    utilization::Float64      # 0-100%
    memory_used::Float64      # GB
    memory_total::Float64     # GB
    temperature::Float64      # Celsius
    power_draw::Float64       # Watts
    clock_speed::Float64      # MHz
    fan_speed::Float64        # 0-100%
end

"""
Progress panel content
"""
struct ProgressPanelContent <: PanelContent
    stage::String             # Current stage name
    overall_progress::Float64 # 0-100%
    stage_progress::Float64   # 0-100%
    items_processed::Int
    items_total::Int
    current_score::Float64
    best_score::Float64
    eta_seconds::Int
end

"""
Metrics panel content
"""
struct MetricsPanelContent <: PanelContent
    nodes_per_second::Float64
    bandwidth_gbps::Float64
    cpu_usage::Float64        # 0-100%
    ram_usage::Float64        # GB
    cache_hit_rate::Float64   # 0-100%
    queue_depth::Int
    active_threads::Int
end

"""
Analysis panel content
"""
struct AnalysisPanelContent <: PanelContent
    total_features::Int
    selected_features::Int
    reduction_percentage::Float64
    top_features::Vector{Tuple{String, Float64}}  # (name, score) pairs
    correlation_matrix_summary::String
end

"""
Log panel content
"""
struct LogPanelContent <: PanelContent
    entries::Vector{Tuple{DateTime, Symbol, String}}  # (timestamp, level, message)
    max_entries::Int
end

"""
Main dashboard layout structure
"""
mutable struct DashboardLayout
    config::DashboardConfig
    terminal_width::Int
    terminal_height::Int
    panels::Dict{Symbol, Any}  # Panel objects from Term.jl
    panel_contents::Dict{Symbol, PanelContent}  # Store content separately
    layout::Any  # Layout object from Term.jl
    
    function DashboardLayout(config::DashboardConfig = DashboardConfig())
        # Get terminal dimensions
        term_width, term_height = get_terminal_size()
        
        # Check minimum requirements
        if term_width < config.min_width || term_height < config.min_height
            error("Terminal too small. Need at least $(config.min_width)x$(config.min_height), got $(term_width)x$(term_height)")
        end
        
        # Create empty panels dictionary
        panels = Dict{Symbol, Any}()
        panel_contents = Dict{Symbol, PanelContent}()
        
        # Create layout
        layout = create_layout_structure(term_width, term_height, config)
        
        new(config, term_width, term_height, panels, panel_contents, layout)
    end
end

"""
Get terminal dimensions
"""
function get_terminal_size()
    # Term.jl provides terminal size functionality
    width = Term.console_width()
    height = Term.console_height()
    return width, height
end

"""
Create the 2x3 grid layout structure
"""
function create_layout_structure(width::Int, height::Int, config::DashboardConfig)
    # Calculate panel dimensions
    # Account for borders and padding
    border_width = 2  # Left and right borders
    border_height = 2 # Top and bottom borders
    padding = 1       # Internal padding
    
    # Calculate available space
    avail_width = width - 3  # Account for column separators
    avail_height = height - 3 # Account for row separators
    
    # Calculate panel sizes (2x3 grid)
    panel_width = div(avail_width, 3)
    panel_height = div(avail_height, 2)
    
    # Adjust for responsive sizing if enabled
    if config.responsive
        panel_width, panel_height = adjust_panel_sizes(width, height, panel_width, panel_height)
    end
    
    # Create layout grid string
    # Use Term.jl layout syntax
    layout_str = """
    ╭─────────────┬─────────────┬─────────────╮
    │ gpu1        │ gpu2        │ progress    │
    ├─────────────┼─────────────┼─────────────┤
    │ metrics     │ analysis    │ log         │
    ╰─────────────┴─────────────┴─────────────╯
    """
    
    # Create the actual layout
    # For now, just return the layout string - will be rendered later
    layout = layout_str
    
    return layout
end

"""
Adjust panel sizes for responsive design
"""
function adjust_panel_sizes(term_width::Int, term_height::Int, base_width::Int, base_height::Int)
    # Implement responsive sizing logic
    # Adjust based on terminal aspect ratio
    aspect_ratio = term_width / term_height
    
    if aspect_ratio > 2.5  # Very wide terminal
        # Make panels wider
        adjusted_width = round(Int, base_width * 1.2)
        adjusted_height = base_height
    elseif aspect_ratio < 1.5  # Narrow terminal
        # Make panels taller
        adjusted_width = base_width
        adjusted_height = round(Int, base_height * 1.2)
    else
        # Standard sizing
        adjusted_width = base_width
        adjusted_height = base_height
    end
    
    return adjusted_width, adjusted_height
end

"""
Create dashboard with initial layout
"""
function create_dashboard(config::DashboardConfig = DashboardConfig())
    dashboard = DashboardLayout(config)
    
    # Initialize panels
    initialize_panels!(dashboard)
    
    return dashboard
end

"""
Initialize all panels with default content
"""
function initialize_panels!(dashboard::DashboardLayout)
    # Get border style
    border_box = get_border_box(dashboard.config.border_style)
    
    # Initialize default content
    dashboard.panel_contents[:gpu1] = GPUPanelContent(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    dashboard.panel_contents[:gpu2] = GPUPanelContent(2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    dashboard.panel_contents[:progress] = ProgressPanelContent("Initializing", 0.0, 0.0, 0, 0, 0.0, 0.0, 0)
    dashboard.panel_contents[:metrics] = MetricsPanelContent(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
    dashboard.panel_contents[:analysis] = AnalysisPanelContent(0, 0, 0.0, Tuple{String, Float64}[], "")
    dashboard.panel_contents[:log] = LogPanelContent(Tuple{DateTime, Symbol, String}[], 100)
    
    # Create GPU panels
    dashboard.panels[:gpu1] = create_gpu_panel(1, dashboard, border_box)
    dashboard.panels[:gpu2] = create_gpu_panel(2, dashboard, border_box)
    
    # Create other panels
    dashboard.panels[:progress] = create_progress_panel(dashboard, border_box)
    dashboard.panels[:metrics] = create_metrics_panel(dashboard, border_box)
    dashboard.panels[:analysis] = create_analysis_panel(dashboard, border_box)
    dashboard.panels[:log] = create_log_panel(dashboard, border_box)
end

"""
Get border box style based on configuration
"""
function get_border_box(style::Symbol)
    if style == :single
        return BOXES[:SQUARE]
    elseif style == :double
        return BOXES[:DOUBLE]
    elseif style == :rounded
        return BOXES[:ROUNDED]
    elseif style == :heavy
        return BOXES[:HEAVY]
    else
        return BOXES[:ROUNDED]
    end
end

"""
Create GPU status panel
"""
function create_gpu_panel(gpu_id::Int, dashboard::DashboardLayout, border_box)
    title = " GPU $gpu_id Status "
    
    # Create empty content initially
    content = render_gpu_content(GPUPanelContent(
        gpu_id, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ), dashboard.config)
    
    panel = Panel(
        content,
        title = title,
        title_style = "bold bright_blue",
        border_box = border_box,
        border_style = "bright_blue",
        fit = false,
        width = div(dashboard.terminal_width, 3) - 2,
        height = div(dashboard.terminal_height, 2) - 2
    )
    
    return panel
end

"""
Create progress panel
"""
function create_progress_panel(dashboard::DashboardLayout, border_box)
    title = " Search Progress "
    
    # Create empty content initially
    content = render_progress_content(ProgressPanelContent(
        "Initializing", 0.0, 0.0, 0, 0, 0.0, 0.0, 0
    ), dashboard.config)
    
    panel = Panel(
        content,
        title = title,
        title_style = "bold bright_green",
        border_box = border_box,
        border_style = "bright_green",
        fit = false,
        width = div(dashboard.terminal_width, 3) - 2,
        height = div(dashboard.terminal_height, 2) - 2
    )
    
    return panel
end

"""
Create metrics panel
"""
function create_metrics_panel(dashboard::DashboardLayout, border_box)
    title = " Performance Metrics "
    
    content = render_metrics_content(MetricsPanelContent(
        0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
    ), dashboard.config)
    
    panel = Panel(
        content,
        title = title,
        title_style = "bold bright_yellow",
        border_box = border_box,
        border_style = "bright_yellow",
        fit = false,
        width = div(dashboard.terminal_width, 3) - 2,
        height = div(dashboard.terminal_height, 2) - 2
    )
    
    return panel
end

"""
Create analysis panel
"""
function create_analysis_panel(dashboard::DashboardLayout, border_box)
    title = " Feature Analysis "
    
    content = render_analysis_content(AnalysisPanelContent(
        0, 0, 0.0, Tuple{String, Float64}[], ""
    ), dashboard.config)
    
    panel = Panel(
        content,
        title = title,
        title_style = "bold bright_magenta",
        border_box = border_box,
        border_style = "bright_magenta",
        fit = false,
        width = div(dashboard.terminal_width, 3) - 2,
        height = div(dashboard.terminal_height, 2) - 2
    )
    
    return panel
end

"""
Create log panel
"""
function create_log_panel(dashboard::DashboardLayout, border_box)
    title = " System Log "
    
    content = render_log_content(LogPanelContent(
        Tuple{DateTime, Symbol, String}[], 100
    ), dashboard.config)
    
    panel = Panel(
        content,
        title = title,
        title_style = "bold bright_cyan",
        border_box = border_box,
        border_style = "bright_cyan",
        fit = false,
        width = div(dashboard.terminal_width, 3) - 2,
        height = div(dashboard.terminal_height, 2) - 2
    )
    
    return panel
end

"""
Render GPU panel content
"""
function render_gpu_content(content::GPUPanelContent, config::DashboardConfig)
    lines = String[]
    
    # Utilization bar
    util_bar = create_progress_bar(content.utilization, 100.0, 20, :utilization)
    push!(lines, "Util: $util_bar $(round(content.utilization, digits=1))%")
    
    # Memory usage
    mem_used_gb = round(content.memory_used, digits=2)
    mem_total_gb = round(content.memory_total, digits=2)
    mem_percent = content.memory_total > 0 ? (content.memory_used / content.memory_total * 100) : 0.0
    mem_bar = create_progress_bar(mem_percent, 100.0, 20, :memory)
    push!(lines, "Mem:  $mem_bar $mem_used_gb/$mem_total_gb GB")
    
    # Temperature
    temp_color = get_temperature_color(content.temperature)
    temp_str = @sprintf("%.1f°C", content.temperature)
    push!(lines, "Temp: $(apply_color(temp_str, temp_color))")
    
    # Power and clock
    push!(lines, @sprintf("Power: %.1f W | Clock: %.0f MHz", content.power_draw, content.clock_speed))
    
    # Fan speed
    fan_bar = create_progress_bar(content.fan_speed, 100.0, 20, :fan)
    push!(lines, "Fan:  $fan_bar $(round(content.fan_speed, digits=1))%")
    
    return join(lines, "\n")
end

"""
Render progress panel content
"""
function render_progress_content(content::ProgressPanelContent, config::DashboardConfig)
    lines = String[]
    
    # Stage info
    push!(lines, "Stage: $(apply_style(content.stage, "bold bright_white"))")
    push!(lines, "")
    
    # Overall progress
    overall_bar = create_progress_bar(content.overall_progress, 100.0, 25, :overall)
    push!(lines, "Overall: $overall_bar $(round(content.overall_progress, digits=1))%")
    
    # Stage progress
    stage_bar = create_progress_bar(content.stage_progress, 100.0, 25, :stage)
    push!(lines, "Stage:   $stage_bar $(round(content.stage_progress, digits=1))%")
    push!(lines, "")
    
    # Items processed
    push!(lines, "Items: $(content.items_processed) / $(content.items_total)")
    
    # Scores
    current_str = @sprintf("%.6f", content.current_score)
    best_str = @sprintf("%.6f", content.best_score)
    push!(lines, "Score: $current_str (Best: $best_str)")
    
    # ETA
    if content.eta_seconds > 0
        eta_str = format_duration(content.eta_seconds)
        push!(lines, "ETA: $eta_str")
    end
    
    return join(lines, "\n")
end

"""
Render metrics panel content
"""
function render_metrics_content(content::MetricsPanelContent, config::DashboardConfig)
    lines = String[]
    
    # Performance metrics
    push!(lines, @sprintf("Nodes/sec: %.2f", content.nodes_per_second))
    push!(lines, @sprintf("Bandwidth: %.2f GB/s", content.bandwidth_gbps))
    push!(lines, "")
    
    # System usage
    cpu_bar = create_progress_bar(content.cpu_usage, 100.0, 15, :cpu)
    push!(lines, "CPU: $cpu_bar $(round(content.cpu_usage, digits=1))%")
    
    push!(lines, @sprintf("RAM: %.2f GB", content.ram_usage))
    
    # Cache
    cache_bar = create_progress_bar(content.cache_hit_rate, 100.0, 15, :cache)
    push!(lines, "Cache: $cache_bar $(round(content.cache_hit_rate, digits=1))%")
    
    # Threading
    push!(lines, "")
    push!(lines, "Threads: $(content.active_threads) | Queue: $(content.queue_depth)")
    
    return join(lines, "\n")
end

"""
Render analysis panel content
"""
function render_analysis_content(content::AnalysisPanelContent, config::DashboardConfig)
    lines = String[]
    
    # Feature statistics
    push!(lines, "Total Features: $(content.total_features)")
    push!(lines, "Selected: $(content.selected_features)")
    
    if content.total_features > 0
        reduction_str = @sprintf("%.1f%%", content.reduction_percentage)
        color = content.reduction_percentage > 80 ? "bright_green" : 
                content.reduction_percentage > 60 ? "bright_yellow" : "bright_red"
        push!(lines, "Reduction: $(apply_color(reduction_str, color))")
    end
    
    push!(lines, "")
    
    # Top features
    if !isempty(content.top_features)
        push!(lines, "Top Features:")
        for (i, (name, score)) in enumerate(content.top_features[1:min(5, end)])
            score_str = @sprintf("%.4f", score)
            push!(lines, "  $i. $name: $score_str")
        end
    end
    
    # Correlation summary
    if !isempty(content.correlation_matrix_summary)
        push!(lines, "")
        push!(lines, content.correlation_matrix_summary)
    end
    
    return join(lines, "\n")
end

"""
Render log panel content
"""
function render_log_content(content::LogPanelContent, config::DashboardConfig)
    lines = String[]
    
    # Show last N entries
    start_idx = max(1, length(content.entries) - 8)  # Show last 8 entries
    
    for (timestamp, level, message) in content.entries[start_idx:end]
        # Format timestamp
        time_str = config.show_timestamps ? 
                   Dates.format(timestamp, "HH:MM:SS") * " " : ""
        
        # Color code by level
        level_color = level == :error ? "bright_red" :
                     level == :warn ? "bright_yellow" :
                     level == :info ? "bright_blue" : "white"
        
        level_str = apply_color(string(level), level_color)
        
        # Truncate message if needed
        max_msg_len = 35 - length(time_str)
        msg = length(message) > max_msg_len ? 
              message[1:max_msg_len-3] * "..." : message
        
        push!(lines, "$time_str[$level_str] $msg")
    end
    
    return join(lines, "\n")
end

"""
Create a progress bar
"""
function create_progress_bar(value::Float64, max_value::Float64, width::Int, type::Symbol)
    # Calculate fill percentage
    percentage = clamp(value / max_value, 0.0, 1.0)
    filled = round(Int, percentage * width)
    
    # Choose characters based on type
    if type == :utilization
        fill_char = "█"
        empty_char = "░"
        color = percentage > 0.9 ? "bright_red" :
                percentage > 0.7 ? "bright_yellow" : "bright_green"
    elseif type == :memory
        fill_char = "▓"
        empty_char = "░"
        color = percentage > 0.9 ? "bright_red" :
                percentage > 0.7 ? "bright_yellow" : "bright_blue"
    elseif type == :overall || type == :stage
        fill_char = "━"
        empty_char = "─"
        color = "bright_cyan"
    elseif type == :cpu
        fill_char = "▪"
        empty_char = "▫"
        color = percentage > 0.8 ? "bright_red" : "bright_green"
    else
        fill_char = "■"
        empty_char = "□"
        color = "white"
    end
    
    # Build bar
    bar = apply_color(fill_char^filled, color) * empty_char^(width - filled)
    
    return "[$bar]"
end

"""
Get color based on temperature
"""
function get_temperature_color(temp::Float64)
    if temp >= 90
        return "bright_red"
    elseif temp >= 80
        return "bright_yellow"
    elseif temp >= 70
        return "yellow"
    else
        return "bright_green"
    end
end

"""
Apply color to text
"""
function apply_color(text::String, color::String)
    # Use Term.jl color styling
    return "{$color}$text{/}"
end

"""
Apply style to text
"""
function apply_style(text::String, style::String)
    # Use Term.jl style syntax
    return "{$style}$text{/}"
end

"""
Format duration in seconds to human readable
"""
function format_duration(seconds::Int)
    if seconds < 60
        return "$(seconds)s"
    elseif seconds < 3600
        mins = div(seconds, 60)
        secs = seconds % 60
        return "$(mins)m $(secs)s"
    else
        hours = div(seconds, 3600)
        mins = div(seconds % 3600, 60)
        return "$(hours)h $(mins)m"
    end
end

"""
Update dashboard with new content
"""
function update_dashboard!(dashboard::DashboardLayout, updates::Dict{Symbol, PanelContent})
    for (panel_key, content) in updates
        if haskey(dashboard.panel_contents, panel_key)
            dashboard.panel_contents[panel_key] = content
        end
    end
    
    # Recreate affected panels
    border_box = get_border_box(dashboard.config.border_style)
    for panel_key in keys(updates)
        if panel_key == :gpu1
            content = render_gpu_content(dashboard.panel_contents[:gpu1], dashboard.config)
            dashboard.panels[:gpu1] = Panel(
                content,
                title = " GPU 1 Status ",
                title_style = "bold bright_blue",
                border_box = border_box,
                border_style = "bright_blue",
                fit = false,
                width = div(dashboard.terminal_width, 3) - 2,
                height = div(dashboard.terminal_height, 2) - 2
            )
        elseif panel_key == :gpu2
            content = render_gpu_content(dashboard.panel_contents[:gpu2], dashboard.config)
            dashboard.panels[:gpu2] = Panel(
                content,
                title = " GPU 2 Status ",
                title_style = "bold bright_blue",
                border_box = border_box,
                border_style = "bright_blue",
                fit = false,
                width = div(dashboard.terminal_width, 3) - 2,
                height = div(dashboard.terminal_height, 2) - 2
            )
        elseif panel_key == :progress
            content = render_progress_content(dashboard.panel_contents[:progress], dashboard.config)
            dashboard.panels[:progress] = Panel(
                content,
                title = " Search Progress ",
                title_style = "bold bright_green",
                border_box = border_box,
                border_style = "bright_green",
                fit = false,
                width = div(dashboard.terminal_width, 3) - 2,
                height = div(dashboard.terminal_height, 2) - 2
            )
        elseif panel_key == :metrics
            content = render_metrics_content(dashboard.panel_contents[:metrics], dashboard.config)
            dashboard.panels[:metrics] = Panel(
                content,
                title = " Performance Metrics ",
                title_style = "bold bright_yellow",
                border_box = border_box,
                border_style = "bright_yellow",
                fit = false,
                width = div(dashboard.terminal_width, 3) - 2,
                height = div(dashboard.terminal_height, 2) - 2
            )
        elseif panel_key == :analysis
            content = render_analysis_content(dashboard.panel_contents[:analysis], dashboard.config)
            dashboard.panels[:analysis] = Panel(
                content,
                title = " Feature Analysis ",
                title_style = "bold bright_magenta",
                border_box = border_box,
                border_style = "bright_magenta",
                fit = false,
                width = div(dashboard.terminal_width, 3) - 2,
                height = div(dashboard.terminal_height, 2) - 2
            )
        elseif panel_key == :log
            content = render_log_content(dashboard.panel_contents[:log], dashboard.config)
            dashboard.panels[:log] = Panel(
                content,
                title = " System Log ",
                title_style = "bold bright_cyan",
                border_box = border_box,
                border_style = "bright_cyan",
                fit = false,
                width = div(dashboard.terminal_width, 3) - 2,
                height = div(dashboard.terminal_height, 2) - 2
            )
        end
    end
end

"""
Update individual panel
"""
function update_panel!(dashboard::DashboardLayout, panel_key::Symbol, content::PanelContent)
    update_dashboard!(dashboard, Dict{Symbol, PanelContent}(panel_key => content))
end

"""
Render the complete dashboard
"""
function render_dashboard(dashboard::DashboardLayout)
    # Create a grid layout
    # Term.jl uses a different approach than Python Rich
    
    # Top row
    top_row = hstack(
        dashboard.panels[:gpu1],
        dashboard.panels[:gpu2],
        dashboard.panels[:progress];
        pad = 1
    )
    
    # Bottom row
    bottom_row = hstack(
        dashboard.panels[:metrics],
        dashboard.panels[:analysis],
        dashboard.panels[:log];
        pad = 1
    )
    
    # Combine rows
    full_layout = vstack(top_row, bottom_row; pad = 0)
    
    return full_layout
end

"""
Check if terminal size has changed
"""
function check_terminal_resize(dashboard::DashboardLayout)
    current_width, current_height = get_terminal_size()
    
    if current_width != dashboard.terminal_width || current_height != dashboard.terminal_height
        # Terminal was resized
        return true, current_width, current_height
    end
    
    return false, current_width, current_height
end

"""
Handle terminal resize
"""
function handle_resize!(dashboard::DashboardLayout, new_width::Int, new_height::Int)
    # Update dimensions
    dashboard.terminal_width = new_width
    dashboard.terminal_height = new_height
    
    # Recreate layout
    dashboard.layout = create_layout_structure(new_width, new_height, dashboard.config)
    
    # Recreate all panels with new dimensions
    border_box = get_border_box(dashboard.config.border_style)
    panel_width = div(new_width, 3) - 2
    panel_height = div(new_height, 2) - 2
    
    # Update all panels
    for (key, content) in dashboard.panel_contents
        rendered_content = if key == :gpu1 || key == :gpu2
            render_gpu_content(content, dashboard.config)
        elseif key == :progress
            render_progress_content(content, dashboard.config)
        elseif key == :metrics
            render_metrics_content(content, dashboard.config)
        elseif key == :analysis
            render_analysis_content(content, dashboard.config)
        elseif key == :log
            render_log_content(content, dashboard.config)
        end
        
        title = if key == :gpu1
            " GPU 1 Status "
        elseif key == :gpu2
            " GPU 2 Status "
        elseif key == :progress
            " Search Progress "
        elseif key == :metrics
            " Performance Metrics "
        elseif key == :analysis
            " Feature Analysis "
        elseif key == :log
            " System Log "
        else
            " Panel "
        end
        
        style = if key in [:gpu1, :gpu2]
            "bright_blue"
        elseif key == :progress
            "bright_green"
        elseif key == :metrics
            "bright_yellow"
        elseif key == :analysis
            "bright_magenta"
        elseif key == :log
            "bright_cyan"
        else
            "white"
        end
        
        dashboard.panels[key] = Panel(
            rendered_content,
            title = title,
            title_style = "bold $style",
            border_box = border_box,
            border_style = style,
            fit = false,
            width = panel_width,
            height = panel_height
        )
    end
    
    return dashboard
end

end # module