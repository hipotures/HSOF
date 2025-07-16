module ColorTheme

using Printf

export ThemeConfig, ColorScheme, ThresholdConfig, MetricThreshold
export create_theme, get_color, interpolate_color, get_status_color
export apply_theme_color, format_with_status, create_gradient
export flash_alert, get_pattern_symbol, get_colorblind_pattern
export status_indicator, set_theme!, set_colorblind_mode!, add_threshold!

"""
Color scheme definition with RGB values
"""
struct ColorScheme
    name::Symbol
    normal::NamedTuple{(:r, :g, :b), Tuple{Int, Int, Int}}
    warning::NamedTuple{(:r, :g, :b), Tuple{Int, Int, Int}}
    critical::NamedTuple{(:r, :g, :b), Tuple{Int, Int, Int}}
    background::NamedTuple{(:r, :g, :b), Tuple{Int, Int, Int}}
    text::NamedTuple{(:r, :g, :b), Tuple{Int, Int, Int}}
    accent::NamedTuple{(:r, :g, :b), Tuple{Int, Int, Int}}
    dim::NamedTuple{(:r, :g, :b), Tuple{Int, Int, Int}}
end

"""
Threshold configuration for a specific metric
"""
struct MetricThreshold
    metric_name::Symbol
    warning_threshold::Float64
    critical_threshold::Float64
    invert::Bool  # If true, lower values are worse (e.g., accuracy)
    unit::String
    
    function MetricThreshold(metric_name::Symbol, warning::Float64, critical::Float64;
                           invert::Bool = false, unit::String = "")
        @assert warning != critical "Warning and critical thresholds must be different"
        if !invert
            @assert warning < critical "Warning threshold must be less than critical (unless inverted)"
        else
            @assert warning > critical "Warning threshold must be greater than critical (when inverted)"
        end
        new(metric_name, warning, critical, invert, unit)
    end
end

"""
Configuration for threshold-based coloring
"""
struct ThresholdConfig
    thresholds::Dict{Symbol, MetricThreshold}
    
    function ThresholdConfig(thresholds::Vector{MetricThreshold} = MetricThreshold[])
        threshold_dict = Dict{Symbol, MetricThreshold}()
        for t in thresholds
            threshold_dict[t.metric_name] = t
        end
        new(threshold_dict)
    end
end

"""
Theme configuration including color schemes and settings
"""
mutable struct ThemeConfig
    current_scheme::Symbol
    schemes::Dict{Symbol, ColorScheme}
    thresholds::ThresholdConfig
    colorblind_mode::Bool
    use_patterns::Bool
    flash_critical::Bool
    smooth_transitions::Bool
    
    function ThemeConfig(;
        current_scheme::Symbol = :dark,
        colorblind_mode::Bool = false,
        use_patterns::Bool = false,
        flash_critical::Bool = true,
        smooth_transitions::Bool = true
    )
        schemes = create_default_schemes()
        thresholds = create_default_thresholds()
        new(current_scheme, schemes, thresholds, colorblind_mode,
            use_patterns, flash_critical, smooth_transitions)
    end
end

"""
Create default color schemes
"""
function create_default_schemes()
    schemes = Dict{Symbol, ColorScheme}()
    
    # Dark theme (default)
    schemes[:dark] = ColorScheme(
        :dark,
        (r=34, g=139, b=34),    # Forest green for normal
        (r=255, g=215, b=0),    # Gold for warning
        (r=220, g=20, b=60),    # Crimson for critical
        (r=0, g=0, b=0),        # Black background
        (r=255, g=255, b=255),  # White text
        (r=30, g=144, b=255),   # Dodger blue accent
        (r=128, g=128, b=128)   # Gray for dim
    )
    
    # Light theme
    schemes[:light] = ColorScheme(
        :light,
        (r=0, g=128, b=0),      # Green for normal
        (r=255, g=140, b=0),    # Dark orange for warning
        (r=178, g=34, b=34),    # Fire brick for critical
        (r=255, g=255, b=255),  # White background
        (r=0, g=0, b=0),        # Black text
        (r=0, g=123, b=255),    # Bright blue accent
        (r=169, g=169, b=169)   # Dark gray for dim
    )
    
    # High contrast theme
    schemes[:high_contrast] = ColorScheme(
        :high_contrast,
        (r=0, g=255, b=0),      # Bright green for normal
        (r=255, g=255, b=0),    # Yellow for warning
        (r=255, g=0, b=0),      # Red for critical
        (r=0, g=0, b=0),        # Black background
        (r=255, g=255, b=255),  # White text
        (r=0, g=255, b=255),    # Cyan accent
        (r=192, g=192, b=192)   # Silver for dim
    )
    
    # Colorblind-friendly theme (deuteranopia/protanopia friendly)
    schemes[:colorblind] = ColorScheme(
        :colorblind,
        (r=0, g=114, b=178),    # Blue for normal
        (r=230, g=159, b=0),    # Orange for warning
        (r=213, g=94, b=0),     # Vermillion for critical
        (r=0, g=0, b=0),        # Black background
        (r=255, g=255, b=255),  # White text
        (r=86, g=180, b=233),   # Sky blue accent
        (r=128, g=128, b=128)   # Gray for dim
    )
    
    return schemes
end

"""
Create default threshold configurations
"""
function create_default_thresholds()
    thresholds = [
        # GPU metrics
        MetricThreshold(:gpu_temp, 80.0, 90.0, unit="°C"),
        MetricThreshold(:gpu_utilization, 80.0, 95.0, unit="%"),
        MetricThreshold(:gpu_memory, 80.0, 95.0, unit="%"),
        MetricThreshold(:gpu_power, 350.0, 450.0, unit="W"),
        MetricThreshold(:gpu_fan, 70.0, 90.0, unit="%"),
        
        # Performance metrics
        MetricThreshold(:nodes_per_sec, 5000.0, 1000.0, invert=true, unit="n/s"),
        MetricThreshold(:memory_bandwidth, 200.0, 100.0, invert=true, unit="GB/s"),
        MetricThreshold(:score, 0.7, 0.5, invert=true),
        
        # System metrics
        MetricThreshold(:cpu_usage, 70.0, 90.0, unit="%"),
        MetricThreshold(:memory_usage, 70.0, 90.0, unit="%"),
        MetricThreshold(:queue_size, 1000.0, 5000.0),
        
        # Error/warning counts
        MetricThreshold(:error_count, 1.0, 10.0),
        MetricThreshold(:warning_count, 5.0, 20.0),
    ]
    
    return ThresholdConfig(thresholds)
end

"""
Create a theme with custom settings
"""
function create_theme(;kwargs...)
    return ThemeConfig(;kwargs...)
end

"""
Get color for a given status
"""
function get_status_color(theme::ThemeConfig, status::Symbol)
    scheme = theme.schemes[theme.current_scheme]
    
    if status == :normal
        return scheme.normal
    elseif status == :warning
        return scheme.warning
    elseif status == :critical
        return scheme.critical
    elseif status == :background
        return scheme.background
    elseif status == :text
        return scheme.text
    elseif status == :accent
        return scheme.accent
    elseif status == :dim
        return scheme.dim
    else
        return scheme.text  # Default to text color
    end
end

"""
Get color based on metric value and thresholds
"""
function get_color(theme::ThemeConfig, metric::Symbol, value::Float64)
    if !haskey(theme.thresholds.thresholds, metric)
        return get_status_color(theme, :normal)
    end
    
    threshold = theme.thresholds.thresholds[metric]
    status = get_status(value, threshold)
    
    if theme.smooth_transitions
        # Interpolate between colors for smooth transitions
        return interpolate_status_color(theme, value, threshold)
    else
        return get_status_color(theme, status)
    end
end

"""
Determine status based on value and threshold
"""
function get_status(value::Float64, threshold::MetricThreshold)
    if threshold.invert
        # Lower is worse (e.g., accuracy, nodes/sec)
        if value <= threshold.critical_threshold
            return :critical
        elseif value <= threshold.warning_threshold
            return :warning
        else
            return :normal
        end
    else
        # Higher is worse (e.g., temperature, utilization)
        if value >= threshold.critical_threshold
            return :critical
        elseif value >= threshold.warning_threshold
            return :warning
        else
            return :normal
        end
    end
end

"""
Interpolate between colors for smooth transitions
"""
function interpolate_color(color1::NamedTuple, color2::NamedTuple, t::Float64)
    t = clamp(t, 0.0, 1.0)
    r = round(Int, color1.r * (1 - t) + color2.r * t)
    g = round(Int, color1.g * (1 - t) + color2.g * t)
    b = round(Int, color1.b * (1 - t) + color2.b * t)
    return (r=r, g=g, b=b)
end

"""
Interpolate color based on status transitions
"""
function interpolate_status_color(theme::ThemeConfig, value::Float64, threshold::MetricThreshold)
    scheme = theme.schemes[theme.current_scheme]
    
    if threshold.invert
        # Lower is worse
        if value <= threshold.critical_threshold
            return scheme.critical
        elseif value <= threshold.warning_threshold
            # Interpolate between critical and warning
            t = (value - threshold.critical_threshold) / (threshold.warning_threshold - threshold.critical_threshold)
            return interpolate_color(scheme.critical, scheme.warning, t)
        else
            # Interpolate between warning and normal
            # Use a reasonable range above warning (2x the warning-critical gap)
            range = threshold.warning_threshold - threshold.critical_threshold
            max_normal = threshold.warning_threshold + range
            if value >= max_normal
                return scheme.normal
            else
                t = (value - threshold.warning_threshold) / range
                return interpolate_color(scheme.warning, scheme.normal, t)
            end
        end
    else
        # Higher is worse
        if value >= threshold.critical_threshold
            return scheme.critical
        elseif value >= threshold.warning_threshold
            # Interpolate between warning and critical
            t = (value - threshold.warning_threshold) / (threshold.critical_threshold - threshold.warning_threshold)
            return interpolate_color(scheme.warning, scheme.critical, t)
        else
            # Interpolate between normal and warning
            # Use a reasonable range below warning (2x the warning-critical gap)
            range = threshold.critical_threshold - threshold.warning_threshold
            min_normal = threshold.warning_threshold - range
            if value <= min_normal || min_normal < 0
                return scheme.normal
            else
                t = (value - min_normal) / (threshold.warning_threshold - min_normal)
                return interpolate_color(scheme.normal, scheme.warning, t)
            end
        end
    end
end

"""
Apply theme color to text using ANSI escape codes
"""
function apply_theme_color(text::String, color::NamedTuple; background::Bool = false)
    if background
        # ANSI background color
        return "\033[48;2;$(color.r);$(color.g);$(color.b)m$text\033[0m"
    else
        # ANSI foreground color
        return "\033[38;2;$(color.r);$(color.g);$(color.b)m$text\033[0m"
    end
end

"""
Format value with appropriate color based on metric
"""
function format_with_status(theme::ThemeConfig, metric::Symbol, value::Float64, 
                          format_string::String = "%.1f")
    color = get_color(theme, metric, value)
    formatted_value = Printf.format(Printf.Format(format_string), value)
    
    # Add unit if available
    if haskey(theme.thresholds.thresholds, metric)
        unit = theme.thresholds.thresholds[metric].unit
        if !isempty(unit)
            formatted_value *= unit
        end
    end
    
    # Add pattern for colorblind mode
    if theme.colorblind_mode && theme.use_patterns
        status = get_status(value, theme.thresholds.thresholds[metric])
        pattern = get_pattern_symbol(status)
        formatted_value = "$pattern $formatted_value"
    end
    
    return apply_theme_color(formatted_value, color)
end

"""
Create a gradient between two colors
"""
function create_gradient(theme::ThemeConfig, start_status::Symbol, end_status::Symbol, steps::Int)
    start_color = get_status_color(theme, start_status)
    end_color = get_status_color(theme, end_status)
    
    gradient = []
    for i in 0:steps-1
        t = i / (steps - 1)
        color = interpolate_color(start_color, end_color, t)
        push!(gradient, color)
    end
    
    return gradient
end

"""
Flash alert for critical conditions
"""
function flash_alert(theme::ThemeConfig, text::String, duration::Float64 = 1.0, 
                    frequency::Float64 = 2.0)
    if !theme.flash_critical
        return text
    end
    
    # Calculate if we should show or hide based on time
    current_time = time()
    cycle_position = (current_time * frequency) % 1.0
    show = cycle_position < 0.5
    
    if show
        color = get_status_color(theme, :critical)
        return apply_theme_color(text, color, background=true)
    else
        return text
    end
end

"""
Get pattern symbol for colorblind mode
"""
function get_pattern_symbol(status::Symbol)
    if status == :normal
        return "●"  # Filled circle
    elseif status == :warning
        return "▲"  # Triangle
    elseif status == :critical
        return "■"  # Square
    else
        return " "
    end
end

"""
Get colorblind-friendly pattern for backgrounds
"""
function get_colorblind_pattern(status::Symbol)
    if status == :normal
        return "····"  # Dots
    elseif status == :warning
        return "////"  # Forward slashes
    elseif status == :critical
        return "XXXX"  # X pattern
    else
        return "    "
    end
end


"""
Set theme for the configuration
"""
function set_theme!(theme::ThemeConfig, scheme::Symbol)
    if haskey(theme.schemes, scheme)
        theme.current_scheme = scheme
    else
        error("Unknown theme: $scheme")
    end
end

"""
Enable or disable colorblind mode
"""
function set_colorblind_mode!(theme::ThemeConfig, enabled::Bool, use_patterns::Bool = true)
    theme.colorblind_mode = enabled
    theme.use_patterns = use_patterns
    if enabled
        theme.current_scheme = :colorblind
    end
end

"""
Add custom threshold for a metric
"""
function add_threshold!(theme::ThemeConfig, threshold::MetricThreshold)
    theme.thresholds.thresholds[threshold.metric_name] = threshold
end

"""
Create a simple status indicator
"""
function status_indicator(theme::ThemeConfig, metric::Symbol, value::Float64)
    if !haskey(theme.thresholds.thresholds, metric)
        return "?"
    end
    
    threshold = theme.thresholds.thresholds[metric]
    status = get_status(value, threshold)
    
    if theme.colorblind_mode && theme.use_patterns
        symbol = get_pattern_symbol(status)
    else
        symbol = "●"
    end
    
    color = get_status_color(theme, status)
    return apply_theme_color(symbol, color)
end

end # module