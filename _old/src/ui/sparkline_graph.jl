module SparklineGraph

using Dates
using Statistics
using Printf

export CircularBuffer, SparklineConfig, SparklineRenderer
export push_data!, get_data, get_recent_data, clear_buffer!
export create_sparkline, create_colored_sparkline, create_bar_sparkline
export smooth_data, apply_gradient_colors

"""
Circular buffer for efficient storage of time-series data
"""
mutable struct CircularBuffer{T}
    data::Vector{T}
    timestamps::Vector{DateTime}
    capacity::Int
    size::Int
    head::Int
    
    function CircularBuffer{T}(capacity::Int) where T
        @assert capacity > 0 "Buffer capacity must be positive"
        data = Vector{T}(undef, capacity)
        timestamps = Vector{DateTime}(undef, capacity)
        new(data, timestamps, capacity, 0, 1)
    end
end

"""
Configuration for sparkline rendering
"""
struct SparklineConfig
    width::Int                    # Number of characters for sparkline
    height::Int                   # Number of vertical levels (usually 8)
    smooth::Bool                  # Apply smoothing
    smooth_window::Int            # Window size for smoothing
    gradient::Bool                # Use color gradient
    min_color::String            # Color for minimum values
    max_color::String            # Color for maximum values
    show_boundaries::Bool         # Show min/max markers
    unicode_blocks::Vector{Char}  # Unicode characters for levels
    
    function SparklineConfig(;
        width::Int = 20,
        height::Int = 8,
        smooth::Bool = false,
        smooth_window::Int = 3,
        gradient::Bool = false,
        min_color::String = "green",
        max_color::String = "red",
        show_boundaries::Bool = false,
        unicode_blocks::Vector{Char} = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    )
        @assert width > 0 "Width must be positive"
        @assert height > 0 "Height must be positive"
        @assert smooth_window > 0 "Smooth window must be positive"
        @assert length(unicode_blocks) == height "Unicode blocks must match height"
        
        new(width, height, smooth, smooth_window, gradient,
            min_color, max_color, show_boundaries, unicode_blocks)
    end
end

"""
Main sparkline renderer
"""
mutable struct SparklineRenderer
    config::SparklineConfig
    buffer::CircularBuffer{Float64}
    min_value::Float64
    max_value::Float64
    auto_scale::Bool
    
    function SparklineRenderer(config::SparklineConfig, buffer_size::Int = 600;
                              auto_scale::Bool = true)
        buffer = CircularBuffer{Float64}(buffer_size)
        new(config, buffer, Inf, -Inf, auto_scale)
    end
end

# Circular Buffer Methods

"""
Push new data point to circular buffer
"""
function push_data!(buffer::CircularBuffer{T}, value::T, timestamp::DateTime = now()) where T
    if buffer.size < buffer.capacity
        buffer.size += 1
    end
    
    buffer.data[buffer.head] = value
    buffer.timestamps[buffer.head] = timestamp
    
    # Move head forward (circular)
    buffer.head = mod1(buffer.head + 1, buffer.capacity)
end

"""
Get all valid data from buffer in chronological order
"""
function get_data(buffer::CircularBuffer{T}) where T
    if buffer.size == 0
        return T[], DateTime[]
    end
    
    if buffer.size < buffer.capacity
        # Buffer not full yet
        return buffer.data[1:buffer.size], buffer.timestamps[1:buffer.size]
    else
        # Buffer is full, return in correct order
        tail = buffer.head
        indices = vcat(tail:buffer.capacity, 1:tail-1)
        return buffer.data[indices], buffer.timestamps[indices]
    end
end

"""
Get recent data points from buffer
"""
function get_recent_data(buffer::CircularBuffer{T}, n::Int) where T
    data, timestamps = get_data(buffer)
    if length(data) <= n
        return data, timestamps
    else
        return data[end-n+1:end], timestamps[end-n+1:end]
    end
end

"""
Clear all data from buffer
"""
function clear_buffer!(buffer::CircularBuffer)
    buffer.size = 0
    buffer.head = 1
end

# Sparkline Rendering Methods

"""
Create basic sparkline from data
"""
function create_sparkline(data::Vector{Float64}, config::SparklineConfig = SparklineConfig())
    if isempty(data)
        return ""
    end
    
    # Apply smoothing if requested
    smoothed_data = config.smooth ? smooth_data(data, config.smooth_window) : data
    
    # Determine sparkline width
    sparkline_width = min(length(smoothed_data), config.width)
    
    # Downsample or interpolate to match desired width
    display_data = if length(smoothed_data) > sparkline_width
        resample_data(smoothed_data, sparkline_width)
    else
        smoothed_data
    end
    
    # Get range for normalization
    min_val = minimum(display_data)
    max_val = maximum(display_data)
    range = max_val - min_val
    
    # Handle flat lines
    if range ≈ 0
        mid_char = config.unicode_blocks[div(config.height, 2)]
        return string(mid_char)^length(display_data)
    end
    
    # Create sparkline
    sparkline = ""
    for val in display_data
        normalized = (val - min_val) / range
        level = round(Int, normalized * (config.height - 1)) + 1
        level = clamp(level, 1, config.height)
        sparkline *= config.unicode_blocks[level]
    end
    
    # Add boundaries if requested
    if config.show_boundaries
        min_str = @sprintf("%.1f", min_val)
        max_str = @sprintf("%.1f", max_val)
        sparkline = "$min_str $sparkline $max_str"
    end
    
    return sparkline
end

"""
Create colored sparkline with gradient
"""
function create_colored_sparkline(data::Vector{Float64}, config::SparklineConfig)
    if !config.gradient
        return create_sparkline(data, config)
    end
    
    # Get basic sparkline
    basic_sparkline = create_sparkline(data, config)
    
    if isempty(basic_sparkline)
        return ""
    end
    
    # Apply smoothing if requested
    smoothed_data = config.smooth ? smooth_data(data, config.smooth_window) : data
    display_data = resample_data(smoothed_data, config.width)
    
    # Apply gradient colors
    min_val = minimum(display_data)
    max_val = maximum(display_data)
    range = max_val - min_val
    
    colored_sparkline = ""
    chars = collect(basic_sparkline)
    
    for (i, val) in enumerate(display_data)
        if i <= length(chars)
            normalized = range > 0 ? (val - min_val) / range : 0.5
            color = interpolate_color(config.min_color, config.max_color, normalized)
            colored_sparkline *= apply_color(string(chars[i]), color)
        end
    end
    
    return colored_sparkline
end

"""
Create bar-style sparkline
"""
function create_bar_sparkline(data::Vector{Float64}, config::SparklineConfig)
    if isempty(data)
        return ""
    end
    
    # Bar characters
    bar_chars = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    bar_config = SparklineConfig(
        width = config.width,
        height = length(bar_chars),
        smooth = config.smooth,
        smooth_window = config.smooth_window,
        gradient = config.gradient,
        min_color = config.min_color,
        max_color = config.max_color,
        show_boundaries = config.show_boundaries,
        unicode_blocks = bar_chars
    )
    
    return create_sparkline(data, bar_config)
end

# Helper Methods

"""
Smooth data using moving average
"""
function smooth_data(data::Vector{Float64}, window::Int)
    if length(data) <= window
        return data
    end
    
    smoothed = similar(data)
    half_window = div(window, 2)
    
    for i in 1:length(data)
        start_idx = max(1, i - half_window)
        end_idx = min(length(data), i + half_window)
        smoothed[i] = mean(data[start_idx:end_idx])
    end
    
    return smoothed
end

"""
Apply exponential smoothing
"""
function exponential_smoothing(data::Vector{Float64}, alpha::Float64 = 0.3)
    if isempty(data)
        return data
    end
    
    smoothed = similar(data)
    smoothed[1] = data[1]
    
    for i in 2:length(data)
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    end
    
    return smoothed
end

"""
Resample data to match target width
"""
function resample_data(data::Vector{Float64}, target_width::Int)
    n = length(data)
    
    if n == target_width
        return data
    elseif n > target_width
        # Downsample
        indices = round.(Int, range(1, n, length=target_width))
        return data[indices]
    else
        # Upsample with linear interpolation
        resampled = zeros(target_width)
        for i in 1:target_width
            src_idx = (i - 1) * (n - 1) / (target_width - 1) + 1
            idx1 = floor(Int, src_idx)
            idx2 = min(ceil(Int, src_idx), n)
            
            if idx1 == idx2
                resampled[i] = data[idx1]
            else
                t = src_idx - idx1
                resampled[i] = data[idx1] * (1 - t) + data[idx2] * t
            end
        end
        return resampled
    end
end

"""
Apply ANSI color to text
"""
function apply_color(text::String, color::String)
    colors = Dict(
        "black" => 30, "red" => 31, "green" => 32, "yellow" => 33,
        "blue" => 34, "magenta" => 35, "cyan" => 36, "white" => 37,
        "bright_black" => 90, "bright_red" => 91, "bright_green" => 92,
        "bright_yellow" => 93, "bright_blue" => 94, "bright_magenta" => 95,
        "bright_cyan" => 96, "bright_white" => 97
    )
    
    code = get(colors, color, 37)  # Default to white
    return "\033[$(code)m$text\033[0m"
end

"""
Interpolate between two colors
"""
function interpolate_color(color1::String, color2::String, t::Float64)
    # Simple interpolation between predefined colors
    t = clamp(t, 0.0, 1.0)
    
    if t < 0.33
        return color1
    elseif t < 0.67
        return "yellow"  # Middle color
    else
        return color2
    end
end

"""
Apply gradient colors to sparkline based on values
"""
function apply_gradient_colors(sparkline::String, values::Vector{Float64}, 
                             min_color::String = "green", max_color::String = "red")
    if isempty(sparkline) || isempty(values)
        return sparkline
    end
    
    chars = collect(sparkline)
    min_val = minimum(values)
    max_val = maximum(values)
    range = max_val - min_val
    
    colored = ""
    for (i, char) in enumerate(chars)
        if i <= length(values)
            normalized = range > 0 ? (values[i] - min_val) / range : 0.5
            color = interpolate_color(min_color, max_color, normalized)
            colored *= apply_color(string(char), color)
        else
            colored *= string(char)
        end
    end
    
    return colored
end

# Specialized Sparkline Types

"""
Create sparkline for score metrics with trend indicator
"""
function create_score_sparkline(scores::Vector{Float64}, width::Int = 20)
    config = SparklineConfig(
        width = width,
        smooth = true,
        smooth_window = 5,
        gradient = true,
        min_color = "red",
        max_color = "green"
    )
    
    sparkline = create_colored_sparkline(scores, config)
    
    # Add trend indicator
    if length(scores) >= 2
        trend = scores[end] > scores[end-1] ? "↑" : 
                scores[end] < scores[end-1] ? "↓" : "→"
        sparkline *= " $trend"
    end
    
    return sparkline
end

"""
Create sparkline for GPU metrics with threshold highlighting
"""
function create_gpu_sparkline(values::Vector{Float64}, threshold::Float64 = 80.0, width::Int = 20)
    config = SparklineConfig(
        width = width,
        smooth = false,  # Keep raw GPU data
        gradient = false
    )
    
    sparkline = create_sparkline(values, config)
    
    # Highlight if current value exceeds threshold
    if !isempty(values) && values[end] > threshold
        sparkline = apply_color(sparkline, "bright_red")
    end
    
    return sparkline
end

"""
Create sparkline for performance metrics with adaptive coloring
"""
function create_performance_sparkline(values::Vector{Float64}, target::Float64, width::Int = 20)
    config = SparklineConfig(
        width = width,
        smooth = true,
        smooth_window = 3
    )
    
    sparkline = create_sparkline(values, config)
    
    # Color based on performance relative to target
    if !isempty(values)
        avg_recent = mean(values[max(1, end-10):end])
        if avg_recent >= target
            sparkline = apply_color(sparkline, "bright_green")
        elseif avg_recent >= target * 0.8
            sparkline = apply_color(sparkline, "yellow")
        else
            sparkline = apply_color(sparkline, "red")
        end
    end
    
    return sparkline
end

# Sparkline Renderer Methods

"""
Push data to renderer and update stats
"""
function push_data!(renderer::SparklineRenderer, value::Float64, timestamp::DateTime = now())
    push_data!(renderer.buffer, value, timestamp)
    
    # Update min/max if auto-scaling
    if renderer.auto_scale
        renderer.min_value = min(renderer.min_value, value)
        renderer.max_value = max(renderer.max_value, value)
    end
end

"""
Render sparkline with current buffer data
"""
function render(renderer::SparklineRenderer)
    data, _ = get_recent_data(renderer.buffer, renderer.config.width * 2)
    
    if isempty(data)
        return ""
    end
    
    if renderer.config.gradient
        return create_colored_sparkline(data, renderer.config)
    else
        return create_sparkline(data, renderer.config)
    end
end

"""
Get statistics for rendered data
"""
function get_stats(renderer::SparklineRenderer)
    data, timestamps = get_data(renderer.buffer)
    
    if isempty(data)
        return (min=0.0, max=0.0, mean=0.0, std=0.0, trend=:stable)
    end
    
    # Calculate trend
    trend = :stable
    if length(data) >= 10
        recent = data[end-9:end]
        older = data[max(1, end-19):end-10]
        if mean(recent) > mean(older) * 1.1
            trend = :increasing
        elseif mean(recent) < mean(older) * 0.9
            trend = :decreasing
        end
    end
    
    return (
        min = minimum(data),
        max = maximum(data),
        mean = mean(data),
        std = std(data),
        trend = trend
    )
end

end # module