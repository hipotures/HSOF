module TerminalCompat

using Dates
using Printf

export TerminalCapabilities, RenderMode, TerminalRenderer
export detect_terminal_capabilities, create_renderer
export batch_ansi_sequences, optimize_cursor_movement
export FrameRateLimiter, should_render_frame, update_frame_time!
export PartialRedrawRegion, mark_dirty!, is_dirty, clear_dirty!
export render_with_fallback, get_terminal_type
export test_terminal_colors, test_unicode_support

"""
Color support levels
"""
@enum ColorMode begin
    COLOR_NONE = 0
    COLOR_8BIT = 1
    COLOR_256 = 2
    COLOR_TRUE = 3
end

"""
Rendering modes for different terminal capabilities
"""
@enum RenderMode begin
    RENDER_BASIC = 1      # ASCII only, no colors
    RENDER_STANDARD = 2   # Basic colors and simple Unicode
    RENDER_ENHANCED = 3   # Full colors and extended Unicode
    RENDER_FULL = 4       # True color and all Unicode
end

"""
Terminal capabilities detection result
"""
struct TerminalCapabilities
    terminal_type::String
    color_mode::ColorMode
    unicode_support::Bool
    unicode_level::Int  # 0=none, 1=basic, 2=extended, 3=full
    width::Int
    height::Int
    supports_cursor_save::Bool
    supports_alternate_screen::Bool
    supports_mouse::Bool
    render_mode::RenderMode
    
    function TerminalCapabilities(;
        terminal_type::String = "unknown",
        color_mode::ColorMode = COLOR_NONE,
        unicode_support::Bool = false,
        unicode_level::Int = 0,
        width::Int = 80,
        height::Int = 24,
        supports_cursor_save::Bool = false,
        supports_alternate_screen::Bool = false,
        supports_mouse::Bool = false,
        render_mode::RenderMode = RENDER_BASIC
    )
        new(terminal_type, color_mode, unicode_support, unicode_level,
            width, height, supports_cursor_save, supports_alternate_screen,
            supports_mouse, render_mode)
    end
end

"""
ANSI sequence batch buffer
"""
mutable struct ANSIBatch
    sequences::Vector{String}
    buffer_size::Int
    auto_flush::Bool
    
    function ANSIBatch(buffer_size::Int = 100; auto_flush::Bool = true)
        new(String[], buffer_size, auto_flush)
    end
end

"""
Frame rate limiter for terminal rendering
"""
mutable struct FrameRateLimiter
    target_fps::Float64
    frame_duration::Float64  # milliseconds
    last_frame_time::DateTime
    frame_times::Vector{Float64}  # Rolling window of frame times
    window_size::Int
    adaptive::Bool
    min_fps::Float64
    max_fps::Float64
    
    function FrameRateLimiter(;
        target_fps::Float64 = 10.0,
        adaptive::Bool = true,
        min_fps::Float64 = 5.0,
        max_fps::Float64 = 30.0,
        window_size::Int = 30
    )
        frame_duration = 1000.0 / target_fps
        new(target_fps, frame_duration, now(), Float64[], 
            window_size, adaptive, min_fps, max_fps)
    end
end

"""
Partial redraw optimization tracking
"""
mutable struct PartialRedrawRegion
    dirty_regions::Dict{Tuple{Int,Int,Int,Int}, Bool}  # (x1,y1,x2,y2) => is_dirty
    full_redraw::Bool
    last_content::Dict{Tuple{Int,Int,Int,Int}, String}  # Cache of last rendered content
    
    function PartialRedrawRegion()
        new(Dict{Tuple{Int,Int,Int,Int}, Bool}(), false, Dict{Tuple{Int,Int,Int,Int}, String}())
    end
end

"""
Terminal-specific renderer
"""
abstract type TerminalRenderer end

# Concrete renderer types
struct BasicRenderer <: TerminalRenderer end
struct StandardRenderer <: TerminalRenderer end
struct EnhancedRenderer <: TerminalRenderer end
struct FullRenderer <: TerminalRenderer end

# Terminal-specific renderers
struct ITerm2Renderer <: TerminalRenderer end
struct WindowsTerminalRenderer <: TerminalRenderer end
struct GnomeTerminalRenderer <: TerminalRenderer end

# Core Functions

"""
Detect terminal capabilities
"""
function detect_terminal_capabilities()
    # Get terminal type from environment
    term = get(ENV, "TERM", "")
    term_program = get(ENV, "TERM_PROGRAM", "")
    colorterm = get(ENV, "COLORTERM", "")
    
    # Detect terminal type
    terminal_type = if !isempty(term_program)
        term_program
    elseif occursin("gnome", lowercase(term))
        "gnome-terminal"
    elseif occursin("xterm", lowercase(term))
        "xterm"
    else
        term
    end
    
    # Detect color support
    color_mode = detect_color_mode(term, colorterm)
    
    # Detect Unicode support
    unicode_level = detect_unicode_level()
    unicode_support = unicode_level > 0
    
    # Get terminal dimensions
    width, height = get_terminal_size()
    
    # Detect advanced features
    supports_cursor_save = !isempty(term) && term != "dumb"
    supports_alternate_screen = occursin("xterm", term) || occursin("screen", term)
    supports_mouse = occursin("xterm", term) || term_program == "iTerm.app"
    
    # Determine render mode
    render_mode = determine_render_mode(color_mode, unicode_level)
    
    return TerminalCapabilities(
        terminal_type = terminal_type,
        color_mode = color_mode,
        unicode_support = unicode_support,
        unicode_level = unicode_level,
        width = width,
        height = height,
        supports_cursor_save = supports_cursor_save,
        supports_alternate_screen = supports_alternate_screen,
        supports_mouse = supports_mouse,
        render_mode = render_mode
    )
end

"""
Detect color mode support
"""
function detect_color_mode(term::String, colorterm::String)
    # True color detection
    if colorterm == "truecolor" || colorterm == "24bit"
        return COLOR_TRUE
    end
    
    # 256 color detection
    if occursin("256color", term)
        return COLOR_256
    end
    
    # Basic 8-bit color
    if !isempty(term) && term != "dumb"
        return COLOR_8BIT
    end
    
    return COLOR_NONE
end

"""
Detect Unicode support level
"""
function detect_unicode_level()
    # Check locale
    lang = get(ENV, "LANG", "")
    lc_all = get(ENV, "LC_ALL", "")
    
    # Check for UTF-8 support
    if occursin("UTF-8", lang) || occursin("UTF-8", lc_all) || 
       occursin("utf8", lowercase(lang)) || occursin("utf8", lowercase(lc_all))
        # Test actual Unicode rendering
        return test_unicode_rendering()
    end
    
    return 0
end

"""
Test actual Unicode rendering capability
"""
function test_unicode_rendering()
    # Try to determine support level by testing different Unicode ranges
    try
        # Basic Unicode (common symbols)
        basic = "●▲■"
        if length(basic) == 3
            # Extended Unicode (box drawing)
            extended = "┌─┐│└┘"
            if length(extended) == 6
                # Full Unicode (emoji and special)
                full = "🚀📊💻"
                if length(full) == 3
                    return 3  # Full support
                end
                return 2  # Extended support
            end
            return 1  # Basic support
        end
    catch
        # Unicode handling failed
    end
    
    return 0  # No support
end

"""
Get terminal dimensions
"""
function get_terminal_size()
    # Try to get from Julia's display size
    try
        rows, cols = displaysize()
        return cols, rows
    catch
        # Fallback to environment or defaults
        cols = tryparse(Int, get(ENV, "COLUMNS", "80"))
        rows = tryparse(Int, get(ENV, "LINES", "24"))
        return something(cols, 80), something(rows, 24)
    end
end

"""
Determine render mode based on capabilities
"""
function determine_render_mode(color_mode::ColorMode, unicode_level::Int)
    if color_mode == COLOR_TRUE && unicode_level >= 3
        return RENDER_FULL
    elseif color_mode >= COLOR_256 && unicode_level >= 2
        return RENDER_ENHANCED
    elseif color_mode >= COLOR_8BIT && unicode_level >= 1
        return RENDER_STANDARD
    else
        return RENDER_BASIC
    end
end

"""
Create appropriate renderer for capabilities
"""
function create_renderer(capabilities::TerminalCapabilities)
    # Check for specific terminals first
    if capabilities.terminal_type == "iTerm.app"
        return ITerm2Renderer()
    elseif occursin("Windows Terminal", capabilities.terminal_type)
        return WindowsTerminalRenderer()
    elseif occursin("gnome", lowercase(capabilities.terminal_type))
        return GnomeTerminalRenderer()
    end
    
    # Fall back to capability-based renderer
    if capabilities.render_mode == RENDER_FULL
        return FullRenderer()
    elseif capabilities.render_mode == RENDER_ENHANCED
        return EnhancedRenderer()
    elseif capabilities.render_mode == RENDER_STANDARD
        return StandardRenderer()
    else
        return BasicRenderer()
    end
end

# ANSI Sequence Batching

"""
Add ANSI sequence to batch
"""
function add_sequence!(batch::ANSIBatch, sequence::String)
    push!(batch.sequences, sequence)
    
    if batch.auto_flush && length(batch.sequences) >= batch.buffer_size
        return flush_batch!(batch)
    end
    
    return ""
end

"""
Flush batched ANSI sequences
"""
function flush_batch!(batch::ANSIBatch)
    if isempty(batch.sequences)
        return ""
    end
    
    result = join(batch.sequences)
    empty!(batch.sequences)
    return result
end

"""
Batch multiple ANSI sequences efficiently
"""
function batch_ansi_sequences(sequences::Vector{String})
    # Remove redundant sequences
    optimized = String[]
    last_seq = ""
    
    for seq in sequences
        if seq != last_seq  # Skip duplicates
            push!(optimized, seq)
            last_seq = seq
        end
    end
    
    return join(optimized)
end

# Cursor Movement Optimization

"""
Optimize cursor movement to minimize sequences
"""
function optimize_cursor_movement(current_x::Int, current_y::Int, 
                                target_x::Int, target_y::Int)
    sequences = String[]
    
    # Calculate deltas
    dx = target_x - current_x
    dy = target_y - current_y
    
    if dx == 0 && dy == 0
        return ""
    end
    
    # Optimize vertical movement
    if dy != 0
        if dy > 0
            push!(sequences, "\033[$(dy)B")  # Move down
        else
            push!(sequences, "\033[$(-dy)A")  # Move up
        end
    end
    
    # Optimize horizontal movement
    if dx != 0
        if dx > 0
            push!(sequences, "\033[$(dx)C")  # Move right
        else
            push!(sequences, "\033[$(-dx)D")  # Move left
        end
    end
    
    # Alternative: use absolute positioning if more efficient
    if length(sequences) > 1
        absolute = "\033[$(target_y);$(target_x)H"
        if length(absolute) < sum(length, sequences)
            return absolute
        end
    end
    
    return join(sequences)
end

# Frame Rate Limiting

"""
Check if frame should be rendered
"""
function should_render_frame(limiter::FrameRateLimiter)
    current_time = now()
    elapsed = Dates.value(current_time - limiter.last_frame_time)
    
    if elapsed < limiter.frame_duration
        return false
    end
    
    # Adaptive frame rate
    if limiter.adaptive && !isempty(limiter.frame_times)
        adapt_frame_rate!(limiter)
    end
    
    return true
end

"""
Update frame timing
"""
function update_frame_time!(limiter::FrameRateLimiter)
    current_time = now()
    
    if limiter.last_frame_time != current_time
        frame_time = Dates.value(current_time - limiter.last_frame_time)
        push!(limiter.frame_times, frame_time)
        
        # Maintain window size
        if length(limiter.frame_times) > limiter.window_size
            popfirst!(limiter.frame_times)
        end
    end
    
    limiter.last_frame_time = current_time
end

"""
Adapt frame rate based on performance
"""
function adapt_frame_rate!(limiter::FrameRateLimiter)
    avg_frame_time = sum(limiter.frame_times) / length(limiter.frame_times)
    current_fps = 1000.0 / avg_frame_time
    
    # Adjust target FPS if consistently missing frames
    if current_fps < limiter.target_fps * 0.9
        # Reduce target FPS
        new_fps = max(limiter.min_fps, limiter.target_fps - 1.0)
        limiter.target_fps = new_fps
        limiter.frame_duration = 1000.0 / new_fps
    elseif current_fps > limiter.target_fps * 1.1
        # Increase target FPS if performance allows
        new_fps = min(limiter.max_fps, limiter.target_fps + 1.0)
        limiter.target_fps = new_fps
        limiter.frame_duration = 1000.0 / new_fps
    end
end

# Partial Redraw Optimization

"""
Mark region as dirty (needs redraw)
"""
function mark_dirty!(region::PartialRedrawRegion, x1::Int, y1::Int, x2::Int, y2::Int)
    region.dirty_regions[(x1, y1, x2, y2)] = true
end

"""
Check if region is dirty
"""
function is_dirty(region::PartialRedrawRegion, x1::Int, y1::Int, x2::Int, y2::Int)
    return get(region.dirty_regions, (x1, y1, x2, y2), false)
end

"""
Clear dirty flags
"""
function clear_dirty!(region::PartialRedrawRegion)
    empty!(region.dirty_regions)
    region.full_redraw = false
end

"""
Mark for full redraw
"""
function mark_full_redraw!(region::PartialRedrawRegion)
    region.full_redraw = true
end

"""
Should perform full redraw
"""
function needs_full_redraw(region::PartialRedrawRegion)
    return region.full_redraw
end

"""
Cache content for region
"""
function cache_content!(region::PartialRedrawRegion, x1::Int, y1::Int, 
                       x2::Int, y2::Int, content::String)
    region.last_content[(x1, y1, x2, y2)] = content
end

"""
Get cached content
"""
function get_cached_content(region::PartialRedrawRegion, x1::Int, y1::Int, 
                          x2::Int, y2::Int)
    return get(region.last_content, (x1, y1, x2, y2), "")
end

# Rendering with Fallbacks

"""
Render with appropriate fallback for terminal
"""
function render_with_fallback(renderer::TerminalRenderer, content::String, 
                            capabilities::TerminalCapabilities)
    return render_content(renderer, content, capabilities)
end

# Renderer implementations

function render_content(::BasicRenderer, content::String, ::TerminalCapabilities)
    # ASCII only, no colors
    # Replace Unicode with ASCII equivalents
    ascii_content = replace(content,
        "█" => "#", "▓" => "=", "▒" => "-", "░" => ".",
        "●" => "*", "▲" => "^", "■" => "#",
        "┌" => "+", "─" => "-", "┐" => "+",
        "│" => "|", "└" => "+", "┘" => "+",
        "├" => "+", "┤" => "+", "┬" => "+", "┴" => "+"
    )
    # Strip ANSI color codes
    return replace(ascii_content, r"\033\[[0-9;]*m" => "")
end

function render_content(::StandardRenderer, content::String, ::TerminalCapabilities)
    # Basic colors and simple Unicode
    # Keep basic box drawing and simple symbols
    return content
end

function render_content(::EnhancedRenderer, content::String, ::TerminalCapabilities)
    # Full colors and extended Unicode
    return content
end

function render_content(::FullRenderer, content::String, ::TerminalCapabilities)
    # True color and all Unicode
    return content
end

# Terminal-specific renderers

function render_content(::ITerm2Renderer, content::String, ::TerminalCapabilities)
    # iTerm2 specific optimizations
    # Can use proprietary escape sequences
    return content
end

function render_content(::WindowsTerminalRenderer, content::String, ::TerminalCapabilities)
    # Windows Terminal specific handling
    return content
end

function render_content(::GnomeTerminalRenderer, content::String, ::TerminalCapabilities)
    # GNOME Terminal specific handling
    return content
end

# Utility Functions

"""
Get detected terminal type
"""
function get_terminal_type()
    capabilities = detect_terminal_capabilities()
    return capabilities.terminal_type
end

"""
Test terminal color support
"""
function test_terminal_colors()
    println("Testing terminal color support...")
    
    # 8-bit colors
    print("8-bit colors: ")
    for i in 0:7
        print("\033[3$(i)m●\033[0m ")
    end
    println()
    
    # 256 colors
    print("256 colors: ")
    for i in 0:15
        print("\033[38;5;$(i)m●\033[0m")
    end
    println()
    
    # True color
    print("True color: ")
    for i in 0:10
        r = round(Int, 255 * i / 10)
        print("\033[38;2;$(r);0;0m●\033[0m")
    end
    println()
end

"""
Test Unicode support
"""
function test_unicode_support()
    println("Testing Unicode support...")
    
    println("Basic: ● ▲ ■ ○ ◐ ◑ ◒ ◓")
    println("Box drawing: ┌─┬─┐ │ │ │ ├─┼─┤ │ │ │ └─┴─┘")
    println("Blocks: █ ▓ ▒ ░")
    println("Emoji: 🚀 📊 💻 ⚡ 🎯")
end

# ANSI Escape Sequences Constants

const ESC = "\033"
const CSI = "$(ESC)["

# Cursor control
const CURSOR_HOME = "$(CSI)H"
const CURSOR_SAVE = "$(CSI)s"
const CURSOR_RESTORE = "$(CSI)u"
const CURSOR_HIDE = "$(CSI)?25l"
const CURSOR_SHOW = "$(CSI)?25h"

# Screen control
const CLEAR_SCREEN = "$(CSI)2J"
const CLEAR_LINE = "$(CSI)2K"
const CLEAR_TO_EOL = "$(CSI)K"

# Alternate screen
const ENTER_ALT_SCREEN = "$(CSI)?1049h"
const EXIT_ALT_SCREEN = "$(CSI)?1049l"

# Colors
const RESET = "$(CSI)0m"
const BOLD = "$(CSI)1m"
const DIM = "$(CSI)2m"
const ITALIC = "$(CSI)3m"
const UNDERLINE = "$(CSI)4m"

end # module