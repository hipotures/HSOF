module LogPanel

using Dates
using Printf
using ..ColorTheme

export LogLevel, LogEntry, CircularLogBuffer, LogPanelState, LogPanelContent
export add_log!, filter_logs, search_logs, export_logs, format_log_entry
export DEBUG, INFO, WARN, ERROR

# Log level enumeration
@enum LogLevel begin
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4
end

# Log entry structure
struct LogEntry
    timestamp::DateTime
    level::LogLevel
    message::String
    source::String
    metadata::Dict{String, Any}
    id::Int64  # Unique identifier for selection
end

# Circular buffer for log entries
mutable struct CircularLogBuffer
    entries::Vector{LogEntry}
    capacity::Int
    head::Int
    size::Int
    next_id::Int64
    lock::ReentrantLock
end

# Panel state for UI
mutable struct LogPanelState
    buffer::CircularLogBuffer
    filter_level::LogLevel
    search_pattern::Union{Nothing, Regex}
    selected_index::Union{Nothing, Int}
    auto_scroll::Bool
    show_details::Bool
    group_repeated::Bool
    last_interaction_time::DateTime
    theme::ThemeConfig
    highlight_matches::Bool
end

# Constructor for CircularLogBuffer
function CircularLogBuffer(capacity::Int=100)
    CircularLogBuffer(
        Vector{LogEntry}(undef, capacity),
        capacity,
        1,
        0,
        1,
        ReentrantLock()
    )
end

# Add log entry to buffer
function add_log!(buffer::CircularLogBuffer, level::LogLevel, message::String, 
                  source::String="", metadata::Dict{String, Any}=Dict{String, Any}())
    lock(buffer.lock) do
        entry = LogEntry(now(), level, message, source, metadata, buffer.next_id)
        buffer.next_id += 1
        
        buffer.entries[buffer.head] = entry
        buffer.head = mod1(buffer.head + 1, buffer.capacity)
        buffer.size = min(buffer.size + 1, buffer.capacity)
    end
end

# Get all entries from buffer in chronological order
function get_entries(buffer::CircularLogBuffer)
    lock(buffer.lock) do
        if buffer.size == 0
            return LogEntry[]
        end
        
        entries = LogEntry[]
        if buffer.size < buffer.capacity
            # Buffer not full yet
            for i in 1:buffer.size
                push!(entries, buffer.entries[i])
            end
        else
            # Buffer is full, wrap around
            start_idx = buffer.head
            for i in 0:(buffer.capacity - 1)
                idx = mod1(start_idx + i, buffer.capacity)
                push!(entries, buffer.entries[idx])
            end
        end
        return entries
    end
end

# Filter logs by level
function filter_logs(entries::Vector{LogEntry}, min_level::LogLevel)
    filter(e -> e.level >= min_level, entries)
end

# Search logs with regex
function search_logs(entries::Vector{LogEntry}, pattern::Regex)
    filter(e -> occursin(pattern, e.message) || occursin(pattern, e.source), entries)
end

# Get color for log level
function get_log_level_color(theme::ThemeConfig, level::LogLevel)
    if level == DEBUG
        return get_status_color(theme, :dim)
    elseif level == INFO
        return get_status_color(theme, :text)
    elseif level == WARN
        return get_status_color(theme, :warning)
    elseif level == ERROR
        return get_status_color(theme, :critical)
    else
        return get_status_color(theme, :text)
    end
end

# Format timestamp with color
function format_timestamp(timestamp::DateTime, theme::ThemeConfig)
    time_str = Dates.format(timestamp, "HH:MM:SS.sss")
    color = get_status_color(theme, :dim)
    return apply_theme_color(time_str, color)
end

# Format log level with color
function format_log_level(level::LogLevel, theme::ThemeConfig)
    level_str = @sprintf("%-5s", string(level))
    color = get_log_level_color(theme, level)
    return apply_theme_color(level_str, color)
end

# Highlight search matches in text
function highlight_matches(text::String, pattern::Regex, theme::ThemeConfig)
    # Find all matches
    matches = collect(eachmatch(pattern, text))
    if isempty(matches)
        return text
    end
    
    # Build highlighted string
    result = ""
    last_end = 1
    accent_color = get_status_color(theme, :accent)
    
    for match in matches
        # Add text before match
        result *= text[last_end:match.offset-1]
        # Add highlighted match
        result *= apply_theme_color(match.match, accent_color, background=true)
        last_end = match.offset + length(match.match)
    end
    
    # Add remaining text
    result *= text[last_end:end]
    return result
end

# Format log entry for display
function format_log_entry(entry::LogEntry; detailed::Bool=false, width::Int=80, 
                         theme::ThemeConfig=create_theme(), search_pattern::Union{Nothing,Regex}=nothing)
    time_str = format_timestamp(entry.timestamp, theme)
    level_str = format_log_level(entry.level, theme)
    
    if detailed
        # Detailed multi-line format
        lines = String[]
        separator_color = get_status_color(theme, :dim)
        push!(lines, apply_theme_color("─" ^ width, separator_color))
        
        # Header line
        header = @sprintf("ID: %d | Time: %s | Level: %s", entry.id, 
                         Dates.format(entry.timestamp, "HH:MM:SS.sss"), string(entry.level))
        push!(lines, header)
        
        if !isempty(entry.source)
            source_line = @sprintf("Source: %s", entry.source)
            if search_pattern !== nothing
                source_line = highlight_matches(source_line, search_pattern, theme)
            end
            push!(lines, source_line)
        end
        
        # Message with potential highlighting
        msg_line = @sprintf("Message: %s", entry.message)
        if search_pattern !== nothing
            msg_line = highlight_matches(msg_line, search_pattern, theme)
        end
        push!(lines, msg_line)
        
        if !isempty(entry.metadata)
            push!(lines, "Metadata:")
            for (key, value) in entry.metadata
                meta_line = @sprintf("  %s: %s", key, string(value))
                push!(lines, meta_line)
            end
        end
        
        push!(lines, apply_theme_color("─" ^ width, separator_color))
        return join(lines, "\n")
    else
        # Compact single-line format
        source_str = ""
        if !isempty(entry.source)
            source_color = get_status_color(theme, :accent)
            source_str = apply_theme_color("[$(entry.source)]", source_color) * " "
        end
        
        msg = entry.message
        
        # Calculate available space for message
        # Account for ANSI escape sequences
        raw_time_len = length(Dates.format(entry.timestamp, "HH:MM:SS.sss"))
        raw_level_len = 5  # Fixed width
        raw_source_len = isempty(entry.source) ? 0 : length(entry.source) + 3  # [source] 
        max_msg_len = width - raw_time_len - raw_level_len - raw_source_len - 4  # spaces
        
        if length(msg) > max_msg_len && max_msg_len > 3
            msg = msg[1:max_msg_len-3] * "..."
        end
        
        # Apply search highlighting if pattern provided
        if search_pattern !== nothing
            msg = highlight_matches(msg, search_pattern, theme)
        end
        
        return @sprintf("%s %s %s%s", time_str, level_str, source_str, msg)
    end
end

# Export logs to file
function export_logs(entries::Vector{LogEntry}, filepath::String; 
                    format::Symbol=:text, include_metadata::Bool=true)
    open(filepath, "w") do io
        if format == :text
            for entry in entries
                println(io, format_log_entry(entry, detailed=include_metadata))
            end
        elseif format == :csv
            # CSV header
            println(io, "timestamp,level,source,message")
            for entry in entries
                time_str = Dates.format(entry.timestamp, "yyyy-mm-dd HH:MM:SS.sss")
                # Escape quotes in message
                msg = replace(entry.message, "\"" => "\"\"")
                println(io, "\"$time_str\",\"$(entry.level)\",\"$(entry.source)\",\"$msg\"")
            end
        elseif format == :json
            # Simple JSON export
            println(io, "[")
            for (i, entry) in enumerate(entries)
                print(io, "  {")
                print(io, "\"timestamp\": \"$(entry.timestamp)\", ")
                print(io, "\"level\": \"$(entry.level)\", ")
                print(io, "\"source\": \"$(entry.source)\", ")
                print(io, "\"message\": \"$(escape_string(entry.message))\"")
                
                if include_metadata && !isempty(entry.metadata)
                    print(io, ", \"metadata\": {")
                    meta_items = []
                    for (k, v) in entry.metadata
                        push!(meta_items, "\"$k\": \"$(escape_string(string(v)))\"")
                    end
                    print(io, join(meta_items, ", "))
                    print(io, "}")
                end
                
                print(io, "}")
                if i < length(entries)
                    println(io, ",")
                else
                    println(io)
                end
            end
            println(io, "]")
        else
            error("Unsupported export format: $format")
        end
    end
end

# Group repeated messages
function group_repeated_messages(entries::Vector{LogEntry})
    if isempty(entries)
        return Tuple{LogEntry, Int}[]
    end
    
    grouped = Tuple{LogEntry, Int}[]
    current_entry = entries[1]
    count = 1
    
    for i in 2:length(entries)
        entry = entries[i]
        # Check if message and level are the same
        if entry.message == current_entry.message && 
           entry.level == current_entry.level &&
           entry.source == current_entry.source
            count += 1
        else
            push!(grouped, (current_entry, count))
            current_entry = entry
            count = 1
        end
    end
    
    push!(grouped, (current_entry, count))
    return grouped
end

# Create panel state
function LogPanelState(buffer::CircularLogBuffer=CircularLogBuffer(); theme::ThemeConfig=create_theme())
    LogPanelState(
        buffer,
        DEBUG,  # Show all levels by default
        nothing,  # No search pattern
        nothing,  # No selection
        true,     # Auto-scroll enabled
        false,    # Details hidden
        false,    # Grouping disabled
        now(),    # Last interaction time
        theme,    # Color theme
        true      # Highlight search matches
    )
end

# Generate panel content for Rich display
function LogPanelContent(state::LogPanelState; width::Int=80, height::Int=20)
    entries = get_entries(state.buffer)
    
    # Apply filters
    if state.filter_level > DEBUG
        entries = filter_logs(entries, state.filter_level)
    end
    
    if state.search_pattern !== nothing
        entries = search_logs(entries, state.search_pattern)
    end
    
    # Group if enabled
    display_items = if state.group_repeated
        grouped = group_repeated_messages(entries)
        # Convert back to display format
        items = []
        for (entry, count) in grouped
            if count > 1
                # Modify message to show count
                modified_entry = LogEntry(
                    entry.timestamp,
                    entry.level,
                    "$(entry.message) [×$count]",
                    entry.source,
                    entry.metadata,
                    entry.id
                )
                push!(items, modified_entry)
            else
                push!(items, entry)
            end
        end
        items
    else
        entries
    end
    
    # Generate content lines
    lines = String[]
    
    # Header with filter info
    header_parts = ["Logs"]
    if state.filter_level > DEBUG
        push!(header_parts, "Filter: ≥$(state.filter_level)")
    end
    if state.search_pattern !== nothing
        push!(header_parts, "Search: $(state.search_pattern.pattern)")
    end
    push!(lines, join(header_parts, " | "))
    push!(lines, "─" * width)
    
    # Calculate visible area
    content_height = height - 3  # Header + separator + footer
    
    # Handle scrolling
    start_idx = if state.auto_scroll || length(display_items) <= content_height
        max(1, length(display_items) - content_height + 1)
    else
        # Manual scroll position would be handled here
        1
    end
    
    end_idx = min(start_idx + content_height - 1, length(display_items))
    
    # Add log entries
    for i in start_idx:end_idx
        entry = display_items[i]
        is_selected = state.selected_index == i
        
        # Format with theme and search highlighting
        line = format_log_entry(entry, detailed=false, width=width-2, 
                               theme=state.theme, 
                               search_pattern=state.highlight_matches ? state.search_pattern : nothing)
        
        # Add selection indicator
        if is_selected
            selector_color = get_status_color(state.theme, :accent)
            line = apply_theme_color("▶ ", selector_color) * line
        else
            line = "  " * line
        end
        
        push!(lines, line)
    end
    
    # Fill remaining space
    while length(lines) < height - 1
        push!(lines, "")
    end
    
    # Footer with status
    footer_parts = []
    push!(footer_parts, "$(length(display_items))/$(state.buffer.size) entries")
    
    if state.auto_scroll
        push!(footer_parts, "Auto-scroll ON")
    else
        push!(footer_parts, "Auto-scroll OFF")
    end
    
    if state.group_repeated
        push!(footer_parts, "Grouping ON")
    end
    
    push!(lines, join(footer_parts, " | "))
    
    return join(lines, "\n")
end

end # module