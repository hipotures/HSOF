module LogPanel

using Dates
using Match
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

# Format log entry for display
function format_log_entry(entry::LogEntry; detailed::Bool=false, width::Int=80)
    time_str = Dates.format(entry.timestamp, "HH:MM:SS.sss")
    level_str = string(entry.level)
    
    if detailed
        # Detailed multi-line format
        lines = String[]
        push!(lines, "─" ^ width)
        push!(lines, @sprintf("ID: %d | Time: %s | Level: %s", 
                             entry.id, time_str, level_str))
        
        if !isempty(entry.source)
            push!(lines, @sprintf("Source: %s", entry.source))
        end
        
        push!(lines, @sprintf("Message: %s", entry.message))
        
        if !isempty(entry.metadata)
            push!(lines, "Metadata:")
            for (key, value) in entry.metadata
                push!(lines, @sprintf("  %s: %s", key, string(value)))
            end
        end
        
        push!(lines, "─" ^ width)
        return join(lines, "\n")
    else
        # Compact single-line format
        source_str = isempty(entry.source) ? "" : "[$(entry.source)] "
        msg = entry.message
        
        # Truncate message if too long
        max_msg_len = width - length(time_str) - length(level_str) - length(source_str) - 10
        if length(msg) > max_msg_len
            msg = msg[1:max_msg_len-3] * "..."
        end
        
        return @sprintf("%s %-5s %s%s", time_str, level_str, source_str, msg)
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
        
        line = format_log_entry(entry, detailed=false, width=width-2)
        
        # Add selection indicator
        if is_selected
            line = "▶ " * line
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