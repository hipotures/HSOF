#!/usr/bin/env julia

# Log Panel Usage Examples
# This file demonstrates how to use the LogPanel module for system logging

using Dates
using Random

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Include the modules
include("../src/ui/color_theme.jl")
include("../src/ui/log_panel.jl")

using .ColorTheme
using .LogPanel

println("Log Panel Usage Examples\n")

# Example 1: Basic Log Panel Setup
println("=" ^ 60)
println("Example 1: Basic Log Panel Setup")
println("=" ^ 60)

# Create a log buffer and panel state
buffer = CircularLogBuffer(50)  # Store up to 50 log entries
theme = create_theme()
state = LogPanelState(buffer, theme=theme)

# Add some sample log entries
add_log!(buffer, INFO, "System initialized successfully", "Main")
add_log!(buffer, DEBUG, "Loading configuration from config.toml", "Config")
add_log!(buffer, INFO, "Connected to database", "Database")
add_log!(buffer, WARN, "API rate limit approaching (80/100 requests)", "API")
add_log!(buffer, ERROR, "Failed to connect to external service", "Network", 
         Dict("error_code" => 500, "retry_count" => 3))

# Display the log panel
println("\nLog Panel Content:")
println(LogPanelContent(state, width=80, height=15))

# Example 2: Log Filtering and Search
println("\n\n" * "=" ^ 60)
println("Example 2: Log Filtering and Search")
println("=" ^ 60)

# Add more diverse log entries
for i in 1:20
    level = rand([DEBUG, INFO, WARN, ERROR])
    sources = ["API", "Database", "Cache", "Network", "Auth", "Worker"]
    messages = [
        "Processing request #$i",
        "Query executed in $(rand(10:100))ms",
        "Cache hit for key: user_$i",
        "Connection timeout after $(rand(1:5))s",
        "User authentication successful",
        "Background job completed"
    ]
    
    add_log!(buffer, level, rand(messages), rand(sources))
    sleep(0.001)  # Small delay to ensure different timestamps
end

# Filter by log level
println("\nFiltering: Show only WARN and ERROR messages")
state.filter_level = WARN
println(LogPanelContent(state, width=80, height=10))

# Search functionality
println("\nSearching for 'timeout' messages:")
state.filter_level = DEBUG  # Reset filter
state.search_pattern = r"timeout"i
println(LogPanelContent(state, width=80, height=10))

# Example 3: Real-time Monitoring Simulation
println("\n\n" * "=" ^ 60)
println("Example 3: Real-time Monitoring Simulation")
println("=" ^ 60)

# Reset state
state.filter_level = DEBUG
state.search_pattern = nothing
state.auto_scroll = true

# Simulate a system under load
println("\nSimulating system activity...")
for i in 1:10
    # Normal operation
    if rand() < 0.7
        add_log!(buffer, INFO, "Request processed successfully ($(rand(10:50))ms)", "API")
    elseif rand() < 0.9
        add_log!(buffer, WARN, "High memory usage detected: $(rand(70:90))%", "Monitor")
    else
        add_log!(buffer, ERROR, "Request failed with error: $(rand(400:599))", "API")
    end
    
    # Show current state
    print("\033[2J\033[H")  # Clear screen
    println("Real-time Log Monitoring (Entry $i/10)")
    println(LogPanelContent(state, width=80, height=15))
    sleep(0.5)
end

# Example 4: Log Export
println("\n\n" * "=" ^ 60)
println("Example 4: Log Export")
println("=" ^ 60)

# Get all entries for export
entries = get_entries(buffer)
println("\nTotal entries in buffer: $(length(entries))")

# Export to different formats
mktempdir() do dir
    # Text export
    text_file = joinpath(dir, "system_logs.txt")
    export_logs(entries, text_file, format=:text)
    println("\nText export (first 5 lines):")
    for (i, line) in enumerate(eachline(text_file))
        println("  $line")
        i >= 5 && break
    end
    
    # CSV export
    csv_file = joinpath(dir, "system_logs.csv")
    export_logs(entries, csv_file, format=:csv)
    println("\nCSV export (first 3 lines):")
    for (i, line) in enumerate(eachline(csv_file))
        println("  $line")
        i >= 3 && break
    end
    
    # JSON export
    json_file = joinpath(dir, "system_logs.json")
    export_logs(entries, json_file, format=:json)
    println("\nJSON export (first 200 chars):")
    content = read(json_file, String)
    println("  " * content[1:min(200, length(content))] * "...")
end

# Example 5: Advanced Features
println("\n\n" * "=" ^ 60)
println("Example 5: Advanced Features")
println("=" ^ 60)

# Message grouping
println("\nTesting message grouping:")
state.group_repeated = true
state.filter_level = DEBUG
state.search_pattern = nothing

# Add repeated messages
for i in 1:3
    add_log!(buffer, INFO, "Database connection established", "DB")
end
add_log!(buffer, WARN, "Different message", "API")
for i in 1:5
    add_log!(buffer, ERROR, "Connection refused", "Network")
end

println(LogPanelContent(state, width=80, height=12))

# Selection and detailed view
println("\nSelecting entry for detailed view:")
state.selected_index = 45  # Select an entry
state.show_details = true

# Get the selected entry
entries = get_entries(buffer)
if state.selected_index <= length(entries)
    selected_entry = entries[state.selected_index]
    println("\nDetailed view of selected entry:")
    println(format_log_entry(selected_entry, detailed=true, width=80, theme=theme))
end

# Example 6: Theme Customization
println("\n\n" * "=" ^ 60)
println("Example 6: Theme Customization")
println("=" ^ 60)

# Switch to high contrast theme
set_theme!(theme, :high_contrast)
state.theme = theme

println("\nHigh Contrast Theme:")
# Add some new entries to show the theme
add_log!(buffer, INFO, "Using high contrast theme", "UI")
add_log!(buffer, WARN, "This is a warning in high contrast", "UI")
add_log!(buffer, ERROR, "This is an error in high contrast", "UI")

println(LogPanelContent(state, width=80, height=10))

# Colorblind mode
set_colorblind_mode!(theme, true)
println("\nColorblind-friendly mode enabled:")
println(LogPanelContent(state, width=80, height=10))

# Example 7: Integration with Dashboard
println("\n\n" * "=" ^ 60)
println("Example 7: Dashboard Integration Pattern")
println("=" ^ 60)

println("""
To integrate the log panel into a dashboard:

```julia
# In your dashboard update loop:
function update_dashboard(state::DashboardState)
    # Update other panels...
    
    # Update log panel
    log_content = LogPanelContent(
        state.log_state,
        width=panel_width,
        height=panel_height
    )
    
    # Render in the appropriate position
    render_panel(log_content, row=2, col=2)
end

# Handle keyboard shortcuts
function handle_log_shortcuts(key::Char, log_state::LogPanelState)
    if key == 'f'  # Filter
        # Cycle through filter levels
        current = Int(log_state.filter_level)
        log_state.filter_level = LogLevel((current % 4) + 1)
    elseif key == '/'  # Search
        # Open search prompt
        pattern = get_user_input("Search pattern: ")
        log_state.search_pattern = Regex(pattern, "i")
    elseif key == 'c'  # Clear search
        log_state.search_pattern = nothing
    elseif key == 'g'  # Toggle grouping
        log_state.group_repeated = !log_state.group_repeated
    elseif key == 'a'  # Toggle auto-scroll
        log_state.auto_scroll = !log_state.auto_scroll
    elseif key == 'e'  # Export
        export_logs(get_entries(log_state.buffer), "dashboard_logs.txt")
    end
end
```
""")

println("\nLog Panel Features Summary:")
println("- Circular buffer with configurable capacity")
println("- Thread-safe operations for concurrent logging")
println("- Log level filtering (DEBUG, INFO, WARN, ERROR)")
println("- Regex-based search with highlighting")
println("- Auto-scroll with pause on interaction")
println("- Message grouping for repeated entries")
println("- Multiple export formats (text, CSV, JSON)")
println("- Color-coded output with theme support")
println("- Selection and detailed view")
println("- Dashboard integration ready")

println("\n✓ Log Panel examples completed!")