using Test
using Dates
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
include("../../src/ui/color_theme.jl")
include("../../src/ui/log_panel.jl")
using .ColorTheme
using .LogPanel

@testset "LogPanel Tests" begin
    
    @testset "LogEntry Creation" begin
        entry = LogEntry(now(), INFO, "Test message", "TestSource", 
                        Dict("key" => "value"), 1)
        
        @test entry.level == INFO
        @test entry.message == "Test message"
        @test entry.source == "TestSource"
        @test entry.metadata["key"] == "value"
        @test entry.id == 1
    end
    
    @testset "CircularLogBuffer" begin
        buffer = CircularLogBuffer(5)
        
        # Test empty buffer
        @test isempty(get_entries(buffer))
        @test buffer.size == 0
        
        # Add entries
        add_log!(buffer, INFO, "Message 1", "Source1")
        add_log!(buffer, DEBUG, "Message 2", "Source2")
        add_log!(buffer, WARN, "Message 3", "Source3")
        
        entries = get_entries(buffer)
        @test length(entries) == 3
        @test entries[1].message == "Message 1"
        @test entries[2].message == "Message 2"
        @test entries[3].message == "Message 3"
        
        # Test circular behavior
        add_log!(buffer, ERROR, "Message 4")
        add_log!(buffer, INFO, "Message 5")
        add_log!(buffer, DEBUG, "Message 6")  # Should overwrite Message 1
        
        entries = get_entries(buffer)
        @test length(entries) == 5
        @test entries[1].message == "Message 2"
        @test entries[5].message == "Message 6"
        
        # Test thread safety
        @sync begin
            for i in 1:10
                @async add_log!(buffer, INFO, "Concurrent $i")
            end
        end
        
        # Buffer should still have exactly 5 entries
        @test length(get_entries(buffer)) == 5
    end
    
    @testset "Log Filtering" begin
        entries = [
            LogEntry(now(), DEBUG, "Debug message", "", Dict(), 1),
            LogEntry(now(), INFO, "Info message", "", Dict(), 2),
            LogEntry(now(), WARN, "Warning message", "", Dict(), 3),
            LogEntry(now(), ERROR, "Error message", "", Dict(), 4),
        ]
        
        # Filter by level
        filtered = filter_logs(entries, INFO)
        @test length(filtered) == 3  # INFO, WARN, ERROR
        @test all(e -> e.level >= INFO, filtered)
        
        filtered = filter_logs(entries, WARN)
        @test length(filtered) == 2  # WARN, ERROR
        @test all(e -> e.level >= WARN, filtered)
        
        filtered = filter_logs(entries, ERROR)
        @test length(filtered) == 1  # ERROR only
        @test filtered[1].level == ERROR
    end
    
    @testset "Log Search" begin
        entries = [
            LogEntry(now(), INFO, "Connection established", "NetworkModule", Dict(), 1),
            LogEntry(now(), WARN, "Connection timeout", "NetworkModule", Dict(), 2),
            LogEntry(now(), ERROR, "Database error", "DatabaseModule", Dict(), 3),
            LogEntry(now(), INFO, "Query executed", "DatabaseModule", Dict(), 4),
        ]
        
        # Search by message
        results = search_logs(entries, r"Connection")
        @test length(results) == 2
        @test results[1].message == "Connection established"
        @test results[2].message == "Connection timeout"
        
        # Search by source
        results = search_logs(entries, r"Database")
        @test length(results) == 2
        @test results[1].source == "DatabaseModule"
        @test results[2].source == "DatabaseModule"
        
        # Case insensitive search
        results = search_logs(entries, r"connection"i)
        @test length(results) == 2
    end
    
    @testset "Log Entry Formatting" begin
        theme = create_theme()
        entry = LogEntry(DateTime(2024, 1, 1, 12, 30, 45, 123), 
                        INFO, "Test message", "TestSource", 
                        Dict("key1" => "value1", "key2" => 42), 1)
        
        # Compact format
        formatted = format_log_entry(entry, detailed=false, width=80, theme=theme)
        @test occursin("12:30:45.123", formatted)
        @test occursin("INFO", formatted)
        @test occursin("[TestSource]", formatted)
        @test occursin("Test message", formatted)
        
        # Detailed format
        formatted = format_log_entry(entry, detailed=true, width=80, theme=theme)
        @test occursin("ID: 1", formatted)
        @test occursin("Time:", formatted)
        @test occursin("Level:", formatted)
        @test occursin("Source: TestSource", formatted)
        @test occursin("Message: Test message", formatted)
        @test occursin("Metadata:", formatted)
        @test occursin("key1: value1", formatted)
        @test occursin("key2: 42", formatted)
        
        # Test message truncation
        long_message = "A" ^ 100
        entry_long = LogEntry(now(), INFO, long_message, "", Dict(), 2)
        formatted = format_log_entry(entry_long, detailed=false, width=50, theme=theme)
        @test occursin("...", formatted)
        @test length(formatted) < 150  # Account for ANSI codes
    end
    
    @testset "Log Export" begin
        entries = [
            LogEntry(DateTime(2024, 1, 1, 12, 0, 0), INFO, "Info message", "Source1", Dict(), 1),
            LogEntry(DateTime(2024, 1, 1, 12, 0, 1), WARN, "Warning message", "Source2", Dict("key" => "value"), 2),
            LogEntry(DateTime(2024, 1, 1, 12, 0, 2), ERROR, "Error \"quoted\" message", "Source3", Dict(), 3),
        ]
        
        # Test text export
        mktempdir() do dir
            filepath = joinpath(dir, "test_logs.txt")
            export_logs(entries, filepath, format=:text, include_metadata=true)
            
            content = read(filepath, String)
            @test occursin("Info message", content)
            @test occursin("Warning message", content)
            @test occursin("Error \"quoted\" message", content)
            @test occursin("key: value", content)
        end
        
        # Test CSV export
        mktempdir() do dir
            filepath = joinpath(dir, "test_logs.csv")
            export_logs(entries, filepath, format=:csv)
            
            lines = readlines(filepath)
            @test lines[1] == "timestamp,level,source,message"
            @test occursin("\"2024-01-01 12:00:00.000\",\"INFO\",\"Source1\",\"Info message\"", lines[2])
            @test occursin("\"Error \"\"quoted\"\" message\"", lines[4])  # Escaped quotes
        end
        
        # Test JSON export
        mktempdir() do dir
            filepath = joinpath(dir, "test_logs.json")
            export_logs(entries, filepath, format=:json, include_metadata=true)
            
            content = read(filepath, String)
            @test occursin("\"timestamp\":", content)
            @test occursin("\"level\":", content)
            @test occursin("\"source\":", content)
            @test occursin("\"message\":", content)
            @test occursin("\"metadata\":", content)
            @test occursin("\\\"quoted\\\"", content)  # Escaped quotes in JSON
        end
    end
    
    @testset "Log Grouping" begin
        entries = [
            LogEntry(now(), INFO, "Repeated message", "Source", Dict(), 1),
            LogEntry(now(), INFO, "Repeated message", "Source", Dict(), 2),
            LogEntry(now(), INFO, "Repeated message", "Source", Dict(), 3),
            LogEntry(now(), WARN, "Different message", "Source", Dict(), 4),
            LogEntry(now(), INFO, "Repeated message", "Source", Dict(), 5),
            LogEntry(now(), INFO, "Another message", "Source2", Dict(), 6),
        ]
        
        grouped = group_repeated_messages(entries)
        @test length(grouped) == 4
        @test grouped[1][2] == 3  # First group has 3 repetitions
        @test grouped[2][2] == 1  # Different message appears once
        @test grouped[3][2] == 1  # Repeated message after different one
        @test grouped[4][2] == 1  # Another message
    end
    
    @testset "LogPanelState" begin
        buffer = CircularLogBuffer(10)
        theme = create_theme()
        state = LogPanelState(buffer, theme=theme)
        
        @test state.filter_level == DEBUG
        @test state.search_pattern === nothing
        @test state.selected_index === nothing
        @test state.auto_scroll == true
        @test state.show_details == false
        @test state.group_repeated == false
        @test state.theme === theme
        @test state.highlight_matches == true
    end
    
    @testset "LogPanelContent Generation" begin
        buffer = CircularLogBuffer(10)
        state = LogPanelState(buffer)
        
        # Add some test entries
        add_log!(buffer, INFO, "First message", "Module1")
        add_log!(buffer, WARN, "Warning message", "Module2")
        add_log!(buffer, ERROR, "Error message", "Module3")
        add_log!(buffer, DEBUG, "Debug message", "Module4")
        
        # Generate content
        content = LogPanelContent(state, width=60, height=10)
        lines = split(content, "\n")
        
        @test length(lines) == 10
        @test occursin("Logs", lines[1])
        @test occursin("─", lines[2])
        @test occursin("4/4 entries", lines[end])
        @test occursin("Auto-scroll ON", lines[end])
        
        # Test with filtering
        state.filter_level = WARN
        content = LogPanelContent(state, width=60, height=10)
        lines = split(content, "\n")
        
        @test occursin("Filter: ≥WARN", lines[1])
        @test occursin("2/4 entries", lines[end])  # Only WARN and ERROR
        
        # Test with search
        state.filter_level = DEBUG
        state.search_pattern = r"Error"
        content = LogPanelContent(state, width=60, height=10)
        lines = split(content, "\n")
        
        @test occursin("Search: Error", lines[1])
        @test occursin("1/4 entries", lines[end])  # Only ERROR message
        
        # Test with grouping
        state.search_pattern = nothing
        state.group_repeated = true
        add_log!(buffer, INFO, "Repeated", "Module1")
        add_log!(buffer, INFO, "Repeated", "Module1")
        add_log!(buffer, INFO, "Repeated", "Module1")
        
        content = LogPanelContent(state, width=60, height=10)
        @test occursin("[×3]", content)  # Grouped message count
    end
    
    @testset "Color Formatting" begin
        theme = create_theme()
        
        # Test log level colors
        @test get_log_level_color(theme, DEBUG) == get_status_color(theme, :dim)
        @test get_log_level_color(theme, INFO) == get_status_color(theme, :text)
        @test get_log_level_color(theme, WARN) == get_status_color(theme, :warning)
        @test get_log_level_color(theme, ERROR) == get_status_color(theme, :critical)
        
        # Test timestamp formatting
        timestamp = DateTime(2024, 1, 1, 12, 30, 45, 123)
        formatted = format_timestamp(timestamp, theme)
        @test occursin("12:30:45.123", formatted)
        @test occursin("\033[", formatted)  # Contains ANSI codes
        
        # Test log level formatting
        formatted = format_log_level(INFO, theme)
        @test occursin("INFO", formatted)
        @test length(strip(formatted)) >= 5  # Padded to 5 chars
        
        # Test search highlighting
        text = "This is a test message with test word"
        pattern = r"test"
        highlighted = highlight_matches(text, pattern, theme)
        @test occursin("\033[48", highlighted)  # Background color
        @test count(s -> occursin("\033[48", s), split(highlighted, "test")) >= 2
    end
    
    @testset "Edge Cases" begin
        # Empty buffer
        buffer = CircularLogBuffer(5)
        @test isempty(get_entries(buffer))
        
        # Single entry
        add_log!(buffer, INFO, "Only entry")
        entries = get_entries(buffer)
        @test length(entries) == 1
        @test entries[1].message == "Only entry"
        
        # Exact capacity
        for i in 1:5
            add_log!(buffer, INFO, "Entry $i")
        end
        @test length(get_entries(buffer)) == 5
        
        # Empty search results
        entries = [LogEntry(now(), INFO, "Message", "Source", Dict(), 1)]
        results = search_logs(entries, r"NotFound")
        @test isempty(results)
        
        # Empty metadata
        entry = LogEntry(now(), INFO, "Message", "", Dict(), 1)
        formatted = format_log_entry(entry, detailed=true)
        @test !occursin("Metadata:", formatted)
        
        # Very long source name
        long_source = "A" ^ 50
        entry = LogEntry(now(), INFO, "Message", long_source, Dict(), 1)
        formatted = format_log_entry(entry, detailed=false, width=80)
        @test occursin("[" * long_source * "]", formatted)
    end
end