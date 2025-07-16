module DebugLogger

using Dates
using JSON3
using Printf

export DebugLogManager, set_log_level!, log_message, log_debug, log_info, log_warn, log_error
export flush_logs, get_log_stats

"""
Log level enumeration
"""
@enum LogLevel begin
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4
end

"""
Log entry structure
"""
struct LogEntry
    timestamp::DateTime
    level::LogLevel
    subsystem::String
    message::String
    data::Dict{Symbol, Any}
    thread_id::Int
    gpu_id::Union{Int, Nothing}
end

"""
Debug log manager for ensemble debugging
"""
mutable struct DebugLogManager
    log_entries::Vector{LogEntry}
    log_level::LogLevel
    output_dir::String
    log_file::Union{IOStream, Nothing}
    max_entries::Int
    subsystem_filters::Set{String}
    config::Dict{String, Any}
    stats::Dict{String, Int}
    
    function DebugLogManager(
        output_dir::String,
        config::Dict{String, Any} = Dict();
        max_entries::Int = 10000
    )
        mkpath(output_dir)
        
        # Open log file
        log_path = joinpath(output_dir, "ensemble_debug_$(Dates.format(now(), "yyyymmdd_HHMMSS")).log")
        log_file = open(log_path, "w")
        
        # Set initial log level
        log_level = get(config, "log_level", :info)
        level = if log_level == :debug
            DEBUG
        elseif log_level == :warn
            WARN
        elseif log_level == :error
            ERROR
        else
            INFO
        end
        
        # Initialize subsystem filters
        filters = get(config, "subsystem_filters", String[])
        subsystem_filters = Set{String}(filters)
        
        # Initialize stats
        stats = Dict{String, Int}(
            "debug" => 0,
            "info" => 0,
            "warn" => 0,
            "error" => 0
        )
        
        new(
            LogEntry[],
            level,
            output_dir,
            log_file,
            max_entries,
            subsystem_filters,
            config,
            stats
        )
    end
end

"""
Set logging level
"""
function set_log_level!(logger::DebugLogManager, level::Symbol)
    logger.log_level = if level == :debug
        DEBUG
    elseif level == :warn
        WARN
    elseif level == :error
        ERROR
    else
        INFO
    end
end

"""
Log a message with specified level
"""
function log_message(
    logger::DebugLogManager,
    level::Symbol,
    message::String;
    subsystem::String = "general",
    gpu_id::Union{Int, Nothing} = nothing,
    kwargs...
)
    # Convert symbol to LogLevel
    log_level = if level == :debug
        DEBUG
    elseif level == :info
        INFO
    elseif level == :warn
        WARN
    elseif level == :error
        ERROR
    else
        INFO
    end
    
    # Check if should log
    if Int(log_level) < Int(logger.log_level)
        return
    end
    
    # Check subsystem filter
    if !isempty(logger.subsystem_filters) && !(subsystem in logger.subsystem_filters)
        return
    end
    
    # Create log entry
    entry = LogEntry(
        now(),
        log_level,
        subsystem,
        message,
        Dict{Symbol, Any}(kwargs...),
        Threads.threadid(),
        gpu_id
    )
    
    # Add to entries (with circular buffer)
    if length(logger.log_entries) >= logger.max_entries
        popfirst!(logger.log_entries)
    end
    push!(logger.log_entries, entry)
    
    # Update stats
    level_str = lowercase(string(level))
    if haskey(logger.stats, level_str)
        logger.stats[level_str] += 1
    end
    
    # Write to file
    write_log_entry(logger, entry)
    
    # Also print to console for warnings and errors
    if log_level >= WARN
        print_log_entry(entry)
    end
end

"""
Convenience functions for different log levels
"""
function log_debug(logger::DebugLogManager, message::String; kwargs...)
    log_message(logger, :debug, message; kwargs...)
end

function log_info(logger::DebugLogManager, message::String; kwargs...)
    log_message(logger, :info, message; kwargs...)
end

function log_warn(logger::DebugLogManager, message::String; kwargs...)
    log_message(logger, :warn, message; kwargs...)
end

function log_error(logger::DebugLogManager, message::String; kwargs...)
    log_message(logger, :error, message; kwargs...)
end

"""
Write log entry to file
"""
function write_log_entry(logger::DebugLogManager, entry::LogEntry)
    if isnothing(logger.log_file)
        return
    end
    
    # Format timestamp
    timestamp_str = Dates.format(entry.timestamp, "yyyy-mm-dd HH:MM:SS.sss")
    
    # Format level
    level_str = string(entry.level)
    
    # Format GPU info
    gpu_str = isnothing(entry.gpu_id) ? "" : "[GPU$(entry.gpu_id)]"
    
    # Format thread info
    thread_str = "[T$(entry.thread_id)]"
    
    # Format data
    data_str = if !isempty(entry.data)
        " | " * join(["$k=$v" for (k, v) in entry.data], ", ")
    else
        ""
    end
    
    # Write formatted entry
    println(logger.log_file, 
        "$timestamp_str $level_str $thread_str$gpu_str [$(entry.subsystem)] $(entry.message)$data_str"
    )
    
    # Flush periodically
    if logger.stats["debug"] + logger.stats["info"] + logger.stats["warn"] + logger.stats["error"] % 100 == 0
        flush(logger.log_file)
    end
end

"""
Print log entry to console
"""
function print_log_entry(entry::LogEntry)
    # Color based on level
    color = if entry.level == ERROR
        :red
    elseif entry.level == WARN
        :yellow
    elseif entry.level == INFO
        :green
    else
        :normal
    end
    
    timestamp_str = Dates.format(entry.timestamp, "HH:MM:SS")
    gpu_str = isnothing(entry.gpu_id) ? "" : "[GPU$(entry.gpu_id)]"
    
    printstyled(
        "$timestamp_str [$(entry.subsystem)]$gpu_str $(entry.message)\n",
        color = color
    )
end

"""
Flush logs to disk
"""
function flush_logs(logger::DebugLogManager)
    if !isnothing(logger.log_file)
        flush(logger.log_file)
        
        # Also save JSON summary
        save_log_summary(logger)
    end
end

"""
Save log summary in JSON format
"""
function save_log_summary(logger::DebugLogManager)
    summary_path = joinpath(logger.output_dir, "log_summary_$(Dates.format(now(), "yyyymmdd_HHMMSS")).json")
    
    summary = Dict{String, Any}(
        "total_entries" => length(logger.log_entries),
        "stats" => logger.stats,
        "subsystems" => count_by_subsystem(logger),
        "errors" => extract_errors(logger),
        "warnings" => extract_warnings(logger),
        "performance_events" => extract_performance_events(logger)
    )
    
    open(summary_path, "w") do io
        JSON3.pretty(io, summary)
    end
end

"""
Count log entries by subsystem
"""
function count_by_subsystem(logger::DebugLogManager)
    counts = Dict{String, Int}()
    
    for entry in logger.log_entries
        if !haskey(counts, entry.subsystem)
            counts[entry.subsystem] = 0
        end
        counts[entry.subsystem] += 1
    end
    
    return counts
end

"""
Extract error entries
"""
function extract_errors(logger::DebugLogManager)
    errors = []
    
    for entry in logger.log_entries
        if entry.level == ERROR
            push!(errors, Dict(
                "timestamp" => entry.timestamp,
                "subsystem" => entry.subsystem,
                "message" => entry.message,
                "data" => entry.data
            ))
        end
    end
    
    return errors
end

"""
Extract warning entries
"""
function extract_warnings(logger::DebugLogManager)
    warnings = []
    
    for entry in logger.log_entries
        if entry.level == WARN
            push!(warnings, Dict(
                "timestamp" => entry.timestamp,
                "subsystem" => entry.subsystem,
                "message" => entry.message,
                "data" => entry.data
            ))
        end
    end
    
    return warnings
end

"""
Extract performance-related events
"""
function extract_performance_events(logger::DebugLogManager)
    perf_events = []
    
    # Keywords that indicate performance events
    perf_keywords = ["time", "duration", "latency", "throughput", "fps", "ms", "seconds"]
    
    for entry in logger.log_entries
        # Check if message or data contains performance keywords
        is_perf = any(occursin(keyword, lowercase(entry.message)) for keyword in perf_keywords)
        
        if !is_perf
            for (k, v) in entry.data
                if any(occursin(keyword, lowercase(string(k))) for keyword in perf_keywords)
                    is_perf = true
                    break
                end
            end
        end
        
        if is_perf
            push!(perf_events, Dict(
                "timestamp" => entry.timestamp,
                "subsystem" => entry.subsystem,
                "message" => entry.message,
                "data" => entry.data
            ))
        end
    end
    
    return perf_events
end

"""
Get logging statistics
"""
function get_log_stats(logger::DebugLogManager)
    return Dict{String, Any}(
        "total_entries" => length(logger.log_entries),
        "by_level" => logger.stats,
        "by_subsystem" => count_by_subsystem(logger),
        "by_thread" => count_by_thread(logger),
        "by_gpu" => count_by_gpu(logger)
    )
end

"""
Count log entries by thread
"""
function count_by_thread(logger::DebugLogManager)
    counts = Dict{Int, Int}()
    
    for entry in logger.log_entries
        if !haskey(counts, entry.thread_id)
            counts[entry.thread_id] = 0
        end
        counts[entry.thread_id] += 1
    end
    
    return counts
end

"""
Count log entries by GPU
"""
function count_by_gpu(logger::DebugLogManager)
    counts = Dict{String, Int}("cpu" => 0)
    
    for entry in logger.log_entries
        if isnothing(entry.gpu_id)
            counts["cpu"] += 1
        else
            gpu_key = "gpu_$( entry.gpu_id)"
            if !haskey(counts, gpu_key)
                counts[gpu_key] = 0
            end
            counts[gpu_key] += 1
        end
    end
    
    return counts
end

"""
Search logs for specific patterns
"""
function search_logs(logger::DebugLogManager, pattern::String; subsystem::Union{String, Nothing} = nothing)
    results = LogEntry[]
    
    for entry in logger.log_entries
        # Check subsystem filter
        if !isnothing(subsystem) && entry.subsystem != subsystem
            continue
        end
        
        # Check if pattern matches message
        if occursin(pattern, entry.message)
            push!(results, entry)
            continue
        end
        
        # Check if pattern matches any data values
        for (k, v) in entry.data
            if occursin(pattern, string(v))
                push!(results, entry)
                break
            end
        end
    end
    
    return results
end

"""
Export logs to different formats
"""
function export_logs(logger::DebugLogManager, format::Symbol, output_path::String)
    if format == :csv
        export_to_csv(logger, output_path)
    elseif format == :json
        export_to_json(logger, output_path)
    else
        error("Unsupported format: $format")
    end
end

"""
Export logs to CSV
"""
function export_to_csv(logger::DebugLogManager, output_path::String)
    open(output_path, "w") do io
        # Write header
        println(io, "timestamp,level,subsystem,thread_id,gpu_id,message,data")
        
        # Write entries
        for entry in logger.log_entries
            data_str = JSON3.write(entry.data)
            gpu_str = isnothing(entry.gpu_id) ? "" : string(entry.gpu_id)
            
            println(io, 
                "$(entry.timestamp),$(entry.level),$(entry.subsystem),$(entry.thread_id),$gpu_str,\"$(entry.message)\",\"$data_str\""
            )
        end
    end
end

"""
Export logs to JSON
"""
function export_to_json(logger::DebugLogManager, output_path::String)
    entries = [
        Dict(
            "timestamp" => entry.timestamp,
            "level" => string(entry.level),
            "subsystem" => entry.subsystem,
            "thread_id" => entry.thread_id,
            "gpu_id" => entry.gpu_id,
            "message" => entry.message,
            "data" => entry.data
        ) for entry in logger.log_entries
    ]
    
    open(output_path, "w") do io
        JSON3.pretty(io, entries)
    end
end

"""
Finalize logger and close files
"""
function Base.close(logger::DebugLogManager)
    flush_logs(logger)
    
    if !isnothing(logger.log_file)
        close(logger.log_file)
        logger.log_file = nothing
    end
end

end # module