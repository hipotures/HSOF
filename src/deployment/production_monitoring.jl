module ProductionMonitoring

using CUDA
using Dates
using JSON3
using Printf
using Statistics
using Base.Threads: @spawn

export MonitoringConfig, MetricsCollector, LogManager, AlertManager
export start_monitoring!, stop_monitoring!, log_event, log_metric
export check_alerts, get_metrics_summary, export_metrics

"""
Monitoring configuration for production systems
"""
struct MonitoringConfig
    # Metrics collection
    metrics_interval_ms::Int  # How often to collect metrics
    metrics_retention_hours::Int  # How long to keep metrics
    
    # Logging settings
    log_level::Symbol  # :debug, :info, :warn, :error
    log_file_prefix::String
    log_rotation_size_mb::Int
    log_retention_days::Int
    
    # Alert thresholds
    gpu_utilization_low::Float64  # Alert if below
    gpu_utilization_high::Float64  # Alert if above
    memory_usage_critical::Float64  # Alert if memory usage above
    temperature_critical::Float64  # Alert if temperature above
    sync_latency_critical_ms::Float64  # Alert if sync takes too long
    
    # Export settings
    export_format::Symbol  # :json, :csv, :prometheus
    export_interval_minutes::Int
    export_path::String
end

"""
Metrics data point
"""
struct MetricPoint
    timestamp::DateTime
    gpu_id::Int
    metric_name::String
    value::Float64
    tags::Dict{String, String}
end

"""
Log entry
"""
struct LogEntry
    timestamp::DateTime
    level::Symbol
    component::String
    message::String
    metadata::Dict{String, Any}
end

"""
Alert definition
"""
struct Alert
    id::String
    timestamp::DateTime
    severity::Symbol  # :warning, :critical
    component::String
    message::String
    value::Float64
    threshold::Float64
end

"""
Metrics collector for GPU and system metrics
"""
mutable struct MetricsCollector
    config::MonitoringConfig
    metrics_buffer::Vector{MetricPoint}
    is_running::Atomic{Bool}
    collection_task::Union{Task, Nothing}
    last_export::DateTime
    
    function MetricsCollector(config::MonitoringConfig)
        new(config, Vector{MetricPoint}(), Atomic{Bool}(false), nothing, now())
    end
end

"""
Log manager for structured logging
"""
mutable struct LogManager
    config::MonitoringConfig
    log_buffer::Vector{LogEntry}
    current_log_file::Union{IOStream, Nothing}
    log_size::Int
    log_lock::ReentrantLock
    
    function LogManager(config::MonitoringConfig)
        new(config, Vector{LogEntry}(), nothing, 0, ReentrantLock())
    end
end

"""
Alert manager for monitoring alerts
"""
mutable struct AlertManager
    config::MonitoringConfig
    active_alerts::Dict{String, Alert}
    alert_history::Vector{Alert}
    alert_callbacks::Vector{Function}
    
    function AlertManager(config::MonitoringConfig)
        new(config, Dict{String, Alert}(), Vector{Alert}(), Vector{Function}())
    end
end

"""
Start metrics collection
"""
function start_monitoring!(collector::MetricsCollector)
    if collector.is_running[]
        @warn "Monitoring already running"
        return
    end
    
    collector.is_running[] = true
    
    collector.collection_task = @spawn begin
        while collector.is_running[]
            try
                collect_metrics!(collector)
                
                # Check if export needed
                if (now() - collector.last_export).value > collector.config.export_interval_minutes * 60000
                    export_metrics(collector)
                    collector.last_export = now()
                end
                
                # Clean old metrics
                clean_old_metrics!(collector)
                
            catch e
                @error "Error in metrics collection" exception=(e, catch_backtrace())
            end
            
            sleep(collector.config.metrics_interval_ms / 1000)
        end
    end
    
    @info "Monitoring started" interval_ms=collector.config.metrics_interval_ms
end

"""
Stop metrics collection
"""
function stop_monitoring!(collector::MetricsCollector)
    collector.is_running[] = false
    
    if !isnothing(collector.collection_task)
        wait(collector.collection_task)
    end
    
    # Final export
    export_metrics(collector)
    
    @info "Monitoring stopped"
end

"""
Collect current metrics
"""
function collect_metrics!(collector::MetricsCollector)
    timestamp = now()
    
    # Collect GPU metrics
    for gpu_id in 0:length(CUDA.devices())-1
        try
            device!(gpu_id)
            
            # GPU utilization (requires NVML)
            # This is a placeholder - actual implementation would use NVML
            gpu_util = 0.0  # Would use NVML_jl or similar
            
            # Memory usage
            free_mem = CUDA.available_memory()
            total_mem = CUDA.total_memory()
            used_mem = total_mem - free_mem
            memory_usage = used_mem / total_mem
            
            # Add metrics
            push!(collector.metrics_buffer, MetricPoint(
                timestamp, gpu_id, "gpu_utilization", gpu_util,
                Dict("unit" => "percent")
            ))
            
            push!(collector.metrics_buffer, MetricPoint(
                timestamp, gpu_id, "memory_usage", memory_usage * 100,
                Dict("unit" => "percent")
            ))
            
            push!(collector.metrics_buffer, MetricPoint(
                timestamp, gpu_id, "memory_used_mb", used_mem / 1024^2,
                Dict("unit" => "megabytes")
            ))
            
            # Temperature (placeholder - would use NVML)
            temperature = 0.0  # Would get actual temperature
            push!(collector.metrics_buffer, MetricPoint(
                timestamp, gpu_id, "temperature", temperature,
                Dict("unit" => "celsius")
            ))
            
        catch e
            @error "Failed to collect metrics for GPU $gpu_id" exception=e
        end
    end
    
    # System metrics
    push!(collector.metrics_buffer, MetricPoint(
        timestamp, -1, "system_memory_usage",
        (Sys.total_memory() - Sys.free_memory()) / Sys.total_memory() * 100,
        Dict("unit" => "percent")
    ))
    
    push!(collector.metrics_buffer, MetricPoint(
        timestamp, -1, "julia_threads", Threads.nthreads(),
        Dict("unit" => "count")
    ))
end

"""
Clean old metrics from buffer
"""
function clean_old_metrics!(collector::MetricsCollector)
    retention_time = now() - Hour(collector.config.metrics_retention_hours)
    
    filter!(m -> m.timestamp > retention_time, collector.metrics_buffer)
end

"""
Log an event
"""
function log_event(
    log_manager::LogManager,
    level::Symbol,
    component::String,
    message::String;
    metadata::Dict{String, Any} = Dict{String, Any}()
)
    # Check log level
    log_levels = Dict(:debug => 0, :info => 1, :warn => 2, :error => 3)
    if log_levels[level] < log_levels[log_manager.config.log_level]
        return
    end
    
    entry = LogEntry(now(), level, component, message, metadata)
    
    lock(log_manager.log_lock) do
        push!(log_manager.log_buffer, entry)
        
        # Write to file
        write_log_entry(log_manager, entry)
        
        # Also print to console for important messages
        if level in [:warn, :error]
            println("[$level] $(entry.timestamp) - $component: $message")
        end
    end
end

"""
Write log entry to file
"""
function write_log_entry(log_manager::LogManager, entry::LogEntry)
    # Open log file if needed
    if isnothing(log_manager.current_log_file)
        log_file = "$(log_manager.config.log_file_prefix)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).log"
        log_manager.current_log_file = open(log_file, "w")
        log_manager.log_size = 0
    end
    
    # Format entry
    log_line = Dict(
        "timestamp" => string(entry.timestamp),
        "level" => string(entry.level),
        "component" => entry.component,
        "message" => entry.message,
        "metadata" => entry.metadata
    )
    
    # Write JSON line
    json_line = JSON3.write(log_line)
    println(log_manager.current_log_file, json_line)
    flush(log_manager.current_log_file)
    
    log_manager.log_size += length(json_line) + 1
    
    # Check rotation
    if log_manager.log_size > log_manager.config.log_rotation_size_mb * 1024^2
        close(log_manager.current_log_file)
        log_manager.current_log_file = nothing
        log_manager.log_size = 0
    end
end

"""
Log a metric value
"""
function log_metric(
    collector::MetricsCollector,
    metric_name::String,
    value::Float64;
    gpu_id::Int = -1,
    tags::Dict{String, String} = Dict{String, String}()
)
    push!(collector.metrics_buffer, MetricPoint(
        now(), gpu_id, metric_name, value, tags
    ))
end

"""
Check for alert conditions
"""
function check_alerts(
    alert_manager::AlertManager,
    collector::MetricsCollector
)::Vector{Alert}
    new_alerts = Alert[]
    current_time = now()
    
    # Get recent metrics (last 60 seconds)
    recent_time = current_time - Second(60)
    recent_metrics = filter(m -> m.timestamp > recent_time, collector.metrics_buffer)
    
    # Group by GPU and metric
    for gpu_id in 0:length(CUDA.devices())-1
        gpu_metrics = filter(m -> m.gpu_id == gpu_id, recent_metrics)
        
        # Check GPU utilization
        util_metrics = filter(m -> m.metric_name == "gpu_utilization", gpu_metrics)
        if !isempty(util_metrics)
            avg_util = mean(m.value for m in util_metrics)
            
            if avg_util < alert_manager.config.gpu_utilization_low
                alert = Alert(
                    "gpu_$(gpu_id)_low_util",
                    current_time,
                    :warning,
                    "GPU_$gpu_id",
                    "Low GPU utilization",
                    avg_util,
                    alert_manager.config.gpu_utilization_low
                )
                push!(new_alerts, alert)
            elseif avg_util > alert_manager.config.gpu_utilization_high
                alert = Alert(
                    "gpu_$(gpu_id)_high_util",
                    current_time,
                    :warning,
                    "GPU_$gpu_id",
                    "High GPU utilization",
                    avg_util,
                    alert_manager.config.gpu_utilization_high
                )
                push!(new_alerts, alert)
            end
        end
        
        # Check memory usage
        mem_metrics = filter(m -> m.metric_name == "memory_usage", gpu_metrics)
        if !isempty(mem_metrics)
            max_mem = maximum(m.value for m in mem_metrics)
            
            if max_mem > alert_manager.config.memory_usage_critical
                alert = Alert(
                    "gpu_$(gpu_id)_high_memory",
                    current_time,
                    :critical,
                    "GPU_$gpu_id",
                    "Critical memory usage",
                    max_mem,
                    alert_manager.config.memory_usage_critical
                )
                push!(new_alerts, alert)
            end
        end
        
        # Check temperature
        temp_metrics = filter(m -> m.metric_name == "temperature", gpu_metrics)
        if !isempty(temp_metrics)
            max_temp = maximum(m.value for m in temp_metrics)
            
            if max_temp > alert_manager.config.temperature_critical
                alert = Alert(
                    "gpu_$(gpu_id)_high_temp",
                    current_time,
                    :critical,
                    "GPU_$gpu_id",
                    "Critical temperature",
                    max_temp,
                    alert_manager.config.temperature_critical
                )
                push!(new_alerts, alert)
            end
        end
    end
    
    # Update alert manager
    for alert in new_alerts
        alert_manager.active_alerts[alert.id] = alert
        push!(alert_manager.alert_history, alert)
        
        # Call alert callbacks
        for callback in alert_manager.alert_callbacks
            try
                callback(alert)
            catch e
                @error "Alert callback failed" exception=e
            end
        end
    end
    
    return new_alerts
end

"""
Get metrics summary
"""
function get_metrics_summary(
    collector::MetricsCollector;
    time_window_minutes::Int = 5
)::Dict{String, Any}
    current_time = now()
    window_start = current_time - Minute(time_window_minutes)
    
    recent_metrics = filter(m -> m.timestamp > window_start, collector.metrics_buffer)
    
    summary = Dict{String, Any}()
    
    # Summary by GPU
    for gpu_id in 0:length(CUDA.devices())-1
        gpu_metrics = filter(m -> m.gpu_id == gpu_id, recent_metrics)
        
        gpu_summary = Dict{String, Any}()
        
        # Aggregate by metric name
        for metric_name in unique(m.metric_name for m in gpu_metrics)
            metric_values = [m.value for m in gpu_metrics if m.metric_name == metric_name]
            
            if !isempty(metric_values)
                gpu_summary[metric_name] = Dict(
                    "mean" => mean(metric_values),
                    "min" => minimum(metric_values),
                    "max" => maximum(metric_values),
                    "count" => length(metric_values)
                )
            end
        end
        
        summary["gpu_$gpu_id"] = gpu_summary
    end
    
    # System metrics
    system_metrics = filter(m -> m.gpu_id == -1, recent_metrics)
    system_summary = Dict{String, Any}()
    
    for metric_name in unique(m.metric_name for m in system_metrics)
        metric_values = [m.value for m in system_metrics if m.metric_name == metric_name]
        
        if !isempty(metric_values)
            system_summary[metric_name] = Dict(
                "mean" => mean(metric_values),
                "min" => minimum(metric_values),
                "max" => maximum(metric_values),
                "count" => length(metric_values)
            )
        end
    end
    
    summary["system"] = system_summary
    summary["time_window_minutes"] = time_window_minutes
    summary["metric_count"] = length(recent_metrics)
    
    return summary
end

"""
Export metrics in specified format
"""
function export_metrics(collector::MetricsCollector)
    if isempty(collector.metrics_buffer)
        return
    end
    
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    
    if collector.config.export_format == :json
        export_json(collector, timestamp)
    elseif collector.config.export_format == :csv
        export_csv(collector, timestamp)
    elseif collector.config.export_format == :prometheus
        export_prometheus(collector, timestamp)
    end
    
    @info "Metrics exported" format=collector.config.export_format count=length(collector.metrics_buffer)
end

"""
Export metrics as JSON
"""
function export_json(collector::MetricsCollector, timestamp::String)
    filename = joinpath(collector.config.export_path, "metrics_$timestamp.json")
    
    metrics_data = [
        Dict(
            "timestamp" => string(m.timestamp),
            "gpu_id" => m.gpu_id,
            "metric" => m.metric_name,
            "value" => m.value,
            "tags" => m.tags
        )
        for m in collector.metrics_buffer
    ]
    
    open(filename, "w") do io
        JSON3.pretty(io, metrics_data)
    end
end

"""
Export metrics as CSV
"""
function export_csv(collector::MetricsCollector, timestamp::String)
    filename = joinpath(collector.config.export_path, "metrics_$timestamp.csv")
    
    open(filename, "w") do io
        # Header
        println(io, "timestamp,gpu_id,metric,value,tags")
        
        # Data
        for m in collector.metrics_buffer
            tags_str = join(["$k=$v" for (k,v) in m.tags], ";")
            println(io, "$(m.timestamp),$(m.gpu_id),$(m.metric_name),$(m.value),\"$tags_str\"")
        end
    end
end

"""
Export metrics in Prometheus format
"""
function export_prometheus(collector::MetricsCollector, timestamp::String)
    filename = joinpath(collector.config.export_path, "metrics_$timestamp.prom")
    
    open(filename, "w") do io
        # Group by metric name
        metrics_by_name = Dict{String, Vector{MetricPoint}}()
        
        for m in collector.metrics_buffer
            if !haskey(metrics_by_name, m.metric_name)
                metrics_by_name[m.metric_name] = MetricPoint[]
            end
            push!(metrics_by_name[m.metric_name], m)
        end
        
        # Write each metric
        for (metric_name, points) in metrics_by_name
            # Metric help
            println(io, "# HELP hsof_$metric_name HSOF metric: $metric_name")
            println(io, "# TYPE hsof_$metric_name gauge")
            
            # Metric values
            for p in points
                labels = ["gpu_id=\"$(p.gpu_id)\""]
                for (k, v) in p.tags
                    push!(labels, "$k=\"$v\"")
                end
                label_str = join(labels, ",")
                
                println(io, "hsof_$metric_name{$label_str} $(p.value)")
            end
            
            println(io)  # Blank line between metrics
        end
    end
end

"""
Create default monitoring configuration
"""
function create_default_monitoring_config()::MonitoringConfig
    return MonitoringConfig(
        # Metrics
        5000,  # Collect every 5 seconds
        24,    # Keep 24 hours
        
        # Logging
        :info,
        "logs/gpu/hsof",
        100,   # 100MB log files
        7,     # Keep 7 days
        
        # Alerts
        20.0,  # Low GPU util
        95.0,  # High GPU util
        90.0,  # Critical memory
        85.0,  # Critical temperature
        100.0, # Critical sync latency
        
        # Export
        :json,
        5,     # Export every 5 minutes
        "logs/metrics"
    )
end

end # module