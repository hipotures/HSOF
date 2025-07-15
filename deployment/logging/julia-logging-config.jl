# Julia Logging Configuration for HSOF Production
# This file configures structured logging with correlation IDs for the HSOF system

using Logging, LoggingExtras, JSON3, UUIDs, Dates

"""
    StructuredLogger

Custom logger that outputs structured JSON logs with correlation IDs
and metadata for the HSOF system.
"""
struct StructuredLogger <: AbstractLogger
    stream::IO
    min_level::LogLevel
    correlation_id::String
    component::String
    
    function StructuredLogger(stream::IO=stdout; 
                             min_level::LogLevel=Logging.Info,
                             correlation_id::String=string(uuid4()),
                             component::String="hsof")
        new(stream, min_level, correlation_id, component)
    end
end

Logging.min_enabled_level(logger::StructuredLogger) = logger.min_level
Logging.shouldlog(logger::StructuredLogger, level, _module, group, id) = true
Logging.catch_exceptions(logger::StructuredLogger) = true

function Logging.handle_message(logger::StructuredLogger, level, message, _module, group, id, filepath, line; kwargs...)
    # Create structured log entry
    log_entry = Dict(
        "timestamp" => Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "level" => string(level),
        "message" => string(message),
        "correlation_id" => logger.correlation_id,
        "component" => logger.component,
        "module" => string(_module),
        "file" => basename(string(filepath)),
        "line" => line,
        "group" => string(group),
        "id" => string(id)
    )
    
    # Add additional context from kwargs
    for (key, value) in kwargs
        if key in [:gpu_id, :stage, :dataset_id, :request_id, :trace_id, 
                  :duration, :features_processed, :batch_size, :error_code,
                  :gpu_utilization, :gpu_memory_used, :gpu_temperature]
            log_entry[string(key)] = value
        elseif key == :exception && value isa Exception
            log_entry["exception"] = Dict(
                "type" => string(typeof(value)),
                "message" => string(value),
                "stacktrace" => string.(stacktrace(catch_backtrace()))
            )
        else
            log_entry["extra_$(string(key))"] = string(value)
        end
    end
    
    # Output as JSON
    json_log = JSON3.write(log_entry)
    println(logger.stream, json_log)
    flush(logger.stream)
end

"""
    GPULogger

Specialized logger for GPU-related operations with automatic GPU context.
"""
struct GPULogger <: AbstractLogger
    base_logger::StructuredLogger
    gpu_id::String
    
    function GPULogger(gpu_id::String; kwargs...)
        base = StructuredLogger(; component="gpu-monitor", kwargs...)
        new(base, gpu_id)
    end
end

Logging.min_enabled_level(logger::GPULogger) = logger.base_logger.min_level
Logging.shouldlog(logger::GPULogger, level, _module, group, id) = true
Logging.catch_exceptions(logger::GPULogger) = true

function Logging.handle_message(logger::GPULogger, level, message, _module, group, id, filepath, line; kwargs...)
    # Add GPU ID to kwargs
    gpu_kwargs = merge(kwargs, (gpu_id=logger.gpu_id,))
    Logging.handle_message(logger.base_logger, level, message, _module, group, id, filepath, line; gpu_kwargs...)
end

"""
    PipelineLogger

Specialized logger for pipeline stages with automatic stage context.
"""
struct PipelineLogger <: AbstractLogger
    base_logger::StructuredLogger
    stage::String
    dataset_id::String
    
    function PipelineLogger(stage::String, dataset_id::String; kwargs...)
        base = StructuredLogger(; component="pipeline", kwargs...)
        new(base, stage, dataset_id)
    end
end

Logging.min_enabled_level(logger::PipelineLogger) = logger.base_logger.min_level
Logging.shouldlog(logger::PipelineLogger, level, _module, group, id) = true
Logging.catch_exceptions(logger::PipelineLogger) = true

function Logging.handle_message(logger::PipelineLogger, level, message, _module, group, id, filepath, line; kwargs...)
    # Add stage and dataset context
    stage_kwargs = merge(kwargs, (stage=logger.stage, dataset_id=logger.dataset_id))
    Logging.handle_message(logger.base_logger, level, message, _module, group, id, filepath, line; stage_kwargs...)
end

"""
    setup_production_logging(correlation_id=nothing)

Sets up production logging configuration with structured JSON output.
"""
function setup_production_logging(correlation_id=nothing)
    # Generate correlation ID if not provided
    if correlation_id === nothing
        correlation_id = string(uuid4())
    end
    
    # Create structured logger
    structured_logger = StructuredLogger(
        stdout,
        min_level=Logging.Info,
        correlation_id=correlation_id,
        component="hsof"
    )
    
    # Set as global logger
    global_logger(structured_logger)
    
    @info "Production logging initialized" correlation_id=correlation_id
    
    return correlation_id
end

"""
    log_gpu_metrics(gpu_id, utilization, memory_used, temperature; logger=nothing)

Log GPU metrics in structured format.
"""
function log_gpu_metrics(gpu_id::String, utilization::Float64, memory_used::Int64, temperature::Float64; logger=nothing)
    if logger === nothing
        logger = current_logger()
    end
    
    @info "GPU metrics" gpu_id=gpu_id gpu_utilization=utilization gpu_memory_used=memory_used gpu_temperature=temperature
end

"""
    log_stage_performance(stage, duration, features_processed; dataset_id="", logger=nothing)

Log pipeline stage performance metrics.
"""
function log_stage_performance(stage::String, duration::Float64, features_processed::Int; dataset_id::String="", logger=nothing)
    if logger === nothing
        logger = current_logger()
    end
    
    @info "Stage completed" stage=stage duration=duration features_processed=features_processed dataset_id=dataset_id
end

"""
    log_error_with_context(message, exception=nothing; kwargs...)

Log errors with full context and exception details.
"""
function log_error_with_context(message::String, exception=nothing; kwargs...)
    if exception !== nothing
        @error message exception=exception kwargs...
    else
        @error message kwargs...
    end
end

"""
    create_correlation_context(correlation_id=nothing)

Create a correlation context that can be passed between functions.
"""
function create_correlation_context(correlation_id=nothing)
    if correlation_id === nothing
        correlation_id = string(uuid4())
    end
    
    return Dict(
        :correlation_id => correlation_id,
        :request_id => correlation_id,
        :trace_id => correlation_id,
        :created_at => now(UTC)
    )
end

"""
    with_correlation_context(f, context)

Execute function with correlation context in logging.
"""
function with_correlation_context(f, context::Dict)
    # Create temporary logger with correlation context
    temp_logger = StructuredLogger(
        stdout,
        correlation_id=context[:correlation_id],
        component=get(context, :component, "hsof")
    )
    
    # Execute function with temporary logger
    with_logger(temp_logger) do
        f()
    end
end

# Export main functions
export StructuredLogger, GPULogger, PipelineLogger
export setup_production_logging, log_gpu_metrics, log_stage_performance
export log_error_with_context, create_correlation_context, with_correlation_context