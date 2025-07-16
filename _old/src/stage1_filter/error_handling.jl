module ErrorHandling

using CUDA
using Logging
using Dates
using Printf

export GPUError, MemoryError, NumericalError, KernelError
export @cuda_check, @memory_check, @numerical_check
export ErrorRecovery, ErrorLogger, RecoveryStrategy
export create_error_recovery, handle_error!, log_error!
export with_gpu_recovery, with_numerical_checks

"""
Custom error types for GPU operations
"""
abstract type GPUError <: Exception end

struct MemoryError <: GPUError
    message::String
    requested_bytes::Int
    available_bytes::Int
    device::Int
end

struct KernelError <: GPUError
    message::String
    kernel_name::String
    error_code::CUDA.CuError
    thread_config::NamedTuple
end

struct NumericalError <: GPUError
    message::String
    operation::String
    values::Dict{String, Any}
end

"""
Recovery strategies for different error types
"""
@enum RecoveryStrategy begin
    RETRY_WITH_SMALLER_BATCH
    FALLBACK_TO_CPU
    CLEAR_CACHE_AND_RETRY
    REDUCE_PRECISION
    ABORT_WITH_LOGGING
end

"""
Error recovery configuration
"""
mutable struct ErrorRecovery
    max_retries::Int
    retry_delay_ms::Int
    memory_pressure_threshold::Float32
    enable_cpu_fallback::Bool
    strategies::Dict{Type{<:GPUError}, Vector{RecoveryStrategy}}
    retry_counts::Dict{String, Int}
end

"""
Error logging system
"""
mutable struct ErrorLogger
    log_file::String
    max_log_size::Int
    rotation_count::Int
    detail_level::Symbol  # :minimal, :normal, :verbose
    include_stack_trace::Bool
    include_gpu_state::Bool
end

"""
Create default error recovery configuration
"""
function create_error_recovery(;
    max_retries::Int = 3,
    retry_delay_ms::Int = 100,
    memory_pressure_threshold::Float32 = 0.9f0,
    enable_cpu_fallback::Bool = true
)
    strategies = Dict{Type{<:GPUError}, Vector{RecoveryStrategy}}(
        MemoryError => [
            CLEAR_CACHE_AND_RETRY,
            RETRY_WITH_SMALLER_BATCH,
            FALLBACK_TO_CPU
        ],
        KernelError => [
            RETRY_WITH_SMALLER_BATCH,
            REDUCE_PRECISION,
            FALLBACK_TO_CPU
        ],
        NumericalError => [
            REDUCE_PRECISION,
            FALLBACK_TO_CPU,
            ABORT_WITH_LOGGING
        ]
    )
    
    return ErrorRecovery(
        max_retries,
        retry_delay_ms,
        memory_pressure_threshold,
        enable_cpu_fallback,
        strategies,
        Dict{String, Int}()
    )
end

"""
Create error logger
"""
function create_error_logger(;
    log_dir::String = ".taskmaster/logs",
    detail_level::Symbol = :normal
)
    mkpath(log_dir)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    log_file = joinpath(log_dir, "gpu_errors_$timestamp.log")
    
    return ErrorLogger(
        log_file,
        10 * 1024 * 1024,  # 10MB max size
        5,                  # Keep 5 rotated logs
        detail_level,
        detail_level !== :minimal,
        detail_level === :verbose
    )
end

"""
CUDA error checking macro
"""
macro cuda_check(expr)
    quote
        local result = $(esc(expr))
        if result != CUDA.SUCCESS
            error_code = CUDA.CuError(result)
            throw(KernelError(
                "CUDA operation failed: $(CUDA.description(error_code))",
                string($(QuoteNode(expr))),
                error_code,
                (threads=0, blocks=0)
            ))
        end
        result
    end
end

"""
Memory allocation checking macro
"""
macro memory_check(allocation_expr, bytes_needed)
    quote
        local mem_info = CUDA.memory_status()
        local bytes = $(esc(bytes_needed))
        
        if mem_info.free < bytes * 1.1  # 10% buffer
            throw(MemoryError(
                "Insufficient GPU memory for allocation",
                bytes,
                mem_info.free,
                CUDA.device().handle
            ))
        end
        
        try
            $(esc(allocation_expr))
        catch e
            if isa(e, OutOfGPUMemoryError)
                throw(MemoryError(
                    "GPU memory allocation failed",
                    bytes,
                    mem_info.free,
                    CUDA.device().handle
                ))
            else
                rethrow()
            end
        end
    end
end

"""
Numerical stability checking macro
"""
macro numerical_check(value_expr, check_type = :finite)
    quote
        local value = $(esc(value_expr))
        local check = $(QuoteNode(check_type))
        
        if check === :finite
            if !all(isfinite.(value))
                invalid_count = sum(.!isfinite.(value))
                throw(NumericalError(
                    "Non-finite values detected",
                    string($(QuoteNode(value_expr))),
                    Dict(
                        "invalid_count" => invalid_count,
                        "total_elements" => length(value),
                        "sample_values" => value[1:min(10, end)]
                    )
                ))
            end
        elseif check === :positive
            if !all(value .> 0)
                negative_count = sum(value .<= 0)
                throw(NumericalError(
                    "Non-positive values detected",
                    string($(QuoteNode(value_expr))),
                    Dict(
                        "negative_count" => negative_count,
                        "min_value" => minimum(value),
                        "sample_values" => value[1:min(10, end)]
                    )
                ))
            end
        elseif check === :variance
            if any(value .< -1e-6)  # Allow small negative values due to numerical errors
                throw(NumericalError(
                    "Negative variance detected",
                    string($(QuoteNode(value_expr))),
                    Dict(
                        "min_variance" => minimum(value),
                        "negative_indices" => findall(value .< 0)
                    )
                ))
            end
        end
        
        value
    end
end

"""
Handle error with recovery strategy
"""
function handle_error!(
    recovery::ErrorRecovery,
    error::GPUError,
    context::Dict{String, Any} = Dict()
)
    error_type = typeof(error)
    error_id = string(error_type) * "_" * string(hash(error.message))
    
    # Check retry count
    retry_count = get(recovery.retry_counts, error_id, 0)
    if retry_count >= recovery.max_retries
        @warn "Max retries exceeded for error" error_type=error_type message=error.message
        return ABORT_WITH_LOGGING
    end
    
    recovery.retry_counts[error_id] = retry_count + 1
    
    # Get recovery strategies for this error type
    strategies = get(recovery.strategies, error_type, [ABORT_WITH_LOGGING])
    
    # Try each strategy
    for (idx, strategy) in enumerate(strategies)
        if strategy == RETRY_WITH_SMALLER_BATCH
            if haskey(context, "batch_size") && context["batch_size"] > 1
                return strategy
            end
        elseif strategy == FALLBACK_TO_CPU
            if recovery.enable_cpu_fallback
                return strategy
            end
        elseif strategy == CLEAR_CACHE_AND_RETRY
            CUDA.reclaim()
            if retry_count == 0  # Only try once
                return strategy
            end
        elseif strategy == REDUCE_PRECISION
            if haskey(context, "precision") && context["precision"] != Float16
                return strategy
            end
        end
    end
    
    return ABORT_WITH_LOGGING
end

"""
Log error with detailed information
"""
function log_error!(
    logger::ErrorLogger,
    error::GPUError,
    context::Dict{String, Any} = Dict()
)
    timestamp = now()
    
    # Rotate log if needed
    if isfile(logger.log_file) && filesize(logger.log_file) > logger.max_log_size
        rotate_log!(logger)
    end
    
    # Open log file
    open(logger.log_file, "a") do io
        println(io, "="^80)
        println(io, "ERROR LOG ENTRY - $(Dates.format(timestamp, "yyyy-mm-dd HH:MM:SS.sss"))")
        println(io, "="^80)
        
        # Basic error info
        println(io, "Error Type: $(typeof(error))")
        println(io, "Message: $(error.message)")
        
        # Error-specific details
        if isa(error, MemoryError)
            println(io, "Requested Memory: $(format_bytes(error.requested_bytes))")
            println(io, "Available Memory: $(format_bytes(error.available_bytes))")
            println(io, "Device: $(error.device)")
        elseif isa(error, KernelError)
            println(io, "Kernel: $(error.kernel_name)")
            println(io, "Error Code: $(error.error_code)")
            println(io, "Thread Config: $(error.thread_config)")
        elseif isa(error, NumericalError)
            println(io, "Operation: $(error.operation)")
            for (key, value) in error.values
                println(io, "  $key: $value")
            end
        end
        
        # Context information
        if !isempty(context)
            println(io, "\nContext:")
            for (key, value) in context
                println(io, "  $key: $value")
            end
        end
        
        # GPU state (if verbose)
        if logger.include_gpu_state
            println(io, "\nGPU State:")
            mem_info = CUDA.memory_status()
            println(io, "  Total Memory: $(format_bytes(mem_info.total))")
            println(io, "  Free Memory: $(format_bytes(mem_info.free))")
            println(io, "  Used Memory: $(format_bytes(mem_info.total - mem_info.free))")
            println(io, "  Memory Pressure: $(round((mem_info.total - mem_info.free) / mem_info.total * 100, digits=1))%")
        end
        
        # Stack trace (if enabled)
        if logger.include_stack_trace
            println(io, "\nStack Trace:")
            for (i, frame) in enumerate(stacktrace())
                if i > 20  # Limit stack trace length
                    println(io, "  ... (truncated)")
                    break
                end
                println(io, "  $frame")
            end
        end
        
        println(io, "\n")
    end
end

"""
Format bytes to human-readable string
"""
function format_bytes(bytes::Int)
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 1
    value = Float64(bytes)
    
    while value >= 1024 && unit_idx < length(units)
        value /= 1024
        unit_idx += 1
    end
    
    return @sprintf("%.2f %s", value, units[unit_idx])
end

"""
Rotate log files
"""
function rotate_log!(logger::ErrorLogger)
    base_name = splitext(logger.log_file)[1]
    ext = splitext(logger.log_file)[2]
    
    # Shift existing rotated logs
    for i in logger.rotation_count-1:-1:1
        old_file = "$(base_name).$(i)$(ext)"
        new_file = "$(base_name).$(i+1)$(ext)"
        if isfile(old_file)
            mv(old_file, new_file, force=true)
        end
    end
    
    # Rotate current log
    if isfile(logger.log_file)
        mv(logger.log_file, "$(base_name).1$(ext)", force=true)
    end
end

"""
Execute GPU operation with automatic recovery
"""
function with_gpu_recovery(
    f::Function,
    recovery::ErrorRecovery,
    logger::ErrorLogger;
    context::Dict{String, Any} = Dict()
)
    retry_count = 0
    last_error = nothing
    
    while retry_count <= recovery.max_retries
        try
            return f()
        catch e
            if !isa(e, GPUError)
                rethrow()
            end
            
            last_error = e
            log_error!(logger, e, context)
            
            strategy = handle_error!(recovery, e, context)
            
            if strategy == ABORT_WITH_LOGGING
                break
            elseif strategy == RETRY_WITH_SMALLER_BATCH
                context["batch_size"] = get(context, "batch_size", 1000) ÷ 2
                @info "Retrying with smaller batch size" new_batch_size=context["batch_size"]
            elseif strategy == FALLBACK_TO_CPU
                @info "Falling back to CPU implementation"
                context["use_cpu"] = true
                return f()  # Caller should check use_cpu flag
            elseif strategy == CLEAR_CACHE_AND_RETRY
                @info "Clearing GPU cache and retrying"
                CUDA.reclaim()
                GC.gc()
            elseif strategy == REDUCE_PRECISION
                old_precision = get(context, "precision", Float32)
                new_precision = old_precision == Float32 ? Float16 : Float16
                context["precision"] = new_precision
                @info "Reducing precision" from=old_precision to=new_precision
            end
            
            retry_count += 1
            if retry_count <= recovery.max_retries
                sleep(recovery.retry_delay_ms / 1000)
            end
        end
    end
    
    # If we get here, all retries failed
    @error "All recovery attempts failed" error=last_error
    throw(last_error)
end

"""
Wrapper for operations with numerical stability checks
"""
function with_numerical_checks(
    f::Function,
    checks::Vector{Symbol} = [:finite];
    logger::Union{ErrorLogger, Nothing} = nothing
)
    result = f()
    
    for check in checks
        try
            if check == :finite
                if !all(isfinite.(result))
                    invalid_count = sum(.!isfinite.(result))
                    throw(NumericalError(
                        "Non-finite values detected",
                        "numerical_check",
                        Dict(
                            "invalid_count" => invalid_count,
                            "total_elements" => length(result)
                        )
                    ))
                end
            elseif check == :positive
                if !all(result .> 0)
                    negative_count = sum(result .<= 0)
                    throw(NumericalError(
                        "Non-positive values detected",
                        "numerical_check",
                        Dict(
                            "negative_count" => negative_count,
                            "min_value" => minimum(result)
                        )
                    ))
                end
            elseif check == :variance
                if any(result .< -1e-6)
                    throw(NumericalError(
                        "Negative variance detected",
                        "numerical_check",
                        Dict(
                            "min_variance" => minimum(result),
                            "negative_indices" => findall(result .< 0)
                        )
                    ))
                end
            end
        catch e
            if !isnothing(logger) && isa(e, NumericalError)
                log_error!(logger, e)
            end
            rethrow()
        end
    end
    
    return result
end

"""
CPU fallback implementation for variance calculation
"""
function cpu_fallback_variance(X::Array{Float32, 2})
    n_features, n_samples = size(X)
    variances = zeros(Float32, n_features)
    
    for i in 1:n_features
        feature_data = @view X[i, :]
        variances[i] = var(feature_data, corrected=true)
    end
    
    return variances
end

"""
CPU fallback implementation for correlation matrix
"""
function cpu_fallback_correlation(X::Array{Float32, 2})
    return cor(X')
end

"""
CPU fallback implementation for mutual information
"""
function cpu_fallback_mutual_information(
    X::Array{Float32, 2},
    y::Array{Int32, 1};
    n_bins::Int = 10
)
    n_features = size(X, 1)
    mi_scores = zeros(Float32, n_features)
    
    # Simple binning-based MI calculation
    for i in 1:n_features
        feature = @view X[i, :]
        
        # Compute histograms
        min_val, max_val = extrema(feature)
        bin_width = (max_val - min_val) / n_bins
        
        # Joint histogram
        joint_hist = zeros(Int, n_bins, maximum(y))
        marginal_x = zeros(Int, n_bins)
        marginal_y = zeros(Int, maximum(y))
        
        for (idx, (x_val, y_val)) in enumerate(zip(feature, y))
            bin = min(floor(Int, (x_val - min_val) / bin_width) + 1, n_bins)
            joint_hist[bin, y_val] += 1
            marginal_x[bin] += 1
            marginal_y[y_val] += 1
        end
        
        # Compute MI
        n_total = length(feature)
        mi = 0.0
        
        for xi in 1:n_bins
            for yi in 1:maximum(y)
                if joint_hist[xi, yi] > 0
                    p_xy = joint_hist[xi, yi] / n_total
                    p_x = marginal_x[xi] / n_total
                    p_y = marginal_y[yi] / n_total
                    
                    if p_x > 0 && p_y > 0
                        mi += p_xy * log(p_xy / (p_x * p_y))
                    end
                end
            end
        end
        
        mi_scores[i] = Float32(mi)
    end
    
    return mi_scores
end

end # module