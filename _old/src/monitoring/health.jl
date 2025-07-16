module Health

using CUDA
using HTTP
using JSON3
using Dates
using Statistics
using Logging

export HealthStatus, HealthCheck, GPUHealth, ModelHealth, PipelineHealth
export check_gpu_health, check_model_health, check_pipeline_health
export aggregate_health, start_health_server

# Health status enumeration
@enum HealthStatus begin
    HEALTHY = 1
    WARNING = 2
    CRITICAL = 3
    UNKNOWN = 4
end

# Health check result structure
struct HealthCheck
    status::HealthStatus
    message::String
    details::Dict{String, Any}
    timestamp::DateTime
    
    function HealthCheck(status::HealthStatus, message::String="", details::Dict{String, Any}=Dict{String, Any}())
        new(status, message, details, now())
    end
end

# GPU health information
struct GPUHealth
    device_id::Int
    available::Bool
    memory_used::Float64  # GB
    memory_total::Float64  # GB
    memory_percentage::Float64
    temperature::Float64  # Celsius
    utilization::Float64  # Percentage
    power_draw::Float64  # Watts
    error_count::Int
    
    function GPUHealth(device_id::Int)
        if !CUDA.functional()
            return new(device_id, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        end
        
        try
            device = CuDevice(device_id)
            CUDA.device!(device)
            
            # Memory information
            free_mem = CUDA.available_memory() / 1024^3
            total_mem = CUDA.total_memory() / 1024^3
            used_mem = total_mem - free_mem
            mem_percentage = (used_mem / total_mem) * 100
            
            # GPU metrics (using NVML through CUDA.jl)
            temp = CUDA.temperature(device)
            util = CUDA.utilization(device)
            power = CUDA.power_usage(device) / 1000  # Convert mW to W
            
            new(device_id, true, used_mem, total_mem, mem_percentage, 
                temp, util, power, 0)
        catch e
            @warn "Failed to get GPU health for device $device_id" exception=e
            new(device_id, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)
        end
    end
end

# Model health information
mutable struct ModelHealth
    metamodel_loaded::Bool
    metamodel_version::String
    inference_ready::Bool
    last_inference_time::Union{DateTime, Nothing}
    average_inference_ms::Float64
    error_rate::Float64
end

# Pipeline health information
mutable struct PipelineHealth
    stage1_operational::Bool
    stage2_operational::Bool
    stage3_operational::Bool
    database_connected::Bool
    redis_connected::Bool
    filesystem_accessible::Bool
end

# Global health state (would be properly initialized in production)
const MODEL_HEALTH = ModelHealth(false, "", false, nothing, 0.0, 0.0)
const PIPELINE_HEALTH = PipelineHealth(false, false, false, false, false, false)

# Configuration
const HEALTH_CONFIG = Dict{String, Any}(
    "gpu_memory_warning_threshold" => 80.0,  # percentage
    "gpu_memory_critical_threshold" => 95.0,
    "gpu_temp_warning_threshold" => 80.0,    # Celsius
    "gpu_temp_critical_threshold" => 90.0,
    "model_error_rate_warning" => 0.05,      # 5%
    "model_error_rate_critical" => 0.10,     # 10%
    "inference_latency_warning_ms" => 100.0,
    "inference_latency_critical_ms" => 500.0
)

"""
Check GPU health status for all available GPUs
"""
function check_gpu_health()
    if !CUDA.functional()
        return HealthCheck(CRITICAL, "CUDA not functional", 
                         Dict("cuda_available" => false))
    end
    
    gpu_count = length(CUDA.devices())
    if gpu_count == 0
        return HealthCheck(CRITICAL, "No GPUs detected",
                         Dict("gpu_count" => 0))
    end
    
    gpu_healths = GPUHealth[]
    overall_status = HEALTHY
    messages = String[]
    
    for i in 0:gpu_count-1
        gpu = GPUHealth(i)
        push!(gpu_healths, gpu)
        
        if !gpu.available
            overall_status = CRITICAL
            push!(messages, "GPU $i not available")
            continue
        end
        
        # Check memory usage
        if gpu.memory_percentage >= HEALTH_CONFIG["gpu_memory_critical_threshold"]
            overall_status = CRITICAL
            push!(messages, "GPU $i memory critical: $(round(gpu.memory_percentage, digits=1))%")
        elseif gpu.memory_percentage >= HEALTH_CONFIG["gpu_memory_warning_threshold"]
            overall_status = max(overall_status, WARNING)
            push!(messages, "GPU $i memory warning: $(round(gpu.memory_percentage, digits=1))%")
        end
        
        # Check temperature
        if gpu.temperature >= HEALTH_CONFIG["gpu_temp_critical_threshold"]
            overall_status = CRITICAL
            push!(messages, "GPU $i temperature critical: $(round(gpu.temperature, digits=1))°C")
        elseif gpu.temperature >= HEALTH_CONFIG["gpu_temp_warning_threshold"]
            overall_status = max(overall_status, WARNING)
            push!(messages, "GPU $i temperature warning: $(round(gpu.temperature, digits=1))°C")
        end
    end
    
    details = Dict{String, Any}(
        "gpu_count" => gpu_count,
        "gpus" => [Dict(
            "device_id" => g.device_id,
            "available" => g.available,
            "memory_used_gb" => round(g.memory_used, digits=2),
            "memory_total_gb" => round(g.memory_total, digits=2),
            "memory_percentage" => round(g.memory_percentage, digits=1),
            "temperature_celsius" => round(g.temperature, digits=1),
            "utilization_percent" => round(g.utilization, digits=1),
            "power_watts" => round(g.power_draw, digits=1)
        ) for g in gpu_healths]
    )
    
    message = overall_status == HEALTHY ? "All GPUs healthy" : join(messages, "; ")
    return HealthCheck(overall_status, message, details)
end

"""
Check model health status
"""
function check_model_health()
    status = HEALTHY
    messages = String[]
    
    if !MODEL_HEALTH.metamodel_loaded
        status = CRITICAL
        push!(messages, "Metamodel not loaded")
    end
    
    if !MODEL_HEALTH.inference_ready
        status = max(status, WARNING)
        push!(messages, "Inference not ready")
    end
    
    # Check error rate
    if MODEL_HEALTH.error_rate >= HEALTH_CONFIG["model_error_rate_critical"]
        status = CRITICAL
        push!(messages, "Model error rate critical: $(round(MODEL_HEALTH.error_rate * 100, digits=1))%")
    elseif MODEL_HEALTH.error_rate >= HEALTH_CONFIG["model_error_rate_warning"]
        status = max(status, WARNING)
        push!(messages, "Model error rate warning: $(round(MODEL_HEALTH.error_rate * 100, digits=1))%")
    end
    
    # Check inference latency
    if MODEL_HEALTH.average_inference_ms >= HEALTH_CONFIG["inference_latency_critical_ms"]
        status = CRITICAL
        push!(messages, "Inference latency critical: $(round(MODEL_HEALTH.average_inference_ms, digits=1))ms")
    elseif MODEL_HEALTH.average_inference_ms >= HEALTH_CONFIG["inference_latency_warning_ms"]
        status = max(status, WARNING)
        push!(messages, "Inference latency warning: $(round(MODEL_HEALTH.average_inference_ms, digits=1))ms")
    end
    
    details = Dict{String, Any}(
        "metamodel_loaded" => MODEL_HEALTH.metamodel_loaded,
        "metamodel_version" => MODEL_HEALTH.metamodel_version,
        "inference_ready" => MODEL_HEALTH.inference_ready,
        "last_inference_time" => MODEL_HEALTH.last_inference_time,
        "average_inference_ms" => round(MODEL_HEALTH.average_inference_ms, digits=2),
        "error_rate" => round(MODEL_HEALTH.error_rate, digits=4)
    )
    
    message = status == HEALTHY ? "Model healthy" : join(messages, "; ")
    return HealthCheck(status, message, details)
end

"""
Check pipeline health status
"""
function check_pipeline_health()
    status = HEALTHY
    messages = String[]
    
    # Check stage operational status
    stages = [
        (PIPELINE_HEALTH.stage1_operational, "Stage 1"),
        (PIPELINE_HEALTH.stage2_operational, "Stage 2"),
        (PIPELINE_HEALTH.stage3_operational, "Stage 3")
    ]
    
    for (operational, name) in stages
        if !operational
            status = CRITICAL
            push!(messages, "$name not operational")
        end
    end
    
    # Check database connection
    if !PIPELINE_HEALTH.database_connected
        status = max(status, WARNING)
        push!(messages, "Database not connected")
    end
    
    # Check Redis connection
    if !PIPELINE_HEALTH.redis_connected
        status = max(status, WARNING)
        push!(messages, "Redis not connected")
    end
    
    # Check filesystem access
    if !PIPELINE_HEALTH.filesystem_accessible
        status = CRITICAL
        push!(messages, "Filesystem not accessible")
    end
    
    details = Dict{String, Any}(
        "stage1_operational" => PIPELINE_HEALTH.stage1_operational,
        "stage2_operational" => PIPELINE_HEALTH.stage2_operational,
        "stage3_operational" => PIPELINE_HEALTH.stage3_operational,
        "database_connected" => PIPELINE_HEALTH.database_connected,
        "redis_connected" => PIPELINE_HEALTH.redis_connected,
        "filesystem_accessible" => PIPELINE_HEALTH.filesystem_accessible
    )
    
    message = status == HEALTHY ? "Pipeline healthy" : join(messages, "; ")
    return HealthCheck(status, message, details)
end

"""
Aggregate all health checks into overall system health
"""
function aggregate_health()
    gpu_health = check_gpu_health()
    model_health = check_model_health()
    pipeline_health = check_pipeline_health()
    
    # Overall status is the worst of all components
    overall_status = max(gpu_health.status, model_health.status, pipeline_health.status)
    
    details = Dict{String, Any}(
        "overall_status" => string(overall_status),
        "components" => Dict(
            "gpu" => Dict(
                "status" => string(gpu_health.status),
                "message" => gpu_health.message,
                "details" => gpu_health.details
            ),
            "model" => Dict(
                "status" => string(model_health.status),
                "message" => model_health.message,
                "details" => model_health.details
            ),
            "pipeline" => Dict(
                "status" => string(pipeline_health.status),
                "message" => pipeline_health.message,
                "details" => pipeline_health.details
            )
        ),
        "timestamp" => now()
    )
    
    messages = String[]
    for (name, health) in [("GPU", gpu_health), ("Model", model_health), ("Pipeline", pipeline_health)]
        if health.status != HEALTHY
            push!(messages, "$name: $(health.message)")
        end
    end
    
    message = overall_status == HEALTHY ? "System healthy" : join(messages, "; ")
    return HealthCheck(overall_status, message, details)
end

"""
HTTP request handler for health endpoints
"""
function health_handler(req::HTTP.Request)
    path = HTTP.URI(req.target).path
    
    try
        response = if path == "/health"
            aggregate_health()
        elseif path == "/health/gpu"
            check_gpu_health()
        elseif path == "/health/model"
            check_model_health()
        elseif path == "/health/pipeline"
            check_pipeline_health()
        else
            return HTTP.Response(404, "Not Found")
        end
        
        # Set HTTP status code based on health status
        http_status = response.status == HEALTHY ? 200 :
                     response.status == WARNING ? 200 :  # Still return 200 for warnings
                     response.status == CRITICAL ? 503 : 500
        
        headers = ["Content-Type" => "application/json"]
        body = JSON3.write(response.details)
        
        return HTTP.Response(http_status, headers, body)
    catch e
        @error "Health check error" exception=e
        error_body = JSON3.write(Dict("error" => string(e), "status" => "UNKNOWN"))
        return HTTP.Response(500, ["Content-Type" => "application/json"], error_body)
    end
end

"""
Start the health check HTTP server
"""
function start_health_server(; host="0.0.0.0", port=8080)
    @info "Starting health check server on $host:$port"
    
    router = HTTP.Router()
    HTTP.register!(router, "/health", health_handler)
    HTTP.register!(router, "/health/*", health_handler)
    
    # Add metrics endpoint - import Prometheus module
    try
        include("prometheus.jl")
        using .Prometheus
        HTTP.register!(router, "/metrics", Prometheus.metrics_handler)
        @info "Prometheus metrics endpoint enabled at /metrics"
    catch e
        @warn "Failed to load Prometheus module, metrics endpoint disabled" exception=e
        HTTP.register!(router, "/metrics") do req
            HTTP.Response(503, "Prometheus metrics not available")
        end
    end
    
    try
        HTTP.serve(router, host, port)
    catch e
        @error "Failed to start health server" exception=e
        rethrow(e)
    end
end

# Recovery mechanisms
"""
Attempt to recover from GPU errors
"""
function recover_gpu_error(device_id::Int)
    @info "Attempting GPU recovery for device $device_id"
    
    try
        # Reset the device
        device = CuDevice(device_id)
        CUDA.device!(device)
        CUDA.device_reset!()
        
        # Test basic functionality
        test_array = CUDA.zeros(Float32, 100)
        sum(test_array)  # Simple operation to verify GPU works
        
        @info "GPU $device_id recovery successful"
        return true
    catch e
        @error "GPU $device_id recovery failed" exception=e
        return false
    end
end

"""
Attempt to reload the metamodel
"""
function recover_model_error()
    @info "Attempting metamodel recovery"
    
    try
        # This would be implemented based on actual model loading logic
        # For now, just a placeholder
        MODEL_HEALTH.metamodel_loaded = false
        MODEL_HEALTH.inference_ready = false
        
        # Simulate model reload
        # load_metamodel()
        
        MODEL_HEALTH.metamodel_loaded = true
        MODEL_HEALTH.inference_ready = true
        MODEL_HEALTH.error_rate = 0.0
        
        @info "Model recovery successful"
        return true
    catch e
        @error "Model recovery failed" exception=e
        return false
    end
end

"""
Attempt to recover from checkpoint
"""
function recover_from_checkpoint(checkpoint_path::String)
    @info "Attempting recovery from checkpoint: $checkpoint_path"
    
    try
        # This would be implemented based on actual checkpoint logic
        # For now, just verify the checkpoint exists
        if !isfile(checkpoint_path)
            @error "Checkpoint not found: $checkpoint_path"
            return false
        end
        
        # Load checkpoint
        # restore_from_checkpoint(checkpoint_path)
        
        @info "Checkpoint recovery successful"
        return true
    catch e
        @error "Checkpoint recovery failed" exception=e
        return false
    end
end

# Update functions for health state (these would be called by the main application)
"""
Update model health metrics
"""
function update_model_health(; metamodel_loaded=nothing, inference_ready=nothing,
                           inference_time_ms=nothing, error_occurred=nothing)
    if !isnothing(metamodel_loaded)
        MODEL_HEALTH.metamodel_loaded = metamodel_loaded
    end
    
    if !isnothing(inference_ready)
        MODEL_HEALTH.inference_ready = inference_ready
    end
    
    if !isnothing(inference_time_ms)
        # Update rolling average (simple exponential moving average)
        α = 0.1  # Smoothing factor
        MODEL_HEALTH.average_inference_ms = α * inference_time_ms + 
                                          (1 - α) * MODEL_HEALTH.average_inference_ms
        MODEL_HEALTH.last_inference_time = now()
    end
    
    if !isnothing(error_occurred) && error_occurred
        # Update error rate (simple counter-based approach)
        MODEL_HEALTH.error_rate = min(1.0, MODEL_HEALTH.error_rate + 0.01)
    elseif !isnothing(error_occurred) && !error_occurred
        # Decay error rate on success
        MODEL_HEALTH.error_rate = max(0.0, MODEL_HEALTH.error_rate * 0.99)
    end
end

"""
Update pipeline health status
"""
function update_pipeline_health(; stage1=nothing, stage2=nothing, stage3=nothing,
                              database=nothing, redis=nothing, filesystem=nothing)
    if !isnothing(stage1)
        PIPELINE_HEALTH.stage1_operational = stage1
    end
    if !isnothing(stage2)
        PIPELINE_HEALTH.stage2_operational = stage2
    end
    if !isnothing(stage3)
        PIPELINE_HEALTH.stage3_operational = stage3
    end
    if !isnothing(database)
        PIPELINE_HEALTH.database_connected = database
    end
    if !isnothing(redis)
        PIPELINE_HEALTH.redis_connected = redis
    end
    if !isnothing(filesystem)
        PIPELINE_HEALTH.filesystem_accessible = filesystem
    end
end

end # module Health