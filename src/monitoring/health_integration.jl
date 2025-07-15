module HealthIntegration

using ..Health
using ..Database
using Redis
using CUDA
using Logging

export initialize_health_monitoring, check_system_dependencies, start_health_services

"""
Initialize health monitoring system with actual component checks
"""
function initialize_health_monitoring()
    @info "Initializing health monitoring system"
    
    # Check and update initial system state
    check_system_dependencies()
    
    # Set up periodic health checks
    health_check_task = @async begin
        while true
            try
                update_system_health()
                sleep(30)  # Update every 30 seconds
            catch e
                @error "Health check update failed" exception=e
            end
        end
    end
    
    return health_check_task
end

"""
Check system dependencies and update health status
"""
function check_system_dependencies()
    # Check GPU availability
    gpu_available = CUDA.functional()
    gpu_count = gpu_available ? length(CUDA.devices()) : 0
    
    # Check database connection
    db_connected = try
        db = Database.get_connection()
        Database.execute(db, "SELECT 1")
        true
    catch
        false
    end
    
    # Check Redis connection
    redis_connected = try
        redis_conn = Redis.RedisConnection(
            host=get(ENV, "REDIS_HOST", "localhost"),
            port=parse(Int, get(ENV, "REDIS_PORT", "6379"))
        )
        Redis.ping(redis_conn)
        Redis.disconnect(redis_conn)
        true
    catch
        false
    end
    
    # Check filesystem access
    fs_accessible = try
        test_dir = joinpath("data", ".health_check")
        mkpath(test_dir)
        test_file = joinpath(test_dir, "test.txt")
        write(test_file, "health check")
        content = read(test_file, String)
        rm(test_dir, recursive=true)
        content == "health check"
    catch
        false
    end
    
    # Update health states
    Health.update_pipeline_health(
        database=db_connected,
        redis=redis_connected,
        filesystem=fs_accessible
    )
    
    @info "System dependency check complete" gpu_available gpu_count db_connected redis_connected fs_accessible
    
    return (gpu_available, db_connected, redis_connected, fs_accessible)
end

"""
Update system health based on actual component states
"""
function update_system_health()
    # This would be called by actual pipeline components
    # For now, we'll simulate some checks
    
    # Update based on actual system state
    gpu_available, db_connected, redis_connected, fs_accessible = check_system_dependencies()
    
    # In a real implementation, these would be set by the actual stages
    # For demonstration, we'll check if the stage modules are loaded
    stage1_operational = isdefined(Main, :Stage1Filter)
    stage2_operational = isdefined(Main, :GPUMCTS) 
    stage3_operational = isdefined(Main, :Stage3Evaluation)
    
    Health.update_pipeline_health(
        stage1=stage1_operational,
        stage2=stage2_operational,
        stage3=stage3_operational,
        database=db_connected,
        redis=redis_connected,
        filesystem=fs_accessible
    )
end

"""
Start health monitoring services
"""
function start_health_services(; port=8080, host="0.0.0.0")
    @info "Starting health monitoring services on $host:$port"
    
    # Initialize monitoring
    health_task = initialize_health_monitoring()
    
    # Start HTTP server
    server_task = @async begin
        try
            Health.start_health_server(host=host, port=port)
        catch e
            @error "Failed to start health server" exception=e
            rethrow(e)
        end
    end
    
    return (health_task, server_task)
end

"""
Integration with main HSOF module for health updates
"""
module Hooks

using ...Health
using Dates

# Hook functions to be called by pipeline stages

export on_stage_start, on_stage_complete, on_stage_error
export on_model_inference, on_model_error, on_model_loaded

function on_stage_start(stage_number::Int)
    @debug "Stage $stage_number started"
    # Could add more detailed tracking here
end

function on_stage_complete(stage_number::Int, duration_seconds::Float64)
    @debug "Stage $stage_number completed" duration=duration_seconds
    
    # Update operational status
    if stage_number == 1
        Health.update_pipeline_health(stage1=true)
    elseif stage_number == 2
        Health.update_pipeline_health(stage2=true)
    elseif stage_number == 3
        Health.update_pipeline_health(stage3=true)
    end
end

function on_stage_error(stage_number::Int, error::Exception)
    @error "Stage $stage_number error" exception=error
    
    # Update operational status
    if stage_number == 1
        Health.update_pipeline_health(stage1=false)
    elseif stage_number == 2
        Health.update_pipeline_health(stage2=false)
    elseif stage_number == 3
        Health.update_pipeline_health(stage3=false)
    end
end

function on_model_inference(inference_time_ms::Float64)
    Health.update_model_health(
        inference_ready=true,
        inference_time_ms=inference_time_ms,
        error_occurred=false
    )
end

function on_model_error(error::Exception)
    @error "Model inference error" exception=error
    Health.update_model_health(error_occurred=true)
end

function on_model_loaded(version::String)
    @info "Model loaded" version=version
    Health.update_model_health(
        metamodel_loaded=true,
        inference_ready=true
    )
    Health.MODEL_HEALTH.metamodel_version = version
end

end # module Hooks

end # module HealthIntegration