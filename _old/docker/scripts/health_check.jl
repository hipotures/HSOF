#!/usr/bin/env julia

# Health check script for HSOF container
# Validates GPU availability, CUDA functionality, and pipeline readiness

using CUDA
using Dates
using JSON

# Health check result structure
mutable struct HealthStatus
    healthy::Bool
    timestamp::DateTime
    checks::Dict{String, Dict{String, Any}}
    errors::Vector{String}
end

function HealthStatus()
    return HealthStatus(
        true,
        now(),
        Dict{String, Dict{String, Any}}(),
        String[]
    )
end

"""
Check CUDA availability and GPU functionality
"""
function check_gpu(status::HealthStatus)
    gpu_check = Dict{String, Any}(
        "name" => "GPU Availability",
        "status" => "unknown",
        "message" => "",
        "details" => Dict{String, Any}()
    )
    
    try
        if CUDA.functional()
            # Get GPU information
            num_gpus = length(CUDA.devices())
            gpu_check["details"]["gpu_count"] = num_gpus
            
            if num_gpus > 0
                # Check each GPU
                gpu_info = []
                for i in 0:num_gpus-1
                    device = CuDevice(i)
                    CUDA.device!(device)
                    
                    info = Dict(
                        "id" => i,
                        "name" => CUDA.name(device),
                        "memory_total_mb" => round(CUDA.totalmem(device) / 1024^2, digits=2),
                        "memory_free_mb" => round(CUDA.available_memory() / 1024^2, digits=2),
                        "compute_capability" => "$(CUDA.capability(device).major).$(CUDA.capability(device).minor)"
                    )
                    
                    # Test GPU computation
                    try
                        test_array = CUDA.rand(100, 100)
                        result = sum(test_array)
                        CUDA.synchronize()
                        info["computation_test"] = "passed"
                    catch e
                        info["computation_test"] = "failed: $e"
                        status.healthy = false
                    end
                    
                    push!(gpu_info, info)
                end
                
                gpu_check["details"]["gpus"] = gpu_info
                gpu_check["status"] = "healthy"
                gpu_check["message"] = "$num_gpus GPU(s) available and functional"
            else
                gpu_check["status"] = "unhealthy"
                gpu_check["message"] = "No GPUs detected"
                status.healthy = false
            end
        else
            gpu_check["status"] = "unhealthy"
            gpu_check["message"] = "CUDA not functional"
            status.healthy = false
            
            # Try to get more details about why CUDA is not functional
            try
                CUDA.version()
                gpu_check["details"]["cuda_version"] = string(CUDA.version())
            catch
                gpu_check["details"]["cuda_version"] = "unable to determine"
            end
        end
    catch e
        gpu_check["status"] = "error"
        gpu_check["message"] = "GPU check failed: $e"
        push!(status.errors, string(e))
        status.healthy = false
    end
    
    status.checks["gpu"] = gpu_check
end

"""
Check memory availability
"""
function check_memory(status::HealthStatus)
    memory_check = Dict{String, Any}(
        "name" => "Memory Availability",
        "status" => "unknown",
        "message" => "",
        "details" => Dict{String, Any}()
    )
    
    try
        # Host memory
        host_memory_mb = Base.Sys.total_memory() / 1024^2
        gc_stats = Base.gc_num()
        host_used_mb = Base.gc_live_bytes() / 1024^2
        
        memory_check["details"]["host_total_mb"] = round(host_memory_mb, digits=2)
        memory_check["details"]["host_used_mb"] = round(host_used_mb, digits=2)
        memory_check["details"]["host_free_mb"] = round(host_memory_mb - host_used_mb, digits=2)
        
        # Check if we have enough memory (at least 4GB free)
        min_free_mb = 4096
        if (host_memory_mb - host_used_mb) < min_free_mb
            memory_check["status"] = "warning"
            memory_check["message"] = "Low host memory available"
        else
            memory_check["status"] = "healthy"
            memory_check["message"] = "Sufficient memory available"
        end
        
        # GPU memory (if available)
        if CUDA.functional()
            for i in 0:length(CUDA.devices())-1
                device = CuDevice(i)
                CUDA.device!(device)
                
                gpu_total = CUDA.totalmem(device) / 1024^2
                gpu_free = CUDA.available_memory() / 1024^2
                
                memory_check["details"]["gpu$(i)_total_mb"] = round(gpu_total, digits=2)
                memory_check["details"]["gpu$(i)_free_mb"] = round(gpu_free, digits=2)
                
                # Check for low GPU memory (less than 2GB free)
                if gpu_free < 2048
                    memory_check["status"] = "warning"
                    memory_check["message"] = "Low GPU memory on device $i"
                end
            end
        end
        
    catch e
        memory_check["status"] = "error"
        memory_check["message"] = "Memory check failed: $e"
        push!(status.errors, string(e))
    end
    
    status.checks["memory"] = memory_check
end

"""
Check pipeline components
"""
function check_pipeline(status::HealthStatus)
    pipeline_check = Dict{String, Any}(
        "name" => "Pipeline Components",
        "status" => "unknown",
        "message" => "",
        "details" => Dict{String, Any}()
    )
    
    try
        # Check if required modules can be loaded
        required_modules = [
            "Models",
            "StatisticalFilter",
            "MCTSFeatureSelection",
            "EnsembleOptimizer",
            "GPU"
        ]
        
        modules_status = Dict{String, Bool}()
        
        for module_name in required_modules
            try
                # Try to include the module
                module_path = joinpath("/app/src", lowercase(module_name) * ".jl")
                if !isfile(module_path)
                    # Check in subdirectories
                    for subdir in ["core", "stage1", "stage2", "stage3", "gpu"]
                        alt_path = joinpath("/app/src", subdir, lowercase(module_name) * ".jl")
                        if isfile(alt_path)
                            module_path = alt_path
                            break
                        end
                    end
                end
                
                if isfile(module_path)
                    modules_status[module_name] = true
                else
                    modules_status[module_name] = false
                    status.healthy = false
                end
            catch e
                modules_status[module_name] = false
                push!(status.errors, "Failed to check module $module_name: $e")
            end
        end
        
        pipeline_check["details"]["modules"] = modules_status
        
        # Check data directories
        data_dirs = ["/app/data", "/app/checkpoints", "/app/logs", "/app/results"]
        dirs_status = Dict{String, Bool}()
        
        for dir in data_dirs
            dirs_status[dir] = isdir(dir) && iswritable(dir)
            if !dirs_status[dir]
                status.healthy = false
            end
        end
        
        pipeline_check["details"]["directories"] = dirs_status
        
        # Overall pipeline status
        all_modules_ok = all(values(modules_status))
        all_dirs_ok = all(values(dirs_status))
        
        if all_modules_ok && all_dirs_ok
            pipeline_check["status"] = "healthy"
            pipeline_check["message"] = "All pipeline components available"
        else
            pipeline_check["status"] = "unhealthy"
            pipeline_check["message"] = "Some pipeline components missing"
            status.healthy = false
        end
        
    catch e
        pipeline_check["status"] = "error"
        pipeline_check["message"] = "Pipeline check failed: $e"
        push!(status.errors, string(e))
        status.healthy = false
    end
    
    status.checks["pipeline"] = pipeline_check
end

"""
Check environment configuration
"""
function check_environment(status::HealthStatus)
    env_check = Dict{String, Any}(
        "name" => "Environment Configuration",
        "status" => "unknown",
        "message" => "",
        "details" => Dict{String, Any}()
    )
    
    try
        # Check Julia version
        env_check["details"]["julia_version"] = string(VERSION)
        
        # Check CUDA environment variables
        cuda_vars = ["CUDA_HOME", "CUDA_VISIBLE_DEVICES", "JULIA_CUDA_MEMORY_POOL"]
        env_vars = Dict{String, String}()
        
        for var in cuda_vars
            env_vars[var] = get(ENV, var, "not set")
        end
        
        env_check["details"]["environment_variables"] = env_vars
        
        # Check if running with proper user permissions
        env_check["details"]["user"] = ENV["USER"]
        env_check["details"]["home"] = ENV["HOME"]
        
        # Check Julia project
        if isfile("/app/Project.toml")
            env_check["details"]["project_toml"] = "found"
        else
            env_check["details"]["project_toml"] = "missing"
            status.healthy = false
        end
        
        env_check["status"] = status.healthy ? "healthy" : "unhealthy"
        env_check["message"] = "Environment configured"
        
    catch e
        env_check["status"] = "error"
        env_check["message"] = "Environment check failed: $e"
        push!(status.errors, string(e))
    end
    
    status.checks["environment"] = env_check
end

"""
Main health check function
"""
function run_health_check()
    status = HealthStatus()
    
    # Run all checks
    check_environment(status)
    check_memory(status)
    check_gpu(status)
    check_pipeline(status)
    
    # Generate summary
    summary = Dict{String, Any}(
        "healthy" => status.healthy,
        "timestamp" => status.timestamp,
        "checks_passed" => count(c -> c["status"] == "healthy", values(status.checks)),
        "checks_warning" => count(c -> c["status"] == "warning", values(status.checks)),
        "checks_failed" => count(c -> c["status"] in ["unhealthy", "error"], values(status.checks)),
        "total_checks" => length(status.checks)
    )
    
    # Create final response
    response = Dict{String, Any}(
        "status" => status.healthy ? "healthy" : "unhealthy",
        "timestamp" => string(status.timestamp),
        "summary" => summary,
        "checks" => status.checks,
        "errors" => status.errors
    )
    
    # Output JSON response
    println(JSON.json(response, 2))
    
    # Exit with appropriate code
    exit(status.healthy ? 0 : 1)
end

# Run the health check
if abspath(PROGRAM_FILE) == @__FILE__
    run_health_check()
end