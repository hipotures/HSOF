#!/usr/bin/env julia

# Health check script for HSOF container
# Returns exit code 0 if healthy, 1 if unhealthy

using Pkg
Pkg.activate(".")

# Load health monitoring modules
include("../src/monitoring/health.jl")
include("../src/monitoring/health_integration.jl")

using .Health
using .HealthIntegration
using JSON3

try
    # Perform comprehensive system check
    gpu_available, db_connected, redis_connected, fs_accessible = HealthIntegration.check_system_dependencies()
    
    # Get aggregated health status
    overall_health = Health.aggregate_health()
    
    # Check if we can reach the health endpoint (if server is running)
    port = parse(Int, get(ENV, "HSOF_PORT", "8080"))
    server_health_ok = try
        using HTTP
        response = HTTP.get("http://localhost:$port/health", readtimeout=5)
        response.status == 200
    catch
        false  # Server might not be running, which is okay for container health check
    end
    
    # Determine exit code based on health status
    if overall_health.status == Health.CRITICAL
        println("ERROR: System health critical - $(overall_health.message)")
        
        # Print detailed component status
        components = overall_health.details["components"]
        for (name, component) in components
            if component["status"] != "HEALTHY"
                println("  $name: $(component["message"])")
            end
        end
        
        exit(1)
    elseif overall_health.status == Health.WARNING
        println("WARNING: System health degraded - $(overall_health.message)")
        # Warnings don't fail the health check
    end
    
    # Print summary
    println("Health check passed:")
    println("  GPU: $(gpu_available ? "Available" : "Not Available") ($(gpu_available ? length(CUDA.devices()) : 0) devices)")
    println("  Database: $(db_connected ? "Connected" : "Disconnected")")
    println("  Redis: $(redis_connected ? "Connected" : "Disconnected")")
    println("  Filesystem: $(fs_accessible ? "Accessible" : "Not Accessible")")
    println("  Health Server: $(server_health_ok ? "Running" : "Not Running")")
    println("  Overall Status: $(overall_health.status)")
    
    exit(0)
    
catch e
    println("ERROR: Health check failed with exception: $e")
    exit(1)
end