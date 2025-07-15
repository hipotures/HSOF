#!/usr/bin/env julia

# Standalone health monitoring server for HSOF
# Can be run independently or as part of the main application

using Pkg
Pkg.activate(".")

# Load required modules
include("../src/monitoring/health.jl")
include("../src/monitoring/health_integration.jl")

using .Health
using .HealthIntegration
using Logging

# Configure logging
global_logger(ConsoleLogger(stdout, Logging.Info))

# Parse command line arguments
port = parse(Int, get(ENV, "HEALTH_PORT", "8080"))
host = get(ENV, "HEALTH_HOST", "0.0.0.0")

@info "Starting HSOF Health Monitoring Server" host=host port=port

# Initialize system checks
@info "Performing initial system check..."
gpu_available, db_connected, redis_connected, fs_accessible = HealthIntegration.check_system_dependencies()

# Start monitoring services
try
    health_task, server_task = HealthIntegration.start_health_services(port=port, host=host)
    
    @info "Health monitoring server started successfully"
    @info "Available endpoints:" endpoints=[
        "http://$host:$port/health",
        "http://$host:$port/health/gpu", 
        "http://$host:$port/health/model",
        "http://$host:$port/health/pipeline",
        "http://$host:$port/metrics"
    ]
    
    # Wait for server task
    wait(server_task)
catch e
    @error "Failed to start health monitoring server" exception=e
    exit(1)
end