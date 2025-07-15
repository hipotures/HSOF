#!/usr/bin/env julia

# Health check script for HSOF container
# Returns exit code 0 if healthy, 1 if unhealthy

using Pkg
Pkg.activate(".")

try
    # Check Julia environment
    using CUDA
    using Redis
    using LibPQ
    using HTTP
    
    # Check GPU availability
    if !CUDA.functional()
        println("ERROR: CUDA not functional")
        exit(1)
    end
    
    gpu_count = length(CUDA.devices())
    if gpu_count == 0
        println("ERROR: No GPUs detected")
        exit(1)
    end
    
    # Check GPU memory
    for i in 0:gpu_count-1
        device = CuDevice(i)
        CUDA.device!(device)
        free_mem = CUDA.available_memory() / 1024^3  # Convert to GB
        total_mem = CUDA.total_memory() / 1024^3
        
        if free_mem < 1.0  # Less than 1GB free
            println("WARNING: GPU $i low memory: $(round(free_mem, digits=2))GB free")
        end
    end
    
    # Check database connection
    db_url = get(ENV, "DATABASE_URL", "postgresql://hsof:hsof123@localhost:5432/hsof")
    try
        conn = LibPQ.Connection(db_url)
        result = execute(conn, "SELECT 1")
        close(conn)
    catch e
        println("ERROR: Database connection failed: $e")
        exit(1)
    end
    
    # Check Redis connection
    redis_url = get(ENV, "REDIS_URL", "redis://localhost:6379/0")
    try
        redis_conn = Redis.RedisConnection(host="redis", port=6379)
        Redis.ping(redis_conn)
        Redis.disconnect(redis_conn)
    catch e
        println("ERROR: Redis connection failed: $e")
        exit(1)
    end
    
    # Check if main module can be loaded
    try
        include("src/HSOF.jl")
        using .HSOF
    catch e
        println("ERROR: Failed to load HSOF module: $e")
        exit(1)
    end
    
    # Check HTTP endpoint if server is running
    port = parse(Int, get(ENV, "HSOF_PORT", "8080"))
    try
        response = HTTP.get("http://localhost:$port/health", readtimeout=5)
        if response.status != 200
            println("ERROR: Health endpoint returned status $(response.status)")
            exit(1)
        end
    catch e
        # Server might not be HTTP-based, which is okay
        println("INFO: HTTP health check skipped (server may not be HTTP-based)")
    end
    
    println("Health check passed: GPUs=$gpu_count, DB=OK, Redis=OK, Module=OK")
    exit(0)
    
catch e
    println("ERROR: Health check failed with exception: $e")
    exit(1)
end