#!/usr/bin/env julia

# Fault Tolerance Demo
# Demonstrates GPU health monitoring and failure recovery

using CUDA
using Printf
using Random
using Dates

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

# Load GPU modules
include("../src/gpu/GPU.jl")
using .GPU

# Import from GPU module
using .GPU.FaultTolerance

"""
Simulate GPU workload with potential failures
"""
function simulate_gpu_work(gpu_id::Int, work_items::Vector{Int}; fail_probability::Float64 = 0.1)
    if !CUDA.functional()
        println("  Simulating work on CPU (CUDA not available)")
        return sum(work_items) * rand()
    end
    
    prev_device = CUDA.device()
    try
        CUDA.device!(gpu_id)
        
        # Simulate failure
        if rand() < fail_probability
            error("Simulated GPU failure")
        end
        
        # Simulate memory pressure
        if rand() < 0.05
            throw(CUDA.CuError(CUDA.ERROR_OUT_OF_MEMORY))
        end
        
        # Do some actual work
        data = CUDA.rand(Float32, 1000, length(work_items))
        result = sum(data)
        
        return result
    finally
        CUDA.device!(prev_device)
    end
end

"""
Demo fault tolerance system
"""
function demo_fault_tolerance()
    println("GPU Fault Tolerance Demo")
    println("=" ^ 60)
    
    # Create health monitor
    num_gpus = CUDA.functional() ? min(2, length(CUDA.devices())) : 2
    monitor = create_health_monitor(
        num_gpus = num_gpus,
        heartbeat_interval = 0.5,
        heartbeat_timeout = 2.0,
        error_threshold = 5
    )
    # Set additional limits
    monitor.consecutive_error_limit = 3
    
    println("\nHealth Monitor Configuration:")
    println("  Number of GPUs: $num_gpus")
    println("  Heartbeat interval: $(monitor.heartbeat_interval)s")
    println("  Error threshold: $(monitor.error_threshold)")
    println("  Consecutive error limit: $(monitor.consecutive_error_limit)")
    
    # Create checkpoint manager
    checkpoint_manager = create_checkpoint_manager(
        checkpoint_dir = joinpath(@__DIR__, ".demo_checkpoints"),
        max_checkpoints = 3
    )
    
    # Demo 1: Basic Health Monitoring
    println("\n1. Basic Health Monitoring:")
    println("-" ^ 40)
    
    # Start monitoring
    start_monitoring!(monitor)
    println("✓ Health monitoring started")
    
    # Check initial status
    for gpu_id in 0:(num_gpus-1)
        status = get_gpu_status(monitor, gpu_id)
        healthy = is_gpu_healthy(monitor, gpu_id)
        println("  GPU $gpu_id: Status=$status, Healthy=$healthy")
    end
    
    # Demo 2: Error Detection and Tracking
    println("\n2. Error Detection and Tracking:")
    println("-" ^ 40)
    
    # Register error callback
    error_callback = function(gpu_id, status, mode)
        println("  ⚠ Error callback: GPU $gpu_id status=$status mode=$mode")
    end
    register_error_callback!(monitor, error_callback)
    
    # Simulate some errors
    println("\nSimulating GPU errors...")
    for i in 1:3
        handle_gpu_error(monitor, 0, CUDA.CuError(CUDA.ERROR_INVALID_VALUE))
        println("  Error $i recorded")
        sleep(0.1)
    end
    
    # Check metrics
    metrics = monitor.metrics[0]
    println("\nGPU 0 Metrics:")
    println("  Total errors: $(metrics.error_count[])")
    println("  Consecutive errors: $(metrics.consecutive_errors[])")
    println("  Memory errors: $(metrics.memory_errors[])")
    
    # Demo 3: Workload Distribution
    println("\n3. Workload Distribution:")
    println("-" ^ 40)
    
    # Create work items (tree IDs)
    total_trees = 100
    work_distribution = Dict{Int, Vector{Int}}()
    
    if num_gpus >= 2
        # Initial distribution: 50-50 split
        work_distribution[0] = collect(1:50)
        work_distribution[1] = collect(51:100)
        
        println("Initial work distribution:")
        for (gpu_id, trees) in work_distribution
            println("  GPU $gpu_id: Trees $(first(trees))-$(last(trees)) ($(length(trees)) trees)")
        end
    else
        # Single GPU gets all work
        work_distribution[0] = collect(1:100)
        println("Single GPU mode: All 100 trees on GPU 0")
    end
    
    # Demo 4: Failure and Recovery
    println("\n4. Failure Simulation and Recovery:")
    println("-" ^ 40)
    
    # Simulate work with potential failures
    iteration = 0
    max_iterations = 10
    
    while iteration < max_iterations
        iteration += 1
        println("\nIteration $iteration:")
        
        # Save checkpoint periodically
        if iteration % 3 == 0
            for (gpu_id, work) in work_distribution
                save_checkpoint(
                    checkpoint_manager,
                    gpu_id,
                    iteration,
                    work,
                    Dict("iteration" => iteration, "status" => "running")
                )
                println("  ✓ Checkpoint saved for GPU $gpu_id")
            end
        end
        
        # Process work on each GPU
        for (gpu_id, work_items) in work_distribution
            if !is_gpu_healthy(monitor, gpu_id)
                println("  ⚠ GPU $gpu_id unhealthy - skipping")
                continue
            end
            
            try
                # Simulate work with 20% failure chance in demo
                result = simulate_gpu_work(gpu_id, work_items, fail_probability = 0.2)
                println("  ✓ GPU $gpu_id completed work: result=$(@sprintf("%.2f", result))")
                
            catch e
                println("  ✗ GPU $gpu_id failed: $e")
                handle_gpu_error(monitor, gpu_id, e)
                
                # Check if GPU needs work redistribution
                if !is_gpu_healthy(monitor, gpu_id)
                    println("\n  Redistributing work from failed GPU $gpu_id...")
                    
                    new_distribution = redistribute_work!(monitor, gpu_id, work_items)
                    
                    if !isnothing(new_distribution)
                        # Remove failed GPU from distribution
                        delete!(work_distribution, gpu_id)
                        
                        # Add redistributed work
                        for (target_gpu, new_work) in new_distribution
                            if haskey(work_distribution, target_gpu)
                                append!(work_distribution[target_gpu], new_work)
                            else
                                work_distribution[target_gpu] = new_work
                            end
                        end
                        
                        println("  Work redistributed:")
                        for (target_gpu, trees) in new_distribution
                            println("    GPU $target_gpu gets $(length(trees)) additional trees")
                        end
                    else
                        println("  ⚠ No healthy GPUs for redistribution!")
                        
                        # Try graceful degradation
                        if enable_graceful_degradation!(monitor)
                            println("  ✓ Enabled graceful degradation to single GPU")
                        else
                            println("  ✗ Complete system failure - no GPUs available")
                            break
                        end
                    end
                end
            end
        end
        
        sleep(0.5)  # Brief pause between iterations
    end
    
    # Demo 5: Recovery from Checkpoint
    println("\n5. Recovery from Checkpoint:")
    println("-" ^ 40)
    
    # Simulate recovery scenario
    println("Simulating system restart...")
    
    # Restore latest checkpoint for each GPU
    for gpu_id in 0:(num_gpus-1)
        checkpoint = restore_checkpoint(checkpoint_manager, gpu_id)
        if !isnothing(checkpoint)
            println("  ✓ Restored GPU $gpu_id from iteration $(checkpoint.iteration)")
            println("    Work items: $(length(checkpoint.work_items)) trees")
            println("    Timestamp: $(checkpoint.timestamp)")
        else
            println("  ⚠ No checkpoint available for GPU $gpu_id")
        end
    end
    
    # Demo 6: Statistics and Monitoring
    println("\n6. Failure Statistics:")
    println("-" ^ 40)
    
    stats = get_failure_statistics(monitor)
    
    println("Overall Statistics:")
    println("  Total failures: $(stats["total_failures"])")
    println("  Total recoveries: $(stats["total_recoveries"])")
    println("  Monitoring active: $(stats["monitoring_active"])")
    
    println("\nPer-GPU Statistics:")
    for (gpu_id, gpu_stats) in stats["gpu_stats"]
        println("  GPU $gpu_id:")
        println("    Status: $(gpu_stats["status"])")
        println("    Failure mode: $(gpu_stats["failure_mode"])")
        println("    Error count: $(gpu_stats["error_count"])")
        println("    Memory errors: $(gpu_stats["memory_errors"])")
    end
    
    # Stop monitoring
    stop_monitoring!(monitor)
    println("\n✓ Health monitoring stopped")
    
    # Demo 7: Best Practices
    println("\n7. Fault Tolerance Best Practices:")
    println("-" ^ 40)
    println("• Set appropriate heartbeat intervals based on workload")
    println("• Configure error thresholds to avoid false positives")
    println("• Implement regular checkpointing for critical state")
    println("• Design work units for easy redistribution")
    println("• Monitor GPU temperature and power limits")
    println("• Test failure scenarios in development")
    println("• Have clear escalation procedures for production")
    
    # Cleanup
    checkpoint_dir = joinpath(@__DIR__, ".demo_checkpoints")
    if isdir(checkpoint_dir)
        rm(checkpoint_dir, recursive=true)
    end
    
    println("\n" * "=" * 60)
    println("Demo completed!")
end

# Custom error callback example
function custom_error_handler(gpu_id::Int, status::GPUStatus, mode::FailureMode)
    timestamp = now()
    
    # Log to file in production
    msg = "[$timestamp] GPU $gpu_id failure: status=$status, mode=$mode"
    
    # Take action based on failure mode
    if mode == MEMORY_ERROR
        println("  → Action: Clearing GPU memory caches")
    elseif mode == THERMAL_THROTTLE
        println("  → Action: Reducing workload")
    elseif mode == HEARTBEAT_FAILURE
        println("  → Action: Attempting GPU reset")
    end
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    demo_fault_tolerance()
end