using Test
using CUDA
using Dates
using Random

# Include required modules
include("../../src/gpu/GPU.jl")
using .GPU

# Import from GPU module
using .GPU.FaultTolerance
using .GPU.FaultTolerance: GPU_HEALTHY, GPU_DEGRADED, GPU_FAILING, GPU_FAILED, GPU_RECOVERING
using .GPU.FaultTolerance: NO_FAILURE, CUDA_ERROR, MEMORY_ERROR, TIMEOUT_ERROR
using .GPU.FaultTolerance: HEARTBEAT_FAILURE, THERMAL_THROTTLE, POWER_LIMIT
using .GPU.FaultTolerance: TimeoutError, handle_gpu_failure, handle_gpu_recovery

@testset "Fault Tolerance Tests" begin
    
    @testset "Health Monitor Creation" begin
        # Test with auto-detection
        monitor = create_health_monitor()
        @test isa(monitor, GPUHealthMonitor)
        @test monitor.heartbeat_interval == 1.0
        @test monitor.heartbeat_timeout == 5.0
        @test monitor.error_threshold == 10
        @test !monitor.monitoring_active[]
        
        # Test with custom parameters
        monitor2 = create_health_monitor(
            num_gpus = 2,
            heartbeat_interval = 0.5,
            heartbeat_timeout = 3.0,
            error_threshold = 5,
            temperature_limit = 80.0f0
        )
        @test length(monitor2.metrics) == 2
        @test monitor2.heartbeat_interval == 0.5
        @test monitor2.temperature_limit == 80.0f0
        
        # Check initialization
        for gpu_id in 0:1
            @test haskey(monitor2.metrics, gpu_id)
            @test monitor2.status[gpu_id] == GPU_HEALTHY
            @test monitor2.failure_modes[gpu_id] == NO_FAILURE
        end
    end
    
    @testset "GPU Status Management" begin
        monitor = create_health_monitor(num_gpus=2)
        
        # Test healthy status
        @test is_gpu_healthy(monitor, 0)
        @test get_gpu_status(monitor, 0) == GPU_HEALTHY
        
        # Manually set degraded status
        monitor.status[0] = GPU_DEGRADED
        @test !is_gpu_healthy(monitor, 0)
        @test get_gpu_status(monitor, 0) == GPU_DEGRADED
        
        # Test with invalid GPU ID
        @test get_gpu_status(monitor, 99) == GPU_FAILED
    end
    
    @testset "Error Tracking" begin
        monitor = create_health_monitor(num_gpus=1)
        
        # Simulate errors
        for i in 1:5
            handle_gpu_error(monitor, 0, CUDA.CuError(CUDA.ERROR_OUT_OF_MEMORY))
        end
        
        metrics = monitor.metrics[0]
        @test metrics.error_count[] == 5
        @test metrics.consecutive_errors[] == 5
        @test metrics.memory_errors[] == 5
        @test monitor.failure_modes[0] == MEMORY_ERROR
        
        # Test error reset
        reset_failure_count(monitor, 0)
        @test metrics.error_count[] == 0
        @test metrics.consecutive_errors[] == 0
        @test metrics.memory_errors[] == 0
    end
    
    @testset "Health Check Logic" begin
        monitor = create_health_monitor(
            num_gpus = 1,
            error_threshold = 5
        )
        # Set limits directly on the monitor
        monitor.consecutive_error_limit = 3
        monitor.memory_error_limit = 2
        
        # Healthy state
        @test check_gpu_health(monitor, 0) == GPU_HEALTHY
        
        # Simulate consecutive errors
        monitor.metrics[0].consecutive_errors[] = 3
        @test check_gpu_health(monitor, 0) == GPU_FAILED
        @test monitor.failure_modes[0] == CUDA_ERROR
        
        # Reset and test memory errors
        monitor.metrics[0].consecutive_errors[] = 0
        monitor.metrics[0].memory_errors[] = 2
        @test check_gpu_health(monitor, 0) == GPU_FAILING
        
        # Test total error threshold
        monitor.metrics[0].memory_errors[] = 0
        monitor.metrics[0].error_count[] = 5
        @test check_gpu_health(monitor, 0) == GPU_DEGRADED
        
        # Test heartbeat timeout
        monitor.metrics[0].error_count[] = 0
        monitor.metrics[0].last_heartbeat = now() - Second(10)
        @test check_gpu_health(monitor, 0) == GPU_FAILED
        @test monitor.failure_modes[0] == HEARTBEAT_FAILURE
    end
    
    @testset "Work Redistribution" begin
        monitor = create_health_monitor(num_gpus=3)
        
        # Mark GPU 1 as failed
        monitor.status[1] = GPU_FAILED
        
        # Test redistribution
        work_items = collect(1:100)
        distribution = redistribute_work!(monitor, 1, work_items)
        
        @test !isnothing(distribution)
        @test haskey(distribution, 0)
        @test haskey(distribution, 2)
        @test !haskey(distribution, 1)  # Failed GPU excluded
        
        # Check work is fully distributed
        total_redistributed = sum(length(items) for items in values(distribution))
        @test total_redistributed == length(work_items)
        
        # Test with all GPUs failed
        monitor.status[0] = GPU_FAILED
        monitor.status[2] = GPU_FAILED
        distribution2 = redistribute_work!(monitor, 1, work_items)
        @test isnothing(distribution2)
    end
    
    @testset "Graceful Degradation" begin
        monitor = create_health_monitor(num_gpus=3)
        
        # All healthy initially
        @test enable_graceful_degradation!(monitor)
        
        # Should have one healthy GPU, others failed
        healthy_count = count(status -> status == GPU_HEALTHY, values(monitor.status))
        failed_count = count(status -> status == GPU_FAILED, values(monitor.status))
        @test healthy_count == 1
        @test failed_count == 2
        
        # Test with all GPUs failed
        for gpu_id in keys(monitor.status)
            monitor.status[gpu_id] = GPU_FAILED
        end
        @test !enable_graceful_degradation!(monitor)
    end
    
    @testset "Checkpoint Management" begin
        # Create temporary checkpoint directory
        temp_dir = mktempdir()
        manager = create_checkpoint_manager(
            checkpoint_dir = temp_dir,
            max_checkpoints = 3,
            checkpoint_interval = 100
        )
        
        @test isdir(temp_dir)
        @test manager.max_checkpoints == 3
        
        # Save checkpoints
        for i in 1:5
            state_data = Dict{String, Any}("iteration" => i * 100, "score" => rand())
            save_checkpoint(manager, 0, i * 100, [1, 2, 3], state_data)
        end
        
        # Should only keep last 3 checkpoints
        @test length(manager.checkpoints[0]) == 3
        @test manager.checkpoints[0][1].iteration == 300  # Oldest kept
        @test manager.checkpoints[0][end].iteration == 500  # Newest
        
        # Test restore
        checkpoint = restore_checkpoint(manager, 0)
        @test !isnothing(checkpoint)
        @test checkpoint.iteration == 500
        
        # Test restore specific iteration
        checkpoint2 = restore_checkpoint(manager, 0, iteration=400)
        @test !isnothing(checkpoint2)
        @test checkpoint2.iteration == 400
        
        # Test restore non-existent
        checkpoint3 = restore_checkpoint(manager, 1)  # No checkpoints for GPU 1
        @test isnothing(checkpoint3)
        
        # Cleanup
        rm(temp_dir, recursive=true)
    end
    
    @testset "Callback System" begin
        monitor = create_health_monitor(num_gpus=1)
        
        # Track callback calls
        error_called = Ref(false)
        recovery_called = Ref(false)
        error_gpu_id = Ref(-1)
        
        # Register callbacks
        error_callback = function(gpu_id, status, mode)
            error_called[] = true
            error_gpu_id[] = gpu_id
        end
        register_error_callback!(monitor, error_callback)
        
        recovery_callback = function(gpu_id)
            recovery_called[] = true
        end
        push!(monitor.recovery_callbacks, recovery_callback)
        
        # Trigger failure
        handle_gpu_failure(monitor, 0, GPU_FAILED)
        @test error_called[]
        @test error_gpu_id[] == 0
        
        # Trigger recovery
        handle_gpu_recovery(monitor, 0)
        @test recovery_called[]
    end
    
    @testset "Failure Statistics" begin
        monitor = create_health_monitor(num_gpus=2)
        
        # Simulate some failures and recoveries
        monitor.total_failures[] = 5
        monitor.total_recoveries[] = 3
        monitor.metrics[0].error_count[] = 10
        monitor.metrics[1].memory_errors[] = 2
        
        stats = get_failure_statistics(monitor)
        
        @test stats["total_failures"] == 5
        @test stats["total_recoveries"] == 3
        @test haskey(stats, "gpu_stats")
        @test stats["gpu_stats"][0]["error_count"] == 10
        @test stats["gpu_stats"][1]["memory_errors"] == 2
    end
    
    @testset "Monitoring Integration" begin
        if CUDA.functional()
            monitor = create_health_monitor(num_gpus=1)
            
            # Start monitoring
            start_monitoring!(monitor)
            @test monitor.monitoring_active[]
            @test haskey(monitor.monitor_tasks, 0)
            @test haskey(monitor.heartbeat_tasks, 0)
            
            # Let it run briefly
            sleep(0.2)
            
            # Stop monitoring
            stop_monitoring!(monitor)
            @test !monitor.monitoring_active[]
            @test isempty(monitor.monitor_tasks)
            @test isempty(monitor.heartbeat_tasks)
        else
            @test_skip "CUDA not functional - skipping monitoring integration"
        end
    end
    
    @testset "Edge Cases" begin
        monitor = create_health_monitor(num_gpus=2)
        
        # Multiple error types
        handle_gpu_error(monitor, 0, CUDA.CuError(CUDA.ERROR_INVALID_VALUE))
        handle_gpu_error(monitor, 0, TimeoutError("Kernel timeout"))
        handle_gpu_error(monitor, 0, ErrorException("Generic error"))
        
        @test monitor.metrics[0].error_count[] == 3
        @test monitor.metrics[0].kernel_timeouts[] == 1
        
        # Work redistribution with empty work
        distribution = redistribute_work!(monitor, 0, Int[])
        @test !isnothing(distribution)
        @test all(isempty, values(distribution))
        
        # Graceful degradation with one GPU
        monitor2 = create_health_monitor(num_gpus=1)
        @test enable_graceful_degradation!(monitor2)
        @test monitor2.status[0] == GPU_HEALTHY
    end
    
end

# Print summary
println("\nFault Tolerance Test Summary:")
println("=============================")
if CUDA.functional()
    println("✓ CUDA functional - Full fault tolerance tests executed")
    println("  Health monitoring validated")
    println("  Error tracking and recovery tested")
    println("  Work redistribution verified")
    println("  Checkpoint system operational")
else
    println("⚠ CUDA not functional - Basic fault tolerance tests only")
end
println("\nAll fault tolerance tests completed!")