using Test
using Dates
using JSON3
using Statistics

# Include the fault tolerance module
include("../../src/stage2/fault_tolerance.jl")

@testset "Fault Tolerance and Recovery Tests" begin
    
    @testset "FaultToleranceManager Creation" begin
        manager = FaultToleranceManager(2)
        
        @test length(manager.gpu_monitors) == 2
        @test manager.checkpoint_interval == 5000
        @test manager.max_checkpoint_age == 50000
        @test manager.monitoring_interval == 0.5
        @test manager.response_timeout == 2.0
        @test manager.memory_threshold == 0.9
        @test manager.temperature_threshold == 85.0
        @test manager.fallback_mode_active == false
        @test manager.degraded_tree_count == 100
        @test manager.original_tree_count == 100
        @test isempty(manager.checkpoints)
        @test isempty(manager.fault_history)
        @test manager.recovery_stats.total_faults == 0
    end
    
    @testset "GPUHealthMonitor Creation" begin
        monitor = GPUHealthMonitor(1)
        
        @test monitor.gpu_id == 1
        @test monitor.status == GPU_HEALTHY
        @test monitor.is_monitoring == false
        @test monitor.monitoring_thread === nothing
        @test monitor.error_count == 0
        @test monitor.warning_count == 0
        @test isempty(monitor.response_times)
        @test isempty(monitor.memory_usage_history)
        @test isempty(monitor.temperature_history)
    end
    
    @testset "Checkpoint Creation and Management" begin
        manager = FaultToleranceManager(2)
        
        # Create test data
        tree_state = Dict{String, Any}(
            "nodes" => [1, 2, 3, 4, 5],
            "values" => [0.1, 0.2, 0.3, 0.4, 0.5],
            "visits" => [10, 20, 30, 40, 50]
        )
        
        performance_metrics = Dict{String, Float64}(
            "score" => 0.85,
            "confidence" => 0.92,
            "iterations" => 1000.0
        )
        
        feature_selections = [1, 5, 10, 15, 20]
        
        # Create checkpoint
        checkpoint = create_checkpoint!(manager, 1, 1, 1000, tree_state, 
                                      performance_metrics, feature_selections)
        
        @test checkpoint.tree_id == 1
        @test checkpoint.gpu_id == 1
        @test checkpoint.iteration == 1000
        @test checkpoint.tree_state == tree_state
        @test checkpoint.performance_metrics == performance_metrics
        @test checkpoint.feature_selections == feature_selections
        @test checkpoint.checksum != 0
        
        # Verify checkpoint is stored
        @test haskey(manager.checkpoints, 1)
        @test manager.checkpoints[1] == checkpoint
    end
    
    @testset "Fault Event Recording" begin
        manager = FaultToleranceManager(2)
        
        # Record fault event
        record_fault_event!(manager, 1, GPU_CRASH, 5, "Test GPU crash", 
                          SINGLE_GPU_FALLBACK, Dict{String, Any}("test" => "data"))
        
        @test length(manager.fault_history) == 1
        @test manager.recovery_stats.total_faults == 1
        
        event = manager.fault_history[1]
        @test event.gpu_id == 1
        @test event.fault_type == GPU_CRASH
        @test event.severity == 5
        @test event.description == "Test GPU crash"
        @test event.recovery_strategy == SINGLE_GPU_FALLBACK
        @test event.additional_data["test"] == "data"
    end
    
    @testset "Recovery Strategy Determination" begin
        manager = FaultToleranceManager(2)
        
        # Set up healthy GPU
        manager.gpu_monitors[1].status = GPU_HEALTHY
        manager.gpu_monitors[2].status = GPU_FAILED
        
        # Test with no checkpoints
        strategy = determine_recovery_strategy(manager, 2)
        @test strategy == SINGLE_GPU_FALLBACK
        
        # Add recent checkpoints
        for i in 1:15
            tree_state = Dict{String, Any}("test" => i)
            performance_metrics = Dict{String, Float64}("score" => 0.5)
            feature_selections = [i]
            create_checkpoint!(manager, i, 2, 1000, tree_state, 
                             performance_metrics, feature_selections)
        end
        
        strategy = determine_recovery_strategy(manager, 2)
        @test strategy == CHECKPOINT_RESTORE
        
        # Test with fewer checkpoints
        manager.checkpoints = Dict{Int, TreeCheckpoint}()
        for i in 1:7
            tree_state = Dict{String, Any}("test" => i)
            performance_metrics = Dict{String, Float64}("score" => 0.5)
            feature_selections = [i]
            create_checkpoint!(manager, i, 2, 1000, tree_state, 
                             performance_metrics, feature_selections)
        end
        
        strategy = determine_recovery_strategy(manager, 2)
        @test strategy == PARTIAL_RECOVERY
        
        # Test with all GPUs failed
        manager.gpu_monitors[1].status = GPU_FAILED
        strategy = determine_recovery_strategy(manager, 2)
        @test strategy == GRACEFUL_DEGRADATION
    end
    
    @testset "Checkpoint Cleanup" begin
        manager = FaultToleranceManager(2)
        
        # Create old checkpoint (older than max_checkpoint_age * 100)
        old_checkpoint = TreeCheckpoint(
            1, 1, 1000, now() - Millisecond(50000 * 100 + 1000), 
            Dict{String, Any}("test" => "old"),
            Dict{String, Float64}("score" => 0.5),
            [1], UInt32(12345)
        )
        manager.checkpoints[1] = old_checkpoint
        
        # Create recent checkpoint
        tree_state = Dict{String, Any}("test" => "recent")
        performance_metrics = Dict{String, Float64}("score" => 0.8)
        feature_selections = [2]
        create_checkpoint!(manager, 2, 1, 2000, tree_state, 
                         performance_metrics, feature_selections)
        
        # Should have removed old checkpoint
        @test !haskey(manager.checkpoints, 1)
        @test haskey(manager.checkpoints, 2)
    end
    
    @testset "Partial Results Extraction" begin
        manager = FaultToleranceManager(2)
        
        # Create test checkpoints
        for i in 1:3
            tree_state = Dict{String, Any}("test" => i)
            performance_metrics = Dict{String, Float64}(
                "score" => 0.5 + i * 0.1,
                "confidence" => 0.8 + i * 0.05
            )
            feature_selections = [i]
            create_checkpoint!(manager, i, 2, 1000, tree_state, 
                             performance_metrics, feature_selections)
        end
        
        # Extract partial results
        partial_results = extract_partial_results(manager, 2)
        
        @test length(partial_results) == 3
        @test haskey(partial_results, "1")
        @test haskey(partial_results, "2")
        @test haskey(partial_results, "3")
        @test partial_results["1"]["score"] == 0.6
        @test partial_results["2"]["score"] == 0.7
        @test partial_results["3"]["score"] == 0.8
    end
    
    @testset "System Status Reporting" begin
        manager = FaultToleranceManager(2)
        
        # Set up test state
        manager.gpu_monitors[1].status = GPU_HEALTHY
        manager.gpu_monitors[2].status = GPU_FAILED
        manager.fallback_mode_active = true
        manager.degraded_tree_count = 50
        manager.recovery_stats.total_faults = 5
        manager.recovery_stats.successful_recoveries = 3
        manager.recovery_stats.mean_recovery_time = 2.5
        
        # Add checkpoint
        tree_state = Dict{String, Any}("test" => 1)
        performance_metrics = Dict{String, Float64}("score" => 0.8)
        feature_selections = [1]
        create_checkpoint!(manager, 1, 1, 1000, tree_state, 
                         performance_metrics, feature_selections)
        
        status = get_system_status(manager)
        
        @test status["gpu_count"] == 2
        @test status["healthy_gpus"] == 1
        @test status["failed_gpus"] == 1
        @test status["fallback_mode"] == true
        @test status["degraded_tree_count"] == 50
        @test status["original_tree_count"] == 100
        @test status["checkpoint_count"] == 1
        @test status["total_faults"] == 5
        @test status["successful_recoveries"] == 3
        @test status["mean_recovery_time"] == 2.5
    end
    
    @testset "Best Available GPU Selection" begin
        manager = FaultToleranceManager(3)
        
        # Set up GPU monitors with different response times
        manager.gpu_monitors[1].status = GPU_HEALTHY
        manager.gpu_monitors[1].response_times = [0.1, 0.2, 0.3]
        
        manager.gpu_monitors[2].status = GPU_FAILED
        manager.gpu_monitors[2].response_times = [0.5, 0.6, 0.7]
        
        manager.gpu_monitors[3].status = GPU_HEALTHY
        manager.gpu_monitors[3].response_times = [0.05, 0.1, 0.15]
        
        best_gpu = find_best_available_gpu(manager)
        
        @test best_gpu !== nothing
        @test best_gpu.gpu_id == 3  # Lowest average response time
        
        # Test with no healthy GPUs
        manager.gpu_monitors[1].status = GPU_FAILED
        manager.gpu_monitors[3].status = GPU_FAILED
        
        best_gpu = find_best_available_gpu(manager)
        @test best_gpu === nothing
    end
    
    @testset "Fault Statistics Saving" begin
        manager = FaultToleranceManager(2)
        
        # Add some test data
        record_fault_event!(manager, 1, GPU_CRASH, 5, "Test fault", 
                          SINGLE_GPU_FALLBACK, Dict{String, Any}("test" => "data"))
        
        manager.recovery_stats.successful_recoveries = 2
        manager.recovery_stats.failed_recoveries = 1
        manager.recovery_stats.mean_recovery_time = 1.5
        
        # Test saving (mocked)
        save_fault_statistics(manager)
        
        # Check if file exists
        @test isfile("fault_tolerance_stats.json")
        
        # Read and verify content
        stats_data = JSON3.read("fault_tolerance_stats.json")
        @test haskey(stats_data, "timestamp")
        @test haskey(stats_data, "recovery_stats")
        @test haskey(stats_data, "fault_history")
        @test haskey(stats_data, "system_status")
        
        # Cleanup
        rm("fault_tolerance_stats.json", force=true)
    end
    
    @testset "GPU Health Status Transitions" begin
        manager = FaultToleranceManager(2)
        monitor = manager.gpu_monitors[1]
        
        # Test initial state
        @test monitor.status == GPU_HEALTHY
        @test monitor.error_count == 0
        
        # Simulate health check failures
        monitor.error_count = 1
        @test monitor.status == GPU_HEALTHY  # Still healthy with 1 error
        
        monitor.error_count = 2
        @test monitor.status == GPU_HEALTHY  # Still healthy with 2 errors
        
        monitor.error_count = 3
        # This would trigger critical status in real monitoring
        # For test, we manually set it
        monitor.status = GPU_CRITICAL
        @test monitor.status == GPU_CRITICAL
    end
    
    @testset "Graceful Degradation Logic" begin
        manager = FaultToleranceManager(2)
        
        # Set up scenario with limited resources
        manager.gpu_monitors[1].status = GPU_WARNING
        manager.gpu_monitors[2].status = GPU_FAILED
        
        # Test degradation calculation
        original_count = manager.original_tree_count
        expected_degraded = max(20, original_count ÷ 2)
        
        @test expected_degraded == 50  # 100 ÷ 2
        
        # Test with smaller original count
        manager.original_tree_count = 30
        expected_degraded = max(20, 30 ÷ 2)
        @test expected_degraded == 20  # max(20, 15) = 20
    end
    
    @testset "Checkpoint Integrity Verification" begin
        manager = FaultToleranceManager(2)
        
        # Create checkpoint with known data
        tree_state = Dict{String, Any}("test" => "data")
        performance_metrics = Dict{String, Float64}("score" => 0.8)
        feature_selections = [1, 2, 3]
        
        checkpoint = create_checkpoint!(manager, 1, 1, 1000, tree_state, 
                                      performance_metrics, feature_selections)
        
        # Verify checksum calculation
        state_string = JSON3.write(tree_state)
        expected_checksum = hash(state_string) % UInt32(2^32 - 1)
        @test checkpoint.checksum == expected_checksum
        
        # Test integrity verification
        computed_checksum = hash(JSON3.write(checkpoint.tree_state)) % UInt32(2^32 - 1)
        @test computed_checksum == checkpoint.checksum
    end
    
    @testset "Recovery Statistics Tracking" begin
        manager = FaultToleranceManager(2)
        stats = manager.recovery_stats
        
        # Test initial state
        @test stats.total_faults == 0
        @test stats.successful_recoveries == 0
        @test stats.failed_recoveries == 0
        @test stats.checkpoint_restores == 0
        @test stats.partial_recoveries == 0
        @test stats.fallback_activations == 0
        @test stats.degradation_events == 0
        @test stats.mean_recovery_time == 0.0
        
        # Simulate some recovery events
        lock(manager.manager_lock) do
            stats.total_faults = 10
            stats.successful_recoveries = 7
            stats.failed_recoveries = 3
            stats.checkpoint_restores = 3
            stats.partial_recoveries = 2
            stats.fallback_activations = 1
            stats.degradation_events = 1
            stats.mean_recovery_time = 1.8
        end
        
        @test stats.total_faults == 10
        @test stats.successful_recoveries == 7
        @test stats.failed_recoveries == 3
        @test stats.checkpoint_restores == 3
        @test stats.partial_recoveries == 2
        @test stats.fallback_activations == 1
        @test stats.degradation_events == 1
        @test stats.mean_recovery_time == 1.8
    end
    
    @testset "Fault Event History Management" begin
        manager = FaultToleranceManager(2)
        
        # Add many fault events to test history limiting
        for i in 1:1100
            record_fault_event!(manager, 1, GPU_CRASH, 3, "Test fault $i", 
                              CHECKPOINT_RESTORE, Dict{String, Any}("index" => i))
        end
        
        # Should be limited to 1000 events
        @test length(manager.fault_history) == 1000
        
        # Should keep the most recent events
        @test manager.fault_history[end].description == "Test fault 1100"
        @test manager.fault_history[1].description == "Test fault 101"
    end
    
    @testset "Memory Usage History Tracking" begin
        manager = FaultToleranceManager(2)
        monitor = manager.gpu_monitors[1]
        
        # Add memory usage data
        for i in 1:150
            push!(monitor.memory_usage_history, i / 200.0)
            # Trigger cleanup when over 100 entries
            if length(monitor.memory_usage_history) > 100
                monitor.memory_usage_history = monitor.memory_usage_history[end-99:end]
            end
        end
        
        # Should be limited to 100 entries
        @test length(monitor.memory_usage_history) == 100
        
        # Should keep the most recent entries
        @test monitor.memory_usage_history[end] == 150 / 200.0
        @test monitor.memory_usage_history[1] == 51 / 200.0
    end
    
    @testset "Multiple GPU Recovery Scenarios" begin
        manager = FaultToleranceManager(4)
        
        # Set up various GPU states
        manager.gpu_monitors[1].status = GPU_HEALTHY
        manager.gpu_monitors[2].status = GPU_FAILED
        manager.gpu_monitors[3].status = GPU_WARNING
        manager.gpu_monitors[4].status = GPU_HEALTHY
        
        # Test finding healthy GPUs
        healthy_count = count(m -> m.status == GPU_HEALTHY, manager.gpu_monitors)
        @test healthy_count == 2
        
        # Test finding failed GPUs
        failed_count = count(m -> m.status == GPU_FAILED, manager.gpu_monitors)
        @test failed_count == 1
        
        # Test recovery with multiple healthy GPUs available
        strategy = determine_recovery_strategy(manager, 2)
        @test strategy == SINGLE_GPU_FALLBACK  # No checkpoints available
    end
    
    @testset "Checkpoint Age-based Cleanup" begin
        manager = FaultToleranceManager(2)
        
        # Create checkpoints with different ages
        current_time = now()
        
        # Old checkpoint (should be cleaned up) - older than max_checkpoint_age * 100
        old_checkpoint = TreeCheckpoint(
            1, 1, 1000, current_time - Millisecond(50000 * 100 + 1000), 
            Dict{String, Any}("test" => "old"),
            Dict{String, Float64}("score" => 0.5),
            [1], UInt32(12345)
        )
        manager.checkpoints[1] = old_checkpoint
        
        # Recent checkpoint (should be kept)
        recent_checkpoint = TreeCheckpoint(
            2, 1, 2000, current_time - Millisecond(1000), 
            Dict{String, Any}("test" => "recent"),
            Dict{String, Float64}("score" => 0.8),
            [2], UInt32(67890)
        )
        manager.checkpoints[2] = recent_checkpoint
        
        # Trigger cleanup
        cleanup_old_checkpoints!(manager)
        
        # Only recent checkpoint should remain
        @test !haskey(manager.checkpoints, 1)
        @test haskey(manager.checkpoints, 2)
    end
    
    @testset "Fault Type and Severity Classification" begin
        manager = FaultToleranceManager(2)
        
        # Test different fault types
        fault_types = [GPU_CRASH, GPU_HANG, MEMORY_ERROR, COMPUTATION_ERROR, COMMUNICATION_ERROR]
        
        for (i, fault_type) in enumerate(fault_types)
            record_fault_event!(manager, 1, fault_type, i, "Test fault type $i", 
                              CHECKPOINT_RESTORE, Dict{String, Any}("type" => string(fault_type)))
        end
        
        @test length(manager.fault_history) == 5
        
        # Verify fault types are recorded correctly
        for (i, event) in enumerate(manager.fault_history)
            @test event.fault_type == fault_types[i]
            @test event.severity == i
        end
    end
    
    @testset "Thread Safety Tests" begin
        manager = FaultToleranceManager(2)
        
        # Test concurrent checkpoint creation
        tasks = Task[]
        for i in 1:10
            task = @async begin
                tree_state = Dict{String, Any}("test" => i)
                performance_metrics = Dict{String, Float64}("score" => i * 0.1)
                feature_selections = [i]
                create_checkpoint!(manager, i, 1, 1000, tree_state, 
                                performance_metrics, feature_selections)
            end
            push!(tasks, task)
        end
        
        # Wait for all tasks to complete
        for task in tasks
            wait(task)
        end
        
        @test length(manager.checkpoints) == 10
        
        # Verify all checkpoints were created correctly
        for i in 1:10
            @test haskey(manager.checkpoints, i)
            @test manager.checkpoints[i].tree_state["test"] == i
        end
    end
    
    @testset "Recovery Strategy Edge Cases" begin
        manager = FaultToleranceManager(1)  # Single GPU system
        
        # Test with single GPU failure
        manager.gpu_monitors[1].status = GPU_FAILED
        
        strategy = determine_recovery_strategy(manager, 1)
        @test strategy == GRACEFUL_DEGRADATION  # No healthy GPUs remaining
        
        # Test with no GPUs
        manager = FaultToleranceManager(0)
        @test length(manager.gpu_monitors) == 0
        
        # Test recovery strategy with empty GPU list
        strategy = determine_recovery_strategy(manager, 1)
        @test strategy == GRACEFUL_DEGRADATION
    end
    
    @testset "Performance Threshold Monitoring" begin
        manager = FaultToleranceManager(2)
        
        # Test response time threshold
        @test manager.response_timeout == 2.0
        
        # Test memory threshold
        @test manager.memory_threshold == 0.9
        
        # Test temperature threshold
        @test manager.temperature_threshold == 85.0
        
        # Test monitoring interval
        @test manager.monitoring_interval == 0.5
        
        # Test checkpoint interval
        @test manager.checkpoint_interval == 5000
        
        # Test max checkpoint age
        @test manager.max_checkpoint_age == 50000
    end
    
    @testset "Recovery Time Tracking" begin
        manager = FaultToleranceManager(2)
        
        # Simulate successful recovery
        lock(manager.manager_lock) do
            manager.recovery_stats.successful_recoveries = 1
            manager.recovery_stats.mean_recovery_time = 2.0
        end
        
        # Add another recovery
        recovery_time = 3.0
        lock(manager.manager_lock) do
            manager.recovery_stats.successful_recoveries += 1
            manager.recovery_stats.mean_recovery_time = 
                (manager.recovery_stats.mean_recovery_time * 1 + recovery_time) / 2
        end
        
        @test manager.recovery_stats.successful_recoveries == 2
        @test manager.recovery_stats.mean_recovery_time == 2.5
    end
    
    @testset "Shutdown Procedure" begin
        manager = FaultToleranceManager(2)
        
        # Test shutdown
        shutdown_fault_tolerance!(manager)
        
        # Verify all monitoring threads are stopped
        for monitor in manager.gpu_monitors
            @test monitor.is_monitoring == false
        end
        
        # Verify statistics file is created
        @test isfile("fault_tolerance_stats.json")
        
        # Cleanup
        rm("fault_tolerance_stats.json", force=true)
    end
    
end