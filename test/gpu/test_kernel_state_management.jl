using Test
using CUDA
using Dates
using Serialization

# Include the kernel state management module
include("../../src/gpu/kernels/kernel_state_management.jl")

using .KernelStateManagement

# Helper function to wait for state
function wait_for_state(manager::StateManager, expected_state::KernelExecutionState, timeout::Float64 = 5.0)
    start_time = time()
    while time() - start_time < timeout
        if get_current_state(manager) == expected_state
            return true
        end
        sleep(0.01)
    end
    return false
end

@testset "Kernel State Management Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU kernel state management tests"
        return
    end
    
    @testset "KernelState Creation" begin
        state = KernelState(version=Int32(2))
        
        CUDA.@allowscalar begin
            @test state.execution_state[1] == Int32(KERNEL_UNINITIALIZED)
            @test state.error_code[1] == Int32(ERROR_NONE)
            @test state.error_count[1] == 0
            @test state.iteration[1] == 0
            @test state.total_iterations[1] == 0
            @test state.last_checkpoint_iter[1] == 0
            @test state.start_time[1] == 0.0
            @test state.last_update_time[1] == 0.0
            @test state.total_runtime[1] == 0.0
            @test state.should_stop[1] == false
            @test state.should_pause[1] == false
            @test state.should_checkpoint[1] == false
            @test state.state_hash[1] == 0
            @test state.version[1] == 2
        end
    end
    
    @testset "StateManager Creation" begin
        # Test with temporary directory
        checkpoint_dir = mktempdir()
        
        manager = StateManager(
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=100,
            max_checkpoints=3,
            auto_recovery=false
        )
        
        @test manager.checkpoint_dir == checkpoint_dir
        @test manager.checkpoint_interval == 100
        @test manager.max_checkpoints == 3
        @test manager.auto_recovery == false
        @test isempty(manager.checkpoints)
        @test isnothing(manager.state_change_callback)
        @test isnothing(manager.error_callback)
        
        # Cleanup
        rm(checkpoint_dir, recursive=true)
    end
    
    @testset "State Transitions" begin
        manager = StateManager(checkpoint_dir=mktempdir(), auto_recovery=false)
        
        # Valid transitions
        @test get_current_state(manager) == KERNEL_UNINITIALIZED
        @test try_transition!(manager, KERNEL_INITIALIZING)
        @test get_current_state(manager) == KERNEL_INITIALIZING
        @test try_transition!(manager, KERNEL_RUNNING)
        @test get_current_state(manager) == KERNEL_RUNNING
        @test try_transition!(manager, KERNEL_PAUSED)
        @test get_current_state(manager) == KERNEL_PAUSED
        @test try_transition!(manager, KERNEL_RUNNING)
        @test get_current_state(manager) == KERNEL_RUNNING
        @test try_transition!(manager, KERNEL_STOPPING)
        @test get_current_state(manager) == KERNEL_STOPPING
        @test try_transition!(manager, KERNEL_STOPPED)
        @test get_current_state(manager) == KERNEL_STOPPED
        
        # Invalid transitions
        @test !try_transition!(manager, KERNEL_PAUSED)  # Can't pause from stopped
        @test get_current_state(manager) == KERNEL_STOPPED
        
        # Cleanup
        rm(manager.checkpoint_dir, recursive=true)
    end
    
    @testset "State Callbacks" begin
        manager = StateManager(checkpoint_dir=mktempdir(), auto_recovery=false)
        
        # Track state changes
        state_changes = Tuple{KernelExecutionState, KernelExecutionState}[]
        manager.state_change_callback = (from, to) -> push!(state_changes, (from, to))
        
        # Track errors
        errors = Tuple{KernelErrorCode, String}[]
        manager.error_callback = (code, msg) -> push!(errors, (code, msg))
        
        # Perform transitions
        try_transition!(manager, KERNEL_INITIALIZING)
        try_transition!(manager, KERNEL_RUNNING)
        
        @test length(state_changes) == 2
        @test state_changes[1] == (KERNEL_UNINITIALIZED, KERNEL_INITIALIZING)
        @test state_changes[2] == (KERNEL_INITIALIZING, KERNEL_RUNNING)
        
        # Trigger error
        set_error!(manager, ERROR_OUT_OF_MEMORY, "Test error")
        
        @test length(errors) == 1
        @test errors[1][1] == ERROR_OUT_OF_MEMORY
        @test errors[1][2] == "Test error"
        @test get_current_state(manager) == KERNEL_ERROR
        
        # Cleanup
        rm(manager.checkpoint_dir, recursive=true)
    end
    
    @testset "State Validation" begin
        manager = StateManager(checkpoint_dir=mktempdir(), auto_recovery=false)
        
        # Initial state should be valid
        @test validate_state(manager)
        
        # Initialize and verify
        initialize_kernel!(manager, Int64(1000))
        @test validate_state(manager)
        
        # Corrupt state hash and verify validation fails
        CUDA.@allowscalar manager.kernel_state.state_hash[1] = UInt64(12345)
        @test !validate_state(manager)
        
        # Error count should increase
        CUDA.@allowscalar begin
            @test manager.kernel_state.error_count[1] == 1
            @test manager.kernel_state.error_code[1] == Int32(ERROR_INVALID_STATE)
        end
        
        # Cleanup
        rm(manager.checkpoint_dir, recursive=true)
    end
    
    @testset "Progress Tracking" begin
        manager = StateManager(
            checkpoint_dir=mktempdir(),
            checkpoint_interval=10,
            auto_recovery=false
        )
        
        # Initialize with total iterations
        initialize_kernel!(manager, Int64(100))
        
        # Check initial checkpoint iteration
        CUDA.@allowscalar @test manager.kernel_state.last_checkpoint_iter[1] == 0
        
        # Update progress
        for i in 1:5
            update_progress!(manager, Int64(1))
        end
        
        # Check progress
        CUDA.@allowscalar begin
            @test manager.kernel_state.iteration[1] == 5
            @test manager.kernel_state.total_runtime[1] > 0.0
        end
        
        # Update with larger delta to trigger checkpoint
        update_progress!(manager, Int64(5))  # Total will be 10, triggering checkpoint
        
        CUDA.@allowscalar begin
            @test manager.kernel_state.iteration[1] == 10
            # Checkpoint should have been performed and flag reset
            @test manager.kernel_state.should_checkpoint[1] == false
            # Verify checkpoint was created by checking the checkpoint iteration was updated
            @test manager.kernel_state.last_checkpoint_iter[1] == 10
        end
        
        # Cleanup
        rm(manager.checkpoint_dir, recursive=true)
    end
    
    @testset "Control Signals" begin
        manager = StateManager(checkpoint_dir=mktempdir(), auto_recovery=false)
        
        initialize_kernel!(manager)
        @test get_current_state(manager) == KERNEL_RUNNING
        
        # Test pause
        request_pause!(manager)
        CUDA.@allowscalar @test manager.kernel_state.should_pause[1] == true
        
        # Test resume
        request_resume!(manager)
        CUDA.@allowscalar @test manager.kernel_state.should_pause[1] == false
        
        # Test stop
        request_stop!(manager)
        CUDA.@allowscalar @test manager.kernel_state.should_stop[1] == true
        
        # Cleanup
        rm(manager.checkpoint_dir, recursive=true)
    end
    
    @testset "Checkpoint Creation" begin
        manager = StateManager(checkpoint_dir=mktempdir(), auto_recovery=false)
        
        initialize_kernel!(manager, Int64(1000))
        update_progress!(manager, Int64(50))
        
        # Create checkpoint with app data
        app_data = Dict{String, Any}(
            "model_weights" => rand(10),
            "training_loss" => 0.123,
            "epoch" => 5
        )
        
        checkpoint = KernelStateManagement.create_checkpoint(manager, app_data)
        
        @test isa(checkpoint, KernelCheckpoint)
        @test checkpoint.iteration == 50
        @test checkpoint.execution_state == KERNEL_RUNNING
        @test checkpoint.error_code == ERROR_NONE
        @test checkpoint.app_data == app_data
        @test checkpoint.checksum != 0
        
        # Cleanup
        rm(manager.checkpoint_dir, recursive=true)
    end
    
    @testset "Checkpoint Save and Load" begin
        checkpoint_dir = mktempdir()
        manager = StateManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=2,
            auto_recovery=false
        )
        
        initialize_kernel!(manager, Int64(1000))
        
        # Create and save multiple checkpoints
        checkpoints = KernelCheckpoint[]
        for i in 1:3
            update_progress!(manager, Int64(100))
            
            app_data = Dict{String, Any}("iteration" => i * 100)
            checkpoint = KernelStateManagement.create_checkpoint(manager, app_data)
            
            @test KernelStateManagement.save_checkpoint(manager, checkpoint)
            push!(checkpoints, checkpoint)
        end
        
        # Should only keep max_checkpoints
        @test length(manager.checkpoints) == 2
        
        # Verify oldest checkpoint file was deleted
        oldest_file = joinpath(
            checkpoint_dir,
            "checkpoint_$(checkpoints[1].checkpoint_id)_iter$(checkpoints[1].iteration).jld2"
        )
        @test !isfile(oldest_file)
        
        # Load a checkpoint
        checkpoint_file = joinpath(
            checkpoint_dir,
            "checkpoint_$(checkpoints[2].checkpoint_id)_iter$(checkpoints[2].iteration).jld2"
        )
        loaded = KernelStateManagement.load_checkpoint(checkpoint_file)
        
        @test !isnothing(loaded)
        @test loaded.iteration == checkpoints[2].iteration
        @test loaded.app_data == checkpoints[2].app_data
        
        # Cleanup
        rm(checkpoint_dir, recursive=true)
    end
    
    @testset "Checkpoint Corruption Detection" begin
        checkpoint_dir = mktempdir()
        
        # Create a checkpoint
        checkpoint = KernelCheckpoint(
            "test-id",
            now(),
            Int64(100),
            Int32(1),
            KERNEL_RUNNING,
            ERROR_NONE,
            10.0,
            Dict{String, Any}("test" => "data"),
            UInt64(0)  # Will be recalculated
        )
        
        # Save with correct checksum
        filename = joinpath(checkpoint_dir, "test_checkpoint.jld2")
        serialize(filename, checkpoint)
        
        # Load should fail due to zero checksum
        loaded = KernelStateManagement.load_checkpoint(filename)
        @test isnothing(loaded)
        
        # Cleanup
        rm(checkpoint_dir, recursive=true)
    end
    
    @testset "Perform Checkpoint" begin
        manager = StateManager(checkpoint_dir=mktempdir(), auto_recovery=false)
        
        initialize_kernel!(manager, Int64(1000))
        update_progress!(manager, Int64(50))
        
        # Perform checkpoint
        app_data = Dict{String, Any}("checkpoint_test" => true)
        @test perform_checkpoint(manager, app_data)
        
        # Should transition back to running
        @test get_current_state(manager) == KERNEL_RUNNING
        
        # Should have saved checkpoint
        @test length(manager.checkpoints) == 1
        @test manager.checkpoints[1].iteration == 50
        
        # Checkpoint flag should be cleared
        CUDA.@allowscalar @test manager.kernel_state.should_checkpoint[1] == false
        
        # Cleanup
        rm(manager.checkpoint_dir, recursive=true)
    end
    
    @testset "Checkpoint Restore" begin
        checkpoint_dir = mktempdir()
        manager = StateManager(checkpoint_dir=checkpoint_dir, auto_recovery=false)
        
        # Initialize and create state
        initialize_kernel!(manager, Int64(2000))
        update_progress!(manager, Int64(500))
        
        # Save checkpoint
        app_data = Dict{String, Any}("model_state" => "checkpoint_data")
        perform_checkpoint(manager, app_data)
        
        saved_checkpoint = manager.checkpoints[1]
        
        # Reset state
        manager2 = StateManager(checkpoint_dir=checkpoint_dir, auto_recovery=false)
        
        # Transition to a valid state for restore (don't fully initialize)
        try_transition!(manager2, KERNEL_INITIALIZING)
        
        # Restore checkpoint
        success, restored_data = restore_checkpoint!(manager2, saved_checkpoint)
        
        @test success
        @test restored_data == app_data
        
        # Verify state was restored
        CUDA.@allowscalar begin
            @test manager2.kernel_state.iteration[1] == 500
            @test manager2.kernel_state.total_runtime[1] == saved_checkpoint.total_runtime
        end
        
        # Should be in paused state after restore if the checkpoint was from running state
        # Otherwise it should be in the checkpoint's state
        expected_state = saved_checkpoint.execution_state == KERNEL_RUNNING ? KERNEL_PAUSED : saved_checkpoint.execution_state
        @test get_current_state(manager2) == expected_state
        
        # Cleanup
        rm(checkpoint_dir, recursive=true)
    end
    
    @testset "Auto Recovery" begin
        checkpoint_dir = mktempdir()
        
        # Create manager with checkpoint
        manager1 = StateManager(checkpoint_dir=checkpoint_dir, auto_recovery=false)
        initialize_kernel!(manager1, Int64(1000))
        update_progress!(manager1, Int64(750))
        perform_checkpoint(manager1, Dict{String, Any}("auto_recovery_test" => true))
        
        # Create new manager with auto-recovery
        manager2 = StateManager(checkpoint_dir=checkpoint_dir, auto_recovery=true)
        
        # Initialize should trigger auto-recovery
        @test initialize_kernel!(manager2)
        
        # Should have recovered to previous state
        CUDA.@allowscalar begin
            @test manager2.kernel_state.iteration[1] == 750
        end
        
        # Cleanup
        rm(checkpoint_dir, recursive=true)
    end
    
    @testset "Find Latest Checkpoint" begin
        checkpoint_dir = mktempdir()
        manager = StateManager(checkpoint_dir=checkpoint_dir, auto_recovery=false)
        
        # No checkpoints initially
        @test isnothing(find_latest_checkpoint(manager))
        
        # Create multiple checkpoints
        initialize_kernel!(manager)
        for i in 1:3
            update_progress!(manager, Int64(100))
            perform_checkpoint(manager, Dict{String, Any}("index" => i))
            sleep(0.1)  # Ensure different modification times
        end
        
        # Find latest
        latest = find_latest_checkpoint(manager)
        @test !isnothing(latest)
        @test latest.app_data["index"] == 3
        @test latest.iteration == 300
        
        # Cleanup
        rm(checkpoint_dir, recursive=true)
    end
    
    @testset "Kernel Statistics" begin
        manager = StateManager(checkpoint_dir=mktempdir(), auto_recovery=false)
        
        initialize_kernel!(manager, Int64(1000))
        
        # Generate some activity
        for i in 1:5
            update_progress!(manager, Int64(100))
        end
        
        # Trigger an error
        set_error!(manager, ERROR_TIMEOUT, "Test timeout")
        
        # Get statistics
        stats = get_kernel_stats(manager)
        
        @test stats["state"] == KERNEL_ERROR
        @test stats["error_code"] == ERROR_TIMEOUT
        @test stats["error_count"] == 1
        @test stats["iteration"] == 500
        @test stats["total_iterations"] == 1000
        @test stats["progress"] ≈ 0.5
        @test stats["total_runtime"] > 0.0
        @test stats["checkpoints_saved"] == 0
        @test stats["last_checkpoint_iter"] == 0
        
        # Cleanup
        rm(manager.checkpoint_dir, recursive=true)
    end
    
    @testset "Error State Recovery" begin
        manager = StateManager(checkpoint_dir=mktempdir(), auto_recovery=false)
        
        initialize_kernel!(manager)
        
        # Trigger error
        set_error!(manager, ERROR_CUDA_ERROR, "GPU error")
        @test get_current_state(manager) == KERNEL_ERROR
        
        # Should be able to reinitialize from error
        @test try_transition!(manager, KERNEL_INITIALIZING)
        @test try_transition!(manager, KERNEL_RUNNING)
        @test get_current_state(manager) == KERNEL_RUNNING
        
        # Error should be cleared
        CUDA.@allowscalar @test manager.kernel_state.error_code[1] == Int32(ERROR_NONE)
        
        # Cleanup
        rm(manager.checkpoint_dir, recursive=true)
    end
    
    @testset "Checkpoint During Different States" begin
        manager = StateManager(checkpoint_dir=mktempdir(), auto_recovery=false)
        
        # Can't checkpoint from uninitialized
        @test !try_transition!(manager, KERNEL_CHECKPOINTING)
        
        # Initialize and run
        initialize_kernel!(manager)
        @test get_current_state(manager) == KERNEL_RUNNING
        
        # Can checkpoint from running
        @test perform_checkpoint(manager)
        
        # Pause and try checkpoint
        try_transition!(manager, KERNEL_PAUSED)
        @test !try_transition!(manager, KERNEL_CHECKPOINTING)  # Can't checkpoint from paused
        
        # Cleanup
        rm(manager.checkpoint_dir, recursive=true)
    end
end

println("\n✅ Kernel state management tests completed!")