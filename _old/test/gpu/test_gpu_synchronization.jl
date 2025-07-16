using Test
using Dates
using Base.Threads

# Include required modules
include("../../src/gpu/GPU.jl")
using .GPU

# Include synchronization module
include("../../src/gpu/gpu_synchronization.jl")
using .GPUSynchronization

@testset "GPU Synchronization Tests" begin
    
    @testset "SyncManager Creation" begin
        # Test with auto-detection
        manager = create_sync_manager()
        @test isa(manager, SyncManager)
        @test manager.timeout_ms == 30000
        
        # Test with explicit GPU count
        manager2 = create_sync_manager(num_gpus=2, timeout_ms=5000)
        @test length(manager2.state.gpu_states) == 2
        @test manager2.timeout_ms == 5000
        @test haskey(manager2.barriers, :compute)
        @test haskey(manager2.barriers, :sync)
        
        # Test with no GPUs
        manager3 = create_sync_manager(num_gpus=0)
        @test length(manager3.state.gpu_states) == 0
        @test isempty(manager3.barriers)
    end
    
    @testset "GPU Registration" begin
        manager = create_sync_manager(num_gpus=0)
        
        # Register GPUs
        register_gpu!(manager, 0)
        register_gpu!(manager, 1)
        
        @test length(get_active_gpus(manager)) == 2
        @test 0 in get_active_gpus(manager)
        @test 1 in get_active_gpus(manager)
        
        # Unregister GPU
        unregister_gpu!(manager, 1)
        @test length(get_active_gpus(manager)) == 1
        @test !(1 in get_active_gpus(manager))
    end
    
    @testset "Phase Management" begin
        manager = create_sync_manager(num_gpus=2)
        
        # Initial state
        @test manager.state.global_phase == PHASE_INIT
        
        # Set phase for one GPU
        set_phase!(manager, 0, PHASE_READY)
        @test manager.state.gpu_states[0].current_phase == PHASE_READY
        @test manager.state.global_phase == PHASE_INIT  # Not all GPUs ready
        
        # Set phase for second GPU
        set_phase!(manager, 1, PHASE_READY)
        @test manager.state.global_phase == PHASE_READY  # All GPUs ready
        
        # Test error phase
        set_phase!(manager, 0, PHASE_ERROR, error_msg="Test error")
        @test manager.state.gpu_states[0].current_phase == PHASE_ERROR
        @test manager.state.gpu_states[0].error_message == "Test error"
        @test !manager.state.gpu_states[0].is_active
    end
    
    @testset "Phase Waiting" begin
        manager = create_sync_manager(num_gpus=2)
        
        # Test immediate return when phase already reached
        set_phase!(manager, 0, PHASE_READY)
        set_phase!(manager, 1, PHASE_READY)
        
        @test wait_for_phase(manager, PHASE_READY, timeout_ms=100)
        
        # Test timeout
        @test !wait_for_phase(manager, PHASE_DONE, timeout_ms=100)
        
        # Test phase completion check
        @test is_phase_complete(manager, PHASE_INIT)
        @test is_phase_complete(manager, PHASE_READY)
        @test !is_phase_complete(manager, PHASE_RUNNING)
    end
    
    @testset "Event Synchronization" begin
        event = SyncEvent()
        
        # Initially not set
        @test !event.is_set
        
        # Test immediate timeout when not set
        @test !wait_for_event(event, timeout_ms=50)
        
        # Signal event
        signal_event!(event)
        @test event.is_set
        @test wait_for_event(event, timeout_ms=50)
        
        # Reset event
        reset_event!(event)
        @test !event.is_set
        
        # Test concurrent signaling
        @async begin
            sleep(0.1)
            signal_event!(event)
        end
        
        @test wait_for_event(event, timeout_ms=500)
    end
    
    @testset "Barrier Synchronization" begin
        barrier = SyncBarrier(2)
        
        @test barrier.required_count == 2
        @test barrier.current_count == 0
        @test barrier.generation == 0
        
        # Test single thread timeout
        @test !enter_barrier!(barrier, timeout_ms=50)
        
        # Test successful barrier synchronization
        barrier = SyncBarrier(2)
        results = Channel{Bool}(2)
        
        @async put!(results, enter_barrier!(barrier, timeout_ms=1000))
        @async put!(results, enter_barrier!(barrier, timeout_ms=1000))
        
        @test take!(results) == true
        @test take!(results) == true
        @test barrier.generation == 1
        @test barrier.current_count == 0
        
        # Test barrier reset
        reset_barrier!(barrier)
        @test barrier.generation == 2
        @test barrier.current_count == 0
    end
    
    @testset "Lock Operations" begin
        lock = ReentrantLock()
        
        # Test successful lock acquisition
        @test acquire_lock!(lock, timeout_ms=100)
        release_lock!(lock)
        
        # Test with_lock
        result = with_lock(lock, timeout_ms=100) do
            return 42
        end
        @test result == 42
        
        # Test nested locking (reentrant)
        with_lock(lock) do
            with_lock(lock) do
                @test true  # Should work with reentrant lock
            end
        end
        
        # Test timeout
        acquire_lock!(lock)
        @async begin
            sleep(0.2)
            release_lock!(lock)
        end
        
        # Should timeout before lock is released
        @test !acquire_lock!(lock, timeout_ms=50)
        
        # Should succeed after lock is released
        sleep(0.3)
        @test acquire_lock!(lock, timeout_ms=50)
        release_lock!(lock)
    end
    
    @testset "Result Storage" begin
        manager = create_sync_manager(num_gpus=2)
        
        # Store results
        set_gpu_result!(manager, 0, [1, 2, 3])
        set_gpu_result!(manager, 1, [4, 5, 6])
        
        # Retrieve results
        @test get_gpu_result(manager, 0) == [1, 2, 3]
        @test get_gpu_result(manager, 1) == [4, 5, 6]
        @test isnothing(get_gpu_result(manager, 2))
        
        # Get all results
        all_results = get_all_results(manager)
        @test length(all_results) == 2
        @test all_results[0] == [1, 2, 3]
        @test all_results[1] == [4, 5, 6]
        
        # Clear results
        clear_results!(manager)
        @test isnothing(get_gpu_result(manager, 0))
        @test isnothing(get_gpu_result(manager, 1))
    end
    
    @testset "State Information" begin
        manager = create_sync_manager(num_gpus=2)
        
        # Get initial state
        state = get_sync_state(manager)
        @test state["global_phase"] == "PHASE_INIT"
        @test state["sync_count"] == 0
        @test haskey(state, "gpu_states")
        @test haskey(state, "barriers")
        
        # Update state
        set_phase!(manager, 0, PHASE_RUNNING)
        update_sync_stats!(manager)
        
        state = get_sync_state(manager)
        @test state["sync_count"] == 1
        @test state["gpu_states"]["GPU0"]["phase"] == "PHASE_RUNNING"
        @test state["gpu_states"]["GPU0"]["active"] == true
    end
    
    @testset "Multi-GPU Workflow" begin
        manager = create_sync_manager(num_gpus=3)
        
        # Simulate GPU workflow
        gpu_tasks = []
        
        for gpu_id in 0:2
            task = @async begin
                # Phase 1: Initialization
                set_phase!(manager, gpu_id, PHASE_READY)
                wait_for_phase(manager, PHASE_READY)
                
                # Phase 2: Computation
                set_phase!(manager, gpu_id, PHASE_RUNNING)
                
                # Simulate work
                result = gpu_id * 100 + rand(1:10)
                sleep(0.1 * (gpu_id + 1))  # Different work times
                
                # Store result
                set_gpu_result!(manager, gpu_id, result)
                
                # Enter sync barrier
                enter_barrier!(manager.barriers[:compute])
                
                # Phase 3: Sync complete
                set_phase!(manager, gpu_id, PHASE_DONE)
                
                return result
            end
            push!(gpu_tasks, task)
        end
        
        # Wait for all tasks
        results = [fetch(task) for task in gpu_tasks]
        
        # Verify all completed
        @test wait_for_phase(manager, PHASE_DONE, timeout_ms=1000)
        @test length(results) == 3
        
        # Verify stored results
        for gpu_id in 0:2
            stored_result = get_gpu_result(manager, gpu_id)
            @test stored_result in results
        end
    end
    
    @testset "Error Handling" begin
        manager = create_sync_manager(num_gpus=2)
        
        # Simulate GPU error
        set_phase!(manager, 0, PHASE_RUNNING)
        set_phase!(manager, 1, PHASE_ERROR, error_msg="GPU memory error")
        
        # Check error state
        state = get_sync_state(manager)
        @test state["gpu_states"]["GPU1"]["phase"] == "PHASE_ERROR"
        @test state["gpu_states"]["GPU1"]["active"] == false
        @test state["gpu_states"]["GPU1"]["error"] == "GPU memory error"
        
        # Only active GPU should be considered
        active_gpus = get_active_gpus(manager)
        @test length(active_gpus) == 1
        @test 0 in active_gpus
        @test !(1 in active_gpus)
    end
    
    @testset "Timeout Handling" begin
        manager = create_sync_manager(num_gpus=2, timeout_ms=200)
        
        # Test phase timeout
        set_phase!(manager, 0, PHASE_READY)
        # GPU 1 never sets phase
        
        start_time = time()
        @test !wait_for_phase(manager, PHASE_READY)
        elapsed = time() - start_time
        
        # Should timeout around 200ms
        @test elapsed >= 0.2
        @test elapsed < 0.3
        
        # Test barrier timeout with mixed results
        barrier = SyncBarrier(3)
        results = Channel{Bool}(3)
        
        # Two threads enter quickly
        @async put!(results, enter_barrier!(barrier, timeout_ms=1000))
        @async put!(results, enter_barrier!(barrier, timeout_ms=1000))
        
        # Third thread enters late (after others timeout)
        @async begin
            sleep(1.5)
            put!(results, enter_barrier!(barrier, timeout_ms=100))
        end
        
        # First two should timeout
        @test take!(results) == false
        @test take!(results) == false
        @test take!(results) == false  # Third also fails
    end
    
end

# Print summary
println("\nGPU Synchronization Test Summary:")
println("=================================")
println("✓ All synchronization primitive tests completed")
println("✓ Multi-GPU coordination tested")
println("✓ Error handling validated")
println("✓ Timeout mechanisms verified")
println("\nSynchronization system ready for GPU coordination!")