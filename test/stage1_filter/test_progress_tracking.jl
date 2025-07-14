using Test
using CUDA
using Dates
using Printf

# Include the progress tracking module
include("../../src/stage1_filter/progress_tracking.jl")

using .ProgressTracking

@testset "Progress Tracking Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU progress tracking tests"
        return
    end
    
    @testset "ProgressConfig Creation" begin
        config = create_progress_config()
        
        @test config.update_interval_ms == 100
        @test config.callback_enabled == true
        @test config.persist_to_disk == false
        @test config.monitor_memory == true
        @test config.show_eta == true
        @test config.progress_file == ".progress.json"
        
        # Test with custom parameters
        config2 = create_progress_config(
            update_interval_ms=500,
            persist_to_disk=true,
            monitor_memory=false,
            progress_file="custom_progress.json"
        )
        
        @test config2.update_interval_ms == 500
        @test config2.persist_to_disk == true
        @test config2.monitor_memory == false
        @test config2.progress_file == "custom_progress.json"
    end
    
    @testset "GPU Progress Counters" begin
        counters = create_gpu_counters()
        
        # Test initial values
        @test CUDA.@allowscalar counters.features_processed[1] == 0
        @test CUDA.@allowscalar counters.samples_processed[1] == 0
        @test CUDA.@allowscalar counters.operations_completed[1] == 0
        @test CUDA.@allowscalar counters.error_count[1] == 0
        
        # Test increment kernel
        @cuda threads=1 blocks=1 increment_progress_kernel!(
            counters.features_processed,
            Int64(10)
        )
        CUDA.synchronize()
        
        @test CUDA.@allowscalar counters.features_processed[1] == 10
        
        # Test reset
        reset_counters!(counters)
        @test CUDA.@allowscalar counters.features_processed[1] == 0
    end
    
    @testset "Batch Progress Update" begin
        counters = create_gpu_counters()
        
        n_blocks = 10
        feature_increments = CUDA.fill(Int32(5), n_blocks)
        
        threads = 32  # Small for testing
        shmem_size = 2 * 32 * sizeof(Int64)  # Adjust for warp size
        
        @cuda threads=threads blocks=1 shmem=shmem_size batch_progress_update_kernel!(
            counters.features_processed,
            counters.samples_processed,
            feature_increments,
            Int32(n_blocks)
        )
        CUDA.synchronize()
        
        # Check that features were incremented
        features_count = CUDA.@allowscalar counters.features_processed[1]
        @test features_count == 5  # First block's increment
        
        samples_count = CUDA.@allowscalar counters.samples_processed[1]
        @test samples_count == 1000  # Example value from kernel
    end
    
    @testset "Progress State Management" begin
        # Create initial state
        state = ProgressState(
            1000,      # total_work
            0,         # completed_work
            now(),     # start_time
            now(),     # last_update_time
            "Testing", # current_phase
            0.0,       # throughput
            0,         # gpu_memory_used
            0,         # gpu_memory_total
            false      # is_cancelled
        )
        
        @test state.total_work == 1000
        @test state.completed_work == 0
        @test state.current_phase == "Testing"
        @test state.is_cancelled == false
        
        # Update state
        counters = create_gpu_counters()
        CUDA.@allowscalar counters.features_processed[1] = 100
        
        update_progress_state!(state, counters, "Processing")
        
        @test state.completed_work == 100
        @test state.current_phase == "Processing"
    end
    
    @testset "Progress Tracker Creation" begin
        config = create_progress_config(update_interval_ms=50)
        
        # Custom callback that counts calls
        callback_count = Ref(0)
        test_callback = function(state::ProgressState)
            callback_count[] += 1
        end
        
        tracker = create_progress_tracker(
            1000,
            config,
            callback=test_callback,
            phase="Test Phase"
        )
        
        @test tracker.state.total_work == 1000
        @test tracker.state.current_phase == "Test Phase"
        @test tracker.active == true
        @test tracker.config.update_interval_ms == 50
        
        # Test that callback is not called immediately
        @test callback_count[] == 0
    end
    
    @testset "Progress Updates and Callbacks" begin
        config = create_progress_config(update_interval_ms=10)
        
        # Track callback data
        callback_data = []
        test_callback = function(state::ProgressState)
            push!(callback_data, (
                completed=state.completed_work,
                phase=state.current_phase,
                time=now()
            ))
        end
        
        tracker = create_progress_tracker(
            100,
            config,
            callback=test_callback,
            phase="Processing"
        )
        
        # Simulate progress
        for i in 1:5
            CUDA.@allowscalar tracker.counters.features_processed[1] = i * 20
            check_progress_update!(tracker)
            sleep(0.02)  # Wait longer than update interval
        end
        
        # Should have multiple callbacks
        @test length(callback_data) >= 3
        
        # Check that progress increased
        if length(callback_data) >= 2
            @test callback_data[2].completed > callback_data[1].completed
        end
        
        # Complete the progress
        complete_progress!(tracker)
        @test tracker.state.completed_work == tracker.state.total_work
        @test !tracker.active
    end
    
    @testset "Duration Formatting" begin
        @test ProgressTracking.format_duration(45.0) == "45s"
        @test ProgressTracking.format_duration(125.0) == "2m 5s"
        @test ProgressTracking.format_duration(3665.0) == "1h 1m"
        @test ProgressTracking.format_duration(7325.0) == "2h 2m"
    end
    
    @testset "Progress Cancellation" begin
        config = create_progress_config()
        tracker = create_progress_tracker(1000, config)
        
        @test tracker.active == true
        @test !tracker.state.is_cancelled
        
        cancel_progress!(tracker)
        
        @test !tracker.active
        @test tracker.state.is_cancelled
    end
    
    @testset "Console Progress Display" begin
        # Test console callback formatting
        config = create_progress_config()
        
        output_buffer = IOBuffer()
        console_callback = function(state::ProgressState)
            percentage = state.completed_work / state.total_work * 100
            write(output_buffer, "Progress: $(round(percentage, digits=1))%")
        end
        
        tracker = create_progress_tracker(
            1000,
            config,
            callback=console_callback
        )
        
        # Update progress
        CUDA.@allowscalar tracker.counters.features_processed[1] = 250
        check_progress_update!(tracker)
        
        output = String(take!(output_buffer))
        @test contains(output, "Progress: 25.0%")
    end
    
    @testset "GPU Memory Monitoring" begin
        config = create_progress_config(monitor_memory=true)
        tracker = create_progress_tracker(1000, config)
        
        # Memory info should be populated
        @test tracker.state.gpu_memory_total > 0
        @test tracker.state.gpu_memory_used >= 0
        @test tracker.state.gpu_memory_used <= tracker.state.gpu_memory_total
        
        # Allocate some GPU memory
        temp_array = CUDA.zeros(Float32, 1000, 1000)
        
        # Update and check memory changed
        initial_memory = tracker.state.gpu_memory_used
        update_progress_state!(tracker.state, tracker.counters)
        
        # Memory usage should have increased (though might not be exact due to pooling)
        @test tracker.state.gpu_memory_used >= initial_memory
    end
    
    @testset "Progress Persistence" begin
        # Create temporary progress file
        progress_file = tempname()
        
        config = create_progress_config(
            persist_to_disk=true,
            progress_file=progress_file
        )
        
        tracker = create_progress_tracker(1000, config, phase="Test Save")
        
        # Update progress
        CUDA.@allowscalar tracker.counters.features_processed[1] = 500
        tracker.state.completed_work = 500
        
        # Save progress
        save_progress(tracker)
        
        @test isfile(progress_file)
        
        # Load progress
        loaded_data = load_progress(config)
        
        @test loaded_data !== nothing
        @test haskey(loaded_data, "total_work")
        @test haskey(loaded_data, "completed_work")
        @test loaded_data["total_work"] == "1000"
        @test loaded_data["completed_work"] == "500"
        @test loaded_data["current_phase"] == "Test Save"
        
        # Clean up
        rm(progress_file, force=true)
    end
    
    @testset "Progress Kernel Macro" begin
        config = create_progress_config(update_interval_ms=10)
        
        callback_count = Ref(0)
        test_callback = function(state::ProgressState)
            callback_count[] += 1
        end
        
        tracker = create_progress_tracker(
            100,
            config,
            callback=test_callback
        )
        
        # Use the macro
        feature_increments = CUDA.ones(Int32, 10)
        
        @progress_kernel tracker begin
            @cuda threads=32 blocks=1 shmem=64*sizeof(Int64) batch_progress_update_kernel!(
                tracker.counters.features_processed,
                tracker.counters.samples_processed,
                feature_increments,
                Int32(1)
            )
        end
        
        # Wait for potential update
        sleep(0.02)
        check_progress_update!(tracker)
        
        # Should have recorded event
        @test isdefined(tracker, :update_event)
    end
    
    @testset "Throughput Calculation" begin
        config = create_progress_config()
        tracker = create_progress_tracker(10000, config)
        
        # Simulate work over time
        start_time = now()
        CUDA.@allowscalar tracker.counters.features_processed[1] = 1000
        update_progress_state!(tracker.state, tracker.counters)
        
        sleep(0.1)  # 100ms
        
        CUDA.@allowscalar tracker.counters.features_processed[1] = 2000
        update_progress_state!(tracker.state, tracker.counters)
        
        # Throughput should be positive (approximately 10000 items/sec)
        @test tracker.state.throughput > 0
        
        # Test ETA calculation in callback
        elapsed = Dates.value(now() - start_time) / 1000.0
        if elapsed > 0 && tracker.state.completed_work > 0
            rate = tracker.state.completed_work / elapsed
            remaining = tracker.state.total_work - tracker.state.completed_work
            eta_seconds = remaining / rate
            @test eta_seconds > 0
        end
    end
end

println("\n✅ Progress tracking tests completed!")