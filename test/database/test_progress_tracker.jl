using Test
using Dates

# Include module
include("../../src/database/progress_tracker.jl")
using .ProgressTracker

@testset "Progress Tracker Tests" begin
    
    @testset "Basic Progress Tracking" begin
        # Create tracker
        tracker = ProgressTracker.ProgressTracker(10000, 10)
        
        @test tracker.total_rows == 10000
        @test tracker.total_chunks == 10
        @test tracker.rows_processed[] == 0
        @test tracker.chunks_processed[] == 0
        @test !tracker.finished
        @test !tracker.cancelled[]
        
        # Update progress
        ProgressTracker.update_progress!(tracker, 1000, 1024*1024)
        
        @test tracker.rows_processed[] == 1000
        @test tracker.bytes_processed[] == 1024*1024
        
        # Update with chunk completion
        ProgressTracker.update_progress!(tracker, 0, 0, chunk_completed=true)
        @test tracker.chunks_processed[] == 1
        
        # Finish progress
        info = ProgressTracker.finish_progress!(tracker)
        @test tracker.finished
        @test info.status == "completed"
    end
    
    @testset "Progress Calculation" begin
        tracker = ProgressTracker.ProgressTracker(5000, 5)
        
        # Process 40% of data
        ProgressTracker.update_progress!(tracker, 2000, 2*1024*1024, chunk_completed=true)
        sleep(0.1)  # Ensure time difference
        ProgressTracker.update_progress!(tracker, 0, 0, chunk_completed=true)
        
        # Get progress info
        info = ProgressTracker.create_progress_info(tracker, "Processing")
        
        @test info.rows_processed == 2000
        @test info.total_rows == 5000
        @test info.chunks_processed == 2
        @test info.total_chunks == 5
        @test isapprox(info.percentage, 40.0, atol=0.1)
        @test info.elapsed_seconds > 0
        @test info.status == "processing"
    end
    
    @testset "Throughput Calculation" begin
        tracker = ProgressTracker.ProgressTracker(100000, 10, update_interval=0.0)
        
        # Simulate processing with delays
        start_time = time()
        ProgressTracker.update_progress!(tracker, 10000, 10*1024*1024)
        sleep(0.5)
        ProgressTracker.update_progress!(tracker, 10000, 10*1024*1024)
        sleep(0.5)
        ProgressTracker.update_progress!(tracker, 10000, 10*1024*1024)
        
        # Calculate throughput
        rows_per_sec, mb_per_sec = ProgressTracker.calculate_throughput(tracker)
        
        # Should be roughly 30000 rows per second (30000 rows / ~1 second)
        @test 20000 < rows_per_sec < 40000
        @test mb_per_sec > 0
    end
    
    @testset "ETA Estimation" begin
        tracker = ProgressTracker.ProgressTracker(100000, 10)
        
        # Simulate steady progress
        elapsed = 10.0  # 10 seconds elapsed
        rows_processed = 25000  # 25% done
        rows_per_sec = 2500.0  # 2500 rows/sec
        
        eta = ProgressTracker.estimate_eta(tracker, rows_processed, elapsed, rows_per_sec)
        
        @test !isnothing(eta)
        # Should need 30 seconds more (75000 remaining / 2500 per sec)
        @test isapprox(eta, 30.0, atol=1.0)
        
        # Test edge cases
        eta_zero_speed = ProgressTracker.estimate_eta(tracker, 50000, 10.0, 0.0)
        @test isnothing(eta_zero_speed)
        
        eta_complete = ProgressTracker.estimate_eta(tracker, 100000, 10.0, 2500.0)
        @test isnothing(eta_complete)
    end
    
    @testset "Progress Formatting" begin
        # Test number formatting
        @test ProgressTracker.format_number(123) == "123"
        @test ProgressTracker.format_number(1234) == "1,234"
        @test ProgressTracker.format_number(1234567) == "1,234,567"
        
        # Test duration formatting
        @test ProgressTracker.format_duration(45.0) == "45s"
        @test ProgressTracker.format_duration(125.0) == "2m5s"
        @test ProgressTracker.format_duration(3665.0) == "1h1m"
        
        # Test progress formatting
        info = ProgressTracker.ProgressInfo(
            2500, 10000, 3, 10, 25.0, 10.0, 30.0,
            250.0, 2.5, 1.5, "processing", "Loading features"
        )
        
        formatted = ProgressTracker.format_progress(info, width=80)
        @test occursin("25.0%", formatted)
        @test occursin("2,500/10,000", formatted)
        @test occursin("Chunks: 3/10", formatted)
        @test occursin("ETA:", formatted)
        @test occursin("Loading features", formatted)
    end
    
    @testset "Progress Callbacks" begin
        # Test console callback
        updates_received = 0
        last_info = nothing
        
        callback = function(info)
            updates_received += 1
            last_info = info
        end
        
        tracker = ProgressTracker.ProgressTracker(
            1000, 10, 
            progress_callback=callback,
            update_interval=0.0  # Immediate updates
        )
        
        ProgressTracker.update_progress!(tracker, 100)
        ProgressTracker.update_progress!(tracker, 200)
        ProgressTracker.finish_progress!(tracker)
        
        @test updates_received >= 3
        @test !isnothing(last_info)
        @test last_info.rows_processed == 300
    end
    
    @testset "Cancellation" begin
        tracker = ProgressTracker.ProgressTracker(10000, 10)
        
        # Process some data
        ProgressTracker.update_progress!(tracker, 1000)
        
        # Cancel
        ProgressTracker.cancel_loading(tracker)
        @test tracker.cancelled[]
        
        # Further updates should be ignored
        ProgressTracker.update_progress!(tracker, 1000)
        @test tracker.rows_processed[] == 1000  # No change
        
        # Create info after cancellation
        info = ProgressTracker.create_progress_info(tracker, "")
        @test info.status == "cancelled"
    end
    
    @testset "Memory Monitoring" begin
        # Test with low memory limit
        tracker = ProgressTracker.ProgressTracker(
            10000, 10,
            memory_limit_gb=0.001  # Very low limit to trigger warning
        )
        
        # This should work without error even if memory exceeded
        ProgressTracker.update_progress!(tracker, 1000)
        
        info = ProgressTracker.create_progress_info(tracker, "")
        @test info.memory_used_gb >= 0  # Should have some value
    end
    
    @testset "Concurrent Updates" begin
        tracker = ProgressTracker.ProgressTracker(100000, 100)
        
        # Simulate concurrent updates
        tasks = []
        for i in 1:10
            task = Threads.@spawn begin
                for j in 1:10
                    ProgressTracker.update_progress!(tracker, 100, 1024*100)
                    sleep(0.01)
                end
            end
            push!(tasks, task)
        end
        
        # Wait for all tasks
        for task in tasks
            wait(task)
        end
        
        # Should have processed all updates
        @test tracker.rows_processed[] == 10000  # 10 tasks * 10 updates * 100 rows
        @test tracker.bytes_processed[] == 10*10*100*1024
    end
    
    @testset "File Progress Callback" begin
        # Create temporary file
        temp_file = tempname()
        
        file_callback = ProgressTracker.file_progress_callback(temp_file)
        
        tracker = ProgressTracker.ProgressTracker(
            1000, 10,
            progress_callback=file_callback,
            update_interval=0.0
        )
        
        ProgressTracker.update_progress!(tracker, 500, 0, message="Halfway")
        ProgressTracker.finish_progress!(tracker, success=true)
        
        # Check file contents
        contents = read(temp_file, String)
        @test occursin("50", contents)  # 50% or 500 rows
        @test occursin("Halfway", contents)
        @test occursin("completed", contents)
        
        # Cleanup
        rm(temp_file, force=true)
    end
    
    @testset "Combined Callbacks" begin
        call_counts = [0, 0]
        
        cb1 = info -> call_counts[1] += 1
        cb2 = info -> call_counts[2] += 1
        
        combined = ProgressTracker.combined_progress_callback([cb1, cb2])
        
        tracker = ProgressTracker.ProgressTracker(
            100, 1,
            progress_callback=combined,
            update_interval=0.0
        )
        
        ProgressTracker.update_progress!(tracker, 50)
        ProgressTracker.finish_progress!(tracker)
        
        @test call_counts[1] >= 2
        @test call_counts[2] >= 2
        @test call_counts[1] == call_counts[2]
    end
end