using Test
using Dates

# Include the module
include("../../src/ui/progress_tracker.jl")
using .ProgressTracker

@testset "Progress Tracker Tests" begin
    
    @testset "Basic Progress Tracker Creation" begin
        tracker = create_progress_tracker()
        @test isa(tracker, ProgressTracker.Tracker)
        @test tracker.state.overall_total == 0
        @test tracker.state.overall_completed == 0
        @test isempty(tracker.state.stages)
        @test tracker.state.current_stage_index == 0
        @test !tracker.state.paused
    end
    
    @testset "Stage Management" begin
        tracker = create_progress_tracker()
        
        # Add stages
        idx1 = add_stage!(tracker, "Stage 1", 100)
        @test idx1 == 1
        @test length(tracker.state.stages) == 1
        @test tracker.state.overall_total == 100
        @test tracker.state.current_stage_index == 1
        
        idx2 = add_stage!(tracker, "Stage 2", 200)
        @test idx2 == 2
        @test length(tracker.state.stages) == 2
        @test tracker.state.overall_total == 300
        
        # Get current stage
        stage_info = get_current_stage(tracker)
        @test stage_info.name == "Stage 1"
        @test stage_info.total == 100
        @test stage_info.completed == 0
        @test stage_info.progress == 0.0
    end
    
    @testset "Progress Updates" begin
        tracker = create_progress_tracker()
        add_stage!(tracker, "Test Stage", 100)
        
        # Update progress
        update_progress!(tracker, 10)
        @test tracker.state.overall_completed == 10
        @test tracker.state.stages[1].completed_items == 10
        
        # Multiple updates
        update_progress!(tracker, 15)
        @test tracker.state.overall_completed == 25
        @test tracker.state.stages[1].completed_items == 25
        
        # Check percentage
        @test get_progress_percentage(tracker) == 25.0
    end
    
    @testset "MCTS Phase Tracking" begin
        tracker = create_progress_tracker()
        add_stage!(tracker, "MCTS Stage", 100)
        
        # Update with different phases
        update_progress!(tracker, 5, phase = ProgressTracker.MCTS_SELECTION)
        stage_info = get_current_stage(tracker)
        @test stage_info.phase == ProgressTracker.MCTS_SELECTION
        
        update_progress!(tracker, 5, phase = ProgressTracker.MCTS_EXPANSION)
        stage_info = get_current_stage(tracker)
        @test stage_info.phase == ProgressTracker.MCTS_EXPANSION
        
        # Test phase names and symbols
        @test get_phase_name(ProgressTracker.MCTS_SELECTION) == "Selection"
        @test get_phase_name(ProgressTracker.MCTS_EXPANSION) == "Expansion"
        @test get_phase_name(ProgressTracker.MCTS_SIMULATION) == "Simulation"
        @test get_phase_name(ProgressTracker.MCTS_BACKPROPAGATION) == "Backpropagation"
        @test get_phase_name(ProgressTracker.MCTS_IDLE) == "Idle"
        
        @test get_phase_symbol(ProgressTracker.MCTS_SELECTION) == "◐"
        @test get_phase_symbol(ProgressTracker.MCTS_EXPANSION) == "◑"
        @test get_phase_symbol(ProgressTracker.MCTS_SIMULATION) == "◒"
        @test get_phase_symbol(ProgressTracker.MCTS_BACKPROPAGATION) == "◓"
        @test get_phase_symbol(ProgressTracker.MCTS_IDLE) == "○"
    end
    
    @testset "ETA Calculations" begin
        tracker = create_progress_tracker()
        add_stage!(tracker, "ETA Test", 1000)
        
        # No ETA without progress
        eta = get_eta(tracker)
        @test isnothing(eta)
        
        # Simulate progress over time
        sleep(0.1)
        update_progress!(tracker, 100)
        sleep(0.1)
        update_progress!(tracker, 100)
        
        # Should have ETA now
        eta = get_eta(tracker)
        @test !isnothing(eta)
        @test isa(eta, Millisecond)
        
        # Test formatted ETA
        formatted = get_formatted_eta(tracker)
        @test isa(formatted, String)
        @test formatted != "Unknown"
        
        # Complete all items
        update_progress!(tracker, 800)
        eta = get_eta(tracker)
        @test eta == Millisecond(0)
        @test get_formatted_eta(tracker) == "Complete"
    end
    
    @testset "Speed Tracking" begin
        tracker = create_progress_tracker()
        add_stage!(tracker, "Speed Test", 1000)
        
        # Initial speed should be 0
        speed_stats = get_speed_stats(tracker)
        @test speed_stats.current == 0.0
        @test speed_stats.average == 0.0
        
        # Add some progress
        for i in 1:10
            update_progress!(tracker, 10)
            sleep(0.01)
        end
        
        speed_stats = get_speed_stats(tracker)
        @test speed_stats.current > 0
        @test speed_stats.average > 0
        @test speed_stats.min >= 0
        @test speed_stats.max >= speed_stats.min
    end
    
    @testset "Pause and Resume" begin
        tracker = create_progress_tracker()
        add_stage!(tracker, "Pause Test", 100)
        
        # Initial state
        @test !is_paused(tracker)
        
        # Pause
        pause_tracker!(tracker)
        @test is_paused(tracker)
        @test !isnothing(tracker.state.pause_start)
        
        # Progress updates should be ignored while paused
        update_progress!(tracker, 10)
        @test tracker.state.overall_completed == 0
        
        # Resume
        sleep(0.1)  # Ensure some pause duration
        resume_tracker!(tracker)
        @test !is_paused(tracker)
        @test isnothing(tracker.state.pause_start)
        @test tracker.state.total_pause_duration.value > 0
        
        # Progress updates should work again
        update_progress!(tracker, 10)
        @test tracker.state.overall_completed == 10
    end
    
    @testset "Progress History" begin
        tracker = create_progress_tracker()
        add_stage!(tracker, "History Test", 100)
        
        # Generate some history
        for i in 1:20
            update_progress!(tracker, 5)
            sleep(0.01)
        end
        
        # Get full history
        history = get_history(tracker)
        @test length(history.timestamps) == 20
        @test length(history.percentages) == 20
        @test length(history.speeds) == 20
        @test all(p -> 0 <= p <= 100, history.percentages)
        
        # Get last N entries
        recent_history = get_history(tracker, last_n = 5)
        @test length(recent_history.timestamps) == 5
        @test recent_history.timestamps == history.timestamps[end-4:end]
    end
    
    @testset "Stage Completion" begin
        tracker = create_progress_tracker()
        add_stage!(tracker, "Stage 1", 50)
        add_stage!(tracker, "Stage 2", 50)
        
        # Complete first stage
        update_progress!(tracker, 50)
        complete_stage!(tracker)
        
        @test tracker.state.current_stage_index == 2
        stage_info = get_current_stage(tracker)
        @test stage_info.name == "Stage 2"
        @test stage_info.completed == 0
        
        # Complete second stage
        update_progress!(tracker, 50)
        complete_stage!(tracker)
        
        # Should stay on last stage
        @test tracker.state.current_stage_index == 2
    end
    
    @testset "Progress Bar Generation" begin
        tracker = create_progress_tracker()
        add_stage!(tracker, "Bar Test", 100)
        
        # Empty bar
        bar = get_progress_bar(tracker, width = 10)
        @test occursin("[░░░░░░░░░░]", bar)
        @test occursin("0.0%", bar)
        
        # Half full
        update_progress!(tracker, 50)
        bar = get_progress_bar(tracker, width = 10)
        @test occursin("[█████░░░░░]", bar)
        @test occursin("50.0%", bar)
        
        # Full bar
        update_progress!(tracker, 50)
        bar = get_progress_bar(tracker, width = 10)
        @test occursin("[██████████]", bar)
        @test occursin("100.0%", bar)
        
        # Custom characters
        bar = get_progress_bar(tracker, width = 10, filled_char = '#', empty_char = '-')
        @test occursin("[##########]", bar)
    end
    
    @testset "Stage Indicator" begin
        tracker = create_progress_tracker()
        
        # No active stage
        indicator = get_stage_indicator(tracker)
        @test indicator == "No active stage"
        
        # With stage
        add_stage!(tracker, "Test Stage", 100)
        update_progress!(tracker, 25, phase = ProgressTracker.MCTS_SIMULATION)
        
        indicator = get_stage_indicator(tracker)
        @test occursin("Test Stage", indicator)
        @test occursin("Simulation", indicator)
        @test occursin("25.0%", indicator)
        @test occursin("◒", indicator)  # Simulation symbol
    end
    
    @testset "Set Total Items" begin
        tracker = create_progress_tracker()
        add_stage!(tracker, "Dynamic Stage")
        
        # Initially no total
        @test tracker.state.stages[1].total_items == 0
        @test tracker.state.overall_total == 0
        
        # Set total
        set_total_items!(tracker, 100)
        @test tracker.state.stages[1].total_items == 100
        @test tracker.state.overall_total == 100
        
        # Update total
        set_total_items!(tracker, 150)
        @test tracker.state.stages[1].total_items == 150
        @test tracker.state.overall_total == 150
    end
    
    @testset "Reset Tracker" begin
        tracker = create_progress_tracker()
        add_stage!(tracker, "Stage 1", 100)
        add_stage!(tracker, "Stage 2", 100)
        
        # Make some progress
        update_progress!(tracker, 50)
        complete_stage!(tracker)
        update_progress!(tracker, 25)
        
        # Reset
        reset_tracker!(tracker)
        
        @test tracker.state.overall_completed == 0
        @test tracker.state.current_stage_index == 1
        @test all(s -> s.completed_items == 0, tracker.state.stages)
        @test all(s -> s.phase == ProgressTracker.MCTS_IDLE, tracker.state.stages)
        @test isempty(tracker.state.history.timestamps)
    end
    
    @testset "Update Callback" begin
        callback_count = Ref(0)
        callback_tracker = Ref{Any}(nothing)
        
        function test_callback(tracker)
            callback_count[] += 1
            callback_tracker[] = tracker
        end
        
        tracker = create_progress_tracker(update_callback = test_callback)
        add_stage!(tracker, "Callback Test", 100)
        
        # Update should trigger callback
        update_progress!(tracker, 10)
        @test callback_count[] == 1
        @test callback_tracker[] === tracker
        
        # Multiple updates
        update_progress!(tracker, 10)
        update_progress!(tracker, 10)
        @test callback_count[] == 3
    end
    
    @testset "Progress Persistence" begin
        # Create temporary file
        temp_file = tempname()
        
        try
            # Create tracker with persistence
            tracker = create_progress_tracker(persistence_file = temp_file)
            add_stage!(tracker, "Persistent Stage 1", 100)
            add_stage!(tracker, "Persistent Stage 2", 200)
            update_progress!(tracker, 50, phase = ProgressTracker.MCTS_EXPANSION)
            complete_stage!(tracker)
            update_progress!(tracker, 75)
            
            # Save explicitly
            save_progress(tracker)
            @test isfile(temp_file)
            
            # Create new tracker and load
            new_tracker = create_progress_tracker()
            @test load_progress(new_tracker, temp_file)
            
            # Verify loaded state
            @test length(new_tracker.state.stages) == 2
            @test new_tracker.state.stages[1].name == "Persistent Stage 1"
            @test new_tracker.state.stages[1].completed_items == 100  # complete_stage! sets to total
            @test new_tracker.state.stages[2].completed_items == 75
            @test new_tracker.state.current_stage_index == 2
            @test new_tracker.state.overall_total == 300
            @test new_tracker.state.overall_completed == 175  # 100 + 75
        finally
            # Cleanup
            isfile(temp_file) && rm(temp_file)
        end
    end
    
    @testset "Edge Cases" begin
        tracker = create_progress_tracker()
        
        # Update without stages
        update_progress!(tracker, 10)
        @test tracker.state.overall_completed == 0
        
        # Complete stage without stages
        complete_stage!(tracker)
        @test tracker.state.current_stage_index == 0
        
        # Set total without stages
        set_total_items!(tracker, 100)
        @test tracker.state.overall_total == 0
        
        # Zero total items
        add_stage!(tracker, "Zero Stage", 0)
        @test get_progress_percentage(tracker) == 0.0
        
        # Negative items (should be handled gracefully)
        update_progress!(tracker, -5)
        @test tracker.state.overall_completed == -5
    end
end

println("All progress tracker tests passed! ✓")