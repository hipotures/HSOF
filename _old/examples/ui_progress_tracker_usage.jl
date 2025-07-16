# Example usage of Progress Tracker System

using Dates
using Printf

# Include the module
include("../src/ui/progress_tracker.jl")
using .ProgressTracker

"""
Basic progress tracking demo
"""
function basic_progress_demo()
    println("Basic Progress Tracking Demo")
    println("===========================")
    
    # Create progress tracker
    tracker = create_progress_tracker()
    
    # Add stages
    add_stage!(tracker, "Data Loading", 1000)
    add_stage!(tracker, "Feature Extraction", 2000)
    add_stage!(tracker, "Model Training", 500)
    
    println("\nCreated progress tracker with 3 stages:")
    println("  Total items to process: $(tracker.state.overall_total)")
    
    # Simulate progress
    println("\n[Stage 1: Data Loading]")
    for i in 1:10
        update_progress!(tracker, 100)
        bar = get_progress_bar(tracker, width=40)
        print("\r$bar")
        flush(stdout)
        sleep(0.1)
    end
    println("\n  Stage 1 complete!")
    
    # Move to next stage
    complete_stage!(tracker)
    
    println("\n[Stage 2: Feature Extraction]")
    for i in 1:20
        update_progress!(tracker, 100)
        bar = get_progress_bar(tracker, width=40)
        eta = get_formatted_eta(tracker)
        print("\r$bar  ETA: $eta  ")
        flush(stdout)
        sleep(0.05)
    end
    println("\n  Stage 2 complete!")
    
    # Show final statistics
    println("\nFinal Statistics:")
    stats = get_speed_stats(tracker)
    println("  Average speed: $(round(stats.average, digits=1)) items/sec")
    println("  Overall progress: $(round(get_progress_percentage(tracker), digits=1))%")
end

"""
MCTS phase tracking demo
"""
function mcts_phase_demo()
    println("\n\nMCTS Phase Tracking Demo")
    println("========================")
    
    tracker = create_progress_tracker()
    add_stage!(tracker, "MCTS Search", 100)
    
    # Simulate MCTS phases
    phases = [
        (ProgressTracker.MCTS_SELECTION, "Selecting nodes", 10),
        (ProgressTracker.MCTS_EXPANSION, "Expanding tree", 20),
        (ProgressTracker.MCTS_SIMULATION, "Running simulations", 50),
        (ProgressTracker.MCTS_BACKPROPAGATION, "Backpropagating results", 20)
    ]
    
    for (phase, description, items) in phases
        println("\n$description...")
        for i in 1:items
            update_progress!(tracker, 1, phase=phase)
            indicator = get_stage_indicator(tracker)
            print("\r  $indicator")
            flush(stdout)
            sleep(0.05)
        end
    end
    
    println("\n\nMCTS search complete!")
end

"""
Multi-stage pipeline demo
"""
function pipeline_demo()
    println("\n\nMulti-Stage Pipeline Demo")
    println("=========================")
    
    tracker = create_progress_tracker()
    
    # Define pipeline stages
    stages = [
        ("Stage 1: Initialization", 50),
        ("Stage 2: Preprocessing", 200),
        ("Stage 3: Feature Selection", 500),
        ("Stage 4: Model Training", 300),
        ("Stage 5: Validation", 100)
    ]
    
    # Add all stages
    for (name, items) in stages
        add_stage!(tracker, name, items)
    end
    
    println("Pipeline with $(length(stages)) stages")
    println("Total work items: $(tracker.state.overall_total)")
    
    # Process pipeline
    for (idx, (name, total_items)) in enumerate(stages)
        println("\n$name")
        
        # Process in chunks
        chunk_size = div(total_items, 10)
        for chunk in 1:10
            update_progress!(tracker, chunk_size)
            
            # Show current status
            stage_info = get_current_stage(tracker)
            overall_pct = round(get_progress_percentage(tracker), digits=1)
            stage_pct = round(stage_info.progress * 100, digits=1)
            
            print("\r  Stage: $stage_pct% | Overall: $overall_pct%")
            flush(stdout)
            sleep(0.1)
        end
        
        # Complete stage and move to next
        if idx < length(stages)
            complete_stage!(tracker)
        end
    end
    
    println("\n\nPipeline complete!")
end

"""
Pause and resume demo
"""
function pause_resume_demo()
    println("\n\nPause and Resume Demo")
    println("====================")
    
    tracker = create_progress_tracker()
    add_stage!(tracker, "Interruptible Task", 100)
    
    println("Processing with pause/resume capability...")
    
    # Process first part
    for i in 1:30
        update_progress!(tracker, 1)
        if i == 30
            println("\n[PAUSE] Pausing at 30%...")
            pause_tracker!(tracker)
            sleep(2.0)  # Simulate pause
            println("[RESUME] Resuming...")
            resume_tracker!(tracker)
        end
    end
    
    # Continue processing
    for i in 31:100
        update_progress!(tracker, 1)
    end
    
    println("\nTask complete!")
    println("Total pause duration: $(tracker.state.total_pause_duration)")
end

"""
Progress persistence demo
"""
function persistence_demo()
    println("\n\nProgress Persistence Demo")
    println("========================")
    
    temp_file = tempname() * "_progress.dat"
    
    println("Creating tracker with persistence to: $temp_file")
    
    # Create and partially complete
    tracker = create_progress_tracker(persistence_file = temp_file)
    add_stage!(tracker, "Persistent Task 1", 100)
    add_stage!(tracker, "Persistent Task 2", 200)
    
    println("\nProcessing first 50 items...")
    update_progress!(tracker, 50)
    save_progress(tracker)
    
    println("Progress saved. Simulating interruption...")
    
    # Create new tracker and restore
    new_tracker = create_progress_tracker(persistence_file = temp_file)
    println("\nRestoring progress from file...")
    
    if isfile(temp_file)
        load_progress(new_tracker, temp_file)
        println("Progress restored successfully!")
        println("  Completed: $(new_tracker.state.overall_completed)")
        println("  Remaining: $(new_tracker.state.overall_total - new_tracker.state.overall_completed)")
    end
    
    # Cleanup
    rm(temp_file, force=true)
end

"""
Advanced metrics demo
"""
function advanced_metrics_demo()
    println("\n\nAdvanced Metrics Demo")
    println("====================")
    
    tracker = create_progress_tracker()
    add_stage!(tracker, "Metrics Collection", 1000)
    
    println("Processing with detailed metrics...")
    
    # Simulate variable processing speeds
    speeds = [50, 100, 75, 200, 150, 100, 80, 120, 90, 110]
    
    for (i, speed) in enumerate(speeds)
        # Process batch
        update_progress!(tracker, speed)
        sleep(1.0)
        
        # Get metrics
        stats = get_speed_stats(tracker)
        eta = get_formatted_eta(tracker)
        pct = round(get_progress_percentage(tracker), digits=1)
        
        println("\nBatch $i processed:")
        println("  Items: $speed")
        println("  Current speed: $(round(stats.current, digits=1)) items/sec")
        println("  Average speed: $(round(stats.average, digits=1)) items/sec")
        println("  Progress: $pct%")
        println("  ETA: $eta")
    end
    
    # Get final history
    history = get_history(tracker, last_n=5)
    println("\nLast 5 measurements:")
    for i in 1:length(history.timestamps)
        println("  $(i): $(round(history.speeds[i], digits=1)) items/sec at $(round(history.percentages[i], digits=1))%")
    end
end

"""
Visual progress demo with animation
"""
function visual_progress_demo()
    println("\n\nVisual Progress Demo")
    println("===================")
    
    tracker = create_progress_tracker()
    add_stage!(tracker, "Visual Task", 100)
    
    # Animation frames
    spinners = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    colors = ["\033[32m", "\033[33m", "\033[31m", "\033[0m"]  # green, yellow, red, reset
    
    println("Processing with visual indicators...")
    
    for i in 1:100
        update_progress!(tracker, 1)
        
        # Get current state
        pct = get_progress_percentage(tracker)
        bar = get_progress_bar(tracker, width=30, filled_char='▓', empty_char='░')
        spinner = spinners[(i % length(spinners)) + 1]
        
        # Choose color based on progress
        color = if pct < 33
            colors[1]  # green
        elseif pct < 66
            colors[2]  # yellow
        else
            colors[3]  # red
        end
        
        # Display
        print("\r$spinner $color$bar$(colors[4])")
        flush(stdout)
        sleep(0.05)
    end
    
    println("\n✓ Complete!")
end

"""
Dashboard integration demo
"""
function dashboard_integration_demo()
    println("\n\nDashboard Integration Demo")
    println("=========================")
    
    # Create tracker with callback
    panel_updates = 0
    
    function dashboard_update(tracker)
        panel_updates += 1
        # In real dashboard, this would update UI panels
    end
    
    tracker = create_progress_tracker(update_callback = dashboard_update)
    add_stage!(tracker, "Dashboard Task", 50)
    
    println("Simulating dashboard integration...")
    println("Each progress update triggers dashboard refresh")
    
    for i in 1:10
        update_progress!(tracker, 5)
        
        # Simulate dashboard panel content
        println("\n--- Dashboard Update #$panel_updates ---")
        println("┌─ Progress Panel ─────────────┐")
        println("│ $(get_progress_bar(tracker, width=27)) │")
        println("├─ Stage Panel ────────────────┤")
        println("│ $(rpad(get_stage_indicator(tracker), 29)) │")
        println("├─ Metrics Panel ──────────────┤")
        stats = get_speed_stats(tracker)
        println("│ Speed: $(rpad(round(stats.current, digits=1), 22)) │")
        println("│ ETA: $(rpad(get_formatted_eta(tracker), 24)) │")
        println("└──────────────────────────────┘")
        
        sleep(0.5)
    end
    
    println("\nDashboard updates completed: $panel_updates")
end

# Main menu
function main()
    println("Progress Tracker System Examples")
    println("===============================")
    println("1. Basic Progress Tracking")
    println("2. MCTS Phase Tracking")
    println("3. Multi-Stage Pipeline")
    println("4. Pause and Resume")
    println("5. Progress Persistence")
    println("6. Advanced Metrics")
    println("7. Visual Progress")
    println("8. Dashboard Integration")
    println("9. Run All Demos")
    println("\nSelect demo (1-9): ")
    
    choice = readline()
    
    if choice == "1"
        basic_progress_demo()
    elseif choice == "2"
        mcts_phase_demo()
    elseif choice == "3"
        pipeline_demo()
    elseif choice == "4"
        pause_resume_demo()
    elseif choice == "5"
        persistence_demo()
    elseif choice == "6"
        advanced_metrics_demo()
    elseif choice == "7"
        visual_progress_demo()
    elseif choice == "8"
        dashboard_integration_demo()
    elseif choice == "9"
        basic_progress_demo()
        mcts_phase_demo()
        pipeline_demo()
        pause_resume_demo()
        persistence_demo()
        advanced_metrics_demo()
        println("\nSkipping visual demo in batch mode...")
        dashboard_integration_demo()
    else
        println("Invalid choice. Running basic demo...")
        basic_progress_demo()
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end