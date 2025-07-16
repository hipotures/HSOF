using Test

# Include debug modules
include("../src/debug/tree_state_visualizer.jl")
include("../src/debug/ensemble_timeline.jl")
include("../src/debug/profiling_hooks.jl")
include("../src/debug/feature_heatmap.jl")
include("../src/debug/debug_logger.jl")
include("../src/debug/ensemble_debugger.jl")

using .TreeStateVisualizer
using .EnsembleTimeline
using .ProfilingHooks
using .FeatureHeatmap
using .DebugLogger
using .EnsembleDebugger

@testset "Ensemble Debugger Tests" begin
    
    @testset "Tree State Visualizer" begin
        viz = TreeVisualizer(mktempdir())
        @test !isnothing(viz)
        
        # Test initialization
        initialize!(viz)
        @test isempty(viz.visualization_cache)
    end
    
    @testset "Timeline Recorder" begin
        recorder = TimelineRecorder()
        @test !recorder.is_recording
        
        # Start recording
        start_recording!(recorder)
        @test recorder.is_recording
        
        # Record events
        record_event!(recorder, :test_event, data = "test")
        @test length(recorder.events) == 1
        @test recorder.event_counters[:test_event] == 1
        
        # Stop recording
        stop_recording!(recorder)
        @test !recorder.is_recording
    end
    
    @testset "Component Profiler" begin
        profiler = ComponentProfiler()
        @test profiler.is_profiling
        
        # Profile a simple function
        result, time = profile_execution(profiler, "test_component", () -> begin
            sum(rand(100))
        end)
        
        @test haskey(profiler.profiles, "test_component")
        @test profiler.profiles["test_component"].call_count == 1
        @test time > 0
        
        # Generate report
        report = generate_report(profiler)
        @test haskey(report, "components")
        @test haskey(report["components"], "test_component")
    end
    
    @testset "Feature Heatmap" begin
        heatmap_gen = HeatmapGenerator(n_features = 50, n_trees = 10)
        @test size(heatmap_gen.feature_counts) == (50, 10)
        
        # Update feature count
        update_feature_count!(heatmap_gen, 5, 3, score = 0.8f0)
        @test heatmap_gen.feature_counts[5, 3] == 1
        @test heatmap_gen.feature_scores[5, 3] == 0.8f0
        
        # Clear heatmap
        clear!(heatmap_gen)
        @test all(heatmap_gen.feature_counts .== 0)
    end
    
    @testset "Debug Logger" begin
        logger = DebugLogManager(mktempdir())
        @test logger.log_level == DebugLogger.INFO
        
        # Test logging at different levels
        log_debug(logger, "Debug message")
        log_info(logger, "Info message")
        log_warn(logger, "Warning message")
        log_error(logger, "Error message")
        
        @test logger.stats["info"] == 1
        @test logger.stats["warn"] == 1
        @test logger.stats["error"] == 1
        # Debug might not be counted if log level is INFO
        
        # Get stats
        stats = get_log_stats(logger)
        @test haskey(stats, "total_entries")
        
        # Close logger
        close(logger)
    end
    
    @testset "Ensemble Debug Session" begin
        session = EnsembleDebugSession(output_dir = mktempdir())
        @test !session.is_active
        
        # Start session
        start_debug_session(session)
        @test session.is_active
        
        # Log some debug info
        log_debug(session, "Test debug message", component = "test")
        
        # Profile a component
        result = profile_component(session, "test_operation", () -> begin
            return sum(1:100)
        end)
        @test result == 5050
        
        # Stop session
        stop_debug_session(session)
        @test !session.is_active
    end
    
    @testset "Integration Test" begin
        # Create a complete debug session
        output_dir = mktempdir()
        session = create_standard_debug_session(
            output_dir = output_dir,
            log_level = :debug
        )
        
        start_debug_session(session)
        
        # Simulate ensemble execution with debugging
        for i in 1:5
            # Record timeline events
            record_event!(session.timeline, :iteration_start, iteration = i)
            
            # Profile tree operations
            profile_component(session, "tree_$i", () -> begin
                # Simulate tree operation
                sleep(0.01)
                return rand(10)
            end)
            
            # Update feature heatmap
            for j in 1:10
                if rand() > 0.5
                    update_feature_count!(session.heatmap_gen, j, i)
                end
            end
            
            # Log progress
            log_info(session, "Completed iteration $i", 
                subsystem = "ensemble",
                trees_processed = i
            )
        end
        
        # Get profiling report
        prof_report = get_profiling_report(session)
        @test length(prof_report["components"]) == 5
        
        stop_debug_session(session)
        
        # Check that files were created
        files = readdir(output_dir)
        @test any(occursin("timeline", f) for f in files)
        @test any(occursin("profiling", f) for f in files)
    end
end

println("All tests passed! ✓")