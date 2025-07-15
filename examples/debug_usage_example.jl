# Example usage of Ensemble Debugger for HSOF project

include("../src/debug/ensemble_debugger.jl")
include("../src/debug/performance_analyzer.jl")

using .EnsembleDebugger
using .PerformanceAnalyzer
using Dates
using JSON3

"""
Example: Debug session for ensemble execution
"""
function debug_ensemble_example()
    println("Starting Ensemble Debug Example...")
    
    # Create debug session with custom configuration
    debug_config = Dict{String, Any}(
        "log_level" => :debug,
        "profile_gpu" => true,
        "track_memory" => true,
        "timeline_resolution_ms" => 50,
        "heatmap_update_interval" => 500
    )
    
    session = EnsembleDebugSession(
        output_dir = "debug_output_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
        config = debug_config
    )
    
    # Start debug session
    start_debug_session(session)
    
    # Example 1: Tree State Visualization
    println("\n1. Visualizing tree states...")
    
    # Simulate tree data
    tree_data = Dict(
        :feature_indices => [1, 15, 27, 45],
        :score => 0.85f0,
        :visit_count => 150,
        :state => "selected",
        :children => [
            Dict(:feature_indices => [1, 15], :score => 0.72f0, :visit_count => 80, :state => "exploring"),
            Dict(:feature_indices => [1, 27], :score => 0.68f0, :visit_count => 70, :state => "rejected")
        ]
    )
    
    # Visualize tree state
    tree_viz = visualize_tree_state(session, 1, tree_data,
        save_path = joinpath(session.output_dir, "tree_1_state.png")
    )
    
    # Example 2: Timeline Recording
    println("\n2. Recording timeline events...")
    
    # Simulate ensemble execution events
    for i in 1:20
        # Tree selection event
        record_event!(session.timeline, :tree_select,
            tree_id = i,
            node_id = rand(1:100),
            gpu_id = i % 2
        )
        
        # Feature evaluation
        record_event!(session.timeline, :feature_eval,
            tree_id = i,
            feature_count = rand(10:50),
            gpu_id = i % 2,
            duration_ms = rand() * 10
        )
        
        # GPU synchronization
        if i % 5 == 0
            record_event!(session.timeline, :gpu_sync,
                gpu_id = 0,
                sync_type = "feature_exchange"
            )
        end
    end
    
    # Generate timeline visualization
    timeline_viz = generate_timeline(session,
        save_path = joinpath(session.output_dir, "ensemble_timeline.png")
    )
    
    # Example 3: Component Profiling
    println("\n3. Profiling components...")
    
    # Profile tree selection
    result = profile_component(session, "tree_selection", () -> begin
        # Simulate tree selection operation
        data = rand(Float32, 1000, 50)
        selected = findall(data .> 0.8)
        return selected
    end)
    
    # Profile feature evaluation
    feature_eval_result = profile_component(session, "feature_evaluation", () -> begin
        # Simulate feature evaluation
        features = rand(Float32, 500, 100)
        scores = sum(features, dims=2)
        return scores
    end)
    
    # Example 4: Feature Heatmap
    println("\n4. Creating feature selection heatmap...")
    
    # Update heatmap with simulated data
    for tree_id in 1:100
        # Each tree selects different features
        n_features = rand(30:50)
        selected_features = rand(1:500, n_features)
        
        for feature_id in selected_features
            update_feature_count!(session.heatmap_gen, feature_id, tree_id,
                score = rand(Float32) * 0.5 + 0.5
            )
        end
    end
    
    # Create heatmap visualization
    heatmap_viz = create_heatmap(session, session.heatmap_gen.feature_scores,
        save_path = joinpath(session.output_dir, "feature_heatmap.png")
    )
    
    # Example 5: Debug Logging
    println("\n5. Debug logging examples...")
    
    # Log at different levels
    log_debug(session, "Starting feature selection phase",
        subsystem = "mcts",
        tree_count = 100,
        gpu_id = 0
    )
    
    log_info(session, "Metamodel loaded successfully",
        subsystem = "metamodel",
        model_size_mb = 125.5,
        load_time_ms = 523.2
    )
    
    log_warn(session, "GPU memory usage high",
        subsystem = "memory",
        gpu_id = 1,
        usage_percentage = 92.5
    )
    
    # Example 6: Performance Analysis
    println("\n6. Analyzing performance...")
    
    # Create mock ensemble data
    ensemble_data = Dict(
        :trees => 100,
        :features => 500,
        :gpus => 2
    )
    
    # Analyze performance
    perf_report = analyze_ensemble_performance(session, ensemble_data, n_iterations = 50)
    
    # Save performance report
    perf_report_path = joinpath(session.output_dir, "performance_report.json")
    open(perf_report_path, "w") do io
        JSON3.pretty(io, perf_report)
    end
    
    # Example 7: Get profiling report
    println("\n7. Generating profiling report...")
    
    prof_report = get_profiling_report(session)
    println("Components profiled: ", length(prof_report["components"]))
    
    # Stop debug session
    stop_debug_session(session)
    
    println("\n✓ Debug session completed!")
    println("Debug output saved to: $(session.output_dir)")
    
    return session
end

"""
Example: Benchmark different implementations
"""
function benchmark_example()
    println("\nRunning Benchmark Example...")
    
    # Define different implementations to compare
    implementations = Dict{String, Function}(
        "baseline" => (data) -> begin
            # Simple implementation
            result = sum(data, dims=2)
            return result
        end,
        
        "optimized" => (data) -> begin
            # Optimized implementation
            result = zeros(Float32, size(data, 1))
            @inbounds for i in 1:size(data, 1)
                result[i] = sum(@view data[i, :])
            end
            return result
        end,
        
        "parallel" => (data) -> begin
            # Parallel implementation
            result = zeros(Float32, size(data, 1))
            Threads.@threads for i in 1:size(data, 1)
                result[i] = sum(@view data[i, :])
            end
            return result
        end
    )
    
    # Test data
    test_data = rand(Float32, 1000, 500)
    
    # Run benchmark
    results = benchmark_component(implementations, test_data, 
        n_runs = 100, 
        warmup_runs = 10
    )
    
    # Compare implementations
    comparison = compare_implementations(results)
    
    println("\nBenchmark Results:")
    for (name, result) in comparison
        println("  $name:")
        println("    Mean time: $(round(result["mean_ms"], digits=3)) ms")
        println("    Speedup: $(round(result["speedup_vs_baseline"], digits=2))x")
    end
    
    return results, comparison
end

"""
Example: Memory profiling
"""
function memory_profiling_example()
    println("\nRunning Memory Profiling Example...")
    
    # Import memory profiling tools
    include("../src/debug/profiling_hooks.jl")
    using .ProfilingHooks.MemoryProfiling
    
    # Profile memory allocation
    result, mem_stats = track_memory_allocation() do
        # Simulate memory-intensive operation
        data = zeros(Float32, 10000, 1000)
        processed = data .* 2.0 .+ 1.0
        return sum(processed)
    end
    
    println("\nMemory Allocation Stats:")
    println("  Allocated: $(round(mem_stats["mb_allocated"], digits=2)) MB")
    println("  Execution time: $(round(mem_stats["execution_time"], digits=3)) seconds")
    println("  Allocation rate: $(round(mem_stats["allocation_rate_mb_per_sec"], digits=2)) MB/s")
    
    # Get current memory report
    mem_report = get_memory_report()
    println("\nCurrent Memory Status:")
    println("  Heap size: $(round(mem_report["heap_size_mb"], digits=2)) MB")
    println("  Used memory: $(round(mem_report["used_memory_mb"], digits=2)) MB")
    println("  Free memory: $(round(mem_report["free_memory_mb"], digits=2)) MB")
    
    return mem_stats, mem_report
end

# Run examples
if abspath(PROGRAM_FILE) == @__FILE__
    println("HSOF Ensemble Debugger Examples")
    println("=" ^ 50)
    
    # Run debug session example
    session = debug_ensemble_example()
    
    # Run benchmark example
    benchmark_results, comparison = benchmark_example()
    
    # Run memory profiling example
    mem_stats, mem_report = memory_profiling_example()
    
    println("\n" * "=" ^ 50)
    println("All examples completed successfully!")
end