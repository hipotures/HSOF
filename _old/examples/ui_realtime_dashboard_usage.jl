# Example usage of Real-time Dashboard Update System for HSOF

using Dates
using Random

# Include the modules
include("../src/ui/realtime_update.jl")
using .RealtimeUpdate
using .RealtimeUpdate.ConsoleDashboard

"""
Simulate GPU metrics that change over time
"""
function simulate_gpu_metrics(gpu_id::Int, base_util::Float64, iteration::Int)
    # Add some realistic variations
    util_variation = 10 * sin(iteration / 20) + 5 * randn()
    utilization = clamp(base_util + util_variation, 0.0, 100.0)
    
    # Memory usage correlates with utilization
    memory_base = 3.0 + (utilization / 100.0) * 6.0
    memory_used = clamp(memory_base + 0.5 * randn(), 0.0, 11.5)
    
    # Temperature follows utilization with lag
    temp_base = 40.0 + (utilization / 100.0) * 45.0
    temperature = clamp(temp_base + 3 * randn(), 30.0, 95.0)
    
    # Power correlates with utilization
    power = 50.0 + (utilization / 100.0) * 200.0 + 10 * randn()
    
    # Clock speed inversely related to temperature
    clock_base = 2100.0 - (temperature - 60.0) * 10
    clock_speed = clamp(clock_base + 50 * randn(), 1200.0, 2100.0)
    
    # Fan speed based on temperature
    fan_speed = clamp(20.0 + (temperature - 50.0) * 1.5 + 5 * randn(), 0.0, 100.0)
    
    return GPUPanelContent(
        gpu_id, utilization, memory_used, 12.0,
        temperature, power, clock_speed, fan_speed
    )
end

"""
Simulate MCTS search progress
"""
function simulate_search_progress(iteration::Int, total_iterations::Int)
    stages = [
        "Stage 1: Coarse Filtering",
        "Stage 2: MCTS Tree Search", 
        "Stage 3: Model Evaluation"
    ]
    
    # Progress through stages
    progress_per_stage = total_iterations ÷ 3
    current_stage_idx = min(div(iteration, progress_per_stage) + 1, 3)
    stage_start = (current_stage_idx - 1) * progress_per_stage
    stage_progress = ((iteration - stage_start) / progress_per_stage) * 100
    
    overall_progress = (iteration / total_iterations) * 100
    
    # Score improves over time with some noise
    base_score = 0.60 + (iteration / total_iterations) * 0.25
    current_score = clamp(base_score + 0.02 * randn(), 0.0, 1.0)
    best_score = max(current_score, 0.60 + (iteration / total_iterations) * 0.30)
    
    # ETA calculation
    if iteration > 0
        time_per_iter = 0.5  # seconds
        remaining_iters = total_iterations - iteration
        eta_seconds = round(Int, remaining_iters * time_per_iter)
    else
        eta_seconds = total_iterations
    end
    
    return ProgressPanelContent(
        stages[current_stage_idx],
        overall_progress,
        stage_progress,
        iteration * 50,  # items processed
        total_iterations * 50,  # total items
        current_score,
        best_score,
        eta_seconds
    )
end

"""
Simulate system metrics
"""
function simulate_system_metrics(iteration::Int)
    # Nodes per second varies with system load
    base_nodes = 2000.0
    nodes_per_sec = base_nodes + 500 * sin(iteration / 15) + 200 * randn()
    nodes_per_sec = max(100.0, nodes_per_sec)
    
    # Bandwidth usage
    bandwidth = 3.0 + 2.0 * sin(iteration / 10) + 0.5 * randn()
    bandwidth = clamp(bandwidth, 0.5, 8.0)
    
    # CPU usage varies
    cpu_usage = 40.0 + 20.0 * sin(iteration / 12) + 10.0 * randn()
    cpu_usage = clamp(cpu_usage, 10.0, 95.0)
    
    # RAM usage gradually increases
    ram_base = 8.0 + (iteration / 200.0)
    ram_usage = clamp(ram_base + 0.5 * randn(), 4.0, 32.0)
    
    # Cache hit rate
    cache_hit = 80.0 + 10.0 * sin(iteration / 8) + 5.0 * randn()
    cache_hit = clamp(cache_hit, 50.0, 99.0)
    
    # Queue and threads
    queue_depth = round(Int, 5 + 3 * sin(iteration / 7) + 2 * randn())
    queue_depth = max(0, queue_depth)
    
    active_threads = 16 + round(Int, 8 * sin(iteration / 9))
    active_threads = clamp(active_threads, 8, 32)
    
    return MetricsPanelContent(
        nodes_per_sec, bandwidth, cpu_usage, ram_usage,
        cache_hit, queue_depth, active_threads
    )
end

"""
Simple real-time dashboard demo
"""
function simple_realtime_demo(; duration_seconds::Int = 30)
    println("Simple Real-time Dashboard Demo")
    println("==============================")
    println("This demo shows basic real-time updates.")
    println("Press Ctrl+C to stop.\n")
    
    # Create dashboard
    dashboard = create_dashboard()
    
    # Create live dashboard with default settings
    live_dashboard = LiveDashboard(dashboard)
    
    println("Starting real-time dashboard...")
    sleep(1)
    
    # Start the live updates
    start_live_dashboard!(live_dashboard)
    
    try
        start_time = now()
        iteration = 0
        
        while (now() - start_time).value < duration_seconds * 1000
            iteration += 1
            
            # Generate updates
            updates = Dict{Symbol, PanelContent}(
                :gpu1 => simulate_gpu_metrics(1, 60.0, iteration),
                :gpu2 => simulate_gpu_metrics(2, 75.0, iteration),
                :progress => simulate_search_progress(iteration, duration_seconds * 2),
                :metrics => simulate_system_metrics(iteration)
            )
            
            # Queue the updates
            update_live_data!(live_dashboard, updates)
            
            # Simulate work being done
            sleep(0.5)
        end
        
        println("\n\nDemo completed!")
        
    catch e
        if isa(e, InterruptException)
            println("\n\nDemo interrupted by user.")
        else
            rethrow(e)
        end
    finally
        # Stop the dashboard
        stop_live_dashboard!(live_dashboard)
        
        # Show performance stats
        stats = get_update_stats(live_dashboard)
        println("\nPerformance Statistics:")
        println("  Average frame time: $(round(stats.avg_frame_time_ms, digits=2))ms")
        println("  Min frame time: $(round(stats.min_frame_time_ms, digits=2))ms") 
        println("  Max frame time: $(round(stats.max_frame_time_ms, digits=2))ms")
        println("  Average FPS: $(round(stats.fps, digits=1))")
        println("  Dropped frames: $(stats.dropped_frames)")
        println("  Uptime: $(round(stats.uptime_seconds, digits=1))s")
    end
end

"""
Advanced demo with custom configuration
"""
function advanced_realtime_demo(; duration_seconds::Int = 30)
    println("Advanced Real-time Dashboard Demo")
    println("================================")
    println("Features: Delta updates, performance tracking, custom refresh rate")
    println("Press Ctrl+C to stop.\n")
    
    # Create dashboard with custom config
    dashboard_config = DashboardConfig(
        refresh_rate_ms = 100,
        color_scheme = :default,
        border_style = :double,
        show_timestamps = true
    )
    dashboard = create_dashboard(dashboard_config)
    
    # Create live dashboard with custom update config
    update_config = DashboardUpdateConfig(
        update_interval_ms = 50,  # 20 FPS
        enable_double_buffer = true,
        enable_delta_updates = true,
        max_queue_size = 50,
        performance_tracking = true
    )
    live_dashboard = LiveDashboard(dashboard, update_config)
    
    println("Configuration:")
    println("  Update interval: $(update_config.update_interval_ms)ms")
    println("  Double buffering: $(update_config.enable_double_buffer)")
    println("  Delta updates: $(update_config.enable_delta_updates)")
    println("\nStarting dashboard...")
    sleep(1)
    
    start_live_dashboard!(live_dashboard)
    
    try
        start_time = now()
        iteration = 0
        last_log_update = now()
        log_entries = Tuple{DateTime, Symbol, String}[]
        
        while (now() - start_time).value < duration_seconds * 1000
            iteration += 1
            
            # Update all panels
            updates = Dict{Symbol, PanelContent}(
                :gpu1 => simulate_gpu_metrics(1, 55.0 + 10 * sin(iteration/30), iteration),
                :gpu2 => simulate_gpu_metrics(2, 70.0 + 10 * sin(iteration/25), iteration),
                :progress => simulate_search_progress(iteration, duration_seconds * 4),
                :metrics => simulate_system_metrics(iteration),
                :analysis => AnalysisPanelContent(
                    10000,
                    round(Int, 500 + 100 * sin(iteration / 20)),
                    85.0 + 5 * sin(iteration / 15),
                    [
                        ("feature_$(i)", 0.9 - 0.1 * i + 0.01 * randn())
                        for i in 1:5
                    ],
                    "Correlation: $(round(0.15 + 0.05 * sin(iteration / 10), digits=3))"
                )
            )
            
            # Update log periodically
            if (now() - last_log_update).value > 2000  # Every 2 seconds
                level = rand([:info, :warn, :error])
                messages = Dict(
                    :info => ["Processing batch", "Checkpoint saved", "GPU sync complete"],
                    :warn => ["High memory usage", "Slow convergence", "Queue buildup"],
                    :error => ["Feature NaN detected", "GPU timeout", "Memory allocation failed"]
                )
                
                push!(log_entries, (now(), level, rand(messages[level])))
                # Keep only last 10 entries
                if length(log_entries) > 10
                    popfirst!(log_entries)
                end
                
                updates[:log] = LogPanelContent(log_entries, 100)
                last_log_update = now()
            end
            
            # Send updates
            update_live_data!(live_dashboard, updates)
            
            # Vary the work simulation
            sleep(0.2 + 0.1 * rand())
        end
        
    catch e
        if isa(e, InterruptException)
            println("\n\nInterrupted.")
        else
            rethrow(e)
        end
    finally
        stop_live_dashboard!(live_dashboard)
        
        # Detailed performance report
        stats = get_update_stats(live_dashboard)
        println("\n\nPerformance Report:")
        println("="^40)
        println("Target FPS: $(1000.0 / update_config.update_interval_ms)")
        println("Actual FPS: $(round(stats.fps, digits=1))")
        println("Frame time (avg/min/max): $(round(stats.avg_frame_time_ms, digits=1))/$(round(stats.min_frame_time_ms, digits=1))/$(round(stats.max_frame_time_ms, digits=1))ms")
        println("Dropped frames: $(stats.dropped_frames)")
        println("Total runtime: $(round(stats.uptime_seconds, digits=1))s")
        
        if stats.dropped_frames > 0
            drop_rate = (stats.dropped_frames / (stats.uptime_seconds * (1000.0 / update_config.update_interval_ms))) * 100
            println("Drop rate: $(round(drop_rate, digits=2))%")
        end
    end
end

"""
Stress test demo
"""
function stress_test_demo(; duration_seconds::Int = 10)
    println("Stress Test Demo")
    println("===============")
    println("High frequency updates to test performance limits")
    println("Press Ctrl+C to stop.\n")
    
    dashboard = create_dashboard()
    
    # Aggressive update settings
    config = DashboardUpdateConfig(
        update_interval_ms = 16,  # ~60 FPS target
        enable_double_buffer = true,
        enable_delta_updates = true,
        max_queue_size = 200,
        performance_tracking = true
    )
    live_dashboard = LiveDashboard(dashboard, config)
    
    start_live_dashboard!(live_dashboard)
    
    try
        start_time = now()
        iteration = 0
        update_count = 0
        
        while (now() - start_time).value < duration_seconds * 1000
            iteration += 1
            
            # Rapid fire updates
            for _ in 1:3
                update_count += 1
                updates = Dict{Symbol, PanelContent}(
                    :gpu1 => simulate_gpu_metrics(1, 80.0 + 20 * rand(), update_count),
                    :gpu2 => simulate_gpu_metrics(2, 85.0 + 15 * rand(), update_count),
                    :metrics => simulate_system_metrics(update_count)
                )
                
                update_live_data!(live_dashboard, updates)
            end
            
            # Brief pause
            sleep(0.05)
        end
        
    finally
        stop_live_dashboard!(live_dashboard)
        
        stats = get_update_stats(live_dashboard)
        println("\n\nStress Test Results:")
        println("  Update count: $update_count")
        println("  Updates per second: $(round(update_count / stats.uptime_seconds, digits=1))")
        println("  Achieved FPS: $(round(stats.fps, digits=1))")
        println("  Dropped frames: $(stats.dropped_frames)")
        println("  Queue overflows: $(stats.dropped_frames > 0 ? "Yes" : "No")")
    end
end

# Run examples
if abspath(PROGRAM_FILE) == @__FILE__
    println("HSOF Real-time Dashboard Examples")
    println("================================\n")
    
    println("1. Simple Real-time Demo (30s)")
    println("2. Advanced Real-time Demo (30s)")
    println("3. Stress Test Demo (10s)")
    println("\nSelect a demo (1-3): ")
    
    choice = readline()
    
    if choice == "1"
        simple_realtime_demo()
    elseif choice == "2"
        advanced_realtime_demo()
    elseif choice == "3"
        stress_test_demo()
    else
        println("Running simple demo...")
        simple_realtime_demo(duration_seconds=10)
    end
end