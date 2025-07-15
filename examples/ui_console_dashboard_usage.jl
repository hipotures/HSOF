# Example usage of Console Dashboard for HSOF

using Dates
using Random

# Include the module
include("../src/ui/console_dashboard.jl")
using .ConsoleDashboard

"""
Simulate GPU data updates
"""
function simulate_gpu_data(gpu_id::Int, iteration::Int)
    base_util = 30.0 + 40.0 * sin(iteration / 10)
    base_mem = 4.0 + 3.0 * sin(iteration / 15)
    
    return GPUPanelContent(
        gpu_id,
        base_util + 10.0 * rand(),          # utilization
        base_mem + rand(),                  # memory used
        12.0,                               # memory total
        55.0 + 20.0 * sin(iteration / 20) + 5.0 * rand(),  # temperature
        100.0 + 50.0 * sin(iteration / 10), # power draw
        1500.0 + 300.0 * sin(iteration / 8), # clock speed
        30.0 + 30.0 * sin(iteration / 25)   # fan speed
    )
end

"""
Simulate progress data
"""
function simulate_progress_data(iteration::Int, total_iterations::Int)
    stages = ["Stage 1: Filtering", "Stage 2: MCTS Search", "Stage 3: Evaluation"]
    current_stage = div(iteration * 3, total_iterations) + 1
    current_stage = min(current_stage, 3)
    
    overall_progress = (iteration / total_iterations) * 100
    stage_progress = ((iteration % (total_iterations ÷ 3)) / (total_iterations ÷ 3)) * 100
    
    return ProgressPanelContent(
        stages[current_stage],
        overall_progress,
        stage_progress,
        iteration * 100,                    # items processed
        total_iterations * 100,             # items total
        0.85 + 0.1 * rand(),               # current score
        0.92,                              # best score
        max(0, (total_iterations - iteration) * 2)  # eta seconds
    )
end

"""
Simulate metrics data
"""
function simulate_metrics_data(iteration::Int)
    return MetricsPanelContent(
        1000.0 + 500.0 * sin(iteration / 10) + 100.0 * rand(),  # nodes/sec
        3.5 + 1.5 * sin(iteration / 8),                         # bandwidth
        40.0 + 30.0 * sin(iteration / 15),                      # CPU usage
        8.0 + 4.0 * sin(iteration / 20),                        # RAM usage
        70.0 + 20.0 * sin(iteration / 12),                      # cache hit rate
        5 + round(Int, 5 * sin(iteration / 10)),                # queue depth
        16 + round(Int, 8 * sin(iteration / 7))                 # active threads
    )
end

"""
Simulate analysis data
"""
function simulate_analysis_data(iteration::Int)
    features = [
        "feature_importance_$(i)" => 0.95 - 0.05 * i + 0.02 * rand()
        for i in 1:10
    ]
    
    selected = 50 + round(Int, 20 * sin(iteration / 10))
    
    return AnalysisPanelContent(
        1000,                              # total features
        selected,                          # selected features
        (1000 - selected) / 10,           # reduction percentage
        features[1:5],                     # top 5 features
        "Correlation: Low multicollinearity detected (avg: 0.15)"
    )
end

"""
Generate log entries
"""
function generate_log_entries(iteration::Int)
    log_types = [
        (:info, [
            "GPU kernel launched successfully",
            "Feature batch processed",
            "Checkpoint saved",
            "Memory pool allocated",
            "Cache warmed up"
        ]),
        (:warn, [
            "GPU temperature above threshold",
            "Memory usage high",
            "Queue depth increasing",
            "Slow kernel execution detected",
            "Cache miss rate elevated"
        ]),
        (:error, [
            "CUDA out of memory",
            "Kernel launch failed",
            "Database connection lost",
            "Invalid feature detected",
            "Synchronization timeout"
        ])
    ]
    
    entries = Tuple{DateTime, Symbol, String}[]
    
    # Add some historical entries
    for i in 1:8
        level_idx = rand(1:3)
        level, messages = log_types[level_idx]
        message = rand(messages)
        timestamp = now() - Minute(10 - i)
        push!(entries, (timestamp, level, message))
    end
    
    # Add current entry
    if iteration % 3 == 0
        level, messages = log_types[1]  # info
        push!(entries, (now(), level, "Iteration $iteration completed"))
    elseif iteration % 7 == 0
        level, messages = log_types[2]  # warn
        push!(entries, (now(), level, rand(messages)))
    elseif iteration % 11 == 0
        level, messages = log_types[3]  # error
        push!(entries, (now(), level, rand(messages)))
    end
    
    return LogPanelContent(entries, 100)
end

"""
Main dashboard demo
"""
function dashboard_demo(; iterations::Int = 100, delay::Float64 = 0.1)
    println("Console Dashboard Demo")
    println("=====================")
    println("This demo simulates a live dashboard for the HSOF system.")
    println("Press Ctrl+C to stop.")
    println()
    
    # Create dashboard with custom configuration
    config = DashboardConfig(
        refresh_rate_ms = 100,
        color_scheme = :default,
        border_style = :rounded,
        show_timestamps = true,
        responsive = true
    )
    
    # Initialize dashboard
    dashboard = create_dashboard(config)
    
    println("Dashboard initialized. Starting simulation...")
    sleep(2)
    
    # Clear screen for dashboard
    print("\033[2J\033[H")
    
    try
        for i in 1:iterations
            # Generate new data
            updates = Dict{Symbol, PanelContent}(
                :gpu1 => simulate_gpu_data(1, i),
                :gpu2 => simulate_gpu_data(2, i),
                :progress => simulate_progress_data(i, iterations),
                :metrics => simulate_metrics_data(i),
                :analysis => simulate_analysis_data(i),
                :log => generate_log_entries(i)
            )
            
            # Update dashboard
            update_dashboard!(dashboard, updates)
            
            # Render dashboard
            rendered = render_dashboard(dashboard)
            
            # Clear screen and display
            print("\033[2J\033[H")
            println(rendered)
            
            # Add status line
            println()
            println("Iteration: $i/$iterations | Press Ctrl+C to stop")
            
            # Check for terminal resize
            resized, new_w, new_h = ConsoleDashboard.check_terminal_resize(dashboard)
            if resized
                dashboard = ConsoleDashboard.handle_resize!(dashboard, new_w, new_h)
                println("Terminal resized to $(new_w)x$(new_h)")
            end
            
            # Delay
            sleep(delay)
        end
        
        println("\n\nDemo completed!")
        
    catch e
        if isa(e, InterruptException)
            println("\n\nDemo interrupted by user.")
        else
            rethrow(e)
        end
    end
end

"""
Static dashboard example
"""
function static_dashboard_example()
    println("Static Dashboard Example")
    println("=======================")
    
    # Create dashboard
    dashboard = create_dashboard()
    
    # Set some example data
    updates = Dict{Symbol, PanelContent}(
        :gpu1 => GPUPanelContent(1, 75.0, 8.5, 12.0, 65.0, 150.0, 1800.0, 50.0),
        :gpu2 => GPUPanelContent(2, 82.0, 9.2, 12.0, 68.0, 165.0, 1850.0, 55.0),
        :progress => ProgressPanelContent(
            "Stage 2: MCTS Search", 
            45.0, 60.0, 4500, 10000, 0.87, 0.92, 3600
        ),
        :metrics => MetricsPanelContent(
            1523.5, 4.2, 65.0, 10.5, 85.3, 8, 24
        ),
        :analysis => AnalysisPanelContent(
            1000, 87, 91.3,
            [
                ("feature_gain_125", 0.943),
                ("feature_importance_23", 0.891),
                ("feature_split_count_89", 0.857),
                ("feature_correlation_12", 0.823),
                ("feature_permutation_45", 0.801)
            ],
            "Strong feature reduction achieved. Low correlation detected."
        ),
        :log => LogPanelContent(
            [
                (now() - Minute(5), :info, "System initialized"),
                (now() - Minute(4), :info, "GPU devices detected: 2x RTX 4090"),
                (now() - Minute(3), :info, "Database connected"),
                (now() - Minute(2), :warn, "High memory usage on GPU 2"),
                (now() - Minute(1), :info, "Stage 2 search started"),
                (now() - Second(30), :error, "Cache miss rate above threshold"),
                (now() - Second(10), :info, "Recovered from cache miss"),
                (now(), :info, "Processing batch 45/100")
            ],
            100
        )
    )
    
    # Update and render
    update_dashboard!(dashboard, updates)
    rendered = render_dashboard(dashboard)
    
    println(rendered)
    println("\nThis is a static view of the dashboard showing various system metrics.")
end

"""
Minimal dashboard example
"""
function minimal_dashboard_example()
    println("Minimal Dashboard Example")
    println("========================")
    
    # Create dashboard with minimal configuration
    config = DashboardConfig(
        border_style = :single,
        show_timestamps = false
    )
    
    dashboard = create_dashboard(config)
    
    # Update only some panels
    updates = Dict{Symbol, PanelContent}(
        :progress => ProgressPanelContent(
            "Initializing", 0.0, 0.0, 0, 100, 0.0, 0.0, 0
        ),
        :log => LogPanelContent(
            [(now(), :info, "System starting...")],
            100
        )
    )
    
    update_dashboard!(dashboard, updates)
    rendered = render_dashboard(dashboard)
    
    println(rendered)
    println("\nMinimal dashboard with just a few panels updated.")
end

# Run examples
if abspath(PROGRAM_FILE) == @__FILE__
    println("HSOF Console Dashboard Examples")
    println("==============================\n")
    
    println("1. Static Dashboard Example")
    println("2. Minimal Dashboard Example")
    println("3. Live Dashboard Demo")
    println("\nSelect an example (1-3): ")
    
    choice = readline()
    
    if choice == "1"
        static_dashboard_example()
    elseif choice == "2"
        minimal_dashboard_example()
    elseif choice == "3"
        println("\nEnter number of iterations (default 100): ")
        iter_input = readline()
        iterations = isempty(iter_input) ? 100 : parse(Int, iter_input)
        
        println("Enter delay between updates in seconds (default 0.1): ")
        delay_input = readline()
        delay = isempty(delay_input) ? 0.1 : parse(Float64, delay_input)
        
        dashboard_demo(iterations=iterations, delay=delay)
    else
        println("Invalid choice. Running static example...")
        static_dashboard_example()
    end
end