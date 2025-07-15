"""
Performance Monitoring and Metrics for MCTS Ensemble Feature Selection
Implements comprehensive monitoring system tracking ensemble performance and resource utilization,
including GPU metrics collection, tree performance tracking, ensemble statistics aggregation,
and real-time visualization for dual RTX 4090 configuration.

This module provides detailed performance monitoring across ensemble trees, GPU utilization tracking,
and comprehensive metrics collection for analyzing and optimizing MCTS ensemble execution.
"""

module PerformanceMonitoring

using Random
using Statistics
using Dates
using Printf
using LinearAlgebra
using Base.Threads

# Import other stage2 modules for integration
include("gpu_load_balancing.jl")
using .GPULoadBalancing

include("consensus_voting.jl")
using .ConsensusVoting

include("convergence_detection.jl")
using .ConvergenceDetection

"""
GPU performance metrics structure
"""
mutable struct GPUPerformanceMetrics
    device_id::Int                          # GPU device identifier
    utilization_percentage::Float32         # GPU utilization (0-100%)
    memory_used_mb::Float32                 # Used memory in MB
    memory_total_mb::Float32                # Total memory in MB
    memory_utilization_percentage::Float32  # Memory utilization (0-100%)
    temperature_celsius::Float32            # GPU temperature
    power_draw_watts::Float32               # Power consumption
    fan_speed_percentage::Float32           # Fan speed (0-100%)
    sm_clock_mhz::Float32                  # Streaming multiprocessor clock
    memory_clock_mhz::Float32              # Memory clock
    pcie_throughput_mb_per_sec::Float32    # PCIe throughput
    compute_processes::Int                  # Number of compute processes
    last_update::DateTime                   # Last metrics update time
    update_count::Int                       # Number of updates
    is_healthy::Bool                        # GPU health status
end

"""
Create GPU performance metrics tracker
"""
function create_gpu_performance_metrics(device_id::Int)
    return GPUPerformanceMetrics(
        device_id, 0.0f0, 0.0f0, 24576.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
        0.0f0, 0.0f0, 0.0f0, 0, now(), 0, true
    )
end

"""
Tree performance metrics for individual MCTS trees
"""
mutable struct TreePerformanceMetrics
    tree_id::Int                         # Tree identifier
    iteration_count::Int                 # Total iterations completed
    depth_mean::Float32                  # Average tree depth
    depth_max::Int                       # Maximum tree depth
    depth_std::Float32                   # Standard deviation of depth
    nodes_expanded::Int                  # Total nodes expanded
    nodes_visited::Int                   # Total nodes visited
    expansion_rate::Float32              # Expansions per second
    visit_rate::Float32                  # Visits per second
    features_selected::Int               # Currently selected features
    features_rejected::Int               # Currently rejected features
    selection_diversity::Float32         # Feature selection diversity score
    performance_score::Float32           # Tree performance metric
    convergence_rate::Float32            # Rate of convergence
    memory_usage_mb::Float32             # Memory usage for this tree
    last_iteration_time::DateTime        # Last iteration timestamp
    total_execution_time::Float64        # Total execution time in seconds
    iterations_per_second::Float32       # Current iteration rate
    gpu_assignment::Int                  # Assigned GPU device
    is_active::Bool                      # Whether tree is actively running
end

"""
Create tree performance metrics tracker
"""
function create_tree_performance_metrics(tree_id::Int, gpu_assignment::Int = 0)
    return TreePerformanceMetrics(
        tree_id, 0, 0.0f0, 0, 0.0f0, 0, 0, 0.0f0, 0.0f0, 0, 0,
        0.0f0, 0.0f0, 0.0f0, 0.0f0, now(), 0.0, 0.0f0, gpu_assignment, true
    )
end

"""
Ensemble-wide performance statistics
"""
mutable struct EnsemblePerformanceStats
    # Basic ensemble metrics
    total_trees::Int                      # Total number of trees
    active_trees::Int                     # Currently active trees
    converged_trees::Int                  # Trees that have converged
    failed_trees::Int                     # Trees that have failed
    
    # Iteration and timing metrics
    total_iterations::Int                 # Total iterations across all trees
    iterations_per_second::Float32        # Global iteration rate
    average_tree_depth::Float32           # Average depth across all trees
    ensemble_diversity_score::Float32     # Overall ensemble diversity
    
    # Feature selection metrics
    consensus_strength::Float32           # Current consensus strength
    feature_selection_stability::Float32  # Stability of feature selections
    unique_features_explored::Int         # Total unique features explored
    common_features_count::Int            # Features selected by most trees
    
    # Resource utilization
    total_memory_usage_mb::Float32        # Total memory usage
    cpu_utilization_percentage::Float32   # CPU utilization
    gpu0_utilization_percentage::Float32  # GPU 0 utilization
    gpu1_utilization_percentage::Float32  # GPU 1 utilization
    
    # Performance trends
    performance_trend::Float32            # Performance improvement trend
    convergence_trend::Float32            # Convergence rate trend
    efficiency_score::Float32             # Overall efficiency metric
    
    # Timing information
    execution_start_time::DateTime        # When execution started
    last_update_time::DateTime           # Last metrics update
    total_execution_time::Float64        # Total execution time in seconds
    estimated_time_remaining::Float64    # Estimated time to completion
end

"""
Initialize ensemble performance statistics
"""
function initialize_ensemble_performance_stats(total_trees::Int)
    return EnsemblePerformanceStats(
        total_trees, 0, 0, 0, 0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0, 0,
        0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
        now(), now(), 0.0, 0.0
    )
end

"""
Performance monitoring configuration
"""
struct PerformanceMonitoringConfig
    # Collection intervals
    gpu_metrics_interval_ms::Int          # GPU metrics collection interval
    tree_metrics_interval_ms::Int         # Tree metrics collection interval
    ensemble_stats_interval_ms::Int       # Ensemble statistics interval
    
    # History and storage
    metrics_history_size::Int             # Number of historical metrics to keep
    enable_detailed_logging::Bool         # Enable detailed performance logging
    log_file_path::String                # Path for performance logs
    
    # Monitoring thresholds
    gpu_utilization_warning_threshold::Float32     # GPU utilization warning
    gpu_temperature_warning_threshold::Float32     # GPU temperature warning
    memory_usage_warning_threshold::Float32        # Memory usage warning
    performance_degradation_threshold::Float32     # Performance degradation alert
    
    # Visualization and reporting
    enable_real_time_visualization::Bool  # Enable real-time charts
    visualization_update_interval_ms::Int # Visualization update frequency
    enable_performance_dashboard::Bool    # Enable performance dashboard
    dashboard_port::Int                   # Port for web dashboard
    
    # Data export and persistence
    enable_metrics_export::Bool          # Export metrics to files
    export_interval_minutes::Int         # Export frequency
    export_format::String               # Export format (JSON, CSV, etc.)
    enable_database_logging::Bool       # Log to database
    
    # Performance optimization
    enable_parallel_collection::Bool     # Parallel metrics collection
    collection_thread_count::Int         # Number of collection threads
    enable_metrics_compression::Bool     # Compress stored metrics
    cache_frequently_accessed_metrics::Bool  # Cache hot metrics
end

"""
Create performance monitoring configuration
"""
function create_performance_monitoring_config(;
    gpu_metrics_interval_ms::Int = 1000,
    tree_metrics_interval_ms::Int = 500,
    ensemble_stats_interval_ms::Int = 2000,
    metrics_history_size::Int = 1000,
    enable_detailed_logging::Bool = true,
    log_file_path::String = "performance_metrics.log",
    gpu_utilization_warning_threshold::Float32 = 95.0f0,
    gpu_temperature_warning_threshold::Float32 = 85.0f0,
    memory_usage_warning_threshold::Float32 = 90.0f0,
    performance_degradation_threshold::Float32 = 0.8f0,
    enable_real_time_visualization::Bool = false,
    visualization_update_interval_ms::Int = 5000,
    enable_performance_dashboard::Bool = false,
    dashboard_port::Int = 8080,
    enable_metrics_export::Bool = true,
    export_interval_minutes::Int = 5,
    export_format::String = "JSON",
    enable_database_logging::Bool = false,
    enable_parallel_collection::Bool = true,
    collection_thread_count::Int = 4,
    enable_metrics_compression::Bool = false,
    cache_frequently_accessed_metrics::Bool = true
)
    return PerformanceMonitoringConfig(
        gpu_metrics_interval_ms, tree_metrics_interval_ms, ensemble_stats_interval_ms,
        metrics_history_size, enable_detailed_logging, log_file_path,
        gpu_utilization_warning_threshold, gpu_temperature_warning_threshold,
        memory_usage_warning_threshold, performance_degradation_threshold,
        enable_real_time_visualization, visualization_update_interval_ms,
        enable_performance_dashboard, dashboard_port,
        enable_metrics_export, export_interval_minutes, export_format, enable_database_logging,
        enable_parallel_collection, collection_thread_count, enable_metrics_compression,
        cache_frequently_accessed_metrics
    )
end

"""
Performance alert system
"""
mutable struct PerformanceAlert
    alert_id::String                     # Unique alert identifier
    alert_type::String                   # Type of alert (GPU, MEMORY, PERFORMANCE)
    severity::String                     # Severity level (INFO, WARNING, ERROR, CRITICAL)
    message::String                      # Alert message
    timestamp::DateTime                  # When alert was triggered
    device_id::Union{Int, Nothing}       # Related device (if applicable)
    tree_id::Union{Int, Nothing}         # Related tree (if applicable)
    metric_value::Float32                # Value that triggered alert
    threshold_value::Float32             # Threshold that was exceeded
    is_resolved::Bool                    # Whether alert has been resolved
    resolution_time::Union{DateTime, Nothing}  # When alert was resolved
end

"""
Create performance alert
"""
function create_performance_alert(alert_type::String, severity::String, message::String,
                                 metric_value::Float32, threshold_value::Float32;
                                 device_id::Union{Int, Nothing} = nothing,
                                 tree_id::Union{Int, Nothing} = nothing)
    alert_id = string(hash((alert_type, message, now())))
    return PerformanceAlert(
        alert_id, alert_type, severity, message, now(),
        device_id, tree_id, metric_value, threshold_value,
        false, nothing
    )
end

"""
Main performance monitoring system
"""
mutable struct PerformanceMonitor
    config::PerformanceMonitoringConfig
    
    # Metrics storage
    gpu_metrics::Dict{Int, GPUPerformanceMetrics}
    tree_metrics::Dict{Int, TreePerformanceMetrics}
    ensemble_stats::EnsemblePerformanceStats
    
    # Historical data
    gpu_metrics_history::Dict{Int, Vector{GPUPerformanceMetrics}}
    tree_metrics_history::Dict{Int, Vector{TreePerformanceMetrics}}
    ensemble_stats_history::Vector{EnsemblePerformanceStats}
    
    # Alert system
    active_alerts::Vector{PerformanceAlert}
    alert_history::Vector{PerformanceAlert}
    
    # Monitoring state
    is_monitoring::Bool
    monitoring_start_time::DateTime
    last_gpu_update::DateTime
    last_tree_update::DateTime
    last_ensemble_update::DateTime
    
    # Performance cache
    cached_metrics::Dict{String, Any}
    cache_timestamps::Dict{String, DateTime}
    
    # Synchronization
    metrics_lock::ReentrantLock
    
    # Status and logging
    monitor_state::String
    error_log::Vector{String}
    performance_log::Union{IOStream, Nothing}
end

"""
Initialize performance monitor
"""
function initialize_performance_monitor(config::PerformanceMonitoringConfig = create_performance_monitoring_config())
    # Initialize GPU metrics for dual RTX 4090 setup
    gpu_metrics = Dict{Int, GPUPerformanceMetrics}()
    gpu_metrics_history = Dict{Int, Vector{GPUPerformanceMetrics}}()
    for device_id in 0:1
        gpu_metrics[device_id] = create_gpu_performance_metrics(device_id)
        gpu_metrics_history[device_id] = GPUPerformanceMetrics[]
    end
    
    # Open log file if logging is enabled
    log_stream = config.enable_detailed_logging ? open(config.log_file_path, "w") : nothing
    
    monitor = PerformanceMonitor(
        config,
        gpu_metrics,
        Dict{Int, TreePerformanceMetrics}(),
        initialize_ensemble_performance_stats(100),  # Default 100 trees
        gpu_metrics_history,
        Dict{Int, Vector{TreePerformanceMetrics}}(),
        EnsemblePerformanceStats[],
        PerformanceAlert[],
        PerformanceAlert[],
        false,
        now(),
        now(),
        now(),
        now(),
        Dict{String, Any}(),
        Dict{String, DateTime}(),
        ReentrantLock(),
        "initialized",
        String[],
        log_stream
    )
    
    @info "Performance monitor initialized with $(length(gpu_metrics)) GPU devices"
    return monitor
end

"""
Mock GPU metrics collection (simulates real GPU monitoring)
"""
function collect_gpu_metrics!(monitor::PerformanceMonitor, device_id::Int)
    if !haskey(monitor.gpu_metrics, device_id)
        return false
    end
    
    metrics = monitor.gpu_metrics[device_id]
    
    # Simulate realistic GPU metrics (in production, would use NVIDIA ML API)
    metrics.utilization_percentage = 70.0f0 + rand() * 25.0f0  # 70-95% utilization
    metrics.memory_used_mb = 20000.0f0 + rand() * 4000.0f0     # 20-24GB usage
    metrics.memory_utilization_percentage = (metrics.memory_used_mb / metrics.memory_total_mb) * 100.0f0
    metrics.temperature_celsius = 75.0f0 + rand() * 10.0f0      # 75-85°C
    metrics.power_draw_watts = 350.0f0 + rand() * 100.0f0       # 350-450W
    metrics.fan_speed_percentage = 60.0f0 + rand() * 30.0f0     # 60-90%
    metrics.sm_clock_mhz = 1800.0f0 + rand() * 400.0f0         # 1800-2200 MHz
    metrics.memory_clock_mhz = 9500.0f0 + rand() * 500.0f0     # 9500-10000 MHz
    metrics.pcie_throughput_mb_per_sec = 5000.0f0 + rand() * 3000.0f0  # 5-8 GB/s
    metrics.compute_processes = rand(1:5)                       # 1-5 processes
    
    metrics.last_update = now()
    metrics.update_count += 1
    
    # Health check
    metrics.is_healthy = (
        metrics.temperature_celsius < monitor.config.gpu_temperature_warning_threshold &&
        metrics.memory_utilization_percentage < monitor.config.memory_usage_warning_threshold
    )
    
    # Check for alerts
    check_gpu_alerts!(monitor, device_id)
    
    return true
end

"""
Update tree performance metrics
"""
function update_tree_metrics!(monitor::PerformanceMonitor, tree_id::Int, 
                              iterations::Int, depth::Float32, nodes_expanded::Int,
                              features_selected::Int, performance_score::Float32)
    
    if !haskey(monitor.tree_metrics, tree_id)
        gpu_assignment = tree_id <= 50 ? 0 : 1  # Trees 1-50 on GPU0, 51-100 on GPU1
        monitor.tree_metrics[tree_id] = create_tree_performance_metrics(tree_id, gpu_assignment)
        monitor.tree_metrics_history[tree_id] = TreePerformanceMetrics[]
    end
    
    metrics = monitor.tree_metrics[tree_id]
    current_time = now()
    time_since_last = Dates.value(current_time - metrics.last_iteration_time) / 1000.0  # seconds
    
    # Update basic metrics
    metrics.iteration_count = iterations
    metrics.depth_mean = depth
    metrics.nodes_expanded = nodes_expanded
    metrics.features_selected = features_selected
    metrics.performance_score = performance_score
    
    # Update rates
    if time_since_last > 0
        iteration_delta = iterations - (length(monitor.tree_metrics_history[tree_id]) > 0 ? 
                                       monitor.tree_metrics_history[tree_id][end].iteration_count : 0)
        metrics.iterations_per_second = iteration_delta / time_since_last
        metrics.expansion_rate = (nodes_expanded - (length(monitor.tree_metrics_history[tree_id]) > 0 ? 
                                 monitor.tree_metrics_history[tree_id][end].nodes_expanded : 0)) / time_since_last
    end
    
    # Estimate memory usage (simplified model)
    metrics.memory_usage_mb = Float32(nodes_expanded * 0.001 + features_selected * 0.01)  # Rough estimate
    
    metrics.last_iteration_time = current_time
    metrics.total_execution_time = Dates.value(current_time - monitor.monitoring_start_time) / 1000.0
    
    # Store in history
    if length(monitor.tree_metrics_history[tree_id]) >= monitor.config.metrics_history_size
        deleteat!(monitor.tree_metrics_history[tree_id], 1)
    end
    push!(monitor.tree_metrics_history[tree_id], deepcopy(metrics))
    
    return true
end

"""
Check for GPU-related performance alerts
"""
function check_gpu_alerts!(monitor::PerformanceMonitor, device_id::Int)
    metrics = monitor.gpu_metrics[device_id]
    
    # Temperature alert
    if metrics.temperature_celsius > monitor.config.gpu_temperature_warning_threshold
        alert = create_performance_alert(
            "GPU_TEMPERATURE", "WARNING",
            "GPU $device_id temperature high: $(metrics.temperature_celsius)°C",
            metrics.temperature_celsius, monitor.config.gpu_temperature_warning_threshold,
            device_id = device_id
        )
        add_alert!(monitor, alert)
    end
    
    # Memory usage alert
    if metrics.memory_utilization_percentage > monitor.config.memory_usage_warning_threshold
        alert = create_performance_alert(
            "GPU_MEMORY", "WARNING",
            "GPU $device_id memory usage high: $(round(metrics.memory_utilization_percentage, digits=1))%",
            metrics.memory_utilization_percentage, monitor.config.memory_usage_warning_threshold,
            device_id = device_id
        )
        add_alert!(monitor, alert)
    end
    
    # GPU utilization alert
    if metrics.utilization_percentage > monitor.config.gpu_utilization_warning_threshold
        alert = create_performance_alert(
            "GPU_UTILIZATION", "INFO",
            "GPU $device_id utilization high: $(round(metrics.utilization_percentage, digits=1))%",
            metrics.utilization_percentage, monitor.config.gpu_utilization_warning_threshold,
            device_id = device_id
        )
        add_alert!(monitor, alert)
    end
end

"""
Add performance alert to monitor
"""
function add_alert!(monitor::PerformanceMonitor, alert::PerformanceAlert)
    # Check if similar alert already exists
    existing_alert = findfirst(a -> a.alert_type == alert.alert_type && 
                              a.device_id == alert.device_id && 
                              a.tree_id == alert.tree_id && 
                              !a.is_resolved, monitor.active_alerts)
    
    if isnothing(existing_alert)
        push!(monitor.active_alerts, alert)
        @warn "Performance alert: $(alert.message)"
        
        # Log to file if logging is enabled
        if monitor.config.enable_detailed_logging && !isnothing(monitor.performance_log)
            write(monitor.performance_log, "[$(alert.timestamp)] $(alert.severity): $(alert.message)\n")
            flush(monitor.performance_log)
        end
    end
end

"""
Update ensemble-wide performance statistics
"""
function update_ensemble_stats!(monitor::PerformanceMonitor)
    stats = monitor.ensemble_stats
    current_time = now()
    
    # Count active trees
    stats.active_trees = count(tree -> tree.is_active, values(monitor.tree_metrics))
    stats.total_trees = length(monitor.tree_metrics)
    
    # Calculate iteration rates
    if !isempty(monitor.tree_metrics)
        total_iterations = sum(tree.iteration_count for tree in values(monitor.tree_metrics))
        stats.total_iterations = total_iterations
        
        time_elapsed = Dates.value(current_time - monitor.monitoring_start_time) / 1000.0
        if time_elapsed > 0
            stats.iterations_per_second = total_iterations / time_elapsed
        end
        
        # Calculate average depth
        stats.average_tree_depth = mean(tree.depth_mean for tree in values(monitor.tree_metrics))
        
        # Calculate ensemble diversity
        if length(monitor.tree_metrics) >= 2
            performance_scores = [tree.performance_score for tree in values(monitor.tree_metrics)]
            stats.ensemble_diversity_score = std(performance_scores)
        end
    end
    
    # GPU utilization
    if haskey(monitor.gpu_metrics, 0)
        stats.gpu0_utilization_percentage = monitor.gpu_metrics[0].utilization_percentage
    end
    if haskey(monitor.gpu_metrics, 1)
        stats.gpu1_utilization_percentage = monitor.gpu_metrics[1].utilization_percentage
    end
    
    # Memory usage
    total_memory = sum(tree.memory_usage_mb for tree in values(monitor.tree_metrics))
    stats.total_memory_usage_mb = total_memory
    
    # Update timing
    stats.last_update_time = current_time
    stats.total_execution_time = Dates.value(current_time - stats.execution_start_time) / 1000.0
    
    # Store in history
    if length(monitor.ensemble_stats_history) >= monitor.config.metrics_history_size
        deleteat!(monitor.ensemble_stats_history, 1)
    end
    push!(monitor.ensemble_stats_history, deepcopy(stats))
end

"""
Run performance monitoring cycle
"""
function run_monitoring_cycle!(monitor::PerformanceMonitor)
    lock(monitor.metrics_lock) do
        current_time = now()
        
        # Collect GPU metrics if interval has passed
        if Dates.value(current_time - monitor.last_gpu_update) >= monitor.config.gpu_metrics_interval_ms
            for device_id in keys(monitor.gpu_metrics)
                collect_gpu_metrics!(monitor, device_id)
                
                # Store in history
                if length(monitor.gpu_metrics_history[device_id]) >= monitor.config.metrics_history_size
                    deleteat!(monitor.gpu_metrics_history[device_id], 1)
                end
                push!(monitor.gpu_metrics_history[device_id], deepcopy(monitor.gpu_metrics[device_id]))
            end
            monitor.last_gpu_update = current_time
        end
        
        # Update ensemble statistics if interval has passed
        if Dates.value(current_time - monitor.last_ensemble_update) >= monitor.config.ensemble_stats_interval_ms
            update_ensemble_stats!(monitor)
            monitor.last_ensemble_update = current_time
        end
        
        # Clean up resolved alerts (older than 1 hour)
        hour_ago = current_time - Dates.Hour(1)
        filter!(alert -> !alert.is_resolved || alert.resolution_time > hour_ago, monitor.active_alerts)
    end
end

"""
Start performance monitoring
"""
function start_monitoring!(monitor::PerformanceMonitor)
    monitor.is_monitoring = true
    monitor.monitoring_start_time = now()
    monitor.ensemble_stats.execution_start_time = now()
    monitor.monitor_state = "monitoring"
    
    @info "Performance monitoring started"
    
    # If parallel collection is enabled, start background task
    if monitor.config.enable_parallel_collection
        @async begin
            while monitor.is_monitoring
                try
                    run_monitoring_cycle!(monitor)
                    sleep(0.1)  # 100ms sleep
                catch e
                    push!(monitor.error_log, "Monitoring error: $e")
                    @error "Performance monitoring error: $e"
                end
            end
        end
    end
end

"""
Stop performance monitoring
"""
function stop_monitoring!(monitor::PerformanceMonitor)
    monitor.is_monitoring = false
    monitor.monitor_state = "stopped"
    
    # Close log file if open
    if monitor.config.enable_detailed_logging && !isnothing(monitor.performance_log)
        close(monitor.performance_log)
    end
    
    @info "Performance monitoring stopped"
end

"""
Get current performance summary
"""
function get_performance_summary(monitor::PerformanceMonitor)
    lock(monitor.metrics_lock) do
        return Dict{String, Any}(
            "monitor_state" => monitor.monitor_state,
            "is_monitoring" => monitor.is_monitoring,
            "monitoring_duration_seconds" => Dates.value(now() - monitor.monitoring_start_time) / 1000.0,
            
            # Ensemble statistics
            "total_trees" => monitor.ensemble_stats.total_trees,
            "active_trees" => monitor.ensemble_stats.active_trees,
            "total_iterations" => monitor.ensemble_stats.total_iterations,
            "iterations_per_second" => monitor.ensemble_stats.iterations_per_second,
            "average_tree_depth" => monitor.ensemble_stats.average_tree_depth,
            "ensemble_diversity_score" => monitor.ensemble_stats.ensemble_diversity_score,
            
            # GPU utilization
            "gpu0_utilization" => haskey(monitor.gpu_metrics, 0) ? monitor.gpu_metrics[0].utilization_percentage : 0.0f0,
            "gpu1_utilization" => haskey(monitor.gpu_metrics, 1) ? monitor.gpu_metrics[1].utilization_percentage : 0.0f0,
            "gpu0_memory_usage_mb" => haskey(monitor.gpu_metrics, 0) ? monitor.gpu_metrics[0].memory_used_mb : 0.0f0,
            "gpu1_memory_usage_mb" => haskey(monitor.gpu_metrics, 1) ? monitor.gpu_metrics[1].memory_used_mb : 0.0f0,
            "gpu0_temperature" => haskey(monitor.gpu_metrics, 0) ? monitor.gpu_metrics[0].temperature_celsius : 0.0f0,
            "gpu1_temperature" => haskey(monitor.gpu_metrics, 1) ? monitor.gpu_metrics[1].temperature_celsius : 0.0f0,
            
            # Memory and resource usage
            "total_memory_usage_mb" => monitor.ensemble_stats.total_memory_usage_mb,
            
            # Alerts
            "active_alerts_count" => length(monitor.active_alerts),
            "critical_alerts_count" => count(alert -> alert.severity == "CRITICAL", monitor.active_alerts),
            "warning_alerts_count" => count(alert -> alert.severity == "WARNING", monitor.active_alerts),
            
            # Performance trends
            "performance_trend" => monitor.ensemble_stats.performance_trend,
            "convergence_trend" => monitor.ensemble_stats.convergence_trend,
            "efficiency_score" => monitor.ensemble_stats.efficiency_score,
            
            # Last update times
            "last_gpu_update" => monitor.last_gpu_update,
            "last_tree_update" => monitor.last_tree_update,
            "last_ensemble_update" => monitor.last_ensemble_update
        )
    end
end

"""
Get detailed tree performance report
"""
function get_tree_performance_report(monitor::PerformanceMonitor, tree_id::Union{Int, Nothing} = nothing)
    lock(monitor.metrics_lock) do
        if isnothing(tree_id)
            # Return summary for all trees
            return Dict{String, Any}(
                "total_trees" => length(monitor.tree_metrics),
                "trees" => Dict(id => Dict(
                    "tree_id" => metrics.tree_id,
                    "iteration_count" => metrics.iteration_count,
                    "depth_mean" => metrics.depth_mean,
                    "depth_max" => metrics.depth_max,
                    "nodes_expanded" => metrics.nodes_expanded,
                    "features_selected" => metrics.features_selected,
                    "performance_score" => metrics.performance_score,
                    "iterations_per_second" => metrics.iterations_per_second,
                    "memory_usage_mb" => metrics.memory_usage_mb,
                    "gpu_assignment" => metrics.gpu_assignment,
                    "is_active" => metrics.is_active,
                    "total_execution_time" => metrics.total_execution_time
                ) for (id, metrics) in monitor.tree_metrics)
            )
        else
            # Return detailed report for specific tree
            if haskey(monitor.tree_metrics, tree_id)
                metrics = monitor.tree_metrics[tree_id]
                history = get(monitor.tree_metrics_history, tree_id, TreePerformanceMetrics[])
                
                return Dict{String, Any}(
                    "tree_id" => metrics.tree_id,
                    "current_metrics" => Dict(
                        "iteration_count" => metrics.iteration_count,
                        "depth_mean" => metrics.depth_mean,
                        "depth_max" => metrics.depth_max,
                        "depth_std" => metrics.depth_std,
                        "nodes_expanded" => metrics.nodes_expanded,
                        "nodes_visited" => metrics.nodes_visited,
                        "expansion_rate" => metrics.expansion_rate,
                        "visit_rate" => metrics.visit_rate,
                        "features_selected" => metrics.features_selected,
                        "features_rejected" => metrics.features_rejected,
                        "selection_diversity" => metrics.selection_diversity,
                        "performance_score" => metrics.performance_score,
                        "convergence_rate" => metrics.convergence_rate,
                        "memory_usage_mb" => metrics.memory_usage_mb,
                        "iterations_per_second" => metrics.iterations_per_second,
                        "gpu_assignment" => metrics.gpu_assignment,
                        "is_active" => metrics.is_active,
                        "total_execution_time" => metrics.total_execution_time
                    ),
                    "history_length" => length(history),
                    "performance_trend" => length(history) > 1 ? 
                        (history[end].performance_score - history[1].performance_score) / length(history) : 0.0
                )
            else
                return Dict{String, Any}("error" => "Tree $tree_id not found")
            end
        end
    end
end

"""
Export performance metrics to file
"""
function export_performance_metrics(monitor::PerformanceMonitor, filepath::String = "performance_export.json")
    summary = get_performance_summary(monitor)
    tree_report = get_tree_performance_report(monitor)
    
    export_data = Dict{String, Any}(
        "export_timestamp" => now(),
        "performance_summary" => summary,
        "tree_performance" => tree_report,
        "gpu_metrics" => Dict(id => Dict(
            "device_id" => metrics.device_id,
            "utilization_percentage" => metrics.utilization_percentage,
            "memory_used_mb" => metrics.memory_used_mb,
            "memory_utilization_percentage" => metrics.memory_utilization_percentage,
            "temperature_celsius" => metrics.temperature_celsius,
            "power_draw_watts" => metrics.power_draw_watts,
            "is_healthy" => metrics.is_healthy,
            "last_update" => metrics.last_update,
            "update_count" => metrics.update_count
        ) for (id, metrics) in monitor.gpu_metrics),
        "active_alerts" => [Dict(
            "alert_id" => alert.alert_id,
            "alert_type" => alert.alert_type,
            "severity" => alert.severity,
            "message" => alert.message,
            "timestamp" => alert.timestamp,
            "device_id" => alert.device_id,
            "tree_id" => alert.tree_id,
            "metric_value" => alert.metric_value,
            "threshold_value" => alert.threshold_value,
            "is_resolved" => alert.is_resolved
        ) for alert in monitor.active_alerts]
    )
    
    # Write to file (simplified JSON export)
    open(filepath, "w") do io
        write(io, string(export_data))
    end
    
    @info "Performance metrics exported to $filepath"
    return filepath
end

"""
Generate performance monitoring report
"""
function generate_performance_report(monitor::PerformanceMonitor)
    summary = get_performance_summary(monitor)
    
    report = String[]
    
    push!(report, "=== Performance Monitoring Report ===")
    push!(report, "Generated: $(now())")
    push!(report, "Monitor State: $(summary["monitor_state"])")
    push!(report, "Monitoring Duration: $(round(summary["monitoring_duration_seconds"], digits=2)) seconds")
    push!(report, "")
    
    # Ensemble performance
    push!(report, "Ensemble Performance:")
    push!(report, "  Total Trees: $(summary["total_trees"])")
    push!(report, "  Active Trees: $(summary["active_trees"])")
    push!(report, "  Total Iterations: $(summary["total_iterations"])")
    push!(report, "  Iterations/Second: $(round(summary["iterations_per_second"], digits=2))")
    push!(report, "  Average Tree Depth: $(round(summary["average_tree_depth"], digits=2))")
    push!(report, "  Ensemble Diversity: $(round(summary["ensemble_diversity_score"], digits=3))")
    push!(report, "")
    
    # GPU utilization
    push!(report, "GPU Utilization:")
    push!(report, "  GPU 0 Utilization: $(round(summary["gpu0_utilization"], digits=1))%")
    push!(report, "  GPU 1 Utilization: $(round(summary["gpu1_utilization"], digits=1))%")
    push!(report, "  GPU 0 Memory: $(round(summary["gpu0_memory_usage_mb"], digits=0)) MB")
    push!(report, "  GPU 1 Memory: $(round(summary["gpu1_memory_usage_mb"], digits=0)) MB")
    push!(report, "  GPU 0 Temperature: $(round(summary["gpu0_temperature"], digits=1))°C")
    push!(report, "  GPU 1 Temperature: $(round(summary["gpu1_temperature"], digits=1))°C")
    push!(report, "")
    
    # Resource utilization
    push!(report, "Resource Utilization:")
    push!(report, "  Total Memory Usage: $(round(summary["total_memory_usage_mb"], digits=0)) MB")
    push!(report, "")
    
    # Alerts
    push!(report, "Alert Summary:")
    push!(report, "  Active Alerts: $(summary["active_alerts_count"])")
    push!(report, "  Critical Alerts: $(summary["critical_alerts_count"])")
    push!(report, "  Warning Alerts: $(summary["warning_alerts_count"])")
    
    if !isempty(monitor.active_alerts)
        push!(report, "")
        push!(report, "Recent Alerts:")
        for alert in monitor.active_alerts[max(1, end-4):end]
            push!(report, "  [$(alert.severity)] $(alert.message)")
        end
    end
    
    push!(report, "")
    push!(report, "=== End Performance Report ===")
    
    return join(report, "\n")
end

"""
Cleanup performance monitor
"""
function cleanup_performance_monitor!(monitor::PerformanceMonitor)
    stop_monitoring!(monitor)
    
    # Clear all stored data
    empty!(monitor.gpu_metrics)
    empty!(monitor.tree_metrics)
    empty!(monitor.gpu_metrics_history)
    empty!(monitor.tree_metrics_history)
    empty!(monitor.ensemble_stats_history)
    empty!(monitor.active_alerts)
    empty!(monitor.alert_history)
    empty!(monitor.cached_metrics)
    empty!(monitor.cache_timestamps)
    empty!(monitor.error_log)
    
    monitor.monitor_state = "shutdown"
    @info "Performance monitor cleaned up"
end

# Export main types and functions
export GPUPerformanceMetrics, TreePerformanceMetrics, EnsemblePerformanceStats
export PerformanceMonitoringConfig, PerformanceAlert, PerformanceMonitor
export create_gpu_performance_metrics, create_tree_performance_metrics, initialize_ensemble_performance_stats
export create_performance_alert, create_performance_monitoring_config, initialize_performance_monitor
export collect_gpu_metrics!, update_tree_metrics!, run_monitoring_cycle!, start_monitoring!, stop_monitoring!
export get_performance_summary, get_tree_performance_report, check_gpu_alerts!, add_alert!, update_ensemble_stats!
export export_performance_metrics, generate_performance_report, cleanup_performance_monitor!

end # module PerformanceMonitoring