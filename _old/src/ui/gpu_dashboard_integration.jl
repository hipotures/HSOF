module GPUDashboardIntegration

using Dates

# Include required modules
include("gpu_monitor.jl")
using .GPUMonitor

include("console_dashboard.jl")
using .ConsoleDashboard

export create_integrated_dashboard, update_gpu_panels!, create_gpu_content_from_metrics

"""
Create console dashboard with integrated GPU monitoring
"""
function create_integrated_dashboard(gpu_monitor::GPUMonitorState, config::DashboardConfig = DashboardConfig())
    dashboard = create_dashboard(config)
    
    # Update GPU panels with real metrics
    update_gpu_panels!(dashboard, gpu_monitor)
    
    return dashboard
end

"""
Update dashboard GPU panels with current metrics
"""
function update_gpu_panels!(dashboard::DashboardLayout, gpu_monitor::GPUMonitorState)
    updates = Dict{Symbol, PanelContent}()
    
    # Update GPU 1 panel
    if length(gpu_monitor.devices) >= 1
        device_id = gpu_monitor.devices[1]
        metrics = get_current_metrics(gpu_monitor, device_id)
        if !isnothing(metrics) && metrics.is_available
            updates[:gpu1] = create_gpu_content_from_metrics(metrics, 1)
        end
    end
    
    # Update GPU 2 panel if available
    if length(gpu_monitor.devices) >= 2
        device_id = gpu_monitor.devices[2]
        metrics = get_current_metrics(gpu_monitor, device_id)
        if !isnothing(metrics) && metrics.is_available
            updates[:gpu2] = create_gpu_content_from_metrics(metrics, 2)
        end
    elseif length(gpu_monitor.devices) == 1
        # Single GPU system - show placeholder for GPU 2
        updates[:gpu2] = GPUPanelContent(2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    end
    
    # Apply updates
    if !isempty(updates)
        update_dashboard!(dashboard, updates)
    end
end

"""
Convert GPU metrics to dashboard panel content
"""
function create_gpu_content_from_metrics(metrics::GPUMetrics, panel_gpu_id::Int)
    return GPUPanelContent(
        panel_gpu_id,                # Panel ID (1 or 2)
        metrics.utilization,         # Utilization percentage
        metrics.memory_used,         # Memory used in GB
        metrics.memory_total,        # Memory total in GB
        metrics.temperature,         # Temperature in Celsius
        metrics.power_draw,          # Power draw in Watts
        metrics.clock_speed,         # Clock speed in MHz
        metrics.fan_speed           # Fan speed percentage
    )
end

"""
Create sparkline data from historical metrics
"""
function create_sparkline_data(history::Vector{GPUMetrics}, metric::Symbol, window_size::Int = 60)
    # Get last window_size data points
    start_idx = max(1, length(history) - window_size + 1)
    data_points = Float64[]
    
    for i in start_idx:length(history)
        metrics = history[i]
        if metrics.is_available
            value = getfield(metrics, metric)
            push!(data_points, value)
        end
    end
    
    return data_points
end

"""
Calculate trend direction from historical data
"""
function calculate_trend(history::Vector{GPUMetrics}, metric::Symbol, window_size::Int = 10)
    if length(history) < 2
        return :stable
    end
    
    # Get recent values
    recent_data = create_sparkline_data(history, metric, window_size)
    
    if length(recent_data) < 2
        return :stable
    end
    
    # Simple linear regression
    n = length(recent_data)
    x = collect(1:n)
    y = recent_data
    
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in 1:n)
    denominator = sum((x[i] - x_mean)^2 for i in 1:n)
    
    if denominator ≈ 0
        return :stable
    end
    
    slope = numerator / denominator
    
    # Determine trend based on slope
    if slope > 0.5
        return :increasing
    elseif slope < -0.5
        return :decreasing
    else
        return :stable
    end
end

"""
Format metrics history for log panel
"""
function format_gpu_events(gpu_monitor::GPUMonitorState, max_entries::Int = 10)
    events = Tuple{DateTime, Symbol, String}[]
    
    # Check for recent errors
    for (device_id, error_count) in gpu_monitor.error_count
        if error_count > 0
            push!(events, (now(), :error, "GPU $device_id: $error_count errors"))
        end
    end
    
    # Check for high utilization
    for (device_id, metrics) in gpu_monitor.current_metrics
        if metrics.is_available
            if metrics.utilization > 95.0
                push!(events, (now(), :warn, "GPU $device_id: High utilization $(round(metrics.utilization, digits=1))%"))
            end
            if metrics.temperature > 85.0
                push!(events, (now(), :warn, "GPU $device_id: High temperature $(round(metrics.temperature, digits=0))°C"))
            end
            if metrics.memory_percent > 90.0
                push!(events, (now(), :warn, "GPU $device_id: High memory usage $(round(metrics.memory_percent, digits=1))%"))
            end
        end
    end
    
    # Add GPU availability changes
    for device_id in gpu_monitor.devices
        if haskey(gpu_monitor.current_metrics, device_id)
            if !gpu_monitor.current_metrics[device_id].is_available
                push!(events, (now(), :error, "GPU $device_id: Not available"))
            end
        end
    end
    
    # Sort by timestamp and limit
    sort!(events, by=x->x[1], rev=true)
    if length(events) > max_entries
        events = events[1:max_entries]
    end
    
    return events
end

"""
Create a complete monitoring loop that updates dashboard with GPU metrics
"""
function run_integrated_monitoring(dashboard::DashboardLayout, gpu_monitor::GPUMonitorState; 
                                 update_interval_ms::Int = 100)
    # Start GPU monitoring
    start_monitoring!(gpu_monitor)
    
    try
        while true
            # Update GPU panels
            update_gpu_panels!(dashboard, gpu_monitor)
            
            # Sleep for update interval
            sleep(update_interval_ms / 1000.0)
        end
    catch e
        if isa(e, InterruptException)
            @info "Monitoring interrupted"
        else
            rethrow(e)
        end
    finally
        # Stop GPU monitoring
        stop_monitoring!(gpu_monitor)
    end
end

end # module