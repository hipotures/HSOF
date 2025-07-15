module GPUMonitor

using CUDA
using Dates
using Printf
using Statistics
using LinearAlgebra

export GPUMetrics, GPUMonitorConfig, GPUMonitorState
export create_gpu_monitor, update_metrics!, get_current_metrics, get_historical_metrics
export select_gpu!, get_available_gpus, is_gpu_available
export start_monitoring!, stop_monitoring!, reset_metrics!, get_metric_history

"""
GPU metrics data structure containing all monitored values
"""
mutable struct GPUMetrics
    gpu_id::Int                    # GPU device ID
    name::String                   # GPU name/model
    utilization::Float64           # GPU utilization (0-100%) - simulated
    memory_used::Float64           # Memory used in MB
    memory_total::Float64          # Total memory in MB
    memory_percent::Float64        # Memory usage percentage
    temperature::Float64           # Temperature in Celsius - simulated
    power_draw::Float64           # Power draw in Watts - simulated
    power_limit::Float64          # Power limit in Watts
    clock_graphics::Float64       # Graphics clock speed in MHz
    clock_memory::Float64         # Memory clock speed in MHz  
    clock_speed::Float64          # Clock speed in MHz - simulated (legacy)
    fan_speed::Float64            # Fan speed (0-100%) - simulated
    timestamp::DateTime           # When metrics were collected
    is_available::Bool            # Whether GPU is available
    error_message::Union{Nothing, String}  # Error if metrics collection failed
end

"""
Configuration for GPU monitoring
"""
struct GPUMonitorConfig
    poll_interval_ms::Int         # How often to poll GPU metrics
    history_size::Int             # Number of historical data points to keep
    history_duration_sec::Int     # Duration of history window in seconds (60 for sparklines)
    enable_caching::Bool          # Whether to cache metrics between polls
    cache_duration_ms::Int        # How long to cache metrics
    simulate_metrics::Bool        # Whether to simulate metrics (for systems without real monitoring)
    
    function GPUMonitorConfig(;
        poll_interval_ms::Int = 100,
        history_size::Int = 600,      # 60 seconds at 100ms intervals
        history_duration_sec::Int = 60,
        enable_caching::Bool = true,
        cache_duration_ms::Int = 50,  # Cache for half the poll interval
        simulate_metrics::Bool = true  # Default to simulation since no NVML
    )
        @assert poll_interval_ms > 0 "Poll interval must be positive"
        @assert history_size > 0 "History size must be positive"
        @assert history_duration_sec > 0 "History duration must be positive"
        @assert cache_duration_ms >= 0 "Cache duration must be non-negative"
        
        new(poll_interval_ms, history_size, history_duration_sec,
            enable_caching, cache_duration_ms, simulate_metrics)
    end
end

"""
GPU monitor state managing metrics collection
"""
mutable struct GPUMonitorState
    config::GPUMonitorConfig
    devices::Vector{Int}                    # Available GPU devices
    selected_device::Int                    # Currently selected device
    current_metrics::Dict{Int, GPUMetrics} # Current metrics per GPU
    metrics_history::Dict{Int, Vector{GPUMetrics}}  # Historical metrics
    last_poll_time::Dict{Int, DateTime}    # Last poll time per GPU
    is_monitoring::Bool                     # Whether monitoring is active
    monitor_task::Union{Task, Nothing}      # Background monitoring task
    error_count::Dict{Int, Int}            # Error count per GPU
    baseline_values::Dict{Int, NamedTuple} # Baseline values for simulation
    
    function GPUMonitorState(config::GPUMonitorConfig)
        # Detect available GPUs
        devices = Int[]
        if CUDA.functional()
            try
                device_count = length(CUDA.devices())
                devices = collect(0:device_count-1)
            catch
                # No GPUs available
            end
        end
        
        selected_device = isempty(devices) ? -1 : devices[1]
        current_metrics = Dict{Int, GPUMetrics}()
        metrics_history = Dict{Int, Vector{GPUMetrics}}()
        last_poll_time = Dict{Int, DateTime}()
        error_count = Dict{Int, Int}()
        baseline_values = Dict{Int, NamedTuple}()
        
        # Initialize for each device
        for device_id in devices
            metrics_history[device_id] = GPUMetrics[]
            last_poll_time[device_id] = DateTime(0)
            error_count[device_id] = 0
            
            # Set baseline values for simulation
            baseline_values[device_id] = (
                utilization = 50.0 + device_id * 5.0,
                temperature = 60.0 + device_id * 2.0,
                power = 200.0 + device_id * 20.0,
                clock = 1800.0 + device_id * 50.0,
                fan = 40.0 + device_id * 5.0
            )
        end
        
        new(config, devices, selected_device, current_metrics, metrics_history,
            last_poll_time, false, nothing, error_count, baseline_values)
    end
end

"""
Create a new GPU monitor
"""
function create_gpu_monitor(config::GPUMonitorConfig = GPUMonitorConfig())
    monitor = GPUMonitorState(config)
    
    # Collect initial metrics for all devices
    for device_id in monitor.devices
        collect_gpu_metrics!(monitor, device_id)
    end
    
    return monitor
end

"""
Collect GPU metrics for a specific device
"""
function collect_gpu_metrics!(monitor::GPUMonitorState, device_id::Int)
    # Check if caching is enabled and metrics are fresh
    if monitor.config.enable_caching && haskey(monitor.last_poll_time, device_id)
        time_since_poll = Dates.value(now() - monitor.last_poll_time[device_id])
        if time_since_poll < monitor.config.cache_duration_ms
            return monitor.current_metrics[device_id]
        end
    end
    
    # Get GPU name
    gpu_name = "Unknown GPU"
    if CUDA.functional() && device_id in 0:length(CUDA.devices())-1
        try
            CUDA.device!(device_id)
            gpu_name = CUDA.name(CUDA.device())
        catch
            # Fallback name
        end
    end
    
    metrics = GPUMetrics(
        device_id,                # gpu_id
        gpu_name,                 # name
        0.0,                     # utilization
        0.0,                     # memory_used (MB)
        0.0,                     # memory_total (MB)
        0.0,                     # memory_percent
        0.0,                     # temperature
        0.0,                     # power_draw
        300.0,                   # power_limit (default)
        0.0,                     # clock_graphics
        0.0,                     # clock_memory
        0.0,                     # clock_speed (legacy)
        0.0,                     # fan_speed
        now(),                   # timestamp
        false,                   # is_available
        nothing                  # error_message
    )
    
    if device_id in monitor.devices && CUDA.functional()
        try
            # Switch to the target device
            old_device = CUDA.device()
            CUDA.device!(device_id)
            
            # Get real memory metrics
            free_mem = CUDA.available_memory() / 1e6  # Convert to MB
            total_mem = CUDA.total_memory() / 1e6     # Convert to MB
            used_mem = total_mem - free_mem
            mem_percent = (used_mem / total_mem) * 100.0
            
            metrics.memory_used = used_mem
            metrics.memory_total = total_mem
            metrics.memory_percent = mem_percent
            metrics.is_available = true
            
            # Simulate other metrics since CUDA.jl doesn't provide them
            if monitor.config.simulate_metrics
                simulate_gpu_metrics!(metrics, monitor, device_id)
            end
            
            # Restore original device
            CUDA.device!(old_device)
            
            # Reset error count on success
            monitor.error_count[device_id] = 0
            
        catch e
            metrics.is_available = false
            metrics.error_message = string(e)
            monitor.error_count[device_id] += 1
        end
    else
        metrics.is_available = false
        metrics.error_message = "GPU not available"
    end
    
    # Update current metrics and history
    monitor.current_metrics[device_id] = metrics
    monitor.last_poll_time[device_id] = now()
    
    # Add to history
    history = monitor.metrics_history[device_id]
    push!(history, metrics)
    
    # Trim history to configured size
    if length(history) > monitor.config.history_size
        deleteat!(history, 1:length(history) - monitor.config.history_size)
    end
    
    # Also trim by time window
    cutoff_time = now() - Second(monitor.config.history_duration_sec)
    filter!(m -> m.timestamp > cutoff_time, history)
    
    return metrics
end

"""
Simulate GPU metrics for values not available through CUDA.jl
"""
function simulate_gpu_metrics!(metrics::GPUMetrics, monitor::GPUMonitorState, device_id::Int)
    baseline = monitor.baseline_values[device_id]
    
    # Generate realistic variations around baseline
    time_factor = sin(Dates.value(now()) / 10000.0)  # Slow oscillation
    noise_factor = 0.1  # 10% noise
    
    # Utilization correlates with memory usage
    memory_factor = metrics.memory_percent / 100.0
    metrics.utilization = baseline.utilization + 20.0 * memory_factor + 
                         10.0 * time_factor + randn() * 5.0 * noise_factor
    metrics.utilization = clamp(metrics.utilization, 0.0, 100.0)
    
    # Temperature follows utilization with some lag
    temp_target = 40.0 + metrics.utilization * 0.5
    metrics.temperature = baseline.temperature * 0.7 + temp_target * 0.3 +
                         5.0 * time_factor + randn() * 2.0 * noise_factor
    metrics.temperature = clamp(metrics.temperature, 30.0, 95.0)
    
    # Power draw correlates with utilization
    power_target = 100.0 + metrics.utilization * 3.5
    metrics.power_draw = baseline.power * 0.8 + power_target * 0.2 +
                        20.0 * time_factor + randn() * 10.0 * noise_factor
    metrics.power_draw = clamp(metrics.power_draw, 50.0, 450.0)
    
    # Clock speed inversely related to temperature
    clock_factor = 1.0 - (metrics.temperature - 60.0) / 100.0
    metrics.clock_speed = baseline.clock * clock_factor +
                         100.0 * time_factor + randn() * 50.0 * noise_factor
    metrics.clock_speed = clamp(metrics.clock_speed, 1200.0, 2500.0)
    
    # Graphics and memory clocks
    metrics.clock_graphics = metrics.clock_speed
    metrics.clock_memory = 8000.0 + 1000.0 * time_factor + randn() * 100.0 * noise_factor
    metrics.clock_memory = clamp(metrics.clock_memory, 7000.0, 11000.0)
    
    # Power limit (static for simulation)
    metrics.power_limit = 350.0  # Typical for high-end GPUs
    
    # Fan speed follows temperature
    fan_target = 20.0 + (metrics.temperature - 40.0) * 1.5
    metrics.fan_speed = baseline.fan * 0.6 + fan_target * 0.4 +
                       5.0 * time_factor + randn() * 3.0 * noise_factor
    metrics.fan_speed = clamp(metrics.fan_speed, 0.0, 100.0)
end

"""
Update metrics for all GPUs
"""
function update_metrics!(monitor::GPUMonitorState)
    for device_id in monitor.devices
        collect_gpu_metrics!(monitor, device_id)
    end
end

"""
Get current metrics for a specific GPU
"""
function get_current_metrics(monitor::GPUMonitorState, device_id::Int)
    if haskey(monitor.current_metrics, device_id)
        return monitor.current_metrics[device_id]
    else
        return nothing
    end
end

"""
Get current metrics for selected GPU
"""
function get_current_metrics(monitor::GPUMonitorState)
    return get_current_metrics(monitor, monitor.selected_device)
end

"""
Get historical metrics for a specific GPU
"""
function get_historical_metrics(monitor::GPUMonitorState, device_id::Int)
    if haskey(monitor.metrics_history, device_id)
        return monitor.metrics_history[device_id]
    else
        return GPUMetrics[]
    end
end

"""
Get historical metrics for selected GPU
"""
function get_historical_metrics(monitor::GPUMonitorState)
    return get_historical_metrics(monitor, monitor.selected_device)
end

"""
Select a GPU for monitoring
"""
function select_gpu!(monitor::GPUMonitorState, device_id::Int)
    if device_id in monitor.devices
        monitor.selected_device = device_id
        return true
    else
        return false
    end
end

"""
Get list of available GPUs
"""
function get_available_gpus(monitor::GPUMonitorState)
    gpu_info = []
    for device_id in monitor.devices
        if CUDA.functional()
            try
                old_device = CUDA.device()
                CUDA.device!(device_id)
                name = CUDA.name(CUDA.device())
                CUDA.device!(old_device)
                push!(gpu_info, (id=device_id, name=name))
            catch
                push!(gpu_info, (id=device_id, name="Unknown GPU $device_id"))
            end
        end
    end
    return gpu_info
end

"""
Check if a GPU is available
"""
function is_gpu_available(monitor::GPUMonitorState, device_id::Int)
    return device_id in monitor.devices && 
           haskey(monitor.current_metrics, device_id) &&
           monitor.current_metrics[device_id].is_available
end

"""
Start background monitoring
"""
function start_monitoring!(monitor::GPUMonitorState)
    if monitor.is_monitoring
        @warn "Monitoring is already active"
        return
    end
    
    monitor.is_monitoring = true
    
    # Start background task
    monitor.monitor_task = @async begin
        while monitor.is_monitoring
            try
                update_metrics!(monitor)
                sleep(monitor.config.poll_interval_ms / 1000.0)
            catch e
                @error "Error in GPU monitoring" exception=e
                sleep(1.0)  # Back off on error
            end
        end
    end
    
    @info "GPU monitoring started"
end

"""
Stop background monitoring
"""
function stop_monitoring!(monitor::GPUMonitorState)
    if !monitor.is_monitoring
        @warn "Monitoring is not active"
        return
    end
    
    monitor.is_monitoring = false
    
    # Wait for task to finish
    if !isnothing(monitor.monitor_task)
        wait(monitor.monitor_task)
        # Keep the task reference for testing
        # monitor.monitor_task = nothing
    end
    
    @info "GPU monitoring stopped"
end

"""
Reset all metrics and history
"""
function reset_metrics!(monitor::GPUMonitorState)
    # Clear current metrics
    empty!(monitor.current_metrics)
    
    # Clear history
    for device_id in keys(monitor.metrics_history)
        empty!(monitor.metrics_history[device_id])
    end
    
    # Reset error counts
    for device_id in keys(monitor.error_count)
        monitor.error_count[device_id] = 0
    end
    
    # Reset poll times
    for device_id in keys(monitor.last_poll_time)
        monitor.last_poll_time[device_id] = DateTime(0)
    end
    
    @info "GPU metrics reset"
end

"""
Format GPU metrics for display
"""
function Base.show(io::IO, metrics::GPUMetrics)
    if metrics.is_available
        print(io, @sprintf(
            "GPU %d: Util %.1f%% | Mem %.1f/%.1f GB (%.1f%%) | Temp %.0f°C | Power %.0fW | Clock %.0f MHz | Fan %.0f%%",
            metrics.gpu_id,
            metrics.utilization,
            metrics.memory_used,
            metrics.memory_total,
            metrics.memory_percent,
            metrics.temperature,
            metrics.power_draw,
            metrics.clock_speed,
            metrics.fan_speed
        ))
    else
        print(io, "GPU $(metrics.gpu_id): Not available")
        if !isnothing(metrics.error_message)
            print(io, " - $(metrics.error_message)")
        end
    end
end

"""
Get history of a specific metric for a GPU
"""
function get_metric_history(monitor::GPUMonitorState, gpu_idx::Int, metric::Symbol)
    if !haskey(monitor.metrics_history, gpu_idx)
        return Float64[]
    end
    
    history = monitor.metrics_history[gpu_idx]
    
    # Extract the requested metric from history
    if metric == :utilization
        return [m.utilization for m in history]
    elseif metric == :temperature
        return [m.temperature for m in history]
    elseif metric == :memory_percent
        return [m.memory_percent for m in history]
    elseif metric == :power_draw
        return [m.power_draw for m in history]
    elseif metric == :clock_speed
        return [m.clock_speed for m in history]
    else
        error("Unknown metric: $metric")
    end
end

end # module