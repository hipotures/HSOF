module GPUProfiler

using CUDA
using Statistics
using Dates

export GPUProfileResult, GPUMetrics, profile_gpu_operation
export start_gpu_monitoring, stop_gpu_monitoring, get_gpu_metrics
export CUDAEventTimer, start_timer!, stop_timer!, elapsed_time

"""
GPU metrics snapshot
"""
struct GPUMetrics
    device_id::Int
    timestamp::DateTime
    utilization_percent::Float64
    memory_used_mb::Float64
    memory_total_mb::Float64
    memory_percent::Float64
    temperature_celsius::Float64
    power_watts::Float64
    sm_clock_mhz::Int
    memory_clock_mhz::Int
    pcie_throughput_mb::Float64
end

"""
Result of GPU profiling
"""
struct GPUProfileResult
    operation_name::String
    device_id::Int
    execution_time_ms::Float64
    memory_allocated_mb::Float64
    memory_peak_mb::Float64
    kernel_launches::Int
    metrics_before::GPUMetrics
    metrics_after::GPUMetrics
    metrics_during::Vector{GPUMetrics}
end

"""
CUDA event-based timer for precise GPU timing
"""
mutable struct CUDAEventTimer
    start_event::CUDA.CuEvent
    stop_event::CUDA.CuEvent
    started::Bool
    device::CuDevice
end

function CUDAEventTimer(device_id::Int = 0)
    device = CuDevice(device_id)
    CUDA.device!(device)
    
    return CUDAEventTimer(
        CUDA.CuEvent(CUDA.EVENT_DEFAULT),
        CUDA.CuEvent(CUDA.EVENT_DEFAULT),
        false,
        device
    )
end

function start_timer!(timer::CUDAEventTimer)
    CUDA.device!(timer.device)
    CUDA.record(timer.start_event)
    timer.started = true
end

function stop_timer!(timer::CUDAEventTimer)
    if !timer.started
        error("Timer not started")
    end
    
    CUDA.device!(timer.device)
    CUDA.record(timer.stop_event)
    CUDA.synchronize(timer.stop_event)
    timer.started = false
end

function elapsed_time(timer::CUDAEventTimer)
    if timer.started
        error("Timer still running")
    end
    
    return CUDA.elapsed(timer.start_event, timer.stop_event)
end

"""
Get current GPU metrics using NVML
"""
function get_gpu_metrics(device_id::Int = 0)::GPUMetrics
    device = CuDevice(device_id)
    
    # Get basic device properties
    props = CUDA.properties(device)
    total_memory = CUDA.totalmem(device) / 1024^2  # Convert to MB
    
    # Get current memory usage
    free_memory = CUDA.available_memory() / 1024^2
    used_memory = total_memory - free_memory
    memory_percent = (used_memory / total_memory) * 100
    
    # Get utilization (requires NVML)
    utilization = try
        CUDA.utilization()
    catch
        0.0  # Default if NVML not available
    end
    
    # Get temperature
    temperature = try
        CUDA.temperature(device)
    catch
        0.0  # Default if not available
    end
    
    # Get power usage
    power = try
        CUDA.power_usage(device) / 1000.0  # Convert to watts
    catch
        0.0  # Default if not available
    end
    
    # Get clock speeds
    sm_clock = try
        CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_CLOCK_RATE) ÷ 1000  # Convert to MHz
    catch
        props.clockrate ÷ 1000
    end
    
    memory_clock = try
        CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE) ÷ 1000
    catch
        0
    end
    
    # PCIe throughput (simplified - would need more complex monitoring)
    pcie_throughput = 0.0
    
    return GPUMetrics(
        device_id,
        now(),
        utilization,
        used_memory,
        total_memory,
        memory_percent,
        Float64(temperature),
        power,
        sm_clock,
        memory_clock,
        pcie_throughput
    )
end

"""
Monitoring state
"""
mutable struct GPUMonitor
    device_id::Int
    monitoring::Bool
    interval_ms::Int
    metrics::Vector{GPUMetrics}
    task::Union{Task, Nothing}
end

# Global monitors
const GPU_MONITORS = Dict{Int, GPUMonitor}()

"""
Start continuous GPU monitoring
"""
function start_gpu_monitoring(device_id::Int = 0; interval_ms::Int = 100)
    if haskey(GPU_MONITORS, device_id) && GPU_MONITORS[device_id].monitoring
        @warn "GPU $device_id already being monitored"
        return
    end
    
    monitor = GPUMonitor(device_id, true, interval_ms, GPUMetrics[], nothing)
    
    # Start monitoring task
    monitor.task = @async begin
        while monitor.monitoring
            try
                push!(monitor.metrics, get_gpu_metrics(device_id))
                sleep(monitor.interval_ms / 1000.0)
            catch e
                @warn "Error monitoring GPU $device_id: $e"
                break
            end
        end
    end
    
    GPU_MONITORS[device_id] = monitor
end

"""
Stop GPU monitoring and return collected metrics
"""
function stop_gpu_monitoring(device_id::Int = 0)::Vector{GPUMetrics}
    if !haskey(GPU_MONITORS, device_id)
        @warn "GPU $device_id not being monitored"
        return GPUMetrics[]
    end
    
    monitor = GPU_MONITORS[device_id]
    monitor.monitoring = false
    
    # Wait for monitoring task to finish
    if !isnothing(monitor.task)
        wait(monitor.task)
    end
    
    metrics = monitor.metrics
    delete!(GPU_MONITORS, device_id)
    
    return metrics
end

"""
Profile a GPU operation with detailed metrics
"""
function profile_gpu_operation(
    operation_name::String,
    operation::Function;
    device_id::Int = 0,
    monitor_interval_ms::Int = 50
)::GPUProfileResult
    
    # Select device
    device = CuDevice(device_id)
    CUDA.device!(device)
    
    # Get initial metrics
    metrics_before = get_gpu_metrics(device_id)
    initial_memory = CUDA.memory_stats().live_bytes / 1024^2
    
    # Start monitoring
    start_gpu_monitoring(device_id, interval_ms=monitor_interval_ms)
    
    # Create timer
    timer = CUDAEventTimer(device_id)
    
    # Count kernel launches
    initial_launches = CUDA.launch_count()
    
    # Run operation
    start_timer!(timer)
    result = operation()
    stop_timer!(timer)
    
    # Get execution time
    execution_time_ms = elapsed_time(timer)
    
    # Stop monitoring
    metrics_during = stop_gpu_monitoring(device_id)
    
    # Get final metrics
    metrics_after = get_gpu_metrics(device_id)
    final_memory = CUDA.memory_stats().live_bytes / 1024^2
    peak_memory = CUDA.memory_stats().peak_bytes / 1024^2
    
    # Calculate memory allocated
    memory_allocated = final_memory - initial_memory
    
    # Count kernel launches
    kernel_launches = CUDA.launch_count() - initial_launches
    
    return GPUProfileResult(
        operation_name,
        device_id,
        execution_time_ms,
        memory_allocated,
        peak_memory,
        kernel_launches,
        metrics_before,
        metrics_after,
        metrics_during
    )
end

"""
Analyze GPU profile results
"""
function analyze_profile(result::GPUProfileResult)::Dict{String, Any}
    metrics = result.metrics_during
    
    if isempty(metrics)
        return Dict(
            "execution_time_ms" => result.execution_time_ms,
            "memory_allocated_mb" => result.memory_allocated_mb,
            "kernel_launches" => result.kernel_launches,
            "avg_utilization" => 0.0,
            "peak_utilization" => 0.0,
            "avg_power_watts" => 0.0,
            "peak_temperature" => 0.0
        )
    end
    
    utilizations = [m.utilization_percent for m in metrics]
    powers = [m.power_watts for m in metrics]
    temperatures = [m.temperature_celsius for m in metrics]
    
    return Dict(
        "execution_time_ms" => result.execution_time_ms,
        "memory_allocated_mb" => result.memory_allocated_mb,
        "memory_peak_mb" => result.memory_peak_mb,
        "kernel_launches" => result.kernel_launches,
        "avg_utilization" => mean(utilizations),
        "peak_utilization" => maximum(utilizations),
        "min_utilization" => minimum(utilizations),
        "avg_power_watts" => mean(powers),
        "peak_power_watts" => maximum(powers),
        "avg_temperature" => mean(temperatures),
        "peak_temperature" => maximum(temperatures),
        "utilization_samples" => length(metrics)
    )
end

"""
Profile multiple GPU operations and compare
"""
function profile_comparison(operations::Dict{String, Function}; device_id::Int = 0)
    results = Dict{String, GPUProfileResult}()
    analyses = Dict{String, Dict{String, Any}}()
    
    for (name, op) in operations
        println("Profiling: $name")
        result = profile_gpu_operation(name, op, device_id=device_id)
        results[name] = result
        analyses[name] = analyze_profile(result)
    end
    
    return results, analyses
end

"""
Generate performance summary
"""
function performance_summary(analyses::Dict{String, Dict{String, Any}})
    println("\nGPU Performance Summary")
    println("=" ^ 80)
    
    # Header
    println("Operation                | Time (ms) | Memory (MB) | Avg Util% | Peak Util% | Kernels")
    println("-" * 80)
    
    # Results
    for (name, analysis) in analyses
        @printf("%-23s | %9.2f | %11.2f | %9.1f | %10.1f | %7d\n",
            name,
            analysis["execution_time_ms"],
            analysis["memory_allocated_mb"],
            analysis["avg_utilization"],
            analysis["peak_utilization"],
            analysis["kernel_launches"]
        )
    end
    
    println("-" * 80)
end

end # module