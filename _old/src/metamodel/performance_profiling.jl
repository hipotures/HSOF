"""
Performance Profiling Framework
Comprehensive profiling system measuring inference latency, training throughput, 
and identifying optimization opportunities with NVIDIA Nsight integration
"""
module PerformanceProfiling

using CUDA
using Statistics
using Dates
using Printf
using JSON3
using HDF5
using DataFrames
using LinearAlgebra
using BenchmarkTools

# Include dependencies for profiling - conditionally load if available
NEURAL_ARCH_AVAILABLE = false
BATCH_INF_AVAILABLE = false  
ONLINE_LEARN_AVAILABLE = false
DIST_TRAIN_AVAILABLE = false

try
    include("neural_architecture.jl")
    using .NeuralArchitecture
    global NEURAL_ARCH_AVAILABLE = true
catch e
    @warn "Neural architecture module not available" exception=e
end

try
    include("batch_inference.jl")
    using .BatchInference
    global BATCH_INF_AVAILABLE = true
catch e
    @warn "Batch inference module not available" exception=e
end

try
    include("online_learning.jl")
    using .OnlineLearning
    global ONLINE_LEARN_AVAILABLE = true
catch e
    @warn "Online learning module not available" exception=e
end

try
    include("distributed_training.jl")
    using .DistributedTraining
    global DIST_TRAIN_AVAILABLE = true
catch e
    @warn "Distributed training module not available" exception=e
end

"""
Configuration for performance profiling system
"""
struct ProfilingConfig
    # Measurement settings
    enable_cuda_events::Bool              # Use CUDA events for GPU timing
    enable_nsight_markers::Bool           # Insert Nsight Systems markers
    enable_memory_profiling::Bool         # Track GPU/CPU memory usage
    enable_power_profiling::Bool          # Monitor GPU power consumption
    
    # Timing granularity
    profile_inference_ops::Bool           # Profile individual inference operations
    profile_training_ops::Bool            # Profile training step components
    profile_memory_ops::Bool              # Profile memory allocations/transfers
    profile_communication_ops::Bool       # Profile inter-GPU communication
    
    # Sampling settings
    sampling_frequency::Float32           # Sampling frequency in Hz
    warmup_iterations::Int                # Warmup iterations before measurement
    measurement_iterations::Int           # Number of measurement iterations
    statistical_confidence::Float32       # Confidence level for measurements
    
    # Reporting settings
    export_format::String                 # Export format: "json", "hdf5", "csv"
    export_path::String                   # Export directory path
    realtime_visualization::Bool          # Enable real-time visualization
    regression_testing::Bool              # Enable automated regression testing
    
    # Performance targets
    inference_latency_target_ms::Float32  # Target inference latency (ms)
    training_throughput_target::Float32   # Target training throughput (samples/sec)
    memory_bandwidth_target_gbps::Float32 # Target memory bandwidth (GB/s)
    scaling_efficiency_target::Float32    # Target multi-GPU scaling efficiency
end

"""
Create default profiling configuration
"""
function create_profiling_config(;
    enable_cuda_events::Bool = true,
    enable_nsight_markers::Bool = true,
    enable_memory_profiling::Bool = true,
    enable_power_profiling::Bool = false,  # Requires nvidia-ml-py
    profile_inference_ops::Bool = true,
    profile_training_ops::Bool = true,
    profile_memory_ops::Bool = true,
    profile_communication_ops::Bool = true,
    sampling_frequency::Float32 = 10.0f0,  # 10 Hz
    warmup_iterations::Int = 50,
    measurement_iterations::Int = 100,
    statistical_confidence::Float32 = 0.95f0,
    export_format::String = "json",
    export_path::String = "profiling_results",
    realtime_visualization::Bool = true,
    regression_testing::Bool = true,
    inference_latency_target_ms::Float32 = 1.0f0,
    training_throughput_target::Float32 = 1000.0f0,
    memory_bandwidth_target_gbps::Float32 = 500.0f0,
    scaling_efficiency_target::Float32 = 0.85f0
)
    return ProfilingConfig(
        enable_cuda_events,
        enable_nsight_markers,
        enable_memory_profiling,
        enable_power_profiling,
        profile_inference_ops,
        profile_training_ops,
        profile_memory_ops,
        profile_communication_ops,
        sampling_frequency,
        warmup_iterations,
        measurement_iterations,
        statistical_confidence,
        export_format,
        export_path,
        realtime_visualization,
        regression_testing,
        inference_latency_target_ms,
        training_throughput_target,
        memory_bandwidth_target_gbps,
        scaling_efficiency_target
    )
end

"""
Performance measurement result
"""
struct PerformanceResult
    operation_name::String
    start_time::Float64
    end_time::Float64
    duration_ms::Float32
    gpu_memory_used::UInt64
    cpu_memory_used::UInt64
    gpu_utilization::Float32
    memory_bandwidth_gbps::Float32
    power_consumption_watts::Float32
    additional_metrics::Dict{String, Any}
end

"""
CUDA event-based timer for precise GPU measurements
"""
mutable struct CUDAEventTimer
    start_event::CuEvent
    end_event::CuEvent
    stream::CuStream
    is_recording::Bool
    
    function CUDAEventTimer(stream::CuStream = CUDA.stream())
        start_event = CuEvent()
        end_event = CuEvent()
        new(start_event, end_event, stream, false)
    end
end

"""
Start CUDA event timing
"""
function start_timer!(timer::CUDAEventTimer)
    if timer.is_recording
        @warn "Timer already recording, stopping previous measurement"
        stop_timer!(timer)
    end
    
    CUDA.record(timer.start_event, timer.stream)
    timer.is_recording = true
end

"""
Stop CUDA event timing and return duration in milliseconds
"""
function stop_timer!(timer::CUDAEventTimer)
    if !timer.is_recording
        @warn "Timer not recording"
        return 0.0f0
    end
    
    CUDA.record(timer.end_event, timer.stream)
    CUDA.synchronize(timer.stream)
    
    duration_ms = Float32(CUDA.elapsed(timer.start_event, timer.end_event))
    timer.is_recording = false
    
    return duration_ms
end

"""
Nsight Systems marker integration
"""
struct NsightMarker
    name::String
    color::UInt32
    
    function NsightMarker(name::String, color::Symbol = :blue)
        color_map = Dict(
            :blue => 0x0000FF,
            :red => 0xFF0000,
            :green => 0x00FF00,
            :yellow => 0xFFFF00,
            :purple => 0xFF00FF,
            :orange => 0xFF8000,
            :cyan => 0x00FFFF
        )
        new(name, get(color_map, color, 0x0000FF))
    end
end

"""
Push Nsight marker (requires NVTX)
"""
function push_nsight_marker(marker::NsightMarker)
    # Note: This requires proper NVTX integration
    # For now, we'll use a simple logging approach
    @debug "NSIGHT_MARKER_PUSH: $(marker.name)"
end

"""
Pop Nsight marker
"""
function pop_nsight_marker()
    @debug "NSIGHT_MARKER_POP"
end

"""
Memory profiler for GPU and CPU usage tracking
"""
mutable struct MemoryProfiler
    initial_gpu_memory::UInt64
    initial_cpu_memory::UInt64
    peak_gpu_memory::UInt64
    peak_cpu_memory::UInt64
    gpu_allocations::Vector{UInt64}
    cpu_allocations::Vector{UInt64}
    timestamps::Vector{Float64}
    
    function MemoryProfiler()
        initial_gpu = UInt64(CUDA.available_memory())
        initial_cpu = UInt64(Sys.free_memory())
        
        new(
            initial_gpu,
            initial_cpu,
            initial_gpu,
            initial_cpu,
            UInt64[],
            UInt64[],
            Float64[]
        )
    end
end

"""
Take memory snapshot
"""
function take_memory_snapshot!(profiler::MemoryProfiler)
    current_gpu = UInt64(CUDA.available_memory())
    current_cpu = UInt64(Sys.free_memory())
    
    # Calculate memory used (handle potential negative values)
    gpu_used = current_gpu < profiler.initial_gpu_memory ? 
               profiler.initial_gpu_memory - current_gpu : UInt64(0)
    cpu_used = current_cpu < profiler.initial_cpu_memory ? 
               profiler.initial_cpu_memory - current_cpu : UInt64(0)
    
    push!(profiler.gpu_allocations, gpu_used)
    push!(profiler.cpu_allocations, cpu_used)
    push!(profiler.timestamps, time())
    
    # Update peaks
    profiler.peak_gpu_memory = max(profiler.peak_gpu_memory, gpu_used)
    profiler.peak_cpu_memory = max(profiler.peak_cpu_memory, cpu_used)
end

"""
Performance profiler coordinator
"""
mutable struct PerformanceProfiler
    config::ProfilingConfig
    cuda_timers::Dict{String, CUDAEventTimer}
    nsight_markers::Dict{String, NsightMarker}
    memory_profiler::MemoryProfiler
    results::Vector{PerformanceResult}
    
    # Real-time monitoring
    current_measurements::Dict{String, Float32}
    measurement_history::Dict{String, Vector{Float32}}
    
    # Regression testing
    baseline_results::Dict{String, Float32}
    regression_threshold::Float32
    
    # Statistics
    total_measurements::Int64
    failed_measurements::Int64
    start_time::Float64
end

"""
Initialize performance profiler
"""
function initialize_profiler(config::ProfilingConfig = create_profiling_config())
    # Create output directory
    mkpath(config.export_path)
    
    # Initialize CUDA timers for common operations
    cuda_timers = Dict{String, CUDAEventTimer}()
    if config.enable_cuda_events
        operations = [
            "inference_forward", "inference_batch", "training_forward",
            "training_backward", "gradient_sync", "memory_copy",
            "attention_computation", "correlation_update"
        ]
        
        for op in operations
            cuda_timers[op] = CUDAEventTimer()
        end
    end
    
    # Initialize Nsight markers
    nsight_markers = Dict{String, NsightMarker}()
    if config.enable_nsight_markers
        nsight_markers["inference"] = NsightMarker("Inference", :blue)
        nsight_markers["training"] = NsightMarker("Training", :red)
        nsight_markers["memory"] = NsightMarker("Memory", :green)
        nsight_markers["communication"] = NsightMarker("Communication", :yellow)
        nsight_markers["synchronization"] = NsightMarker("Synchronization", :purple)
    end
    
    # Initialize memory profiler
    memory_profiler = config.enable_memory_profiling ? MemoryProfiler() : nothing
    
    return PerformanceProfiler(
        config,
        cuda_timers,
        nsight_markers,
        memory_profiler,
        PerformanceResult[],
        Dict{String, Float32}(),
        Dict{String, Vector{Float32}}(),
        Dict{String, Float32}(),
        0.05f0,  # 5% regression threshold
        0,
        0,
        time()
    )
end

"""
Profile a specific operation with comprehensive measurements
"""
function profile_operation!(
    profiler::PerformanceProfiler,
    operation_name::String,
    operation_func::Function;
    category::String = "general",
    expected_duration_ms::Union{Nothing, Float32} = nothing
)
    if profiler.config.enable_nsight_markers && haskey(profiler.nsight_markers, category)
        push_nsight_marker(profiler.nsight_markers[category])
    end
    
    # Pre-operation memory snapshot
    if !isnothing(profiler.memory_profiler)
        take_memory_snapshot!(profiler.memory_profiler)
    end
    
    start_time = time()
    start_gpu_memory = UInt64(CUDA.available_memory())
    start_cpu_memory = UInt64(Sys.free_memory())
    
    # Start CUDA timer if available
    cuda_timer = get(profiler.cuda_timers, operation_name, nothing)
    if !isnothing(cuda_timer)
        start_timer!(cuda_timer)
    end
    
    # Execute operation
    result = try
        operation_func()
    catch e
        profiler.failed_measurements += 1
        @error "Operation $operation_name failed" exception=e
        nothing
    end
    
    # Stop timing
    duration_ms = if !isnothing(cuda_timer)
        stop_timer!(cuda_timer)
    else
        Float32((time() - start_time) * 1000)
    end
    
    end_time = time()
    end_gpu_memory = UInt64(CUDA.available_memory())
    end_cpu_memory = UInt64(Sys.free_memory())
    
    # Calculate memory usage (handle potential negative values)
    gpu_memory_used = end_gpu_memory < start_gpu_memory ? 
                     start_gpu_memory - end_gpu_memory : UInt64(0)
    cpu_memory_used = end_cpu_memory < start_cpu_memory ? 
                     start_cpu_memory - end_cpu_memory : UInt64(0)
    
    # Estimate GPU utilization (simplified)
    gpu_utilization = estimate_gpu_utilization(duration_ms)
    
    # Estimate memory bandwidth
    memory_bandwidth = estimate_memory_bandwidth(gpu_memory_used, duration_ms)
    
    # Create performance result
    perf_result = PerformanceResult(
        operation_name,
        start_time,
        end_time,
        duration_ms,
        gpu_memory_used,
        cpu_memory_used,
        gpu_utilization,
        memory_bandwidth,
        0.0f0,  # Power consumption (requires nvidia-ml-py)
        Dict{String, Any}()
    )
    
    # Store result
    push!(profiler.results, perf_result)
    profiler.total_measurements += 1
    
    # Update real-time measurements
    profiler.current_measurements[operation_name] = duration_ms
    
    if !haskey(profiler.measurement_history, operation_name)
        profiler.measurement_history[operation_name] = Float32[]
    end
    push!(profiler.measurement_history[operation_name], duration_ms)
    
    # Regression testing
    if profiler.config.regression_testing
        check_performance_regression!(profiler, operation_name, duration_ms)
    end
    
    # Post-operation memory snapshot
    if !isnothing(profiler.memory_profiler)
        take_memory_snapshot!(profiler.memory_profiler)
    end
    
    if profiler.config.enable_nsight_markers && haskey(profiler.nsight_markers, category)
        pop_nsight_marker()
    end
    
    return result, perf_result
end

"""
Estimate GPU utilization based on timing
"""
function estimate_gpu_utilization(duration_ms::Float32)
    # Simplified estimation - would require nvidia-ml-py for accurate measurement
    # For now, assume high utilization for operations > 0.1ms
    return duration_ms > 0.1f0 ? 85.0f0 : 20.0f0
end

"""
Estimate memory bandwidth usage
"""
function estimate_memory_bandwidth(memory_bytes::UInt64, duration_ms::Float32)
    if duration_ms <= 0 || memory_bytes <= 0
        return 0.0f0
    end
    
    # Convert to GB/s
    duration_s = duration_ms / 1000.0f0
    memory_gb = Float32(memory_bytes) / (1024^3)
    
    return memory_gb / duration_s
end

"""
Check for performance regression
"""
function check_performance_regression!(
    profiler::PerformanceProfiler,
    operation_name::String,
    current_duration::Float32
)
    if !haskey(profiler.baseline_results, operation_name)
        # Set first measurement as baseline
        profiler.baseline_results[operation_name] = current_duration
        return
    end
    
    baseline = profiler.baseline_results[operation_name]
    regression_ratio = current_duration / baseline
    
    if regression_ratio > (1.0f0 + profiler.regression_threshold)
        @warn "Performance regression detected for $operation_name" baseline=baseline current=current_duration regression_ratio=regression_ratio
    end
end

"""
Benchmark metamodel inference performance
"""
function benchmark_inference(
    profiler::PerformanceProfiler,
    model,
    batch_sizes::Vector{Int} = [1, 16, 32, 64, 128, 256, 512, 1024]
)
    println("Benchmarking metamodel inference performance...")
    
    inference_results = Dict{Int, Dict{String, Float32}}()
    
    for batch_size in batch_sizes
        println("  Testing batch size: $batch_size")
        
        # Create test input
        input_dim = model.config.input_dim
        test_input = CUDA.randn(Float32, input_dim, batch_size)
        
        # Warmup
        for _ in 1:profiler.config.warmup_iterations
            model(test_input)
        end
        CUDA.synchronize()
        
        # Benchmark
        durations = Float32[]
        for _ in 1:profiler.config.measurement_iterations
            _, result = profile_operation!(profiler, "inference_batch_$batch_size", () -> model(test_input), category="inference")
            push!(durations, result.duration_ms)
        end
        
        # Calculate statistics
        mean_duration = mean(durations)
        std_duration = std(durations)
        min_duration = minimum(durations)
        max_duration = maximum(durations)
        throughput = batch_size / (mean_duration / 1000.0f0)  # samples/sec
        
        inference_results[batch_size] = Dict(
            "mean_duration_ms" => mean_duration,
            "std_duration_ms" => std_duration,
            "min_duration_ms" => min_duration,
            "max_duration_ms" => max_duration,
            "throughput_samples_per_sec" => throughput,
            "per_sample_latency_ms" => mean_duration / batch_size
        )
        
        println("    Mean: $(round(mean_duration, digits=3)) ms")
        println("    Throughput: $(round(throughput, digits=1)) samples/sec")
        println("    Per-sample: $(round(mean_duration / batch_size, digits=4)) ms")
    end
    
    return inference_results
end

"""
Benchmark training performance
"""
function benchmark_training(
    profiler::PerformanceProfiler,
    online_state,
    replay_buffer,
    iterations::Int = 100
)
    println("Benchmarking training performance...")
    
    training_durations = Float32[]
    sync_durations = Float32[]
    total_samples = 0
    
    for i in 1:iterations
        # Profile training step
        _, train_result = profile_operation!(profiler, "training_step", () -> begin
            # Simulate training step
            online_update!(online_state, replay_buffer, create_online_config())
        end, category="training")
        
        push!(training_durations, train_result.duration_ms)
        total_samples += 32  # Assume batch size of 32
        
        if i % 10 == 0
            println("  Iteration $i: $(round(train_result.duration_ms, digits=3)) ms")
        end
    end
    
    # Calculate training statistics
    mean_training_duration = mean(training_durations)
    training_throughput = total_samples / (sum(training_durations) / 1000.0f0)
    
    training_results = Dict(
        "mean_training_duration_ms" => mean_training_duration,
        "std_training_duration_ms" => std(training_durations),
        "min_training_duration_ms" => minimum(training_durations),
        "max_training_duration_ms" => maximum(training_durations),
        "training_throughput_samples_per_sec" => training_throughput,
        "total_iterations" => iterations,
        "total_samples" => total_samples
    )
    
    println("Training Results:")
    println("  Mean duration: $(round(mean_training_duration, digits=3)) ms")
    println("  Throughput: $(round(training_throughput, digits=1)) samples/sec")
    
    return training_results
end

"""
Benchmark multi-GPU scaling efficiency
"""
function benchmark_scaling_efficiency(
    profiler::PerformanceProfiler,
    coordinator,
    iterations::Int = 50
)
    println("Benchmarking multi-GPU scaling efficiency...")
    
    scaling_durations = Float32[]
    
    for i in 1:iterations
        _, result = profile_operation!(profiler, "distributed_step", () -> begin
            distributed_training_step!(coordinator)
        end, category="communication")
        
        push!(scaling_durations, result.duration_ms)
        
        if i % 10 == 0
            println("  Iteration $i: $(round(result.duration_ms, digits=3)) ms")
        end
    end
    
    # Get scaling statistics
    stats = get_distributed_stats(coordinator)
    
    scaling_results = Dict(
        "mean_step_duration_ms" => mean(scaling_durations),
        "std_step_duration_ms" => std(scaling_durations),
        "scaling_efficiency" => stats.scaling_efficiency,
        "total_throughput" => stats.total_throughput,
        "avg_sync_time_ms" => stats.avg_sync_time,
        "healthy_gpus" => stats.healthy_gpus
    )
    
    println("Scaling Results:")
    println("  Efficiency: $(round(stats.scaling_efficiency * 100, digits=1))%")
    println("  Total throughput: $(round(stats.total_throughput, digits=1)) samples/sec")
    
    return scaling_results
end

"""
Memory bandwidth benchmark
"""
function benchmark_memory_bandwidth(profiler::PerformanceProfiler)
    println("Benchmarking memory bandwidth...")
    
    # Test different data sizes
    data_sizes_mb = [1, 10, 100, 500, 1000]  # MB
    bandwidth_results = Dict{Int, Dict{String, Float32}}()
    
    for size_mb in data_sizes_mb
        println("  Testing data size: $(size_mb) MB")
        
        size_bytes = size_mb * 1024 * 1024
        n_elements = div(size_bytes, sizeof(Float32))
        
        # Host to device transfer
        host_data = rand(Float32, n_elements)
        _, h2d_result = profile_operation!(profiler, "h2d_transfer_$(size_mb)mb", () -> begin
            CuArray(host_data)
        end, category="memory")
        
        # Device to host transfer
        device_data = CuArray(host_data)
        _, d2h_result = profile_operation!(profiler, "d2h_transfer_$(size_mb)mb", () -> begin
            Array(device_data)
        end, category="memory")
        
        # Device to device copy
        _, d2d_result = profile_operation!(profiler, "d2d_copy_$(size_mb)mb", () -> begin
            copy(device_data)
        end, category="memory")
        
        # Calculate bandwidths
        h2d_bandwidth = (size_mb * 1000.0f0) / h2d_result.duration_ms  # MB/s
        d2h_bandwidth = (size_mb * 1000.0f0) / d2h_result.duration_ms  # MB/s
        d2d_bandwidth = (size_mb * 1000.0f0) / d2d_result.duration_ms  # MB/s
        
        bandwidth_results[size_mb] = Dict(
            "h2d_bandwidth_mbps" => h2d_bandwidth,
            "d2h_bandwidth_mbps" => d2h_bandwidth,
            "d2d_bandwidth_mbps" => d2d_bandwidth,
            "h2d_duration_ms" => h2d_result.duration_ms,
            "d2h_duration_ms" => d2h_result.duration_ms,
            "d2d_duration_ms" => d2d_result.duration_ms
        )
        
        println("    H2D: $(round(h2d_bandwidth / 1000, digits=2)) GB/s")
        println("    D2H: $(round(d2h_bandwidth / 1000, digits=2)) GB/s")
        println("    D2D: $(round(d2d_bandwidth / 1000, digits=2)) GB/s")
    end
    
    return bandwidth_results
end

"""
Generate comprehensive performance report
"""
function generate_performance_report(profiler::PerformanceProfiler)
    println("Generating comprehensive performance report...")
    
    # Calculate overall statistics
    total_runtime = time() - profiler.start_time
    success_rate = 1.0f0 - (Float32(profiler.failed_measurements) / Float32(profiler.total_measurements))
    
    # Aggregate results by operation
    operation_stats = Dict{String, Dict{String, Float32}}()
    operation_durations = Dict{String, Vector{Float32}}()
    
    for result in profiler.results
        op_name = result.operation_name
        
        if !haskey(operation_durations, op_name)
            operation_durations[op_name] = Float32[]
            operation_stats[op_name] = Dict{String, Float32}()
        end
        
        push!(operation_durations[op_name], result.duration_ms)
    end
    
    # Calculate statistics for each operation
    for (op_name, durations) in operation_durations
        stats = operation_stats[op_name]
        
        stats["count"] = Float32(length(durations))
        stats["mean_ms"] = mean(durations)
        stats["std_ms"] = length(durations) > 1 ? std(durations) : 0.0f0
        stats["min_ms"] = minimum(durations)
        stats["max_ms"] = maximum(durations)
        stats["p50_ms"] = quantile(durations, 0.5)
        stats["p95_ms"] = quantile(durations, 0.95)
        stats["p99_ms"] = quantile(durations, 0.99)
    end
    
    # Create comprehensive report
    report = Dict(
        "profiling_config" => profiler.config,
        "overall_statistics" => Dict(
            "total_runtime_seconds" => total_runtime,
            "total_measurements" => profiler.total_measurements,
            "failed_measurements" => profiler.failed_measurements,
            "success_rate" => success_rate,
            "measurements_per_second" => Float32(profiler.total_measurements) / Float32(total_runtime)
        ),
        "operation_statistics" => operation_stats,
        "memory_profiling" => nothing,
        "regression_analysis" => profiler.baseline_results,
        "performance_targets" => Dict(
            "inference_latency_target_ms" => profiler.config.inference_latency_target_ms,
            "training_throughput_target" => profiler.config.training_throughput_target,
            "memory_bandwidth_target_gbps" => profiler.config.memory_bandwidth_target_gbps,
            "scaling_efficiency_target" => profiler.config.scaling_efficiency_target
        ),
        "timestamp" => Dates.now(),
        "gpu_info" => Dict(
            "device_count" => CUDA.ndevices(),
            "device_name" => CUDA.name(CUDA.device()),
            "total_memory_gb" => CUDA.totalmem(CUDA.device()) / (1024^3),
            "available_memory_gb" => CUDA.available_memory() / (1024^3)
        )
    )
    
    # Add memory profiling results if available
    if !isnothing(profiler.memory_profiler)
        mp = profiler.memory_profiler
        report["memory_profiling"] = Dict(
            "peak_gpu_memory_mb" => mp.peak_gpu_memory / (1024^2),
            "peak_cpu_memory_mb" => mp.peak_cpu_memory / (1024^2),
            "gpu_allocation_count" => length(mp.gpu_allocations),
            "cpu_allocation_count" => length(mp.cpu_allocations),
            "avg_gpu_allocation_mb" => length(mp.gpu_allocations) > 0 ? mean(mp.gpu_allocations) / (1024^2) : 0.0,
            "avg_cpu_allocation_mb" => length(mp.cpu_allocations) > 0 ? mean(mp.cpu_allocations) / (1024^2) : 0.0
        )
    end
    
    return report
end

"""
Export performance report
"""
function export_performance_report(profiler::PerformanceProfiler, filename::Union{Nothing, String} = nothing)
    report = generate_performance_report(profiler)
    
    if isnothing(filename)
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        filename = "performance_report_$timestamp"
    end
    
    export_path = joinpath(profiler.config.export_path, filename)
    
    if profiler.config.export_format == "json"
        full_path = export_path * ".json"
        open(full_path, "w") do f
            JSON3.pretty(f, report)
        end
    elseif profiler.config.export_format == "hdf5"
        # Note: HDF5 export would require more complex serialization
        full_path = export_path * ".h5"
        @warn "HDF5 export not fully implemented, using JSON fallback"
        full_path = export_path * ".json"
        open(full_path, "w") do f
            JSON3.pretty(f, report)
        end
    else
        @warn "Unknown export format: $(profiler.config.export_format), using JSON"
        full_path = export_path * ".json"
        open(full_path, "w") do f
            JSON3.pretty(f, report)
        end
    end
    
    println("Performance report exported to: $full_path")
    return full_path
end

"""
Real-time performance dashboard data
"""
function get_realtime_dashboard_data(profiler::PerformanceProfiler)
    current_time = time()
    runtime = current_time - profiler.start_time
    
    # Calculate recent performance (last 60 seconds)
    recent_results = filter(r -> (current_time - r.start_time) <= 60.0, profiler.results)
    
    dashboard_data = Dict(
        "runtime_seconds" => runtime,
        "total_measurements" => profiler.total_measurements,
        "measurements_per_second" => Float32(profiler.total_measurements) / Float32(runtime),
        "recent_measurements" => length(recent_results),
        "success_rate" => 1.0f0 - (Float32(profiler.failed_measurements) / max(1.0f0, Float32(profiler.total_measurements))),
        "current_measurements" => profiler.current_measurements,
        "gpu_memory_available_gb" => CUDA.available_memory() / (1024^3),
        "gpu_memory_total_gb" => CUDA.totalmem(CUDA.device()) / (1024^3)
    )
    
    return dashboard_data
end

"""
Cleanup profiler resources
"""
function cleanup_profiler!(profiler::PerformanceProfiler)
    # Clean up CUDA events
    for (_, timer) in profiler.cuda_timers
        try
            # Note: CUDA.jl should handle cleanup automatically
        catch e
            @warn "Error cleaning up CUDA timer" exception=e
        end
    end
    
    println("Performance profiler cleanup completed")
end

# Export types and functions
export ProfilingConfig, create_profiling_config
export PerformanceProfiler, initialize_profiler
export profile_operation!, benchmark_inference, benchmark_training
export benchmark_scaling_efficiency, benchmark_memory_bandwidth
export generate_performance_report, export_performance_report
export get_realtime_dashboard_data, cleanup_profiler!
export CUDAEventTimer, start_timer!, stop_timer!
export NsightMarker, push_nsight_marker, pop_nsight_marker
export MemoryProfiler, take_memory_snapshot!

end # module PerformanceProfiling