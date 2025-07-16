module LatencyProfiler

using Statistics
using Dates
using BenchmarkTools
using CUDA

export LatencyMeasurement, LatencyProfile, LatencyBenchmark
export measure_latency, profile_latencies, benchmark_metamodel_inference
export analyze_latency_distribution, detect_outliers

"""
Single latency measurement
"""
struct LatencyMeasurement
    operation::String
    timestamp::DateTime
    latency_us::Float64  # microseconds
    batch_size::Int
    is_gpu::Bool
    is_outlier::Bool
end

"""
Latency profile for an operation
"""
struct LatencyProfile
    operation::String
    measurements::Vector{LatencyMeasurement}
    percentiles::Dict{Int, Float64}  # 50, 90, 95, 99, 99.9
    mean_us::Float64
    median_us::Float64
    std_us::Float64
    min_us::Float64
    max_us::Float64
    throughput_ops_per_sec::Float64
end

"""
Benchmark configuration
"""
struct LatencyBenchmark
    warmup_iterations::Int
    measurement_iterations::Int
    batch_sizes::Vector{Int}
    timeout_seconds::Float64
end

"""
Measure latency of a single operation
"""
function measure_latency(
    operation::Function,
    operation_name::String;
    batch_size::Int = 1,
    is_gpu::Bool = false,
    iterations::Int = 100
)::Vector{LatencyMeasurement}
    
    measurements = LatencyMeasurement[]
    
    # Warmup
    for _ in 1:min(10, iterations ÷ 10)
        operation()
        if is_gpu
            CUDA.synchronize()
        end
    end
    
    # Measurements
    latencies = Float64[]
    for _ in 1:iterations
        start_time = time_ns()
        
        operation()
        
        if is_gpu
            CUDA.synchronize()
        end
        
        end_time = time_ns()
        latency_ns = end_time - start_time
        latency_us = latency_ns / 1000.0
        push!(latencies, latency_us)
    end
    
    # Detect outliers using IQR method
    q1 = quantile(latencies, 0.25)
    q3 = quantile(latencies, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Create measurements
    for latency in latencies
        is_outlier = latency < lower_bound || latency > upper_bound
        push!(measurements, LatencyMeasurement(
            operation_name,
            now(),
            latency,
            batch_size,
            is_gpu,
            is_outlier
        ))
    end
    
    return measurements
end

"""
Profile latencies across different configurations
"""
function profile_latencies(
    operation::Function,
    operation_name::String;
    batch_sizes::Vector{Int} = [1, 10, 100, 1000],
    is_gpu::Bool = false,
    iterations_per_batch::Int = 100
)::Dict{Int, LatencyProfile}
    
    profiles = Dict{Int, LatencyProfile}()
    
    for batch_size in batch_sizes
        println("Profiling batch size: $batch_size")
        
        # Create batched operation
        batched_op = () -> begin
            for _ in 1:batch_size
                operation()
            end
        end
        
        # Measure latencies
        measurements = measure_latency(
            batched_op,
            "$operation_name (batch=$batch_size)",
            batch_size=batch_size,
            is_gpu=is_gpu,
            iterations=iterations_per_batch
        )
        
        # Analyze results
        profile = analyze_latency_measurements(measurements)
        profiles[batch_size] = profile
    end
    
    return profiles
end

"""
Analyze latency measurements
"""
function analyze_latency_measurements(measurements::Vector{LatencyMeasurement})::LatencyProfile
    # Filter out outliers for statistics
    valid_measurements = filter(m -> !m.is_outlier, measurements)
    latencies = [m.latency_us for m in valid_measurements]
    
    if isempty(latencies)
        latencies = [m.latency_us for m in measurements]  # Use all if no valid ones
    end
    
    # Calculate percentiles
    percentiles = Dict{Int, Float64}()
    for p in [50, 90, 95, 99]
        percentiles[p] = quantile(latencies, p / 100.0)
    end
    
    # Calculate 99.9th percentile if enough samples
    if length(latencies) >= 1000
        percentiles[999] = quantile(latencies, 0.999)
    end
    
    # Calculate throughput
    mean_latency_s = mean(latencies) / 1_000_000  # Convert to seconds
    batch_size = isempty(measurements) ? 1 : measurements[1].batch_size
    throughput = batch_size / mean_latency_s
    
    return LatencyProfile(
        isempty(measurements) ? "unknown" : measurements[1].operation,
        measurements,
        percentiles,
        mean(latencies),
        median(latencies),
        std(latencies),
        minimum(latencies),
        maximum(latencies),
        throughput
    )
end

"""
Benchmark metamodel inference latency
"""
function benchmark_metamodel_inference(
    metamodel_predict::Function;
    input_sizes::Vector{Int} = [100, 500, 1000, 5000],
    batch_sizes::Vector{Int} = [1, 10, 100],
    use_gpu::Bool = false,
    config::LatencyBenchmark = LatencyBenchmark(100, 1000, [1, 10, 100], 60.0)
)
    
    results = Dict{String, Dict{Int, LatencyProfile}}()
    
    for input_size in input_sizes
        println("\nBenchmarking input size: $input_size features")
        
        # Create test input
        if use_gpu
            test_input = CUDA.rand(Float32, input_size)
        else
            test_input = rand(Float32, input_size)
        end
        
        # Profile different batch sizes
        profiles = Dict{Int, LatencyProfile}()
        
        for batch_size in batch_sizes
            # Create batched prediction function
            predict_batch = if batch_size == 1
                () -> metamodel_predict(test_input)
            else
                () -> begin
                    # Simulate batch prediction
                    results = Vector{Any}(undef, batch_size)
                    for i in 1:batch_size
                        results[i] = metamodel_predict(test_input)
                    end
                    results
                end
            end
            
            # Measure latency
            measurements = measure_latency(
                predict_batch,
                "metamodel_inference",
                batch_size=batch_size,
                is_gpu=use_gpu,
                iterations=config.measurement_iterations
            )
            
            profiles[batch_size] = analyze_latency_measurements(measurements)
        end
        
        results["input_$input_size"] = profiles
    end
    
    return results
end

"""
Analyze latency distribution
"""
function analyze_latency_distribution(profile::LatencyProfile)
    latencies = [m.latency_us for m in profile.measurements if !m.is_outlier]
    
    if isempty(latencies)
        return Dict{String, Any}("error" => "No valid measurements")
    end
    
    # Calculate distribution metrics
    cv = std(latencies) / mean(latencies)  # Coefficient of variation
    skewness = sum((latencies .- mean(latencies)).^3) / (length(latencies) * std(latencies)^3)
    
    # Tail latency analysis
    p99_to_median_ratio = profile.percentiles[99] / profile.median_us
    
    # Jitter analysis
    consecutive_diffs = diff(latencies)
    jitter = std(consecutive_diffs)
    
    return Dict{String, Any}(
        "coefficient_of_variation" => cv,
        "skewness" => skewness,
        "p99_to_median_ratio" => p99_to_median_ratio,
        "jitter_us" => jitter,
        "outlier_percentage" => count(m -> m.is_outlier, profile.measurements) / length(profile.measurements) * 100,
        "stability" => cv < 0.1 ? "excellent" : cv < 0.3 ? "good" : cv < 0.5 ? "fair" : "poor"
    )
end

"""
Detect latency outliers and anomalies
"""
function detect_outliers(measurements::Vector{LatencyMeasurement}; threshold_sigma::Float64 = 3.0)
    latencies = [m.latency_us for m in measurements]
    
    # Z-score method
    mean_latency = mean(latencies)
    std_latency = std(latencies)
    
    outliers = LatencyMeasurement[]
    for (i, m) in enumerate(measurements)
        z_score = abs(m.latency_us - mean_latency) / std_latency
        if z_score > threshold_sigma
            push!(outliers, m)
        end
    end
    
    # Temporal clustering detection
    if length(outliers) > 5
        timestamps = [m.timestamp for m in outliers]
        time_diffs = [Dates.value(timestamps[i+1] - timestamps[i]) / 1000.0 for i in 1:length(timestamps)-1]
        
        if !isempty(time_diffs) && median(time_diffs) < 1000  # Less than 1 second apart
            println("⚠️  Detected temporal clustering of outliers - possible systemic issue")
        end
    end
    
    return outliers
end

"""
Generate latency report
"""
function latency_report(profiles::Dict{Int, LatencyProfile}; target_latency_us::Float64 = 1000.0)
    println("\nLatency Profile Report")
    println("=" ^ 100)
    
    # Header
    println("Batch | Mean μs | Median | P90    | P95    | P99    | Max    | Throughput | Meets Target")
    println("-" * 100)
    
    # Results
    for batch_size in sort(collect(keys(profiles)))
        profile = profiles[batch_size]
        meets_target = profile.percentiles[99] < target_latency_us
        
        @printf("%5d | %7.1f | %6.1f | %6.1f | %6.1f | %6.1f | %6.1f | %10.0f | %s\n",
            batch_size,
            profile.mean_us,
            profile.median_us,
            profile.percentiles[90],
            profile.percentiles[95],
            profile.percentiles[99],
            profile.max_us,
            profile.throughput_ops_per_sec,
            meets_target ? "✓ Yes" : "✗ No"
        )
        
        # Distribution analysis
        dist_analysis = analyze_latency_distribution(profile)
        println("      Stability: $(dist_analysis["stability"]), " *
                "CV: $(round(dist_analysis["coefficient_of_variation"], digits=3)), " *
                "Outliers: $(round(dist_analysis["outlier_percentage"], digits=1))%")
    end
    
    println("-" * 100)
end

"""
Compare latency profiles across different implementations
"""
function compare_latencies(
    implementations::Dict{String, Function};
    batch_sizes::Vector{Int} = [1, 10, 100],
    iterations::Int = 1000
)
    all_profiles = Dict{String, Dict{Int, LatencyProfile}}()
    
    for (name, impl) in implementations
        println("\nProfiling implementation: $name")
        profiles = Dict{Int, LatencyProfile}()
        
        for batch_size in batch_sizes
            measurements = measure_latency(
                impl,
                name,
                batch_size=batch_size,
                iterations=iterations
            )
            profiles[batch_size] = analyze_latency_measurements(measurements)
        end
        
        all_profiles[name] = profiles
    end
    
    # Comparison report
    println("\nLatency Comparison")
    println("=" ^ 80)
    
    for batch_size in batch_sizes
        println("\nBatch Size: $batch_size")
        println("Implementation | Mean μs | P99 μs | Throughput ops/s")
        println("-" * 60)
        
        for (name, profiles) in all_profiles
            profile = profiles[batch_size]
            @printf("%-13s | %7.1f | %6.1f | %16.0f\n",
                name,
                profile.mean_us,
                profile.percentiles[99],
                profile.throughput_ops_per_sec
            )
        end
    end
end

end # module