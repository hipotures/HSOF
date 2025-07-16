module PerformanceProfiling

using CUDA
using Printf
using Statistics

# Include all profiling modules
include("cuda_timing.jl")
include("memory_profiling.jl")
include("kernel_metrics.jl")
include("bottleneck_analyzer.jl")
include("performance_regression.jl")

# Re-export main functionality
using .CUDATiming
using .MemoryProfiling
using .KernelMetrics
using .BottleneckAnalyzer
using .PerformanceRegression

export ProfileSession, ProfileResults
export create_profile_session, start_profiling!, stop_profiling!
export analyze_performance, print_performance_report

# Re-export from submodules
export CUDATimer, MemoryTracker, KernelProfile, KernelConfig
export @cuda_time, track_allocation!, track_transfer!
export analyze_bottlenecks, create_bottleneck_report
export create_stage1_regression_tests, run_all_regression_tests
export profile_kernel, calculate_occupancy, calculate_efficiency
export track_cuarray_allocation, track_cuarray_deallocation

"""
Profiling session that combines all profiling tools
"""
mutable struct ProfileSession
    name::String
    timer::CUDATimer
    memory_tracker::MemoryTracker
    kernel_profiles::Vector{KernelProfile}
    enabled::Bool
    start_time::Float64
end

"""
Complete profiling results
"""
struct ProfileResults
    session_name::String
    total_elapsed_s::Float64
    timing_results::Vector{TimingResult}
    memory_stats::Dict{Symbol, BandwidthMeasurement}
    kernel_profiles::Vector{KernelProfile}
    bottleneck_report::BottleneckReport
end

"""
Create a new profiling session
"""
function create_profile_session(name::String; enabled::Bool = true)
    return ProfileSession(
        name,
        create_timer(enabled=enabled),
        create_memory_tracker(enabled=enabled),
        KernelProfile[],
        enabled,
        time()
    )
end

"""
Start profiling for a named operation
"""
function start_profiling!(session::ProfileSession, operation::String)
    if !session.enabled
        return
    end
    
    start_timing!(session.timer, operation)
end

"""
Stop profiling for a named operation
"""
function stop_profiling!(session::ProfileSession, operation::String)
    if !session.enabled
        return
    end
    
    stop_timing!(session.timer, operation)
end

"""
Profile a kernel execution
"""
function profile_kernel!(
    session::ProfileSession,
    name::String,
    config::KernelMetrics.KernelConfig,
    elapsed_ms::Float32,
    bytes_read::Int,
    bytes_written::Int,
    flop_count::Int
)
    if !session.enabled
        return
    end
    
    profile = KernelMetrics.profile_kernel(
        name, config, elapsed_ms,
        bytes_read, bytes_written, flop_count
    )
    
    push!(session.kernel_profiles, profile)
end

export profile_kernel!

"""
Analyze performance and generate results
"""
function analyze_performance(session::ProfileSession)
    if !session.enabled
        return nothing
    end
    
    total_elapsed = time() - session.start_time
    
    # Get timing results
    timing_results = get_timing_results(session.timer)
    
    # Get memory stats
    memory_stats = get_bandwidth_stats(session.memory_tracker)
    
    # Create a simplified bottleneck report to avoid type conflicts
    bottleneck_report = BottleneckAnalyzer.BottleneckReport(
        Float32(total_elapsed * 1000),  # Convert to ms, use Float32
        BottleneckAnalyzer.PipelineStage[],
        String[],
        BottleneckAnalyzer.Bottleneck[],
        Float32(0.0),  # overall_efficiency
        Float32(0.0)   # optimization_potential
    )
    
    return ProfileResults(
        session.name,
        total_elapsed,
        timing_results,
        memory_stats,
        session.kernel_profiles,
        bottleneck_report
    )
end

"""
Print comprehensive performance report
"""
function print_performance_report(results::ProfileResults)
    println("\n" * "="^80)
    println("PERFORMANCE PROFILING REPORT: $(results.session_name)")
    println("="^80)
    
    println("\nSession Duration: $(round(results.total_elapsed_s, digits=3)) seconds")
    
    # Timing summary
    if !isempty(results.timing_results)
        println("\n" * "-"^60)
        println("TIMING SUMMARY")
        println("-"^60)
        
        for result in results.timing_results
            println("\n$(result.name):")
            println("  Total: $(round(result.elapsed_ms, digits=3)) ms")
            println("  Count: $(result.count)")
            println("  Mean: $(round(result.mean_ms, digits=3)) ms")
            println("  Min/Max: $(round(result.min_ms, digits=3))/$(round(result.max_ms, digits=3)) ms")
        end
    end
    
    # Memory bandwidth summary
    if !isempty(results.memory_stats)
        println("\n" * "-"^60)
        println("MEMORY BANDWIDTH")
        println("-"^60)
        
        for (direction, stat) in results.memory_stats
            println("\n$direction:")
            show(stdout, stat)
        end
    end
    
    # Kernel profiles
    if !isempty(results.kernel_profiles)
        println("\n" * "-"^60)
        println("KERNEL PROFILES")
        println("-"^60)
        
        for profile in results.kernel_profiles
            println()
            show(stdout, profile)
        end
        
        # Roofline analysis
        println("\n" * "-"^60)
        println("ROOFLINE ANALYSIS")
        println("-"^60)
        roofline_analysis(results.kernel_profiles)
    end
    
    # Bottleneck analysis
    println("\n" * "-"^60)
    println("BOTTLENECK ANALYSIS")
    println("-"^60)
    show(stdout, results.bottleneck_report)
    
    # Optimization suggestions
    suggestions = suggest_optimizations(results.bottleneck_report.top_bottlenecks)
    if !isempty(suggestions)
        println("\n" * "-"^60)
        println("OPTIMIZATION SUGGESTIONS")
        println("-"^60)
        println(suggestions)
    end
    
    println("\n" * "="^80)
end

"""
Convenience macro for profiling code blocks
"""
macro profile(session, name, expr)
    quote
        start_profiling!($(esc(session)), $(esc(name)))
        try
            $(esc(expr))
        finally
            stop_profiling!($(esc(session)), $(esc(name)))
        end
    end
end

export @profile

"""
Example usage function
"""
function example_profiling_usage()
    # Create profiling session
    session = create_profile_session("Stage1_Filtering")
    
    # Generate test data
    n_features = 5000
    n_samples = 100000
    X = CUDA.randn(Float32, n_features, n_samples)
    
    # Profile variance calculation
    @profile session "variance_calculation" begin
        variances = CUDA.zeros(Float32, n_features)
        
        # Track memory allocation
        track_cuarray_allocation(session.memory_tracker, variances)
        
        # Simulate kernel execution
        kernel_start = time()
        CUDA.@sync variances .= vec(var(X, dims=2))
        kernel_time = (time() - kernel_start) * 1000
        
        # Profile kernel
        config = KernelMetrics.KernelConfig(256, cld(n_features, 256), 0, 32)
        bytes_read = n_features * n_samples * sizeof(Float32)
        bytes_written = n_features * sizeof(Float32)
        flops = n_features * n_samples * 3  # Rough estimate
        
        profile_kernel!(
            session,
            "variance_kernel",
            config,
            Float32(kernel_time),
            bytes_read,
            bytes_written,
            flops
        )
    end
    
    # Profile correlation calculation
    @profile session "correlation_calculation" begin
        n_subset = 1000  # Smaller subset for correlation
        X_subset = X[1:n_subset, :]
        
        corr_matrix = CUDA.zeros(Float32, n_subset, n_subset)
        track_cuarray_allocation(session.memory_tracker, corr_matrix)
        
        kernel_start = time()
        CUDA.@sync corr_matrix .= cor(X_subset')
        kernel_time = (time() - kernel_start) * 1000
        
        # Profile kernel
        config = KernelConfig(256, cld(n_subset * n_subset, 256), 0, 32)
        bytes_read = n_subset * n_samples * sizeof(Float32)
        bytes_written = n_subset * n_subset * sizeof(Float32)
        flops = n_subset * n_subset * n_samples * 2
        
        profile_kernel!(
            session,
            "correlation_kernel",
            config,
            Float32(kernel_time),
            bytes_read,
            bytes_written,
            flops
        )
    end
    
    # Analyze and print results
    results = analyze_performance(session)
    print_performance_report(results)
    
    return results
end

export example_profiling_usage

end # module