module BottleneckAnalyzer

using CUDA
using Statistics
using Printf

# Include profiling modules
include("cuda_timing.jl")
include("memory_profiling.jl")
include("kernel_metrics.jl")

using .CUDATiming
using .MemoryProfiling
using .KernelMetrics

export Bottleneck, BottleneckReport, PipelineStage
export analyze_bottlenecks, identify_critical_path
export suggest_optimizations, create_bottleneck_report

"""
Types of bottlenecks
"""
@enum BottleneckType begin
    MEMORY_BANDWIDTH
    COMPUTE_THROUGHPUT
    KERNEL_LAUNCH_OVERHEAD
    HOST_DEVICE_TRANSFER
    SYNCHRONIZATION
    MEMORY_ALLOCATION
    OCCUPANCY_LIMITED
    CACHE_THRASHING
end

"""
Severity levels for bottlenecks
"""
@enum SeverityLevel begin
    CRITICAL  # >50% performance impact
    HIGH      # 20-50% impact
    MEDIUM    # 10-20% impact
    LOW       # <10% impact
end

"""
Individual bottleneck finding
"""
struct Bottleneck
    type::BottleneckType
    severity::SeverityLevel
    location::String
    impact_percent::Float32
    description::String
    suggested_fix::String
end

"""
Pipeline stage analysis
"""
struct PipelineStage
    name::String
    elapsed_ms::Float32
    percent_of_total::Float32
    is_critical_path::Bool
    bottlenecks::Vector{Bottleneck}
end

"""
Complete bottleneck analysis report
"""
struct BottleneckReport
    total_elapsed_ms::Float32
    stages::Vector{PipelineStage}
    critical_path::Vector{String}
    top_bottlenecks::Vector{Bottleneck}
    overall_efficiency::Float32
    optimization_potential::Float32
end

"""
Analyze bottlenecks from profiling data
"""
function analyze_bottlenecks(
    timing_results::Vector{TimingResult},
    memory_stats::Dict{Symbol, BandwidthMeasurement},
    kernel_profiles::Vector{KernelProfile}
)
    bottlenecks = Bottleneck[]
    
    # Analyze memory bandwidth bottlenecks
    append!(bottlenecks, analyze_memory_bottlenecks(memory_stats, kernel_profiles))
    
    # Analyze compute bottlenecks
    append!(bottlenecks, analyze_compute_bottlenecks(kernel_profiles))
    
    # Analyze launch overhead
    append!(bottlenecks, analyze_launch_overhead(timing_results, kernel_profiles))
    
    # Analyze transfer bottlenecks
    append!(bottlenecks, analyze_transfer_bottlenecks(memory_stats, timing_results))
    
    # Analyze occupancy issues
    append!(bottlenecks, analyze_occupancy_bottlenecks(kernel_profiles))
    
    return bottlenecks
end

"""
Analyze memory bandwidth bottlenecks
"""
function analyze_memory_bottlenecks(
    memory_stats::Dict{Symbol, BandwidthMeasurement},
    kernel_profiles::Vector{KernelProfile}
)
    bottlenecks = Bottleneck[]
    
    # Check overall bandwidth utilization
    for profile in kernel_profiles
        bw_efficiency = profile.efficiency.bandwidth_efficiency
        
        if profile.efficiency.is_memory_bound && bw_efficiency < 0.5
            severity = if bw_efficiency < 0.2
                CRITICAL
            elseif bw_efficiency < 0.35
                HIGH
            else
                MEDIUM
            end
            
            push!(bottlenecks, Bottleneck(
                MEMORY_BANDWIDTH,
                severity,
                profile.name,
                (1.0 - bw_efficiency) * 100,
                "Kernel achieving only $(round(bw_efficiency * 100, digits=1))% of peak bandwidth",
                "Improve memory access patterns, use shared memory, increase arithmetic intensity"
            ))
        end
    end
    
    # Check for uncoalesced access patterns
    for profile in kernel_profiles
        if profile.efficiency.memory_bandwidth_gbps < 200.0  # Very low for RTX 4090
            push!(bottlenecks, Bottleneck(
                MEMORY_BANDWIDTH,
                HIGH,
                profile.name,
                80.0,  # Estimated impact
                "Suspected uncoalesced memory access ($(round(profile.efficiency.memory_bandwidth_gbps, digits=1)) GB/s)",
                "Ensure coalesced memory access, use structure of arrays (SoA) layout"
            ))
        end
    end
    
    return bottlenecks
end

"""
Analyze compute bottlenecks
"""
function analyze_compute_bottlenecks(kernel_profiles::Vector{KernelProfile})
    bottlenecks = Bottleneck[]
    
    for profile in kernel_profiles
        if !profile.efficiency.is_memory_bound
            compute_eff = profile.efficiency.compute_efficiency
            
            if compute_eff < 0.5
                severity = if compute_eff < 0.1
                    CRITICAL
                elseif compute_eff < 0.25
                    HIGH
                else
                    MEDIUM
                end
                
                push!(bottlenecks, Bottleneck(
                    COMPUTE_THROUGHPUT,
                    severity,
                    profile.name,
                    (1.0 - compute_eff) * 100,
                    "Low compute utilization ($(round(compute_eff * 100, digits=1))% of peak)",
                    "Increase instruction-level parallelism, use FMA operations, consider tensor cores"
                ))
            end
        end
    end
    
    return bottlenecks
end

"""
Analyze kernel launch overhead
"""
function analyze_launch_overhead(
    timing_results::Vector{TimingResult},
    kernel_profiles::Vector{KernelProfile}
)
    bottlenecks = Bottleneck[]
    
    # Find very fast kernels that might have launch overhead issues
    for result in timing_results
        if result.mean_ms < 0.1  # Kernel executes in <0.1ms
            # Launch overhead is typically 5-10 microseconds
            launch_overhead_ms = 0.01  # 10 microseconds
            overhead_percent = (launch_overhead_ms / result.mean_ms) * 100
            
            if overhead_percent > 20
                push!(bottlenecks, Bottleneck(
                    KERNEL_LAUNCH_OVERHEAD,
                    overhead_percent > 50 ? HIGH : MEDIUM,
                    result.name,
                    overhead_percent,
                    "Kernel launch overhead is $(round(overhead_percent, digits=1))% of execution time",
                    "Batch operations, use persistent kernels, or fuse kernels"
                ))
            end
        end
    end
    
    return bottlenecks
end

"""
Analyze host-device transfer bottlenecks
"""
function analyze_transfer_bottlenecks(
    memory_stats::Dict{Symbol, BandwidthMeasurement},
    timing_results::Vector{TimingResult}
)
    bottlenecks = Bottleneck[]
    
    # Check H2D and D2H transfer efficiency
    for (direction, expected_bw) in [(:h2d, 25.0), (:d2h, 25.0)]  # PCIe 4.0 theoretical
        if haskey(memory_stats, direction)
            stat = memory_stats[direction]
            efficiency = stat.bandwidth_gbps / expected_bw
            
            if efficiency < 0.7
                push!(bottlenecks, Bottleneck(
                    HOST_DEVICE_TRANSFER,
                    efficiency < 0.4 ? HIGH : MEDIUM,
                    string(direction),
                    (1.0 - efficiency) * 100,
                    "PCIe transfer at $(round(stat.bandwidth_gbps, digits=1)) GB/s ($(round(efficiency * 100, digits=1))% efficiency)",
                    "Use pinned memory, batch transfers, overlap with computation"
                ))
            end
        end
    end
    
    # Check if transfers dominate execution time
    total_kernel_time = sum(r.elapsed_ms for r in timing_results)
    total_transfer_time = sum(
        haskey(memory_stats, d) ? memory_stats[d].total_time_ms : 0.0
        for d in [:h2d, :d2h]
    )
    
    if total_transfer_time > 0 && total_transfer_time > total_kernel_time * 0.3
        transfer_percent = (total_transfer_time / (total_kernel_time + total_transfer_time)) * 100
        push!(bottlenecks, Bottleneck(
            HOST_DEVICE_TRANSFER,
            transfer_percent > 50 ? CRITICAL : HIGH,
            "Overall",
            transfer_percent,
            "Data transfers consume $(round(transfer_percent, digits=1))% of total time",
            "Minimize transfers, keep data on GPU, use unified memory wisely"
        ))
    end
    
    return bottlenecks
end

"""
Analyze occupancy-related bottlenecks
"""
function analyze_occupancy_bottlenecks(kernel_profiles::Vector{KernelProfile})
    bottlenecks = Bottleneck[]
    
    for profile in kernel_profiles
        occupancy = profile.occupancy.occupancy
        
        if occupancy < 0.5
            severity = if occupancy < 0.25
                CRITICAL
            elseif occupancy < 0.4
                HIGH
            else
                MEDIUM
            end
            
            impact = (1.0 - occupancy) * 50  # Rough estimate
            
            fix_suggestion = if profile.occupancy.limiting_factor == :registers
                "Reduce register usage per thread, use shared memory"
            elseif profile.occupancy.limiting_factor == :shared_memory
                "Reduce shared memory per block or use smaller block size"
            elseif profile.occupancy.limiting_factor == :blocks
                "Increase block size to improve occupancy"
            else
                "Adjust thread block dimensions"
            end
            
            push!(bottlenecks, Bottleneck(
                OCCUPANCY_LIMITED,
                severity,
                profile.name,
                impact,
                "Low occupancy ($(round(occupancy * 100, digits=1))%) limited by $(profile.occupancy.limiting_factor)",
                fix_suggestion
            ))
        end
    end
    
    return bottlenecks
end

"""
Identify critical path through pipeline stages
"""
function identify_critical_path(timing_results::Vector{TimingResult})
    # Sort by total elapsed time
    sorted_results = sort(timing_results, by=r->r.elapsed_ms, rev=true)
    
    # Critical path includes stages that cumulatively account for 80% of time
    total_time = sum(r.elapsed_ms for r in timing_results)
    cumulative_time = 0.0
    critical_path = String[]
    
    for result in sorted_results
        push!(critical_path, result.name)
        cumulative_time += result.elapsed_ms
        
        if cumulative_time >= total_time * 0.8
            break
        end
    end
    
    return critical_path
end

"""
Suggest optimizations based on bottlenecks
"""
function suggest_optimizations(bottlenecks::Vector{Bottleneck})
    # Group by type
    by_type = Dict{BottleneckType, Vector{Bottleneck}}()
    for bottleneck in bottlenecks
        if !haskey(by_type, bottleneck.type)
            by_type[bottleneck.type] = Bottleneck[]
        end
        push!(by_type[bottleneck.type], bottleneck)
    end
    
    suggestions = String[]
    
    # Memory bandwidth optimizations
    if haskey(by_type, MEMORY_BANDWIDTH)
        push!(suggestions, "Memory Bandwidth Optimizations:")
        push!(suggestions, "  - Use shared memory to reduce global memory traffic")
        push!(suggestions, "  - Ensure coalesced memory access patterns")
        push!(suggestions, "  - Consider using texture memory for spatial locality")
        push!(suggestions, "  - Increase arithmetic intensity with kernel fusion")
    end
    
    # Compute optimizations
    if haskey(by_type, COMPUTE_THROUGHPUT)
        push!(suggestions, "\nCompute Throughput Optimizations:")
        push!(suggestions, "  - Use FMA (fused multiply-add) operations")
        push!(suggestions, "  - Increase instruction-level parallelism")
        push!(suggestions, "  - Consider using tensor cores for applicable operations")
        push!(suggestions, "  - Unroll loops to hide latency")
    end
    
    # Launch overhead optimizations
    if haskey(by_type, KERNEL_LAUNCH_OVERHEAD)
        push!(suggestions, "\nKernel Launch Optimizations:")
        push!(suggestions, "  - Batch multiple operations into single kernels")
        push!(suggestions, "  - Use persistent kernels for iterative algorithms")
        push!(suggestions, "  - Consider CUDA graphs for complex kernel sequences")
    end
    
    # Transfer optimizations
    if haskey(by_type, HOST_DEVICE_TRANSFER)
        push!(suggestions, "\nData Transfer Optimizations:")
        push!(suggestions, "  - Use pinned (page-locked) memory")
        push!(suggestions, "  - Overlap transfers with computation using streams")
        push!(suggestions, "  - Keep data on GPU between operations")
        push!(suggestions, "  - Use unified memory for infrequent accesses")
    end
    
    # Occupancy optimizations
    if haskey(by_type, OCCUPANCY_LIMITED)
        push!(suggestions, "\nOccupancy Optimizations:")
        push!(suggestions, "  - Adjust block size for better occupancy")
        push!(suggestions, "  - Reduce register usage with -maxrregcount")
        push!(suggestions, "  - Use dynamic shared memory allocation")
        push!(suggestions, "  - Consider occupancy calculator for tuning")
    end
    
    return join(suggestions, "\n")
end

"""
Create comprehensive bottleneck report
"""
function create_bottleneck_report(
    timing_results::Vector{TimingResult},
    memory_stats::Dict{Symbol, BandwidthMeasurement},
    kernel_profiles::Vector{KernelProfile}
)
    # Analyze bottlenecks
    bottlenecks = analyze_bottlenecks(timing_results, memory_stats, kernel_profiles)
    
    # Sort by impact
    sorted_bottlenecks = sort(bottlenecks, by=b->b.impact_percent, rev=true)
    top_bottlenecks = length(sorted_bottlenecks) > 5 ? sorted_bottlenecks[1:5] : sorted_bottlenecks
    
    # Create pipeline stages
    total_time = sum(r.elapsed_ms for r in timing_results)
    critical_path = identify_critical_path(timing_results)
    
    stages = PipelineStage[]
    for result in timing_results
        stage_bottlenecks = filter(b -> b.location == result.name, bottlenecks)
        
        push!(stages, PipelineStage(
            result.name,
            result.elapsed_ms,
            (result.elapsed_ms / total_time) * 100,
            result.name in critical_path,
            stage_bottlenecks
        ))
    end
    
    # Calculate overall efficiency
    avg_memory_eff = mean([p.efficiency.bandwidth_efficiency for p in kernel_profiles if p.efficiency.is_memory_bound])
    avg_compute_eff = mean([p.efficiency.compute_efficiency for p in kernel_profiles if !p.efficiency.is_memory_bound])
    overall_efficiency = (avg_memory_eff + avg_compute_eff) / 2
    
    # Estimate optimization potential
    optimization_potential = mean([b.impact_percent for b in top_bottlenecks])
    
    return BottleneckReport(
        total_time,
        stages,
        critical_path,
        top_bottlenecks,
        overall_efficiency,
        optimization_potential
    )
end

"""
Pretty print bottleneck report
"""
function Base.show(io::IO, report::BottleneckReport)
    println(io, "Performance Bottleneck Analysis")
    println(io, "="^60)
    
    println(io, "\nOverall Performance:")
    println(io, "  Total time: $(round(report.total_elapsed_ms, digits=2)) ms")
    println(io, "  Overall efficiency: $(round(report.overall_efficiency * 100, digits=1))%")
    println(io, "  Optimization potential: $(round(report.optimization_potential, digits=1))%")
    
    println(io, "\nCritical Path:")
    for (i, stage) in enumerate(report.critical_path)
        println(io, "  $i. $stage")
    end
    
    println(io, "\nTop Bottlenecks:")
    for (i, bottleneck) in enumerate(report.top_bottlenecks)
        println(io, "\n$i. $(bottleneck.type) in $(bottleneck.location)")
        println(io, "   Severity: $(bottleneck.severity)")
        println(io, "   Impact: $(round(bottleneck.impact_percent, digits=1))%")
        println(io, "   Issue: $(bottleneck.description)")
        println(io, "   Fix: $(bottleneck.suggested_fix)")
    end
    
    println(io, "\nPipeline Stage Analysis:")
    for stage in report.stages
        marker = stage.is_critical_path ? "!" : " "
        println(io, "$marker $(stage.name): $(round(stage.elapsed_ms, digits=2)) ms ($(round(stage.percent_of_total, digits=1))%)")
        
        if !isempty(stage.bottlenecks)
            for bottleneck in stage.bottlenecks
                println(io, "    - $(bottleneck.type): $(bottleneck.description)")
            end
        end
    end
end

end # module