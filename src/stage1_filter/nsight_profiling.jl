module NsightProfiling

using CUDA
using Printf
using DataFrames
using CSV
using Dates

export ProfileConfig, profile_kernel_memory, generate_nsight_report
export analyze_memory_patterns, optimize_kernel_config

"""
Configuration for Nsight-compatible profiling
"""
struct ProfileConfig
    profile_name::String
    output_dir::String
    metrics::Vector{String}
    events::Vector{String}
    iterations::Int
end

function ProfileConfig(;
    profile_name::String = "memory_profile",
    output_dir::String = "profiling_results",
    metrics::Vector{String} = [
        "dram_read_throughput",
        "dram_write_throughput", 
        "gld_efficiency",
        "gst_efficiency",
        "shared_efficiency",
        "achieved_occupancy"
    ],
    events::Vector{String} = [
        "global_load",
        "global_store",
        "shared_load",
        "shared_store"
    ],
    iterations::Int = 10
)
    return ProfileConfig(profile_name, output_dir, metrics, events, iterations)
end

"""
CUDA kernel wrapper for profiling with nvprof markers
"""
macro profile_kernel(name, kernel_call)
    quote
        # Start NVTX range for profiling
        CUDA.@profile $name begin
            $kernel_call
        end
    end
end

"""
Profile memory access patterns of a kernel
"""
function profile_kernel_memory(kernel_func, args...; config::ProfileConfig)
    mkpath(config.output_dir)
    
    results = Dict{String, Any}()
    results["kernel_name"] = config.profile_name
    results["timestamp"] = Dates.now()
    
    # Warm up
    kernel_func(args...)
    CUDA.synchronize()
    
    # Time measurements
    times = Float64[]
    
    for i in 1:config.iterations
        CUDA.@sync begin
            t = CUDA.@elapsed kernel_func(args...)
            push!(times, t)
        end
    end
    
    results["avg_time_ms"] = mean(times) * 1000
    results["min_time_ms"] = minimum(times) * 1000
    results["max_time_ms"] = maximum(times) * 1000
    
    # Memory info
    mem_info = CUDA.memory_status()
    results["memory_used_mb"] = mem_info.used / 1024^2
    results["memory_free_mb"] = mem_info.free / 1024^2
    
    # Calculate theoretical metrics
    results["metrics"] = calculate_theoretical_metrics(kernel_func, args...)
    
    return results
end

"""
Calculate theoretical memory metrics
"""
function calculate_theoretical_metrics(kernel_func, args...)
    metrics = Dict{String, Float64}()
    
    # Extract array dimensions from arguments
    total_bytes_read = 0
    total_bytes_written = 0
    
    for arg in args
        if isa(arg, CuArray)
            bytes = length(arg) * sizeof(eltype(arg))
            # Assume read for inputs, write for outputs (simplified)
            if isa(arg, CuArray{<:Any, 1})  # Vectors likely outputs
                total_bytes_written += bytes
            else  # Matrices likely inputs
                total_bytes_read += bytes
            end
        end
    end
    
    # Execute once to get timing
    t = CUDA.@elapsed begin
        kernel_func(args...)
        CUDA.synchronize()
    end
    
    # Calculate bandwidth
    metrics["read_bandwidth_gb_s"] = total_bytes_read / t / 1e9
    metrics["write_bandwidth_gb_s"] = total_bytes_written / t / 1e9
    metrics["total_bandwidth_gb_s"] = (total_bytes_read + total_bytes_written) / t / 1e9
    
    # RTX 4090 theoretical peak: ~1008 GB/s
    theoretical_peak = 1008.0  # GB/s
    metrics["bandwidth_utilization"] = metrics["total_bandwidth_gb_s"] / theoretical_peak
    
    return metrics
end

"""
Analyze memory access patterns from profiling data
"""
function analyze_memory_patterns(X::CuArray{Float32, 2}, kernel_type::Symbol)
    n_features, n_samples = size(X)
    
    analysis = Dict{String, Any}()
    analysis["dataset_size"] = (n_features, n_samples)
    analysis["memory_footprint_mb"] = n_features * n_samples * sizeof(Float32) / 1024^2
    
    if kernel_type == :variance
        # Variance kernel access pattern
        analysis["access_pattern"] = "Sequential row-wise"
        analysis["coalescing"] = "Good - consecutive threads access consecutive elements"
        analysis["bank_conflicts"] = "Minimal with padding"
        analysis["cache_usage"] = "L1 cache beneficial for reduction"
        
        # Optimization suggestions
        analysis["optimizations"] = [
            "Use vectorized loads (float4) for 4x bandwidth",
            "Implement warp shuffle for reduction",
            "Pad shared memory to avoid bank conflicts",
            "Process multiple elements per thread"
        ]
        
    elseif kernel_type == :mutual_information
        # MI kernel access pattern  
        analysis["access_pattern"] = "Random within feature"
        analysis["coalescing"] = "Moderate - depends on histogram binning"
        analysis["bank_conflicts"] = "Possible in histogram updates"
        analysis["cache_usage"] = "L2 cache helps with repeated feature access"
        
        analysis["optimizations"] = [
            "Use texture memory for feature data",
            "Implement histogram with stride to avoid conflicts",
            "Batch process samples for better locality",
            "Use shared memory for local histograms"
        ]
        
    elseif kernel_type == :correlation
        # Correlation kernel access pattern
        analysis["access_pattern"] = "Tiled matrix multiplication"
        analysis["coalescing"] = "Excellent with proper tiling"
        analysis["bank_conflicts"] = "None with correct tile size"
        analysis["cache_usage"] = "Shared memory critical for tile reuse"
        
        analysis["optimizations"] = [
            "Use optimal tile size (16x16 or 32x32)",
            "Implement double buffering for tiles",
            "Leverage tensor cores if available",
            "Minimize global memory accesses"
        ]
    end
    
    # Memory bandwidth analysis
    element_size = sizeof(Float32)
    cache_line_size = 128  # bytes
    warp_size = 32
    
    analysis["cache_lines_per_warp"] = cld(warp_size * element_size, cache_line_size)
    analysis["bytes_per_warp_access"] = analysis["cache_lines_per_warp"] * cache_line_size
    analysis["efficiency"] = (warp_size * element_size) / analysis["bytes_per_warp_access"]
    
    return analysis
end

"""
Generate comprehensive Nsight-compatible report
"""
function generate_nsight_report(profile_results::Vector{Dict{String, Any}}, 
                              output_file::String = "nsight_report.md")
    
    open(output_file, "w") do f
        write(f, """
        # GPU Memory Profiling Report
        
        Generated: $(Dates.now())
        GPU: $(CUDA.name(CUDA.device()))
        
        ## Executive Summary
        
        This report analyzes memory access patterns and optimization opportunities
        for Stage 1 Fast Filtering GPU kernels.
        
        ## Profiling Results
        
        """)
        
        for result in profile_results
            write(f, """
            ### $(result["kernel_name"])
            
            **Performance Metrics:**
            - Average execution time: $(round(result["avg_time_ms"], digits=2))ms
            - Min/Max time: $(round(result["min_time_ms"], digits=2))ms / $(round(result["max_time_ms"], digits=2))ms
            - Memory used: $(round(result["memory_used_mb"], digits=1))MB
            
            **Bandwidth Utilization:**
            - Read bandwidth: $(round(result["metrics"]["read_bandwidth_gb_s"], digits=1)) GB/s
            - Write bandwidth: $(round(result["metrics"]["write_bandwidth_gb_s"], digits=1)) GB/s
            - Total bandwidth: $(round(result["metrics"]["total_bandwidth_gb_s"], digits=1)) GB/s
            - Utilization: $(round(result["metrics"]["bandwidth_utilization"] * 100, digits=1))%
            
            """)
        end
        
        write(f, """
        ## Memory Access Pattern Analysis
        
        ### Coalesced Access Guidelines
        
        1. **Warp-aligned access**: Ensure consecutive threads access consecutive memory addresses
        2. **128-byte transactions**: Align data structures to cache line boundaries
        3. **Avoid strided access**: Use structure-of-arrays (SoA) instead of array-of-structures (AoS)
        
        ### Bank Conflict Avoidance
        
        1. **Shared memory padding**: Add padding to avoid 32-way bank conflicts
        2. **Stride access patterns**: Use prime number strides for conflict-free access
        3. **Warp-level primitives**: Use shuffle operations to reduce shared memory pressure
        
        ### Optimization Recommendations
        
        1. **Vectorized Loads**
           ```julia
           # Use float4 for 4x bandwidth
           vec4 = reinterpret(NTuple{4,Float32}, ptr)
           ```
        
        2. **Texture Memory**
           - Use for read-only data with spatial locality
           - Automatic caching improves random access performance
        
        3. **Shared Memory Tiling**
           - Optimal tile sizes: 16×16 or 32×32
           - Double buffering for overlapped computation
        
        4. **Memory Alignment**
           - Align allocations to 128-byte boundaries
           - Pad arrays to power-of-2 sizes when beneficial
        
        ## Nsight Profiler Commands
        
        To profile with Nsight Compute:
        ```bash
        ncu --set full --export profile_results julia your_script.jl
        ```
        
        Key metrics to monitor:
        - `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` - Load efficiency
        - `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct` - Store efficiency  
        - `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` - L1 cache traffic
        - `lts__t_sectors_op_read.sum` + `lts__t_sectors_op_write.sum` - L2 cache traffic
        
        """)
    end
    
    println("Report generated: $output_file")
end

"""
Optimize kernel configuration based on profiling
"""
function optimize_kernel_config(n_features::Int, n_samples::Int, kernel_type::Symbol)
    config = Dict{String, Any}()
    
    # Calculate optimal block size
    if kernel_type == :variance
        # Variance benefits from larger blocks for reduction
        config["block_size"] = 256
        config["items_per_thread"] = max(1, cld(n_samples, 256 * 4))  # Process 4 at a time
        config["use_vectorized"] = true
        config["shared_mem_bytes"] = (256 + 1) * 2 * sizeof(Float32)  # Padded
        
    elseif kernel_type == :mutual_information
        # MI needs balance between parallelism and shared memory
        config["block_size"] = 128
        config["histogram_bins"] = 10
        config["histogram_stride"] = 11  # Avoid bank conflicts
        n_classes = 3  # Typical
        config["shared_mem_bytes"] = 11 * n_classes * sizeof(Int32) + 256 * sizeof(Float32)
        
    elseif kernel_type == :correlation
        # Correlation uses 2D thread blocks
        config["tile_size"] = 32
        config["block_dim"] = (32, 32)
        config["shared_mem_bytes"] = 2 * 32 * 32 * sizeof(Float32)
    end
    
    # Calculate occupancy
    sm_count = 128  # RTX 4090
    max_threads_per_sm = 1536
    max_blocks_per_sm = 16
    
    if haskey(config, "block_size")
        blocks_per_sm = min(max_blocks_per_sm, max_threads_per_sm ÷ config["block_size"])
        config["theoretical_occupancy"] = blocks_per_sm * config["block_size"] / max_threads_per_sm
    end
    
    return config
end

"""
Example profiling workflow
"""
function example_profiling_workflow()
    # Create test data
    X = CUDA.randn(Float32, 1000, 10000)
    y = CuArray(Int32.(rand(1:3, 10000)))
    
    # Profile variance kernel
    println("Profiling Variance Kernel...")
    config = ProfileConfig(profile_name="variance_kernel")
    
    results = profile_kernel_memory(
        (variances, X) -> begin
            @cuda threads=256 blocks=1000 optimized_variance_kernel!(
                variances, X, Int32(1000), Int32(10000)
            )
        end,
        CUDA.zeros(Float32, 1000), X;
        config=config
    )
    
    # Analyze patterns
    analysis = analyze_memory_patterns(X, :variance)
    
    # Generate report
    generate_nsight_report([results])
    
    return results, analysis
end

end # module