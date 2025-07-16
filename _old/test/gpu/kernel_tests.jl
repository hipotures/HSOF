# GPU Kernel Testing Framework
# Provides comprehensive testing for CUDA kernels

push!(LOAD_PATH, joinpath(@__DIR__, "../.."))
using HSOF
using CUDA
using Test
using BenchmarkTools
using Statistics
using Printf
using JSON
using Dates

# Include GPU modules
include("../../src/gpu/device_manager.jl")

module KernelTests

using CUDA
using Test
using BenchmarkTools
using Statistics
using Printf
using JSON
using Dates
using ..DeviceManager
using ..GPUManager
using ..MemoryManager
using ..StreamManager

# Test result structure
struct KernelTestResult
    name::String
    passed::Bool
    execution_time_ms::Float64
    memory_bandwidth_gb_s::Float64
    occupancy::Float64
    error_message::String
end

# Performance baseline structure
mutable struct PerformanceBaseline
    kernel_name::String
    baseline_time_ms::Float64
    tolerance_percent::Float64
end

# Global performance baselines
const PERFORMANCE_BASELINES = Dict{String, PerformanceBaseline}()

# Define kernels at module level
function vadd_kernel(a, b, c)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(c)
        c[i] = a[i] + b[i]
    end
    return
end

function reduce_sum_kernel(input, output, n)
    shared = @cuDynamicSharedMem(Float32, blockDim().x)
    tid = threadIdx().x
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    # Load and sum
    shared[tid] = i <= n ? input[i] : 0.0f0
    sync_threads()
    
    # Reduction in shared memory
    stride = blockDim().x ÷ 2
    while stride > 0
        if tid <= stride && tid + stride <= blockDim().x
            shared[tid] += shared[tid + stride]
        end
        sync_threads()
        stride ÷= 2
    end
    
    # Write result
    if tid == 1
        output[blockIdx().x] = shared[1]
    end
    return
end

function transpose_kernel(output, input, m, n)
    tile = @cuDynamicSharedMem(Float32, (32, 33))  # 33 to avoid bank conflicts
    
    bx = blockIdx().x
    by = blockIdx().y
    tx = threadIdx().x
    ty = threadIdx().y
    
    # Global indices
    row = (by - 1) * 32 + ty
    col = (bx - 1) * 32 + tx
    
    # Load tile to shared memory
    if row <= m && col <= n
        tile[ty, tx] = input[row, col]
    end
    sync_threads()
    
    # Write transposed tile
    row = (bx - 1) * 32 + ty
    col = (by - 1) * 32 + tx
    
    if row <= n && col <= m
        output[row, col] = tile[tx, ty]
    end
    return
end

function coalesced_kernel(output, input)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(output)
        output[i] = input[i] * 2.0f0
    end
    return
end

function strided_kernel(output, input, stride)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    i = 1 + (tid - 1) * stride
    if i <= length(output)
        output[i] = input[i] * 2.0f0
    end
    return
end

"""
    test_vector_addition()

Test basic vector addition kernel.
"""
function test_vector_addition(n::Int = 1_000_000)
    @info "Testing vector addition kernel (n=$n)"
    
    # Prepare data
    a_cpu = rand(Float32, n)
    b_cpu = rand(Float32, n)
    c_cpu = a_cpu .+ b_cpu  # Reference result
    
    # GPU arrays
    a_gpu = CuArray(a_cpu)
    b_gpu = CuArray(b_cpu)
    c_gpu = CUDA.zeros(Float32, n)
    
    # Launch configuration
    threads = 256
    blocks = cld(n, threads)
    
    # Warmup
    CUDA.CUDA.@cuda threads=threads blocks=blocks vadd_kernel(a_gpu, b_gpu, c_gpu)
    CUDA.synchronize()
    
    # Benchmark
    stats = @benchmark begin
        CUDA.CUDA.@cuda threads=$threads blocks=$blocks vadd_kernel($a_gpu, $b_gpu, $c_gpu)
        CUDA.synchronize()
    end samples=100
    
    # Verify correctness
    c_result = Array(c_gpu)
    max_error = maximum(abs.(c_result .- c_cpu))
    passed = max_error < 1e-5
    
    # Calculate metrics
    exec_time_ms = median(stats).time / 1e6  # Convert to ms
    bytes_transferred = 3 * n * sizeof(Float32)  # 3 arrays
    bandwidth_gb_s = (bytes_transferred / 1e9) / (exec_time_ms / 1e3)
    
    # Estimate occupancy (simplified)
    sm_count = attribute(device(), CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    active_blocks_per_sm = min(blocks ÷ sm_count, 32)  # Max 32 blocks per SM
    occupancy = active_blocks_per_sm / 32.0
    
    result = KernelTestResult(
        "vector_addition",
        passed,
        exec_time_ms,
        bandwidth_gb_s,
        occupancy,
        passed ? "" : "Maximum error: $max_error"
    )
    
    # Free GPU memory
    CUDA.unsafe_free!(a_gpu)
    CUDA.unsafe_free!(b_gpu)
    CUDA.unsafe_free!(c_gpu)
    
    return result
end

"""
    test_reduction_sum()

Test parallel reduction kernel.
"""
function test_reduction_sum(n::Int = 1_000_000)
    @info "Testing reduction sum kernel (n=$n)"
    
    # Prepare data
    data_cpu = rand(Float32, n)
    expected_sum = sum(data_cpu)
    
    # GPU arrays
    data_gpu = CuArray(data_cpu)
    threads = 256
    blocks = cld(n, threads)
    partial_sums = CUDA.zeros(Float32, blocks)
    
    # Warmup
    CUDA.@cuda threads=threads blocks=blocks shmem=threads*sizeof(Float32) reduce_sum_kernel(data_gpu, partial_sums, n)
    CUDA.synchronize()
    
    # Benchmark
    stats = @benchmark begin
        CUDA.@cuda threads=$threads blocks=$blocks shmem=$threads*sizeof(Float32) reduce_sum_kernel($data_gpu, $partial_sums, $n)
        CUDA.synchronize()
    end samples=100
    
    # Complete reduction on CPU
    gpu_sum = sum(Array(partial_sums))
    
    # Verify correctness
    relative_error = abs(gpu_sum - expected_sum) / abs(expected_sum)
    passed = relative_error < 1e-4
    
    # Calculate metrics
    exec_time_ms = median(stats).time / 1e6
    bytes_transferred = n * sizeof(Float32) + blocks * sizeof(Float32)
    bandwidth_gb_s = (bytes_transferred / 1e9) / (exec_time_ms / 1e3)
    
    # Occupancy estimation
    sm_count = attribute(device(), CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    occupancy = min(blocks / sm_count, 1.0)
    
    result = KernelTestResult(
        "reduction_sum",
        passed,
        exec_time_ms,
        bandwidth_gb_s,
        occupancy,
        passed ? "" : "Relative error: $relative_error"
    )
    
    # Cleanup
    CUDA.unsafe_free!(data_gpu)
    CUDA.unsafe_free!(partial_sums)
    
    return result
end

"""
    test_matrix_transpose()

Test matrix transpose kernel with coalesced memory access.
"""
function test_matrix_transpose(m::Int = 1024, n::Int = 1024)
    @info "Testing matrix transpose kernel ($m x $n)"
    
    # Prepare data
    A_cpu = rand(Float32, m, n)
    A_T_cpu = transpose(A_cpu)
    
    # GPU arrays
    A_gpu = CuArray(A_cpu)
    A_T_gpu = CUDA.zeros(Float32, n, m)
    
    # Launch configuration
    threads = (32, 32)
    blocks = (cld(n, 32), cld(m, 32))
    
    # Warmup
    CUDA.@cuda threads=threads blocks=blocks shmem=32*33*sizeof(Float32) transpose_kernel(A_T_gpu, A_gpu, m, n)
    CUDA.synchronize()
    
    # Benchmark
    stats = @benchmark begin
        CUDA.@cuda threads=$threads blocks=$blocks shmem=32*33*sizeof(Float32) transpose_kernel($A_T_gpu, $A_gpu, $m, $n)
        CUDA.synchronize()
    end samples=100
    
    # Verify correctness
    A_T_result = Array(A_T_gpu)
    max_error = maximum(abs.(A_T_result .- A_T_cpu))
    passed = max_error < 1e-5
    
    # Calculate metrics
    exec_time_ms = median(stats).time / 1e6
    bytes_transferred = 2 * m * n * sizeof(Float32)
    bandwidth_gb_s = (bytes_transferred / 1e9) / (exec_time_ms / 1e3)
    
    # Occupancy
    total_blocks = blocks[1] * blocks[2]
    sm_count = attribute(device(), CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    occupancy = min(total_blocks / sm_count / 8, 1.0)  # Assuming max 8 blocks per SM
    
    result = KernelTestResult(
        "matrix_transpose",
        passed,
        exec_time_ms,
        bandwidth_gb_s,
        occupancy,
        passed ? "" : "Maximum error: $max_error"
    )
    
    # Cleanup
    CUDA.unsafe_free!(A_gpu)
    CUDA.unsafe_free!(A_T_gpu)
    
    return result
end

"""
    test_memory_patterns()

Test different memory access patterns.
"""
function test_memory_patterns(n::Int = 1_000_000)
    @info "Testing memory access patterns (n=$n)"
    
    results = KernelTestResult[]
    
    # Test data
    input_cpu = rand(Float32, n)
    output_cpu = similar(input_cpu)
    
    input_gpu = CuArray(input_cpu)
    output_gpu = CUDA.zeros(Float32, n)
    
    threads = 256
    blocks = cld(n, threads)
    
    # Test coalesced access
    stats_coalesced = @benchmark begin
        CUDA.@cuda threads=$threads blocks=$blocks coalesced_kernel($output_gpu, $input_gpu)
        CUDA.synchronize()
    end samples=50
    
    exec_time_coalesced = median(stats_coalesced).time / 1e6
    bandwidth_coalesced = (2 * n * sizeof(Float32) / 1e9) / (exec_time_coalesced / 1e3)
    
    push!(results, KernelTestResult(
        "memory_coalesced",
        true,
        exec_time_coalesced,
        bandwidth_coalesced,
        1.0,
        ""
    ))
    
    # Test strided access (stride = 32)
    stride = 32
    blocks_strided = cld(n ÷ stride, threads)
    
    stats_strided = @benchmark begin
        CUDA.@cuda threads=$threads blocks=$blocks_strided strided_kernel($output_gpu, $input_gpu, $stride)
        CUDA.synchronize()
    end samples=50
    
    exec_time_strided = median(stats_strided).time / 1e6
    bandwidth_strided = (2 * (n ÷ stride) * sizeof(Float32) / 1e9) / (exec_time_strided / 1e3)
    
    push!(results, KernelTestResult(
        "memory_strided",
        true,
        exec_time_strided,
        bandwidth_strided,
        1.0,
        "Stride: $stride"
    ))
    
    # Cleanup
    CUDA.unsafe_free!(input_gpu)
    CUDA.unsafe_free!(output_gpu)
    
    return results
end

"""
    benchmark_kernel(kernel_func, args...; kwargs...)

Generic kernel benchmarking function.
"""
function benchmark_kernel(kernel_func, args...; 
                         threads=256, blocks=1, shmem=0, samples=100)
    # Warmup
    CUDA.@cuda threads=threads blocks=blocks shmem=shmem kernel_func(args...)
    CUDA.synchronize()
    
    # Benchmark
    stats = @benchmark begin
        CUDA.@cuda threads=$threads blocks=$blocks shmem=$shmem $kernel_func($args...)
        CUDA.synchronize()
    end samples=samples
    
    return stats
end

"""
    check_memory_leaks(f::Function)

Run function and check for GPU memory leaks.
"""
function check_memory_leaks(f::Function)
    # Force GC and get initial memory
    GC.gc()
    CUDA.reclaim()
    initial_free = CUDA.available_memory()
    
    # Run function
    f()
    
    # Force GC and get final memory
    GC.gc()
    CUDA.reclaim()
    final_free = CUDA.available_memory()
    
    # Check for leaks (allowing small variations)
    leaked_bytes = initial_free - final_free
    leaked_mb = leaked_bytes / 1024^2
    
    return leaked_mb, leaked_mb < 1.0  # Allow up to 1MB variation
end

"""
    run_all_tests()

Run all kernel tests and generate report.
"""
function run_all_tests(; verbose::Bool = true)
    @info "Running comprehensive GPU kernel tests..."
    
    # Initialize GPU
    config = Dict("gpu" => Dict("cuda" => Dict("stream_count" => 4)))
    DeviceManager.initialize_devices(config)
    
    results = KernelTestResult[]
    
    # Run tests
    try
        # Basic kernels
        push!(results, test_vector_addition())
        push!(results, test_reduction_sum())
        push!(results, test_matrix_transpose())
        
        # Memory patterns
        append!(results, test_memory_patterns())
        
        # Memory leak test
        @info "Checking for memory leaks..."
        leaked_mb, no_leak = check_memory_leaks() do
            test_vector_addition(100_000)
            test_reduction_sum(100_000)
        end
        
        if no_leak
            @info "✓ No memory leaks detected"
        else
            @warn "Memory leak detected: $leaked_mb MB"
        end
        
    catch e
        @error "Test failed with error: $e"
        rethrow(e)
    finally
        # Cleanup
        DeviceManager.cleanup()
    end
    
    # Generate report
    if verbose
        generate_report(results)
    end
    
    return results
end

"""
    generate_report(results::Vector{KernelTestResult})

Generate detailed test report.
"""
function generate_report(results::Vector{KernelTestResult})
    println("\n" * "="^80)
    println("GPU Kernel Test Report")
    println("="^80)
    
    # Summary
    total_tests = length(results)
    passed_tests = count(r -> r.passed, results)
    
    println("\nSummary:")
    println("  Total Tests: $total_tests")
    println("  Passed: $passed_tests")
    println("  Failed: $(total_tests - passed_tests)")
    println("  Success Rate: $(round(100 * passed_tests / total_tests, digits=1))%")
    
    # Detailed results
    println("\nDetailed Results:")
    println("-"^80)
    println(@sprintf("%-20s %-8s %-12s %-15s %-10s %s", 
                     "Kernel", "Status", "Time (ms)", "Bandwidth (GB/s)", "Occupancy", "Notes"))
    println("-"^80)
    
    for result in results
        status = result.passed ? "PASS" : "FAIL"
        status_color = result.passed ? :green : :red
        
        printstyled(@sprintf("%-20s", result.name); color=:normal)
        printstyled(@sprintf(" %-8s", status); color=status_color)
        println(@sprintf(" %-12.3f %-15.2f %-10.2f %s",
                        result.execution_time_ms,
                        result.memory_bandwidth_gb_s,
                        result.occupancy,
                        result.error_message))
    end
    
    println("-"^80)
    
    # Performance summary
    println("\nPerformance Summary:")
    
    # Calculate aggregate metrics
    total_bandwidth = sum(r -> r.memory_bandwidth_gb_s, results)
    avg_bandwidth = total_bandwidth / length(results)
    max_bandwidth = maximum(r -> r.memory_bandwidth_gb_s, results)
    
    println("  Average Bandwidth: $(round(avg_bandwidth, digits=2)) GB/s")
    println("  Peak Bandwidth: $(round(max_bandwidth, digits=2)) GB/s")
    
    # Device info
    dev_info = DeviceManager.get_device_info()
    if !isempty(dev_info["devices"])
        dev = dev_info["devices"][1]
        println("\nGPU Information:")
        println("  Device: $(dev["name"])")
        println("  Compute Capability: $(dev["compute_capability"])")
        println("  Memory: $(dev["total_memory_gb"]) GB")
    end
    
    println("="^80)
end

"""
    save_performance_baseline(results::Vector{KernelTestResult}, filename::String)

Save current performance as baseline for regression testing.
"""
function save_performance_baseline(results::Vector{KernelTestResult}, 
                                 filename::String = "kernel_performance_baseline.json")
    baselines = Dict{String, Any}()
    
    for result in results
        if result.passed
            baselines[result.name] = Dict(
                "time_ms" => result.execution_time_ms,
                "bandwidth_gb_s" => result.memory_bandwidth_gb_s,
                "occupancy" => result.occupancy
            )
        end
    end
    
    # Add metadata
    baselines["metadata"] = Dict(
        "timestamp" => string(now()),
        "cuda_version" => string(CUDA.runtime_version()),
        "julia_version" => string(VERSION)
    )
    
    # Save to file
    open(filename, "w") do f
        JSON.print(f, baselines, 4)
    end
    
    @info "Performance baseline saved to $filename"
end

"""
    check_performance_regression(results::Vector{KernelTestResult}, 
                               baseline_file::String;
                               tolerance::Float64 = 0.1)

Check for performance regressions against baseline.
"""
function check_performance_regression(results::Vector{KernelTestResult}, 
                                    baseline_file::String;
                                    tolerance::Float64 = 0.1)
    if !isfile(baseline_file)
        @warn "Baseline file not found: $baseline_file"
        return false
    end
    
    # Load baseline
    baselines = JSON.parsefile(baseline_file)
    regressions = String[]
    
    for result in results
        if result.passed && haskey(baselines, result.name)
            baseline = baselines[result.name]
            baseline_time = baseline["time_ms"]
            
            # Check if current time is significantly worse
            regression_factor = result.execution_time_ms / baseline_time
            if regression_factor > 1.0 + tolerance
                push!(regressions, 
                      "$(result.name): $(round(regression_factor, digits=2))x slower")
            end
        end
    end
    
    if isempty(regressions)
        @info "✓ No performance regressions detected"
        return true
    else
        @warn "Performance regressions detected:"
        for reg in regressions
            println("  - $reg")
        end
        return false
    end
end

end # module

# Run tests if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    using .KernelTests
    results = KernelTests.run_all_tests()
end