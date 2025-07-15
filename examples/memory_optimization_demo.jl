#!/usr/bin/env julia

# Memory Optimization Demo
# Demonstrates GPU memory access pattern optimizations for improved performance

using CUDA
using Printf
using Statistics
using BenchmarkTools

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

# Load GPU modules
include("../src/gpu/GPU.jl")
using .GPU

# Import memory optimization module
using .GPU.MemoryOptimization

"""
Benchmark unoptimized memory access
"""
function benchmark_unoptimized(data::CuArray{Float32,2}, indices::CuArray{Int})
    n = length(indices)
    result = CUDA.zeros(Float32, n)
    
    function kernel(result, data, indices, n)
        idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        
        if idx <= n
            # Random access pattern - poor memory coalescing
            i = indices[idx]
            if i > 0 && i <= size(data, 2)
                # Access multiple elements with strided pattern
                sum = 0.0f0
                for j in 1:size(data, 1)
                    sum += data[j, i]
                end
                result[idx] = sum
            end
        end
        
        return nothing
    end
    
    threads = 256
    blocks = cld(n, threads)
    
    # Warmup
    @cuda threads=threads blocks=blocks kernel(result, data, indices, n)
    CUDA.synchronize()
    
    # Benchmark
    elapsed = @belapsed begin
        @cuda threads=$threads blocks=$blocks $kernel($result, $data, $indices, $n)
        CUDA.synchronize()
    end samples=10
    
    return elapsed, result
end

"""
Benchmark optimized memory access
"""
function benchmark_optimized(data::CuArray{Float32,2}, indices::CuArray{Int})
    n = length(indices)
    result = CUDA.zeros(Float32, n)
    
    # Optimize data layout for column-major access
    data_tiled = optimize_data_layout(data, access_pattern=:tiled, tile_size=32)
    
    # Use coalesced access
    coalesced_result = CUDA.zeros(Float32, n)
    
    # Profile memory access
    function optimized_kernel(result, data, indices, n)
        # Use shared memory for frequently accessed data
        shared_cache = @cuDynamicSharedMem(Float32, (blockDim().x,))
        
        tid = threadIdx().x
        gid = tid + (blockIdx().x - 1) * blockDim().x
        
        if gid <= n
            # Coalesced read of indices
            idx = indices[gid]
            
            if idx > 0 && idx <= size(data, 2)
                # Use shared memory for reduction
                local_sum = 0.0f0
                
                # Process in coalesced chunks
                for j in 1:size(data, 1)
                    local_sum += data[j, idx]
                end
                
                shared_cache[tid] = local_sum
                sync_threads()
                
                result[gid] = shared_cache[tid]
            end
        end
        
        return nothing
    end
    
    threads = 256
    blocks = cld(n, threads)
    shmem = threads * sizeof(Float32)
    
    # Warmup
    @cuda threads=threads blocks=blocks shmem=shmem optimized_kernel(result, data_tiled, indices, n)
    CUDA.synchronize()
    
    # Benchmark
    elapsed = @belapsed begin
        @cuda threads=$threads blocks=$blocks shmem=$shmem $optimized_kernel($result, $data_tiled, $indices, $n)
        CUDA.synchronize()
    end samples=10
    
    return elapsed, result
end

"""
Demo memory optimization techniques
"""
function demo_memory_optimization()
    println("GPU Memory Optimization Demo")
    println("=" ^ 60)
    
    if !CUDA.functional()
        println("⚠ CUDA not functional - demo requires GPU support")
        return
    end
    
    # Display GPU info
    device!(0)
    dev = device()
    println("GPU: $(CUDA.name(dev))")
    println("Memory: $(round(CUDA.totalmem(dev) / 1024^3, digits=2)) GB")
    println("Compute capability: $(CUDA.capability(dev))")
    println()
    
    # Demo 1: Coalesced vs Non-coalesced Access
    println("1. Coalesced vs Non-coalesced Memory Access")
    println("-" * 40)
    
    n = 100_000
    src = CUDA.rand(Float32, n * 2)
    
    # Non-coalesced access (random indices)
    random_indices = CuArray(shuffle!(collect(1:2:2n))[1:n])
    
    # Coalesced access (sequential indices)
    sequential_indices = CuArray(1:2:2n)
    
    # Benchmark non-coalesced
    dst1 = CUDA.zeros(Float32, n)
    t1 = @belapsed begin
        @cuda threads=256 blocks=cld($n, 256) (dst, src, indices, n) -> begin
            idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            if idx <= n
                dst[idx] = src[indices[idx]]
            end
            return nothing
        end($dst1, $src, $random_indices, $n)
        CUDA.synchronize()
    end samples=10
    
    # Benchmark coalesced
    dst2 = CUDA.zeros(Float32, n)
    t2 = @belapsed begin
        coalesced_read!($dst2, $src, $sequential_indices)
        CUDA.synchronize()
    end samples=10
    
    println("Random access time: $(round(t1 * 1000, digits=3)) ms")
    println("Coalesced access time: $(round(t2 * 1000, digits=3)) ms")
    println("Speedup: $(round(t1 / t2, digits=2))x")
    
    # Demo 2: Memory Pool Performance
    println("\n2. Memory Pool Allocation Performance")
    println("-" * 40)
    
    pool = create_memory_pool(Float32, block_size=10_000, num_blocks=10)
    
    # Benchmark regular allocation
    alloc_times = Float64[]
    for _ in 1:100
        t = @elapsed begin
            arr = CUDA.zeros(Float32, 5000)
            CUDA.synchronize()
        end
        push!(alloc_times, t)
    end
    regular_time = mean(alloc_times) * 1000
    
    # Benchmark pool allocation
    pool_times = Float64[]
    arrays = []
    for _ in 1:100
        t = @elapsed begin
            arr = allocate!(pool, 5000)
            push!(arrays, arr)
        end
        push!(pool_times, t)
    end
    pool_time = mean(pool_times) * 1000
    
    # Deallocate back to pool
    for arr in arrays[1:50]
        deallocate!(pool, arr)
    end
    
    println("Regular allocation: $(round(regular_time, digits=3)) ms/alloc")
    println("Pool allocation: $(round(pool_time, digits=3)) ms/alloc")
    println("Speedup: $(round(regular_time / pool_time, digits=2))x")
    println("Pool reuses: $(pool.reuses)")
    
    # Demo 3: Data Layout Optimization
    println("\n3. Data Layout Optimization")
    println("-" * 40)
    
    rows, cols = 1024, 1024
    matrix = CUDA.rand(Float32, rows, cols)
    indices = CuArray(shuffle!(collect(1:cols))[1:100])
    
    println("Matrix size: $rows × $cols")
    println("Accessing 100 random columns")
    
    # Benchmark different layouts
    t_original, _ = benchmark_unoptimized(matrix, indices)
    t_optimized, _ = benchmark_optimized(matrix, indices)
    
    println("Original layout time: $(round(t_original * 1000, digits=3)) ms")
    println("Optimized layout time: $(round(t_optimized * 1000, digits=3)) ms")
    println("Speedup: $(round(t_original / t_optimized, digits=2))x")
    
    # Demo 4: Memory Bandwidth Utilization
    println("\n4. Memory Bandwidth Analysis")
    println("-" * 40)
    
    data_size = 10_000_000  # 10M floats
    data = CUDA.rand(Float32, data_size)
    
    # Simple copy kernel for bandwidth test
    function copy_kernel(dst, src, n)
        idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        if idx <= n
            dst[idx] = src[idx]
        end
        return nothing
    end
    
    dst = similar(data)
    
    # Warmup
    @cuda threads=256 blocks=cld(data_size, 256) copy_kernel(dst, data, data_size)
    CUDA.synchronize()
    
    # Benchmark
    elapsed = @belapsed begin
        @cuda threads=256 blocks=cld($data_size, 256) $copy_kernel($dst, $data, $data_size)
        CUDA.synchronize()
    end samples=10
    
    bandwidth_stats = analyze_bandwidth_utilization(data_size, elapsed)
    
    println("Data transferred: $(round(bandwidth_stats["bytes_transferred"] / 1024^2, digits=1)) MB")
    println("Time: $(round(bandwidth_stats["elapsed_time_ms"], digits=2)) ms")
    println("Achieved bandwidth: $(round(bandwidth_stats["achieved_bandwidth_gb_s"], digits=1)) GB/s")
    println("Utilization: $(round(bandwidth_stats["utilization_percent"], digits=1))%")
    println("Efficiency: $(bandwidth_stats["efficiency_rating"])")
    
    # Demo 5: Memory Profiling
    println("\n5. Memory Access Pattern Profiling")
    println("-" * 40)
    
    # Profile a simple reduction kernel
    function reduction_kernel(output, input, n)
        tid = threadIdx().x
        bid = blockIdx().x
        
        shared_mem = @cuDynamicSharedMem(Float32, (blockDim().x,))
        
        # Load data with coalesced access
        idx = tid + (bid - 1) * blockDim().x
        shared_mem[tid] = idx <= n ? input[idx] : 0.0f0
        sync_threads()
        
        # Reduction in shared memory
        stride = blockDim().x ÷ 2
        while stride > 0
            if tid <= stride && tid + stride <= blockDim().x
                shared_mem[tid] += shared_mem[tid + stride]
            end
            sync_threads()
            stride ÷= 2
        end
        
        # Write result
        if tid == 1
            output[bid] = shared_mem[1]
        end
        
        return nothing
    end
    
    n = 1_000_000
    input = CUDA.rand(Float32, n)
    blocks = 1024
    threads = 256
    output = CUDA.zeros(Float32, blocks)
    
    profile = profile_memory_access(
        (inp, out) -> begin
            @cuda threads=threads blocks=blocks shmem=threads*sizeof(Float32) reduction_kernel(out, inp, n)
        end,
        input, output,
        warmup_runs=5,
        profile_runs=20
    )
    
    println("Profile Results:")
    println("  Total time: $(round(profile.total_time * 1000, digits=2)) ms")
    println("  Achieved bandwidth: $(round(profile.achieved_bandwidth, digits=1)) GB/s")
    println("  Bandwidth utilization: $(round(profile.bandwidth_utilization * 100, digits=1))%")
    
    if haskey(profile.read_patterns, "main")
        pattern = profile.read_patterns["main"]
        println("  Access pattern: $(pattern.pattern_type)")
        println("  Cache hit rate: $(round(pattern.cache_hit_rate * 100, digits=1))%")
    end
    
    # Demo 6: Prefetching Strategy
    println("\n6. Prefetching for Sequential Access")
    println("-" * 40)
    
    data = CUDA.rand(Float32, 1_000_000)
    cached_data = CachedArray(data, cache_line_size=128, prefetch_distance=8)
    
    # Simulate sequential access with prefetching
    access_pattern = collect(1:1000:100_000)
    prefetch_data!(cached_data, access_pattern)
    
    println("Prefetch configuration:")
    println("  Cache line size: $(cached_data.cache_line_size) bytes")
    println("  Prefetch distance: $(cached_data.prefetch_distance) elements")
    println("  Access history: $(length(cached_data.access_history)) accesses recorded")
    
    # Demo 7: Memory Statistics
    println("\n7. Current Memory Usage")
    println("-" * 40)
    
    mem_stats = get_memory_stats()
    
    println("GPU Memory Status:")
    println("  Allocated: $(round(mem_stats["allocated_mb"], digits=1)) MB")
    println("  Reserved: $(round(mem_stats["reserved_mb"], digits=1)) MB")
    println("  Free: $(round(mem_stats["free_mb"], digits=1)) MB")
    println("  Total: $(round(mem_stats["total_mb"], digits=1)) MB")
    println("  Utilization: $(round(mem_stats["utilization_percent"], digits=1))%")
    
    # Summary
    println("\n" * "=" ^ 60)
    println("Memory Optimization Summary")
    println("=" ^ 60)
    println("Key techniques demonstrated:")
    println("  ✓ Coalesced memory access patterns")
    println("  ✓ Memory pooling for reduced allocation overhead")
    println("  ✓ Data layout optimization (tiling, Z-order)")
    println("  ✓ Bandwidth utilization monitoring")
    println("  ✓ Memory access pattern profiling")
    println("  ✓ Prefetching strategies")
    println("\nThese optimizations can significantly improve GPU performance!")
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    demo_memory_optimization()
end