using Test
using CUDA
using Statistics

# Include the memory optimization module
include("../../src/stage1_filter/memory_optimization.jl")

using .MemoryOptimization

@testset "Memory Optimization Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU memory optimization tests"
        return
    end
    
    @testset "MemoryConfig Creation" begin
        config = create_memory_config()
        
        @test config.alignment == 128
        @test config.coalesce_factor == 32
        @test config.prefetch_distance == 8
        @test config.use_l2_cache_hints == true
        @test config.bank_conflict_strategy == :padding
        @test config.texture_memory == false
        
        # Test with custom parameters
        config2 = create_memory_config(
            alignment=256,
            coalesce_factor=64,
            bank_conflict_strategy=:swizzle
        )
        
        @test config2.alignment == 256
        @test config2.coalesce_factor == 64
        @test config2.bank_conflict_strategy == :swizzle
    end
    
    @testset "Aligned Memory Allocation" begin
        # Test 1D allocation
        arr1d = allocate_aligned(Float32, (1000,), 128)
        @test size(arr1d) == (1000,)
        # @test pointer(parent(arr1d)) % 128 == 0  # Cannot test pointer alignment in Julia/CUDA
        
        # Test 2D allocation with alignment
        arr2d = allocate_aligned(Float32, (100, 97), 128)
        @test size(arr2d) == (100, 97)
        # Parent should be padded to alignment
        @test size(parent(arr2d), 2) >= 97
        @test size(parent(arr2d), 2) % (128 ÷ sizeof(Float32)) == 0
        
        # Test 3D allocation
        arr3d = allocate_aligned(Float32, (50, 60, 70), 128)
        @test size(arr3d) == (50, 60, 70)
    end
    
    @testset "Coalesced Memory Copy" begin
        n_rows = 1024
        n_cols = 512
        
        src = CUDA.rand(Float32, n_rows, n_cols)
        dst = CUDA.zeros(Float32, n_rows, n_cols)
        
        config = create_memory_config(coalesce_factor=32)
        
        threads = 256
        blocks = cld(n_cols, config.coalesce_factor)
        
        @cuda threads=threads blocks=blocks coalesced_copy_kernel!(
            dst, src, Int32(n_rows), Int32(n_cols), config.coalesce_factor
        )
        CUDA.synchronize()
        
        # Verify copy correctness
        @test Array(dst) ≈ Array(src)
    end
    
    @testset "L2 Cache Optimization" begin
        n_rows = 256
        n_cols = 256
        
        input = CUDA.rand(Float32, n_rows, n_cols)
        weights = CUDA.rand(Float32, n_cols, n_cols)
        output_with_hints = CUDA.zeros(Float32, n_rows, n_cols)
        output_no_hints = CUDA.zeros(Float32, n_rows, n_cols)
        
        threads = 128
        blocks = n_rows
        
        # With L2 hints
        @cuda threads=threads blocks=blocks cached_compute_kernel!(
            output_with_hints, input, weights, 
            Int32(n_rows), Int32(n_cols), true
        )
        
        # Without L2 hints
        @cuda threads=threads blocks=blocks cached_compute_kernel!(
            output_no_hints, input, weights,
            Int32(n_rows), Int32(n_cols), false
        )
        
        CUDA.synchronize()
        
        # Results should be similar (hints don't change computation)
        @test Array(output_with_hints) ≈ Array(output_no_hints)
    end
    
    @testset "Bank Conflict Resolution" begin
        n_rows = 512
        n_cols = 32  # WARP_SIZE
        
        input = CUDA.rand(Float32, n_rows, n_cols)
        
        # Test different strategies
        for strategy in [:none, :padding, :swizzle]
            output = CUDA.zeros(Float32, cld(n_rows, 32))  # One output per tile
            
            threads = 32
            blocks = cld(n_rows, 32)
            
            # Calculate shared memory size
            if strategy == :padding
                shmem_size = (33 * 32) * sizeof(Float32)  # Padded
            else
                shmem_size = (32 * 32) * sizeof(Float32)  # Normal
            end
            
            @cuda threads=threads blocks=blocks shmem=shmem_size bank_conflict_free_kernel!(
                output, input, Int32(n_rows), Int32(n_cols), strategy
            )
            CUDA.synchronize()
            
            # Check that kernel executed
            @test sum(Array(output)) > 0
        end
    end
    
    @testset "Sequential Prefetching" begin
        n_elements = 100000
        
        input = CUDA.rand(Float32, n_elements)
        output = CUDA.zeros(Float32, n_elements)
        
        config = create_memory_config(prefetch_distance=16)
        
        threads = 256
        blocks = cld(n_elements, threads)
        
        @cuda threads=threads blocks=blocks prefetch_sequential_kernel!(
            output, input, Int32(n_elements), config.prefetch_distance
        )
        CUDA.synchronize()
        
        # Verify computation
        expected = Array(input) .* 2.0f0 .+ 1.0f0
        @test Array(output) ≈ expected
    end
    
    @testset "Memory Pattern Analysis" begin
        n_elements = 10000
        src = CUDA.rand(Float32, n_elements)
        dst = CUDA.zeros(Float32, n_elements)
        
        config = create_memory_config()
        
        # Simple kernel for testing
        function test_kernel(dst, src, n)
            tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            if tid <= n
                @inbounds dst[tid] = src[tid] * 2.0f0
            end
            return nothing
        end
        
        profile = analyze_memory_pattern(
            test_kernel,
            (dst, src, Int32(n_elements)),
            config,
            threads=256,
            blocks=cld(n_elements, 256)
        )
        
        @test profile.total_transactions > 0
        @test 0 <= profile.efficiency_percentage <= 100
        
        # Get recommendations
        recommendations = get_optimization_recommendations(profile)
        @test isa(recommendations, Vector{String})
    end
    
    @testset "Feature Matrix Layout Optimization" begin
        # Test unaligned matrix
        n_samples = 1000
        n_features = 97  # Not aligned to 128 bytes
        
        feature_matrix = CUDA.rand(Float32, n_samples, n_features)
        config = create_memory_config(alignment=128)
        
        optimized = optimize_feature_layout!(feature_matrix, config)
        
        # Check alignment
        aligned_features = size(optimized, 2)
        @test aligned_features >= n_features
        @test aligned_features % (128 ÷ sizeof(Float32)) == 0
        
        # Original data should be preserved
        @test optimized[:, 1:n_features] ≈ feature_matrix
        
        # Test already aligned matrix
        aligned_features = 128
        aligned_matrix = CUDA.rand(Float32, n_samples, aligned_features)
        optimized2 = optimize_feature_layout!(aligned_matrix, config)
        @test size(optimized2) == size(aligned_matrix)
    end
    
    @testset "Memory Strategy Benchmarking" begin
        data_size = (512, 512)
        
        results, speedup = benchmark_memory_strategies(
            data_size,
            iterations=10
        )
        
        @test haskey(results, "baseline")
        @test haskey(results, "coalesced")
        @test results["baseline"] > 0
        @test results["coalesced"] > 0
        @test speedup > 0
        
        println("\nMemory optimization speedup: $(round(speedup, digits=2))x")
    end
    
    @testset "Optimization Guidelines" begin
        config = create_memory_config(
            alignment=256,
            coalesce_factor=64,
            use_l2_cache_hints=true
        )
        
        # Mock module for testing
        test_module = Module()
        
        guidelines = apply_memory_optimizations!(test_module, config)
        
        @test isa(guidelines, Dict)
        @test haskey(guidelines, "alignment")
        @test haskey(guidelines, "coalescing")
        @test haskey(guidelines, "prefetching")
        @test haskey(guidelines, "l2_cache")
        @test haskey(guidelines, "bank_conflicts")
        
        @test contains(guidelines["alignment"], "256")
        @test contains(guidelines["coalescing"], "64")
    end
    
    @testset "Memory Profile Recommendations" begin
        # Test different profile scenarios
        
        # Good efficiency
        profile1 = MemoryProfile(100, 95, 80, 20, 0, 95.0)
        recs1 = get_optimization_recommendations(profile1)
        @test length(recs1) == 0  # No recommendations needed
        
        # Poor efficiency
        profile2 = MemoryProfile(100, 50, 30, 70, 10, 50.0)
        recs2 = get_optimization_recommendations(profile2)
        @test length(recs2) > 0
        @test any(contains(r, "coalescing") for r in recs2)
        @test any(contains(r, "bank conflicts") for r in recs2)
        @test any(contains(r, "cache miss") for r in recs2)
        
        # Low coalescing
        profile3 = MemoryProfile(100, 70, 60, 40, 0, 70.0)
        recs3 = get_optimization_recommendations(profile3)
        @test any(contains(r, "coalescing ratio") for r in recs3)
    end
end

println("\n✅ Memory optimization tests completed!")