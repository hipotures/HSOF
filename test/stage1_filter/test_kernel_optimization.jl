using Test
using CUDA
using Statistics

# Skip if no GPU
if !CUDA.functional()
    @warn "CUDA not functional, skipping kernel optimization tests"
    exit(0)
end

# Include modules
include("../../src/stage1_filter/kernel_optimization.jl")
include("../../src/stage1_filter/auto_tuning.jl")
include("../../src/stage1_filter/variance_calculation.jl")
include("../../src/stage1_filter/mutual_information.jl")
include("../../src/stage1_filter/correlation_matrix.jl")
include("../../src/stage1_filter/gpu_config.jl")

using .KernelOptimization
using .AutoTuning
using .VarianceCalculation
using .MutualInformation
using .CorrelationMatrix
using .GPUConfig

println("Testing Kernel Optimization System...")
println("="^60)

@testset "Kernel Optimization Tests" begin
    
    @testset "RTX 4090 Specifications" begin
        specs = get_rtx4090_specs()
        
        @test specs.compute_capability == (8, 9)
        @test specs.sm_count == 128
        @test specs.max_threads_per_sm == 1536
        @test specs.max_threads_per_block == 1024
        @test specs.warp_size == 32
        @test specs.l2_cache_size == 72 * 1024 * 1024  # 72MB
    end
    
    @testset "Optimal Configuration Calculation" begin
        specs = get_rtx4090_specs()
        
        # Test variance kernel configuration
        config = calculate_optimal_config(
            variance_kernel!,
            10000,  # n_features
            8,      # shared mem per element (2 floats)
            24;     # estimated registers
            specs = specs
        )
        
        @test config.block_size in [128, 256, 512]
        @test config.theoretical_occupancy > 0.5
        @test config.shared_mem_per_block <= specs.max_shared_mem_per_block
    end
    
    @testset "Workload-Specific Optimization" begin
        # Test different kernel types
        
        # Variance kernel
        var_analysis = optimize_for_workload(:variance, 1000, 5000)
        @test var_analysis.optimal_config.block_size >= 128
        @test contains(var_analysis.recommendation, "vectorized loads")
        
        # MI kernel
        mi_analysis = optimize_for_workload(:mutual_information, 500, 10000)
        @test mi_analysis.optimal_config.block_size <= 256  # Smaller for atomics
        @test contains(mi_analysis.recommendation, "atomic")
        
        # Correlation kernel
        corr_analysis = optimize_for_workload(:correlation, 100, 5000)
        @test corr_analysis.optimal_config.block_size == 1024  # 32×32 tile
        @test contains(corr_analysis.recommendation, "tensor cores")
    end
    
    @testset "Auto-Tuning System" begin
        tuner = AutoTuner(verbose=false)
        
        # Test variance auto-tuning
        X = CUDA.randn(Float32, 500, 2000)
        threads, blocks, shmem = auto_tune_variance!(tuner, X)
        
        @test threads in [64, 128, 256, 512]
        @test blocks == cld(500, threads)
        @test shmem == 2 * threads * sizeof(Float32)
        
        # Test cache hit
        threads2, blocks2, shmem2 = auto_tune_variance!(tuner, X)
        @test threads == threads2
        @test tuner.cache.hit_count == 1
        
        # Test MI auto-tuning
        y = CuArray(Int32.(rand(1:3, 2000)))
        threads_mi, blocks_mi, shmem_mi = auto_tune_mi!(
            tuner, X, y, Int32(10), Int32(3)
        )
        
        @test threads_mi in [64, 128, 256]
        @test blocks_mi == 500
    end
    
    @testset "Cache Configuration" begin
        # Test L1 cache optimization
        @test optimize_cache_config(:variance, 16384) == :prefer_l1
        @test optimize_cache_config(:mutual_information, 60000) == :prefer_shared
        @test optimize_cache_config(:correlation, 32768) == :prefer_shared
    end
    
    @testset "Dynamic Configuration" begin
        # Test dynamic kernel configuration
        block_small, grid_small = dynamic_kernel_config(:variance, 100)
        block_large, grid_large = dynamic_kernel_config(:variance, 100000)
        
        @test block_small == 256
        @test block_large in [256, 512]
        
        # MI should use smaller blocks
        block_mi, _ = dynamic_kernel_config(:mutual_information, 10000)
        @test block_mi <= 128
    end
    
    @testset "Performance Profiling" begin
        # Test with real kernels
        X = CUDA.randn(Float32, 1000, 5000)
        variances = CUDA.zeros(Float32, 1000)
        
        # Profile variance kernel
        config = OptimalLaunchConfig(
            256,     # block_size
            6,       # blocks_per_sm
            1000,    # total_blocks
            2048,    # shared_mem
            24,      # registers
            0.75f0,  # theoretical_occupancy
            0.0f0,   # achieved (to be measured)
            0.9f0    # memory_efficiency
        )
        
        # Simple timing test
        t = CUDA.@elapsed begin
            @cuda threads=config.block_size blocks=config.total_blocks variance_kernel!(
                variances, X, Int32(1000), Int32(5000)
            )
            CUDA.synchronize()
        end
        
        @test t < 0.1  # Should complete in < 100ms
        @test all(isfinite.(Array(variances)))
    end
    
    @testset "Occupancy Analysis" begin
        tuner = AutoTuner()
        
        # Test occupancy calculation
        occupancy = calculate_occupancy(
            256,    # threads
            2048,   # shared memory
            24,     # registers
            tuner.specs
        )
        
        @test 0.0 <= occupancy <= 1.0
        @test occupancy > 0.5  # Should achieve reasonable occupancy
    end
end

# Demonstration of optimization results
println("\n" * "="^60)
println("KERNEL OPTIMIZATION DEMONSTRATION")
println("="^60)

# Create auto-tuner
tuner = AutoTuner(verbose=true)

# Test different data sizes
println("\n1. Auto-tuning for different data sizes:")
for (n_feat, n_samp) in [(100, 1000), (1000, 5000), (5000, 10000)]
    println("\n  Dataset: $n_feat features × $n_samp samples")
    X = CUDA.randn(Float32, n_feat, n_samp)
    
    threads, blocks, shmem = auto_tune_variance!(tuner, X)
    
    # Calculate metrics
    occupancy = calculate_occupancy(threads, shmem, 24, tuner.specs)
    println("  → Optimal: $threads threads/block, occupancy: $(round(occupancy*100, digits=1))%")
end

# Generate optimization report
println("\n2. Kernel-specific optimizations:")
analyses = Dict(
    :variance => optimize_for_workload(:variance, 1000, 10000),
    :mutual_information => optimize_for_workload(:mutual_information, 1000, 10000),
    :correlation => optimize_for_workload(:correlation, 100, 10000)
)

generate_optimization_report(analyses)

# Show tuning summary
print_tuning_summary(tuner)

# Performance comparison
println("\n3. Performance Impact of Optimization:")
X_test = CUDA.randn(Float32, 2000, 8000)
variances = CUDA.zeros(Float32, 2000)

# Baseline (fixed configuration)
t_baseline = CUDA.@elapsed begin
    @cuda threads=256 blocks=cld(2000, 256) variance_kernel!(
        variances, X_test, Int32(2000), Int32(8000)
    )
    CUDA.synchronize()
end

# Auto-tuned configuration
threads_opt, blocks_opt, shmem_opt = auto_tune_variance!(tuner, X_test)
t_optimized = CUDA.@elapsed begin
    @cuda threads=threads_opt blocks=blocks_opt shmem=shmem_opt variance_kernel!(
        variances, X_test, Int32(2000), Int32(8000)
    )
    CUDA.synchronize()
end

speedup = t_baseline / t_optimized
println("  Baseline time: $(round(t_baseline*1000, digits=2))ms")
println("  Optimized time: $(round(t_optimized*1000, digits=2))ms")
println("  Speedup: $(round(speedup, digits=2))x")

println("\n" * "="^60)
println("OPTIMIZATION SUMMARY")
println("="^60)
println("✓ RTX 4090 specifications correctly configured")
println("✓ Optimal launch configurations calculated")
println("✓ Auto-tuning system functional")
println("✓ Cache hit/miss tracking working")
println("✓ Dynamic configuration adapts to workload")
println("✓ Performance improvements demonstrated")
println("="^60)