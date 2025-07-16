#!/usr/bin/env julia

"""
Test GPU utilization in Stage 1
"""

push!(LOAD_PATH, "src/")
using CUDA
using BenchmarkTools
using Statistics

include("src/gpu_stage1.jl")

function test_gpu_vs_cpu_correlation()
    println("="^80)
    println("GPU vs CPU CORRELATION COMPUTATION TEST")
    println("="^80)
    
    # Test different dataset sizes
    test_configs = [
        (samples=1000, features=100),
        (samples=1000, features=500),
        (samples=1000, features=1000),
        (samples=5000, features=500),
    ]
    
    for config in test_configs
        n_samples = config.samples
        n_features = config.features
        
        println("\n" * "-"^60)
        println("Dataset: $n_samples samples × $n_features features")
        println("-"^60)
        
        # Generate random data
        X = randn(Float32, n_samples, n_features)
        y = randn(Float32, n_samples)
        feature_names = ["f$i" for i in 1:n_features]
        
        # GPU memory usage
        data_size_mb = (sizeof(X) + sizeof(y)) / 1024^2
        println("Data size: $(round(data_size_mb, digits=2)) MB")
        
        # CPU correlation (sequential)
        println("\nCPU Correlation (sequential):")
        cpu_time = @elapsed begin
            cpu_correlations = zeros(Float32, n_features)
            for i in 1:n_features
                cpu_correlations[i] = abs(cor(X[:, i], y))
            end
        end
        println("  Time: $(round(cpu_time * 1000, digits=2)) ms")
        println("  Features/second: $(round(n_features / cpu_time, digits=0))")
        
        # GPU Stage 1
        println("\nGPU Stage 1 (full pipeline):")
        gpu_time = @elapsed begin
            X_filtered, features_selected, indices = gpu_stage1_filter(
                X, y, feature_names,
                correlation_threshold=0.1,
                min_features_to_keep=10,
                variance_threshold=0.01
            )
        end
        
        # Just GPU kernel timing
        println("\nGPU Kernel Only (correlation):")
        X_gpu = CuArray(X)
        y_gpu = CuArray(y)
        correlation_scores = CUDA.zeros(Float32, n_features)
        
        # Warm-up
        threads_per_block = 256
        blocks = cld(n_features, threads_per_block)
        @cuda threads=threads_per_block blocks=blocks correlation_kernel!(
            correlation_scores, X_gpu, y_gpu, Int32(n_samples), Int32(n_features)
        )
        CUDA.synchronize()
        
        # Actual timing
        kernel_time = @elapsed begin
            @cuda threads=threads_per_block blocks=blocks correlation_kernel!(
                correlation_scores, X_gpu, y_gpu, Int32(n_samples), Int32(n_features)
            )
            CUDA.synchronize()
        end
        
        println("  Kernel time: $(round(kernel_time * 1000, digits=2)) ms")
        println("  Features/second: $(round(n_features / kernel_time, digits=0))")
        
        # Calculate speedup
        speedup = cpu_time / kernel_time
        println("\nSpeedup:")
        println("  GPU kernel vs CPU: $(round(speedup, digits=1))x faster")
        println("  Theoretical max threads: $n_features")
        println("  Actual GPU threads: $(threads_per_block * blocks)")
        println("  Thread efficiency: $(round(100 * n_features / (threads_per_block * blocks), digits=1))%")
        
        # Memory bandwidth utilization
        data_processed = 2 * sizeof(X) + sizeof(y)  # Read X twice (for mean and correlation)
        bandwidth_gb_s = (data_processed / kernel_time) / 1024^3
        println("\nMemory bandwidth:")
        println("  Data processed: $(round(data_processed / 1024^2, digits=2)) MB")
        println("  Effective bandwidth: $(round(bandwidth_gb_s, digits=2)) GB/s")
        
        # Verify results match
        gpu_results = Array(correlation_scores)
        max_diff = maximum(abs.(gpu_results - cpu_correlations))
        println("\nAccuracy check:")
        println("  Max difference GPU vs CPU: $(max_diff)")
        println("  Results match: $(max_diff < 1e-4 ? "✅ Yes" : "❌ No")")
        
        # Cleanup
        CUDA.reclaim()
    end
    
    # GPU info
    println("\n" * "="^80)
    println("GPU INFORMATION")
    println("="^80)
    println("Device: $(CUDA.name(CUDA.device()))")
    println("Compute capability: $(CUDA.capability(CUDA.device()))")
    println("Total memory: $(round(CUDA.total_memory() / 1024^3, digits=2)) GB")
    println("Available memory: $(round(CUDA.available_memory() / 1024^3, digits=2)) GB")
    
    # Theoretical limits
    device = CUDA.device()
    max_threads = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    max_blocks = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
    sm_count = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    
    println("\nGPU Limits:")
    println("  Max threads per block: $max_threads")
    println("  Max blocks: $max_blocks")
    println("  Streaming Multiprocessors: $sm_count")
    println("  Max total threads: $(max_blocks * max_threads)")
end

# Run test
if abspath(PROGRAM_FILE) == @__FILE__
    test_gpu_vs_cpu_correlation()
end