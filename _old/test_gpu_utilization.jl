#!/usr/bin/env julia

"""
Simple test to check GPU utilization with actual GPU kernels from HSOF
"""

using Pkg
Pkg.activate(".")

using CUDA
using Statistics
using Random
using BenchmarkTools

println("GPU Utilization Test for HSOF")
println("=" ^ 50)

# Check if GPU is available
if !CUDA.functional()
    println("❌ CUDA not functional!")
    exit(1)
end

println("✅ CUDA functional")
println("GPU: ", CUDA.name(CUDA.device()))
println("Memory: ", round(CUDA.totalmem(CUDA.device())/1024^3, digits=2), " GB")

# Generate test data
println("\n📊 Generating test data...")
n_samples = 10000
n_features = 1000

Random.seed!(42)
X_cpu = randn(Float32, n_samples, n_features)
y_cpu = Float32.(rand(0:1, n_samples))

# Transfer to GPU
println("📤 Transferring data to GPU...")
X_gpu = CuArray(X_cpu)
y_gpu = CuArray(y_cpu)

println("✅ Data on GPU: ", sizeof(X_gpu) ÷ 1024^2, " MB + ", sizeof(y_gpu) ÷ 1024, " KB")

# Test 1: Simple CUDA operations to warm up
println("\n🔥 Warming up GPU...")

function warmup_kernel!(data)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= length(data)
        data[idx] = data[idx] * 2.0f0
    end
    return nothing
end

@cuda threads=256 blocks=64 warmup_kernel!(X_gpu)
CUDA.synchronize()
println("✅ GPU warmed up")

# Test 2: Load the HSOF GPU modules
println("\n🔧 Loading HSOF GPU modules...")
try
    include("src/stage1_filter/gpu_memory_layout.jl")
    using .GPUMemoryLayout
    
    include("src/stage1_filter/mutual_information.jl")
    using .MutualInformation
    
    println("✅ HSOF GPU modules loaded")
    
    # Test mutual information GPU kernel
    println("\n⚡ Testing HSOF Mutual Information GPU kernels...")
    
    # Create histogram buffers
    histogram_buffers = GPUMemoryLayout.create_histogram_buffers(n_features)
    mi_config = MutualInformation.create_mi_config(n_features, n_samples)
    
    # Allocate MI scores on GPU
    mi_scores_gpu = CUDA.zeros(Float32, n_features)
    
    println("🚀 Running GPU mutual information calculation...")
    println("   This should utilize GPU significantly...")
    
    # Run the calculation
    start_time = time()
    MutualInformation.compute_mutual_information!(
        mi_scores_gpu,
        X_gpu,
        y_gpu,
        histogram_buffers,
        mi_config
    )
    CUDA.synchronize()
    elapsed_time = time() - start_time
    
    println("✅ GPU MI calculation completed in $(round(elapsed_time, digits=3))s")
    
    # Get results back to CPU
    mi_scores_cpu = Array(mi_scores_gpu)
    println("📈 Top 5 MI scores: ", round.(sort(mi_scores_cpu, rev=true)[1:5], digits=4))
    
catch e
    println("❌ Failed to load HSOF GPU modules: $e")
    println("Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

# Test 3: Intensive GPU computation to really stress the GPU
println("\n🔥 Running intensive GPU computation...")
println("   (This should definitely show up in nvidia-smi)")

# Create large arrays for intensive computation
large_n = 100_000_000  # 100M elements
println("📤 Allocating large GPU arrays ($(round(large_n * 4 / 1024^2)) MB each)...")

try
    a_gpu = CUDA.rand(Float32, large_n)
    b_gpu = CUDA.rand(Float32, large_n)
    c_gpu = CUDA.similar(a_gpu)
    
    println("🚀 Running intensive matrix operations...")
    println("   Monitor GPU utilization with: watch -n 0.5 nvidia-smi")
    
    # Run multiple iterations of intensive operations
    for i in 1:10
        println("   Iteration $i/10...")
        
        # Vector addition
        c_gpu .= a_gpu .+ b_gpu
        CUDA.synchronize()
        
        # Element-wise operations
        c_gpu .= sin.(a_gpu) .* cos.(b_gpu) .+ sqrt.(abs.(a_gpu))
        CUDA.synchronize()
        
        # Matrix multiplication (reshape to square-ish matrices)
        n_side = Int(floor(sqrt(large_n / 4)))  # Make it smaller for matrix ops
        if i == 1
            println("     Matrix size: $(n_side)x$(n_side)")
        end
        
        a_mat = reshape(a_gpu[1:n_side^2], n_side, n_side)
        b_mat = reshape(b_gpu[1:n_side^2], n_side, n_side)
        c_mat = a_mat * b_mat
        CUDA.synchronize()
        
        sleep(0.1)  # Small delay to allow observation
    end
    
    println("✅ Intensive computation completed")
    
    # Free memory
    a_gpu = nothing
    b_gpu = nothing
    c_gpu = nothing
    GC.gc()
    CUDA.reclaim()
    
catch e
    println("❌ Intensive computation failed: $e")
    if isa(e, CUDA.OutOfGPUMemoryError)
        println("   GPU out of memory - trying smaller arrays...")
        
        # Try with smaller arrays
        small_n = 10_000_000  # 10M elements
        a_gpu = CUDA.rand(Float32, small_n)
        b_gpu = CUDA.rand(Float32, small_n)
        
        for i in 1:20
            c_gpu = a_gpu .+ b_gpu
            c_gpu .= sin.(c_gpu) .* cos.(a_gpu)
            CUDA.synchronize()
        end
        
        println("✅ Smaller computation completed")
    end
end

# Test 4: Check memory usage
println("\n💾 GPU Memory Status:")
println("   Available: $(round(CUDA.available_memory()/1024^2)) MB")
println("   Used: $(round((CUDA.totalmem(CUDA.device()) - CUDA.available_memory())/1024^2)) MB")
println("   Total: $(round(CUDA.totalmem(CUDA.device())/1024^2)) MB")

println("\n" * "=" ^ 50)
println("GPU utilization test completed!")
println("Check nvidia-smi output during execution to see if GPU was utilized.")
println("If GPU utilization was 0% throughout, there may be an issue with:")
println("1. GPU kernel execution")
println("2. CUDA context initialization") 
println("3. GPU driver configuration")