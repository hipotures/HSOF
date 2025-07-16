# CUDA Environment Validation Script

push!(LOAD_PATH, "@")
using HSOF
include("../configs/cuda_config.jl")

println("=" ^ 60)
println("CUDA Environment Validation for HSOF Project")
println("=" ^ 60)

# Load and apply CUDA configuration
config = CUDAConfig.load_config()
CUDAConfig.apply_config!(config)

# Run validation
println("\nRunning CUDA environment validation...")
results = CUDAConfig.validate_cuda_environment()

# Display results
println("\n📊 CUDA Environment Summary:")
println("-" ^ 40)
println("CUDA Functional: ", results["cuda_functional"] ? "✓" : "✗")

if results["cuda_functional"]
    println("CUDA Version: ", results["cuda_version"])
    println("Driver Version: ", results["driver_version"])
    println("GPU Count: ", results["gpu_count"])
    
    println("\n🎮 GPU Information:")
    for gpu in results["gpus"]
        println("\n  GPU $(gpu["index"]):")
        println("    Name: ", gpu["name"])
        println("    Compute Capability: ", gpu["compute_capability"][1], ".", gpu["compute_capability"][2], 
                gpu["compute_capability_check"] ? " ✓" : " ✗ (8.9+ required)")
        println("    Total Memory: ", round(gpu["total_memory_gb"], digits=2), " GB",
                gpu["memory_check"] ? " ✓" : " ✗ (20GB+ required)")
        println("    Free Memory: ", round(gpu["free_memory_gb"], digits=2), " GB")
        println("    Multiprocessors: ", gpu["multiprocessor_count"])
        println("    Max Threads/Block: ", gpu["max_threads_per_block"])
    end
    
    println("\n🧪 Basic CUDA Test: ", results["basic_cuda_test"] ? "✓ Passed" : "✗ Failed")
    
    # Test kernel compilation
    println("\n🔧 Testing CUDA kernel compilation...")
    kernel_test = CUDAConfig.test_cuda_kernel()
    println("Kernel Test: ", kernel_test ? "✓ Passed" : "✗ Failed")
    
    # Test PCIe bandwidth between GPUs
    if results["gpu_count"] >= 2
        println("\n📡 Testing multi-GPU capabilities...")
        using CUDA
        
        # Test peer access
        CUDA.device!(0)
        can_access = CUDA.can_access_peer(CUDA.CuDevice(1))
        println("GPU 0 → GPU 1 peer access: ", can_access ? "✓ Available" : "✗ Not available")
        
        # Simple bandwidth test
        println("\n📊 PCIe Bandwidth Test (GPU 0 ↔ GPU 1):")
        n = 100_000_000  # 100M floats = 400MB
        
        CUDA.device!(0)
        a_gpu0 = CUDA.rand(Float32, n)
        
        CUDA.device!(1)
        a_gpu1 = CUDA.zeros(Float32, n)
        
        # Warm up
        copyto!(a_gpu1, a_gpu0)
        CUDA.synchronize()
        
        # Measure transfer time
        start_time = time()
        for i in 1:10
            copyto!(a_gpu1, a_gpu0)
            CUDA.synchronize()
        end
        elapsed = time() - start_time
        
        bandwidth_gb_s = (10 * n * sizeof(Float32) / 1e9) / elapsed
        println("    Transfer Rate: ", round(bandwidth_gb_s, digits=2), " GB/s",
                bandwidth_gb_s >= 8.0 ? " ✓" : " ⚠️ (8GB/s recommended)")
    end
    
    # Overall verdict
    println("\n" * ("=" ^ 60))
    all_checks = results["cuda_functional"] && 
                 results["gpu_count"] >= 1 &&
                 all(gpu["compute_capability_check"] && gpu["memory_check"] for gpu in results["gpus"]) &&
                 results["basic_cuda_test"] &&
                 kernel_test
    
    if all_checks
        printstyled("✓ CUDA environment is ready for HSOF!\n"; color=:green, bold=true)
    else
        printstyled("✗ CUDA environment needs attention!\n"; color=:red, bold=true)
        if results["gpu_count"] < 2
            println("  ⚠️  Only $(results["gpu_count"]) GPU(s) found. Project is optimized for 2 GPUs.")
        end
    end
else
    printstyled("✗ CUDA is not functional!\n"; color=:red, bold=true)
    println("Please check your CUDA installation.")
end

println("=" ^ 60)