# PCIe Bandwidth and Multi-GPU Validation

push!(LOAD_PATH, joinpath(@__DIR__, "../.."))
using HSOF
using CUDA
include("../../src/gpu/gpu_manager.jl")

println("=" ^ 60)
println("PCIe and Multi-GPU Validation")
println("=" ^ 60)

# Initialize GPU Manager
manager = GPUManager.initialize()

if GPUManager.is_single_gpu_mode()
    println("\n⚠️  Running in SINGLE GPU mode")
    println("   This project is optimized for dual RTX 4090 GPUs")
    println("   Performance will be limited with only 1 GPU")
    
    # Show single GPU info
    dev = GPUManager.get_device(0)
    println("\nAvailable GPU:")
    println("  $(dev.name)")
    println("  Compute Capability: $(dev.compute_capability[1]).$(dev.compute_capability[2])")
    println("  Memory: $(round(dev.total_memory/1024^3, digits=2)) GB")
    
    # Test single GPU performance
    println("\n📊 Single GPU Memory Bandwidth Test:")
    GPUManager.set_device!(0)
    
    # Allocate large arrays
    n = 100_000_000  # 400MB per array
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)
    
    # Warm up
    c .= a .+ b
    CUDA.synchronize()
    
    # Measure bandwidth
    start_time = time()
    for i in 1:10
        c .= a .+ b
        CUDA.synchronize()
    end
    elapsed_time = time() - start_time
    
    bandwidth_gb_s = (10 * 3 * n * sizeof(Float32) / 1e9) / elapsed_time
    println("  Memory Bandwidth: $(round(bandwidth_gb_s, digits=2)) GB/s")
    println("  Expected for RTX 4070 Ti: ~500-600 GB/s")
    
else
    println("\n✓ Multi-GPU mode active with $(GPUManager.device_count()) GPUs")
    
    # Show GPU info
    println("\nAvailable GPUs:")
    for i in 0:GPUManager.device_count()-1
        dev = GPUManager.get_device(i)
        println("  GPU $i: $(dev.name)")
        println("    Compute Capability: $(dev.compute_capability[1]).$(dev.compute_capability[2])")
        println("    Memory: $(round(dev.total_memory/1024^3, digits=2)) GB")
    end
    
    # Test PCIe bandwidth
    println("\n📡 PCIe Bandwidth Test:")
    results = GPUManager.benchmark_pcie_bandwidth(100)
    
    if results !== nothing
        for (pair, bandwidth) in sort(collect(results))
            status = bandwidth >= 8.0 ? "✓" : "⚠️"
            println("  $pair: $(round(bandwidth, digits=2)) GB/s $status")
        end
        
        avg_bandwidth = sum(values(results)) / length(results)
        println("\n  Average: $(round(avg_bandwidth, digits=2)) GB/s")
        
        if avg_bandwidth < 8.0
            println("  ⚠️  PCIe bandwidth below recommended 8 GB/s")
            println("     This may impact multi-GPU performance")
        end
    end
    
    # Test workload distribution
    println("\n📊 Workload Distribution Test:")
    total_work = 1_000_000
    distribution = GPUManager.distribute_workload(total_work)
    
    for (i, work) in enumerate(distribution)
        percentage = round(100 * work / total_work, digits=1)
        println("  GPU $(i-1): $work items ($percentage%)")
    end
end

# Memory status
println("\n💾 GPU Memory Status:")
mem_info = GPUManager.get_memory_info()
global total_vram = 0.0
global available_vram = 0.0

for (gpu_id, info) in sort(collect(mem_info))
    used_percent = round(100 * info["used_gb"] / info["total_gb"], digits=1)
    println("  GPU $gpu_id: $(round(info["used_gb"], digits=2))/$(round(info["total_gb"], digits=2)) GB ($used_percent% used)")
    global total_vram += info["total_gb"]
    global available_vram += info["free_gb"]
end

println("\n  Total VRAM: $(round(total_vram, digits=2)) GB")
println("  Available: $(round(available_vram, digits=2)) GB")

# Recommendations
println("\n📋 Recommendations:")
if GPUManager.is_single_gpu_mode()
    println("  • Add a second RTX 4090 GPU for optimal performance")
    println("  • Current GPU has less VRAM than recommended (24GB)")
    println("  • Feature selection will be limited to smaller datasets")
else
    println("  • Multi-GPU setup detected - optimal for large-scale processing")
    if total_vram < 48.0
        println("  • Total VRAM below recommended 48GB (2x24GB)")
    end
end

println("=" ^ 60)