# Test GPU Device Manager

push!(LOAD_PATH, joinpath(@__DIR__, "../.."))
using HSOF
using CUDA
using Test

# Load configurations
include("../../configs/config_loader.jl")
ConfigLoader.load_configs("dev")

# Test device manager
include("../../src/gpu/device_manager.jl")

println("=" ^ 60)
println("GPU Device Manager Test")
println("=" ^ 60)

# Initialize with configuration
config = ConfigLoader.get_config()
gpu_manager = DeviceManager.initialize_devices(Dict("gpu" => config.gpu_config))

# Get device information
println("\n📊 Device Information:")
device_info = DeviceManager.get_device_info()
println("  Device Count: ", device_info["device_count"])
println("  Single GPU Mode: ", device_info["single_gpu_mode"])

for dev in device_info["devices"]
    println("\n  GPU $(dev["index"]):")
    println("    Name: $(dev["name"])")
    println("    Compute Capability: $(dev["compute_capability"])")
    println("    Total Memory: $(dev["total_memory_gb"]) GB")
    println("    Memory Stats:")
    if haskey(dev, "memory")
        mem = dev["memory"]
        println("      Used: $(round(mem["used_gb"], digits=2)) GB ($(round(mem["usage_percent"], digits=1))%)")
        println("      Free: $(round(mem["free_gb"], digits=2)) GB")
        if haskey(mem, "limit_gb")
            println("      Limit: $(mem["limit_gb"]) GB")
        end
    end
    println("    Streams: $(dev["streams"])")
end

# Validate environment
println("\n🔍 Environment Validation:")
results, issues = DeviceManager.validate_gpu_environment()

for (check, passed) in sort(collect(results))
    status = passed ? "✓" : "✗"
    color = passed ? :green : :red
    printstyled("  $check: $status\n"; color=color)
end

if !isempty(issues)
    println("\n⚠️  Issues found:")
    for issue in issues
        println("  - $issue")
    end
end

# Test memory allocation
println("\n💾 Testing Memory Allocation:")
try
    # Allocate some memory
    GPUManager.set_device!(0)
    test_array = MemoryManager.allocate(Float32, 1000, 1000)
    println("  ✓ Allocated 1000x1000 Float32 array")
    
    # Check memory stats
    stats = MemoryManager.get_memory_stats(0)
    println("  Tracked allocations: $(stats["allocations_count"])")
    println("  Tracked memory: $(round(stats["tracked_gb"], digits=3)) GB")
    
    # Free memory
    MemoryManager.free(test_array)
    println("  ✓ Freed test array")
catch e
    println("  ✗ Memory allocation failed: $e")
end

# Test stream operations
println("\n🌊 Testing CUDA Streams:")
if StreamManager.get_stream_count(0) > 0
    try
        # Simple kernel for testing
        function add_kernel(a, b, c)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            if i <= length(c)
                c[i] = a[i] + b[i]
            end
            return
        end
        
        # Test on different streams
        n = 10000
        a = CUDA.rand(Float32, n)
        b = CUDA.rand(Float32, n)
        c1 = CUDA.zeros(Float32, n)
        c2 = CUDA.zeros(Float32, n)
        
        # Launch on stream 1
        StreamManager.with_stream(0, 1) do
            @cuda threads=256 blocks=cld(n, 256) add_kernel(a, b, c1)
        end
        
        # Launch on stream 2 (if available)
        if StreamManager.get_stream_count(0) >= 2
            StreamManager.with_stream(0, 2) do
                @cuda threads=256 blocks=cld(n, 256) add_kernel(a, b, c2)
            end
        end
        
        # Synchronize all streams
        StreamManager.synchronize_all_streams(0)
        
        println("  ✓ Stream operations completed")
        println("  Stream count: $(StreamManager.get_stream_count(0))")
    catch e
        println("  ✗ Stream operations failed: $e")
    end
else
    println("  ⚠️  No streams initialized")
end

# Memory pool info
println("\n📊 Memory Pool Information:")
pool_info = MemoryManager.get_pool_info()
for (device_id, info) in sort(collect(pool_info))
    println("  GPU $device_id:")
    println("    Limit: $(info["limit_gb"]) GB")
    println("    Allocated: $(round(info["allocated_gb"], digits=3)) GB")
    println("    Peak: $(round(info["peak_allocated_gb"], digits=3)) GB")
end

# Cleanup
println("\n🧹 Cleaning up...")
DeviceManager.cleanup()
println("✓ Cleanup complete")

println("=" ^ 60)