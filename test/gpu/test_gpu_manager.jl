# Test GPU Manager Module

push!(LOAD_PATH, joinpath(@__DIR__, "../.."))
using HSOF
include("../../src/gpu/gpu_manager.jl")
using Test
using CUDA

@testset "GPU Manager Tests" begin
    # Initialize GPU Manager
    @testset "Initialization" begin
        manager = GPUManager.initialize()
        @test manager !== nothing
        @test manager.devices !== nothing
        @test length(manager.devices) >= 1
        @test manager.single_gpu_mode == (length(manager.devices) < 2)
    end
    
    # Test device operations
    @testset "Device Operations" begin
        # Get device count
        n_devices = GPUManager.device_count()
        @test n_devices >= 1
        
        # Test device switching
        for i in 0:n_devices-1
            dev = GPUManager.set_device!(i)
            @test dev.index == i
            @test GPUManager.current_device().index == i
        end
        
        # Test invalid device index
        @test_throws ErrorException GPUManager.get_device(-1)
        @test_throws ErrorException GPUManager.get_device(n_devices)
    end
    
    # Test workload distribution
    @testset "Workload Distribution" begin
        total_work = 1000
        distribution = GPUManager.distribute_workload(total_work)
        
        if GPUManager.is_single_gpu_mode()
            @test length(distribution) == 1
            @test distribution[1] == total_work
        else
            @test length(distribution) == GPUManager.device_count()
            @test sum(distribution) == total_work
        end
    end
    
    # Test memory allocation
    @testset "Memory Allocation" begin
        n_devices = GPUManager.device_count()
        
        for i in 0:n_devices-1
            # Allocate array on specific device
            arr = GPUManager.allocate_on_device(Float32, 100, 100; device=i)
            @test size(arr) == (100, 100)
            @test eltype(arr) == Float32
            @test arr isa CuArray
        end
    end
    
    # Test memory info
    @testset "Memory Info" begin
        mem_info = GPUManager.get_memory_info()
        @test mem_info !== nothing
        
        for i in 0:GPUManager.device_count()-1
            @test haskey(mem_info, i)
            @test haskey(mem_info[i], "total_gb")
            @test haskey(mem_info[i], "free_gb")
            @test haskey(mem_info[i], "used_gb")
            @test mem_info[i]["total_gb"] > 0
            @test mem_info[i]["free_gb"] > 0
        end
    end
    
    # Test synchronization
    @testset "Synchronization" begin
        # This should not error
        GPUManager.synchronize_all()
        @test true
    end
    
    # Test transfers (if multi-GPU)
    if !GPUManager.is_single_gpu_mode()
        @testset "GPU Transfers" begin
            # Create array on GPU 0
            GPUManager.set_device!(0)
            src_array = CUDA.rand(Float32, 1000)
            
            # Transfer to GPU 1
            dst_array = GPUManager.transfer_between_gpus(src_array, 1)
            @test size(dst_array) == size(src_array)
            
            # Verify data integrity
            GPUManager.set_device!(0)
            src_host = Array(src_array)
            GPUManager.set_device!(1)
            dst_host = Array(dst_array)
            @test src_host ≈ dst_host
        end
        
        @testset "PCIe Bandwidth" begin
            results = GPUManager.benchmark_pcie_bandwidth(10)  # Small size for test
            @test results !== nothing
            @test length(results) > 0
            
            # Check bandwidth is reasonable (> 1 GB/s)
            for (pair, bandwidth) in results
                @test bandwidth > 1.0
            end
        end
    end
end

# Print summary
println("\n" * "=" ^ 60)
println("GPU Manager Test Summary")
println("=" ^ 60)
println("Device Count: ", GPUManager.device_count())
println("Single GPU Mode: ", GPUManager.is_single_gpu_mode())

for i in 0:GPUManager.device_count()-1
    dev = GPUManager.get_device(i)
    println("\nGPU $i: $(dev.name)")
    println("  Compute Capability: $(dev.compute_capability[1]).$(dev.compute_capability[2])")
    println("  Memory: $(round(dev.total_memory/1024^3, digits=2)) GB")
end

mem_info = GPUManager.get_memory_info()
println("\nMemory Usage:")
for (gpu_id, info) in sort(collect(mem_info))
    println("  GPU $gpu_id: $(round(info["used_gb"], digits=2))/$(round(info["total_gb"], digits=2)) GB used")
end

println("=" ^ 60)