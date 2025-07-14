# GPU Device Manager Module
# Main entry point for GPU detection and initialization

include("gpu_manager.jl")
include("memory_manager.jl")
include("stream_manager.jl")

module DeviceManager

using ..GPUManager
using ..MemoryManager
using ..StreamManager
using CUDA
using Logging

export initialize_devices, get_device_info, validate_gpu_environment

"""
    initialize_devices(config::Dict = Dict())

Initialize all GPU devices with the given configuration.
"""
function initialize_devices(config::Dict = Dict())
    @info "Initializing GPU devices..."
    
    # Initialize GPU manager
    gpu_manager = GPUManager.initialize()
    
    # Apply configuration
    if haskey(config, "gpu") && haskey(config["gpu"], "cuda")
        cuda_config = config["gpu"]["cuda"]
        
        # Set memory limits per device
        if haskey(cuda_config, "memory_limit_gb")
            limit_gb = cuda_config["memory_limit_gb"]
            for i in 0:GPUManager.device_count()-1
                GPUManager.set_device!(i)
                MemoryManager.set_memory_limit(Float64(limit_gb), i)
            end
        end
        
        # Initialize streams
        if haskey(cuda_config, "stream_count")
            stream_count = cuda_config["stream_count"]
            for i in 0:GPUManager.device_count()-1
                StreamManager.initialize_streams(i, stream_count)
            end
        end
    end
    
    # Initialize memory manager
    MemoryManager.initialize()
    
    @info "GPU devices initialized successfully"
    return gpu_manager
end

"""
    get_device_info()

Get comprehensive information about all GPU devices.
"""
function get_device_info()
    info = Dict{String, Any}()
    
    # Basic GPU info
    info["device_count"] = GPUManager.device_count()
    info["single_gpu_mode"] = GPUManager.is_single_gpu_mode()
    
    # Per-device information
    info["devices"] = []
    for i in 0:GPUManager.device_count()-1
        dev = GPUManager.get_device(i)
        dev_info = Dict(
            "index" => dev.index,
            "name" => dev.name,
            "compute_capability" => dev.compute_capability,
            "total_memory_gb" => round(dev.total_memory / 1024^3, digits=2),
            "uuid" => dev.uuid
        )
        
        # Add memory info
        GPUManager.set_device!(i)
        mem_info = MemoryManager.get_memory_stats(i)
        dev_info["memory"] = mem_info
        
        # Add stream info
        dev_info["streams"] = StreamManager.get_stream_count(i)
        
        push!(info["devices"], dev_info)
    end
    
    # Memory pool info
    info["memory_pools"] = MemoryManager.get_pool_info()
    
    return info
end

"""
    validate_gpu_environment(requirements::Dict = Dict())

Validate GPU environment against project requirements.
"""
function validate_gpu_environment(requirements::Dict = Dict())
    results = Dict{String, Bool}()
    issues = String[]
    
    # Default requirements
    default_reqs = Dict(
        "min_gpu_count" => 2,
        "min_compute_capability" => (8, 9),
        "min_memory_gb" => 20,
        "cuda_version_min" => v"11.8"
    )
    
    reqs = merge(default_reqs, requirements)
    
    # Check CUDA functionality
    results["cuda_functional"] = CUDA.functional()
    if !results["cuda_functional"]
        push!(issues, "CUDA is not functional")
        return results, issues
    end
    
    # Check GPU count
    gpu_count = GPUManager.device_count()
    results["gpu_count_ok"] = gpu_count >= reqs["min_gpu_count"]
    if !results["gpu_count_ok"]
        push!(issues, "Found $gpu_count GPU(s), need at least $(reqs["min_gpu_count"])")
    end
    
    # Check CUDA version
    cuda_version = CUDA.runtime_version()
    results["cuda_version_ok"] = cuda_version >= reqs["cuda_version_min"]
    if !results["cuda_version_ok"]
        push!(issues, "CUDA version $cuda_version < required $(reqs["cuda_version_min"])")
    end
    
    # Check each GPU
    for i in 0:gpu_count-1
        dev = GPUManager.get_device(i)
        
        # Compute capability
        cc_ok = dev.compute_capability[1] > reqs["min_compute_capability"][1] ||
                (dev.compute_capability[1] == reqs["min_compute_capability"][1] &&
                 dev.compute_capability[2] >= reqs["min_compute_capability"][2])
        results["gpu$(i)_compute_capability_ok"] = cc_ok
        if !cc_ok
            push!(issues, "GPU $i compute capability $(dev.compute_capability) < required $(reqs["min_compute_capability"])")
        end
        
        # Memory
        mem_gb = dev.total_memory / 1024^3
        mem_ok = mem_gb >= reqs["min_memory_gb"]
        results["gpu$(i)_memory_ok"] = mem_ok
        if !mem_ok
            push!(issues, "GPU $i memory $(round(mem_gb, digits=1))GB < required $(reqs["min_memory_gb"])GB")
        end
    end
    
    # Overall result
    results["all_checks_passed"] = all(values(results))
    
    return results, issues
end

"""
    cleanup()

Cleanup all GPU resources.
"""
function cleanup()
    @info "Cleaning up GPU resources..."
    
    # Cleanup streams
    StreamManager.cleanup_all()
    
    # Cleanup memory pools
    MemoryManager.cleanup()
    
    # Cleanup GPU manager
    GPUManager.cleanup()
    
    @info "GPU resources cleaned up"
end

end # module