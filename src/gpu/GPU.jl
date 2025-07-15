module GPU

using CUDA

# GPU Management and Configuration
include("gpu_manager.jl")
include("memory_manager.jl")
include("stream_manager.jl")

# Re-export main functionality from included modules
using .GPUManager
using .MemoryManager
using .StreamManager

# Now include device_manager which depends on the above modules
include("device_manager.jl")
using .DeviceManager

# MCTS GPU Implementation
include("mcts_gpu.jl")
using .MCTSGPU

# GPU Management exports
export initialize_devices, get_device_info, validate_gpu_environment
export GPUManager, DeviceManager, StreamManager, MemoryManager

# MCTS GPU exports
export MCTSGPUEngine, initialize!, start!, stop!, get_statistics
export select_features, get_best_features, reset_tree!

# Module initialization
function __init__()
    if CUDA.functional()
        # Initialize GPU subsystem
        try
            gpu_manager = GPUManager.initialize()
            device_count = GPUManager.device_count()
            println("HSOF GPU Module initialized with $(device_count) GPU(s)")
            for i in 0:device_count-1
                dev = GPUManager.get_device(i)
                println("  GPU $i: $(dev.name) ($(round(dev.total_memory/1024^3, digits=2))GB)")
            end
        catch e
            @warn "Failed to initialize GPU manager: $e"
        end
    else
        @warn "CUDA.jl not functional - GPU features disabled"
    end
end

end # module GPU