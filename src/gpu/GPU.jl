module GPU

# GPU Management and Configuration
include("gpu_management.jl")
include("device_manager.jl")
include("stream_manager.jl")
include("memory_manager.jl")
include("cuda_config.jl")

# MCTS GPU Implementation
include("mcts_gpu.jl")

# Re-export main functionality
using .GPUManagement
using .DeviceManager
using .StreamManager
using .MemoryManager
using .CUDAConfig
using .MCTSGPU

# GPU Management exports
export detect_gpus, select_device, get_device_info, initialize_gpu
export DeviceInfo, DeviceSelector, GPUSelector
export StreamPool, get_stream, return_stream, synchronize_streams
export set_memory_limit, get_memory_info, allocate_gpu_memory, free_gpu_memory
export CUDAConfiguration, validate_cuda_setup, get_cuda_config

# MCTS GPU exports
export MCTSGPUEngine, initialize!, start!, stop!, get_statistics
export select_features, get_best_features, reset_tree!

# Module initialization
function __init__()
    if CUDA.functional()
        # Initialize GPU subsystem
        devices = detect_gpus()
        if !isempty(devices)
            println("HSOF GPU Module initialized with $(length(devices)) GPU(s)")
            for (i, dev) in enumerate(devices)
                println("  GPU $i: $(dev.name) ($(dev.memory_gb)GB)")
            end
        else
            @warn "No CUDA-capable GPUs detected"
        end
    else
        @warn "CUDA.jl not functional - GPU features disabled"
    end
end

end # module GPU