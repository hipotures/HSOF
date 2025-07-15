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

# Work Distribution for Multi-GPU
include("work_distribution.jl")
using .WorkDistribution

# GPU Management exports
export initialize_devices, get_device_info, validate_gpu_environment
export GPUManager, DeviceManager, StreamManager, MemoryManager

# MCTS GPU exports
export MCTSGPUEngine, initialize!, start!, stop!, get_statistics
export select_features, get_best_features, reset_tree!

# Work Distribution exports
export WorkDistributor, GPUWorkAssignment, WorkloadMetrics
export create_work_distributor, assign_tree_work, assign_metamodel_work
export get_gpu_for_tree, get_tree_range, get_load_balance_ratio
export update_metrics!, rebalance_if_needed!, get_work_summary
export set_gpu_affinity, execute_on_gpu

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