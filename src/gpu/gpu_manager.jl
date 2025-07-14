# GPU Manager Module for Dual GPU Setup
module GPUManager

using CUDA
using Logging
using Statistics

# GPU device information structure
struct GPUDevice
    index::Int
    cuda_device::CUDA.CuDevice
    name::String
    compute_capability::Tuple{Int,Int}
    total_memory::Int
    uuid::String
end

# GPU manager state
mutable struct GPUManagerState
    devices::Vector{GPUDevice}
    current_device::Int
    peer_access_matrix::Matrix{Bool}
    workload_distribution::Vector{Float64}
    single_gpu_mode::Bool
end

# Global manager instance
const MANAGER = Ref{GPUManagerState}()

"""
    initialize()

Initialize GPU manager and detect available devices.
"""
function initialize()
    @info "Initializing GPU Manager..."
    
    if !CUDA.functional()
        error("CUDA is not functional. Cannot initialize GPU Manager.")
    end
    
    # Detect devices
    cuda_devices = CUDA.devices()
    n_devices = length(cuda_devices)
    
    if n_devices == 0
        error("No CUDA devices found!")
    end
    
    devices = GPUDevice[]
    
    for (i, dev) in enumerate(cuda_devices)
        CUDA.device!(dev)
        cc = CUDA.capability(dev)
        
        push!(devices, GPUDevice(
            i - 1,  # 0-based index
            dev,
            CUDA.name(dev),
            (cc.major, cc.minor),
            CUDA.totalmem(dev),
            string(CUDA.uuid(dev))
        ))
    end
    
    # Initialize peer access matrix
    peer_access_matrix = zeros(Bool, n_devices, n_devices)
    
    for i in 1:n_devices
        for j in 1:n_devices
            if i != j
                CUDA.device!(devices[i].cuda_device)
                peer_access_matrix[i,j] = CUDA.can_access_peer(devices[j].cuda_device)
            else
                peer_access_matrix[i,j] = true
            end
        end
    end
    
    # Enable peer access where available
    for i in 1:n_devices
        for j in 1:n_devices
            if i != j && peer_access_matrix[i,j]
                try
                    CUDA.device!(devices[i].cuda_device)
                    CUDA.enable_peer_access(devices[j].cuda_device)
                    @info "Enabled peer access: GPU $(i-1) → GPU $(j-1)"
                catch e
                    @warn "Failed to enable peer access: GPU $(i-1) → GPU $(j-1): $e"
                end
            end
        end
    end
    
    # Initialize workload distribution
    workload_distribution = ones(n_devices) / n_devices
    
    # Determine if we're in single GPU mode
    single_gpu_mode = n_devices < 2
    
    if single_gpu_mode
        @warn "Running in single GPU mode. Only $(n_devices) GPU(s) detected."
        @info "Project is optimized for 2 GPUs but will work with reduced performance."
    else
        @info "Multi-GPU mode enabled with $(n_devices) GPUs"
    end
    
    # Create manager state
    MANAGER[] = GPUManagerState(
        devices,
        0,  # Current device
        peer_access_matrix,
        workload_distribution,
        single_gpu_mode
    )
    
    # Log device information
    for dev in devices
        @info "GPU $(dev.index): $(dev.name) (CC $(dev.compute_capability[1]).$(dev.compute_capability[2]), $(round(dev.total_memory/1024^3, digits=2)) GB)"
    end
    
    return MANAGER[]
end

"""
    get_device(index::Int)

Get GPU device by index.
"""
function get_device(index::Int)
    if !isassigned(MANAGER)
        error("GPU Manager not initialized. Call GPUManager.initialize() first.")
    end
    
    manager = MANAGER[]
    if index < 0 || index >= length(manager.devices)
        error("Invalid GPU index: $index. Available: 0-$(length(manager.devices)-1)")
    end
    
    return manager.devices[index + 1]
end

"""
    set_device!(index::Int)

Set the current CUDA device.
"""
function set_device!(index::Int)
    dev = get_device(index)
    CUDA.device!(dev.cuda_device)
    MANAGER[].current_device = index
    return dev
end

"""
    current_device()

Get the currently active GPU device.
"""
function current_device()
    if !isassigned(MANAGER)
        error("GPU Manager not initialized.")
    end
    
    return MANAGER[].devices[MANAGER[].current_device + 1]
end

"""
    device_count()

Get the number of available GPU devices.
"""
function device_count()
    if !isassigned(MANAGER)
        return 0
    end
    
    return length(MANAGER[].devices)
end

"""
    is_single_gpu_mode()

Check if running in single GPU mode.
"""
function is_single_gpu_mode()
    if !isassigned(MANAGER)
        return true
    end
    
    return MANAGER[].single_gpu_mode
end

"""
    distribute_workload(total_work::Int)

Distribute workload across available GPUs.
Returns array of work items per GPU.
"""
function distribute_workload(total_work::Int)
    if !isassigned(MANAGER)
        error("GPU Manager not initialized.")
    end
    
    manager = MANAGER[]
    n_gpus = length(manager.devices)
    
    if manager.single_gpu_mode
        return [total_work]
    end
    
    # Distribute based on workload distribution weights
    work_per_gpu = zeros(Int, n_gpus)
    remaining = total_work
    
    for i in 1:n_gpus-1
        work_per_gpu[i] = round(Int, total_work * manager.workload_distribution[i])
        remaining -= work_per_gpu[i]
    end
    work_per_gpu[end] = remaining
    
    return work_per_gpu
end

"""
    synchronize_all()

Synchronize all GPU devices.
"""
function synchronize_all()
    if !isassigned(MANAGER)
        return
    end
    
    current = MANAGER[].current_device
    
    for dev in MANAGER[].devices
        CUDA.device!(dev.cuda_device)
        CUDA.synchronize()
    end
    
    # Restore current device
    set_device!(current)
end

"""
    allocate_on_device(T::Type, dims...; device::Int)

Allocate array on specific GPU device.
"""
function allocate_on_device(T::Type, dims...; device::Int)
    prev_device = MANAGER[].current_device
    set_device!(device)
    
    arr = CUDA.zeros(T, dims...)
    
    # Restore previous device
    set_device!(prev_device)
    
    return arr
end

"""
    transfer_between_gpus(src_array::CuArray, dst_device::Int)

Transfer array from current GPU to destination GPU.
"""
function transfer_between_gpus(src_array::CuArray, dst_device::Int)
    if !isassigned(MANAGER)
        error("GPU Manager not initialized.")
    end
    
    manager = MANAGER[]
    src_device = MANAGER[].current_device
    
    # If same device, just return the array
    if src_device == dst_device
        return src_array
    end
    
    # Check if peer access is available
    if manager.peer_access_matrix[src_device + 1, dst_device + 1]
        # Direct peer-to-peer copy
        set_device!(dst_device)
        dst_array = similar(src_array)
        copyto!(dst_array, src_array)
        return dst_array
    else
        # Copy through host
        @warn "No peer access between GPU $src_device and GPU $dst_device. Using host memory."
        host_array = Array(src_array)
        set_device!(dst_device)
        return CuArray(host_array)
    end
end

"""
    benchmark_pcie_bandwidth(size_mb::Int = 100)

Benchmark PCIe bandwidth between GPUs.
"""
function benchmark_pcie_bandwidth(size_mb::Int = 100)
    if !isassigned(MANAGER) || is_single_gpu_mode()
        @warn "Cannot benchmark PCIe bandwidth in single GPU mode."
        return nothing
    end
    
    n_elements = size_mb * 1024 * 1024 ÷ sizeof(Float32)
    results = Dict{String, Float64}()
    
    # Test all GPU pairs
    for i in 0:device_count()-1
        for j in 0:device_count()-1
            if i != j
                # Allocate on source GPU
                set_device!(i)
                src_array = CUDA.rand(Float32, n_elements)
                
                # Warm up
                dst_array = transfer_between_gpus(src_array, j)
                synchronize_all()
                
                # Measure transfer time
                start_time = time()
                for _ in 1:10
                    dst_array = transfer_between_gpus(src_array, j)
                    synchronize_all()
                end
                elapsed = time() - start_time
                
                bandwidth_gb_s = (10 * n_elements * sizeof(Float32) / 1e9) / elapsed
                results["GPU$i→GPU$j"] = bandwidth_gb_s
                
                @info "PCIe bandwidth GPU$i → GPU$j: $(round(bandwidth_gb_s, digits=2)) GB/s"
            end
        end
    end
    
    return results
end

"""
    get_memory_info()

Get memory information for all GPUs.
"""
function get_memory_info()
    if !isassigned(MANAGER)
        return nothing
    end
    
    info = Dict{Int, Dict{String, Float64}}()
    current = MANAGER[].current_device
    
    for dev in MANAGER[].devices
        CUDA.device!(dev.cuda_device)
        info[dev.index] = Dict(
            "total_gb" => dev.total_memory / 1024^3,
            "free_gb" => CUDA.available_memory() / 1024^3,
            "used_gb" => (dev.total_memory - CUDA.available_memory()) / 1024^3
        )
    end
    
    # Restore current device
    set_device!(current)
    
    return info
end

"""
    cleanup()

Cleanup GPU manager resources.
"""
function cleanup()
    if isassigned(MANAGER)
        synchronize_all()
        # Reset the global manager reference
        global MANAGER
        MANAGER = Ref{GPUManagerState}()
    end
end

end # module