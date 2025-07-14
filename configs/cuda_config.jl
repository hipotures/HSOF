# CUDA Configuration Module
module CUDAConfig

using CUDA
using TOML

# Default CUDA configuration
const DEFAULT_CONFIG = Dict(
    "cuda" => Dict(
        "memory_pool_growth" => 2^30,  # 1GB increments
        "math_mode" => "FAST_MATH",
        "device_preferences" => [0, 1],  # Prefer GPU 0, then GPU 1
        "memory_limit_gb" => 22,  # Leave 2GB for system on 24GB cards
        "stream_count" => 4,  # Number of CUDA streams per device
        "allow_memory_growth" => true,
        "enable_profiling" => false,
        "compute_capability_min" => [8, 9]  # RTX 4090 is 8.9
    ),
    "kernel_launch" => Dict(
        "default_block_size" => 256,
        "max_threads_per_block" => 1024,
        "default_grid_size" => 0,  # 0 means auto-calculate
        "occupancy_target" => 0.8
    ),
    "memory" => Dict(
        "enable_unified_memory" => false,
        "async_malloc_threshold" => 2^20,  # 1MB
        "enable_peer_access" => true,
        "pool_type" => "binned"  # or "none", "cuda"
    )
)

"""
    load_config(config_path::String = "configs/gpu_config.toml")

Load CUDA configuration from TOML file, falling back to defaults.
"""
function load_config(config_path::String = "configs/gpu_config.toml")
    config = deepcopy(DEFAULT_CONFIG)
    
    if isfile(config_path)
        try
            user_config = TOML.parsefile(config_path)
            merge!(config, user_config)
            @info "Loaded CUDA configuration from $config_path"
        catch e
            @warn "Failed to load config from $config_path: $e"
            @info "Using default configuration"
        end
    else
        @info "No config file found at $config_path, using defaults"
    end
    
    return config
end

"""
    apply_config!(config::Dict)

Apply CUDA configuration settings.
"""
function apply_config!(config::Dict)
    cuda_cfg = config["cuda"]
    
    # Set memory pool configuration
    if cuda_cfg["allow_memory_growth"]
        # Memory pool configuration is automatic in newer CUDA.jl versions
        @info "Memory growth is enabled (automatic in CUDA.jl 5.x)"
    end
    
    # Set math mode
    if cuda_cfg["math_mode"] == "FAST_MATH"
        CUDA.math_mode!(CUDA.FAST_MATH)
    elseif cuda_cfg["math_mode"] == "DEFAULT"
        CUDA.math_mode!(CUDA.DEFAULT_MATH)
    end
    
    # Enable peer access if available
    mem_cfg = config["memory"]
    if get(mem_cfg, "enable_peer_access", true) && CUDA.functional() && length(CUDA.devices()) > 1
        for i in 0:length(CUDA.devices())-1
            for j in 0:length(CUDA.devices())-1
                if i != j
                    try
                        CUDA.device!(i)
                        if CUDA.can_access_peer(CUDA.CuDevice(j))
                            CUDA.enable_peer_access(CUDA.CuDevice(j))
                            @info "Enabled peer access from GPU $i to GPU $j"
                        end
                    catch e
                        @warn "Could not enable peer access from GPU $i to GPU $j: $e"
                    end
                end
            end
        end
    end
    
    @info "Applied CUDA configuration"
end

"""
    validate_cuda_environment()

Validate CUDA installation and GPU capabilities.
"""
function validate_cuda_environment()
    results = Dict{String, Any}()
    
    # Check CUDA functionality
    results["cuda_functional"] = CUDA.functional()
    
    if !results["cuda_functional"]
        @error "CUDA is not functional!"
        return results
    end
    
    # Get CUDA version info
    results["cuda_version"] = string(CUDA.runtime_version())
    results["driver_version"] = string(CUDA.driver_version())
    
    # Check GPUs
    devices = CUDA.devices()
    results["gpu_count"] = length(devices)
    results["gpus"] = []
    
    for (i, dev) in enumerate(devices)
        CUDA.device!(dev)
        gpu_info = Dict{String, Any}()
        
        gpu_info["index"] = i - 1
        gpu_info["name"] = CUDA.name(dev)
        gpu_info["uuid"] = CUDA.uuid(dev)
        
        # Compute capability
        cc = CUDA.capability(dev)
        gpu_info["compute_capability"] = (cc.major, cc.minor)
        gpu_info["compute_capability_check"] = cc.major >= 8 && cc.minor >= 9
        
        # Memory info
        gpu_info["total_memory_gb"] = CUDA.totalmem(dev) / 1024^3
        gpu_info["free_memory_gb"] = CUDA.available_memory() / 1024^3
        gpu_info["memory_check"] = gpu_info["total_memory_gb"] >= 20
        
        # Additional properties
        gpu_info["warp_size"] = CUDA.warpsize(dev)
        gpu_info["multiprocessor_count"] = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        gpu_info["max_threads_per_block"] = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
        gpu_info["max_block_dim"] = (
            CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
            CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
            CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)
        )
        gpu_info["max_grid_dim"] = (
            CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X),
            CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y),
            CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
        )
        
        push!(results["gpus"], gpu_info)
    end
    
    # Test basic CUDA functionality
    try
        CUDA.device!(0)
        test_arr = CUDA.ones(Float32, 1000, 1000)
        test_result = sum(test_arr)
        results["basic_cuda_test"] = test_result == 1_000_000.0f0
    catch e
        results["basic_cuda_test"] = false
        results["cuda_test_error"] = string(e)
    end
    
    return results
end

"""
    test_cuda_kernel()

Test basic CUDA kernel compilation and execution.
"""
function test_cuda_kernel()
    # Simple vector addition kernel
    function vadd_kernel(a, b, c)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        if i <= length(c)
            c[i] = a[i] + b[i]
        end
        return
    end
    
    # Test data
    n = 1_000_000
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)
    
    # Launch kernel
    threads = 256
    blocks = cld(n, threads)
    
    @cuda threads=threads blocks=blocks vadd_kernel(a, b, c)
    CUDA.synchronize()
    
    # Verify result
    c_host = Array(c)
    a_host = Array(a)
    b_host = Array(b)
    
    is_correct = all(abs.(c_host .- (a_host .+ b_host)) .< 1e-5)
    
    return is_correct
end

"""
    save_default_config(path::String = "configs/gpu_config.toml")

Save the default configuration to a TOML file.
"""
function save_default_config(path::String = "configs/gpu_config.toml")
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, DEFAULT_CONFIG)
    end
    @info "Saved default CUDA configuration to $path"
end

end # module