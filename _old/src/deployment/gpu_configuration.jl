module GPUConfiguration

using CUDA
using JSON3
using Dates
using TOML

export GPUTopology, MemoryLimits, PerformanceTargets, DeploymentConfig
export load_config, save_config, validate_config, merge_configs
export detect_gpu_topology, calculate_memory_limits, optimize_performance_targets

"""
GPU topology configuration for multi-GPU setups
"""
struct GPUTopology
    # GPU identification
    num_gpus::Int
    gpu_devices::Vector{Int}  # CUDA device IDs
    gpu_names::Vector{String}
    
    # Connectivity
    peer_access_matrix::Matrix{Bool}  # peer_access_matrix[i,j] = can GPU i access GPU j
    pcie_bandwidth::Matrix{Float64}   # GB/s between GPUs
    nvlink_available::Bool
    
    # NUMA affinity
    cpu_affinity::Dict{Int, Vector{Int}}  # GPU ID -> CPU cores
    numa_nodes::Dict{Int, Int}  # GPU ID -> NUMA node
end

"""
Memory limits and allocation strategies
"""
struct MemoryLimits
    # Per-GPU memory limits
    total_memory::Dict{Int, Int}  # GPU ID -> bytes
    reserved_memory::Dict{Int, Int}  # OS/driver reserved
    available_memory::Dict{Int, Int}  # Usable memory
    
    # Allocation limits
    max_allocation_percent::Float64  # Max % of available memory to use
    dataset_memory_limit::Int  # Max memory for dataset storage
    tree_memory_limit::Int  # Max memory per MCTS tree
    buffer_memory_limit::Int  # Communication buffer limits
    
    # Safety margins
    oom_safety_margin::Float64  # Keep this fraction free
    fragmentation_reserve::Float64  # Reserve for fragmentation
end

"""
Performance targets and thresholds
"""
struct PerformanceTargets
    # Efficiency targets
    scaling_efficiency_target::Float64  # Target scaling efficiency (e.g., 0.85)
    min_acceptable_efficiency::Float64  # Minimum before rebalancing
    
    # Utilization targets
    gpu_utilization_target::Float64  # Target GPU utilization %
    memory_bandwidth_target::Float64  # Target bandwidth utilization %
    
    # Timing constraints
    max_sync_latency_ms::Float64  # Max synchronization latency
    max_transfer_time_ms::Float64  # Max PCIe transfer time
    kernel_timeout_ms::Float64  # Kernel execution timeout
    
    # Throughput targets
    min_trees_per_second::Float64  # Minimum tree evaluations/sec
    min_features_per_second::Float64  # Minimum feature evaluations/sec
end

"""
Complete deployment configuration
"""
struct DeploymentConfig
    # Core configuration
    topology::GPUTopology
    memory_limits::MemoryLimits
    performance_targets::PerformanceTargets
    
    # Runtime settings
    environment::Dict{String, String}
    cuda_settings::Dict{String, Any}
    
    # Metadata
    created_at::DateTime
    version::String
    hardware_id::String  # Unique hardware fingerprint
end

"""
Detect GPU topology automatically
"""
function detect_gpu_topology()::GPUTopology
    if !CUDA.functional()
        error("CUDA not functional - cannot detect GPU topology")
    end
    
    num_gpus = length(CUDA.devices())
    gpu_devices = collect(0:num_gpus-1)
    gpu_names = String[]
    
    # Get GPU names
    for i in gpu_devices
        device!(i)
        push!(gpu_names, CUDA.name(device()))
    end
    
    # Check peer access
    peer_access_matrix = zeros(Bool, num_gpus, num_gpus)
    for i in 1:num_gpus
        for j in 1:num_gpus
            if i != j
                try
                    device!(i-1)
                    peer_access_matrix[i,j] = CUDA.can_access_peer(device(), CuDevice(j-1))
                catch
                    peer_access_matrix[i,j] = false
                end
            else
                peer_access_matrix[i,i] = true
            end
        end
    end
    
    # Estimate PCIe bandwidth (default values, can be measured)
    pcie_bandwidth = zeros(Float64, num_gpus, num_gpus)
    for i in 1:num_gpus
        for j in 1:num_gpus
            if i == j
                pcie_bandwidth[i,j] = 0.0  # No transfer needed
            elseif peer_access_matrix[i,j]
                pcie_bandwidth[i,j] = 32.0  # PCIe 4.0 x16 theoretical
            else
                pcie_bandwidth[i,j] = 16.0  # CPU-mediated transfer
            end
        end
    end
    
    # Check for NVLink (simplified check)
    nvlink_available = any(peer_access_matrix[i,j] && i != j 
                          for i in 1:num_gpus, j in 1:num_gpus)
    
    # CPU affinity (placeholder - requires system-specific detection)
    cpu_affinity = Dict{Int, Vector{Int}}()
    numa_nodes = Dict{Int, Int}()
    
    for i in gpu_devices
        # Simple default: distribute CPUs evenly
        total_cpus = Sys.CPU_THREADS
        cpus_per_gpu = total_cpus ÷ num_gpus
        start_cpu = i * cpus_per_gpu
        cpu_affinity[i] = collect(start_cpu:start_cpu+cpus_per_gpu-1)
        numa_nodes[i] = i ÷ 2  # Assume 2 GPUs per NUMA node
    end
    
    return GPUTopology(
        num_gpus,
        gpu_devices,
        gpu_names,
        peer_access_matrix,
        pcie_bandwidth,
        nvlink_available,
        cpu_affinity,
        numa_nodes
    )
end

"""
Calculate memory limits based on GPU capabilities
"""
function calculate_memory_limits(topology::GPUTopology)::MemoryLimits
    total_memory = Dict{Int, Int}()
    reserved_memory = Dict{Int, Int}()
    available_memory = Dict{Int, Int}()
    
    for gpu_id in topology.gpu_devices
        device!(gpu_id)
        
        # Get total memory
        total = CUDA.totalmem(device())
        total_memory[gpu_id] = total
        
        # Estimate reserved memory (typically 5-10% for driver/OS)
        reserved = Int(ceil(total * 0.08))
        reserved_memory[gpu_id] = reserved
        
        # Calculate available
        available_memory[gpu_id] = total - reserved
    end
    
    # Conservative defaults for production
    max_allocation_percent = 0.85  # Use max 85% of available memory
    
    # Calculate limits based on smallest GPU
    min_available = minimum(values(available_memory))
    
    # Dataset can use up to 40% of available memory
    dataset_memory_limit = Int(floor(min_available * 0.4))
    
    # Each tree gets equal share of 40% of memory (for 50 trees per GPU)
    tree_memory_limit = Int(floor(min_available * 0.4 / 50))
    
    # Buffers use 10% of available memory
    buffer_memory_limit = Int(floor(min_available * 0.1))
    
    # Safety margins
    oom_safety_margin = 0.1  # Keep 10% free
    fragmentation_reserve = 0.05  # 5% for fragmentation
    
    return MemoryLimits(
        total_memory,
        reserved_memory,
        available_memory,
        max_allocation_percent,
        dataset_memory_limit,
        tree_memory_limit,
        buffer_memory_limit,
        oom_safety_margin,
        fragmentation_reserve
    )
end

"""
Optimize performance targets based on hardware
"""
function optimize_performance_targets(topology::GPUTopology)::PerformanceTargets
    # Base targets
    scaling_efficiency_target = 0.85
    min_acceptable_efficiency = 0.75
    
    # Adjust based on peer access
    has_good_connectivity = topology.nvlink_available || 
                           all(topology.peer_access_matrix[i,j] || i == j 
                               for i in 1:topology.num_gpus, j in 1:topology.num_gpus)
    
    if has_good_connectivity
        scaling_efficiency_target = 0.90  # Better with NVLink
    end
    
    # GPU utilization targets
    gpu_utilization_target = 0.85  # Target 85% utilization
    memory_bandwidth_target = 0.70  # Target 70% bandwidth utilization
    
    # Timing constraints based on connectivity
    if topology.nvlink_available
        max_sync_latency_ms = 5.0
        max_transfer_time_ms = 10.0
    elseif has_good_connectivity
        max_sync_latency_ms = 10.0
        max_transfer_time_ms = 20.0
    else
        max_sync_latency_ms = 20.0
        max_transfer_time_ms = 50.0
    end
    
    kernel_timeout_ms = 5000.0  # 5 second timeout
    
    # Throughput targets (conservative estimates)
    # Assume each tree can process 100 iterations/second
    min_trees_per_second = topology.num_gpus * 50 * 100  # trees * iterations
    
    # Assume 1000 features evaluated per tree per second
    min_features_per_second = min_trees_per_second * 1000
    
    return PerformanceTargets(
        scaling_efficiency_target,
        min_acceptable_efficiency,
        gpu_utilization_target,
        memory_bandwidth_target,
        max_sync_latency_ms,
        max_transfer_time_ms,
        kernel_timeout_ms,
        min_trees_per_second,
        min_features_per_second
    )
end

"""
Load configuration from file
"""
function load_config(filename::String)::DeploymentConfig
    if endswith(filename, ".toml")
        config_dict = TOML.parsefile(filename)
    elseif endswith(filename, ".json")
        config_dict = JSON3.read(read(filename, String))
    else
        error("Unsupported config format. Use .toml or .json")
    end
    
    # Parse topology
    topo_dict = config_dict["topology"]
    topology = GPUTopology(
        topo_dict["num_gpus"],
        Vector{Int}(topo_dict["gpu_devices"]),
        Vector{String}(topo_dict["gpu_names"]),
        Matrix{Bool}(hcat(topo_dict["peer_access_matrix"]...)),
        Matrix{Float64}(hcat(topo_dict["pcie_bandwidth"]...)),
        topo_dict["nvlink_available"],
        Dict(parse(Int, k) => v for (k,v) in topo_dict["cpu_affinity"]),
        Dict(parse(Int, k) => v for (k,v) in topo_dict["numa_nodes"])
    )
    
    # Parse memory limits
    mem_dict = config_dict["memory_limits"]
    memory_limits = MemoryLimits(
        Dict(parse(Int, k) => v for (k,v) in mem_dict["total_memory"]),
        Dict(parse(Int, k) => v for (k,v) in mem_dict["reserved_memory"]),
        Dict(parse(Int, k) => v for (k,v) in mem_dict["available_memory"]),
        mem_dict["max_allocation_percent"],
        mem_dict["dataset_memory_limit"],
        mem_dict["tree_memory_limit"],
        mem_dict["buffer_memory_limit"],
        mem_dict["oom_safety_margin"],
        mem_dict["fragmentation_reserve"]
    )
    
    # Parse performance targets
    perf_dict = config_dict["performance_targets"]
    performance_targets = PerformanceTargets(
        perf_dict["scaling_efficiency_target"],
        perf_dict["min_acceptable_efficiency"],
        perf_dict["gpu_utilization_target"],
        perf_dict["memory_bandwidth_target"],
        perf_dict["max_sync_latency_ms"],
        perf_dict["max_transfer_time_ms"],
        perf_dict["kernel_timeout_ms"],
        perf_dict["min_trees_per_second"],
        perf_dict["min_features_per_second"]
    )
    
    # Parse runtime settings
    environment = Dict{String, String}(config_dict["environment"])
    cuda_settings = Dict{String, Any}(config_dict["cuda_settings"])
    
    # Parse metadata
    created_at = DateTime(config_dict["metadata"]["created_at"])
    version = config_dict["metadata"]["version"]
    hardware_id = config_dict["metadata"]["hardware_id"]
    
    return DeploymentConfig(
        topology,
        memory_limits,
        performance_targets,
        environment,
        cuda_settings,
        created_at,
        version,
        hardware_id
    )
end

"""
Save configuration to file
"""
function save_config(config::DeploymentConfig, filename::String)
    # Convert to dictionary
    config_dict = Dict{String, Any}()
    
    # Topology
    config_dict["topology"] = Dict(
        "num_gpus" => config.topology.num_gpus,
        "gpu_devices" => config.topology.gpu_devices,
        "gpu_names" => config.topology.gpu_names,
        "peer_access_matrix" => [config.topology.peer_access_matrix[i,:] 
                                 for i in 1:config.topology.num_gpus],
        "pcie_bandwidth" => [config.topology.pcie_bandwidth[i,:] 
                            for i in 1:config.topology.num_gpus],
        "nvlink_available" => config.topology.nvlink_available,
        "cpu_affinity" => Dict(string(k) => v for (k,v) in config.topology.cpu_affinity),
        "numa_nodes" => Dict(string(k) => v for (k,v) in config.topology.numa_nodes)
    )
    
    # Memory limits
    config_dict["memory_limits"] = Dict(
        "total_memory" => Dict(string(k) => v for (k,v) in config.memory_limits.total_memory),
        "reserved_memory" => Dict(string(k) => v for (k,v) in config.memory_limits.reserved_memory),
        "available_memory" => Dict(string(k) => v for (k,v) in config.memory_limits.available_memory),
        "max_allocation_percent" => config.memory_limits.max_allocation_percent,
        "dataset_memory_limit" => config.memory_limits.dataset_memory_limit,
        "tree_memory_limit" => config.memory_limits.tree_memory_limit,
        "buffer_memory_limit" => config.memory_limits.buffer_memory_limit,
        "oom_safety_margin" => config.memory_limits.oom_safety_margin,
        "fragmentation_reserve" => config.memory_limits.fragmentation_reserve
    )
    
    # Performance targets
    config_dict["performance_targets"] = Dict(
        "scaling_efficiency_target" => config.performance_targets.scaling_efficiency_target,
        "min_acceptable_efficiency" => config.performance_targets.min_acceptable_efficiency,
        "gpu_utilization_target" => config.performance_targets.gpu_utilization_target,
        "memory_bandwidth_target" => config.performance_targets.memory_bandwidth_target,
        "max_sync_latency_ms" => config.performance_targets.max_sync_latency_ms,
        "max_transfer_time_ms" => config.performance_targets.max_transfer_time_ms,
        "kernel_timeout_ms" => config.performance_targets.kernel_timeout_ms,
        "min_trees_per_second" => config.performance_targets.min_trees_per_second,
        "min_features_per_second" => config.performance_targets.min_features_per_second
    )
    
    # Runtime settings
    config_dict["environment"] = config.environment
    config_dict["cuda_settings"] = config.cuda_settings
    
    # Metadata
    config_dict["metadata"] = Dict(
        "created_at" => string(config.created_at),
        "version" => config.version,
        "hardware_id" => config.hardware_id
    )
    
    # Save based on extension
    if endswith(filename, ".toml")
        open(filename, "w") do io
            TOML.print(io, config_dict)
        end
    elseif endswith(filename, ".json")
        open(filename, "w") do io
            JSON3.pretty(io, config_dict)
        end
    else
        error("Unsupported config format. Use .toml or .json")
    end
end

"""
Validate configuration for consistency and feasibility
"""
function validate_config(config::DeploymentConfig)::Vector{String}
    issues = String[]
    
    # Validate topology
    if config.topology.num_gpus < 1
        push!(issues, "Invalid number of GPUs: $(config.topology.num_gpus)")
    end
    
    if length(config.topology.gpu_devices) != config.topology.num_gpus
        push!(issues, "GPU device count mismatch")
    end
    
    # Validate memory limits
    for (gpu_id, available) in config.memory_limits.available_memory
        total_allocated = config.memory_limits.dataset_memory_limit + 
                         50 * config.memory_limits.tree_memory_limit +
                         config.memory_limits.buffer_memory_limit
        
        max_allowed = available * config.memory_limits.max_allocation_percent
        
        if total_allocated > max_allowed
            push!(issues, "Memory overcommit on GPU $gpu_id: $total_allocated > $max_allowed")
        end
    end
    
    # Validate performance targets
    if config.performance_targets.scaling_efficiency_target > 1.0
        push!(issues, "Invalid scaling efficiency target: $(config.performance_targets.scaling_efficiency_target)")
    end
    
    if config.performance_targets.min_acceptable_efficiency > config.performance_targets.scaling_efficiency_target
        push!(issues, "Minimum efficiency exceeds target")
    end
    
    return issues
end

"""
Merge configurations with override precedence
"""
function merge_configs(base::DeploymentConfig, override::DeploymentConfig)::DeploymentConfig
    # For now, override completely replaces base
    # In future, could implement field-by-field merging
    return override
end

"""
Generate unique hardware fingerprint
"""
function generate_hardware_id(topology::GPUTopology)::String
    # Create a unique ID based on GPU configuration
    gpu_info = join(topology.gpu_names, "_")
    gpu_count = topology.num_gpus
    nvlink = topology.nvlink_available ? "nvlink" : "pcie"
    
    return "$(gpu_count)x_$(gpu_info)_$(nvlink)"
end

"""
Create default production configuration
"""
function create_default_config()::DeploymentConfig
    # Detect hardware
    topology = detect_gpu_topology()
    memory_limits = calculate_memory_limits(topology)
    performance_targets = optimize_performance_targets(topology)
    
    # Default environment
    environment = Dict{String, String}(
        "CUDA_VISIBLE_DEVICES" => join(topology.gpu_devices, ","),
        "JULIA_NUM_THREADS" => string(Sys.CPU_THREADS),
        "CUDA_CACHE_DISABLE" => "0",
        "CUDA_LAUNCH_BLOCKING" => "0"
    )
    
    # CUDA settings
    cuda_settings = Dict{String, Any}(
        "memory_pool" => true,
        "memory_pool_size" => 0,  # 0 = automatic
        "device_synchronize" => false,
        "stream_ordered_allocator" => true
    )
    
    # Create config
    return DeploymentConfig(
        topology,
        memory_limits,
        performance_targets,
        environment,
        cuda_settings,
        now(),
        "1.0.0",
        generate_hardware_id(topology)
    )
end

end # module