module DatasetStorage

using CUDA
using DataFrames
using Dates
using Printf
using Statistics

export DatasetReplica, DatasetManager, DatasetVersion, MemoryStats
export create_dataset_manager, replicate_dataset!, get_dataset_replica
export update_dataset!, get_memory_usage, has_sufficient_memory
export load_dataset_to_gpu!, clear_dataset!, sync_datasets!
export get_feature_column, get_sample_batch, get_dataset_info

# Dataset version information
struct DatasetVersion
    version_id::Int
    timestamp::DateTime
    n_features::Int
    n_samples::Int
    checksum::UInt64
    
    function DatasetVersion(version_id::Int, n_features::Int, n_samples::Int)
        new(version_id, now(), n_features, n_samples, hash((n_features, n_samples)))
    end
end

"""
GPU-optimized dataset replica with column-wise storage
"""
mutable struct DatasetReplica
    gpu_id::Int
    version::DatasetVersion
    
    # Column-wise storage for efficient GPU access
    feature_data::Union{Nothing, CuMatrix{Float32}}  # [n_samples × n_features]
    feature_names::Vector{String}
    feature_indices::Dict{String, Int}  # name -> column index
    
    # Sample metadata
    sample_ids::Union{Nothing, CuVector{Int32}}
    n_samples::Int
    n_features::Int
    
    # Memory management
    allocated_memory::Int64  # in bytes
    last_access::DateTime
    is_loaded::Bool
    
    function DatasetReplica(gpu_id::Int, version::DatasetVersion)
        new(
            gpu_id,
            version,
            nothing,  # feature_data
            String[],
            Dict{String, Int}(),
            nothing,  # sample_ids
            version.n_samples,
            version.n_features,
            0,
            now(),
            false
        )
    end
end

"""
Memory statistics for GPU dataset storage
"""
struct MemoryStats
    gpu_id::Int
    total_memory::Int64
    free_memory::Int64
    dataset_memory::Int64
    utilization::Float64
end

"""
Main dataset manager for multi-GPU storage
"""
mutable struct DatasetManager
    replicas::Dict{Int, DatasetReplica}  # gpu_id -> replica
    current_version::DatasetVersion
    memory_limit_per_gpu::Int64  # in bytes
    enable_compression::Bool
    sync_interval::Int  # iterations between syncs
    last_sync::DateTime
    
    function DatasetManager(;
        memory_limit_per_gpu::Int64 = 8 * 1024^3,  # 8GB default
        enable_compression::Bool = false,
        sync_interval::Int = 10000
    )
        new(
            Dict{Int, DatasetReplica}(),
            DatasetVersion(1, 0, 0),
            memory_limit_per_gpu,
            enable_compression,
            sync_interval,
            now()
        )
    end
end

"""
Create a dataset manager for multi-GPU storage
"""
function create_dataset_manager(;
    num_gpus::Int = -1,
    memory_limit_per_gpu::Int64 = 8 * 1024^3,
    enable_compression::Bool = false
)
    # Auto-detect GPUs if not specified
    if num_gpus == -1
        num_gpus = CUDA.functional() ? length(CUDA.devices()) : 0
    end
    
    manager = DatasetManager(
        memory_limit_per_gpu = memory_limit_per_gpu,
        enable_compression = enable_compression
    )
    
    # Initialize replicas for each GPU
    for gpu_id in 0:(num_gpus-1)
        manager.replicas[gpu_id] = DatasetReplica(gpu_id, manager.current_version)
    end
    
    return manager
end

"""
Get memory usage statistics for a GPU
"""
function get_memory_usage(gpu_id::Int)
    if !CUDA.functional()
        return MemoryStats(gpu_id, 0, 0, 0, 0.0)
    end
    
    # Save current device
    prev_device = CUDA.device()
    
    try
        # Switch to target GPU
        CUDA.device!(gpu_id)
        
        # Get memory info
        total_mem = CUDA.total_memory()
        free_mem = CUDA.available_memory()
        used_mem = total_mem - free_mem
        utilization = used_mem / total_mem
        
        return MemoryStats(
            gpu_id,
            total_mem,
            free_mem,
            used_mem,
            utilization
        )
    finally
        # Restore device
        CUDA.device!(prev_device)
    end
end

"""
Check if GPU has sufficient memory for dataset
"""
function has_sufficient_memory(
    manager::DatasetManager,
    gpu_id::Int,
    n_features::Int,
    n_samples::Int
)
    # Calculate required memory
    # Float32 matrix + Int32 sample IDs + overhead
    element_size = sizeof(Float32)
    matrix_size = n_features * n_samples * element_size
    ids_size = n_samples * sizeof(Int32)
    overhead = 1.2  # 20% overhead for allocations
    
    required_memory = Int64(ceil((matrix_size + ids_size) * overhead))
    
    # Check against limit
    if required_memory > manager.memory_limit_per_gpu
        return false
    end
    
    # Check actual GPU memory
    if CUDA.functional()
        stats = get_memory_usage(gpu_id)
        return stats.free_memory >= required_memory
    end
    
    return true
end

"""
Load dataset to GPU memory
"""
function load_dataset_to_gpu!(
    replica::DatasetReplica,
    data::AbstractMatrix{Float32},
    feature_names::Vector{String};
    sample_ids::Union{Nothing, Vector{Int32}} = nothing
)
    if !CUDA.functional()
        @warn "CUDA not functional - storing reference only"
        replica.feature_names = feature_names
        replica.n_features = length(feature_names)
        replica.n_samples = size(data, 1)
        replica.is_loaded = false
        return replica
    end
    
    # Save current device
    prev_device = CUDA.device()
    
    try
        # Switch to replica's GPU
        CUDA.device!(replica.gpu_id)
        
        # Clear existing data
        if replica.feature_data !== nothing
            CUDA.unsafe_free!(replica.feature_data)
            replica.feature_data = nothing
        end
        if replica.sample_ids !== nothing
            CUDA.unsafe_free!(replica.sample_ids)
            replica.sample_ids = nothing
        end
        
        # Ensure data is in correct format (samples × features)
        if size(data, 2) != length(feature_names)
            error("Data dimensions don't match feature names")
        end
        
        # Allocate and copy data
        replica.feature_data = CuMatrix{Float32}(data)
        replica.n_samples = size(data, 1)
        replica.n_features = size(data, 2)
        
        # Copy sample IDs
        if sample_ids === nothing
            sample_ids = Int32.(1:replica.n_samples)
        end
        replica.sample_ids = CuVector{Int32}(sample_ids)
        
        # Update metadata
        replica.feature_names = feature_names
        replica.feature_indices = Dict(name => i for (i, name) in enumerate(feature_names))
        
        # Calculate memory usage
        replica.allocated_memory = sizeof(replica.feature_data) + sizeof(replica.sample_ids)
        replica.last_access = now()
        replica.is_loaded = true
        
        # Force synchronization
        CUDA.synchronize()
        
        @info "Loaded dataset to GPU $(replica.gpu_id): $(replica.n_samples) samples × $(replica.n_features) features ($(round(replica.allocated_memory / 1024^2, digits=2)) MB)"
        
    catch e
        @error "Failed to load dataset to GPU $(replica.gpu_id): $e"
        replica.is_loaded = false
        rethrow()
    finally
        # Restore device
        CUDA.device!(prev_device)
    end
    
    return replica
end

"""
Replicate dataset across multiple GPUs
"""
function replicate_dataset!(
    manager::DatasetManager,
    data::AbstractMatrix{Float32},
    feature_names::Vector{String};
    sample_ids::Union{Nothing, Vector{Int32}} = nothing,
    gpu_ids::Union{Nothing, Vector{Int}} = nothing
)
    # Validate input
    if isempty(data) || isempty(feature_names)
        error("Cannot replicate empty dataset")
    end
    
    if size(data, 2) != length(feature_names)
        error("Data columns ($(size(data, 2))) don't match feature names ($(length(feature_names)))")
    end
    
    # Update version
    manager.current_version = DatasetVersion(
        manager.current_version.version_id + 1,
        length(feature_names),
        size(data, 1)
    )
    
    # Determine target GPUs
    if gpu_ids === nothing
        gpu_ids = collect(keys(manager.replicas))
    end
    
    # Check memory on all GPUs first
    for gpu_id in gpu_ids
        if !has_sufficient_memory(manager, gpu_id, length(feature_names), size(data, 1))
            error("Insufficient memory on GPU $gpu_id for dataset")
        end
    end
    
    # Replicate to each GPU
    success_count = 0
    for gpu_id in gpu_ids
        if haskey(manager.replicas, gpu_id)
            try
                replica = manager.replicas[gpu_id]
                replica.version = manager.current_version
                load_dataset_to_gpu!(replica, data, feature_names, sample_ids=sample_ids)
                success_count += 1
            catch e
                @error "Failed to replicate to GPU $gpu_id: $e"
            end
        end
    end
    
    @info "Dataset replicated to $success_count GPUs (version $(manager.current_version.version_id))"
    
    return success_count
end

"""
Get dataset replica for a specific GPU
"""
function get_dataset_replica(manager::DatasetManager, gpu_id::Int)
    if !haskey(manager.replicas, gpu_id)
        error("No replica found for GPU $gpu_id")
    end
    
    replica = manager.replicas[gpu_id]
    replica.last_access = now()
    
    return replica
end

"""
Get feature column by name or index
"""
function get_feature_column(
    replica::DatasetReplica,
    feature::Union{String, Int};
    as_host::Bool = false
)
    if !replica.is_loaded
        error("Dataset not loaded on GPU $(replica.gpu_id)")
    end
    
    # Get column index
    col_idx = if isa(feature, String)
        get(replica.feature_indices, feature, 0)
    else
        feature
    end
    
    if col_idx <= 0 || col_idx > replica.n_features
        error("Invalid feature index: $col_idx")
    end
    
    # Save current device
    prev_device = CUDA.device()
    
    try
        CUDA.device!(replica.gpu_id)
        
        # Get column data
        column_data = @view replica.feature_data[:, col_idx]
        
        if as_host
            return Array(column_data)
        else
            return column_data
        end
    finally
        CUDA.device!(prev_device)
    end
end

"""
Get batch of samples by indices
"""
function get_sample_batch(
    replica::DatasetReplica,
    sample_indices::AbstractVector{Int};
    feature_indices::Union{Nothing, AbstractVector{Int}} = nothing,
    as_host::Bool = false
)
    if !replica.is_loaded
        error("Dataset not loaded on GPU $(replica.gpu_id)")
    end
    
    # Save current device
    prev_device = CUDA.device()
    
    try
        CUDA.device!(replica.gpu_id)
        
        # Get data subset
        if feature_indices === nothing
            batch_data = @view replica.feature_data[sample_indices, :]
        else
            batch_data = @view replica.feature_data[sample_indices, feature_indices]
        end
        
        if as_host
            return Array(batch_data)
        else
            return batch_data
        end
    finally
        CUDA.device!(prev_device)
    end
end

"""
Update dataset on specific GPU(s)
"""
function update_dataset!(
    manager::DatasetManager,
    gpu_id::Int,
    feature_updates::Dict{String, Vector{Float32}};
    increment_version::Bool = true
)
    if !haskey(manager.replicas, gpu_id)
        error("No replica found for GPU $gpu_id")
    end
    
    replica = manager.replicas[gpu_id]
    if !replica.is_loaded
        error("Dataset not loaded on GPU $gpu_id")
    end
    
    # Save current device
    prev_device = CUDA.device()
    
    try
        CUDA.device!(gpu_id)
        
        # Apply updates
        for (feature_name, new_values) in feature_updates
            if haskey(replica.feature_indices, feature_name)
                col_idx = replica.feature_indices[feature_name]
                
                if length(new_values) != replica.n_samples
                    error("Update size mismatch for feature $feature_name")
                end
                
                # Update column
                replica.feature_data[:, col_idx] .= CuArray{Float32}(new_values)
            else
                @warn "Feature $feature_name not found in dataset"
            end
        end
        
        # Update version if requested
        if increment_version
            manager.current_version = DatasetVersion(
                manager.current_version.version_id + 1,
                replica.n_features,
                replica.n_samples
            )
            replica.version = manager.current_version
        end
        
        replica.last_access = now()
        CUDA.synchronize()
        
    finally
        CUDA.device!(prev_device)
    end
end

"""
Synchronize datasets across GPUs
"""
function sync_datasets!(manager::DatasetManager; force::Bool = false)
    # Check if sync is needed
    time_since_sync = Dates.value(now() - manager.last_sync) / 1000  # Convert to seconds
    if !force && time_since_sync < manager.sync_interval
        return
    end
    
    # Find reference GPU (most recent version)
    ref_gpu_id = -1
    latest_version = 0
    
    for (gpu_id, replica) in manager.replicas
        if replica.is_loaded && replica.version.version_id > latest_version
            latest_version = replica.version.version_id
            ref_gpu_id = gpu_id
        end
    end
    
    if ref_gpu_id == -1
        @warn "No loaded datasets to sync"
        return
    end
    
    # Get reference data
    ref_replica = manager.replicas[ref_gpu_id]
    ref_data = Array(ref_replica.feature_data)
    
    # Sync to other GPUs
    sync_count = 0
    for (gpu_id, replica) in manager.replicas
        if gpu_id != ref_gpu_id && replica.version.version_id < latest_version
            try
                load_dataset_to_gpu!(
                    replica,
                    ref_data,
                    ref_replica.feature_names,
                    sample_ids = Array(ref_replica.sample_ids)
                )
                replica.version = ref_replica.version
                sync_count += 1
            catch e
                @error "Failed to sync GPU $gpu_id: $e"
            end
        end
    end
    
    manager.last_sync = now()
    
    if sync_count > 0
        @info "Synchronized $sync_count GPUs to version $latest_version"
    end
end

"""
Clear dataset from GPU memory
"""
function clear_dataset!(manager::DatasetManager, gpu_id::Int)
    if !haskey(manager.replicas, gpu_id)
        return
    end
    
    replica = manager.replicas[gpu_id]
    
    if CUDA.functional() && replica.is_loaded
        prev_device = CUDA.device()
        try
            CUDA.device!(gpu_id)
            
            if replica.feature_data !== nothing
                CUDA.unsafe_free!(replica.feature_data)
                replica.feature_data = nothing
            end
            
            if replica.sample_ids !== nothing
                CUDA.unsafe_free!(replica.sample_ids)
                replica.sample_ids = nothing
            end
            
            CUDA.synchronize()
        finally
            CUDA.device!(prev_device)
        end
    end
    
    replica.is_loaded = false
    replica.allocated_memory = 0
    
    @info "Cleared dataset from GPU $gpu_id"
end

"""
Get dataset information
"""
function get_dataset_info(manager::DatasetManager)
    info = Dict{String, Any}()
    
    info["version"] = manager.current_version.version_id
    info["timestamp"] = manager.current_version.timestamp
    info["memory_limit_per_gpu"] = manager.memory_limit_per_gpu
    info["compression_enabled"] = manager.enable_compression
    
    gpu_info = Dict{String, Any}()
    for (gpu_id, replica) in manager.replicas
        gpu_data = Dict(
            "loaded" => replica.is_loaded,
            "version" => replica.version.version_id,
            "n_samples" => replica.n_samples,
            "n_features" => replica.n_features,
            "memory_mb" => round(replica.allocated_memory / 1024^2, digits=2),
            "last_access" => replica.last_access
        )
        
        if CUDA.functional()
            stats = get_memory_usage(gpu_id)
            gpu_data["total_memory_gb"] = round(stats.total_memory / 1024^3, digits=2)
            gpu_data["free_memory_gb"] = round(stats.free_memory / 1024^3, digits=2)
            gpu_data["utilization"] = round(stats.utilization * 100, digits=1)
        end
        
        gpu_info["GPU$gpu_id"] = gpu_data
    end
    
    info["gpus"] = gpu_info
    
    return info
end

end # module