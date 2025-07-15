module GPUMemoryManagement

using CUDA
using Statistics
using Printf
using JSON3
using Dates

export MemoryPool, MemoryConfig, MemoryStats, MemoryBlock
export create_memory_manager, allocate_memory!, free_memory!, defragment!, defragment_pool!
export get_memory_stats, check_memory_pressure, set_adaptive_batch_size!
export register_component!, unregister_component!, get_component_usage
export save_memory_profile, optimize_memory_layout!, reset_memory_manager!

"""
Configuration for GPU memory management system
"""
struct MemoryConfig
    total_vram_gb::Float64              # Total VRAM budget (e.g., 24.0 for RTX 4090)
    reserved_system_gb::Float64         # Reserved for system operations
    pool_size_ratios::Dict{Symbol, Float64}  # Memory pool allocation ratios
    enable_defragmentation::Bool        # Enable automatic defragmentation
    defrag_threshold::Float64           # Trigger defragmentation when fragmentation > threshold
    pressure_warning_threshold::Float64 # Memory pressure warning level (0.8 = 80%)
    pressure_critical_threshold::Float64 # Critical memory pressure level (0.95 = 95%)
    adaptive_batch_sizing::Bool         # Enable adaptive batch size adjustment
    min_batch_size::Int                 # Minimum batch size for adaptive sizing
    max_batch_size::Int                 # Maximum batch size
    enable_memory_profiling::Bool       # Enable detailed memory profiling
    cache_cleanup_interval::Int         # Cache cleanup frequency (iterations)
end

"""
Default memory configuration for RTX 4090 (24GB VRAM)
"""
function MemoryConfig(;
    total_vram_gb::Float64 = 24.0,
    reserved_system_gb::Float64 = 2.0,
    pool_size_ratios::Dict{Symbol, Float64} = Dict(
        :model_weights => 0.3,     # 30% for model parameters
        :replay_buffer => 0.25,    # 25% for experience replay
        :inference_batch => 0.2,   # 20% for batch inference
        :training_cache => 0.15,   # 15% for training temporaries
        :general_pool => 0.1       # 10% for general allocations
    ),
    enable_defragmentation::Bool = true,
    defrag_threshold::Float64 = 0.3,
    pressure_warning_threshold::Float64 = 0.8,
    pressure_critical_threshold::Float64 = 0.95,
    adaptive_batch_sizing::Bool = true,
    min_batch_size::Int = 32,
    max_batch_size::Int = 2048,
    enable_memory_profiling::Bool = true,
    cache_cleanup_interval::Int = 1000
)
    # Validate pool ratios sum to 1.0
    total_ratio = sum(values(pool_size_ratios))
    if abs(total_ratio - 1.0) > 0.01
        throw(ArgumentError("Pool size ratios must sum to 1.0, got $total_ratio"))
    end
    
    MemoryConfig(
        total_vram_gb, reserved_system_gb, pool_size_ratios,
        enable_defragmentation, defrag_threshold,
        pressure_warning_threshold, pressure_critical_threshold,
        adaptive_batch_sizing, min_batch_size, max_batch_size,
        enable_memory_profiling, cache_cleanup_interval
    )
end

"""
Memory block representation for tracking allocations
"""
mutable struct MemoryBlock
    id::String                          # Unique block identifier
    ptr::CuPtr{Nothing}                 # GPU memory pointer
    size_bytes::Int64                   # Block size in bytes
    pool::Symbol                        # Pool this block belongs to
    component::String                   # Component that owns this block
    allocated_time::DateTime            # When block was allocated
    last_access_time::DateTime          # Last access timestamp
    access_count::Int64                 # Number of accesses
    is_free::Bool                       # Whether block is available
    fragmentation_score::Float64        # Fragmentation metric
end

"""
Memory pool for managing allocations of specific types
"""
mutable struct MemoryPool
    name::Symbol                        # Pool identifier
    total_size_bytes::Int64             # Total pool size
    allocated_bytes::Int64              # Currently allocated bytes
    free_bytes::Int64                   # Available bytes
    blocks::Vector{MemoryBlock}         # All blocks in this pool
    free_blocks::Vector{MemoryBlock}    # Available blocks
    largest_free_block::Int64           # Size of largest contiguous free block
    fragmentation_ratio::Float64        # Current fragmentation level
    allocation_count::Int64             # Total allocations made
    deallocation_count::Int64           # Total deallocations made
end

"""
Memory statistics for monitoring and profiling
"""
mutable struct MemoryStats
    timestamp::DateTime
    
    # Overall memory usage
    total_vram_bytes::Int64
    system_reserved_bytes::Int64
    allocated_bytes::Int64
    free_bytes::Int64
    fragmentation_ratio::Float64
    
    # Pool-specific stats
    pool_stats::Dict{Symbol, Dict{String, Any}}
    
    # Component usage
    component_usage::Dict{String, Int64}
    
    # Performance metrics
    allocation_latency_ms::Float64
    deallocation_latency_ms::Float64
    defragmentation_count::Int64
    
    # Pressure indicators
    memory_pressure::Float64
    pressure_level::Symbol  # :normal, :warning, :critical
    
    # Adaptive sizing
    current_batch_size::Int
    batch_size_adjustments::Int64
    
    # System health
    cuda_memory_info::Tuple{Int64, Int64}  # (free, total) from CUDA
    last_defrag_time::DateTime
end

"""
Main GPU memory manager
"""
mutable struct GPUMemoryManager
    config::MemoryConfig
    device_id::Int
    pools::Dict{Symbol, MemoryPool}
    
    # Global tracking
    total_allocated_bytes::Int64
    total_free_bytes::Int64
    block_registry::Dict{String, MemoryBlock}
    
    # Component registration
    registered_components::Dict{String, Dict{String, Any}}
    
    # Performance tracking
    stats_history::Vector{MemoryStats}
    allocation_times::Vector{Float64}
    deallocation_times::Vector{Float64}
    
    # Adaptive batch sizing
    current_batch_size::Int
    batch_performance_history::Vector{Float64}
    
    # Defragmentation
    last_defrag_time::DateTime
    defrag_operations::Int64
    
    # Caching and cleanup
    cache_objects::Dict{String, Any}
    last_cleanup_time::DateTime
    cleanup_count::Int64
    
    # System RAM fallback
    fallback_storage::Dict{String, Array}
    fallback_usage_bytes::Int64
    
    # Profiling
    memory_profile::Dict{String, Any}
    profiling_enabled::Bool
end

"""
Create GPU memory manager with specified configuration
"""
function create_memory_manager(config::MemoryConfig; device_id::Int = 0)
    # Verify GPU availability
    if !CUDA.functional()
        throw(ErrorException("CUDA not functional, cannot create GPU memory manager"))
    end
    
    # Switch to target device
    if device_id >= 0 && device_id < length(CUDA.devices())
        CUDA.device!(device_id)
    else
        throw(ArgumentError("Invalid device_id: $device_id"))
    end
    
    # Get actual GPU memory info
    local free_bytes, total_bytes
    try
        memory_info = CUDA.memory_status()
        if memory_info !== nothing
            free_bytes, total_bytes = memory_info
        else
            # Fallback values if memory_status returns nothing
            total_bytes = Int64(config.total_vram_gb * 1024^3)
            free_bytes = total_bytes - Int64(1024^3)  # Assume 1GB used
        end
    catch
        # Fallback values if memory_status fails
        total_bytes = Int64(config.total_vram_gb * 1024^3)
        free_bytes = total_bytes - Int64(1024^3)  # Assume 1GB used
    end
    total_gb = total_bytes / (1024^3)
    
    if total_gb < config.total_vram_gb
        @warn "Actual GPU memory ($(round(total_gb, digits=2))GB) less than configured ($(config.total_vram_gb)GB)"
    end
    
    # Calculate available memory for pools
    system_reserved = Int64(config.reserved_system_gb * 1024^3)
    available_bytes = total_bytes - system_reserved
    
    # Create memory pools
    pools = Dict{Symbol, MemoryPool}()
    
    for (pool_name, ratio) in config.pool_size_ratios
        pool_size = Int64(round(available_bytes * ratio))
        pools[pool_name] = MemoryPool(
            pool_name,
            pool_size,
            0,                          # allocated_bytes
            pool_size,                  # free_bytes
            Vector{MemoryBlock}(),      # blocks
            Vector{MemoryBlock}(),      # free_blocks
            pool_size,                  # largest_free_block
            0.0,                        # fragmentation_ratio
            0,                          # allocation_count
            0                           # deallocation_count
        )
        
        @info "Created memory pool '$pool_name': $(round(pool_size / 1024^3, digits=2))GB"
    end
    
    # Initialize manager
    manager = GPUMemoryManager(
        config,
        device_id,
        pools,
        0,                              # total_allocated_bytes
        available_bytes,                # total_free_bytes
        Dict{String, MemoryBlock}(),    # block_registry
        Dict{String, Dict{String, Any}}(), # registered_components
        Vector{MemoryStats}(),          # stats_history
        Vector{Float64}(),              # allocation_times
        Vector{Float64}(),              # deallocation_times
        config.max_batch_size,          # current_batch_size
        Vector{Float64}(),              # batch_performance_history
        now(),                          # last_defrag_time
        0,                              # defrag_operations
        Dict{String, Any}(),            # cache_objects
        now(),                          # last_cleanup_time
        0,                              # cleanup_count
        Dict{String, Array}(),          # fallback_storage
        0,                              # fallback_usage_bytes
        Dict{String, Any}(),            # memory_profile
        config.enable_memory_profiling  # profiling_enabled
    )
    
    # Initialize memory profile
    if manager.profiling_enabled
        initialize_memory_profile!(manager)
    end
    
    @info "GPU Memory Manager initialized on device $device_id"
    @info "Available memory: $(round(available_bytes / 1024^3, digits=2))GB"
    @info "Pool configuration: $(config.pool_size_ratios)"
    
    return manager
end

"""
Initialize memory profiling data
"""
function initialize_memory_profile!(manager::GPUMemoryManager)
    manager.memory_profile = Dict{String, Any}(
        "device_id" => manager.device_id,
        "total_vram_gb" => manager.config.total_vram_gb,
        "pools" => Dict{String, Any}(),
        "allocation_patterns" => Dict{String, Vector{Int64}}(),
        "fragmentation_history" => Vector{Float64}(),
        "pressure_history" => Vector{Float64}(),
        "batch_size_history" => Vector{Int}(),
        "performance_metrics" => Dict{String, Vector{Float64}}(),
        "component_profiles" => Dict{String, Dict{String, Any}}(),
        "created_time" => now()
    )
    
    for pool_name in keys(manager.pools)
        manager.memory_profile["pools"][string(pool_name)] = Dict{String, Any}(
            "allocations" => Vector{Int64}(),
            "deallocations" => Vector{Int64}(),
            "peak_usage" => 0,
            "avg_block_size" => 0.0,
            "fragmentation_events" => Vector{DateTime}()
        )
    end
end

"""
Register a component with the memory manager
"""
function register_component!(manager::GPUMemoryManager, component_name::String; 
                           max_memory_gb::Float64 = 1.0,
                           preferred_pools::Vector{Symbol} = [:general_pool],
                           priority::Int = 1)
    
    if haskey(manager.registered_components, component_name)
        @warn "Component '$component_name' already registered, updating..."
    end
    
    manager.registered_components[component_name] = Dict{String, Any}(
        "max_memory_bytes" => Int64(max_memory_gb * 1024^3),
        "preferred_pools" => preferred_pools,
        "priority" => priority,
        "allocated_blocks" => Vector{String}(),
        "total_allocated_bytes" => 0,
        "allocation_count" => 0,
        "registration_time" => now()
    )
    
    if manager.profiling_enabled
        manager.memory_profile["component_profiles"][component_name] = Dict{String, Any}(
            "max_memory_gb" => max_memory_gb,
            "allocation_history" => Vector{Int64}(),
            "peak_usage" => 0,
            "access_patterns" => Vector{DateTime}()
        )
    end
    
    @info "Registered component '$component_name' with $(max_memory_gb)GB limit"
    return true
end

"""
Allocate GPU memory for a component
"""
function allocate_memory!(manager::GPUMemoryManager, component_name::String, 
                         size_bytes::Int64; pool::Symbol = :general_pool,
                         alignment::Int = 256)
    
    start_time = time()
    
    # Check if component is registered
    if !haskey(manager.registered_components, component_name)
        throw(ArgumentError("Component '$component_name' not registered"))
    end
    
    # Check component memory limits
    component = manager.registered_components[component_name]
    if component["total_allocated_bytes"] + size_bytes > component["max_memory_bytes"]
        # Try fallback to system RAM
        if try_fallback_allocation!(manager, component_name, size_bytes)
            @warn "Using system RAM fallback for component '$component_name'"
            return "fallback_$(component_name)_$(length(manager.fallback_storage))"
        else
            throw(OutOfMemoryError("Component '$component_name' would exceed memory limit"))
        end
    end
    
    # Align size to requested alignment
    aligned_size = ((size_bytes + alignment - 1) ÷ alignment) * alignment
    
    # Check if pool exists and has enough space
    if !haskey(manager.pools, pool)
        throw(ArgumentError("Pool '$pool' does not exist"))
    end
    
    target_pool = manager.pools[pool]
    
    # Check memory pressure and trigger adaptive responses
    check_and_respond_to_pressure!(manager, aligned_size)
    
    # Try to allocate from target pool
    block = try_allocate_from_pool!(manager, target_pool, component_name, aligned_size)
    
    if block === nothing
        # Try defragmentation if enabled
        if manager.config.enable_defragmentation && 
           target_pool.fragmentation_ratio > manager.config.defrag_threshold
            
            @info "Triggering defragmentation for pool '$pool'"
            defragment_pool!(manager, pool)
            
            # Try allocation again after defragmentation
            block = try_allocate_from_pool!(manager, target_pool, component_name, aligned_size)
        end
        
        # If still no success, try other pools
        if block === nothing
            for (other_pool_name, other_pool) in manager.pools
                if other_pool_name != pool && other_pool.free_bytes >= aligned_size
                    block = try_allocate_from_pool!(manager, other_pool, component_name, aligned_size)
                    if block !== nothing
                        @info "Allocated from alternative pool '$other_pool_name'"
                        break
                    end
                end
            end
        end
        
        # Final fallback to system RAM
        if block === nothing
            fallback_key = try_fallback_allocation!(manager, component_name, aligned_size)
            if fallback_key !== nothing
                allocation_time = (time() - start_time) * 1000
                push!(manager.allocation_times, allocation_time)
                @warn "Using system RAM fallback after GPU allocation failure"
                return "fallback_$(fallback_key)"
            else
                throw(OutOfMemoryError("Unable to allocate $(aligned_size) bytes for '$component_name'"))
            end
        end
    end
    
    # Update statistics
    allocation_time = (time() - start_time) * 1000
    push!(manager.allocation_times, allocation_time)
    
    # Register allocation with component
    push!(component["allocated_blocks"], block.id)
    component["total_allocated_bytes"] += aligned_size
    component["allocation_count"] += 1
    
    # Update profiling data
    if manager.profiling_enabled
        update_allocation_profile!(manager, component_name, aligned_size, pool)
    end
    
    @debug "Allocated $(aligned_size) bytes for '$component_name' in pool '$pool'"
    return block.id
end

"""
Try to allocate memory from a specific pool
"""
function try_allocate_from_pool!(manager::GPUMemoryManager, pool::MemoryPool, 
                                component_name::String, size_bytes::Int64)
    
    if pool.free_bytes < size_bytes
        return nothing
    end
    
    # Try to find existing free block
    suitable_block = nothing
    for block in pool.free_blocks
        if block.size_bytes >= size_bytes && block.is_free
            suitable_block = block
            break
        end
    end
    
    # If no suitable free block, try to allocate new one
    if suitable_block === nothing
        try
            # Allocate new GPU memory
            gpu_ptr = CUDA.malloc(size_bytes)
            
            # Create memory block
            block_id = "$(pool.name)_$(component_name)_$(time_ns())"
            suitable_block = MemoryBlock(
                block_id,
                gpu_ptr,
                size_bytes,
                pool.name,
                component_name,
                now(),
                now(),
                0,
                false,
                0.0
            )
            
            push!(pool.blocks, suitable_block)
            manager.block_registry[block_id] = suitable_block
            
        catch e
            @warn "GPU allocation failed: $e"
            return nothing
        end
    else
        # Reuse existing block
        suitable_block.component = component_name
        suitable_block.allocated_time = now()
        suitable_block.is_free = false
        
        # Remove from free blocks list
        filter!(b -> b.id != suitable_block.id, pool.free_blocks)
    end
    
    # Update pool statistics
    pool.allocated_bytes += size_bytes
    pool.free_bytes -= size_bytes
    pool.allocation_count += 1
    
    # Update global statistics
    manager.total_allocated_bytes += size_bytes
    manager.total_free_bytes -= size_bytes
    
    # Update fragmentation metrics
    update_fragmentation_metrics!(pool)
    
    return suitable_block
end

"""
Try fallback allocation to system RAM
"""
function try_fallback_allocation!(manager::GPUMemoryManager, component_name::String, size_bytes::Int64)
    try
        # Allocate in system RAM
        fallback_array = zeros(UInt8, size_bytes)
        fallback_key = "$(component_name)_$(time_ns())"
        manager.fallback_storage[fallback_key] = fallback_array
        manager.fallback_usage_bytes += size_bytes
        
        @info "Allocated $(size_bytes) bytes in system RAM fallback for '$component_name'"
        return fallback_key
    catch e
        @error "System RAM fallback allocation failed: $e"
        return nothing
    end
end

"""
Free allocated memory
"""
function free_memory!(manager::GPUMemoryManager, block_id::String)
    start_time = time()
    
    # Check if it's a fallback allocation
    if startswith(block_id, "fallback_")
        return free_fallback_memory!(manager, block_id)
    end
    
    # Find block in registry
    if !haskey(manager.block_registry, block_id)
        @warn "Block '$block_id' not found in registry"
        return false
    end
    
    block = manager.block_registry[block_id]
    pool = manager.pools[block.pool]
    
    # Mark block as free
    block.is_free = true
    block.last_access_time = now()
    push!(pool.free_blocks, block)
    
    # Update pool statistics
    pool.allocated_bytes -= block.size_bytes
    pool.free_bytes += block.size_bytes
    pool.deallocation_count += 1
    
    # Update global statistics
    manager.total_allocated_bytes -= block.size_bytes
    manager.total_free_bytes += block.size_bytes
    
    # Update component tracking
    component = manager.registered_components[block.component]
    filter!(id -> id != block_id, component["allocated_blocks"])
    component["total_allocated_bytes"] -= block.size_bytes
    
    # Update fragmentation metrics
    update_fragmentation_metrics!(pool)
    
    # Record deallocation time
    deallocation_time = (time() - start_time) * 1000
    push!(manager.deallocation_times, deallocation_time)
    
    @debug "Freed $(block.size_bytes) bytes from block '$block_id'"
    return true
end

"""
Free fallback memory allocation
"""
function free_fallback_memory!(manager::GPUMemoryManager, fallback_block_id::String)
    # Extract actual key from fallback identifier (remove "fallback_" prefix)
    actual_key = replace(fallback_block_id, "fallback_" => "", count=1)
    
    if !haskey(manager.fallback_storage, actual_key)
        @warn "Fallback allocation '$actual_key' not found"
        return false
    end
    
    array = manager.fallback_storage[actual_key]
    size_bytes = sizeof(array)
    
    delete!(manager.fallback_storage, actual_key)
    manager.fallback_usage_bytes -= size_bytes
    
    @debug "Freed $(size_bytes) bytes from system RAM fallback"
    return true
end

"""
Update fragmentation metrics for a pool
"""
function update_fragmentation_metrics!(pool::MemoryPool)
    if isempty(pool.blocks)
        pool.fragmentation_ratio = 0.0
        pool.largest_free_block = pool.free_bytes
        return
    end
    
    # Calculate fragmentation ratio
    total_free_blocks = length(pool.free_blocks)
    if total_free_blocks <= 1
        pool.fragmentation_ratio = 0.0
    else
        # Fragmentation = (number of free blocks - 1) / number of free blocks
        pool.fragmentation_ratio = (total_free_blocks - 1) / total_free_blocks
    end
    
    # Find largest contiguous free block
    pool.largest_free_block = 0
    for block in pool.free_blocks
        if block.size_bytes > pool.largest_free_block
            pool.largest_free_block = block.size_bytes
        end
    end
end

"""
Check memory pressure and respond adaptively
"""
function check_and_respond_to_pressure!(manager::GPUMemoryManager, additional_bytes::Int64 = 0)
    total_available = manager.total_free_bytes
    total_capacity = manager.total_allocated_bytes + manager.total_free_bytes
    
    # Calculate pressure including potential allocation
    pressure_ratio = (manager.total_allocated_bytes + additional_bytes) / total_capacity
    
    if pressure_ratio > manager.config.pressure_critical_threshold
        @warn "Critical memory pressure: $(round(pressure_ratio * 100, digits=1))%"
        
        # Trigger aggressive cleanup
        cleanup_caches!(manager)
        
        # Reduce batch size if adaptive sizing enabled
        if manager.config.adaptive_batch_sizing
            new_batch_size = max(manager.config.min_batch_size, 
                               manager.current_batch_size ÷ 2)
            set_adaptive_batch_size!(manager, new_batch_size)
        end
        
        # Force defragmentation
        if manager.config.enable_defragmentation
            for pool_name in keys(manager.pools)
                defragment_pool!(manager, pool_name)
            end
        end
        
    elseif pressure_ratio > manager.config.pressure_warning_threshold
        @info "Memory pressure warning: $(round(pressure_ratio * 100, digits=1))%"
        
        # Trigger cache cleanup
        if time() - manager.last_cleanup_time > 30.0  # 30 second cooldown
            cleanup_caches!(manager)
        end
        
        # Reduce batch size moderately
        if manager.config.adaptive_batch_sizing
            new_batch_size = max(manager.config.min_batch_size,
                               Int(round(manager.current_batch_size * 0.8)))
            set_adaptive_batch_size!(manager, new_batch_size)
        end
    end
end

"""
Set adaptive batch size for inference operations
"""
function set_adaptive_batch_size!(manager::GPUMemoryManager, new_batch_size::Int)
    old_batch_size = manager.current_batch_size
    manager.current_batch_size = clamp(new_batch_size, 
                                     manager.config.min_batch_size,
                                     manager.config.max_batch_size)
    
    if manager.current_batch_size != old_batch_size
        @info "Adjusted batch size: $(old_batch_size) → $(manager.current_batch_size)"
        
        if manager.profiling_enabled
            push!(manager.memory_profile["batch_size_history"], manager.current_batch_size)
        end
    end
end

"""
Perform defragmentation on a memory pool
"""
function defragment_pool!(manager::GPUMemoryManager, pool_name::Symbol)
    if !haskey(manager.pools, pool_name)
        return false
    end
    
    pool = manager.pools[pool_name]
    
    if pool.fragmentation_ratio < manager.config.defrag_threshold
        return false  # No defragmentation needed
    end
    
    start_time = time()
    
    # Sort free blocks by memory address to identify contiguous regions
    sort!(pool.free_blocks, by = b -> UInt64(b.ptr))
    
    # Merge contiguous free blocks
    merged_count = 0
    i = 1
    while i < length(pool.free_blocks)
        current_block = pool.free_blocks[i]
        next_block = pool.free_blocks[i + 1]
        
        # Check if blocks are contiguous
        if UInt64(current_block.ptr) + current_block.size_bytes == UInt64(next_block.ptr)
            # Merge blocks
            current_block.size_bytes += next_block.size_bytes
            
            # Remove merged block from pool and registry
            delete!(manager.block_registry, next_block.id)
            deleteat!(pool.free_blocks, i + 1)
            filter!(b -> b.id != next_block.id, pool.blocks)
            
            merged_count += 1
            
            # Don't increment i to check for further merges
        else
            i += 1
        end
    end
    
    # Update fragmentation metrics
    update_fragmentation_metrics!(pool)
    
    # Record defragmentation
    manager.last_defrag_time = now()
    manager.defrag_operations += 1
    
    defrag_time = (time() - start_time) * 1000
    
    @info "Defragmented pool '$pool_name': merged $merged_count blocks in $(round(defrag_time, digits=2))ms"
    @info "New fragmentation ratio: $(round(pool.fragmentation_ratio, digits=3))"
    
    if manager.profiling_enabled
        pool_profile = manager.memory_profile["pools"][string(pool_name)]
        push!(pool_profile["fragmentation_events"], now())
    end
    
    return true
end

"""
Cleanup cached objects and temporary allocations
"""
function cleanup_caches!(manager::GPUMemoryManager)
    start_time = time()
    cleaned_bytes = 0
    
    # Clean up cache objects
    for (key, obj) in manager.cache_objects
        if obj isa CuArray
            cleaned_bytes += sizeof(obj)
            CUDA.unsafe_free!(obj)
        end
    end
    
    empty!(manager.cache_objects)
    
    # Trigger CUDA garbage collection
    GC.gc()
    CUDA.reclaim()
    
    manager.last_cleanup_time = now()
    manager.cleanup_count += 1
    
    cleanup_time = (time() - start_time) * 1000
    
    @info "Cache cleanup completed: freed $(round(cleaned_bytes / 1024^2, digits=2))MB in $(round(cleanup_time, digits=2))ms"
    
    return cleaned_bytes
end

"""
Get comprehensive memory statistics
"""
function get_memory_stats(manager::GPUMemoryManager)
    # Get current CUDA memory info
    local cuda_free, cuda_total
    try
        memory_info = CUDA.memory_status()
        if memory_info !== nothing
            cuda_free, cuda_total = memory_info
        else
            cuda_free = manager.total_free_bytes
            cuda_total = manager.total_allocated_bytes + manager.total_free_bytes
        end
    catch
        cuda_free = manager.total_free_bytes
        cuda_total = manager.total_allocated_bytes + manager.total_free_bytes
    end
    
    # Calculate overall fragmentation
    total_fragmentation = 0.0
    for pool in values(manager.pools)
        total_fragmentation += pool.fragmentation_ratio
    end
    avg_fragmentation = total_fragmentation / length(manager.pools)
    
    # Calculate memory pressure
    total_capacity = manager.total_allocated_bytes + manager.total_free_bytes
    memory_pressure = manager.total_allocated_bytes / total_capacity
    
    pressure_level = if memory_pressure > manager.config.pressure_critical_threshold
        :critical
    elseif memory_pressure > manager.config.pressure_warning_threshold
        :warning
    else
        :normal
    end
    
    # Calculate average allocation/deallocation times
    avg_alloc_time = isempty(manager.allocation_times) ? 0.0 : mean(manager.allocation_times)
    avg_dealloc_time = isempty(manager.deallocation_times) ? 0.0 : mean(manager.deallocation_times)
    
    # Build pool statistics
    pool_stats = Dict{Symbol, Dict{String, Any}}()
    for (name, pool) in manager.pools
        pool_stats[name] = Dict{String, Any}(
            "total_size_gb" => pool.total_size_bytes / 1024^3,
            "allocated_bytes" => pool.allocated_bytes,
            "free_bytes" => pool.free_bytes,
            "utilization_ratio" => pool.allocated_bytes / pool.total_size_bytes,
            "fragmentation_ratio" => pool.fragmentation_ratio,
            "largest_free_block_mb" => pool.largest_free_block / 1024^2,
            "allocation_count" => pool.allocation_count,
            "deallocation_count" => pool.deallocation_count,
            "block_count" => length(pool.blocks),
            "free_block_count" => length(pool.free_blocks)
        )
    end
    
    # Build component usage statistics
    component_usage = Dict{String, Int64}()
    for (name, component) in manager.registered_components
        component_usage[name] = component["total_allocated_bytes"]
    end
    
    # Create comprehensive stats object
    stats = MemoryStats(
        now(),
        Int64(manager.config.total_vram_gb * 1024^3),
        Int64(manager.config.reserved_system_gb * 1024^3),
        manager.total_allocated_bytes,
        manager.total_free_bytes,
        avg_fragmentation,
        pool_stats,
        component_usage,
        avg_alloc_time,
        avg_dealloc_time,
        manager.defrag_operations,
        memory_pressure,
        pressure_level,
        manager.current_batch_size,
        length(manager.batch_performance_history),
        (cuda_free, cuda_total),
        manager.last_defrag_time
    )
    
    # Add to history
    push!(manager.stats_history, stats)
    
    # Keep only recent history (last 1000 entries)
    if length(manager.stats_history) > 1000
        deleteat!(manager.stats_history, 1:length(manager.stats_history)-1000)
    end
    
    return stats
end

"""
Check current memory pressure level
"""
function check_memory_pressure(manager::GPUMemoryManager)
    total_capacity = manager.total_allocated_bytes + manager.total_free_bytes
    pressure_ratio = manager.total_allocated_bytes / total_capacity
    
    pressure_info = Dict{String, Any}(
        "pressure_ratio" => pressure_ratio,
        "pressure_percentage" => pressure_ratio * 100,
        "allocated_gb" => manager.total_allocated_bytes / 1024^3,
        "free_gb" => manager.total_free_bytes / 1024^3,
        "fallback_usage_mb" => manager.fallback_usage_bytes / 1024^2,
        "level" => if pressure_ratio > manager.config.pressure_critical_threshold
            "critical"
        elseif pressure_ratio > manager.config.pressure_warning_threshold
            "warning"
        else
            "normal"
        end,
        "recommendation" => if pressure_ratio > manager.config.pressure_critical_threshold
            "Immediate action required: cleanup caches, reduce batch sizes, or restart"
        elseif pressure_ratio > manager.config.pressure_warning_threshold
            "Consider cleanup or reducing memory usage"
        else
            "Memory usage normal"
        end
    )
    
    return pressure_info
end

"""
Update allocation profile for component
"""
function update_allocation_profile!(manager::GPUMemoryManager, component_name::String, 
                                  size_bytes::Int64, pool::Symbol)
    if !manager.profiling_enabled
        return
    end
    
    # Update component profile
    if haskey(manager.memory_profile["component_profiles"], component_name)
        profile = manager.memory_profile["component_profiles"][component_name]
        push!(profile["allocation_history"], size_bytes)
        profile["peak_usage"] = max(profile["peak_usage"], 
                                  manager.registered_components[component_name]["total_allocated_bytes"])
        push!(profile["access_patterns"], now())
    end
    
    # Update pool profile
    pool_profile = manager.memory_profile["pools"][string(pool)]
    push!(pool_profile["allocations"], size_bytes)
    pool_profile["peak_usage"] = max(pool_profile["peak_usage"], 
                                   manager.pools[pool].allocated_bytes)
    
    # Update allocation patterns
    pattern_key = "$(component_name)_$(pool)"
    if !haskey(manager.memory_profile["allocation_patterns"], pattern_key)
        manager.memory_profile["allocation_patterns"][pattern_key] = Vector{Int64}()
    end
    push!(manager.memory_profile["allocation_patterns"][pattern_key], size_bytes)
    
    # Update global metrics
    push!(manager.memory_profile["fragmentation_history"], 
          mean([pool.fragmentation_ratio for pool in values(manager.pools)]))
    
    total_capacity = manager.total_allocated_bytes + manager.total_free_bytes
    pressure = manager.total_allocated_bytes / total_capacity
    push!(manager.memory_profile["pressure_history"], pressure)
end

"""
Get component usage statistics
"""
function get_component_usage(manager::GPUMemoryManager, component_name::String)
    if !haskey(manager.registered_components, component_name)
        return nothing
    end
    
    component = manager.registered_components[component_name]
    
    usage_stats = Dict{String, Any}(
        "allocated_blocks" => length(component["allocated_blocks"]),
        "total_allocated_bytes" => component["total_allocated_bytes"],
        "total_allocated_gb" => component["total_allocated_bytes"] / 1024^3,
        "max_memory_gb" => component["max_memory_bytes"] / 1024^3,
        "utilization_ratio" => component["total_allocated_bytes"] / component["max_memory_bytes"],
        "allocation_count" => component["allocation_count"],
        "preferred_pools" => component["preferred_pools"],
        "priority" => component["priority"],
        "registration_time" => component["registration_time"]
    )
    
    # Add profiling data if available
    if manager.profiling_enabled && haskey(manager.memory_profile["component_profiles"], component_name)
        profile = manager.memory_profile["component_profiles"][component_name]
        usage_stats["allocation_history"] = profile["allocation_history"]
        usage_stats["peak_usage"] = profile["peak_usage"]
        usage_stats["access_count"] = length(profile["access_patterns"])
    end
    
    return usage_stats
end

"""
Unregister a component and free its memory
"""
function unregister_component!(manager::GPUMemoryManager, component_name::String)
    if !haskey(manager.registered_components, component_name)
        @warn "Component '$component_name' not registered"
        return false
    end
    
    component = manager.registered_components[component_name]
    
    # Free all allocated blocks
    for block_id in copy(component["allocated_blocks"])
        free_memory!(manager, block_id)
    end
    
    # Remove from registry
    delete!(manager.registered_components, component_name)
    
    # Remove from profiling
    if manager.profiling_enabled
        delete!(manager.memory_profile["component_profiles"], component_name)
    end
    
    @info "Unregistered component '$component_name'"
    return true
end

"""
Optimize memory layout for better performance
"""
function optimize_memory_layout!(manager::GPUMemoryManager)
    @info "Optimizing memory layout..."
    
    start_time = time()
    optimizations = 0
    
    # Defragment all pools
    for pool_name in keys(manager.pools)
        if defragment_pool!(manager, pool_name)
            optimizations += 1
        end
    end
    
    # Cleanup caches
    cleaned_bytes = cleanup_caches!(manager)
    
    # Force garbage collection
    GC.gc()
    CUDA.reclaim()
    
    optimization_time = (time() - start_time) * 1000
    
    @info "Memory optimization completed in $(round(optimization_time, digits=2))ms"
    @info "Performed $optimizations defragmentation operations"
    @info "Cleaned $(round(cleaned_bytes / 1024^2, digits=2))MB from caches"
    
    return Dict{String, Any}(
        "optimization_time_ms" => optimization_time,
        "defragmentation_operations" => optimizations,
        "cleaned_bytes" => cleaned_bytes,
        "final_pressure" => check_memory_pressure(manager)
    )
end

"""
Save memory profile to file
"""
function save_memory_profile(manager::GPUMemoryManager, filename::String = "memory_profile.json")
    if !manager.profiling_enabled
        @warn "Memory profiling not enabled"
        return false
    end
    
    try
        # Add current statistics
        manager.memory_profile["final_stats"] = get_memory_stats(manager)
        manager.memory_profile["save_time"] = now()
        
        # Save to file
        open(filename, "w") do io
            JSON3.write(io, manager.memory_profile)
        end
        
        @info "Memory profile saved to $filename"
        return true
        
    catch e
        @error "Failed to save memory profile: $e"
        return false
    end
end

"""
Reset memory manager to initial state
"""
function reset_memory_manager!(manager::GPUMemoryManager)
    @info "Resetting memory manager..."
    
    # Free all allocated blocks
    for (component_name, _) in manager.registered_components
        unregister_component!(manager, component_name)
    end
    
    # Clear fallback storage
    empty!(manager.fallback_storage)
    manager.fallback_usage_bytes = 0
    
    # Clear caches
    cleanup_caches!(manager)
    
    # Reset statistics
    empty!(manager.stats_history)
    empty!(manager.allocation_times)
    empty!(manager.deallocation_times)
    empty!(manager.batch_performance_history)
    
    # Reset counters
    manager.total_allocated_bytes = 0
    manager.total_free_bytes = sum(pool.total_size_bytes for pool in values(manager.pools))
    manager.defrag_operations = 0
    manager.cleanup_count = 0
    manager.current_batch_size = manager.config.max_batch_size
    
    # Reset pools
    for pool in values(manager.pools)
        # Free all blocks
        for block in pool.blocks
            if !block.is_free
                try
                    CUDA.unsafe_free!(block.ptr)
                catch e
                    @debug "Error freeing block during reset: $e"
                end
            end
        end
        
        # Reset pool state
        empty!(pool.blocks)
        empty!(pool.free_blocks)
        pool.allocated_bytes = 0
        pool.free_bytes = pool.total_size_bytes
        pool.largest_free_block = pool.total_size_bytes
        pool.fragmentation_ratio = 0.0
        pool.allocation_count = 0
        pool.deallocation_count = 0
    end
    
    # Clear block registry
    empty!(manager.block_registry)
    
    # Force garbage collection
    GC.gc()
    CUDA.reclaim()
    
    @info "Memory manager reset completed"
    return true
end

end # module GPUMemoryManagement