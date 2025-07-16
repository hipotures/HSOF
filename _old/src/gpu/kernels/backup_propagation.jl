module BackupPropagation

using CUDA
using Statistics

# Include required types
include("mcts_types.jl")
using .MCTSTypes: MAX_NODES, NODE_ACTIVE, NODE_EXPANDED, NODE_TERMINAL, WARP_SIZE

"""
Configuration for backup propagation
"""
struct BackupConfig
    virtual_loss::Float32           # Virtual loss amount for selection consistency
    backup_batch_size::Int32        # Number of paths to backup in parallel
    max_path_length::Int32          # Maximum path length to trace
    convergence_threshold::Float32  # Threshold for detecting convergence
    damping_factor::Float32         # Damping factor for value updates (0-1)
    use_path_compression::Bool      # Enable path compression optimization
end

"""
Path information for backup
"""
struct BackupPath
    leaf_idx::Int32                 # Leaf node index
    path_length::Int32              # Length of path to root
    leaf_value::Float32             # Value at leaf node
    is_terminal::Bool               # Whether leaf is terminal
end

"""
Batch backup buffer for parallel processing
"""
mutable struct BackupBuffer
    # Path storage
    paths::CuArray{Int32, 2}        # [max_path_length, backup_batch_size]
    path_lengths::CuArray{Int32, 1} # [backup_batch_size]
    leaf_values::CuArray{Float32, 1} # [backup_batch_size]
    
    # Virtual loss tracking
    virtual_losses::CuArray{Float32, 1} # [MAX_NODES]
    
    # Convergence tracking
    value_deltas::CuArray{Float32, 1}   # [MAX_NODES] - tracks value changes
    convergence_flags::CuArray{Bool, 1}  # [MAX_NODES] - node convergence status
    
    # Statistics
    total_backups::CuArray{Int64, 1}
    avg_path_length::CuArray{Float32, 1}
    convergence_rate::CuArray{Float32, 1}
end

"""
Create a new backup buffer
"""
function BackupBuffer(config::BackupConfig)
    BackupBuffer(
        CUDA.zeros(Int32, config.max_path_length, config.backup_batch_size),
        CUDA.zeros(Int32, config.backup_batch_size),
        CUDA.zeros(Float32, config.backup_batch_size),
        CUDA.zeros(Float32, MAX_NODES),
        CUDA.zeros(Float32, MAX_NODES),
        CUDA.zeros(Bool, MAX_NODES),
        CUDA.zeros(Int64, 1),
        CUDA.zeros(Float32, 1),
        CUDA.zeros(Float32, 1)
    )
end

"""
Trace path from leaf to root kernel
"""
function trace_path_kernel!(
    parent_ids::CuDeviceArray{Int32, 1},
    paths::CuDeviceArray{Int32, 2},
    path_lengths::CuDeviceArray{Int32, 1},
    leaf_indices::CuDeviceArray{Int32, 1},
    batch_size::Int32,
    max_path_length::Int32
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= batch_size
        leaf_idx = leaf_indices[tid]
        
        if leaf_idx > 0 && leaf_idx <= MAX_NODES
            path_length = Int32(0)
            current = leaf_idx
            
            # Trace path from leaf to root
            while current > 0 && path_length < max_path_length
                paths[path_length + 1, tid] = current
                path_length += 1
                
                parent = parent_ids[current]
                if parent <= 0
                    break
                end
                current = parent
            end
            
            path_lengths[tid] = path_length
        else
            path_lengths[tid] = 0
        end
    end
    
    return nothing
end

"""
Apply virtual loss kernel - adds virtual loss during selection
"""
function apply_virtual_loss_kernel!(
    virtual_losses::CuDeviceArray{Float32, 1},
    visit_counts::CuDeviceArray{Int32, 1},
    paths::CuDeviceArray{Int32, 2},
    path_lengths::CuDeviceArray{Int32, 1},
    virtual_loss_amount::Float32,
    batch_size::Int32
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= batch_size
        path_length = path_lengths[tid]
        
        # Apply virtual loss to all nodes in path
        for i in 1:path_length
            node_idx = paths[i, tid]
            if node_idx > 0 && node_idx <= MAX_NODES
                # Add virtual loss
                CUDA.atomic_add!(pointer(virtual_losses, node_idx), virtual_loss_amount)
                # Increment virtual visit count
                CUDA.atomic_add!(pointer(visit_counts, node_idx), Int32(1))
            end
        end
    end
    
    return nothing
end

"""
Remove virtual loss kernel - removes virtual loss after evaluation
"""
function remove_virtual_loss_kernel!(
    virtual_losses::CuDeviceArray{Float32, 1},
    visit_counts::CuDeviceArray{Int32, 1},
    paths::CuDeviceArray{Int32, 2},
    path_lengths::CuDeviceArray{Int32, 1},
    virtual_loss_amount::Float32,
    batch_size::Int32
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= batch_size
        path_length = path_lengths[tid]
        
        # Remove virtual loss from all nodes in path
        for i in 1:path_length
            node_idx = paths[i, tid]
            if node_idx > 0 && node_idx <= MAX_NODES
                # Remove virtual loss
                CUDA.atomic_sub!(pointer(virtual_losses, node_idx), virtual_loss_amount)
                # Note: We don't decrement visit counts as the actual visit happened
            end
        end
    end
    
    return nothing
end

"""
Backup values kernel - propagates values from leaves to root
"""
function backup_values_kernel!(
    total_scores::CuDeviceArray{Float32, 1},
    visit_counts::CuDeviceArray{Int32, 1},
    value_deltas::CuDeviceArray{Float32, 1},
    paths::CuDeviceArray{Int32, 2},
    path_lengths::CuDeviceArray{Int32, 1},
    leaf_values::CuDeviceArray{Float32, 1},
    damping_factor::Float32,
    batch_size::Int32
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= batch_size
        path_length = path_lengths[tid]
        leaf_value = leaf_values[tid]
        
        # Propagate value up the path
        current_value = leaf_value
        
        for i in 1:path_length
            node_idx = paths[i, tid]
            if node_idx > 0 && node_idx <= MAX_NODES
                # Get old average value
                visits = visit_counts[node_idx]
                old_total = total_scores[node_idx]
                old_avg = visits > 0 ? old_total / Float32(visits) : 0.0f0
                
                # Apply damping to propagated value
                propagated_value = damping_factor * current_value + 
                                  (1.0f0 - damping_factor) * old_avg
                
                # Update total score
                CUDA.atomic_add!(pointer(total_scores, node_idx), propagated_value)
                
                # Track value change for convergence detection
                new_avg = (old_total + propagated_value) / Float32(visits + 1)
                delta = abs(new_avg - old_avg)
                
                # Atomic max using CAS (compare-and-swap) pattern
                old_delta = value_deltas[node_idx]
                while old_delta < delta
                    old_val = CUDA.atomic_cas!(pointer(value_deltas, node_idx), old_delta, delta)
                    if old_val == old_delta
                        break
                    end
                    old_delta = old_val
                end
                
                # Propagate value up (can apply decay here if needed)
                current_value = propagated_value
            end
        end
    end
    
    return nothing
end

"""
Path compression kernel - optimizes paths by skipping intermediate nodes
"""
function compress_paths_kernel!(
    parent_ids::CuDeviceArray{Int32, 1},
    visit_counts::CuDeviceArray{Int32, 1},
    paths::CuDeviceArray{Int32, 2},
    path_lengths::CuDeviceArray{Int32, 1},
    batch_size::Int32,
    compression_threshold::Int32
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= batch_size
        path_length = path_lengths[tid]
        
        if path_length > 2  # Need at least 3 nodes to compress
            # Check for compressible segments
            write_idx = 1
            
            for read_idx in 1:path_length
                node_idx = paths[read_idx, tid]
                
                if read_idx == 1 || read_idx == path_length
                    # Always keep leaf and root
                    paths[write_idx, tid] = node_idx
                    write_idx += 1
                else
                    # Check if node has low visit count (candidate for skipping)
                    visits = visit_counts[node_idx]
                    
                    if visits >= compression_threshold
                        # Keep important nodes
                        paths[write_idx, tid] = node_idx
                        write_idx += 1
                    end
                    # Otherwise skip this node
                end
            end
            
            # Update path length
            path_lengths[tid] = write_idx - 1
        end
    end
    
    return nothing
end

"""
Convergence detection kernel
"""
function detect_convergence_kernel!(
    convergence_flags::CuDeviceArray{Bool, 1},
    value_deltas::CuDeviceArray{Float32, 1},
    visit_counts::CuDeviceArray{Int32, 1},
    convergence_threshold::Float32,
    min_visits_for_convergence::Int32
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= MAX_NODES
        visits = visit_counts[tid]
        delta = value_deltas[tid]
        
        # Node is converged if:
        # 1. Has sufficient visits
        # 2. Value changes are below threshold
        is_converged = visits >= min_visits_for_convergence && 
                      delta < convergence_threshold
        
        convergence_flags[tid] = is_converged
        
        # Reset delta for next iteration
        value_deltas[tid] = 0.0f0
    end
    
    return nothing
end

"""
Batch backup statistics kernel
"""
function update_backup_stats_kernel!(
    total_backups::CuDeviceArray{Int64, 1},
    avg_path_length::CuDeviceArray{Float32, 1},
    path_lengths::CuDeviceArray{Int32, 1},
    batch_size::Int32
)
    # Shared memory for reduction
    shared_sum = @cuDynamicSharedMem(Int32, blockDim().x)
    shared_count = @cuDynamicSharedMem(Int32, blockDim().x, sizeof(Int32) * blockDim().x)
    
    tid = threadIdx().x
    bid = blockIdx().x
    gid = tid + (bid - 1) * blockDim().x
    
    # Load path length
    path_len = gid <= batch_size ? path_lengths[gid] : Int32(0)
    valid = gid <= batch_size && path_len > 0 ? Int32(1) : Int32(0)
    
    shared_sum[tid] = path_len
    shared_count[tid] = valid
    sync_threads()
    
    # Parallel reduction
    stride = blockDim().x ÷ 2
    while stride > 0
        if tid <= stride
            shared_sum[tid] += shared_sum[tid + stride]
            shared_count[tid] += shared_count[tid + stride]
        end
        sync_threads()
        stride ÷= 2
    end
    
    # Update global statistics
    if tid == 1
        if shared_count[1] > 0
            CUDA.atomic_add!(pointer(total_backups), Int64(shared_count[1]))
            
            # Update running average
            current_avg = avg_path_length[1]
            new_samples = Float32(shared_sum[1]) / Float32(shared_count[1])
            α = 0.1f0  # Exponential moving average factor
            avg_path_length[1] = α * new_samples + (1 - α) * current_avg
        end
    end
    
    return nothing
end

"""
Perform batch backup propagation
"""
function backup_batch!(
    buffer::BackupBuffer,
    config::BackupConfig,
    parent_ids::CuArray{Int32, 1},
    total_scores::CuArray{Float32, 1},
    visit_counts::CuArray{Int32, 1},
    leaf_indices::CuArray{Int32, 1},
    leaf_values::CuArray{Float32, 1},
    batch_size::Int32
)
    if batch_size == 0
        return
    end
    
    # Ensure batch size doesn't exceed buffer capacity
    actual_batch_size = min(batch_size, config.backup_batch_size)
    
    # Copy leaf values to buffer
    copyto!(buffer.leaf_values, 1, leaf_values, 1, actual_batch_size)
    
    # 1. Trace paths from leaves to root
    threads = 256
    blocks = cld(actual_batch_size, threads)
    
    @cuda threads=threads blocks=blocks trace_path_kernel!(
        parent_ids,
        buffer.paths,
        buffer.path_lengths,
        leaf_indices,
        actual_batch_size,
        config.max_path_length
    )
    
    # 2. Apply virtual loss (for selection consistency)
    @cuda threads=threads blocks=blocks apply_virtual_loss_kernel!(
        buffer.virtual_losses,
        visit_counts,
        buffer.paths,
        buffer.path_lengths,
        config.virtual_loss,
        actual_batch_size
    )
    
    # 3. Optional: Apply path compression
    if config.use_path_compression
        compression_threshold = Int32(10)  # Skip nodes with < 10 visits
        
        @cuda threads=threads blocks=blocks compress_paths_kernel!(
            parent_ids,
            visit_counts,
            buffer.paths,
            buffer.path_lengths,
            actual_batch_size,
            compression_threshold
        )
    end
    
    # 4. Backup values along paths
    @cuda threads=threads blocks=blocks backup_values_kernel!(
        total_scores,
        visit_counts,
        buffer.value_deltas,
        buffer.paths,
        buffer.path_lengths,
        buffer.leaf_values,
        config.damping_factor,
        actual_batch_size
    )
    
    # 5. Remove virtual loss
    @cuda threads=threads blocks=blocks remove_virtual_loss_kernel!(
        buffer.virtual_losses,
        visit_counts,
        buffer.paths,
        buffer.path_lengths,
        config.virtual_loss,
        actual_batch_size
    )
    
    # 6. Update statistics
    shmem = 2 * threads * sizeof(Int32)
    @cuda threads=threads blocks=blocks shmem=shmem update_backup_stats_kernel!(
        buffer.total_backups,
        buffer.avg_path_length,
        buffer.path_lengths,
        actual_batch_size
    )
    
    # 7. Check convergence periodically
    if CUDA.@allowscalar buffer.total_backups[1] % 1000 == 0
        min_visits = Int32(100)
        
        @cuda threads=1024 blocks=cld(MAX_NODES, 1024) detect_convergence_kernel!(
            buffer.convergence_flags,
            buffer.value_deltas,
            visit_counts,
            config.convergence_threshold,
            min_visits
        )
        
        # Calculate convergence rate
        CUDA.@allowscalar begin
            converged_count = sum(buffer.convergence_flags)
            total_expanded = sum(visit_counts .> 0)
            buffer.convergence_rate[1] = Float32(converged_count) / Float32(max(1, total_expanded))
        end
    end
end

"""
Get backup statistics
"""
function get_backup_stats(buffer::BackupBuffer)
    stats = Dict{String, Any}()
    
    CUDA.@allowscalar begin
        stats["total_backups"] = buffer.total_backups[1]
        stats["avg_path_length"] = buffer.avg_path_length[1]
        stats["convergence_rate"] = buffer.convergence_rate[1]
        
        # Count converged nodes
        converged_count = sum(buffer.convergence_flags)
        stats["converged_nodes"] = converged_count
        
        # Get max virtual loss (indicates contention)
        max_virtual_loss = maximum(buffer.virtual_losses)
        stats["max_virtual_loss"] = max_virtual_loss
    end
    
    return stats
end

"""
Reset convergence tracking
"""
function reset_convergence!(buffer::BackupBuffer)
    fill!(buffer.value_deltas, 0.0f0)
    fill!(buffer.convergence_flags, false)
    CUDA.@allowscalar buffer.convergence_rate[1] = 0.0f0
end

# Export types and functions
export BackupConfig, BackupPath, BackupBuffer
export backup_batch!, get_backup_stats, reset_convergence!

end # module BackupPropagation