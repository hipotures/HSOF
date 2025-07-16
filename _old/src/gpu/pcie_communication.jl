module PCIeCommunication

using CUDA
using Dates
using Statistics
using Serialization

export PCIeTransferManager, CandidateData, TransferBuffer, TransferStats
export create_transfer_manager, select_top_candidates, transfer_candidates
export enable_peer_access!, can_access_peer, get_transfer_stats
export reset_buffer!, should_transfer, compress_candidates, decompress_candidates
export add_candidates!, broadcast_candidates

"""
Data structure for feature candidates to transfer
"""
struct CandidateData
    feature_indices::Vector{Int32}  # Feature IDs (Int32 for smaller transfers)
    scores::Vector{Float32}         # Feature scores (Float32 for smaller transfers)
    source_gpu::Int
    iteration::Int
    timestamp::DateTime
    
    function CandidateData(indices::Vector{<:Integer}, scores::Vector{<:Real}, 
                          source_gpu::Int, iteration::Int)
        @assert length(indices) == length(scores) "Indices and scores must have same length"
        new(Int32.(indices), Float32.(scores), source_gpu, iteration, now())
    end
end

"""
Transfer buffer for batching candidates
"""
mutable struct TransferBuffer
    candidates::Vector{CandidateData}
    last_transfer_iteration::Int
    transfer_interval::Int  # Transfer every N iterations
    max_candidates::Int     # Maximum candidates to accumulate
    
    function TransferBuffer(;transfer_interval::Int = 1000, max_candidates::Int = 10)
        new(CandidateData[], 0, transfer_interval, max_candidates)
    end
end

"""
Statistics for transfer monitoring
"""
mutable struct TransferStats
    total_transfers::Int
    total_bytes::Int64
    total_time_ms::Float64
    last_transfer_size::Int64
    last_transfer_time_ms::Float64
    peer_access_enabled::Bool
    compression_ratio::Float64
    
    TransferStats() = new(0, 0, 0.0, 0, 0.0, false, 1.0)
end

"""
Main PCIe transfer manager
"""
mutable struct PCIeTransferManager
    gpu_pairs::Vector{Tuple{Int, Int}}  # (source_gpu, dest_gpu) pairs
    buffers::Dict{Tuple{Int, Int}, TransferBuffer}
    stats::Dict{Tuple{Int, Int}, TransferStats}
    peer_access_matrix::Matrix{Bool}
    use_compression::Bool
    
    function PCIeTransferManager(;
        num_gpus::Int = -1,
        use_compression::Bool = true,
        transfer_interval::Int = 1000
    )
        # Auto-detect GPUs if not specified
        if num_gpus == -1
            num_gpus = CUDA.functional() ? length(CUDA.devices()) : 0
        end
        
        # Create GPU pairs for bidirectional communication
        pairs = Tuple{Int, Int}[]
        for i in 0:(num_gpus-1)
            for j in 0:(num_gpus-1)
                if i != j
                    push!(pairs, (i, j))
                end
            end
        end
        
        # Initialize buffers and stats
        buffers = Dict{Tuple{Int, Int}, TransferBuffer}()
        stats = Dict{Tuple{Int, Int}, TransferStats}()
        for pair in pairs
            buffers[pair] = TransferBuffer(transfer_interval=transfer_interval)
            stats[pair] = TransferStats()
        end
        
        # Initialize peer access matrix
        peer_matrix = zeros(Bool, num_gpus, num_gpus)
        
        new(pairs, buffers, stats, peer_matrix, use_compression)
    end
end

"""
Create a PCIe transfer manager
"""
function create_transfer_manager(;kwargs...)
    manager = PCIeTransferManager(;kwargs...)
    
    # Try to enable peer access between all GPU pairs
    if CUDA.functional()
        for (src, dst) in manager.gpu_pairs
            if enable_peer_access!(manager, src, dst)
                @info "Enabled P2P access: GPU $src → GPU $dst"
            end
        end
    end
    
    return manager
end

"""
Enable peer-to-peer access between GPUs
"""
function enable_peer_access!(manager::PCIeTransferManager, src_gpu::Int, dst_gpu::Int)
    if !CUDA.functional() || src_gpu == dst_gpu
        return false
    end
    
    try
        # Get CUDA devices
        devices = collect(CUDA.devices())
        if src_gpu + 1 > length(devices) || dst_gpu + 1 > length(devices)
            return false
        end
        
        src_device = devices[src_gpu + 1]
        dst_device = devices[dst_gpu + 1]
        
        # Check if peer access is possible
        CUDA.device!(src_device) do
            if CUDA.can_access_peer(dst_device)
                # Enable peer access
                # Note: CUDA.jl handles peer access internally
                manager.peer_access_matrix[src_gpu + 1, dst_gpu + 1] = true
                if haskey(manager.stats, (src_gpu, dst_gpu))
                    manager.stats[(src_gpu, dst_gpu)].peer_access_enabled = true
                end
                return true
            end
        end
    catch e
        @warn "Failed to enable P2P access between GPU $src_gpu and $dst_gpu: $e"
    end
    
    return false
end

"""
Check if peer access is available between GPUs
"""
function can_access_peer(manager::PCIeTransferManager, src_gpu::Int, dst_gpu::Int)
    if src_gpu == dst_gpu || !CUDA.functional()
        return false
    end
    
    # Check bounds
    n_gpus = size(manager.peer_access_matrix, 1)
    if src_gpu + 1 > n_gpus || dst_gpu + 1 > n_gpus
        return false
    end
    
    return manager.peer_access_matrix[src_gpu + 1, dst_gpu + 1]
end

"""
Select top N candidates from ensemble results
"""
function select_top_candidates(
    feature_scores::Dict{Int, Float64},
    n_candidates::Int = 10;
    min_score_threshold::Float64 = 0.0
)
    # Filter by threshold
    filtered = [(idx, score) for (idx, score) in feature_scores if score >= min_score_threshold]
    
    # Sort by score (descending)
    sort!(filtered, by=x->x[2], rev=true)
    
    # Take top N
    top_n = min(n_candidates, length(filtered))
    indices = Int32[x[1] for x in filtered[1:top_n]]
    scores = Float32[x[2] for x in filtered[1:top_n]]
    
    return indices, scores
end

"""
Add candidates to transfer buffer
"""
function add_candidates!(
    manager::PCIeTransferManager,
    src_gpu::Int,
    dst_gpu::Int,
    candidates::CandidateData
)
    key = (src_gpu, dst_gpu)
    if haskey(manager.buffers, key)
        buffer = manager.buffers[key]
        push!(buffer.candidates, candidates)
        
        # Keep only the most recent candidates if buffer is full
        if length(buffer.candidates) > buffer.max_candidates
            # Sort by score and keep top candidates
            all_indices = Int32[]
            all_scores = Float32[]
            
            for cand in buffer.candidates
                append!(all_indices, cand.feature_indices)
                append!(all_scores, cand.scores)
            end
            
            # Get unique top candidates
            unique_features = Dict{Int32, Float32}()
            for (idx, score) in zip(all_indices, all_scores)
                if !haskey(unique_features, idx) || unique_features[idx] < score
                    unique_features[idx] = score
                end
            end
            
            # Select top N
            sorted_features = sort(collect(unique_features), by=x->x[2], rev=true)
            top_n = min(buffer.max_candidates, length(sorted_features))
            
            new_indices = Int32[x[1] for x in sorted_features[1:top_n]]
            new_scores = Float32[x[2] for x in sorted_features[1:top_n]]
            
            # Replace buffer with consolidated candidates
            empty!(buffer.candidates)
            push!(buffer.candidates, CandidateData(
                new_indices, new_scores, src_gpu, candidates.iteration
            ))
        end
    end
end

"""
Check if transfer should happen based on iteration count
"""
function should_transfer(manager::PCIeTransferManager, src_gpu::Int, dst_gpu::Int, 
                        current_iteration::Int)
    key = (src_gpu, dst_gpu)
    if !haskey(manager.buffers, key)
        return false
    end
    
    buffer = manager.buffers[key]
    return (current_iteration - buffer.last_transfer_iteration) >= buffer.transfer_interval
end

"""
Transfer candidates between GPUs
"""
function transfer_candidates(
    manager::PCIeTransferManager,
    src_gpu::Int,
    dst_gpu::Int,
    current_iteration::Int
) 
    key = (src_gpu, dst_gpu)
    if !haskey(manager.buffers, key) || !haskey(manager.stats, key)
        return nothing
    end
    
    buffer = manager.buffers[key]
    stats = manager.stats[key]
    
    # Check if we have candidates to transfer
    if isempty(buffer.candidates)
        return nothing
    end
    
    # Consolidate all candidates
    all_indices = Int32[]
    all_scores = Float32[]
    
    for cand in buffer.candidates
        append!(all_indices, cand.feature_indices)
        append!(all_scores, cand.scores)
    end
    
    # Create transfer data
    transfer_data = CandidateData(all_indices, all_scores, src_gpu, current_iteration)
    
    # Measure transfer time
    start_time = time()
    
    # Perform transfer
    received_data = if can_access_peer(manager, src_gpu, dst_gpu)
        # Use P2P transfer
        transfer_p2p(src_gpu, dst_gpu, transfer_data, manager.use_compression)
    else
        # Use CPU-mediated transfer
        transfer_via_cpu(src_gpu, dst_gpu, transfer_data, manager.use_compression)
    end
    
    transfer_time = (time() - start_time) * 1000  # Convert to ms
    
    # Update statistics  
    transfer_size = length(all_indices) * sizeof(Int32) + length(all_scores) * sizeof(Float32)
    stats.total_transfers += 1
    stats.total_bytes += transfer_size
    stats.total_time_ms += transfer_time
    stats.last_transfer_size = transfer_size
    stats.last_transfer_time_ms = transfer_time
    
    # Clear buffer and update iteration
    empty!(buffer.candidates)
    buffer.last_transfer_iteration = current_iteration
    
    return received_data
end

"""
Perform peer-to-peer GPU transfer
"""
function transfer_p2p(src_gpu::Int, dst_gpu::Int, data::CandidateData, use_compression::Bool)
    if !CUDA.functional()
        return transfer_via_cpu(src_gpu, dst_gpu, data, use_compression)
    end
    
    try
        devices = collect(CUDA.devices())
        src_device = devices[src_gpu + 1]
        dst_device = devices[dst_gpu + 1]
        
        # Compress if enabled
        compressed_data = use_compression ? compress_candidates(data) : data
        
        # Create GPU arrays on source
        CUDA.device!(src_device) do
            gpu_indices = CuArray(compressed_data.feature_indices)
            gpu_scores = CuArray(compressed_data.scores)
            
            # Transfer to destination GPU
            CUDA.device!(dst_device) do
                # Copy data (CUDA.jl handles P2P internally)
                dst_indices = copy(gpu_indices)
                dst_scores = copy(gpu_scores)
                
                # Copy back to host on destination
                host_indices = Array(dst_indices)
                host_scores = Array(dst_scores)
                
                # Decompress if needed
                received = CandidateData(
                    host_indices, host_scores, 
                    compressed_data.source_gpu, 
                    compressed_data.iteration
                )
                
                return use_compression ? decompress_candidates(received) : received
            end
        end
    catch e
        @warn "P2P transfer failed, falling back to CPU: $e"
        return transfer_via_cpu(src_gpu, dst_gpu, data, use_compression)
    end
end

"""
Perform CPU-mediated transfer (fallback)
"""
function transfer_via_cpu(src_gpu::Int, dst_gpu::Int, data::CandidateData, use_compression::Bool)
    # For CPU transfer, we just copy the data
    # In a real multi-GPU system, this would involve host memory copies
    
    # Simulate compression effect
    compressed = use_compression ? compress_candidates(data) : data
    
    # Create a copy (simulating transfer)
    received = CandidateData(
        copy(compressed.feature_indices),
        copy(compressed.scores),
        compressed.source_gpu,
        compressed.iteration
    )
    
    return use_compression ? decompress_candidates(received) : received
end

"""
Compress candidate data for smaller transfers
"""
function compress_candidates(data::CandidateData)
    # Simple compression: remove duplicates and keep highest scores
    unique_features = Dict{Int32, Float32}()
    
    for (idx, score) in zip(data.feature_indices, data.scores)
        if !haskey(unique_features, idx) || unique_features[idx] < score
            unique_features[idx] = score
        end
    end
    
    # Sort by score to maintain top candidates
    sorted = sort(collect(unique_features), by=x->x[2], rev=true)
    
    indices = Int32[x[1] for x in sorted]
    scores = Float32[x[2] for x in sorted]
    
    return CandidateData(indices, scores, data.source_gpu, data.iteration)
end

"""
Decompress candidate data (currently a no-op as compression removes duplicates)
"""
function decompress_candidates(data::CandidateData)
    return data
end

"""
Get transfer statistics
"""
function get_transfer_stats(manager::PCIeTransferManager)
    summary = Dict{String, Any}()
    summary["num_gpu_pairs"] = length(manager.gpu_pairs)
    summary["use_compression"] = manager.use_compression
    summary["peer_access_enabled"] = sum(manager.peer_access_matrix) > 0
    
    # Aggregate stats
    total_transfers = 0
    total_bytes = 0
    total_time = 0.0
    
    pair_stats = Dict{String, Any}()
    for (pair, stats) in manager.stats
        key = "GPU$(pair[1])→GPU$(pair[2])"
        pair_stats[key] = Dict(
            "transfers" => stats.total_transfers,
            "total_MB" => round(stats.total_bytes / 1024^2, digits=2),
            "avg_time_ms" => stats.total_transfers > 0 ? 
                          round(stats.total_time_ms / stats.total_transfers, digits=2) : 0.0,
            "bandwidth_MB/s" => stats.total_time_ms > 0 ?
                              round((stats.total_bytes / 1024^2) / (stats.total_time_ms / 1000), digits=2) : 0.0,
            "p2p_enabled" => stats.peer_access_enabled
        )
        
        total_transfers += stats.total_transfers
        total_bytes += stats.total_bytes
        total_time += stats.total_time_ms
    end
    
    summary["pair_stats"] = pair_stats
    summary["total_transfers"] = total_transfers
    summary["total_MB"] = round(total_bytes / 1024^2, digits=2)
    summary["avg_transfer_size_KB"] = total_transfers > 0 ? 
                                     round((total_bytes / total_transfers) / 1024, digits=2) : 0.0
    summary["avg_bandwidth_MB/s"] = total_time > 0 ?
                                   round((total_bytes / 1024^2) / (total_time / 1000), digits=2) : 0.0
    
    return summary
end

"""
Reset transfer buffer for a GPU pair
"""
function reset_buffer!(manager::PCIeTransferManager, src_gpu::Int, dst_gpu::Int)
    key = (src_gpu, dst_gpu)
    if haskey(manager.buffers, key)
        empty!(manager.buffers[key].candidates)
        manager.buffers[key].last_transfer_iteration = 0
    end
end

"""
Batch transfer to all destination GPUs
"""
function broadcast_candidates(
    manager::PCIeTransferManager,
    src_gpu::Int,
    candidates::CandidateData,
    current_iteration::Int
)
    received_data = Dict{Int, CandidateData}()
    
    # Transfer to all other GPUs
    for (src, dst) in manager.gpu_pairs
        if src == src_gpu
            add_candidates!(manager, src, dst, candidates)
            
            if should_transfer(manager, src, dst, current_iteration)
                result = transfer_candidates(manager, src, dst, current_iteration)
                if !isnothing(result)
                    received_data[dst] = result
                end
            end
        end
    end
    
    return received_data
end

end # module