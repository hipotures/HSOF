"""
PCIe Communication Module for MCTS Ensemble Feature Selection
Implements efficient inter-GPU communication system for sharing top candidates with minimal overhead,
including candidate serialization, PCIe transfer scheduling, asynchronous communication,
candidate merging logic, and transfer monitoring for dual RTX 4090 configuration.

This module provides high-performance GPU-to-GPU communication using PCIe for exchanging
top-performing feature sets between ensemble trees with bandwidth optimization.
"""

module PCIeCommunication

using Random
using Statistics
using Dates
using Printf
using LinearAlgebra
using Base.Threads
using Serialization

# Import CUDA for GPU operations
using CUDA

# Import other stage2 modules for integration
include("gpu_load_balancing.jl")
using .GPULoadBalancing

include("performance_monitoring.jl")
using .PerformanceMonitoring

"""
Communication strategy types for PCIe transfers
"""
@enum CommunicationStrategy begin
    ROUND_ROBIN_TRANSFER = 1      # Round-robin between GPUs
    PRIORITY_BASED_TRANSFER = 2   # Based on candidate quality
    LOAD_BALANCED_TRANSFER = 3    # Based on GPU utilization
    ADAPTIVE_TRANSFER = 4         # Dynamic strategy selection
    BANDWIDTH_OPTIMIZED = 5       # Minimize bandwidth usage
end

"""
Transfer scheduling types
"""
@enum TransferSchedule begin
    FIXED_INTERVAL = 1            # Fixed iteration intervals
    ADAPTIVE_INTERVAL = 2         # Based on convergence rate
    PERFORMANCE_BASED = 3         # Based on improvement metrics
    LOAD_BASED_SCHEDULE = 4       # Based on GPU load
    HYBRID_SCHEDULE = 5           # Combination of strategies
end

"""
Serialization format types
"""
@enum SerializationFormat begin
    COMPACT_BINARY = 1            # Minimal binary format
    COMPRESSED_BINARY = 2         # Compressed binary
    EFFICIENT_MSGPACK = 3         # MessagePack format
    CUSTOM_PROTOCOL = 4           # Custom optimization
end

"""
Feature set candidate for communication
"""
struct FeatureCandidate
    feature_indices::Vector{Int}           # Selected feature indices
    performance_score::Float32             # Candidate performance
    confidence_score::Float32              # Selection confidence
    iteration_discovered::Int              # Discovery iteration
    tree_id::Int                          # Source tree identifier
    gpu_id::Int                           # Source GPU identifier
    validation_score::Float32             # Cross-validation score
    stability_metric::Float32             # Feature stability
    selection_count::Int                  # Times selected
    last_updated::DateTime                # Last update timestamp
end

"""
Create feature candidate
"""
function create_feature_candidate(feature_indices::Vector{Int}, performance_score::Float32,
                                 tree_id::Int, gpu_id::Int;
                                 confidence_score::Float32 = 0.8f0,
                                 iteration_discovered::Int = 0,
                                 validation_score::Float32 = 0.0f0,
                                 stability_metric::Float32 = 0.0f0,
                                 selection_count::Int = 1)
    return FeatureCandidate(
        feature_indices, performance_score, confidence_score,
        iteration_discovered, tree_id, gpu_id, validation_score,
        stability_metric, selection_count, now()
    )
end

"""
Serialized candidate data for efficient transfer
"""
struct SerializedCandidate
    data::Vector{UInt8}                   # Serialized binary data
    checksum::UInt32                      # Data integrity checksum
    size_bytes::Int                       # Size in bytes
    compression_ratio::Float32            # Compression efficiency
    serialization_time::Float64          # Time to serialize (ms)
    format::SerializationFormat          # Serialization format used
end

"""
PCIe transfer metadata
"""
mutable struct TransferMetadata
    transfer_id::String                   # Unique transfer identifier
    source_gpu::Int                       # Source GPU device
    target_gpu::Int                       # Target GPU device
    candidate_count::Int                  # Number of candidates
    total_size_bytes::Int                 # Total transfer size
    bandwidth_used_mbps::Float64         # Bandwidth utilization
    transfer_start_time::DateTime        # Transfer start
    transfer_end_time::Union{DateTime, Nothing}  # Transfer completion
    transfer_duration_ms::Float64        # Transfer time
    success::Bool                        # Transfer success status
    error_message::Union{String, Nothing}  # Error details if any
    retry_count::Int                     # Number of retries
    queue_time_ms::Float64              # Time spent in queue
end

"""
Create transfer metadata
"""
function create_transfer_metadata(source_gpu::Int, target_gpu::Int, candidate_count::Int)
    transfer_id = string(hash((source_gpu, target_gpu, now(), candidate_count)))
    return TransferMetadata(
        transfer_id, source_gpu, target_gpu, candidate_count, 0, 0.0,
        now(), nothing, 0.0, false, nothing, 0, 0.0
    )
end

"""
PCIe communication configuration
"""
struct PCIeConfig
    # Transfer scheduling
    communication_strategy::CommunicationStrategy
    transfer_schedule::TransferSchedule
    base_transfer_interval::Int           # Base iterations between transfers
    adaptive_interval_range::Tuple{Int, Int}  # Min/max adaptive intervals
    
    # Candidate management
    max_candidates_per_transfer::Int      # Maximum candidates per transfer
    candidate_quality_threshold::Float32  # Minimum quality threshold
    candidate_age_limit_iterations::Int   # Maximum candidate age
    duplicate_detection_enabled::Bool     # Enable duplicate filtering
    
    # Serialization settings
    serialization_format::SerializationFormat
    enable_compression::Bool              # Enable data compression
    compression_level::Int               # Compression level (1-9)
    checksum_validation::Bool            # Enable checksum validation
    
    # Bandwidth management
    max_bandwidth_mb_per_sec::Float64    # Maximum bandwidth limit
    bandwidth_monitoring_enabled::Bool   # Enable bandwidth monitoring
    transfer_queue_size::Int             # Maximum queued transfers
    priority_queue_enabled::Bool         # Enable priority queuing
    
    # Error handling and reliability
    max_retry_attempts::Int              # Maximum transfer retries
    retry_backoff_ms::Int               # Retry delay multiplier
    timeout_ms::Int                     # Transfer timeout
    enable_fault_tolerance::Bool        # Enable fault tolerance
    
    # Performance optimization
    async_transfer_enabled::Bool         # Enable asynchronous transfers
    batching_enabled::Bool              # Enable transfer batching
    prefetch_enabled::Bool              # Enable candidate prefetching
    cache_enabled::Bool                 # Enable transfer caching
    
    # Monitoring and logging
    detailed_logging_enabled::Bool       # Enable detailed logging
    performance_metrics_enabled::Bool   # Enable performance tracking
    transfer_history_size::Int          # Number of transfers to track
    bandwidth_history_size::Int         # Bandwidth measurements to keep
end

"""
Create PCIe communication configuration
"""
function create_pcie_config(;
    communication_strategy::CommunicationStrategy = LOAD_BALANCED_TRANSFER,
    transfer_schedule::TransferSchedule = ADAPTIVE_INTERVAL,
    base_transfer_interval::Int = 1000,
    adaptive_interval_range::Tuple{Int, Int} = (500, 2000),
    max_candidates_per_transfer::Int = 10,
    candidate_quality_threshold::Float32 = 0.7f0,
    candidate_age_limit_iterations::Int = 5000,
    duplicate_detection_enabled::Bool = true,
    serialization_format::SerializationFormat = COMPACT_BINARY,
    enable_compression::Bool = true,
    compression_level::Int = 6,
    checksum_validation::Bool = true,
    max_bandwidth_mb_per_sec::Float64 = 100.0,
    bandwidth_monitoring_enabled::Bool = true,
    transfer_queue_size::Int = 16,
    priority_queue_enabled::Bool = true,
    max_retry_attempts::Int = 3,
    retry_backoff_ms::Int = 100,
    timeout_ms::Int = 5000,
    enable_fault_tolerance::Bool = true,
    async_transfer_enabled::Bool = true,
    batching_enabled::Bool = true,
    prefetch_enabled::Bool = false,
    cache_enabled::Bool = true,
    detailed_logging_enabled::Bool = false,
    performance_metrics_enabled::Bool = true,
    transfer_history_size::Int = 1000,
    bandwidth_history_size::Int = 100
)
    return PCIeConfig(
        communication_strategy, transfer_schedule, base_transfer_interval, adaptive_interval_range,
        max_candidates_per_transfer, candidate_quality_threshold, candidate_age_limit_iterations, duplicate_detection_enabled,
        serialization_format, enable_compression, compression_level, checksum_validation,
        max_bandwidth_mb_per_sec, bandwidth_monitoring_enabled, transfer_queue_size, priority_queue_enabled,
        max_retry_attempts, retry_backoff_ms, timeout_ms, enable_fault_tolerance,
        async_transfer_enabled, batching_enabled, prefetch_enabled, cache_enabled,
        detailed_logging_enabled, performance_metrics_enabled, transfer_history_size, bandwidth_history_size
    )
end

"""
PCIe communication statistics
"""
mutable struct PCIeStats
    total_transfers::Int
    successful_transfers::Int
    failed_transfers::Int
    total_candidates_transferred::Int
    total_bytes_transferred::Int
    average_transfer_time_ms::Float64
    peak_bandwidth_mbps::Float64
    current_bandwidth_mbps::Float64
    queue_overflows::Int
    retry_events::Int
    compression_savings_percentage::Float64
    duplicate_candidates_filtered::Int
    bandwidth_violations::Int
    fault_tolerance_activations::Int
    cache_hit_rate::Float64
    last_transfer_time::DateTime
end

"""
Initialize PCIe communication statistics
"""
function initialize_pcie_stats()
    return PCIeStats(
        0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0, 0, 0, 0.0, now()
    )
end

"""
Candidate queue for managing transfer candidates
"""
mutable struct CandidateQueue
    candidates::Vector{FeatureCandidate}
    max_size::Int
    priority_sorted::Bool
    last_cleanup::DateTime
    insert_count::Int
    pop_count::Int
    overflow_count::Int
end

"""
Create candidate queue
"""
function create_candidate_queue(max_size::Int = 100; priority_sorted::Bool = true)
    return CandidateQueue(
        FeatureCandidate[], max_size, priority_sorted, now(), 0, 0, 0
    )
end

"""
PCIe communication manager
"""
mutable struct PCIeCommunicationManager
    config::PCIeConfig
    stats::PCIeStats
    
    # GPU device management
    gpu_devices::Vector{Int}              # Available GPU device IDs
    gpu_contexts::Dict{Int, Any}          # GPU contexts (placeholder for CUDA contexts)
    peer_access_enabled::Dict{Tuple{Int, Int}, Bool}  # P2P access status
    
    # Candidate management
    candidate_queues::Dict{Int, CandidateQueue}       # Per-GPU candidate queues
    candidate_cache::Dict{String, FeatureCandidate}   # Candidate cache
    candidate_history::Vector{FeatureCandidate}       # Transfer history
    
    # Transfer management
    transfer_queue::Vector{TransferMetadata}          # Pending transfers
    active_transfers::Dict{String, TransferMetadata}  # Active transfers
    transfer_history::Vector{TransferMetadata}        # Completed transfers
    bandwidth_history::Vector{Tuple{DateTime, Float64}}  # Bandwidth measurements
    
    # Scheduling and timing
    last_transfer_iteration::Dict{Tuple{Int, Int}, Int}  # Last transfer per GPU pair
    next_transfer_iteration::Dict{Tuple{Int, Int}, Int}  # Scheduled next transfer
    iteration_counter::Int                            # Current iteration
    
    # Synchronization
    communication_lock::ReentrantLock
    transfer_tasks::Vector{Task}          # Async transfer tasks
    
    # Status and error handling
    manager_state::String
    error_log::Vector{String}
    is_running::Bool
    fault_recovery_active::Bool
end

"""
Initialize PCIe communication manager
"""
function initialize_pcie_communication_manager(config::PCIeConfig = create_pcie_config(),
                                              gpu_devices::Vector{Int} = [0, 1])
    # Initialize candidate queues for each GPU
    candidate_queues = Dict{Int, CandidateQueue}()
    for gpu_id in gpu_devices
        candidate_queues[gpu_id] = create_candidate_queue(config.transfer_queue_size * 2)
    end
    
    # Initialize GPU contexts (placeholder - would use actual CUDA contexts)
    gpu_contexts = Dict{Int, Any}()
    for gpu_id in gpu_devices
        gpu_contexts[gpu_id] = nothing  # Placeholder for CUDA context
    end
    
    # Initialize peer access mapping
    peer_access = Dict{Tuple{Int, Int}, Bool}()
    for src in gpu_devices, dst in gpu_devices
        if src != dst
            peer_access[(src, dst)] = false  # Will be detected during initialization
        end
    end
    
    # Initialize timing dictionaries
    last_transfer = Dict{Tuple{Int, Int}, Int}()
    next_transfer = Dict{Tuple{Int, Int}, Int}()
    for src in gpu_devices, dst in gpu_devices
        if src != dst
            last_transfer[(src, dst)] = 0
            next_transfer[(src, dst)] = 0  # Allow first transfer immediately
        end
    end
    
    manager = PCIeCommunicationManager(
        config,
        initialize_pcie_stats(),
        gpu_devices,
        gpu_contexts,
        peer_access,
        candidate_queues,
        Dict{String, FeatureCandidate}(),
        FeatureCandidate[],
        TransferMetadata[],
        Dict{String, TransferMetadata}(),
        TransferMetadata[],
        Tuple{DateTime, Float64}[],
        last_transfer,
        next_transfer,
        0,
        ReentrantLock(),
        Task[],
        "initialized",
        String[],
        false,
        false
    )
    
    @info "PCIe communication manager initialized with $(length(gpu_devices)) GPU devices"
    return manager
end

"""
Serialize feature candidate for efficient transfer
"""
function serialize_candidate(candidate::FeatureCandidate, format::SerializationFormat)::SerializedCandidate
    start_time = time()
    
    if format == COMPACT_BINARY
        # Create compact binary representation
        data = Vector{UInt8}()
        
        # Pack feature indices (variable length encoding)
        append!(data, reinterpret(UInt8, [length(candidate.feature_indices)]))
        for idx in candidate.feature_indices
            append!(data, reinterpret(UInt8, [idx]))
        end
        
        # Pack performance metrics
        append!(data, reinterpret(UInt8, [candidate.performance_score]))
        append!(data, reinterpret(UInt8, [candidate.confidence_score]))
        append!(data, reinterpret(UInt8, [candidate.iteration_discovered]))
        append!(data, reinterpret(UInt8, [candidate.tree_id]))
        append!(data, reinterpret(UInt8, [candidate.gpu_id]))
        append!(data, reinterpret(UInt8, [candidate.validation_score]))
        append!(data, reinterpret(UInt8, [candidate.stability_metric]))
        append!(data, reinterpret(UInt8, [candidate.selection_count]))
        
        # Pack timestamp (as Unix timestamp)
        timestamp = Dates.datetime2unix(candidate.last_updated)
        append!(data, reinterpret(UInt8, [timestamp]))
        
    elseif format == COMPRESSED_BINARY
        # First create compact binary, then compress
        compact_data = serialize_candidate(candidate, COMPACT_BINARY).data
        # Simulated compression (would use actual compression library)
        data = compact_data[1:max(1, length(compact_data) ÷ 2)]  # Simulate 50% compression
        
    else
        # Fallback to Julia serialization
        io = IOBuffer()
        serialize(io, candidate)
        data = take!(io)
    end
    
    # Calculate checksum
    checksum = sum(data) % UInt32(2^32 - 1)
    
    # Calculate compression ratio
    original_size = sizeof(candidate.feature_indices) * length(candidate.feature_indices) + 
                   sizeof(Float32) * 4 + sizeof(Int) * 4 + 8  # Rough estimate
    compression_ratio = Float32(length(data) / original_size)
    
    serialization_time = (time() - start_time) * 1000  # Convert to milliseconds
    
    return SerializedCandidate(
        data, checksum, length(data), compression_ratio, serialization_time, format
    )
end

"""
Deserialize feature candidate from binary data
"""
function deserialize_candidate(serialized::SerializedCandidate)::Union{FeatureCandidate, Nothing}
    try
        data = serialized.data
        
        if serialized.format == COMPACT_BINARY
            offset = 1
            
            # Unpack feature indices
            num_features = reinterpret(Int, data[offset:offset+sizeof(Int)-1])[1]
            offset += sizeof(Int)
            
            feature_indices = Vector{Int}(undef, num_features)
            for i in 1:num_features
                feature_indices[i] = reinterpret(Int, data[offset:offset+sizeof(Int)-1])[1]
                offset += sizeof(Int)
            end
            
            # Unpack performance metrics
            performance_score = reinterpret(Float32, data[offset:offset+sizeof(Float32)-1])[1]
            offset += sizeof(Float32)
            
            confidence_score = reinterpret(Float32, data[offset:offset+sizeof(Float32)-1])[1]
            offset += sizeof(Float32)
            
            iteration_discovered = reinterpret(Int, data[offset:offset+sizeof(Int)-1])[1]
            offset += sizeof(Int)
            
            tree_id = reinterpret(Int, data[offset:offset+sizeof(Int)-1])[1]
            offset += sizeof(Int)
            
            gpu_id = reinterpret(Int, data[offset:offset+sizeof(Int)-1])[1]
            offset += sizeof(Int)
            
            validation_score = reinterpret(Float32, data[offset:offset+sizeof(Float32)-1])[1]
            offset += sizeof(Float32)
            
            stability_metric = reinterpret(Float32, data[offset:offset+sizeof(Float32)-1])[1]
            offset += sizeof(Float32)
            
            selection_count = reinterpret(Int, data[offset:offset+sizeof(Int)-1])[1]
            offset += sizeof(Int)
            
            # Unpack timestamp
            timestamp = reinterpret(Float64, data[offset:offset+sizeof(Float64)-1])[1]
            last_updated = Dates.unix2datetime(timestamp)
            
            return FeatureCandidate(
                feature_indices, performance_score, confidence_score,
                iteration_discovered, tree_id, gpu_id, validation_score,
                stability_metric, selection_count, last_updated
            )
            
        else
            # Fallback to Julia deserialization
            io = IOBuffer(data)
            return deserialize(io)
        end
        
    catch e
        @error "Failed to deserialize candidate: $e"
        return nothing
    end
end

"""
Add candidate to queue for transfer
"""
function add_candidate_to_queue!(manager::PCIeCommunicationManager, gpu_id::Int, 
                                candidate::FeatureCandidate)::Bool
    if !haskey(manager.candidate_queues, gpu_id)
        return false
    end
    
    queue = manager.candidate_queues[gpu_id]
    
    # Check for duplicates if enabled
    if manager.config.duplicate_detection_enabled
        candidate_signature = hash(candidate.feature_indices)  # Only check feature indices for duplicates
        if haskey(manager.candidate_cache, string(candidate_signature))
            manager.stats.duplicate_candidates_filtered += 1
            return false
        end
        manager.candidate_cache[string(candidate_signature)] = candidate
    end
    
    # Add candidate to queue
    push!(queue.candidates, candidate)
    queue.insert_count += 1
    
    # Maintain queue size limit
    if length(queue.candidates) > queue.max_size
        # Remove oldest/lowest quality candidate
        if queue.priority_sorted
            sort!(queue.candidates, by = c -> c.performance_score, rev = true)
            pop!(queue.candidates)  # Remove lowest quality (last in sorted list)
        else
            popfirst!(queue.candidates)  # Remove oldest
        end
        queue.overflow_count += 1
        manager.stats.queue_overflows += 1
    end
    
    # Sort by priority if enabled
    if queue.priority_sorted
        sort!(queue.candidates, by = c -> c.performance_score, rev = true)
    end
    
    return true
end

"""
Get top candidates from queue for transfer
"""
function get_top_candidates_for_transfer(manager::PCIeCommunicationManager, gpu_id::Int,
                                       max_count::Int)::Vector{FeatureCandidate}
    if !haskey(manager.candidate_queues, gpu_id)
        return FeatureCandidate[]
    end
    
    queue = manager.candidate_queues[gpu_id]
    candidates = FeatureCandidate[]
    
    # Filter candidates by quality threshold and age
    current_iteration = manager.iteration_counter
    
    for candidate in queue.candidates
        # Check quality threshold
        if candidate.performance_score < manager.config.candidate_quality_threshold
            continue
        end
        
        # Check age limit
        age = current_iteration - candidate.iteration_discovered
        if age > manager.config.candidate_age_limit_iterations
            continue
        end
        
        push!(candidates, candidate)
        
        # Stop when we have enough candidates
        if length(candidates) >= max_count
            break
        end
    end
    
    # Remove transferred candidates from queue
    for candidate in candidates
        idx = findfirst(c -> c === candidate, queue.candidates)
        if !isnothing(idx)
            deleteat!(queue.candidates, idx)
            queue.pop_count += 1
        end
    end
    
    return candidates
end

"""
Calculate current bandwidth utilization
"""
function calculate_current_bandwidth(manager::PCIeCommunicationManager)::Float64
    if length(manager.bandwidth_history) < 2
        return 0.0
    end
    
    # Calculate bandwidth over last few measurements
    recent_history = manager.bandwidth_history[max(1, end-10):end]
    if length(recent_history) < 2
        return 0.0
    end
    
    total_bytes = 0.0
    time_span = Dates.value(recent_history[end][1] - recent_history[1][1]) / 1000.0  # seconds
    
    for (_, bytes) in recent_history
        total_bytes += bytes
    end
    
    return time_span > 0 ? (total_bytes / (1024 * 1024)) / time_span : 0.0  # MB/s
end

"""
Check if transfer should be scheduled
"""
function should_schedule_transfer(manager::PCIeCommunicationManager, source_gpu::Int, 
                                target_gpu::Int)::Bool
    # Validate GPU IDs
    if !in(source_gpu, manager.gpu_devices) || !in(target_gpu, manager.gpu_devices)
        return false
    end
    
    current_iteration = manager.iteration_counter
    gpu_pair = (source_gpu, target_gpu)
    
    # Check if enough iterations have passed
    if !haskey(manager.next_transfer_iteration, gpu_pair)
        return true
    end
    
    if current_iteration < manager.next_transfer_iteration[gpu_pair]
        return false
    end
    
    # Check bandwidth constraints
    current_bandwidth = calculate_current_bandwidth(manager)
    if current_bandwidth > manager.config.max_bandwidth_mb_per_sec * 0.8  # 80% threshold
        return false
    end
    
    # Check queue capacity
    if length(manager.transfer_queue) >= manager.config.transfer_queue_size
        return false
    end
    
    # Check if source GPU has candidates
    if !haskey(manager.candidate_queues, source_gpu)
        return false
    end
    
    queue = manager.candidate_queues[source_gpu]
    available_candidates = count(c -> c.performance_score >= manager.config.candidate_quality_threshold, 
                                queue.candidates)
    
    return available_candidates > 0
end

"""
Schedule transfer between GPUs
"""
function schedule_transfer!(manager::PCIeCommunicationManager, source_gpu::Int, target_gpu::Int)::Bool
    if !should_schedule_transfer(manager, source_gpu, target_gpu)
        return false
    end
    
    # Get candidates for transfer
    candidates = get_top_candidates_for_transfer(
        manager, source_gpu, manager.config.max_candidates_per_transfer
    )
    
    if isempty(candidates)
        return false
    end
    
    # Create transfer metadata
    metadata = create_transfer_metadata(source_gpu, target_gpu, length(candidates))
    
    # Calculate transfer size
    total_size = 0
    for candidate in candidates
        serialized = serialize_candidate(candidate, manager.config.serialization_format)
        total_size += serialized.size_bytes
    end
    metadata.total_size_bytes = total_size
    
    # Add to transfer queue
    push!(manager.transfer_queue, metadata)
    
    # Update scheduling
    gpu_pair = (source_gpu, target_gpu)
    manager.last_transfer_iteration[gpu_pair] = manager.iteration_counter
    
    # Calculate next transfer interval (adaptive scheduling)
    next_interval = calculate_next_transfer_interval(manager, source_gpu, target_gpu)
    manager.next_transfer_iteration[gpu_pair] = manager.iteration_counter + next_interval
    
    @debug "Scheduled transfer from GPU $source_gpu to GPU $target_gpu with $(length(candidates)) candidates"
    
    return true
end

"""
Calculate next transfer interval based on scheduling strategy
"""
function calculate_next_transfer_interval(manager::PCIeCommunicationManager, 
                                        source_gpu::Int, target_gpu::Int)::Int
    base_interval = manager.config.base_transfer_interval
    
    if manager.config.transfer_schedule == FIXED_INTERVAL
        return base_interval
        
    elseif manager.config.transfer_schedule == ADAPTIVE_INTERVAL
        # Adapt based on queue sizes and performance
        source_queue_size = length(manager.candidate_queues[source_gpu].candidates)
        max_queue_size = manager.candidate_queues[source_gpu].max_size
        
        # More frequent transfers when queue is full
        queue_factor = source_queue_size / max_queue_size
        
        min_interval, max_interval = manager.config.adaptive_interval_range
        interval = base_interval - Int(round((base_interval - min_interval) * queue_factor))
        
        return clamp(interval, min_interval, max_interval)
        
    elseif manager.config.transfer_schedule == PERFORMANCE_BASED
        # Adapt based on recent candidate quality
        if !isempty(manager.candidate_history)
            recent_performance = mean(c.performance_score for c in manager.candidate_history[max(1, end-10):end])
            # More frequent transfers for high-quality candidates
            performance_factor = recent_performance  # 0.0 to 1.0
            interval = base_interval - Int(round(base_interval * 0.5 * performance_factor))
            return max(interval, manager.config.adaptive_interval_range[1])
        end
        
    elseif manager.config.transfer_schedule == LOAD_BASED_SCHEDULE
        # Adapt based on GPU load (would need load balancer integration)
        return base_interval
        
    else  # HYBRID_SCHEDULE
        # Combine multiple factors
        adaptive_interval = calculate_next_transfer_interval(manager, source_gpu, target_gpu)
        return adaptive_interval
    end
    
    return base_interval
end

"""
Execute transfer between GPUs
"""
function execute_transfer!(manager::PCIeCommunicationManager, metadata::TransferMetadata)::Bool
    lock(manager.communication_lock) do
        try
            metadata.transfer_start_time = now()
            manager.active_transfers[metadata.transfer_id] = metadata
            
            # Get candidates for this transfer
            candidates = get_top_candidates_for_transfer(
                manager, metadata.source_gpu, metadata.candidate_count
            )
            
            if isempty(candidates)
                metadata.success = false
                metadata.error_message = "No candidates available for transfer"
                metadata.transfer_end_time = now()
                manager.stats.failed_transfers += 1
                delete!(manager.active_transfers, metadata.transfer_id)
                push!(manager.transfer_history, metadata)
                return false
            end
            
            # Serialize candidates
            serialized_candidates = SerializedCandidate[]
            total_bytes = 0
            
            for candidate in candidates
                serialized = serialize_candidate(candidate, manager.config.serialization_format)
                push!(serialized_candidates, serialized)
                total_bytes += serialized.size_bytes
            end
            
            metadata.total_size_bytes = total_bytes
            
            # Simulate PCIe transfer (in real implementation, would use CUDA P2P)
            transfer_start = time()
            
            # Simulate bandwidth-limited transfer
            simulated_bandwidth_mbps = min(manager.config.max_bandwidth_mb_per_sec, 
                                         800.0)  # PCIe 3.0 x16 theoretical bandwidth
            transfer_time_sec = (total_bytes / (1024 * 1024)) / simulated_bandwidth_mbps
            
            # Simulate transfer delay
            sleep(max(0.001, transfer_time_sec))  # Minimum 1ms delay
            
            transfer_end = time()
            
            # Update metadata
            metadata.transfer_end_time = now()
            metadata.transfer_duration_ms = (transfer_end - transfer_start) * 1000
            metadata.bandwidth_used_mbps = (total_bytes / (1024 * 1024)) / max(0.001, transfer_time_sec)
            metadata.success = true
            
            # Merge candidates into target GPU
            merge_transferred_candidates!(manager, metadata.target_gpu, candidates)
            
            # Update statistics
            manager.stats.total_transfers += 1
            manager.stats.successful_transfers += 1
            manager.stats.total_candidates_transferred += length(candidates)
            manager.stats.total_bytes_transferred += total_bytes
            manager.stats.current_bandwidth_mbps = metadata.bandwidth_used_mbps
            manager.stats.peak_bandwidth_mbps = max(manager.stats.peak_bandwidth_mbps, 
                                                   metadata.bandwidth_used_mbps)
            manager.stats.last_transfer_time = now()
            
            # Update bandwidth history
            push!(manager.bandwidth_history, (now(), Float64(total_bytes)))
            if length(manager.bandwidth_history) > manager.config.bandwidth_history_size
                deleteat!(manager.bandwidth_history, 1)
            end
            
            # Clean up
            delete!(manager.active_transfers, metadata.transfer_id)
            push!(manager.transfer_history, metadata)
            if length(manager.transfer_history) > manager.config.transfer_history_size
                deleteat!(manager.transfer_history, 1)
            end
            
            if manager.config.detailed_logging_enabled
                @info "Transfer $(metadata.transfer_id) completed: $(length(candidates)) candidates, $(total_bytes) bytes, $(round(metadata.bandwidth_used_mbps, digits=2)) MB/s"
            end
            
            return true
            
        catch e
            metadata.success = false
            metadata.error_message = string(e)
            metadata.transfer_end_time = now()
            
            manager.stats.failed_transfers += 1
            delete!(manager.active_transfers, metadata.transfer_id)
            push!(manager.transfer_history, metadata)
            
            @error "Transfer $(metadata.transfer_id) failed: $e"
            push!(manager.error_log, "Transfer $(metadata.transfer_id) failed: $e")
            
            return false
        end
    end
end

"""
Merge transferred candidates into target GPU queue
"""
function merge_transferred_candidates!(manager::PCIeCommunicationManager, target_gpu::Int,
                                     candidates::Vector{FeatureCandidate})
    if !haskey(manager.candidate_queues, target_gpu)
        return
    end
    
    target_queue = manager.candidate_queues[target_gpu]
    merged_count = 0
    
    for candidate in candidates
        # Check for duplicates in target queue
        duplicate_found = false
        if manager.config.duplicate_detection_enabled
            for existing in target_queue.candidates
                if existing.feature_indices == candidate.feature_indices
                    # Update existing candidate if new one is better
                    if candidate.performance_score > existing.performance_score
                        idx = findfirst(c -> c === existing, target_queue.candidates)
                        if !isnothing(idx)
                            target_queue.candidates[idx] = candidate
                            merged_count += 1
                        end
                    end
                    duplicate_found = true
                    break
                end
            end
        end
        
        if !duplicate_found
            # Add as new candidate
            push!(target_queue.candidates, candidate)
            merged_count += 1
            
            # Maintain queue size
            if length(target_queue.candidates) > target_queue.max_size
                if target_queue.priority_sorted
                    sort!(target_queue.candidates, by = c -> c.performance_score, rev = true)
                    pop!(target_queue.candidates)
                else
                    popfirst!(target_queue.candidates)
                end
            end
        end
    end
    
    # Sort queue by priority
    if target_queue.priority_sorted && merged_count > 0
        sort!(target_queue.candidates, by = c -> c.performance_score, rev = true)
    end
    
    # Add to candidate history
    append!(manager.candidate_history, candidates)
    if length(manager.candidate_history) > manager.config.transfer_history_size
        manager.candidate_history = manager.candidate_history[end-manager.config.transfer_history_size+1:end]
    end
    
    @debug "Merged $merged_count candidates into GPU $target_gpu queue"
end

"""
Process transfer queue (run transfers)
"""
function process_transfer_queue!(manager::PCIeCommunicationManager)
    if isempty(manager.transfer_queue)
        return
    end
    
    # Check bandwidth limits
    current_bandwidth = calculate_current_bandwidth(manager)
    if current_bandwidth > manager.config.max_bandwidth_mb_per_sec
        manager.stats.bandwidth_violations += 1
        return
    end
    
    # Process transfers based on priority
    if manager.config.priority_queue_enabled
        sort!(manager.transfer_queue, by = t -> t.candidate_count, rev = true)
    end
    
    # Execute transfers (limit concurrent transfers)
    max_concurrent = min(2, length(manager.gpu_devices))
    active_count = length(manager.active_transfers)
    
    while !isempty(manager.transfer_queue) && active_count < max_concurrent
        metadata = popfirst!(manager.transfer_queue)
        
        if manager.config.async_transfer_enabled
            # Execute asynchronously
            task = @async execute_transfer!(manager, metadata)
            push!(manager.transfer_tasks, task)
            
            # Clean up completed tasks
            filter!(t -> !istaskdone(t), manager.transfer_tasks)
        else
            # Execute synchronously
            execute_transfer!(manager, metadata)
        end
        
        active_count += 1
    end
end

"""
Update PCIe communication manager (called each iteration)
"""
function update_pcie_communication!(manager::PCIeCommunicationManager, iteration::Int)
    manager.iteration_counter = iteration
    
    # Check if manager is running
    if !manager.is_running
        return
    end
    
    # Process pending transfers
    process_transfer_queue!(manager)
    
    # Schedule new transfers if needed
    for source_gpu in manager.gpu_devices
        for target_gpu in manager.gpu_devices
            if source_gpu != target_gpu
                if should_schedule_transfer(manager, source_gpu, target_gpu)
                    schedule_transfer!(manager, source_gpu, target_gpu)
                end
            end
        end
    end
    
    # Clean up old candidate cache entries
    if iteration % 1000 == 0  # Every 1000 iterations
        cleanup_candidate_cache!(manager)
    end
    
    # Update average transfer time
    if !isempty(manager.transfer_history)
        total_time = sum(t.transfer_duration_ms for t in manager.transfer_history)
        manager.stats.average_transfer_time_ms = total_time / length(manager.transfer_history)
    end
end

"""
Cleanup old entries from candidate cache
"""
function cleanup_candidate_cache!(manager::PCIeCommunicationManager)
    if length(manager.candidate_cache) > 10000  # Arbitrary limit
        # Keep only most recent 5000 entries
        cache_entries = collect(pairs(manager.candidate_cache))
        sort!(cache_entries, by = p -> p.second.last_updated, rev = true)
        
        manager.candidate_cache = Dict{String, FeatureCandidate}()
        for (key, candidate) in cache_entries[1:min(5000, length(cache_entries))]
            manager.candidate_cache[key] = candidate
        end
    end
end

"""
Start PCIe communication manager
"""
function start_pcie_communication!(manager::PCIeCommunicationManager)
    manager.is_running = true
    manager.manager_state = "running"
    
    # Initialize peer-to-peer access (simulation)
    for src in manager.gpu_devices, dst in manager.gpu_devices
        if src != dst
            # In real implementation, would enable CUDA P2P access
            manager.peer_access_enabled[(src, dst)] = true
        end
    end
    
    @info "PCIe communication manager started"
end

"""
Stop PCIe communication manager
"""
function stop_pcie_communication!(manager::PCIeCommunicationManager)
    manager.is_running = false
    manager.manager_state = "stopped"
    
    # Wait for active transfers to complete
    for task in manager.transfer_tasks
        if !istaskdone(task)
            wait(task)
        end
    end
    empty!(manager.transfer_tasks)
    
    # Clear pending transfers
    empty!(manager.transfer_queue)
    empty!(manager.active_transfers)
    
    @info "PCIe communication manager stopped"
end

"""
Get PCIe communication status
"""
function get_pcie_status(manager::PCIeCommunicationManager)
    return Dict{String, Any}(
        "manager_state" => manager.manager_state,
        "is_running" => manager.is_running,
        "iteration_counter" => manager.iteration_counter,
        "gpu_devices" => manager.gpu_devices,
        "total_transfers" => manager.stats.total_transfers,
        "successful_transfers" => manager.stats.successful_transfers,
        "failed_transfers" => manager.stats.failed_transfers,
        "success_rate" => manager.stats.total_transfers > 0 ? 
                         manager.stats.successful_transfers / manager.stats.total_transfers : 0.0,
        "total_candidates_transferred" => manager.stats.total_candidates_transferred,
        "total_bytes_transferred" => manager.stats.total_bytes_transferred,
        "average_transfer_time_ms" => manager.stats.average_transfer_time_ms,
        "current_bandwidth_mbps" => manager.stats.current_bandwidth_mbps,
        "peak_bandwidth_mbps" => manager.stats.peak_bandwidth_mbps,
        "queue_overflows" => manager.stats.queue_overflows,
        "retry_events" => manager.stats.retry_events,
        "duplicate_candidates_filtered" => manager.stats.duplicate_candidates_filtered,
        "bandwidth_violations" => manager.stats.bandwidth_violations,
        "pending_transfers" => length(manager.transfer_queue),
        "active_transfers" => length(manager.active_transfers),
        "candidate_queues" => Dict(gpu_id => length(queue.candidates) 
                                 for (gpu_id, queue) in manager.candidate_queues),
        "last_transfer_time" => manager.stats.last_transfer_time
    )
end

"""
Generate PCIe communication report
"""
function generate_pcie_report(manager::PCIeCommunicationManager)
    status = get_pcie_status(manager)
    
    report = String[]
    
    push!(report, "=== PCIe Communication Report ===")
    push!(report, "Generated: $(now())")
    push!(report, "Manager State: $(status["manager_state"])")
    push!(report, "")
    
    # Transfer statistics
    push!(report, "Transfer Statistics:")
    push!(report, "  Total Transfers: $(status["total_transfers"])")
    push!(report, "  Successful: $(status["successful_transfers"])")
    push!(report, "  Failed: $(status["failed_transfers"])")
    push!(report, "  Success Rate: $(round(status["success_rate"] * 100, digits=2))%")
    push!(report, "  Candidates Transferred: $(status["total_candidates_transferred"])")
    push!(report, "  Total Bytes: $(status["total_bytes_transferred"])")
    push!(report, "")
    
    # Performance metrics
    push!(report, "Performance Metrics:")
    push!(report, "  Average Transfer Time: $(round(status["average_transfer_time_ms"], digits=2)) ms")
    push!(report, "  Current Bandwidth: $(round(status["current_bandwidth_mbps"], digits=2)) MB/s")
    push!(report, "  Peak Bandwidth: $(round(status["peak_bandwidth_mbps"], digits=2)) MB/s")
    push!(report, "  Bandwidth Violations: $(status["bandwidth_violations"])")
    push!(report, "")
    
    # Queue status
    push!(report, "Queue Status:")
    push!(report, "  Pending Transfers: $(status["pending_transfers"])")
    push!(report, "  Active Transfers: $(status["active_transfers"])")
    push!(report, "  Queue Overflows: $(status["queue_overflows"])")
    push!(report, "  Duplicates Filtered: $(status["duplicate_candidates_filtered"])")
    push!(report, "")
    
    # GPU candidate queues
    push!(report, "Candidate Queues:")
    for (gpu_id, count) in status["candidate_queues"]
        push!(report, "  GPU $gpu_id: $count candidates")
    end
    
    push!(report, "")
    push!(report, "=== End PCIe Report ===")
    
    return join(report, "\n")
end

"""
Cleanup PCIe communication manager
"""
function cleanup_pcie_communication!(manager::PCIeCommunicationManager)
    stop_pcie_communication!(manager)
    
    # Clear all data structures
    empty!(manager.candidate_queues)
    empty!(manager.candidate_cache)
    empty!(manager.candidate_history)
    empty!(manager.transfer_history)
    empty!(manager.bandwidth_history)
    empty!(manager.active_transfers)
    empty!(manager.error_log)
    
    manager.manager_state = "shutdown"
    @info "PCIe communication manager cleaned up"
end

# Export main types and functions
export CommunicationStrategy, TransferSchedule, SerializationFormat
export ROUND_ROBIN_TRANSFER, PRIORITY_BASED_TRANSFER, LOAD_BALANCED_TRANSFER, ADAPTIVE_TRANSFER, BANDWIDTH_OPTIMIZED
export FIXED_INTERVAL, ADAPTIVE_INTERVAL, PERFORMANCE_BASED, LOAD_BASED_SCHEDULE, HYBRID_SCHEDULE
export COMPACT_BINARY, COMPRESSED_BINARY, EFFICIENT_MSGPACK, CUSTOM_PROTOCOL
export FeatureCandidate, SerializedCandidate, TransferMetadata, PCIeConfig, PCIeCommunicationManager
export create_feature_candidate, create_transfer_metadata, create_pcie_config, initialize_pcie_communication_manager
export serialize_candidate, deserialize_candidate, add_candidate_to_queue!, get_top_candidates_for_transfer
export calculate_current_bandwidth, should_schedule_transfer, schedule_transfer!, execute_transfer!
export calculate_next_transfer_interval, process_transfer_queue!, merge_transferred_candidates!
export start_pcie_communication!, stop_pcie_communication!, update_pcie_communication!
export get_pcie_status, generate_pcie_report, cleanup_pcie_communication!

end # module PCIeCommunication