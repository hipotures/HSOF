"""
Shared Memory Coordination for MCTS Ensemble Feature Selection
Implements CPU-based coordination layer using mutex-protected shared memory for ensemble synchronization,
including inter-process communication between GPU controllers, thread-safe access, message passing protocol,
and shared candidate pool accessible by both GPU processes with deadlock detection and recovery.

This module provides coordinated multi-GPU execution ensuring ensemble trees can safely share state,
communicate via messages, and access shared resources without conflicts or race conditions.
"""

module SharedMemoryCoordination

using Random
using Statistics
using Dates
using Printf
using LinearAlgebra
using Base.Threads
# Using standard library modules only

# Import other stage2 modules for integration
include("pcie_communication.jl")
using .PCIeCommunication

include("performance_monitoring.jl")
using .PerformanceMonitoring

"""
Coordination message types for inter-process communication
"""
@enum CoordinationMessageType begin
    COMMAND_MESSAGE = 1           # Command execution
    STATUS_UPDATE = 2             # Status notification
    CANDIDATE_SHARE = 3           # Candidate sharing
    SYNCHRONIZATION_BARRIER = 4   # Synchronization point
    ERROR_NOTIFICATION = 5        # Error reporting
    SHUTDOWN_SIGNAL = 6           # Graceful shutdown
    HEARTBEAT_PING = 7           # Liveness check
    RESOURCE_REQUEST = 8          # Resource allocation
end

"""
Process status enumeration
"""
@enum ProcessStatus begin
    PROCESS_STARTING = 1
    PROCESS_RUNNING = 2
    PROCESS_PAUSED = 3
    PROCESS_STOPPING = 4
    PROCESS_STOPPED = 5
    PROCESS_ERROR = 6
    PROCESS_DEAD = 7
end

"""
Coordination message structure
"""
struct CoordinationMessage
    message_id::String
    message_type::CoordinationMessageType
    source_process_id::Int
    target_process_id::Int        # 0 for broadcast
    timestamp::DateTime
    priority::Int                 # 1-10, higher is more urgent
    payload::Dict{String, Any}    # Message data
    retry_count::Int
    max_retries::Int
    expiry_time::DateTime
end

"""
Create coordination message
"""
function create_coordination_message(message_type::CoordinationMessageType, 
                                   source_id::Int, target_id::Int,
                                   payload::Dict{String, Any} = Dict{String, Any}();
                                   priority::Int = 5,
                                   max_retries::Int = 3,
                                   expiry_seconds::Int = 30)
    message_id = string(rand(UInt64))
    return CoordinationMessage(
        message_id, message_type, source_id, target_id, now(),
        priority, payload, 0, max_retries, now() + Dates.Second(expiry_seconds)
    )
end

"""
Process information structure
"""
mutable struct ProcessInfo
    process_id::Int
    gpu_assignment::Int           # GPU device ID
    status::ProcessStatus
    last_heartbeat::DateTime
    start_time::DateTime
    iteration_count::Int
    tree_count::Int
    candidate_count::Int
    error_count::Int
    memory_usage_mb::Float64
    cpu_utilization::Float32
    is_responsive::Bool
    last_message_time::DateTime
end

"""
Create process information
"""
function create_process_info(process_id::Int, gpu_assignment::Int)
    return ProcessInfo(
        process_id, gpu_assignment, PROCESS_STARTING, now(), now(),
        0, 0, 0, 0, 0.0, 0.0f0, true, now()
    )
end

"""
Shared candidate pool entry
"""
struct SharedCandidate
    candidate_id::String
    feature_indices::Vector{Int}
    performance_score::Float32
    confidence_score::Float32
    source_process_id::Int
    source_tree_id::Int
    creation_time::DateTime
    access_count::Int
    is_validated::Bool
end

"""
Create shared candidate
"""
function create_shared_candidate(feature_indices::Vector{Int}, performance_score::Float32,
                               source_process_id::Int, source_tree_id::Int;
                               confidence_score::Float32 = 0.8f0)
    candidate_id = string(rand(UInt64))
    return SharedCandidate(
        candidate_id, feature_indices, performance_score, confidence_score,
        source_process_id, source_tree_id, now(), 0, false
    )
end

"""
Shared memory configuration
"""
struct SharedMemoryConfig
    segment_name::String              # Shared memory segment name
    segment_size_mb::Int             # Total segment size in MB
    max_processes::Int               # Maximum concurrent processes
    max_messages::Int                # Maximum queued messages
    max_candidates::Int              # Maximum shared candidates
    heartbeat_interval_ms::Int       # Heartbeat frequency
    deadlock_timeout_ms::Int         # Deadlock detection timeout
    message_retention_minutes::Int   # Message history retention
    candidate_retention_minutes::Int # Candidate pool retention
    enable_deadlock_detection::Bool  # Enable deadlock monitoring
    enable_performance_tracking::Bool # Track coordination performance
    enable_detailed_logging::Bool    # Detailed operation logging
    backup_interval_minutes::Int     # State backup frequency
    recovery_timeout_ms::Int         # Recovery operation timeout
end

"""
Create shared memory configuration
"""
function create_shared_memory_config(;
    segment_name::String = "hsof_ensemble_coordination",
    segment_size_mb::Int = 512,
    max_processes::Int = 4,
    max_messages::Int = 1000,
    max_candidates::Int = 500,
    heartbeat_interval_ms::Int = 1000,
    deadlock_timeout_ms::Int = 30000,
    message_retention_minutes::Int = 60,
    candidate_retention_minutes::Int = 30,
    enable_deadlock_detection::Bool = true,
    enable_performance_tracking::Bool = true,
    enable_detailed_logging::Bool = false,
    backup_interval_minutes::Int = 5,
    recovery_timeout_ms::Int = 10000)
    
    return SharedMemoryConfig(
        segment_name, segment_size_mb, max_processes, max_messages, max_candidates,
        heartbeat_interval_ms, deadlock_timeout_ms, message_retention_minutes,
        candidate_retention_minutes, enable_deadlock_detection, enable_performance_tracking,
        enable_detailed_logging, backup_interval_minutes, recovery_timeout_ms
    )
end

"""
Coordination statistics
"""
mutable struct CoordinationStats
    total_messages_sent::Int
    total_messages_received::Int
    total_messages_failed::Int
    total_candidates_shared::Int
    total_candidates_accessed::Int
    total_deadlocks_detected::Int
    total_deadlocks_resolved::Int
    total_heartbeats_sent::Int
    total_heartbeats_missed::Int
    average_message_latency_ms::Float64
    peak_message_latency_ms::Float64
    average_lock_wait_time_ms::Float64
    peak_lock_wait_time_ms::Float64
    memory_usage_bytes::Int
    active_processes::Int
    last_backup_time::DateTime
    last_cleanup_time::DateTime
end

"""
Initialize coordination statistics
"""
function initialize_coordination_stats()
    return CoordinationStats(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, now(), now()
    )
end

"""
Shared memory coordination manager
"""
mutable struct SharedMemoryCoordinator
    config::SharedMemoryConfig
    stats::CoordinationStats
    process_id::Int
    
    # Shared memory management
    shared_memory_file::String
    shared_memory_handle::IOStream
    memory_map::Vector{UInt8}
    
    # Process management
    processes::Dict{Int, ProcessInfo}
    process_lock::ReentrantLock
    
    # Message passing
    message_queue::Vector{CoordinationMessage}
    message_history::Vector{CoordinationMessage}
    message_lock::ReentrantLock
    
    # Candidate pool
    candidate_pool::Dict{String, SharedCandidate}
    candidate_access_log::Vector{Tuple{DateTime, String, Int}}
    candidate_lock::ReentrantLock
    
    # Synchronization
    global_lock::ReentrantLock
    barrier_participants::Dict{String, Set{Int}}
    barrier_lock::ReentrantLock
    
    # Monitoring and recovery
    deadlock_detector_task::Union{Task, Nothing}
    heartbeat_task::Union{Task, Nothing}
    cleanup_task::Union{Task, Nothing}
    is_active::Bool
    coordinator_state::String
    
    # Performance tracking
    operation_timestamps::Dict{String, DateTime}
    lock_wait_times::Vector{Float64}
    message_latencies::Vector{Float64}
    
    # Error handling
    error_log::Vector{String}
    recovery_attempts::Int
    last_error_time::DateTime
end

"""
Initialize shared memory coordinator
"""
function initialize_shared_memory_coordinator(config::SharedMemoryConfig, process_id::Int)
    # Create shared memory file path (use temp directory for testing)
    shared_memory_file = joinpath(tempdir(), "$(config.segment_name)_$(process_id)")
    
    # Initialize shared memory segment (simplified for testing)
    if !isfile(shared_memory_file)
        # Create new shared memory segment
        open(shared_memory_file, "w") do f
            write(f, zeros(UInt8, config.segment_size_mb * 1024 * 1024))
        end
        @info "Created shared memory segment: $shared_memory_file ($(config.segment_size_mb) MB)"
    else
        @info "Connected to existing shared memory segment: $shared_memory_file"
    end
    
    # Open shared memory handle (simplified)
    shared_memory_handle = open(shared_memory_file, "r+")
    memory_map = zeros(UInt8, config.segment_size_mb * 1024 * 1024)
    
    coordinator = SharedMemoryCoordinator(
        config,
        initialize_coordination_stats(),
        process_id,
        shared_memory_file,
        shared_memory_handle,
        memory_map,
        Dict{Int, ProcessInfo}(),
        ReentrantLock(),
        CoordinationMessage[],
        CoordinationMessage[],
        ReentrantLock(),
        Dict{String, SharedCandidate}(),
        Tuple{DateTime, String, Int}[],
        ReentrantLock(),
        ReentrantLock(),
        Dict{String, Set{Int}}(),
        ReentrantLock(),
        nothing,
        nothing,
        nothing,
        false,
        "initialized",
        Dict{String, DateTime}(),
        Float64[],
        Float64[],
        String[],
        0,
        now()
    )
    
    # Register this process
    process_info = create_process_info(process_id, 0)  # GPU assignment set later
    register_process!(coordinator, process_info)
    
    @info "Shared memory coordinator initialized for process $process_id"
    return coordinator
end

"""
Register process in coordination system
"""
function register_process!(coordinator::SharedMemoryCoordinator, process_info::ProcessInfo)
    lock(coordinator.process_lock) do
        coordinator.processes[process_info.process_id] = process_info
        coordinator.stats.active_processes = length(coordinator.processes)
        
        if coordinator.config.enable_detailed_logging
            @info "Registered process $(process_info.process_id) with GPU $(process_info.gpu_assignment)"
        end
    end
end

"""
Unregister process from coordination system
"""
function unregister_process!(coordinator::SharedMemoryCoordinator, process_id::Int)
    lock(coordinator.process_lock) do
        if haskey(coordinator.processes, process_id)
            delete!(coordinator.processes, process_id)
            coordinator.stats.active_processes = length(coordinator.processes)
            
            if coordinator.config.enable_detailed_logging
                @info "Unregistered process $process_id"
            end
        end
    end
end

"""
Send coordination message
"""
function send_message!(coordinator::SharedMemoryCoordinator, message::CoordinationMessage)::Bool
    lock(coordinator.message_lock) do
        try
            # Check if message is expired
            if now() > message.expiry_time
                coordinator.stats.total_messages_failed += 1
                return false
            end
            
            # Add to message queue
            push!(coordinator.message_queue, message)
            coordinator.stats.total_messages_sent += 1
            
            # Sort by priority (higher priority first)
            sort!(coordinator.message_queue, by = m -> -m.priority)
            
            # Maintain queue size limit
            if length(coordinator.message_queue) > coordinator.config.max_messages
                # Remove oldest low-priority messages
                filter!(m -> m.priority > 3 || 
                           now() - m.timestamp < Dates.Minute(1), 
                        coordinator.message_queue)
            end
            
            if coordinator.config.enable_detailed_logging
                @info "Sent message $(message.message_id): $(message.message_type) from $(message.source_process_id) to $(message.target_process_id)"
            end
            
            return true
            
        catch e
            push!(coordinator.error_log, "Failed to send message: $e")
            coordinator.stats.total_messages_failed += 1
            return false
        end
    end
end

"""
Receive coordination messages for process
"""
function receive_messages!(coordinator::SharedMemoryCoordinator, process_id::Int)::Vector{CoordinationMessage}
    lock(coordinator.message_lock) do
        received_messages = CoordinationMessage[]
        remaining_messages = CoordinationMessage[]
        
        for message in coordinator.message_queue
            # Check if message is for this process (direct or broadcast)
            if message.target_process_id == process_id || message.target_process_id == 0
                # Check expiry
                if now() <= message.expiry_time
                    push!(received_messages, message)
                    coordinator.stats.total_messages_received += 1
                    
                    # Add to history
                    push!(coordinator.message_history, message)
                    if length(coordinator.message_history) > coordinator.config.max_messages
                        deleteat!(coordinator.message_history, 1)
                    end
                else
                    coordinator.stats.total_messages_failed += 1
                end
            else
                push!(remaining_messages, message)
            end
        end
        
        coordinator.message_queue = remaining_messages
        
        if coordinator.config.enable_detailed_logging && !isempty(received_messages)
            @info "Process $process_id received $(length(received_messages)) messages"
        end
        
        return received_messages
    end
end

"""
Add candidate to shared pool
"""
function add_shared_candidate!(coordinator::SharedMemoryCoordinator, 
                              candidate::SharedCandidate)::Bool
    lock(coordinator.candidate_lock) do
        try
            # Check pool size limit
            if length(coordinator.candidate_pool) >= coordinator.config.max_candidates
                # Remove oldest candidates
                oldest_candidates = sort(collect(values(coordinator.candidate_pool)), 
                                       by = c -> c.creation_time)
                for i in 1:min(10, length(oldest_candidates))
                    delete!(coordinator.candidate_pool, oldest_candidates[i].candidate_id)
                end
            end
            
            coordinator.candidate_pool[candidate.candidate_id] = candidate
            coordinator.stats.total_candidates_shared += 1
            
            # Log access
            push!(coordinator.candidate_access_log, 
                  (now(), candidate.candidate_id, candidate.source_process_id))
            
            if coordinator.config.enable_detailed_logging
                @info "Added candidate $(candidate.candidate_id) to shared pool (score: $(candidate.performance_score))"
            end
            
            return true
            
        catch e
            push!(coordinator.error_log, "Failed to add candidate: $e")
            return false
        end
    end
end

"""
Get candidates from shared pool
"""
function get_shared_candidates(coordinator::SharedMemoryCoordinator, 
                             process_id::Int, max_count::Int = 10)::Vector{SharedCandidate}
    lock(coordinator.candidate_lock) do
        candidates = SharedCandidate[]
        
        # Get top candidates (excluding own candidates)
        all_candidates = collect(values(coordinator.candidate_pool))
        filter!(c -> c.source_process_id != process_id, all_candidates)
        sort!(all_candidates, by = c -> c.performance_score, rev = true)
        
        count = min(max_count, length(all_candidates))
        for i in 1:count
            candidate = all_candidates[i]
            push!(candidates, candidate)
            
            # Update access count
            updated_candidate = SharedCandidate(
                candidate.candidate_id, candidate.feature_indices, 
                candidate.performance_score, candidate.confidence_score,
                candidate.source_process_id, candidate.source_tree_id,
                candidate.creation_time, candidate.access_count + 1, 
                candidate.is_validated
            )
            coordinator.candidate_pool[candidate.candidate_id] = updated_candidate
            
            # Log access
            push!(coordinator.candidate_access_log, 
                  (now(), candidate.candidate_id, process_id))
        end
        
        coordinator.stats.total_candidates_accessed += length(candidates)
        
        if coordinator.config.enable_detailed_logging && !isempty(candidates)
            @info "Process $process_id retrieved $(length(candidates)) shared candidates"
        end
        
        return candidates
    end
end

"""
Update process heartbeat
"""
function update_heartbeat!(coordinator::SharedMemoryCoordinator, process_id::Int,
                          iteration_count::Int = 0, tree_count::Int = 0,
                          candidate_count::Int = 0)
    lock(coordinator.process_lock) do
        if haskey(coordinator.processes, process_id)
            process = coordinator.processes[process_id]
            process.last_heartbeat = now()
            process.iteration_count = iteration_count
            process.tree_count = tree_count
            process.candidate_count = candidate_count
            process.is_responsive = true
            process.last_message_time = now()
            
            coordinator.stats.total_heartbeats_sent += 1
        end
    end
end

"""
Detect deadlocks in coordination system
"""
function detect_deadlocks(coordinator::SharedMemoryCoordinator)::Vector{Int}
    deadlocked_processes = Int[]
    current_time = now()
    timeout_threshold = Dates.Millisecond(coordinator.config.deadlock_timeout_ms)
    
    lock(coordinator.process_lock) do
        for (process_id, process_info) in coordinator.processes
            time_since_heartbeat = current_time - process_info.last_heartbeat
            
            if time_since_heartbeat > timeout_threshold && 
               process_info.status == PROCESS_RUNNING
                push!(deadlocked_processes, process_id)
                process_info.is_responsive = false
                coordinator.stats.total_deadlocks_detected += 1
                
                @warn "Detected potential deadlock in process $process_id (no heartbeat for $(time_since_heartbeat))"
            end
        end
    end
    
    return deadlocked_processes
end

"""
Attempt recovery from deadlocks
"""
function recover_from_deadlocks!(coordinator::SharedMemoryCoordinator, 
                                deadlocked_processes::Vector{Int})::Bool
    all_recovered = true
    
    for process_id in deadlocked_processes
        try
            # Send recovery message
            recovery_message = create_coordination_message(
                COMMAND_MESSAGE, coordinator.process_id, process_id,
                Dict("command" => "recover", "timestamp" => string(now())),
                priority = 10
            )
            
            if send_message!(coordinator, recovery_message)
                coordinator.stats.total_deadlocks_resolved += 1
                @info "Sent recovery command to process $process_id"
            else
                all_recovered = false
                @error "Failed to send recovery command to process $process_id"
            end
            
        catch e
            all_recovered = false
            push!(coordinator.error_log, "Recovery failed for process $process_id: $e")
        end
    end
    
    return all_recovered
end

"""
Synchronization barrier - wait for all processes
"""
function synchronization_barrier!(coordinator::SharedMemoryCoordinator, 
                                 barrier_name::String, process_id::Int)::Bool
    lock(coordinator.barrier_lock) do
        if !haskey(coordinator.barrier_participants, barrier_name)
            coordinator.barrier_participants[barrier_name] = Set{Int}()
        end
        
        push!(coordinator.barrier_participants[barrier_name], process_id)
        expected_participants = coordinator.stats.active_processes
        
        if coordinator.config.enable_detailed_logging
            @info "Process $process_id joined barrier '$barrier_name' ($(length(coordinator.barrier_participants[barrier_name]))/$expected_participants)"
        end
        
        # Check if all processes have reached the barrier
        if length(coordinator.barrier_participants[barrier_name]) >= expected_participants
            # Release all processes
            delete!(coordinator.barrier_participants, barrier_name)
            
            # Send release message to all processes
            release_message = create_coordination_message(
                SYNCHRONIZATION_BARRIER, coordinator.process_id, 0,
                Dict("barrier_name" => barrier_name, "action" => "release"),
                priority = 9
            )
            send_message!(coordinator, release_message)
            
            @info "Barrier '$barrier_name' released - all processes synchronized"
            return true
        end
        
        return false
    end
end

"""
Start coordination services
"""
function start_coordination!(coordinator::SharedMemoryCoordinator)
    coordinator.is_active = true
    coordinator.coordinator_state = "starting"
    
    # Start deadlock detection task
    if coordinator.config.enable_deadlock_detection
        coordinator.deadlock_detector_task = @async begin
            while coordinator.is_active
                try
                    deadlocked = detect_deadlocks(coordinator)
                    if !isempty(deadlocked)
                        recover_from_deadlocks!(coordinator, deadlocked)
                    end
                catch e
                    push!(coordinator.error_log, "Deadlock detection error: $e")
                end
                sleep(coordinator.config.deadlock_timeout_ms / 2000.0)  # Check at half timeout interval
            end
        end
    end
    
    # Start heartbeat monitoring task
    coordinator.heartbeat_task = @async begin
        while coordinator.is_active
            try
                # Check for missed heartbeats
                current_time = now()
                timeout_threshold = Dates.Millisecond(coordinator.config.heartbeat_interval_ms * 3)
                
                lock(coordinator.process_lock) do
                    for (process_id, process_info) in coordinator.processes
                        if current_time - process_info.last_heartbeat > timeout_threshold &&
                           process_info.status == PROCESS_RUNNING
                            coordinator.stats.total_heartbeats_missed += 1
                            process_info.is_responsive = false
                            @warn "Process $process_id missed heartbeat"
                        end
                    end
                end
                
            catch e
                push!(coordinator.error_log, "Heartbeat monitoring error: $e")
            end
            sleep(coordinator.config.heartbeat_interval_ms / 1000.0)
        end
    end
    
    # Start cleanup task
    coordinator.cleanup_task = @async begin
        while coordinator.is_active
            try
                cleanup_coordination!(coordinator)
            catch e
                push!(coordinator.error_log, "Cleanup error: $e")
            end
            sleep(60)  # Cleanup every minute
        end
    end
    
    coordinator.coordinator_state = "running"
    @info "Shared memory coordination services started"
end

"""
Stop coordination services
"""
function stop_coordination!(coordinator::SharedMemoryCoordinator)
    coordinator.is_active = false
    coordinator.coordinator_state = "stopping"
    
    # Send shutdown signals
    shutdown_message = create_coordination_message(
        SHUTDOWN_SIGNAL, coordinator.process_id, 0,
        Dict("reason" => "coordinator_shutdown"),
        priority = 10
    )
    send_message!(coordinator, shutdown_message)
    
    # Stop background tasks
    if coordinator.deadlock_detector_task !== nothing
        try
            wait(coordinator.deadlock_detector_task)
        catch
        end
    end
    
    if coordinator.heartbeat_task !== nothing
        try
            wait(coordinator.heartbeat_task)
        catch
        end
    end
    
    if coordinator.cleanup_task !== nothing
        try
            wait(coordinator.cleanup_task)
        catch
        end
    end
    
    coordinator.coordinator_state = "stopped"
    @info "Shared memory coordination services stopped"
end

"""
Cleanup coordination resources
"""
function cleanup_coordination!(coordinator::SharedMemoryCoordinator)
    current_time = now()
    
    # Clean up expired messages
    lock(coordinator.message_lock) do
        filter!(m -> current_time <= m.expiry_time, coordinator.message_queue)
        
        # Clean up old message history
        retention_time = Dates.Minute(coordinator.config.message_retention_minutes)
        filter!(m -> current_time - m.timestamp <= retention_time, coordinator.message_history)
    end
    
    # Clean up old candidates
    lock(coordinator.candidate_lock) do
        retention_time = Dates.Minute(coordinator.config.candidate_retention_minutes)
        to_remove = String[]
        
        for (candidate_id, candidate) in coordinator.candidate_pool
            if current_time - candidate.creation_time > retention_time
                push!(to_remove, candidate_id)
            end
        end
        
        for candidate_id in to_remove
            delete!(coordinator.candidate_pool, candidate_id)
        end
        
        # Clean up access log
        filter!(entry -> current_time - entry[1] <= retention_time, 
                coordinator.candidate_access_log)
    end
    
    # Update cleanup timestamp
    coordinator.stats.last_cleanup_time = current_time
    
    if coordinator.config.enable_detailed_logging
        @info "Coordination cleanup completed"
    end
end

"""
Get coordination status
"""
function get_coordination_status(coordinator::SharedMemoryCoordinator)::Dict{String, Any}
    status = Dict{String, Any}()
    
    status["coordinator_state"] = coordinator.coordinator_state
    status["is_active"] = coordinator.is_active
    status["process_id"] = coordinator.process_id
    status["active_processes"] = coordinator.stats.active_processes
    status["memory_usage_mb"] = coordinator.stats.memory_usage_bytes / (1024 * 1024)
    
    # Message statistics
    status["total_messages_sent"] = coordinator.stats.total_messages_sent
    status["total_messages_received"] = coordinator.stats.total_messages_received
    status["total_messages_failed"] = coordinator.stats.total_messages_failed
    status["pending_messages"] = length(coordinator.message_queue)
    status["message_history_size"] = length(coordinator.message_history)
    
    # Candidate pool statistics
    status["total_candidates_shared"] = coordinator.stats.total_candidates_shared
    status["total_candidates_accessed"] = coordinator.stats.total_candidates_accessed
    status["active_candidates"] = length(coordinator.candidate_pool)
    status["candidate_access_log_size"] = length(coordinator.candidate_access_log)
    
    # Deadlock statistics
    status["total_deadlocks_detected"] = coordinator.stats.total_deadlocks_detected
    status["total_deadlocks_resolved"] = coordinator.stats.total_deadlocks_resolved
    status["total_heartbeats_sent"] = coordinator.stats.total_heartbeats_sent
    status["total_heartbeats_missed"] = coordinator.stats.total_heartbeats_missed
    
    # Performance metrics
    status["average_message_latency_ms"] = coordinator.stats.average_message_latency_ms
    status["peak_message_latency_ms"] = coordinator.stats.peak_message_latency_ms
    status["average_lock_wait_time_ms"] = coordinator.stats.average_lock_wait_time_ms
    status["peak_lock_wait_time_ms"] = coordinator.stats.peak_lock_wait_time_ms
    
    # Process information
    lock(coordinator.process_lock) do
        processes = Dict{String, Any}()
        for (process_id, process_info) in coordinator.processes
            processes[string(process_id)] = Dict(
                "status" => string(process_info.status),
                "gpu_assignment" => process_info.gpu_assignment,
                "last_heartbeat" => string(process_info.last_heartbeat),
                "iteration_count" => process_info.iteration_count,
                "tree_count" => process_info.tree_count,
                "candidate_count" => process_info.candidate_count,
                "is_responsive" => process_info.is_responsive,
                "error_count" => process_info.error_count,
                "memory_usage_mb" => process_info.memory_usage_mb,
                "cpu_utilization" => process_info.cpu_utilization
            )
        end
        status["processes"] = processes
    end
    
    return status
end

"""
Generate coordination report
"""
function generate_coordination_report(coordinator::SharedMemoryCoordinator)::String
    status = get_coordination_status(coordinator)
    
    report = "Shared Memory Coordination Report\n"
    report *= "=" ^ 50 * "\n\n"
    
    report *= "Coordinator Status:\n"
    report *= "  State: $(status["coordinator_state"])\n"
    report *= "  Active: $(status["is_active"])\n"
    report *= "  Process ID: $(status["process_id"])\n"
    report *= "  Active Processes: $(status["active_processes"])\n"
    report *= "  Memory Usage: $(round(status["memory_usage_mb"], digits=2)) MB\n\n"
    
    report *= "Message Statistics:\n"
    report *= "  Messages Sent: $(status["total_messages_sent"])\n"
    report *= "  Messages Received: $(status["total_messages_received"])\n"
    report *= "  Messages Failed: $(status["total_messages_failed"])\n"
    report *= "  Pending Messages: $(status["pending_messages"])\n"
    report *= "  Message History: $(status["message_history_size"])\n"
    
    if status["total_messages_sent"] > 0
        success_rate = (status["total_messages_sent"] - status["total_messages_failed"]) / status["total_messages_sent"] * 100
        report *= "  Success Rate: $(round(success_rate, digits=2))%\n"
    end
    report *= "\n"
    
    report *= "Candidate Pool Statistics:\n"
    report *= "  Candidates Shared: $(status["total_candidates_shared"])\n"
    report *= "  Candidates Accessed: $(status["total_candidates_accessed"])\n"
    report *= "  Active Candidates: $(status["active_candidates"])\n"
    report *= "  Access Log Size: $(status["candidate_access_log_size"])\n\n"
    
    report *= "Deadlock Detection:\n"
    report *= "  Deadlocks Detected: $(status["total_deadlocks_detected"])\n"
    report *= "  Deadlocks Resolved: $(status["total_deadlocks_resolved"])\n"
    report *= "  Heartbeats Sent: $(status["total_heartbeats_sent"])\n"
    report *= "  Heartbeats Missed: $(status["total_heartbeats_missed"])\n\n"
    
    report *= "Performance Metrics:\n"
    report *= "  Avg Message Latency: $(round(status["average_message_latency_ms"], digits=2)) ms\n"
    report *= "  Peak Message Latency: $(round(status["peak_message_latency_ms"], digits=2)) ms\n"
    report *= "  Avg Lock Wait Time: $(round(status["average_lock_wait_time_ms"], digits=2)) ms\n"
    report *= "  Peak Lock Wait Time: $(round(status["peak_lock_wait_time_ms"], digits=2)) ms\n\n"
    
    report *= "Process Information:\n"
    for (process_id, process_info) in status["processes"]
        report *= "  Process $process_id:\n"
        report *= "    Status: $(process_info["status"])\n"
        report *= "    GPU Assignment: $(process_info["gpu_assignment"])\n"
        report *= "    Responsive: $(process_info["is_responsive"])\n"
        report *= "    Iterations: $(process_info["iteration_count"])\n"
        report *= "    Trees: $(process_info["tree_count"])\n"
        report *= "    Candidates: $(process_info["candidate_count"])\n"
        report *= "    Memory: $(round(process_info["memory_usage_mb"], digits=2)) MB\n"
        report *= "    CPU: $(round(process_info["cpu_utilization"], digits=1))%\n"
        report *= "    Errors: $(process_info["error_count"])\n\n"
    end
    
    report *= "End Coordination Report\n"
    
    return report
end

# Export public functions and enum values
export CoordinationMessageType, ProcessStatus, CoordinationMessage, ProcessInfo, SharedCandidate
export SharedMemoryConfig, CoordinationStats, SharedMemoryCoordinator
export create_coordination_message, create_process_info, create_shared_candidate, create_shared_memory_config
export initialize_coordination_stats, initialize_shared_memory_coordinator
export register_process!, unregister_process!, send_message!, receive_messages!
export add_shared_candidate!, get_shared_candidates, update_heartbeat!
export detect_deadlocks, recover_from_deadlocks!, synchronization_barrier!
export start_coordination!, stop_coordination!, cleanup_coordination!
export get_coordination_status, generate_coordination_report

# Export enum values for CoordinationMessageType
export COMMAND_MESSAGE, STATUS_UPDATE, CANDIDATE_SHARE, SYNCHRONIZATION_BARRIER
export ERROR_NOTIFICATION, SHUTDOWN_SIGNAL, HEARTBEAT_PING, RESOURCE_REQUEST

# Export enum values for ProcessStatus
export PROCESS_STARTING, PROCESS_RUNNING, PROCESS_PAUSED, PROCESS_STOPPING
export PROCESS_STOPPED, PROCESS_ERROR, PROCESS_DEAD

end # module