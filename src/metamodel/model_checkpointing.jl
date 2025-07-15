module ModelCheckpointing

using CUDA
using Statistics
using Dates
using Printf
using JSON3
using Serialization
using CodecZlib
using CRC32c

# Optional Flux import for model state extraction
const FLUX_AVAILABLE = try
    using Flux
    true
catch
    false
end

export CheckpointConfig, CheckpointManager, CheckpointMetadata
export create_checkpoint_manager, save_checkpoint!, load_checkpoint!
export list_checkpoints, cleanup_old_checkpoints!, rollback_to_checkpoint!
export validate_checkpoint, get_checkpoint_info, compress_checkpoint!

"""
Configuration for model checkpointing system
"""
struct CheckpointConfig
    checkpoint_dir::String              # Directory for storing checkpoints
    max_checkpoints::Int                # Maximum number of checkpoints to keep
    save_interval::Int                  # Save every N training iterations
    async_save::Bool                    # Use asynchronous saving
    compression_enabled::Bool           # Enable checkpoint compression
    compression_level::Int              # Compression level (1-9)
    incremental_saves::Bool             # Save only changed weights
    metadata_tracking::Bool             # Track comprehensive metadata
    backup_to_remote::Bool              # Enable remote backup (future)
    validation_enabled::Bool            # Validate checkpoints after saving
end

"""
Default checkpoint configuration
"""
function CheckpointConfig(;
    checkpoint_dir::String = "checkpoints",
    max_checkpoints::Int = 10,
    save_interval::Int = 1000,
    async_save::Bool = true,
    compression_enabled::Bool = true,
    compression_level::Int = 6,
    incremental_saves::Bool = true,
    metadata_tracking::Bool = true,
    backup_to_remote::Bool = false,
    validation_enabled::Bool = true
)
    CheckpointConfig(
        checkpoint_dir, max_checkpoints, save_interval, async_save,
        compression_enabled, compression_level, incremental_saves,
        metadata_tracking, backup_to_remote, validation_enabled
    )
end

"""
Metadata for checkpoint tracking
"""
mutable struct CheckpointMetadata
    checkpoint_id::String               # Unique checkpoint identifier
    timestamp::DateTime                 # Creation timestamp
    iteration::Int64                    # Training iteration number
    epoch::Int64                        # Training epoch number
    
    # Model information
    model_hash::UInt64                  # Hash of model weights
    model_size_bytes::Int64             # Size of model in bytes
    parameter_count::Int64              # Total number of parameters
    
    # Training metrics
    training_loss::Float64              # Current training loss
    validation_loss::Float64            # Current validation loss
    correlation_score::Float64          # Model correlation with actual scores
    learning_rate::Float64              # Current learning rate
    
    # Performance metrics
    inference_latency_ms::Float64       # Average inference latency
    throughput_samples_per_sec::Float64 # Training throughput
    
    # System information
    gpu_memory_used_mb::Float64         # GPU memory usage
    system_memory_used_mb::Float64      # System memory usage
    cuda_version::String                # CUDA version
    julia_version::String               # Julia version
    
    # File information
    checkpoint_size_bytes::Int64        # Checkpoint file size
    compression_ratio::Float64          # Compression ratio achieved
    save_duration_ms::Float64           # Time taken to save
    validation_passed::Bool             # Checkpoint validation result
    
    # Additional metadata
    tags::Vector{String}                # Custom tags for organization
    notes::String                       # Optional notes
end

"""
Create checkpoint metadata
"""
function CheckpointMetadata(;
    checkpoint_id::String = generate_checkpoint_id(),
    iteration::Int64 = 0,
    epoch::Int64 = 0,
    training_loss::Float64 = 0.0,
    validation_loss::Float64 = 0.0,
    correlation_score::Float64 = 0.0,
    learning_rate::Float64 = 0.001,
    inference_latency_ms::Float64 = 0.0,
    throughput_samples_per_sec::Float64 = 0.0,
    tags::Vector{String} = String[],
    notes::String = ""
)
    CheckpointMetadata(
        checkpoint_id,
        now(),
        iteration,
        epoch,
        UInt64(0),  # model_hash - will be computed
        Int64(0),   # model_size_bytes - will be computed
        Int64(0),   # parameter_count - will be computed
        training_loss,
        validation_loss,
        correlation_score,
        learning_rate,
        inference_latency_ms,
        throughput_samples_per_sec,
        get_gpu_memory_usage(),
        get_system_memory_usage(),
        string(CUDA.runtime_version()),
        string(VERSION),
        Int64(0),   # checkpoint_size_bytes - will be computed
        0.0,        # compression_ratio - will be computed
        0.0,        # save_duration_ms - will be computed
        false,      # validation_passed - will be set
        tags,
        notes
    )
end

"""
Checkpoint data structure
"""
struct CheckpointData
    # Model state
    model_state::Dict{String, Any}      # Model parameters and buffers
    optimizer_state::Dict{String, Any}  # Optimizer state
    
    # Training state
    replay_buffer::Dict{String, Any}    # Experience replay buffer state
    training_history::Vector{Float64}   # Training loss history
    validation_history::Vector{Float64} # Validation loss history
    
    # Configuration
    model_config::Dict{String, Any}     # Model configuration
    training_config::Dict{String, Any}  # Training configuration
    
    # Metadata
    metadata::CheckpointMetadata         # Checkpoint metadata
end

"""
Model checkpointing manager
"""
mutable struct CheckpointManager
    config::CheckpointConfig
    checkpoint_registry::Dict{String, CheckpointMetadata}
    active_saves::Set{String}           # Currently saving checkpoints
    save_stream::CuStream               # Dedicated CUDA stream for saving
    last_checkpoint_time::Float64       # Time of last checkpoint
    
    # Incremental saving state
    previous_model_hash::UInt64         # Hash of previous model state
    weight_deltas::Dict{String, Any}    # Changed weights since last save
    
    # Statistics
    total_saves::Int64                  # Total number of saves
    total_save_time::Float64            # Total time spent saving
    bytes_saved::Int64                  # Total bytes saved
    bytes_compressed::Int64             # Total bytes after compression
end

"""
Create checkpoint manager
"""
function create_checkpoint_manager(config::CheckpointConfig)
    # Create checkpoint directory
    mkpath(config.checkpoint_dir)
    
    # Create metadata directory
    metadata_dir = joinpath(config.checkpoint_dir, "metadata")
    mkpath(metadata_dir)
    
    # Load existing checkpoint registry
    registry_file = joinpath(config.checkpoint_dir, "registry.json")
    registry = if isfile(registry_file)
        try
            data = JSON3.read(read(registry_file, String))
            Dict(k => CheckpointMetadata(;
                checkpoint_id=v.checkpoint_id,
                iteration=v.iteration,
                epoch=v.epoch,
                training_loss=v.training_loss,
                validation_loss=v.validation_loss,
                correlation_score=v.correlation_score,
                learning_rate=v.learning_rate,
                inference_latency_ms=v.inference_latency_ms,
                throughput_samples_per_sec=v.throughput_samples_per_sec,
                tags=v.tags,
                notes=v.notes
            ) for (k, v) in data)
        catch e
            @warn "Failed to load checkpoint registry: $e"
            Dict{String, CheckpointMetadata}()
        end
    else
        Dict{String, CheckpointMetadata}()
    end
    
    CheckpointManager(
        config,
        registry,
        Set{String}(),
        CuStream(),
        time(),
        UInt64(0),
        Dict{String, Any}(),
        0,
        0.0,
        0,
        0
    )
end

"""
Generate unique checkpoint ID
"""
function generate_checkpoint_id()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    random_suffix = string(rand(UInt32), base=16, pad=8)
    return "checkpoint_$(timestamp)_$(random_suffix)"
end

"""
Get GPU memory usage in MB
"""
function get_gpu_memory_usage()
    try
        if CUDA.functional()
            used, total = CUDA.memory_status()
            return Float64(used) / (1024^2)
        else
            return 0.0
        end
    catch
        return 0.0
    end
end

"""
Get system memory usage in MB
"""
function get_system_memory_usage()
    try
        # This is a simplified version - real implementation would use system APIs
        return Float64(Sys.total_memory() - Sys.free_memory()) / (1024^2)
    catch
        return 0.0
    end
end

"""
Compute model hash for change detection
"""
function compute_model_hash(model_state::Dict{String, <:Any})
    # Simple hash combining all parameter hashes
    combined_hash = UInt64(0)
    
    for (name, param) in model_state
        if param isa AbstractArray
            param_hash = hash(Array(param))
            combined_hash = hash(combined_hash, param_hash)
        end
    end
    
    return combined_hash
end

"""
Extract model state from Flux model or custom model
"""
function extract_model_state(model)
    state = Dict{String, Any}()
    
    if FLUX_AVAILABLE
        # Extract parameters using Flux.state
        try
            model_state = Flux.state(model)
            for (name, param) in pairs(model_state)
                if param isa AbstractArray
                    # Convert to CPU for saving
                    state[string(name)] = Array(param)
                else
                    state[string(name)] = param
                end
            end
        catch e
            @warn "Failed to extract model state: $e"
            # Fallback: try to extract manually
            try
                if hasfield(typeof(model), :layers)
                    for (i, layer) in enumerate(model.layers)
                        if hasfield(typeof(layer), :weight)
                            state["layer_$(i)_weight"] = Array(layer.weight)
                        end
                        if hasfield(typeof(layer), :bias) && layer.bias !== nothing
                            state["layer_$(i)_bias"] = Array(layer.bias)
                        end
                    end
                end
            catch e2
                @warn "Fallback model state extraction failed: $e2"
            end
        end
    else
        # Handle custom models without Flux
        if hasfield(typeof(model), :weights) && model.weights isa Dict
            state = copy(model.weights)
        elseif hasfield(typeof(model), :parameters) && model.parameters isa Dict
            state = copy(model.parameters)
        else
            @warn "No Flux available and model doesn't have recognizable state structure"
        end
    end
    
    return state
end

"""
Extract optimizer state
"""
function extract_optimizer_state(optimizer)
    state = Dict{String, Any}()
    
    try
        # Handle different optimizer types
        if hasfield(typeof(optimizer), :state)
            state["optimizer_state"] = optimizer.state
        end
        
        if hasfield(typeof(optimizer), :eta)
            state["learning_rate"] = optimizer.eta
        elseif hasfield(typeof(optimizer), :α)
            state["learning_rate"] = optimizer.α
        end
        
        if hasfield(typeof(optimizer), :beta)
            state["beta"] = optimizer.beta
        end
        
        if hasfield(typeof(optimizer), :momentum)
            state["momentum"] = optimizer.momentum
        end
        
        # Store optimizer type
        state["optimizer_type"] = string(typeof(optimizer))
        
    catch e
        @warn "Failed to extract optimizer state: $e"
    end
    
    return state
end

"""
Save checkpoint asynchronously
"""
function save_checkpoint!(
    manager::CheckpointManager,
    model,
    optimizer,
    replay_buffer = nothing;
    metadata::Union{CheckpointMetadata, Nothing} = nothing,
    force::Bool = false
)
    # Check if we should save based on interval
    current_time = time()
    if !force && (current_time - manager.last_checkpoint_time) < manager.config.save_interval
        return nothing
    end
    
    checkpoint_id = generate_checkpoint_id()
    
    # Create metadata if not provided
    if metadata === nothing
        metadata = CheckpointMetadata(checkpoint_id=checkpoint_id)
    else
        metadata.checkpoint_id = checkpoint_id
    end
    
    start_time = time()
    
    # Extract model state
    model_state = extract_model_state(model)
    optimizer_state = extract_optimizer_state(optimizer)
    
    # Compute model hash
    model_hash = compute_model_hash(model_state)
    metadata.model_hash = model_hash
    
    # Check for incremental save
    if manager.config.incremental_saves && manager.previous_model_hash != 0
        # Only save changed weights
        changed_weights = Dict{String, Any}()
        for (name, param) in model_state
            if !haskey(manager.weight_deltas, name) || 
               hash(param) != hash(manager.weight_deltas[name])
                changed_weights[name] = param
                manager.weight_deltas[name] = param
            end
        end
        
        if isempty(changed_weights) && !force
            @info "No model changes detected, skipping checkpoint"
            return nothing
        end
        
        model_state = changed_weights
        metadata.notes *= " [incremental]"
    else
        # Full save
        manager.weight_deltas = copy(model_state)
    end
    
    manager.previous_model_hash = model_hash
    
    # Extract replay buffer state
    replay_buffer_state = if replay_buffer !== nothing
        try
            extract_replay_buffer_state(replay_buffer)
        catch e
            @warn "Failed to extract replay buffer state: $e"
            Dict{String, Any}()
        end
    else
        Dict{String, Any}()
    end
    
    # Create checkpoint data
    checkpoint_data = CheckpointData(
        model_state,
        optimizer_state,
        replay_buffer_state,
        Float64[],  # training_history - would be populated from training loop
        Float64[],  # validation_history - would be populated from training loop
        Dict{String, Any}("model_type" => string(typeof(model))),
        Dict{String, Any}("save_time" => current_time),
        metadata
    )
    
    # Update metadata with computed values
    metadata.parameter_count = sum(length(p) for p in values(model_state) if p isa AbstractArray)
    metadata.model_size_bytes = sum(sizeof(p) for p in values(model_state) if p isa AbstractArray)
    metadata.timestamp = now()
    
    if manager.config.async_save
        # Save asynchronously
        @async begin
            try
                _save_checkpoint_sync(manager, checkpoint_id, checkpoint_data)
                manager.last_checkpoint_time = current_time
                @info "Checkpoint $checkpoint_id saved successfully"
            catch e
                @error "Failed to save checkpoint $checkpoint_id: $e"
                delete!(manager.active_saves, checkpoint_id)
            end
        end
        
        # Add to active saves
        push!(manager.active_saves, checkpoint_id)
    else
        # Save synchronously
        _save_checkpoint_sync(manager, checkpoint_id, checkpoint_data)
        manager.last_checkpoint_time = current_time
    end
    
    return checkpoint_id
end

"""
Extract replay buffer state
"""
function extract_replay_buffer_state(replay_buffer)
    state = Dict{String, Any}()
    
    # This would be customized based on the specific replay buffer implementation
    # For now, provide a generic interface
    
    try
        if hasfield(typeof(replay_buffer), :buffer)
            # Convert GPU arrays to CPU for saving
            buffer_data = replay_buffer.buffer
            if buffer_data isa CuArray
                state["buffer_data"] = Array(buffer_data)
            else
                state["buffer_data"] = buffer_data
            end
        end
        
        if hasfield(typeof(replay_buffer), :size)
            state["buffer_size"] = replay_buffer.size
        end
        
        if hasfield(typeof(replay_buffer), :capacity)
            state["buffer_capacity"] = replay_buffer.capacity
        end
        
        if hasfield(typeof(replay_buffer), :head)
            state["buffer_head"] = replay_buffer.head
        end
        
        if hasfield(typeof(replay_buffer), :tail)
            state["buffer_tail"] = replay_buffer.tail
        end
        
        state["buffer_type"] = string(typeof(replay_buffer))
        
    catch e
        @warn "Failed to extract replay buffer state: $e"
    end
    
    return state
end

"""
Internal synchronous checkpoint saving
"""
function _save_checkpoint_sync(
    manager::CheckpointManager,
    checkpoint_id::String,
    checkpoint_data::CheckpointData
)
    start_time = time()
    
    try
        # Create checkpoint file path
        checkpoint_file = joinpath(manager.config.checkpoint_dir, "$checkpoint_id.jls")
        
        # Serialize checkpoint data
        serialized_data = serialize_checkpoint(checkpoint_data)
        
        # Compress if enabled
        if manager.config.compression_enabled
            original_size = length(serialized_data)
            compressed_data = compress_data(serialized_data, manager.config.compression_level)
            compression_ratio = length(compressed_data) / original_size
            
            checkpoint_data.metadata.compression_ratio = compression_ratio
            checkpoint_data.metadata.checkpoint_size_bytes = length(compressed_data)
            
            # Save compressed data
            write(checkpoint_file, compressed_data)
            
            manager.bytes_saved += original_size
            manager.bytes_compressed += length(compressed_data)
        else
            checkpoint_data.metadata.checkpoint_size_bytes = length(serialized_data)
            write(checkpoint_file, serialized_data)
            
            manager.bytes_saved += length(serialized_data)
            manager.bytes_compressed += length(serialized_data)
        end
        
        # Save metadata separately
        metadata_file = joinpath(manager.config.checkpoint_dir, "metadata", "$checkpoint_id.json")
        save_metadata(checkpoint_data.metadata, metadata_file)
        
        # Validate checkpoint if enabled
        if manager.config.validation_enabled
            checkpoint_data.metadata.validation_passed = validate_checkpoint(checkpoint_file)
        else
            checkpoint_data.metadata.validation_passed = true
        end
        
        # Update timing
        save_duration = (time() - start_time) * 1000
        checkpoint_data.metadata.save_duration_ms = save_duration
        
        # Update registry
        manager.checkpoint_registry[checkpoint_id] = checkpoint_data.metadata
        save_registry(manager)
        
        # Update statistics
        manager.total_saves += 1
        manager.total_save_time += save_duration / 1000
        
        # Cleanup old checkpoints
        if length(manager.checkpoint_registry) > manager.config.max_checkpoints
            cleanup_old_checkpoints!(manager)
        end
        
        # Remove from active saves
        delete!(manager.active_saves, checkpoint_id)
        
        @info "Checkpoint saved" checkpoint_id=checkpoint_id size_mb=round(checkpoint_data.metadata.checkpoint_size_bytes/1024^2, digits=2) duration_ms=round(save_duration, digits=1)
        
    catch e
        @error "Error saving checkpoint: $e" checkpoint_id=checkpoint_id
        delete!(manager.active_saves, checkpoint_id)
        rethrow(e)
    end
end

"""
Serialize checkpoint data
"""
function serialize_checkpoint(checkpoint_data::CheckpointData)
    io = IOBuffer()
    serialize(io, checkpoint_data)
    return take!(io)
end

"""
Compress data using CodecZlib
"""
function compress_data(data::Vector{UInt8}, level::Int)
    compressed = transcode(GzipCompressor(level=level), data)
    return compressed
end

"""
Decompress data using CodecZlib
"""
function decompress_data(compressed_data::Vector{UInt8})
    decompressed = transcode(GzipDecompressor(), compressed_data)
    return decompressed
end

"""
Save metadata to JSON file
"""
function save_metadata(metadata::CheckpointMetadata, filepath::String)
    # Convert metadata to dictionary for JSON serialization
    metadata_dict = Dict(
        "checkpoint_id" => metadata.checkpoint_id,
        "timestamp" => string(metadata.timestamp),
        "iteration" => metadata.iteration,
        "epoch" => metadata.epoch,
        "model_hash" => string(metadata.model_hash, base=16),
        "model_size_bytes" => metadata.model_size_bytes,
        "parameter_count" => metadata.parameter_count,
        "training_loss" => metadata.training_loss,
        "validation_loss" => metadata.validation_loss,
        "correlation_score" => metadata.correlation_score,
        "learning_rate" => metadata.learning_rate,
        "inference_latency_ms" => metadata.inference_latency_ms,
        "throughput_samples_per_sec" => metadata.throughput_samples_per_sec,
        "gpu_memory_used_mb" => metadata.gpu_memory_used_mb,
        "system_memory_used_mb" => metadata.system_memory_used_mb,
        "cuda_version" => metadata.cuda_version,
        "julia_version" => metadata.julia_version,
        "checkpoint_size_bytes" => metadata.checkpoint_size_bytes,
        "compression_ratio" => metadata.compression_ratio,
        "save_duration_ms" => metadata.save_duration_ms,
        "validation_passed" => metadata.validation_passed,
        "tags" => metadata.tags,
        "notes" => metadata.notes
    )
    
    JSON3.write(filepath, metadata_dict)
end

"""
Save checkpoint registry
"""
function save_registry(manager::CheckpointManager)
    registry_file = joinpath(manager.config.checkpoint_dir, "registry.json")
    
    # Convert registry to serializable format
    registry_dict = Dict()
    for (id, metadata) in manager.checkpoint_registry
        registry_dict[id] = Dict(
            "checkpoint_id" => metadata.checkpoint_id,
            "timestamp" => string(metadata.timestamp),
            "iteration" => metadata.iteration,
            "epoch" => metadata.epoch,
            "training_loss" => metadata.training_loss,
            "validation_loss" => metadata.validation_loss,
            "correlation_score" => metadata.correlation_score,
            "learning_rate" => metadata.learning_rate,
            "inference_latency_ms" => metadata.inference_latency_ms,
            "throughput_samples_per_sec" => metadata.throughput_samples_per_sec,
            "checkpoint_size_bytes" => metadata.checkpoint_size_bytes,
            "compression_ratio" => metadata.compression_ratio,
            "save_duration_ms" => metadata.save_duration_ms,
            "validation_passed" => metadata.validation_passed,
            "tags" => metadata.tags,
            "notes" => metadata.notes
        )
    end
    
    JSON3.write(registry_file, registry_dict)
end

"""
Load checkpoint from file
"""
function load_checkpoint!(
    manager::CheckpointManager,
    checkpoint_id::String,
    model = nothing,
    optimizer = nothing,
    replay_buffer = nothing
)
    if !haskey(manager.checkpoint_registry, checkpoint_id)
        throw(ArgumentError("Checkpoint $checkpoint_id not found in registry"))
    end
    
    checkpoint_file = joinpath(manager.config.checkpoint_dir, "$checkpoint_id.jls")
    
    if !isfile(checkpoint_file)
        throw(ArgumentError("Checkpoint file $checkpoint_file not found"))
    end
    
    @info "Loading checkpoint: $checkpoint_id"
    start_time = time()
    
    try
        # Read checkpoint file
        checkpoint_data_raw = read(checkpoint_file)
        
        # Decompress if needed
        if manager.config.compression_enabled
            checkpoint_data_raw = decompress_data(checkpoint_data_raw)
        end
        
        # Deserialize
        io = IOBuffer(checkpoint_data_raw)
        checkpoint_data = deserialize(io)
        
        # Restore model state
        if model !== nothing
            restore_model_state!(model, checkpoint_data.model_state)
        end
        
        # Restore optimizer state
        if optimizer !== nothing
            restore_optimizer_state!(optimizer, checkpoint_data.optimizer_state)
        end
        
        # Restore replay buffer state
        if replay_buffer !== nothing
            restore_replay_buffer_state!(replay_buffer, checkpoint_data.replay_buffer)
        end
        
        load_duration = (time() - start_time) * 1000
        @info "Checkpoint loaded successfully" checkpoint_id=checkpoint_id duration_ms=round(load_duration, digits=1)
        
        return checkpoint_data
        
    catch e
        @error "Failed to load checkpoint: $e" checkpoint_id=checkpoint_id
        rethrow(e)
    end
end

"""
Restore model state from checkpoint
"""
function restore_model_state!(model, model_state::Dict{String, Any})
    if FLUX_AVAILABLE
        try
            # Get current model state
            current_state = Flux.state(model)
            
            # Restore parameters
            for (name, saved_param) in model_state
                if haskey(current_state, Symbol(name))
                    current_param = current_state[Symbol(name)]
                    if current_param isa AbstractArray && saved_param isa AbstractArray
                        # Move to appropriate device and copy
                        if current_param isa CuArray
                            current_param .= CuArray(saved_param)
                        else
                            current_param .= saved_param
                        end
                    end
                end
            end
            
            @info "Model state restored successfully"
            
        catch e
            @warn "Failed to restore model state: $e"
            # Fallback restoration method
            try
                restore_model_state_fallback!(model, model_state)
            catch e2
                @error "Fallback model restoration also failed: $e2"
            end
        end
    else
        # Handle custom models without Flux
        if hasfield(typeof(model), :weights) && model.weights isa Dict
            # Direct restoration for custom models
            for (name, saved_param) in model_state
                if haskey(model.weights, name)
                    model.weights[name] = copy(saved_param)
                end
            end
            @info "Custom model state restored successfully"
        else
            @warn "No Flux available and model doesn't support direct state restoration"
        end
    end
end

"""
Fallback method for restoring model state
"""
function restore_model_state_fallback!(model, model_state::Dict{String, Any})
    if hasfield(typeof(model), :layers)
        # Manual restoration for common layer types
        for (i, layer) in enumerate(model.layers)
            weight_key = "layer_$(i)_weight"
            bias_key = "layer_$(i)_bias"
            
            if haskey(model_state, weight_key) && hasfield(typeof(layer), :weight)
                saved_weight = model_state[weight_key]
                if layer.weight isa CuArray
                    layer.weight .= CuArray(saved_weight)
                else
                    layer.weight .= saved_weight
                end
            end
            
            if haskey(model_state, bias_key) && hasfield(typeof(layer), :bias) && layer.bias !== nothing
                saved_bias = model_state[bias_key]
                if layer.bias isa CuArray
                    layer.bias .= CuArray(saved_bias)
                else
                    layer.bias .= saved_bias
                end
            end
        end
    else
        @warn "Fallback restoration not possible - model has no layers field"
    end
end

"""
Restore optimizer state from checkpoint
"""
function restore_optimizer_state!(optimizer, optimizer_state::Dict{String, Any})
    try
        if haskey(optimizer_state, "learning_rate")
            if hasfield(typeof(optimizer), :eta)
                optimizer.eta = optimizer_state["learning_rate"]
            elseif hasfield(typeof(optimizer), :α)
                optimizer.α = optimizer_state["learning_rate"]
            end
        end
        
        if haskey(optimizer_state, "momentum") && hasfield(typeof(optimizer), :momentum)
            optimizer.momentum = optimizer_state["momentum"]
        end
        
        if haskey(optimizer_state, "beta") && hasfield(typeof(optimizer), :beta)
            optimizer.beta = optimizer_state["beta"]
        end
        
        if haskey(optimizer_state, "optimizer_state") && hasfield(typeof(optimizer), :state)
            optimizer.state = optimizer_state["optimizer_state"]
        end
        
        @info "Optimizer state restored successfully"
        
    catch e
        @warn "Failed to restore optimizer state: $e"
    end
end

"""
Restore replay buffer state from checkpoint
"""
function restore_replay_buffer_state!(replay_buffer, replay_buffer_state::Dict{String, Any})
    try
        if haskey(replay_buffer_state, "buffer_data") && hasfield(typeof(replay_buffer), :buffer)
            saved_buffer = replay_buffer_state["buffer_data"]
            if replay_buffer.buffer isa CuArray
                replay_buffer.buffer .= CuArray(saved_buffer)
            else
                replay_buffer.buffer .= saved_buffer
            end
        end
        
        if haskey(replay_buffer_state, "buffer_size") && hasfield(typeof(replay_buffer), :size)
            replay_buffer.size = replay_buffer_state["buffer_size"]
        end
        
        if haskey(replay_buffer_state, "buffer_head") && hasfield(typeof(replay_buffer), :head)
            replay_buffer.head = replay_buffer_state["buffer_head"]
        end
        
        if haskey(replay_buffer_state, "buffer_tail") && hasfield(typeof(replay_buffer), :tail)
            replay_buffer.tail = replay_buffer_state["buffer_tail"]
        end
        
        @info "Replay buffer state restored successfully"
        
    catch e
        @warn "Failed to restore replay buffer state: $e"
    end
end

"""
List available checkpoints
"""
function list_checkpoints(manager::CheckpointManager; sort_by::Symbol = :timestamp)
    checkpoints = collect(values(manager.checkpoint_registry))
    
    if sort_by == :timestamp
        sort!(checkpoints, by = c -> c.timestamp, rev = true)
    elseif sort_by == :iteration
        sort!(checkpoints, by = c -> c.iteration, rev = true)
    elseif sort_by == :correlation
        sort!(checkpoints, by = c -> c.correlation_score, rev = true)
    elseif sort_by == :loss
        sort!(checkpoints, by = c -> c.training_loss)
    end
    
    return checkpoints
end

"""
Cleanup old checkpoints keeping only the most recent
"""
function cleanup_old_checkpoints!(manager::CheckpointManager)
    if length(manager.checkpoint_registry) <= manager.config.max_checkpoints
        return
    end
    
    # Sort by timestamp and keep only the most recent
    sorted_checkpoints = sort(collect(values(manager.checkpoint_registry)), 
                             by = c -> c.timestamp, rev = true)
    
    checkpoints_to_remove = sorted_checkpoints[(manager.config.max_checkpoints + 1):end]
    
    for metadata in checkpoints_to_remove
        checkpoint_id = metadata.checkpoint_id
        
        # Remove files
        checkpoint_file = joinpath(manager.config.checkpoint_dir, "$checkpoint_id.jls")
        metadata_file = joinpath(manager.config.checkpoint_dir, "metadata", "$checkpoint_id.json")
        
        try
            if isfile(checkpoint_file)
                rm(checkpoint_file)
            end
            if isfile(metadata_file)
                rm(metadata_file)
            end
            
            # Remove from registry
            delete!(manager.checkpoint_registry, checkpoint_id)
            
            @info "Cleaned up old checkpoint: $checkpoint_id"
            
        catch e
            @warn "Failed to cleanup checkpoint $checkpoint_id: $e"
        end
    end
    
    # Save updated registry
    save_registry(manager)
end

"""
Rollback to a specific checkpoint
"""
function rollback_to_checkpoint!(
    manager::CheckpointManager,
    checkpoint_id::String,
    model,
    optimizer,
    replay_buffer = nothing
)
    @info "Rolling back to checkpoint: $checkpoint_id"
    
    # Load the checkpoint
    checkpoint_data = load_checkpoint!(manager, checkpoint_id, model, optimizer, replay_buffer)
    
    # Update manager state
    manager.previous_model_hash = checkpoint_data.metadata.model_hash
    manager.weight_deltas = copy(checkpoint_data.model_state)
    
    @info "Rollback completed successfully" checkpoint_id=checkpoint_id
    
    return checkpoint_data
end

"""
Validate checkpoint integrity
"""
function validate_checkpoint(checkpoint_file::String)
    try
        # Check if file exists and is readable
        if !isfile(checkpoint_file)
            return false
        end
        
        # Check file size
        file_size = filesize(checkpoint_file)
        if file_size == 0
            return false
        end
        
        # Try to read and deserialize
        checkpoint_data_raw = read(checkpoint_file)
        
        # Check for gzip header if compressed
        if length(checkpoint_data_raw) >= 2 && checkpoint_data_raw[1] == 0x1f && checkpoint_data_raw[2] == 0x8b
            # Decompress
            try
                checkpoint_data_raw = decompress_data(checkpoint_data_raw)
            catch
                return false
            end
        end
        
        # Try to deserialize
        io = IOBuffer(checkpoint_data_raw)
        checkpoint_data = deserialize(io)
        
        # Basic structure validation
        if !isa(checkpoint_data, CheckpointData)
            return false
        end
        
        # Check required fields
        if isempty(checkpoint_data.model_state)
            return false
        end
        
        return true
        
    catch e
        @warn "Checkpoint validation failed: $e"
        return false
    end
end

"""
Get checkpoint information
"""
function get_checkpoint_info(manager::CheckpointManager, checkpoint_id::String)
    if !haskey(manager.checkpoint_registry, checkpoint_id)
        return nothing
    end
    
    metadata = manager.checkpoint_registry[checkpoint_id]
    
    info = Dict(
        "checkpoint_id" => metadata.checkpoint_id,
        "timestamp" => metadata.timestamp,
        "iteration" => metadata.iteration,
        "epoch" => metadata.epoch,
        "training_loss" => metadata.training_loss,
        "validation_loss" => metadata.validation_loss,
        "correlation_score" => metadata.correlation_score,
        "learning_rate" => metadata.learning_rate,
        "model_size_mb" => round(metadata.model_size_bytes / 1024^2, digits=2),
        "checkpoint_size_mb" => round(metadata.checkpoint_size_bytes / 1024^2, digits=2),
        "compression_ratio" => round(metadata.compression_ratio, digits=3),
        "save_duration_ms" => round(metadata.save_duration_ms, digits=1),
        "validation_passed" => metadata.validation_passed,
        "parameter_count" => metadata.parameter_count,
        "tags" => metadata.tags,
        "notes" => metadata.notes
    )
    
    return info
end

"""
Compress existing checkpoint
"""
function compress_checkpoint!(manager::CheckpointManager, checkpoint_id::String)
    checkpoint_file = joinpath(manager.config.checkpoint_dir, "$checkpoint_id.jls")
    
    if !isfile(checkpoint_file)
        throw(ArgumentError("Checkpoint file not found: $checkpoint_file"))
    end
    
    # Read current data
    data = read(checkpoint_file)
    
    # Check if already compressed
    if length(data) >= 2 && data[1] == 0x1f && data[2] == 0x8b
        @info "Checkpoint already compressed: $checkpoint_id"
        return
    end
    
    # Compress data
    compressed_data = compress_data(data, manager.config.compression_level)
    
    # Update metadata
    if haskey(manager.checkpoint_registry, checkpoint_id)
        metadata = manager.checkpoint_registry[checkpoint_id]
        metadata.compression_ratio = length(compressed_data) / length(data)
        metadata.checkpoint_size_bytes = length(compressed_data)
    end
    
    # Write compressed data
    write(checkpoint_file, compressed_data)
    
    # Update registry
    save_registry(manager)
    
    compression_ratio = length(compressed_data) / length(data)
    @info "Checkpoint compressed" checkpoint_id=checkpoint_id original_size_mb=round(length(data)/1024^2, digits=2) compressed_size_mb=round(length(compressed_data)/1024^2, digits=2) ratio=round(compression_ratio, digits=3)
end

"""
Get manager statistics
"""
function get_manager_stats(manager::CheckpointManager)
    active_saves_count = length(manager.active_saves)
    total_checkpoints = length(manager.checkpoint_registry)
    
    avg_save_time = manager.total_saves > 0 ? manager.total_save_time / manager.total_saves : 0.0
    total_compression_ratio = manager.bytes_saved > 0 ? manager.bytes_compressed / manager.bytes_saved : 1.0
    
    stats = Dict(
        "total_checkpoints" => total_checkpoints,
        "active_saves" => active_saves_count,
        "total_saves" => manager.total_saves,
        "avg_save_time_ms" => round(avg_save_time * 1000, digits=1),
        "total_bytes_saved_mb" => round(manager.bytes_saved / 1024^2, digits=2),
        "total_bytes_compressed_mb" => round(manager.bytes_compressed / 1024^2, digits=2),
        "overall_compression_ratio" => round(total_compression_ratio, digits=3),
        "disk_usage_mb" => round(sum(c.checkpoint_size_bytes for c in values(manager.checkpoint_registry)) / 1024^2, digits=2)
    )
    
    return stats
end

end # module ModelCheckpointing