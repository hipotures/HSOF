"""
Distributed Training Support for Metamodel
Multi-GPU training capabilities using data parallelism across dual RTX 4090 setup
"""
module DistributedTraining

using CUDA
using Flux
using Statistics
using Distributed
using SharedArrays
using Dates
using LinearAlgebra
using JLD2
using Printf

# Include dependencies
include("neural_architecture.jl")
include("experience_replay.jl")
include("online_learning.jl")

using .NeuralArchitecture: Metamodel, MetamodelConfig, create_metamodel_config, create_metamodel
using .ExperienceReplay: ReplayBuffer, sample_batch, get_buffer_stats
using .OnlineLearning: OnlineLearningConfig, OnlineLearningState, create_online_config

"""
Configuration for distributed training system
"""
mutable struct DistributedTrainingConfig
    # Multi-GPU settings
    gpu_devices::Vector{Int}             # GPU device IDs to use
    primary_gpu::Int                     # Primary GPU for coordination
    data_parallel::Bool                  # Use data parallelism
    
    # Gradient synchronization
    sync_method::Symbol                  # :nccl, :custom_reduce, :parameter_server
    sync_frequency::Int32                # Sync every N updates
    gradient_compression::Bool           # Compress gradients for communication
    compression_ratio::Float32           # Compression ratio (0.1-1.0)
    
    # Data sharding
    shard_replay_buffer::Bool           # Shard replay buffer across GPUs
    overlap_data_compute::Bool          # Overlap data loading with compute
    prefetch_batches::Int32             # Number of batches to prefetch
    
    # Load balancing
    dynamic_batch_sizing::Bool          # Adjust batch sizes based on GPU load
    load_balance_frequency::Int32       # Check load balance every N updates
    max_batch_size_ratio::Float32       # Max ratio between GPU batch sizes
    
    # Fault tolerance
    enable_fault_tolerance::Bool        # Enable fault tolerance
    checkpoint_frequency::Int32         # Checkpoint every N updates
    single_gpu_fallback::Bool          # Fall back to single GPU on failure
    heartbeat_interval::Float32         # Heartbeat interval in seconds
    
    # Performance optimization
    enable_mixed_precision::Bool        # Use FP16 for gradients
    enable_overlap_comm::Bool           # Overlap communication with computation
    pin_memory::Bool                    # Pin CPU memory for transfers
    async_gradient_copy::Bool           # Async gradient copying
end

"""
Create default distributed training configuration
"""
function create_distributed_config(;
    gpu_devices::Vector{Int} = [0, 1],
    primary_gpu::Int = 0,
    data_parallel::Bool = true,
    sync_method::Symbol = :custom_reduce,  # NCCL not available without NVLink
    sync_frequency::Int = 1,
    gradient_compression::Bool = false,
    compression_ratio::Float32 = 0.5f0,
    shard_replay_buffer::Bool = true,
    overlap_data_compute::Bool = true,
    prefetch_batches::Int = 2,
    dynamic_batch_sizing::Bool = true,
    load_balance_frequency::Int = 100,
    max_batch_size_ratio::Float32 = 2.0f0,
    enable_fault_tolerance::Bool = true,
    checkpoint_frequency::Int = 1000,
    single_gpu_fallback::Bool = true,
    heartbeat_interval::Float32 = 5.0f0,
    enable_mixed_precision::Bool = true,
    enable_overlap_comm::Bool = true,
    pin_memory::Bool = true,
    async_gradient_copy::Bool = true
)
    return DistributedTrainingConfig(
        gpu_devices,
        primary_gpu,
        data_parallel,
        sync_method,
        Int32(sync_frequency),
        gradient_compression,
        compression_ratio,
        shard_replay_buffer,
        overlap_data_compute,
        Int32(prefetch_batches),
        dynamic_batch_sizing,
        Int32(load_balance_frequency),
        max_batch_size_ratio,
        enable_fault_tolerance,
        Int32(checkpoint_frequency),
        single_gpu_fallback,
        heartbeat_interval,
        enable_mixed_precision,
        enable_overlap_comm,
        pin_memory,
        async_gradient_copy
    )
end

"""
Per-GPU training state
"""
mutable struct GPUTrainingState
    gpu_id::Int
    device::CuDevice
    model::Metamodel
    local_optimizer::Flux.Optimiser
    replay_buffer::Union{ReplayBuffer, Nothing}
    
    # Streams for async operations
    compute_stream::CuStream
    comm_stream::CuStream
    
    # Performance metrics
    local_updates::Int64
    avg_batch_time::Float32
    avg_sync_time::Float32
    throughput_samples_per_sec::Float32
    
    # Fault tolerance
    is_healthy::Bool
    last_heartbeat::Float64
    error_count::Int32
end

"""
Distributed training coordinator
"""
mutable struct DistributedTrainingCoordinator
    config::DistributedTrainingConfig
    gpu_states::Vector{GPUTrainingState}
    shared_model_params::SharedArray{Float32, 1}  # For parameter server approach
    
    # Synchronization
    sync_lock::ReentrantLock
    gradient_ready_flags::Vector{Bool}
    
    # Global statistics
    global_update_count::Int64
    total_samples_processed::Int64
    avg_scaling_efficiency::Float32
    
    # Fault tolerance
    coordinator_healthy::Bool
    failed_gpus::Set{Int}
    fallback_active::Bool
end

"""
Initialize distributed training system
"""
function initialize_distributed_training(
    base_model::Metamodel,
    replay_buffer::ReplayBuffer,
    dist_config::DistributedTrainingConfig = create_distributed_config(),
    online_config::OnlineLearningConfig = create_online_config()
)
    println("Initializing distributed training on $(length(dist_config.gpu_devices)) GPUs...")
    
    # Verify GPU availability
    available_gpus = collect(0:CUDA.ndevices()-1)
    for gpu_id in dist_config.gpu_devices
        if !(gpu_id in available_gpus)
            error("GPU $gpu_id not available. Available GPUs: $available_gpus")
        end
    end
    
    # Initialize GPU states
    gpu_states = GPUTrainingState[]
    
    for gpu_id in dist_config.gpu_devices
        println("  Initializing GPU $gpu_id...")
        
        # Set device
        CUDA.device!(gpu_id)
        device = CUDA.device()
        
        # Create model copy on this GPU
        model = deepcopy(base_model) |> gpu
        
        # Create local optimizer
        local_optimizer = Flux.Optimiser(
            Flux.AdamW(online_config.learning_rate, (0.9f0, 0.999f0), online_config.weight_decay),
            Flux.ClipGrad(online_config.gradient_clip)
        )
        
        # Create local replay buffer if sharding enabled
        local_buffer = if dist_config.shard_replay_buffer
            shard_replay_buffer(replay_buffer, gpu_id, length(dist_config.gpu_devices))
        else
            replay_buffer
        end
        
        # Create CUDA streams
        compute_stream = CuStream(flags=CUDA.STREAM_NON_BLOCKING)
        comm_stream = CuStream(flags=CUDA.STREAM_NON_BLOCKING)
        
        gpu_state = GPUTrainingState(
            gpu_id,
            device,
            model,
            local_optimizer,
            local_buffer,
            compute_stream,
            comm_stream,
            0,      # local_updates
            0.0f0,  # avg_batch_time
            0.0f0,  # avg_sync_time
            0.0f0,  # throughput_samples_per_sec
            true,   # is_healthy
            time(), # last_heartbeat
            0       # error_count
        )
        
        push!(gpu_states, gpu_state)
    end
    
    # Initialize shared parameters for parameter server
    total_params = count_parameters(base_model)
    shared_model_params = SharedArray{Float32}(total_params)
    
    # Copy initial model parameters to shared array
    flatten_model_params!(shared_model_params, base_model)
    
    # Initialize coordinator
    coordinator = DistributedTrainingCoordinator(
        dist_config,
        gpu_states,
        shared_model_params,
        ReentrantLock(),
        fill(false, length(gpu_states)),
        0,     # global_update_count
        0,     # total_samples_processed
        0.0f0, # avg_scaling_efficiency
        true,  # coordinator_healthy
        Set{Int}(),  # failed_gpus
        false  # fallback_active
    )
    
    println("Distributed training initialized successfully!")
    println("  - GPUs: $(dist_config.gpu_devices)")
    println("  - Sync method: $(dist_config.sync_method)")
    println("  - Data sharding: $(dist_config.shard_replay_buffer)")
    println("  - Total parameters: $total_params")
    
    return coordinator
end

"""
Shard replay buffer across GPUs
"""
function shard_replay_buffer(buffer::ReplayBuffer, gpu_id::Int, n_gpus::Int)
    # Create smaller buffer for this GPU
    shard_size = div(buffer.config.capacity, n_gpus)
    
    # Note: This is a simplified implementation
    # In practice, you'd need to modify the ReplayBuffer to support sharding
    return buffer  # Return original for now
end

"""
Perform one distributed training step
"""
function distributed_training_step!(
    coordinator::DistributedTrainingCoordinator;
    step_callback::Union{Nothing, Function} = nothing
)
    dist_config = coordinator.config
    
    # Check for failed GPUs if fault tolerance enabled
    if dist_config.enable_fault_tolerance
        check_gpu_health!(coordinator)
        
        if coordinator.fallback_active
            return single_gpu_fallback_step!(coordinator)
        end
    end
    
    # Determine active GPUs (excluding failed ones)
    active_gpus = [gpu for gpu in coordinator.gpu_states if gpu.is_healthy && !(gpu.gpu_id in coordinator.failed_gpus)]
    
    if isempty(active_gpus)
        @warn "No healthy GPUs available for training"
        return false
    end
    
    # Adjust batch sizes for load balancing if enabled
    if dist_config.dynamic_batch_sizing && coordinator.global_update_count % dist_config.load_balance_frequency == 0
        adjust_batch_sizes!(active_gpus, dist_config)
    end
    
    # Parallel forward and backward pass on each GPU
    training_futures = []
    
    for gpu_state in active_gpus
        future = @async begin
            try
                local_training_step!(gpu_state, dist_config)
            catch e
                @error "Training step failed on GPU $(gpu_state.gpu_id)" exception=e
                gpu_state.is_healthy = false
                gpu_state.error_count += 1
                return false
            end
        end
        push!(training_futures, future)
    end
    
    # Wait for all local training steps to complete
    local_results = [fetch(f) for f in training_futures]
    
    # Check if any GPU failed
    successful_gpus = active_gpus[local_results]
    
    if isempty(successful_gpus)
        @warn "All GPU training steps failed"
        return false
    end
    
    # Gradient synchronization
    if coordinator.global_update_count % dist_config.sync_frequency == 0
        sync_success = synchronize_gradients!(successful_gpus, dist_config)
        
        if !sync_success
            @warn "Gradient synchronization failed"
            return false
        end
    end
    
    # Update global statistics
    coordinator.global_update_count += 1
    coordinator.total_samples_processed += sum(length(gpu.replay_buffer.features) for gpu in successful_gpus if !isnothing(gpu.replay_buffer))
    
    # Calculate scaling efficiency
    if length(successful_gpus) > 1
        coordinator.avg_scaling_efficiency = calculate_scaling_efficiency(successful_gpus)
    end
    
    # Checkpoint if needed
    if dist_config.enable_fault_tolerance && 
       coordinator.global_update_count % dist_config.checkpoint_frequency == 0
        save_distributed_checkpoint(coordinator)
    end
    
    # Call callback if provided
    if !isnothing(step_callback)
        step_callback(coordinator)
    end
    
    return true
end

"""
Perform local training step on a single GPU
"""
function local_training_step!(gpu_state::GPUTrainingState, config::DistributedTrainingConfig)
    CUDA.device!(gpu_state.gpu_id)
    
    start_time = time()
    
    # Use compute stream for training
    CUDA.stream!(gpu_state.compute_stream) do
        # Sample batch from local replay buffer
        if isnothing(gpu_state.replay_buffer)
            return false
        end
        
        batch = sample_batch(gpu_state.replay_buffer, 32)  # Base batch size
        
        if isnothing(batch)
            return false
        end
        
        # Prepare inputs (same as online learning)
        features = batch.features
        actual_scores = batch.actual
        weights = batch.weights
        
        # Convert sparse features to dense input
        batch_inputs = prepare_batch_inputs_distributed(features, batch.n_features, gpu_state.model.config)
        
        # Forward and backward pass
        loss, grads = Flux.withgradient(gpu_state.model) do model
            predictions = model(batch_inputs)
            
            # MSE loss with importance weights
            mse = mean((predictions .- actual_scores).^2 .* weights)
            
            # Add regularization
            l2_reg = sum(sum(p.^2) for p in Flux.params(model)) * 1f-5
            
            mse + l2_reg
        end
        
        # Store gradients for synchronization
        gpu_state.model.gradients = grads[1]
        
        gpu_state.local_updates += 1
    end
    
    # Update timing statistics
    elapsed = time() - start_time
    gpu_state.avg_batch_time = 0.9f0 * gpu_state.avg_batch_time + 0.1f0 * Float32(elapsed)
    
    # Calculate throughput
    if gpu_state.avg_batch_time > 0
        gpu_state.throughput_samples_per_sec = 32.0f0 / gpu_state.avg_batch_time
    end
    
    return true
end

"""
Prepare batch inputs for distributed training
"""
function prepare_batch_inputs_distributed(
    feature_indices::CuArray{Int32, 2},
    n_features::CuArray{Int32, 1},
    model_config::MetamodelConfig
)
    max_features, batch_size = size(feature_indices)
    input_dim = model_config.input_dim
    
    # Create dense input matrix
    inputs = CUDA.zeros(Float32, input_dim, batch_size)
    
    # CUDA kernel for sparse to dense conversion
    function sparse_to_dense_kernel!(
        inputs::CuDeviceMatrix{Float32},
        indices::CuDeviceMatrix{Int32},
        n_feats::CuDeviceVector{Int32},
        input_dim::Int32
    )
        idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        batch_idx = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        
        if batch_idx <= size(inputs, 2) && idx <= size(indices, 1)
            if idx <= n_feats[batch_idx]
                feat_idx = indices[idx, batch_idx]
                if 1 <= feat_idx <= input_dim
                    @inbounds inputs[feat_idx, batch_idx] = 1.0f0
                end
            end
        end
        
        return nothing
    end
    
    threads = (32, 32)
    blocks = (cld(max_features, 32), cld(batch_size, 32))
    
    @cuda threads=threads blocks=blocks sparse_to_dense_kernel!(
        inputs, feature_indices, n_features, Int32(input_dim)
    )
    
    return inputs
end

"""
Synchronize gradients across GPUs
"""
function synchronize_gradients!(gpu_states::Vector{GPUTrainingState}, config::DistributedTrainingConfig)
    if config.sync_method == :custom_reduce
        return custom_reduce_gradients!(gpu_states, config)
    elseif config.sync_method == :parameter_server
        return parameter_server_sync!(gpu_states, config)
    else
        @warn "Unknown sync method: $(config.sync_method)"
        return false
    end
end

"""
Custom all-reduce implementation for gradient synchronization
"""
function custom_reduce_gradients!(gpu_states::Vector{GPUTrainingState}, config::DistributedTrainingConfig)
    n_gpus = length(gpu_states)
    
    if n_gpus <= 1
        return true
    end
    
    sync_start = time()
    
    try
        # Step 1: Collect gradients from all GPUs to primary GPU
        primary_gpu_state = gpu_states[1]  # Use first GPU as primary
        CUDA.device!(primary_gpu_state.gpu_id)
        
        # Initialize averaged gradients on primary GPU
        averaged_grads = deepcopy(primary_gpu_state.model.gradients)
        
        # Collect and sum gradients from other GPUs
        for i in 2:n_gpus
            gpu_state = gpu_states[i]
            
            # Copy gradients to primary GPU
            for (key, grad) in pairs(gpu_state.model.gradients)
                if haskey(averaged_grads, key) && !isnothing(grad)
                    # Transfer gradient to primary GPU and add
                    transferred_grad = grad |> CuArray{Float32}  # Ensure on GPU
                    averaged_grads[key] .+= transferred_grad
                end
            end
        end
        
        # Step 2: Average gradients
        for (key, grad) in pairs(averaged_grads)
            if !isnothing(grad)
                grad ./= Float32(n_gpus)
            end
        end
        
        # Step 3: Broadcast averaged gradients back to all GPUs and apply updates
        for gpu_state in gpu_states
            CUDA.device!(gpu_state.gpu_id)
            
            # Copy averaged gradients to this GPU
            local_grads = deepcopy(averaged_grads)
            
            # Apply gradients using local optimizer
            Flux.update!(gpu_state.local_optimizer, gpu_state.model, local_grads)
        end
        
        sync_time = Float32(time() - sync_start)
        
        # Update sync timing for all GPUs
        for gpu_state in gpu_states
            gpu_state.avg_sync_time = 0.9f0 * gpu_state.avg_sync_time + 0.1f0 * sync_time
        end
        
        return true
        
    catch e
        @error "Gradient synchronization failed" exception=e
        return false
    end
end

"""
Parameter server approach for gradient synchronization
"""
function parameter_server_sync!(gpu_states::Vector{GPUTrainingState}, config::DistributedTrainingConfig)
    # Simplified parameter server implementation
    # In practice, this would involve more sophisticated coordination
    return custom_reduce_gradients!(gpu_states, config)
end

"""
Flatten model parameters to vector
"""
function flatten_model_params!(dest::SharedArray{Float32, 1}, model::Metamodel)
    idx = 1
    for p in Flux.params(model)
        p_flat = vec(Array(p))
        n = length(p_flat)
        dest[idx:idx+n-1] = p_flat
        idx += n
    end
end

"""
Check GPU health and handle failures
"""
function check_gpu_health!(coordinator::DistributedTrainingCoordinator)
    current_time = time()
    
    for gpu_state in coordinator.gpu_states
        # Check heartbeat
        if current_time - gpu_state.last_heartbeat > coordinator.config.heartbeat_interval * 2
            @warn "GPU $(gpu_state.gpu_id) heartbeat timeout"
            gpu_state.is_healthy = false
            push!(coordinator.failed_gpus, gpu_state.gpu_id)
        end
        
        # Check error count
        if gpu_state.error_count > 5
            @warn "GPU $(gpu_state.gpu_id) has too many errors"
            gpu_state.is_healthy = false
            push!(coordinator.failed_gpus, gpu_state.gpu_id)
        end
        
        # Update heartbeat for healthy GPUs
        if gpu_state.is_healthy
            gpu_state.last_heartbeat = current_time
        end
    end
    
    # Activate fallback if too many GPUs failed
    healthy_gpus = sum(gpu.is_healthy for gpu in coordinator.gpu_states)
    
    if coordinator.config.single_gpu_fallback && healthy_gpus <= 1
        coordinator.fallback_active = true
        @warn "Activating single GPU fallback mode"
    end
end

"""
Single GPU fallback training step
"""
function single_gpu_fallback_step!(coordinator::DistributedTrainingCoordinator)
    # Find a healthy GPU
    healthy_gpu = findfirst(gpu -> gpu.is_healthy, coordinator.gpu_states)
    
    if isnothing(healthy_gpu)
        @error "No healthy GPUs available for fallback"
        return false
    end
    
    gpu_state = coordinator.gpu_states[healthy_gpu]
    
    # Perform normal single-GPU training step
    return local_training_step!(gpu_state, coordinator.config)
end

"""
Adjust batch sizes for load balancing
"""
function adjust_batch_sizes!(gpu_states::Vector{GPUTrainingState}, config::DistributedTrainingConfig)
    # Calculate relative performance of each GPU
    throughputs = [gpu.throughput_samples_per_sec for gpu in gpu_states]
    avg_throughput = mean(throughputs)
    
    if avg_throughput <= 0
        return  # No data yet
    end
    
    # Adjust batch sizes based on relative performance
    for gpu_state in gpu_states
        relative_perf = gpu_state.throughput_samples_per_sec / avg_throughput
        relative_perf = clamp(relative_perf, 1.0f0 / config.max_batch_size_ratio, config.max_batch_size_ratio)
        
        # This would require modifying the batch sampling logic
        # For now, we just log the adjustment
        @debug "GPU $(gpu_state.gpu_id) relative performance: $(round(relative_perf, digits=2))"
    end
end

"""
Calculate scaling efficiency
"""
function calculate_scaling_efficiency(gpu_states::Vector{GPUTrainingState})
    if length(gpu_states) <= 1
        return 1.0f0
    end
    
    # Calculate actual throughput vs ideal throughput
    total_throughput = sum(gpu.throughput_samples_per_sec for gpu in gpu_states)
    single_gpu_throughput = gpu_states[1].throughput_samples_per_sec
    
    if single_gpu_throughput <= 0
        return 0.0f0
    end
    
    ideal_throughput = single_gpu_throughput * length(gpu_states)
    efficiency = total_throughput / ideal_throughput
    
    return clamp(efficiency, 0.0f0, 1.0f0)
end

"""
Save distributed training checkpoint
"""
function save_distributed_checkpoint(coordinator::DistributedTrainingCoordinator)
    checkpoint_dir = "checkpoints/distributed"
    mkpath(checkpoint_dir)
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    checkpoint_path = joinpath(checkpoint_dir, "checkpoint_$timestamp.jld2")
    
    try
        # Save primary model state
        primary_gpu = coordinator.gpu_states[1]
        
        checkpoint = Dict(
            "model_state" => Flux.state(primary_gpu.model),
            "global_update_count" => coordinator.global_update_count,
            "total_samples_processed" => coordinator.total_samples_processed,
            "failed_gpus" => collect(coordinator.failed_gpus),
            "avg_scaling_efficiency" => coordinator.avg_scaling_efficiency,
            "gpu_statistics" => [(
                gpu_id = gpu.gpu_id,
                local_updates = gpu.local_updates,
                avg_batch_time = gpu.avg_batch_time,
                avg_sync_time = gpu.avg_sync_time,
                throughput = gpu.throughput_samples_per_sec,
                is_healthy = gpu.is_healthy,
                error_count = gpu.error_count
            ) for gpu in coordinator.gpu_states]
        )
        
        @save checkpoint_path checkpoint
        @info "Distributed checkpoint saved to $checkpoint_path"
        
    catch e
        @error "Failed to save distributed checkpoint" exception=e
    end
end

"""
Get distributed training statistics
"""
function get_distributed_stats(coordinator::DistributedTrainingCoordinator)
    healthy_gpus = [gpu for gpu in coordinator.gpu_states if gpu.is_healthy]
    
    total_throughput = sum(gpu.throughput_samples_per_sec for gpu in healthy_gpus)
    avg_batch_time = mean(gpu.avg_batch_time for gpu in healthy_gpus)
    avg_sync_time = mean(gpu.avg_sync_time for gpu in healthy_gpus)
    
    return (
        global_update_count = coordinator.global_update_count,
        total_samples_processed = coordinator.total_samples_processed,
        healthy_gpus = length(healthy_gpus),
        failed_gpus = length(coordinator.failed_gpus),
        total_throughput = total_throughput,
        avg_batch_time = avg_batch_time,
        avg_sync_time = avg_sync_time,
        scaling_efficiency = coordinator.avg_scaling_efficiency,
        fallback_active = coordinator.fallback_active
    )
end

"""
Shutdown distributed training system
"""
function shutdown_distributed_training!(coordinator::DistributedTrainingCoordinator)
    @info "Shutting down distributed training system..."
    
    # Save final checkpoint
    if coordinator.config.enable_fault_tolerance
        save_distributed_checkpoint(coordinator)
    end
    
    # Clean up GPU resources
    for gpu_state in coordinator.gpu_states
        try
            CUDA.device!(gpu_state.gpu_id)
            # Free streams
            CUDA.unsafe_free!(gpu_state.compute_stream)
            CUDA.unsafe_free!(gpu_state.comm_stream)
        catch e
            @warn "Error cleaning up GPU $(gpu_state.gpu_id)" exception=e
        end
    end
    
    @info "Distributed training shutdown complete"
end

# Export types and functions
export DistributedTrainingConfig, create_distributed_config
export DistributedTrainingCoordinator, initialize_distributed_training
export distributed_training_step!, get_distributed_stats
export shutdown_distributed_training!

end # module DistributedTraining