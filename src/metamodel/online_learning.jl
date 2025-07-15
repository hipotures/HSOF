module OnlineLearning

using CUDA
using Flux
using Statistics
using Dates
using LinearAlgebra
using ChainRulesCore
using JLD2

# Include dependencies
include("neural_architecture.jl")
include("experience_replay.jl")

using .NeuralArchitecture: Metamodel, MetamodelConfig, create_metamodel_config, create_metamodel, MultiHeadAttention
using .ExperienceReplay: ReplayBuffer, sample_batch, get_buffer_stats, update_sum_tree!

"""
Configuration for online learning system
"""
mutable struct OnlineLearningConfig
    # Training parameters
    batch_size::Int32
    update_frequency::Int32          # Update every N MCTS iterations
    learning_rate::Float32
    weight_decay::Float32
    gradient_clip::Float32
    
    # Double buffering
    use_double_buffering::Bool
    
    # Learning rate scheduling
    lr_scheduler::Symbol             # :constant, :exponential, :polynomial, :adaptive
    lr_decay_rate::Float32
    lr_min::Float32
    
    # Gradient accumulation
    accumulation_steps::Int32
    
    # Performance tracking
    correlation_window::Int32        # Window for tracking accuracy
    min_correlation::Float32         # Minimum acceptable correlation
    
    # CUDA streams
    n_cuda_streams::Int32
end

"""
Create default online learning configuration
"""
function create_online_config(;
    batch_size::Int = 32,
    update_frequency::Int = 100,
    learning_rate::Float32 = 1f-4,
    weight_decay::Float32 = 1f-5,
    gradient_clip::Float32 = 1.0f0,
    use_double_buffering::Bool = true,
    lr_scheduler::Symbol = :adaptive,
    lr_decay_rate::Float32 = 0.95f0,
    lr_min::Float32 = 1f-6,
    accumulation_steps::Int = 4,
    correlation_window::Int = 1000,
    min_correlation::Float32 = 0.9f0,
    n_cuda_streams::Int = 2
)
    return OnlineLearningConfig(
        Int32(batch_size),
        Int32(update_frequency),
        learning_rate,
        weight_decay,
        gradient_clip,
        use_double_buffering,
        lr_scheduler,
        lr_decay_rate,
        lr_min,
        Int32(accumulation_steps),
        Int32(correlation_window),
        min_correlation,
        Int32(n_cuda_streams)
    )
end

"""
Online learning state tracking
"""
mutable struct OnlineLearningState
    # Model states
    primary_model::Metamodel
    inference_model::Union{Nothing, Metamodel}  # For double buffering
    
    # Optimizer
    optimizer::Flux.Optimiser
    current_lr::Float32
    
    # CUDA streams
    training_stream::CuStream
    inference_stream::CuStream
    
    # Gradient accumulation
    accumulated_grads::Union{Nothing, NamedTuple}
    accumulation_count::Int32
    
    # Statistics
    iteration_count::Int64
    update_count::Int64
    recent_predictions::CuArray{Float32, 1}
    recent_actuals::CuArray{Float32, 1}
    recent_idx::Int32
    correlation_history::Vector{Float32}
    
    # Timing
    last_update_time::Float64
    total_training_time::Float64
end

"""
Initialize online learning system
"""
function initialize_online_learning(
    model::Metamodel,
    config::OnlineLearningConfig = create_online_config()
)
    # Create optimizer with weight decay
    optimizer = Flux.Optimiser(
        Flux.AdamW(config.learning_rate, (0.9f0, 0.999f0), config.weight_decay),
        Flux.ClipGrad(config.gradient_clip)
    )
    
    # Initialize CUDA streams
    training_stream = CuStream(flags=CUDA.STREAM_NON_BLOCKING)
    inference_stream = CuStream(flags=CUDA.STREAM_NON_BLOCKING)
    
    # Create inference model copy if using double buffering
    inference_model = if config.use_double_buffering
        # Deep copy model to GPU
        deepcopy(model)
    else
        nothing
    end
    
    # Initialize tracking arrays
    window_size = config.correlation_window
    recent_predictions = CUDA.zeros(Float32, window_size)
    recent_actuals = CUDA.zeros(Float32, window_size)
    
    return OnlineLearningState(
        model,
        inference_model,
        optimizer,
        config.learning_rate,
        training_stream,
        inference_stream,
        nothing,  # accumulated_grads
        Int32(0),
        0,
        0,
        recent_predictions,
        recent_actuals,
        Int32(1),
        Float32[],
        time(),
        0.0
    )
end

"""
Get model for inference (uses inference copy if double buffering)
"""
function get_inference_model(state::OnlineLearningState, config::OnlineLearningConfig)
    if config.use_double_buffering && !isnothing(state.inference_model)
        return state.inference_model
    else
        return state.primary_model
    end
end

"""
Perform online learning update
"""
function online_update!(
    state::OnlineLearningState,
    replay_buffer::ReplayBuffer,
    config::OnlineLearningConfig
)
    # Check if update is needed
    if state.iteration_count % config.update_frequency != 0
        state.iteration_count += 1
        return false
    end
    
    # Use training stream
    CUDA.stream!(state.training_stream) do
        # Sample mini-batch from replay buffer
        batch = sample_batch(replay_buffer, Int(config.batch_size))
        
        if isnothing(batch)
            return false
        end
        
        # Prepare inputs
        features = batch.features
        actual_scores = batch.actual
        weights = batch.weights
        
        # Convert sparse features to dense input
        batch_inputs = prepare_batch_inputs(features, batch.n_features, state.primary_model.config)
        
        # Compute loss and gradients
        loss, grads = Flux.withgradient(state.primary_model) do model
            predictions = model(batch_inputs)
            
            # MSE loss with importance weights
            mse = mean((predictions .- actual_scores).^2 .* weights)
            
            # Add correlation penalty if needed
            correlation = compute_correlation(predictions, actual_scores)
            correlation_penalty = max(0, config.min_correlation - correlation) * 0.1f0
            
            mse + correlation_penalty
        end
        
        # Accumulate gradients if configured
        if config.accumulation_steps > 1
            if isnothing(state.accumulated_grads)
                state.accumulated_grads = grads[1]
                state.accumulation_count = 1
            else
                # Add gradients
                accumulate_gradients!(state.accumulated_grads, grads[1])
                state.accumulation_count += 1
            end
            
            # Apply accumulated gradients if reached steps
            if state.accumulation_count >= config.accumulation_steps
                # Average accumulated gradients
                scale_gradients!(state.accumulated_grads, 1.0f0 / Float32(state.accumulation_count))
                
                # Update weights
                Flux.update!(state.optimizer, state.primary_model, state.accumulated_grads)
                
                # Reset accumulation
                state.accumulated_grads = nothing
                state.accumulation_count = 0
                
                state.update_count += 1
            end
        else
            # Direct update without accumulation
            Flux.update!(state.optimizer, state.primary_model, grads[1])
            state.update_count += 1
        end
        
        # Update learning rate if needed
        update_learning_rate!(state, config, loss)
        
        # Track predictions for correlation monitoring
        update_correlation_tracking!(state, batch_inputs, actual_scores, config)
        
        # Copy weights to inference model if double buffering
        if config.use_double_buffering && state.update_count % 10 == 0
            sync_inference_model!(state)
        end
        
        # Update timing
        current_time = time()
        state.total_training_time += current_time - state.last_update_time
        state.last_update_time = current_time
    end
    
    state.iteration_count += 1
    return true
end

"""
Prepare batch inputs from sparse feature indices
"""
function prepare_batch_inputs(
    feature_indices::CuArray{Int32, 2},
    n_features::CuArray{Int32, 1},
    model_config::MetamodelConfig
)
    max_features, batch_size = size(feature_indices)
    input_dim = model_config.input_dim
    
    # Create dense input matrix
    inputs = CUDA.zeros(Float32, input_dim, batch_size)
    
    # Convert to dense representation
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
Compute correlation between predictions and actuals
"""
function compute_correlation(predictions::CuArray{Float32}, actuals::CuArray{Float32})
    # Move to CPU for correlation calculation
    pred_cpu = Array(predictions)
    actual_cpu = Array(actuals)
    
    # Pearson correlation
    if length(pred_cpu) > 1
        return cor(pred_cpu, actual_cpu)
    else
        return 0.0f0
    end
end

"""
Accumulate gradients element-wise
"""
function accumulate_gradients!(accumulated, new_grads)
    for (key, grad) in pairs(new_grads)
        if haskey(accumulated, key) && !isnothing(grad)
            accumulated[key] .+= grad
        end
    end
end

"""
Scale gradients by factor
"""
function scale_gradients!(grads, scale::Float32)
    for (key, grad) in pairs(grads)
        if !isnothing(grad)
            grad .*= scale
        end
    end
end

"""
Update learning rate based on schedule
"""
function update_learning_rate!(state::OnlineLearningState, config::OnlineLearningConfig, loss::Float32)
    if config.lr_scheduler == :constant
        return
    elseif config.lr_scheduler == :exponential
        # Decay every 1000 updates
        if state.update_count % 1000 == 0
            state.current_lr *= config.lr_decay_rate
            state.current_lr = max(state.current_lr, config.lr_min)
            update_optimizer_lr!(state.optimizer, state.current_lr)
        end
    elseif config.lr_scheduler == :polynomial
        # Polynomial decay
        progress = Float32(state.update_count) / 10000.0f0
        decay = (1.0f0 - progress)^2
        state.current_lr = config.learning_rate * decay
        state.current_lr = max(state.current_lr, config.lr_min)
        update_optimizer_lr!(state.optimizer, state.current_lr)
    elseif config.lr_scheduler == :adaptive
        # Adaptive based on correlation
        if length(state.correlation_history) >= 10
            recent_corr = mean(state.correlation_history[end-9:end])
            if recent_corr < config.min_correlation
                # Increase learning rate to adapt faster
                state.current_lr = min(state.current_lr * 1.1f0, config.learning_rate)
            else
                # Decay normally
                state.current_lr *= config.lr_decay_rate
            end
            state.current_lr = max(state.current_lr, config.lr_min)
            update_optimizer_lr!(state.optimizer, state.current_lr)
        end
    end
end

"""
Update optimizer learning rate
"""
function update_optimizer_lr!(opt::Flux.Optimiser, new_lr::Float32)
    for o in opt
        if o isa Flux.AdamW
            o.eta = new_lr
        end
    end
end

"""
Update correlation tracking
"""
function update_correlation_tracking!(
    state::OnlineLearningState,
    inputs::CuArray{Float32, 2},
    actuals::CuArray{Float32, 1},
    config::OnlineLearningConfig
)
    # Get predictions
    predictions = state.primary_model(inputs)
    
    # Update circular buffers
    batch_size = length(actuals)
    window_size = config.correlation_window
    
    for i in 1:batch_size
        idx = mod1(state.recent_idx + i - 1, window_size)
        CUDA.@allowscalar state.recent_predictions[idx] = predictions[i]
        CUDA.@allowscalar state.recent_actuals[idx] = actuals[i]
    end
    
    state.recent_idx = mod1(state.recent_idx + batch_size, window_size)
    
    # Compute correlation periodically
    if state.update_count % 10 == 0
        valid_count = min(state.iteration_count, window_size)
        if valid_count >= 10
            correlation = compute_correlation(
                state.recent_predictions[1:valid_count],
                state.recent_actuals[1:valid_count]
            )
            push!(state.correlation_history, correlation)
            
            # Keep history bounded
            if length(state.correlation_history) > 1000
                popfirst!(state.correlation_history)
            end
        end
    end
end

"""
Synchronize inference model with primary model
"""
function sync_inference_model!(state::OnlineLearningState)
    if isnothing(state.inference_model)
        return
    end
    
    # Use inference stream for copy
    CUDA.stream!(state.inference_stream) do
        # Copy weights from primary to inference model
        copy_model_weights!(state.inference_model, state.primary_model)
    end
end

"""
Copy weights between models
"""
function copy_model_weights!(dest::Metamodel, src::Metamodel)
    # Copy each layer's parameters
    Flux.loadmodel!(dest, Flux.state(src))
end

"""
Asynchronous training step for concurrent execution
"""
function async_training_step!(
    state::OnlineLearningState,
    replay_buffer::ReplayBuffer,
    config::OnlineLearningConfig;
    callback::Union{Nothing, Function} = nothing
)
    # Launch training on separate stream
    @async begin
        try
            updated = online_update!(state, replay_buffer, config)
            
            if updated && !isnothing(callback)
                callback(state)
            end
        catch e
            @warn "Error in async training step" exception=e
        end
    end
end

"""
Get online learning statistics
"""
function get_online_stats(state::OnlineLearningState)
    avg_correlation = if length(state.correlation_history) > 0
        mean(state.correlation_history)
    else
        0.0f0
    end
    
    recent_correlation = if length(state.correlation_history) >= 10
        mean(state.correlation_history[end-9:end])
    else
        avg_correlation
    end
    
    return (
        iteration_count = state.iteration_count,
        update_count = state.update_count,
        current_lr = state.current_lr,
        avg_correlation = avg_correlation,
        recent_correlation = recent_correlation,
        total_training_time = state.total_training_time,
        updates_per_second = state.update_count / max(1.0, state.total_training_time)
    )
end

"""
Save model checkpoint
"""
function save_checkpoint(
    state::OnlineLearningState,
    filepath::String;
    include_optimizer::Bool = true,
    include_buffer::Bool = false
)
    checkpoint = Dict(
        "model_state" => Flux.state(state.primary_model),
        "iteration_count" => state.iteration_count,
        "update_count" => state.update_count,
        "correlation_history" => state.correlation_history,
        "total_training_time" => state.total_training_time
    )
    
    if include_optimizer
        checkpoint["optimizer_state"] = Flux.state(state.optimizer)
        checkpoint["current_lr"] = state.current_lr
    end
    
    # Save using JLD2 or BSON
    @save filepath checkpoint
end

"""
Load model checkpoint
"""
function load_checkpoint!(
    state::OnlineLearningState,
    filepath::String
)
    @load filepath checkpoint
    
    # Restore model weights
    Flux.loadmodel!(state.primary_model, checkpoint["model_state"])
    
    # Restore training state
    state.iteration_count = checkpoint["iteration_count"]
    state.update_count = checkpoint["update_count"]
    state.correlation_history = checkpoint["correlation_history"]
    state.total_training_time = checkpoint["total_training_time"]
    
    if haskey(checkpoint, "optimizer_state")
        Flux.loadmodel!(state.optimizer, checkpoint["optimizer_state"])
    end
    
    if haskey(checkpoint, "current_lr")
        state.current_lr = checkpoint["current_lr"]
        update_optimizer_lr!(state.optimizer, state.current_lr)
    end
    
    # Sync inference model if using double buffering
    sync_inference_model!(state)
end

# Export types and functions
export OnlineLearningConfig, create_online_config
export OnlineLearningState, initialize_online_learning
export get_inference_model, online_update!
export async_training_step!, get_online_stats
export save_checkpoint, load_checkpoint!
export prepare_batch_inputs

end # module OnlineLearning