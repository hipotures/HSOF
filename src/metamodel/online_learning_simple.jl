module OnlineLearningSimple

using CUDA
using Flux
using Statistics
using JLD2

# Include dependencies
include("neural_architecture.jl")
include("experience_replay.jl")

using .NeuralArchitecture: Metamodel, MetamodelConfig, create_metamodel_config, create_gpu_metamodel
using .ExperienceReplay: ReplayBuffer, sample_batch, update_sum_tree!

"""
Simple online learning configuration
"""
struct SimpleOnlineConfig
    batch_size::Int
    learning_rate::Float32
    update_frequency::Int
end

"""
Simple online learning state
"""
mutable struct SimpleOnlineState
    model::Metamodel
    optimizer::Flux.Optimiser
    iteration_count::Int
    update_count::Int
end

"""
Initialize simple online learning
"""
function init_online_learning(
    model::Metamodel,
    lr::Float32 = 1f-4
)
    optimizer = Flux.Optimiser(
        Flux.Adam(lr),
        Flux.ClipGrad(1.0f0)
    )
    
    return SimpleOnlineState(model, optimizer, 0, 0)
end

"""
Perform simple online update
"""
function simple_update!(
    state::SimpleOnlineState,
    replay_buffer::ReplayBuffer,
    config::SimpleOnlineConfig
)
    state.iteration_count += 1
    
    # Check update frequency
    if state.iteration_count % config.update_frequency != 0
        return false
    end
    
    # Sample batch
    batch = sample_batch(replay_buffer, config.batch_size)
    if isnothing(batch)
        return false
    end
    
    # Prepare inputs
    n_batch = length(batch.n_features)
    input_dim = state.model.config.input_dim
    
    # Convert sparse to dense on CPU
    inputs_cpu = zeros(Float32, input_dim, n_batch)
    features_cpu = Array(batch.features)
    n_features_cpu = Array(batch.n_features)
    
    for j in 1:n_batch
        for i in 1:n_features_cpu[j]
            feat_idx = features_cpu[i, j]
            if 1 <= feat_idx <= input_dim
                inputs_cpu[feat_idx, j] = 1.0f0
            end
        end
    end
    
    # Move to GPU
    inputs = CuArray(inputs_cpu)
    
    actuals = batch.actual
    
    # Compute gradients
    loss, grads = Flux.withgradient(state.model) do m
        preds = m(inputs)
        Flux.mse(preds, actuals)
    end
    
    # Update
    Flux.update!(state.optimizer, state.model, grads[1])
    state.update_count += 1
    
    return true
end

# Export
export SimpleOnlineConfig, SimpleOnlineState
export init_online_learning, simple_update!

end # module