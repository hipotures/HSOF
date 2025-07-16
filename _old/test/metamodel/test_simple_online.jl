using Test
using CUDA
using Flux

# Include simple online learning
include("../../src/metamodel/online_learning_simple.jl")

using .OnlineLearningSimple
using .OnlineLearningSimple.NeuralArchitecture
using .OnlineLearningSimple.ExperienceReplay

println("Testing Simple Online Learning...")

# Skip tests if no GPU available
if !CUDA.functional()
    @warn "CUDA not functional, skipping GPU tests"
    exit(0)
end

# Create configuration
config = SimpleOnlineConfig(4, 1f-3, 2)
println("✓ Config created")

# Create model
model_config = create_metamodel_config(input_dim=20, hidden_dims=[16, 8, 4])
model = create_gpu_metamodel(model_config)
println("✓ GPU model created")

# Initialize online learning
state = init_online_learning(model, config.learning_rate)
println("✓ Online learning initialized")

# Create replay buffer
replay_config = create_replay_config(buffer_size=20, max_features=5)
buffer = create_replay_buffer(replay_config)

# Add experiences
for i in 1:10
    features = Int32[i, mod1(i+1, 20)]
    predicted = 0.5f0
    actual = 0.5f0 + 0.01f0 * i
    insert_experience!(buffer, features, predicted, actual)
end
update_sum_tree!(buffer)
println("✓ Buffer populated with $(get_buffer_stats(buffer).n_valid) experiences")

# Test updates
initial_updates = state.update_count
for i in 1:6
    updated = simple_update!(state, buffer, config)
    if updated
        println("✓ Update #$(state.update_count) at iteration $(state.iteration_count)")
    end
end

println("\n✅ Simple online learning test completed!")
println("Total updates: $(state.update_count)")
println("Total iterations: $(state.iteration_count)")