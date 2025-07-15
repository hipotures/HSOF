using Test
using CUDA
using Flux
using Statistics

# Include only the online learning module which includes the others
include("../../src/metamodel/online_learning.jl")

using .OnlineLearning
using .OnlineLearning.NeuralArchitecture
using .OnlineLearning.ExperienceReplay

println("Testing Online Learning Basic Functionality...")

# Skip tests if no GPU available
if !CUDA.functional()
    @warn "CUDA not functional, skipping GPU tests"
    exit(0)
end

# Test 1: Configuration
config = create_online_config(
    batch_size = 8,
    update_frequency = 2,
    learning_rate = 1f-3,
    accumulation_steps = 1
)
println("✓ Configuration created")

# Test 2: Model and state initialization
model_config = create_metamodel_config(input_dim=50, hidden_dims=[32, 16, 8])
model = create_gpu_metamodel(model_config)
state = initialize_online_learning(model, config)
println("✓ Online learning state initialized")

# Test 3: Create and populate replay buffer
replay_config = create_replay_config(buffer_size=50, max_features=10)
buffer = create_replay_buffer(replay_config)

# Add experiences
for i in 1:20
    features = Int32[i, i+1, i+2]
    predicted = 0.7f0 + 0.1f0 * rand(Float32)
    actual = 0.7f0 + 0.05f0 * rand(Float32)
    insert_experience!(buffer, features, predicted, actual)
end
update_sum_tree!(buffer)
println("✓ Replay buffer populated with $(get_buffer_stats(buffer).n_valid) experiences")

# Test 4: Perform online updates
initial_update_count = state.update_count
for i in 1:4
    updated = online_update!(state, buffer, config)
    if updated
        println("✓ Update performed at iteration $(state.iteration_count)")
    end
end
println("✓ Performed $(state.update_count - initial_update_count) updates")

# Test 5: Check statistics
stats = get_online_stats(state)
println("✓ Statistics: iterations=$(stats.iteration_count), updates=$(stats.update_count), lr=$(stats.current_lr)")

# Test 6: Test inference model
inference_model = get_inference_model(state, config)
test_input = CUDA.zeros(Float32, 50, 1)
test_input[1:3, 1] .= 1.0f0
output = inference_model(test_input)
println("✓ Inference model output: $(Array(output)[1])")

println("\n✅ All basic tests passed!")