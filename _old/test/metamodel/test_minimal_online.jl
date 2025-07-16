using Test
using CUDA
using Flux

# Include online learning module
include("../../src/metamodel/online_learning.jl")

using .OnlineLearning
using .OnlineLearning.NeuralArchitecture
using .OnlineLearning.ExperienceReplay

println("Testing Minimal Online Learning...")

# Skip tests if no GPU available
if !CUDA.functional()
    @warn "CUDA not functional, skipping GPU tests"
    exit(0)
end

# Test 1: Simple configuration without double buffering
config = create_online_config(
    batch_size = 4,
    update_frequency = 1,
    learning_rate = 1f-3,
    accumulation_steps = 1,
    use_double_buffering = false  # Disable double buffering
)
println("✓ Configuration created (double buffering disabled)")

# Test 2: Create GPU model
model_config = create_metamodel_config(input_dim=20, hidden_dims=[16, 8, 4])
model = create_gpu_metamodel(model_config)
println("✓ GPU model created")

# Test 3: Initialize online learning
state = initialize_online_learning(model, config)
println("✓ Online learning state initialized")

# Test 4: Test model forward pass
test_input = CUDA.rand(Float32, 20, 2)
output = state.primary_model(test_input)
println("✓ Model forward pass works: output shape = $(size(output))")

# Test 5: Create minimal replay buffer
replay_config = create_replay_config(buffer_size=10, max_features=5)
buffer = create_replay_buffer(replay_config)
println("✓ Replay buffer created")

# Test 6: Add a few experiences
for i in 1:5
    features = Int32[i, i+1]
    predicted = 0.5f0
    actual = 0.5f0 + 0.01f0 * i
    insert_experience!(buffer, features, predicted, actual)
end
update_sum_tree!(buffer)
println("✓ Added $(get_buffer_stats(buffer).n_valid) experiences")

# Test 7: Try one update
try
    updated = online_update!(state, buffer, config)
    if updated
        println("✓ Online update successful")
    else
        println("✓ Online update skipped (expected behavior)")
    end
catch e
    println("✗ Online update failed: ", e)
    rethrow(e)
end

println("\n✅ Minimal test completed!")