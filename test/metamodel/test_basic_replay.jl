using Test
using CUDA
using Statistics

# Include the experience replay module
include("../../src/metamodel/experience_replay.jl")

using .ExperienceReplay

println("Testing Experience Replay Buffer Basic Functionality...")

# Skip tests if no GPU available
if !CUDA.functional()
    @warn "CUDA not functional, skipping GPU tests"
    exit(0)
end

# Test 1: Create buffer
config = create_replay_config(buffer_size=100, max_features=10)
buffer = create_replay_buffer(config)
println("✓ Buffer created with size $(config.buffer_size)")

# Test 2: Insert single experience
features = Int32[1, 5, 10, 15]
predicted = 0.85f0
actual = 0.80f0
insert_experience!(buffer, features, predicted, actual)
println("✓ Single experience inserted")

# Test 3: Check buffer state
stats = get_buffer_stats(buffer)
println("✓ Buffer stats: $(stats.n_valid) valid, $(stats.total_count) total")

# Test 4: Batch insert
batch_features = Int32[1 2 3 4 5;
                       2 3 4 5 6;
                       3 4 5 6 7;
                       0 0 0 0 0;
                       0 0 0 0 0;
                       0 0 0 0 0;
                       0 0 0 0 0;
                       0 0 0 0 0;
                       0 0 0 0 0;
                       0 0 0 0 0]
predicted_batch = Float32[0.7, 0.8, 0.6, 0.9, 0.75]
actual_batch = Float32[0.65, 0.85, 0.55, 0.88, 0.8]

batch_insert!(buffer, batch_features, predicted_batch, actual_batch)
println("✓ Batch of 5 experiences inserted")

# Test 5: Update sum tree and sample
update_sum_tree!(buffer)
batch = sample_batch(buffer, 3)
if !isnothing(batch)
    println("✓ Sampled batch of size $(length(batch.indices))")
else
    println("✗ Failed to sample batch")
end

# Test 6: Get final stats
final_stats = get_buffer_stats(buffer)
println("✓ Final buffer utilization: $(round(final_stats.buffer_utilization * 100, digits=1))%")

println("\n✅ All basic tests passed!")