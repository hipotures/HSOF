using Test
using CUDA
using Statistics

# Include the experience replay module
include("../../src/metamodel/experience_replay.jl")

using .ExperienceReplay

println("Testing batch insertion debug...")

# Skip tests if no GPU available
if !CUDA.functional()
    @warn "CUDA not functional, skipping GPU tests"
    exit(0)
end

# Create a small buffer
config = create_replay_config(buffer_size=10, max_features=5)
buffer = create_replay_buffer(config)

# Prepare simple batch data
batch_size = 3
features_batch = Int32[1 2 3;
                      2 3 4;
                      3 4 5;
                      0 0 0;
                      0 0 0]
predicted_batch = Float32[0.7, 0.8, 0.9]
actual_batch = Float32[0.65, 0.85, 0.95]

println("Inserting batch of size $batch_size")
batch_insert!(buffer, features_batch, predicted_batch, actual_batch)

# Check state
stats = get_buffer_stats(buffer)
println("Buffer stats: n_valid=$(stats.n_valid), total_count=$(stats.total_count)")

# Check each inserted item
for i in 1:batch_size
    n_feat = CUDA.@allowscalar buffer.experience.n_features[i]
    pred = CUDA.@allowscalar buffer.experience.predicted_scores[i]
    actual = CUDA.@allowscalar buffer.experience.actual_scores[i]
    valid = CUDA.@allowscalar buffer.experience.valid[i]
    
    println("Item $i: n_features=$n_feat, predicted=$pred, actual=$actual, valid=$valid")
end

println("\nDone!")