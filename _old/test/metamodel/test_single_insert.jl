using Test
using CUDA

# Include the experience replay module
include("../../src/metamodel/experience_replay.jl")

using .ExperienceReplay

println("Testing single insertion...")

# Skip tests if no GPU available
if !CUDA.functional()
    @warn "CUDA not functional, skipping GPU tests"
    exit(0)
end

# Create a small buffer
config = create_replay_config(buffer_size=5, max_features=3)
buffer = create_replay_buffer(config)

# Insert one experience
features = Int32[1, 2, 3]
predicted = 0.8f0
actual = 0.75f0

println("Before insertion:")
println("  current_idx: ", CUDA.@allowscalar buffer.current_idx[1])
println("  total_count: ", CUDA.@allowscalar buffer.total_count[1])

insert_experience!(buffer, features, predicted, actual)

println("\nAfter insertion:")
println("  current_idx: ", CUDA.@allowscalar buffer.current_idx[1])
println("  total_count: ", CUDA.@allowscalar buffer.total_count[1])
println("  n_features[1]: ", CUDA.@allowscalar buffer.experience.n_features[1])
println("  predicted[1]: ", CUDA.@allowscalar buffer.experience.predicted_scores[1])
println("  actual[1]: ", CUDA.@allowscalar buffer.experience.actual_scores[1])
println("  valid[1]: ", CUDA.@allowscalar buffer.experience.valid[1])

println("\nTest completed successfully!")